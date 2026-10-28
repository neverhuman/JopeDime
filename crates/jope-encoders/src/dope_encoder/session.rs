use super::{Dope512, DopeFusion, FusionComponents, LandmarkEncodingReceipt};
use anyhow::{ensure, Result};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender, TrySendError},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionOutcome {
    Running,
    Complete,
    Cancelled,
    Failed { message: String },
}

#[derive(Clone, Debug)]
pub struct EncodingSnapshot {
    pub receipt: LandmarkEncodingReceipt,
    pub embedding: Dope512,
    pub fusion: Option<DopeFusion>,
    pub consumer: Option<super::DopeConsumedEncoding>,
}

impl EncodingSnapshot {
    pub fn with_components(mut self, components: &FusionComponents) -> Result<Self> {
        self.fusion = Some(self.embedding.assemble(components)?);
        Ok(self)
    }
    pub fn with_resident_components(
        mut self,
        components: &super::ResidentFusionComponents,
    ) -> Result<Self> {
        self.fusion = Some(self.embedding.assemble_resident(components)?);
        Ok(self)
    }
}

struct State {
    last: EncodingSnapshot,
    outcome: SessionOutcome,
}

/// Ordered bounded snapshots. The initial snapshot is available before any worker starts.
/// Dropping or cancelling a session stops publication at the last valid snapshot.
pub struct EncodingSession {
    initial: EncodingSnapshot,
    events: Receiver<EncodingSnapshot>,
    state: Arc<Mutex<State>>,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    admission: Option<Arc<super::AuthenticatedFusionComponents>>,
}

pub(crate) struct Publisher {
    events: SyncSender<EncodingSnapshot>,
    state: Arc<Mutex<State>>,
    cancelled: Arc<AtomicBool>,
    external_cancel: Option<Arc<AtomicBool>>,
    admission: Option<Arc<super::AuthenticatedFusionComponents>>,
}

impl Publisher {
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
            || self
                .external_cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Acquire))
    }

    /// Compare effective conditioning, not bundle values: native confidence can change the input.
    pub fn publish(&self, mut next: EncodingSnapshot) -> Result<bool> {
        if self.cancelled() {
            return Ok(false);
        }
        if let Some(admission) = &self.admission {
            admission.require_active()?;
        }
        ensure!(
            next.receipt.source_digest == next.embedding.source_digest
                && next.receipt.conditioning_digest == next.embedding.conditioning_digest
                && next.receipt.weights_sha256
                    == next.embedding.checkpoint_identity.weights_sha256()
                && next.receipt.embedding_identity
                    == next.embedding.checkpoint_identity.embedding_recipe()
                && next.receipt.training_receipt_sha256.as_deref()
                    == next.embedding.checkpoint_identity.training_receipt_sha256(),
            "encoding receipt differs from resident embedding"
        );
        match (&next.consumer, &next.receipt.consumer) {
            (Some(consumer), Some(_)) => {
                consumer.require_snapshot(&next.embedding, &next.receipt)?
            }
            (None, None) => {}
            _ => anyhow::bail!("consumer receipt and resident product differ"),
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            ensure!(
                next.receipt.source_digest == state.last.receipt.source_digest,
                "incompatible encoding source/options scope"
            );
            ensure!(
                next.receipt.fit_scope == state.last.receipt.fit_scope,
                "consumer physical FIT/fold preparation scope cannot change within a session"
            );
            // Validate the fixed product binding before duplicate elision.
            match (&state.last.consumer, &next.consumer) {
                (Some(previous), Some(current)) => previous.require_same_binding(current)?,
                (None, None) => {}
                _ => anyhow::bail!("consumer selection cannot change within a session"),
            }
            let unchanged =
                next.receipt.conditioning_digest == state.last.receipt.conditioning_digest;
            ensure!(
                next.receipt.weights_sha256 == state.last.receipt.weights_sha256
                    && next.receipt.embedding_identity == state.last.receipt.embedding_identity
                    && next.receipt.training_receipt_sha256
                        == state.last.receipt.training_receipt_sha256,
                "encoding checkpoint cannot change within a session"
            );
            if unchanged && next.receipt.mask == state.last.receipt.mask {
                return Ok(false);
            }
            // Coverage can advance for cells unused at this model level, without a new encoding.
            next.receipt.version = if unchanged {
                state.last.receipt.version
            } else {
                state
                    .last
                    .receipt
                    .version
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("encoding version overflow"))?
            };
            if self.cancelled() {
                return Ok(false);
            }
            if let Some(admission) = &self.admission {
                admission.require_active()?;
            }
            state.last = next.clone();
        }
        loop {
            if self.cancelled() {
                return Ok(false);
            }
            match self.events.try_send(next) {
                Ok(()) => return Ok(true),
                Err(TrySendError::Full(value)) => {
                    next = value;
                    thread::park_timeout(Duration::from_millis(2));
                }
                Err(TrySendError::Disconnected(_)) => return Ok(false),
            }
        }
    }
}

impl EncodingSession {
    #[cfg(test)]
    pub(crate) fn spawn(
        initial: EncodingSnapshot,
        worker: impl FnOnce(Publisher) -> Result<()> + Send + 'static,
    ) -> Self {
        Self::spawn_with_cancel(initial, None, worker)
    }

    #[cfg(test)]
    pub(crate) fn spawn_with_cancel(
        initial: EncodingSnapshot,
        external_cancel: Option<Arc<AtomicBool>>,
        worker: impl FnOnce(Publisher) -> Result<()> + Send + 'static,
    ) -> Self {
        Self::spawn_with_admission(initial, external_cancel, None, worker)
    }

    pub(crate) fn spawn_with_admission(
        initial: EncodingSnapshot,
        external_cancel: Option<Arc<AtomicBool>>,
        admission: Option<Arc<super::AuthenticatedFusionComponents>>,
        worker: impl FnOnce(Publisher) -> Result<()> + Send + 'static,
    ) -> Self {
        let (tx, events) = mpsc::sync_channel(4);
        // This send completes before thread creation, establishing initial-before-worker order.
        tx.send(initial.clone()).expect("new snapshot channel");
        let cancelled = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(State {
            last: initial.clone(),
            outcome: SessionOutcome::Running,
        }));
        let publisher = Publisher {
            events: tx,
            state: state.clone(),
            cancelled: cancelled.clone(),
            external_cancel: external_cancel.clone(),
            admission: admission.clone(),
        };
        let worker_state = state.clone();
        let worker_cancelled = cancelled.clone();
        let handle = thread::spawn(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker(publisher)));
            let outcome = if worker_cancelled.load(Ordering::Acquire)
                || external_cancel
                    .as_ref()
                    .is_some_and(|c| c.load(Ordering::Acquire))
            {
                SessionOutcome::Cancelled
            } else {
                match result {
                    Ok(Ok(())) => SessionOutcome::Complete,
                    Ok(Err(error)) => SessionOutcome::Failed {
                        message: format!("{error:#}"),
                    },
                    Err(_) => SessionOutcome::Failed {
                        message: "encoding worker panicked".into(),
                    },
                }
            };
            worker_state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .outcome = outcome;
        });
        Self {
            initial,
            events,
            state,
            cancelled,
            worker: Some(handle),
            admission,
        }
    }

    pub fn initial(&self) -> &EncodingSnapshot {
        &self.initial
    }

    pub fn recv(&self) -> Option<EncodingSnapshot> {
        self.events.recv().ok()
    }

    pub fn try_recv(&self) -> Option<EncodingSnapshot> {
        self.events.try_recv().ok()
    }

    pub fn last(&self) -> EncodingSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last
            .clone()
    }

    pub fn outcome(&self) -> SessionOutcome {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .outcome
            .clone()
    }

    pub fn cancellation_token(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.state.lock().unwrap_or_else(|e| e.into_inner()).outcome = SessionOutcome::Cancelled;
    }

    /// Drain updates and join. Errors remain recorded alongside the last valid snapshot.
    pub fn finish(
        mut self,
        mut on_snapshot: impl FnMut(EncodingSnapshot) -> Result<()>,
    ) -> Result<(EncodingSnapshot, SessionOutcome)> {
        let mut authorization_failure = None;
        while let Some(snapshot) = self.recv() {
            if let Err(error) = self.require_active() {
                authorization_failure = Some(format!("{error:#}"));
                self.cancelled.store(true, Ordering::Release);
                break;
            }
            on_snapshot(snapshot)?;
            if let Err(error) = self.require_active() {
                authorization_failure = Some(format!("{error:#}"));
                self.cancelled.store(true, Ordering::Release);
                break;
            }
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Err(error) = self.require_active() {
            authorization_failure = Some(format!("{error:#}"));
        }
        if let Some(message) = authorization_failure {
            self.state.lock().unwrap_or_else(|e| e.into_inner()).outcome =
                SessionOutcome::Failed { message };
        }
        Ok((self.last(), self.outcome()))
    }

    fn require_active(&self) -> Result<()> {
        if let Some(admission) = &self.admission {
            admission.require_active()?;
        }
        Ok(())
    }
}

impl Drop for EncodingSession {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
