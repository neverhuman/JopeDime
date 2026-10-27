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
}

pub(crate) struct Publisher {
    events: SyncSender<EncodingSnapshot>,
    state: Arc<Mutex<State>>,
    cancelled: Arc<AtomicBool>,
    external_cancel: Option<Arc<AtomicBool>>,
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
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            ensure!(
                next.receipt.source_digest == state.last.receipt.source_digest,
                "incompatible encoding source/options scope"
            );
            let unchanged =
                next.receipt.conditioning_digest == state.last.receipt.conditioning_digest;
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

    pub(crate) fn spawn_with_cancel(
        initial: EncodingSnapshot,
        external_cancel: Option<Arc<AtomicBool>>,
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
        while let Some(snapshot) = self.recv() {
            on_snapshot(snapshot)?;
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        Ok((self.last(), self.outcome()))
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
mod tests {
    use super::*;
    use crate::dope_encoder::{
        LandmarkStatus, DOPE512_RECIPE, DOPE_WEIGHTS_SHA256, ENCODING_RECEIPT_SCHEMA,
    };
    use candle_core::{DType, Device, Tensor};

    fn snapshot(digest: &str) -> EncodingSnapshot {
        EncodingSnapshot {
            embedding: Dope512 {
                tensor: Tensor::zeros((1, 512), DType::F32, &Device::Cpu).unwrap(),
                source_digest: "a".repeat(64),
                conditioning_digest: digest.into(),
            },
            receipt: LandmarkEncodingReceipt {
                schema: ENCODING_RECEIPT_SCHEMA.into(),
                status: LandmarkStatus::None,
                mask: [false; 23],
                subsample_n: Some(300),
                source_digest: "a".repeat(64),
                conditioning_digest: digest.into(),
                options_digest: None,
                work_budget: None,
                weights_sha256: DOPE_WEIGHTS_SHA256.into(),
                embedding_identity: DOPE512_RECIPE.into(),
                device: "cpu".into(),
                version: 0,
                outcome: None,
                embedding: None,
            },
            fusion: None,
        }
    }

    #[test]
    fn initial_precedes_worker_completion_and_updates_are_ordered_idempotent() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker_barrier = barrier.clone();
        let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
            worker_barrier.wait();
            assert!(publisher.publish(snapshot("macro"))?);
            assert!(!publisher.publish(snapshot("macro"))?);
            assert!(publisher.publish(snapshot("full"))?);
            Ok(())
        });
        assert_eq!(session.initial().receipt.version, 0);
        assert_eq!(session.recv().unwrap().receipt.conditioning_digest, "zero");
        barrier.wait();
        let mut versions = Vec::new();
        let (last, outcome) = session
            .finish(|s| {
                versions.push(s.receipt.version);
                Ok(())
            })
            .unwrap();
        assert_eq!(versions, vec![1, 2]);
        assert_eq!(last.receipt.version, 2);
        assert_eq!(outcome, SessionOutcome::Complete);
    }

    #[test]
    fn new_coverage_is_published_without_advancing_an_unchanged_encoding() {
        let session = EncodingSession::spawn(snapshot("zero"), |publisher| {
            let mut macro_snapshot = snapshot("macro");
            macro_snapshot.receipt.mask[0] = true;
            macro_snapshot.receipt.status = LandmarkStatus::Partial(1);
            assert!(publisher.publish(macro_snapshot.clone())?);
            macro_snapshot.receipt.mask[17] = true;
            macro_snapshot.receipt.status = LandmarkStatus::Partial(2);
            assert!(publisher.publish(macro_snapshot.clone())?);
            assert!(!publisher.publish(macro_snapshot)?);
            Ok(())
        });
        let mut versions = Vec::new();
        let (last, outcome) = session
            .finish(|s| {
                versions.push(s.receipt.version);
                Ok(())
            })
            .unwrap();
        assert_eq!(versions, vec![0, 1, 1]);
        assert_eq!(last.receipt.status, LandmarkStatus::Partial(2));
        assert_eq!(last.receipt.version, 1);
        assert_eq!(outcome, SessionOutcome::Complete);
    }

    #[test]
    fn incompatible_scope_and_worker_panic_preserve_last_valid_snapshot() {
        for panic in [false, true] {
            let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
                publisher.publish(snapshot("macro"))?;
                if panic {
                    panic!("injected worker failure");
                }
                let mut wrong = snapshot("other");
                wrong.receipt.source_digest = "b".repeat(64);
                publisher.publish(wrong)?;
                Ok(())
            });
            let (last, outcome) = session.finish(|_| Ok(())).unwrap();
            assert_eq!(last.receipt.conditioning_digest, "macro");
            assert_eq!(last.receipt.version, 1);
            assert!(matches!(outcome, SessionOutcome::Failed { .. }));
        }
    }

    #[test]
    fn cancellation_keeps_last_published_embedding() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker_barrier = barrier.clone();
        let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
            worker_barrier.wait();
            assert!(publisher.cancelled());
            assert!(!publisher.publish(snapshot("late"))?);
            Ok(())
        });
        session.cancel();
        barrier.wait();
        let (last, outcome) = session.finish(|_| Ok(())).unwrap();
        assert_eq!(last.receipt.version, 0);
        assert_eq!(outcome, SessionOutcome::Cancelled);
    }
    #[test]
    fn external_cancellation_is_observed_before_late_publication() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker_barrier = barrier.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let session = EncodingSession::spawn_with_cancel(
            snapshot("zero"),
            Some(cancel.clone()),
            move |publisher| {
                worker_barrier.wait();
                assert!(publisher.cancelled());
                assert!(!publisher.publish(snapshot("late"))?);
                Ok(())
            },
        );
        assert_eq!(session.recv().unwrap().receipt.version, 0);
        cancel.store(true, Ordering::Release);
        barrier.wait();
        let (last, outcome) = session.finish(|_| Ok(())).unwrap();
        assert_eq!(last.receipt.version, 0);
        assert_eq!(outcome, SessionOutcome::Cancelled);
    }
}
