//! Batch ownership and interruption at the shared engine/API boundary (§3.1, §6.5).
//!
//! CLI discovery/dispatch must opt into these helpers. They do not implement run,
//! control, or export handlers; a refused control never counts as a committed pause.
#![cfg(not(windows))] // signal-hook's iterator is unavailable on Windows.
#![allow(dead_code)] // Store-bound transport wiring awaits the shared engine owner.

use jope_api::{ids::RunId, RunSnapshot, RunState};
use jope_client::{Request, Transport};
#[cfg(feature = "engine")]
use jope_service::{engine::InProcessEngine, store::Store};
use signal_hook::iterator::{Handle, Signals};
#[cfg(feature = "engine")]
use std::path::Path;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

/// Scope the store and its advisory lock to this command, on both success and error.
/// The borrowed store cannot escape the operation. No listener or child is started.
#[cfg(feature = "engine")]
pub fn with_engine<R>(
    home: &Path,
    operation: impl FnOnce(&Store, &Interrupts) -> anyhow::Result<R>,
) -> anyhow::Result<R> {
    let engine = InProcessEngine::open(home)?;
    // Reverse local drop order joins/unregisters the signal owner before the
    // engine closes its Store and releases the advisory lock, including unwind.
    let interrupts = Interrupts::install()?;
    operation(engine.store(), &interrupts)
}

/// A durable paused/terminal acknowledgment, suitable for the batch entrypoint.
#[derive(Debug)]
pub struct Interrupted {
    pub run_id: RunId,
    pub state: RunState,
}

impl Interrupted {
    pub fn exit_code(&self) -> u8 {
        130
    }
}

impl std::fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "run {} acknowledged {:?} after SIGINT",
            self.run_id, self.state
        )
    }
}

impl std::error::Error for Interrupted {}

/// Safe signal delivery; API calls and store IO never run in a signal handler.
pub struct Interrupts {
    count: Arc<AtomicUsize>,
    handle: Handle,
    reader: Option<JoinHandle<()>>,
}

impl Interrupts {
    pub fn install() -> std::io::Result<Self> {
        let mut signals = Signals::new([signal_hook::consts::SIGINT])?;
        let handle = signals.handle();
        let count = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&count);
        let reader = thread::Builder::new()
            .name("jope-batch-signals".into())
            .spawn(move || {
                for _ in signals.forever() {
                    // SIGINT may be coalesced by the OS; retain every delivered
                    // occurrence, including a second received during API/store IO.
                    let _ = observed.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                        Some(n.saturating_add(1).min(2))
                    });
                }
            })?;
        Ok(Self {
            count,
            handle,
            reader: Some(reader),
        })
    }

    pub fn requested(&self) -> bool {
        self.count.load(Ordering::SeqCst) != 0
    }

    #[cfg(test)]
    pub(crate) fn delivered_for_test(count: usize) -> Self {
        let interrupts = Self::install().unwrap();
        interrupts.count.store(count.min(2), Ordering::SeqCst);
        interrupts
    }

    /// Called by the command at a request boundary after SIGINT. Wait for the
    /// engine's committed acknowledgment; a second SIGINT escalates to cancel.
    /// Errors propagate instead of manufacturing a successful exit130.
    pub fn finish<T: Transport>(
        &self,
        transport: &T,
        mut snapshot: RunSnapshot,
    ) -> anyhow::Result<Option<Interrupted>> {
        if !self.requested() {
            return Ok(None);
        }
        let run_id = snapshot.run_id.clone();
        // The command may have retained a snapshot across a long request. A
        // stale paused snapshot is not proof of the run's current durable state.
        snapshot = transport.run_status(&run_id)?;
        let mut cancelling = false;
        let mut pausing = false;
        loop {
            anyhow::ensure!(
                snapshot.run_id == run_id,
                "control returned a different run"
            );
            let cancel = self.count.load(Ordering::SeqCst) >= 2;
            let terminal = matches!(
                snapshot.state,
                RunState::Completed | RunState::Cancelled | RunState::Failed
            );
            if terminal || (snapshot.state == RunState::Paused && !cancel) {
                return Ok(Some(Interrupted {
                    run_id,
                    state: snapshot.state,
                }));
            }
            let action = if cancel && !cancelling {
                cancelling = true;
                Some("cancel")
            } else if !cancel && !pausing && snapshot.state == RunState::Running {
                pausing = true;
                Some("pause")
            } else {
                None
            };
            if let Some(action) = action {
                let key = format!("cli_{}", ulid::Ulid::new());
                snapshot = transport
                    .execute(
                        Request::new(8, 0, &[("id", &run_id.to_string())])?
                            .key(Some(&key))?
                            .if_match(snapshot.revision)
                            .json(&serde_json::json!({"action": action}))?,
                    )?
                    .json()?;
            } else {
                // A short status request leaves room for cancellation; do not
                // enter the batch command's normal 30-second long poll here.
                thread::sleep(Duration::from_millis(20));
                snapshot = transport.run_status(&run_id)?;
            }
        }
    }
}

impl Drop for Interrupts {
    fn drop(&mut self) {
        self.handle.close();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        // The reader and this final handle drop this owner's registrations.
    }
}

#[cfg(test)]
pub(super) fn exit_process_for_test(code: u8) -> ! {
    std::process::exit(i32::from(code));
}

#[cfg(all(test, feature = "engine"))]
#[path = "inproc/tests.rs"]
mod tests;
