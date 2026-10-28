//! Shared observer lifetime for the pipeline's public entrypoints.
//! Search and output remain authoritative; observer startup/shutdown failures affect only
//! commentary. Stdout remains reserved for the existing command/MCP response contract.
use anyhow::Result;
use jope_narrator::{LineageFact, Tap};

pub(crate) fn with_tap<T>(
    tap: Option<Tap>,
    run: impl FnOnce(Option<&Tap>) -> Result<T>,
    cancelled: impl FnOnce() -> bool,
) -> Result<T> {
    let result = run(tap.as_ref());
    if let Some(tap) = tap {
        tap.emit(LineageFact::RunStopped {
            status: if result.is_ok() {
                "completed"
            } else if cancelled() {
                "cancelled"
            } else {
                "failed"
            }
            .to_owned(),
        });
    }
    result
}

pub(crate) fn from_env<T>(
    run: impl FnOnce(Option<&Tap>) -> Result<T>,
    cancelled: impl FnOnce() -> bool,
) -> Result<T> {
    let journal = jope_narrator::journal_from_env().unwrap_or_else(|error| {
        eprintln!("jope pipeline: narration disabled - {error:#}");
        None
    });
    let Some((tap, journal)) = journal else {
        return with_tap(None, run, cancelled);
    };
    let result = with_tap(Some(tap), run, cancelled);
    if let Err(error) = journal.finish() {
        eprintln!("jope pipeline: narration incomplete - {error:#}");
    }
    result
}
