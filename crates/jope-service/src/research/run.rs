//! Drive the existing capture engine after trusted job resolution.
use super::{outbox, CaptureContext, RecordEvidence, RecordInventory};
use crate::{
    discover::r#loop::{Discovery, Executor, Plan, RoundAdmission, SearchSpace},
    store::Store,
};
use anyhow::{ensure, Result};
use std::{path::Path, sync::Arc};

/// Execution objects supplied by the qualified job's resolver, never JSON inputs.
pub struct RecordingBackend<'a, E, S, A> {
    pub executor: &'a E,
    pub space: &'a S,
    pub admission: &'a A,
    pub live_workers: usize,
}

/// Execute or reopen the pinned research job, then publish its complete outbox.
/// No retries, provider construction, partial export or policy substitution.
pub fn record_run<E: Executor, S: SearchSpace, A: RoundAdmission>(
    store: Arc<Store>,
    plan: Plan,
    context: Arc<CaptureContext>,
    backend: RecordingBackend<'_, E, S, A>,
    output: &Path,
) -> Result<RecordInventory> {
    record_inner(store, plan, context, backend, output, false)
}

fn record_inner<E: Executor, S: SearchSpace, A: RoundAdmission>(
    store: Arc<Store>,
    plan: Plan,
    context: Arc<CaptureContext>,
    backend: RecordingBackend<'_, E, S, A>,
    output: &Path,
    allow_synthetic: bool,
) -> Result<RecordInventory> {
    outbox::validate_destination(output)?;
    ensure!(
        allow_synthetic || context.binding.identity.evidence == RecordEvidence::Engine,
        "synthetic control is not an Engine recording job"
    );
    ensure!(
        backend.live_workers > 0 && backend.live_workers <= plan.planned_workers as usize,
        "live worker count exceeds the admitted research plan"
    );
    context.validate(&plan)?;
    let run = plan.run_id.clone();
    let mut discovery = Discovery::open_research(Arc::clone(&store), plan, context)?;
    while discovery.completed().is_none() {
        discovery.advance(
            backend.space,
            backend.admission,
            backend.executor,
            backend.live_workers,
            usize::MAX,
        )?;
    }
    outbox::materialize(&store, &run, output, allow_synthetic)
}

#[cfg(test)]
mod tests;
