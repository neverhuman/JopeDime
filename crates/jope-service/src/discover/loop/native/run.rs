use super::*;

/// Hidden M1 adapter. The plan is immutable and contains a FIT-only mapping;
/// `home` is a private store directory, `live_workers` affects throughput only.
/// Public run/start/attach is implemented by BAT-02/CLT-02, not this probe.
pub fn run_m1(path: &Path, executable: &Path) -> Result<Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Input {
        home: PathBuf,
        plan: Plan,
        live_workers: usize,
    }
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        input.live_workers > 0 && input.live_workers <= 1024,
        "invalid live workers"
    );
    let store = Arc::new(Store::open(&input.home)?);
    let pool = ProcessExecutor::new(
        executable,
        input.live_workers,
        &input.home.join("worker-scratch"),
    )?;
    let mut discovery = Discovery::open(store, input.plan)?;
    while discovery.completed().is_none() {
        discovery.advance(
            &NativeSpace,
            &HeuristicAdmission,
            &pool,
            input.live_workers,
            usize::MAX,
        )?;
    }
    ensure!(
        discovery.champion().is_some(),
        "search produced no finite rung-1 champion"
    );
    Ok(
        json!({"stage":"search_finished", "completion_reason":discovery.completed(),
        "budget_bound_by":discovery.budget_bound_by(),
        "fitting":"selection_only",
        "threads_per_worker":1,
        "wall_recovery_policy":"interruption is charged through recovery; explicit pause excludes idle time; backwards clock refuses resume",
        "champion":discovery.snapshot().best, "snapshot":discovery.snapshot(),
        "nondeterminism":discovery.nondeterminism(), "projection":discovery.projection()?,
        "qualification":"M1 heuristic numeric regression; finalization/export and bundle-surrogate parity remain separate gates"}),
    )
}
