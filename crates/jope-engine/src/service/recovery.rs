use super::*;

/// Recover compact durable results before dispatch starts. Retain only pending
/// jobs, preserving their order; never walk the much larger observation ledgers.
pub(crate) fn restore_completed(
    root: &Path,
    jobs: &mut Vec<GpJob>,
    coverage: &mut coverage::Ledger,
    restored: bool,
) -> Result<(usize, u64)> {
    let mut pending = Vec::with_capacity(jobs.len());
    let mut completed = 0usize;
    let mut evaluations = 0u64;
    for job in jobs.drain(..) {
        let summary_path = resolve(root, &job.output)?.join("search.json");
        if summary_path.is_file() {
            let summary: serde_json::Value = read_json(&summary_path)?;
            ensure!(
                summary["schema"] == "parallel.search.v1",
                "invalid completed GP summary"
            );
            if let Some(seed) = job.rng_seed {
                ensure!(
                    summary["rng_seed"] == seed,
                    "completed GP RNG identity changed"
                );
                ensure!(
                    summary["seeds_sha256"] == serde_json::to_value(&job.seeds_sha256)?,
                    "completed GP population changed"
                );
            }
            if job.generation.is_some() {
                ensure!(
                    read_json::<serde_json::Value>(
                        &resolve(root, &job.output)?.join("job-binding.json")
                    )? == serde_json::to_value(&job)?,
                    "completed GP generation changed"
                );
            }
            let count = summary["completed_evaluations"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("completed GP evaluation count absent"))?;
            ensure!(count <= 1000, "completed GP count exceeds view budget");
            if !restored || coverage.tracks(&job)? {
                ensure!(
                    summary["terminal"] == true
                        && summary["completed_at"].as_u64().is_some_and(|n| n > 0)
                        && summary["prepared_sha256"] == job.prepared_sha256
                        && summary["cut"] == job.cut
                        && summary["observations_sha256"]
                            == file_hash(&summary_path.with_file_name("observations.json"))?,
                    "worker completion payload or cut binding absent"
                );
            }
            coverage.reused(&job, now(), restored)?;
            completed += 1;
            evaluations += count;
        } else {
            pending.push(job);
        }
    }
    *jobs = pending;
    Ok((completed, evaluations))
}
