use super::*;

/// Generations are qualification teaching work over eligible frozen contexts.
pub fn validate_generation(root: &Path, job: &GpJob) -> Result<()> {
    if job.generation.is_none()
        && job.parent_checkpoint.is_none()
        && job.parent_checkpoint_sha256.is_none()
    {
        return Ok(());
    }
    ensure!(
        job.generation.is_some() && job.rng_seed.is_some(),
        "generation and RNG seed required"
    );
    let parent = job
        .parent_checkpoint
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("generation parent missing"))?;
    ensure!(
        Some(file_hash(&resolve(root, parent)?.join("state.json"))?)
            == job.parent_checkpoint_sha256,
        "generation parent checkpoint changed"
    );
    let prepared: serde_json::Value =
        read_json(&resolve(root, &job.prepared)?.join("prepared.json"))?;
    ensure!(
        matches!(
            prepared["meta_partition"].as_str(),
            Some("meta_train" | "meta_validation")
        ),
        "sealed-test jobs cannot enter teaching generations"
    );
    ensure!(
        job.seeds.is_some() && job.seeds_sha256.is_some(),
        "generation seed population missing"
    );
    Ok(())
}

/// The CPUs in `/proc/self/status`'s `Cpus_allowed_list` (affinity mask and cpuset), e.g.
/// `0-3,8,10-11`.
pub(super) fn allowed_cpus(status: &str) -> Result<std::collections::BTreeSet<usize>> {
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("CPU affinity unavailable")?
        .trim();
    let mut cpus = std::collections::BTreeSet::new();
    for part in list.split(',') {
        let (start, end) = match part.split_once('-') {
            Some((start, end)) => (start.parse::<usize>()?, end.parse::<usize>()?),
            None => {
                let cpu = part.parse::<usize>()?;
                (cpu, cpu)
            }
        };
        ensure!(
            start <= end && end < 1 << 16,
            "invalid allowed CPU range {part}"
        );
        cpus.extend(start..=end);
    }
    ensure!(!cpus.is_empty(), "allowed CPU list is empty");
    Ok(cpus)
}
