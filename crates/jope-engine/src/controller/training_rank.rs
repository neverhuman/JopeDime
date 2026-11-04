use super::*;

pub(super) fn training_step_window(
    initial_step: u64,
    increment: usize,
) -> Result<serde_json::Value> {
    ensure!(increment > 0, "training increment must be positive");
    let target = initial_step
        .checked_add(increment as u64)
        .context("training step overflow")?;
    Ok(
        serde_json::json!({"schema":"parallel.training_step_window.v1",
        "initial_optimizer_step":initial_step,"requested_step_increment":increment,
        "target_optimizer_step":target}),
    )
}

pub(super) fn verify_training_rank(
    receipt: &serde_json::Value,
    window: &serde_json::Value,
    rank: usize,
    world: usize,
    corpus_hash: &str,
) -> Result<()> {
    ensure!(
        receipt["rank"] == rank
            && receipt["world"] == world
            && receipt["steps"] == window["target_optimizer_step"]
            && receipt["local_steps"] == window["requested_step_increment"]
            && receipt["corpus_sha256"] == corpus_hash,
        "rank completion step window or identity mismatch"
    );
    let initial = receipt["initial_optimizer_step"]
        .as_u64()
        .context("rank initial step absent")?;
    let executed = receipt["executed_steps"]
        .as_u64()
        .context("rank executed steps absent")?;
    let target = window["target_optimizer_step"]
        .as_u64()
        .context("target step absent")?;
    ensure!(
        initial
            >= window["initial_optimizer_step"]
                .as_u64()
                .context("initial step absent")?
            && initial.checked_add(executed) == Some(target),
        "rank execution exposure mismatch"
    );
    Ok(())
}
