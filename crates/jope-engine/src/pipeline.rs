//! Resumable real-data DL -> adaptive GP -> DL integration campaign.
use crate::{
    controller, corpus,
    grammar::Vocab,
    manifest::*,
    service::GpJobs,
    telemetry::{now, StageHeartbeat},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

fn check_gp_health(
    service: &Value,
    pending: &[PathBuf],
    elapsed: Duration,
    timestamp: u64,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let observation = crate::telemetry::observed_status(service, timestamp);
    ensure!((service.is_null() && elapsed < Duration::from_secs(10)) || observation["current_activity"] == "current", "GP service activity is stale, terminal or unbound while {} cycle jobs remain; reconcile the service before resuming this cycle", pending.len());
    ensure!(
        service["phase"] != "failed",
        "GP service reports failure while cycle jobs remain: {}",
        service["failure"]
    );
    ensure!(elapsed < Duration::from_secs(3) || service["phase"] != "completed", "GP service finished its queue but {} cycle jobs have no complete search receipts; restart the GP service to ingest pending priority jobs", pending.len());
    for row in service["failed_rows"].as_array().into_iter().flatten() {
        if row["output"]
            .as_str()
            .is_some_and(|p| pending.iter().any(|j| j == Path::new(p)))
        {
            anyhow::bail!("cycle GP search failed: {row}; recover the job and resume the cycle");
        }
    }
    Ok(())
}

pub(crate) use crate::corpus_publication::verify as verify_corpus;

fn build_corpus_atomic(
    root: &Path,
    contexts: &Path,
    vocab: &Path,
    observations: &[PathBuf],
    output: &Path,
    heartbeat: Option<&StageHeartbeat>,
) -> Result<()> {
    crate::corpus_publication::build(
        root,
        crate::corpus_publication::PublicationSources {
            contexts,
            vocab,
            observations,
        },
        output,
        true,
        Some(&corpus_cache_path(root, output)?),
        heartbeat,
    )
}

/// Reuse across cycles in one campaign, with an isolated sibling for standalone
/// cycles. The corpus API verifies namespace separation and all artifact hashes.
fn corpus_cache_path(root: &Path, corpus_output: &Path) -> Result<PathBuf> {
    let cycle = corpus_output
        .parent()
        .context("corpus cycle directory absent")?;
    let parent = cycle.parent().context("corpus campaign directory absent")?;
    let namespace = if cycle
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("cycle-"))
        && parent != root.join("runs")
    {
        parent
    } else {
        cycle
    };
    let relative = namespace
        .strip_prefix(root)
        .context("corpus cache outside workspace")?;
    ensure!(
        relative.starts_with("runs") && relative.components().count() >= 2,
        "corpus cache requires a private campaign namespace"
    );
    Ok(namespace.join("corpus-scan-cache"))
}

#[cfg(test)]
mod corpus_cache_tests;

pub(crate) fn latest(root: &Path, run: &Path) -> Result<PathBuf> {
    let v: Value = read_json(&run.join("latest.json"))?;
    let cp = resolve(
        root,
        Path::new(v["checkpoint"].as_str().context("checkpoint path")?),
    )?;
    ensure!(
        file_hash(&cp.join("state.json"))? == v["state_sha256"],
        "checkpoint receipt changed"
    );
    Ok(cp)
}
fn progress(
    root: &Path,
    phase: &str,
    proposals: (usize, usize),
    gp: (usize, usize),
    train: (usize, usize),
    complete: bool,
    reason: &str,
) -> Result<()> {
    write_json(
        &root.join("runs/cycle-progress.json"),
        &json!({"schema":"parallel.cycle_progress.v1","timestamp":now(),"phase":phase,"proposals":{"completed":proposals.0,"total":proposals.1},"neural_seeded_gp":{"completed":gp.0,"total":gp.1},"retraining":{"step":train.0,"steps":train.1},"complete":complete,"reason":reason,"pilot_started":false,"qualification":true}),
    )
}
use crate::tournament::baseline;

pub(crate) fn verify_cycle_identity(
    root: &Path,
    corpus_path: &Path,
    output: &Path,
    steps: usize,
    cuts: usize,
    cut_start: usize,
    contexts_path: Option<&Path>,
    initialize: Option<&Path>,
    extra_observations: &[PathBuf],
    require_existing: bool,
) -> Result<(PathBuf, String)> {
    for arm in ["neural", "shuffled"] {
        controller::verify_proposal_protocol(
            &output.join(arm),
            require_existing || output.join("complete.json").exists(),
        )?;
    }
    let configured_executable = PathBuf::from(
        std::env::var_os("JOPE_GP_EXECUTABLE")
            .context("JOPE_GP_EXECUTABLE must name the pinned CPU-only predictor evaluator")?,
    );
    ensure!(
        configured_executable.is_absolute(),
        "JOPE_GP_EXECUTABLE must be absolute"
    );
    let executable = resolve(
        root,
        configured_executable
            .strip_prefix(root)
            .context("predictor evaluator must be inside the workspace")?,
    )?;
    let executable_sha = file_hash(&executable)?;
    let evaluator_receipt = output.join("predictor-evaluator.json");
    let evaluator = json!({"schema":"parallel.predictor_evaluator.v1","executable":executable.strip_prefix(root)?,"sha256":executable_sha,"timeout_seconds":30,"memory_limit_gib":8,"threads":1});
    ensure!(
        !require_existing || evaluator_receipt.is_file(),
        "completed cycle evaluator receipt absent"
    );
    if evaluator_receipt.exists() {
        ensure!(
            read_json::<Value>(&evaluator_receipt)? == evaluator,
            "cycle predictor evaluator identity changed"
        );
    } else {
        write_json(&evaluator_receipt, &evaluator)?;
    }
    let run_spec = output.join("cycle.json");
    let mut spec = json!({"schema":"parallel.cycle.v1","corpus_sha256":file_hash(&corpus_path.join("index.json"))?,"steps_each_training_block":steps,"cuts_each_dataset":cuts,"qualification":true,"evaluation_bias":"adaptive_cut_heldout_teacher","final_tests_opened":false});
    controller::bind_proposal_protocol(&mut spec);
    spec["champion_snapshot_protocol"] = "cycle-shared-cut-evaluator-v1".into();
    spec["auxiliary_training_protocol"] = crate::auxiliary_training::PROTOCOL.into();
    spec["auxiliary_training_options"] =
        serde_json::to_value(crate::auxiliary_training::Options::for_cycle(steps))?;
    if cut_start != 0 {
        spec["cut_start"] = cut_start.into();
    }
    if let Some(path) = contexts_path {
        spec["contexts_sha256"] = file_hash(path)?.into();
    }
    if let Some(path) = initialize {
        spec["initialization_sha256"] = file_hash(&path.join("state.json"))?.into();
    }
    if !extra_observations.is_empty() {
        spec["extra_observations"] = json!(extra_observations);
    }
    ensure!(
        !require_existing || run_spec.is_file(),
        "completed cycle specification absent"
    );
    if run_spec.exists() {
        ensure!(
            read_json::<Value>(&run_spec)? == spec,
            "cycle resume specification changed"
        );
    } else {
        write_json(&run_spec, &spec)?;
    }
    if require_existing || output.join("champion-snapshot.json").exists() {
        crate::champions::verify_cycle_snapshot(
            output,
            spec["corpus_sha256"].as_str().unwrap(),
            cuts,
            cut_start,
        )?;
    }
    Ok((executable, executable_sha))
}

mod run;
pub use run::run;

#[cfg(test)]
mod campaign_identity_tests;
