//! Recover interrupted checkpoint publication without deleting any saved work.
//! Call only after all previous training ranks have been reaped, before spawning
//! the next attempt. The controller's exclusive GPU lease must remain held.
use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const CONTINUATION_SCHEMA: &str = "parallel.checkpoint_continuation.v1";

/// These originals become part of the immutable resident input closure, too.
pub fn continuation_references(policy: &Value) -> Result<Vec<(PathBuf, String)>> {
    ensure!(
        policy["schema"] == CONTINUATION_SCHEMA,
        "unknown continuation schema"
    );
    let mut references = Vec::new();
    for field in [
        "original_run",
        "original_completion",
        "original_initialization",
        "parent_state",
        "replay_policy_ref",
    ] {
        if field == "replay_policy_ref" && policy.get(field).is_none() {
            continue;
        }
        let reference = policy[field]
            .as_object()
            .context("continuation original reference absent")?;
        ensure!(
            reference.len() == 2,
            "continuation reference must contain only path and sha256"
        );
        let path = PathBuf::from(
            reference
                .get("path")
                .and_then(Value::as_str)
                .context("continuation reference path absent")?,
        );
        relative(&path)?;
        let sha = reference
            .get("sha256")
            .and_then(Value::as_str)
            .context("continuation original digest absent")?;
        ensure!(
            sha.len() == 64
                && sha
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid continuation original digest"
        );
        references.push((path, sha.to_owned()));
    }
    Ok(references)
}

pub fn continuation_replay(root: &Path, policy: &Value) -> Result<Option<Value>> {
    if policy["schema"] != CONTINUATION_SCHEMA {
        return Ok(Some(policy.clone()));
    }
    let Some(reference) = policy.get("replay_policy_ref") else {
        return Ok(None);
    };
    let path = Path::new(
        reference["path"]
            .as_str()
            .context("continuation replay path absent")?,
    );
    let bytes = fs::read(resolve(root, path)?)?;
    ensure!(
        hash(&bytes) == reference["sha256"],
        "continuation replay policy changed"
    );
    let replay: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        replay["schema"] == "parallel.replay_policy.v2"
            && replay["optimizer"] == "carry"
            && replay["sampling_seed"].as_u64() == policy["sampling_seed"].as_u64()
            && replay["parent_checkpoint_sha256"] == policy["parent_checkpoint_sha256"]
            && replay["corpus_sha256"] == policy["corpus_sha256"]
            && replay["vocabulary_sha256"] == policy["vocabulary_sha256"],
        "continuation replay parent, seed, optimizer or cohort differs"
    );
    Ok(Some(replay))
}

/// Requested rank, microbatch and global-batch partition.
#[derive(Clone, Copy)]
pub struct TrainingTopology {
    pub world: usize,
    pub micro: usize,
    pub global: usize,
}

/// Recover a completed parent's observed topology; authorize no execution and
/// never rewrite that historical checkpoint as an exact-resume checkpoint.
pub fn verify_continuation(
    root: &Path,
    policy: &Value,
    corpus_sha: &str,
    vocabulary_sha: &str,
    options: TrainingTopology,
    new_steps: usize,
) -> Result<Value> {
    let TrainingTopology {
        world,
        micro,
        global,
    } = options;

    verify_continuation_partition(
        root,
        policy,
        corpus_sha,
        vocabulary_sha,
        &crate::task_contract::training_partition(world, micro, global)?,
        new_steps,
    )
}

/// [`verify_continuation`] for an explicit requested execution-partition
/// record, v1 or v2 (W7-T06). The original partition is recovered from the
/// original launch topology under the parent's own partition version; a
/// continuation keeps it exactly.
pub fn verify_continuation_partition(
    root: &Path,
    policy: &Value,
    corpus_sha: &str,
    vocabulary_sha: &str,
    requested: &Value,
    new_steps: usize,
) -> Result<Value> {
    let global = usize::try_from(
        requested["global_batch"]
            .as_u64()
            .context("requested global batch absent")?,
    )?;
    let allowed = [
        "schema",
        "mode",
        "optimizer",
        "exact_resume",
        "sampling_seed",
        "parent_checkpoint_sha256",
        "corpus_sha256",
        "vocabulary_sha256",
        "execution_partition",
        "requested_step_increment",
        "original_run",
        "original_completion",
        "original_initialization",
        "parent_state",
        "replay_policy_ref",
    ];
    ensure!(
        policy
            .as_object()
            .context("continuation policy must be an object")?
            .keys()
            .all(|key| allowed.contains(&key.as_str())),
        "mixed replay/continuation fields are not permitted"
    );
    ensure!(
        policy["schema"] == CONTINUATION_SCHEMA
            && policy["mode"] == "optimizer_carry_new_window"
            && policy["optimizer"] == "carry"
            && policy["exact_resume"] == false,
        "continuation must explicitly carry optimizer into a new stochastic window"
    );
    let mut originals = Vec::new();
    for (path, expected) in continuation_references(policy)?.into_iter().take(4) {
        let bytes = fs::read(resolve(root, &path)?)?;
        ensure!(
            hash(&bytes) == expected,
            "continuation original changed: {}",
            path.display()
        );
        originals.push(serde_json::from_slice::<Value>(&bytes)?);
    }
    let [run, completion, initialization, state]: [Value; 4] = originals.try_into().unwrap();
    let parent_sha = policy["parent_state"]["sha256"]
        .as_str()
        .context("parent state digest absent")?;
    let parent_step = state["step"]
        .as_u64()
        .context("parent optimizer step absent")?;
    let original_world = usize::try_from(run["world"].as_u64().context("original world absent")?)?;
    let original_micro = usize::try_from(
        run["microbatch"]
            .as_u64()
            .context("original microbatch absent")?,
    )?;
    let original_global = usize::try_from(
        run["global_batch"]
            .as_u64()
            .context("original global batch absent")?,
    )?;
    // A v2 parent names its partition; every other parent is v1, recovered
    // exactly as before.
    let partition = if state["execution_partition"]["schema"]
        == crate::task_contract::PARTITION_V2_SCHEMA
    {
        let partition = crate::task_contract::training_partition_v2(
            original_world,
            original_micro,
            original_global,
        )?;
        ensure!(
            state["execution_partition"] == partition,
            "original partition-v2 record differs from its launch topology"
        );
        partition
    } else {
        crate::task_contract::training_partition(original_world, original_micro, original_global)?
    };
    ensure!(
        state["schema"] == "parallel.checkpoint.v1"
            && run["schema"] == "parallel.training_run.v1"
            && completion["schema"] == "parallel.training_complete.v1"
            && completion["latest"]["state_sha256"] == parent_sha
            && completion["latest"]["step"].as_u64() == Some(parent_step)
            && completion["steps"].as_u64() == Some(parent_step)
            && completion["world"].as_u64() == Some(original_world as u64)
            && completion["microbatch"].as_u64() == Some(original_micro as u64)
            && state["world"].as_u64() == Some(original_world as u64)
            && state["global_batch"].as_u64() == Some(original_global as u64)
            && state["initialization"] == initialization
            && run["initialization_sha256"] == initialization["state_sha256"],
        "original completion, launch topology or initialization disagree"
    );
    ensure!(
        state["corpus_sha256"] == run["corpus_sha256"]
            && completion["corpus_sha256"] == run["corpus_sha256"]
            && state["vocabulary_sha256"] == vocabulary_sha
            && policy["parent_checkpoint_sha256"] == parent_sha
            && policy["corpus_sha256"] == corpus_sha
            && policy["vocabulary_sha256"] == vocabulary_sha,
        "continuation corpus, vocabulary or parent binding changed"
    );
    ensure!(
        policy["execution_partition"] == partition && partition == *requested,
        "requested continuation partition differs from recovered original topology"
    );
    let original_seed = state["seed"].as_u64().context("original seed absent")?;
    let seed = policy["sampling_seed"]
        .as_u64()
        .context("explicit new-window seed absent")?;
    ensure!(state["sampler_id"] == "family_uniform_champion_plus_seven_quantiles_v2"
        && run["sampler_id"] == state["sampler_id"]
        && state["rng"] == "counter reset before every microbatch: mix(seed ^ step*65537 ^ rank*257 ^ microbatch); LibTorch CPU/CUDA manual_seed"
        && state["sampler"] == format!("family-uniform best teacher plus seven quantile comparators; next global example = step*{global}")
        && state["ranking_backward"] == "dense_pairwise_v2"
        && state["runtime_objective"] == "huber_log1p_seconds_v2"
        && state["autocast_cache_lifetime"] == "scoped_v2",
        "recovered stochastic or objective contract is incompatible");
    // A checkpoint trained with a declared preference objective may not be
    // continued without re-declaring the same one (a silent objective change).
    // Starting a declared preference run from a parent that has none is the
    // intended new-identity path and is allowed.
    ensure!(
        state
            .get("preference_objective")
            .is_none_or(|declared| policy.get("preference_objective") == Some(declared)),
        "recovered checkpoint declares a preference objective the continuation does not"
    );
    ensure!(
        new_steps > 0 && policy["requested_step_increment"].as_u64() == Some(new_steps as u64),
        "continuation requested increment changed"
    );
    let target = parent_step
        .checked_add(new_steps as u64)
        .context("continuation step overflow")?;
    let replay = continuation_replay(root, policy)?;
    Ok(
        json!({"schema":"parallel.recovered_continuation.v1", "mode":"optimizer_carry_new_window",
        "parent_checkpoint_sha256":parent_sha,"optimizer_parent_step":parent_step,
        "requested_step_increment":new_steps,"target_optimizer_step":target,
        "execution_partition":partition,"seed":seed,"original_seed":original_seed,
        "rng":state["rng"],"sampler":state["sampler"],"replay_policy_ref":policy.get("replay_policy_ref"),
        "new_window_replay":replay.as_ref().map(|p| &p["replay"]),
        "sampler_id":state["sampler_id"],"ranking_backward":state["ranking_backward"],
        "runtime_objective":state["runtime_objective"],"original_executable_sha256":run["executable_sha256"],
        "original_next_global_example":parent_step.checked_mul(global as u64).context("sampler offset overflow")?,
        "new_window_counter_origin":0,"exact_same_experiment_resume":false,
        "numerical_equivalence_verified":false,"execution_authority":false,
        "evidence":{"parent_state":policy["parent_state"],"run":policy["original_run"],
            "completion":policy["original_completion"],"initialization":policy["original_initialization"]}}),
    )
}

fn validate_checkpoint(root: &Path, path: &Path, step: u64) -> Result<String> {
    #[cfg(not(feature = "neural"))]
    let _ = root;
    let state_path = path.join("state.json");
    ensure!(
        !fs::symlink_metadata(&state_path)?.file_type().is_symlink(),
        "checkpoint state is a symlink"
    );
    let state: Value = read_json(&state_path)?;
    ensure!(
        state["schema"] == "parallel.checkpoint.v1" && state["step"] == step,
        "checkpoint step or schema changed: {}",
        path.display()
    );
    if state["checkpoint_format"] == "parallel.checkpoint_commit.v3" {
        #[cfg(feature = "neural")]
        {
            let loaded = crate::training::checkpoint_commit::verify_dir(
                root,
                path,
                step,
                &crate::training::checkpoint_commit::Expectation {
                    require_complete: true,
                    ..Default::default()
                },
            )?;
            ensure!(
                loaded.format == crate::training::checkpoint_commit::Format::Successor,
                "successor checkpoint did not load as its declared format"
            );
        }
        #[cfg(not(feature = "neural"))]
        anyhow::bail!("successor checkpoint recovery requires the neural checkpoint verifier");
    }
    for (name, key) in [
        ("model.ot", "model_sha256"),
        ("optimizer.ot", "optimizer_sha256"),
        ("optimizer.json", "optimizer_metadata_sha256"),
        ("config.json", "config_sha256"),
    ] {
        let artifact = path.join(name);
        ensure!(
            !fs::symlink_metadata(&artifact)?.file_type().is_symlink()
                && file_hash(&artifact)? == state[key],
            "checkpoint artifact changed: {}",
            artifact.display()
        );
    }
    file_hash(&state_path)
}

pub fn reconcile(root: &Path, run: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let relative_run = if run.is_absolute() {
        run.strip_prefix(&root)?.to_path_buf()
    } else {
        run.to_path_buf()
    };
    let run = resolve(&root, &relative_run)?;
    let directory = resolve(&root, &relative_run.join("checkpoints"))?;
    let latest_path = resolve(&root, &relative_run.join("latest.json"))?;
    if !directory.exists() {
        ensure!(
            !latest_path.exists(),
            "published checkpoint directory is missing"
        );
        return Ok(());
    }
    let latest_hash = if latest_path.exists() {
        Some(file_hash(&latest_path)?)
    } else {
        None
    };
    let published_step = if latest_hash.is_some() {
        let latest: Value = read_json(&latest_path)?;
        let step = latest["step"]
            .as_u64()
            .context("published checkpoint step absent")?;
        let path = resolve(
            &root,
            Path::new(
                latest["checkpoint"]
                    .as_str()
                    .context("published checkpoint path absent")?,
            ),
        )?;
        ensure!(
            path == directory.join(format!("step-{step}")),
            "latest checkpoint is outside this run or has a mismatched step"
        );
        ensure!(
            validate_checkpoint(&root, &path, step)? == latest["state_sha256"],
            "published checkpoint state changed"
        );
        Some(step)
    } else {
        ensure!(
            !run.join("complete.json").exists(),
            "completed training run lost latest.json; refusing to quarantine committed checkpoints"
        );
        None
    };
    let mut moves = Vec::<Value>::new();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let (step, kind) = if let Some(number) = name.strip_prefix("step-") {
            let Ok(step) = number.parse::<u64>() else {
                continue;
            };
            if published_step.is_some_and(|published| step <= published) {
                continue;
            }
            (step, "unpublished_complete_checkpoint")
        } else if let Some(rest) = name.strip_prefix(".step-") {
            let Some((number, suffix)) = rest.split_once('-') else {
                continue;
            };
            let Ok(step) = number.parse::<u64>() else {
                continue;
            };
            if suffix.is_empty() {
                continue;
            }
            (step, "interrupted_staging_checkpoint")
        } else {
            continue;
        };
        let metadata = entry.file_type()?;
        ensure!(
            metadata.is_dir() && !metadata.is_symlink(),
            "unsafe checkpoint recovery entry: {}",
            entry.path().display()
        );
        let state_sha = if kind == "unpublished_complete_checkpoint" {
            Some(validate_checkpoint(&root, &entry.path(), step)?)
        } else if entry.path().join("state.json").is_file() {
            Some(file_hash(&entry.path().join("state.json"))?)
        } else {
            None
        };
        moves.push(
            json!({"name":name,"step":step,"kind":kind,"state_sha256":state_sha,"moved":false}),
        );
    }
    if moves.is_empty() {
        return Ok(());
    }
    moves.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let recovery = resolve(
        &root,
        &relative_run
            .join("checkpoint-recovery")
            .join(format!("{stamp}-{}", std::process::id())),
    )?;
    fs::create_dir_all(recovery.parent().context("quarantine parent absent")?)?;
    fs::create_dir(&recovery)?;
    let mut receipt = json!({"schema":"parallel.checkpoint_recovery.v1","run":relative_run,
        "published_step":published_step,"latest_sha256":latest_hash,"caller_boundary":"all ranks reaped before next spawn",
        "moves":moves,"completed":false,"deleted_bytes":0});
    write_json(&recovery.join("receipt.json"), &receipt)?;
    for index in 0..receipt["moves"].as_array().unwrap().len() {
        ensure!(
            if let Some(expected) = &latest_hash {
                latest_path.exists() && file_hash(&latest_path)? == *expected
            } else {
                !latest_path.exists()
            },
            "latest checkpoint changed during stopped-rank recovery"
        );
        let name = receipt["moves"][index]["name"].as_str().unwrap();
        let destination = recovery.join(name);
        ensure!(!destination.exists(), "checkpoint quarantine collision");
        fs::rename(directory.join(name), &destination)?;
        fs::File::open(&directory)?.sync_all()?;
        fs::File::open(&recovery)?.sync_all()?;
        receipt["moves"][index]["moved"] = true.into();
        write_json(&recovery.join("receipt.json"), &receipt)?;
    }
    receipt["completed"] = true.into();
    write_json(&recovery.join("receipt.json"), &receipt)?;
    Ok(())
}

#[cfg(test)]
mod tests;
