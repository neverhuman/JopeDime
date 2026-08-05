//! Shared filesystem, provenance, and scheduler primitives for self-play.

use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
pub const DEFAULT_SCQ: &str = "/home/ubuntu/smart_cluster/target/release/scq";
pub const DEFAULT_EVALUATOR: &str = "/home/ubuntu/jope-dime/target/release/corpus_eval";
pub const DEFAULT_RUNNER: &str = "/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh";

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis()
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    // serde_json maps are backed by deterministic key ordering without the
    // preserve_order feature, matching Python sort_keys compact output.
    Ok(serde_json::to_vec(value)?)
}

pub fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("JSON target has no filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    output.write_all(&canonical_json_bytes(value)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read JSON {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

pub fn checkpoint_epoch(checkpoint_sha256: &str) -> Result<String> {
    if checkpoint_sha256.len() != 64
        || !checkpoint_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("checkpoint is not a SHA-256 value");
    }
    Ok(format!("checkpoint-{checkpoint_sha256}"))
}

pub fn active_checkpoint(root: &Path) -> Result<Option<(String, String)>> {
    for relative in ["training/learning.json", "training/accepted.json"] {
        let path = root.join(relative);
        if !path.is_file() {
            continue;
        }
        let value = read_json(&path)?;
        let digest = value
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("checkpoint pointer lacks checkpoint_sha256")?;
        let epoch = checkpoint_epoch(digest)?;
        if let Some(checkpoint) = value.get("checkpoint").and_then(Value::as_str) {
            let checkpoint_path = Path::new(checkpoint);
            if checkpoint_path.is_file() && sha256_file(checkpoint_path)? != digest {
                bail!("checkpoint pointer content drift: {}", path.display());
            }
        }
        return Ok(Some((digest.to_owned(), epoch)));
    }
    // Bootstrap compatibility is explicit and content-verified. The original
    // flywheel treated this immutable warm checkpoint as accepted before the
    // first promotion pointer existed. Never infer an arbitrary checkpoint:
    // require the proposal pointer's immutable legacy-adoption receipt and the
    // matching content-addressed artifact.
    let proposal_pointer = root.join("proposals/current.json");
    if proposal_pointer.is_file() {
        let value = read_json(&proposal_pointer)?;
        if value.get("legacy_adopted").and_then(Value::as_bool) == Some(true) {
            let digest = value
                .get("checkpoint_sha256")
                .and_then(Value::as_str)
                .context("legacy proposal pointer lacks checkpoint_sha256")?;
            let checkpoint = PathBuf::from(format!(
                "/home/ubuntu/jope-flywheel/training/round1/checkpoints/{digest}.model.pt"
            ));
            if checkpoint.is_file() && sha256_file(&checkpoint)? == digest {
                return Ok(Some((digest.to_owned(), checkpoint_epoch(digest)?)));
            }
            bail!(
                "legacy-adopted checkpoint artifact failed verification: {}",
                checkpoint.display()
            );
        }
    }
    Ok(None)
}

pub fn manifest_freshness_scoped(manifest: &Value) -> bool {
    let lane = manifest
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if matches!(
        lane,
        "checkpoint_gate" | "sealed_audit" | "evaluator_canary"
    ) {
        return false;
    }
    manifest.get("schema").and_then(Value::as_str) == Some("JopeFlywheelCorpusEvalShard.v1")
        && manifest
            .get("candidates")
            .and_then(Value::as_array)
            .is_some()
}

pub fn manifest_epoch(manifest: &Value) -> String {
    if let Some(epoch) = manifest.get("checkpoint_epoch").and_then(Value::as_str) {
        return epoch.to_owned();
    }
    if let Some(checkpoint) = manifest
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .or_else(|| {
            manifest
                .get("label_authority")
                .and_then(|value| value.get("model_checkpoint_sha256"))
                .and_then(Value::as_str)
        })
    {
        if checkpoint.len() == 64 {
            return format!("checkpoint-{checkpoint}");
        }
    }
    "legacy-jope-testonly".to_owned()
}

pub fn manifest_is_current(root: &Path, manifest: &Value) -> Result<bool> {
    if !manifest_freshness_scoped(manifest) {
        return Ok(true);
    }
    Ok(active_checkpoint(root)?
        .map(|(_, epoch)| manifest_epoch(manifest) == epoch)
        .unwrap_or(true))
}

pub fn scq_json(scq: &Path, arguments: &[&str]) -> Result<Value> {
    let mut command = Command::new(scq);
    command.arg("--json").args(arguments);
    let output = command
        .output()
        .with_context(|| format!("run {} --json {}", scq.display(), arguments.join(" ")))?;
    if !output.status.success() {
        bail!(
            "SCQ command failed: {}",
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    if output.stdout.iter().all(u8::is_ascii_whitespace) {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_slice(&output.stdout).context("parse SCQ JSON output")
}

pub fn run(command: &Path, arguments: &[String]) -> Result<Output> {
    Command::new(command)
        .args(arguments)
        .output()
        .with_context(|| format!("run {}", command.display()))
}

/// Per-task authority metric (R1/D2); must agree with
/// `crate::task::TaskKind::primary_metric`.
///
/// Public so callers that are about to build an evaluation protocol can ask for the
/// metric instead of keeping their own copy — `evaluation_protocol_sha256` validates
/// against this exact function, so any private duplicate is a latent fail-closed bug.
pub fn expected_evaluation_metric(task: &str) -> Result<&'static str> {
    match task {
        "regression" => Ok("r2"),
        "binary" => Ok("auc"),
        "timeseries" => Ok("negative_mase"),
        "multiclass" => Ok("macro_f1"),
        _ => bail!("unsupported evaluation task"),
    }
}

pub fn evaluation_protocol_sha256(
    task: &str,
    metric: &str,
    score_tier: &str,
    train_sha256: &str,
    test_sha256: &str,
) -> Result<String> {
    let expected = expected_evaluation_metric(task)?;
    if metric != expected || score_tier != "test_exact" {
        bail!("evaluation protocol task/metric/tier mismatch");
    }
    let value = serde_json::json!({
        "metric": metric,
        "score_tier": score_tier,
        "task": task,
        "test_sha256": test_sha256,
        "train_sha256": train_sha256,
        "version": "test-exact-v1",
    });
    Ok(sha256_bytes(&canonical_json_bytes(&value)?))
}

pub fn protocol_for_manifest(manifest: &Value) -> Result<String> {
    if let Some(protocol) = manifest.get("protocol_sha256").and_then(Value::as_str) {
        let authority = manifest
            .get("label_authority")
            .context("manifest label authority is absent")?;
        let computed = evaluation_protocol_sha256(
            manifest["task"]
                .as_str()
                .context("manifest task is absent")?,
            authority["metric"]
                .as_str()
                .context("manifest authority metric is absent")?,
            authority["score_tier"]
                .as_str()
                .context("manifest authority tier is absent")?,
            manifest["train_sha256"]
                .as_str()
                .context("manifest train hash is absent")?,
            manifest["test_sha256"]
                .as_str()
                .context("manifest test hash is absent")?,
        )?;
        if protocol != computed {
            bail!("manifest protocol hash drift");
        }
        return Ok(computed);
    }
    let authority = manifest
        .get("label_authority")
        .context("manifest label authority is absent")?;
    evaluation_protocol_sha256(
        manifest["task"]
            .as_str()
            .context("manifest task is absent")?,
        authority["metric"]
            .as_str()
            .context("manifest authority metric is absent")?,
        authority["score_tier"]
            .as_str()
            .context("manifest authority tier is absent")?,
        manifest["train_sha256"]
            .as_str()
            .context("manifest train hash is absent")?,
        manifest["test_sha256"]
            .as_str()
            .context("manifest test hash is absent")?,
    )
}

pub fn collect_json_files(directory: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    if !directory.is_dir() {
        return Ok(result);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() && recursive {
            result.extend(collect_json_files(&path, true)?);
        } else if path.extension().and_then(|value| value.to_str()) == Some("json") {
            result.push(path);
        }
    }
    result.sort();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_metric_map_agrees_with_task_authority() {
        use crate::task::TaskKind;
        for kind in [
            TaskKind::Regression,
            TaskKind::Binary,
            TaskKind::Timeseries,
            TaskKind::Multiclass,
        ] {
            assert_eq!(
                expected_evaluation_metric(kind.as_str()).unwrap(),
                kind.primary_metric()
            );
        }
        assert!(expected_evaluation_metric("multi_label").is_err());
    }

    #[test]
    fn protocol_hash_matches_frozen_python_vector() {
        let value = evaluation_protocol_sha256(
            "binary",
            "auc",
            "test_exact",
            &"1".repeat(64),
            &"2".repeat(64),
        )
        .unwrap();
        assert_eq!(value.len(), 64);
        assert_eq!(
            value,
            "d08fcb5928bacc37fc5723c08eff2ad78dd1af7ab4bf0b0bdbaa595c17d69d74"
        );
    }

    #[test]
    fn non_adaptive_lanes_are_outside_freshness() {
        let gate = serde_json::json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "evaluation_lane": "checkpoint_gate",
            "candidates": [],
        });
        assert!(!manifest_freshness_scoped(&gate));
    }
}
