//! Explicit artifact test: actual Rust measurements on a fresh synthetic train-only table.
//! This proves diagnostic archive/reload, never neural training or production authority.
use anyhow::{Context, Result};
use jope_learning::{hwm, record::UNBOUND, sources::pipeline, state::State};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn digest(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn child(executable: &Path, args: &[&str], cwd: &Path) -> Result<(u32, Output)> {
    let process = Command::new("nice")
        .args(["-n", "15"])
        .arg(executable)
        .args(args)
        .current_dir(cwd)
        .env_remove("JOPE_LEARNING_URL")
        .env_remove("JOPE_DATASETS_URL")
        .env_remove("JOPE_MODELS_URL")
        .env_remove("DOPE_WEIGHTS")
        .env("JOPE_STATE", cwd.join("unused-global-state"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let pid = process.id();
    Ok((pid, process.wait_with_output()?))
}

fn succeeded(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn finite512(value: &Value) {
    let values = value.as_array().expect("explicit host vector");
    assert_eq!(values.len(), 512);
    assert!(values
        .iter()
        .all(|x| x.as_f64().is_some_and(f64::is_finite)));
}

#[test]
#[ignore = "requires explicit JOPE_ONE_SHOT_EXE and released Dope weights; real bounded CPU execution"]
fn bounded_one_shot_measures_reloads_and_refuses_reuse_and_tampering() -> Result<()> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let executable = fs::canonicalize(
        std::env::var_os("JOPE_ONE_SHOT_EXE")
            .context("set JOPE_ONE_SHOT_EXE to the built landmark_one_shot example")?,
    )?;
    let weights = std::env::var_os("DOPE_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            manifest.join("../../deploy/appliance/assets/dope-mlc-v3/model.safetensors")
        });
    let weights = fs::canonicalize(weights)?;
    let git = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest)
        .output()?;
    succeeded(&git);
    let commit = String::from_utf8(git.stdout)?.trim().to_owned();
    assert_eq!(commit.len(), 40);
    let base = manifest.join("../../target/landmarks/p6-one-shot-tests");
    fs::create_dir_all(&base)?;
    let temporary = tempfile::tempdir_in(base)?;
    let existing = std::env::var_os("JOPE_ONE_SHOT_RUN");
    let root = existing
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().join("one-shot"));
    let args = [
        "--output",
        root.to_str().unwrap(),
        "--weights",
        weights.to_str().unwrap(),
        "--source-commit",
        &commit,
        "--device",
        "cpu",
    ];
    // Reuse an explicitly retained one-shot when validating the owner's single run budget.
    // Verification/reuse/tamper subprocesses do not run another acquisition round.
    let pid = if existing.is_some() {
        json(&root.join("receipt.json"))?["pid"]
            .as_u64()
            .context("missing retained PID")? as u32
    } else {
        let (pid, output) = child(&executable, &args, temporary.path())?;
        succeeded(&output);
        pid
    };
    let receipt_path = root.join("receipt.json");
    let receipt = json(&receipt_path)?;
    assert_eq!(receipt["schema"], "jope.landmark-one-shot/v1");
    assert_eq!(receipt["pid"], pid);
    assert_ne!(pid, std::process::id());
    assert_eq!(receipt["binary_sha256"], digest(&executable)?);
    assert_eq!(receipt["source_commit"], commit);
    assert_eq!(receipt["bounded_one_shot"], true);
    assert_eq!(receipt["rounds"], 1);
    for field in [
        "continuous_service_demonstrated",
        "production_admission",
        "promotion",
        "remote_global_mutations",
        "teaching_eligible",
    ] {
        assert_eq!(receipt[field], false, "{field}");
    }
    assert_eq!(
        receipt["production_snapshot_publication"],
        "held_missing_authenticated_evidence"
    );
    assert_eq!(receipt["cas_refusal_preserved"], true);
    assert_eq!(receipt["reload"]["verified"], true);
    assert_ne!(receipt["reload"]["child_pid"], receipt["pid"]);

    let run_path = root.join("run/receipt.json");
    let verified = jope_pipeline::evidence::verify_champion_run(&run_path)?;
    let native = &verified.receipt;
    assert_eq!(native.search.evaluations, 2);
    assert_eq!(native.search.seconds, 15);
    assert_eq!(native.search.cuts.len(), 1);
    let cut = &native.search.cuts[0];
    assert_eq!(cut.completed_evaluations, 2);
    assert!(cut.successful_measurements > 0);
    assert_eq!(
        cut.successful_measurements + cut.failed_or_unavailable_measurements,
        cut.completed_evaluations
    );
    assert!(native
        .search
        .best
        .as_ref()
        .unwrap()
        .validation_rmse
        .is_finite());
    let (mut candidates, _) = pipeline::scan(&run_path, None)?;
    assert_eq!(candidates.len(), 1);
    let record = candidates.remove(0).record;
    assert_eq!(
        record.identity.comparison.as_ref().unwrap().stage,
        pipeline::STAGE
    );
    assert!(!record.benchmark_valid && !record.sealed && !record.replayable);
    assert_eq!(record.evaluator.sha256, UNBOUND);

    let encoding = json(&root.join("run/encoding/cut-000.json"))?;
    assert_eq!(encoding["schema"], "jope.landmark-encoding/v1");
    assert_eq!(encoding["status"], "none");
    assert_eq!(encoding["version"], 0);
    assert_eq!(encoding["device"], "cpu");
    assert_eq!(encoding["mask"], serde_json::to_value([false; 23])?);
    assert_eq!(encoding["consumer"]["mode"], "encoding-only");
    assert_eq!(encoding["embedding"], encoding["consumer_embedding"]);
    assert_eq!(
        encoding["weights_sha256"],
        starforge::dope_encoder::DOPE_WEIGHTS_SHA256
    );
    finite512(&encoding["embedding"]);
    let events = fs::read_to_string(root.join("run/encoding/cut-000.jsonl"))?;
    let events = events
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["embedding"], encoding["embedding"]);
    assert_eq!(
        events[0]["receipt"]["conditioning_digest"],
        encoding["conditioning_digest"]
    );
    assert_eq!(events[0]["receipt"]["version"], 0);

    let state = State::at(&root.join("state"));
    let before = hwm::verify(&state, &[])?;
    assert!(before.ok(), "{:?}", before.breaks);
    assert_eq!(
        (
            before.initialized,
            before.improved,
            before.accepted,
            before.rejected
        ),
        (1, 0, 1, 0)
    );
    assert_eq!(
        (
            before.identities,
            before.pointers_on_disk,
            before.blobs_checked
        ),
        (1, 1, 1)
    );
    let pointers = state.pointers()?;
    assert_eq!(pointers.len(), 1);
    assert_eq!(
        pointers[0].1.provenance.record_sha256,
        record.provenance.record_sha256
    );
    assert_eq!(
        state.read_blob(&record.provenance.record_sha256)?.unwrap(),
        fs::read(&run_path)?
    );
    let archive = json(&root.join("archive/manifest.json"))?;
    assert_eq!(archive["schema"], hwm::FREEZE_SCHEMA);
    assert_eq!(archive["scope"], "diagnostic_hwm_archive");
    assert_eq!(archive["teaching_eligible"], false);
    assert_eq!(archive["include_unbound"], true);
    assert_eq!(archive["included"], 1);
    assert_eq!(
        archive["entries"][0]["record_sha256"],
        record.provenance.record_sha256
    );
    let archived = root
        .join("archive")
        .join(archive["entries"][0]["blob"].as_str().unwrap());
    assert_eq!(digest(&archived)?, record.provenance.record_sha256);
    assert!(!root.join("state/selfplay/authority/ROOT.json").exists());
    assert!(!temporary.path().join("unused-global-state").exists());

    let checkpoint_path = root.join("checkpoint.json");
    let checkpoint_before = fs::read(&checkpoint_path)?;
    let mut checkpoint: Value = serde_json::from_slice(&checkpoint_before)?;
    assert_eq!(
        checkpoint["schema"],
        "jope.landmark-diagnostic-checkpoint/v1"
    );
    assert_eq!(checkpoint["pipeline_receipt_sha256"], digest(&run_path)?);
    assert_eq!(
        checkpoint["encoding_sha256"],
        digest(&root.join("run/encoding/cut-000.json"))?
    );
    let (verify_pid, output) = child(
        &executable,
        &["--verify-checkpoint", checkpoint_path.to_str().unwrap()],
        temporary.path(),
    )?;
    succeeded(&output);
    assert_ne!(verify_pid, pid);
    let _: Value = serde_json::from_slice(&output.stdout)?;

    let receipt_before = fs::read(&receipt_path)?;
    let (_, reused) = child(&executable, &args, temporary.path())?;
    assert!(!reused.status.success(), "existing output must be refused");
    assert_eq!(fs::read(&receipt_path)?, receipt_before);
    let wrong = if checkpoint["encoding_sha256"] == "0".repeat(64) {
        "1"
    } else {
        "0"
    };
    checkpoint["encoding_sha256"] = Value::String(wrong.repeat(64));
    let corrupt = temporary.path().join("tampered-checkpoint.json");
    fs::write(&corrupt, serde_json::to_vec(&checkpoint)?)?;
    let (_, refused) = child(
        &executable,
        &["--verify-checkpoint", corrupt.to_str().unwrap()],
        temporary.path(),
    )?;
    assert!(
        !refused.status.success(),
        "changed retained digest must be refused"
    );
    assert_eq!(fs::read(checkpoint_path)?, checkpoint_before);
    assert_eq!(state.pointers()?, pointers);
    assert!(hwm::verify(&state, &[])?.ok());
    Ok(())
}
