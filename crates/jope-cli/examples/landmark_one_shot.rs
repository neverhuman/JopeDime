//! One bounded, unqualified local execution. This installs no production admission or authority.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_learning::{hwm, sources::pipeline, state::State};
use jope_pipeline::{
    run::{Policy, RunOptions},
    search::Budget,
};
use jope_runtime::landmark_args::{LandmarkConsumer, LandmarkMode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    weights: Option<PathBuf>,
    #[arg(long)]
    source_commit: Option<String>,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    verify_checkpoint: Option<PathBuf>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    root: PathBuf,
    source_commit: String,
    binary_sha256: String,
    pipeline_receipt_sha256: String,
    encoding_sha256: String,
    archive_manifest_sha256: String,
    pointer_rel: String,
    pointer_sha256: String,
    input_train_sha256: String,
}
fn now_ms() -> Result<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}
fn hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        digest.update(&chunk[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn write_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}
fn fixture(root: &Path) -> Result<PathBuf> {
    let dir = root.join("input");
    fs::create_dir(&dir)?;
    let mut csv = String::new();
    for i in 0..80 {
        let a = i as f64 / 10.;
        let b = (i % 7) as f64;
        let c = (i as f64 * 0.31).cos();
        let y = 2. * a - 0.2 * b + c * c;
        csv.push_str(&format!("{a:.12},{b:.12},{c:.12},{y:.12}\n"));
    }
    fs::write(dir.join("train.csv"), &csv)?;
    fs::write(dir.join("test.csv"), b"")?;
    let train_sha = jope_runtime::sha256_hex(csv.as_bytes());
    let canonical = jope_runtime::sha256_hex(&serde_json::to_vec(&json!([
        "jope.p6-synthetic-train/v1",
        train_sha
    ]))?);
    let value = json!({"schema":"jope.dataset-catalog/v1","id":format!("synth_{}",&canonical[..16]),
        "dataset_hash":&canonical[..16],"canonical_sha256":canonical,"display_name":"P6 fresh synthetic TRAIN only",
        "task":"regression","origin":"synthetic","shape":{"rows":80,"features":3,"categorical_features":0,"columns":4},
        "target":{"name":"target","index":-1,"type":"numeric"},
        "files":{"train":{"sha256":train_sha,"bytes":csv.len(),"rows":80,"columns":4,"has_header":false},
            "test":{"sha256":jope_runtime::sha256_hex(b""),"bytes":0,"rows":0,"columns":4,"has_header":false}},
        "split":{"method":"fresh synthetic TRAIN only; no outer data","seed":20261027,"train_rows":80,"test_rows":0},
        "provenance":{"platform":"bounded-local-diagnostic","license":{"status":"unknown","note":"fresh generated fixture; no corpus qualification claim"}},
        "generator":{"name":"jope.p6-synthetic-train","version":"1","seed":20261027},"collections":[],"aliases":[]});
    jope_datasets::catalog::CatalogRow::validate_value(&value)
        .map_err(|e| anyhow::anyhow!("{}", e.join("; ")))?;
    write_new(&dir.join("metadata.json"), &value)?;
    Ok(dir)
}
fn verify(path: &Path) -> Result<Value> {
    let bytes = fs::read(path)?;
    let checkpoint: Checkpoint = serde_json::from_slice(&bytes)?;
    ensure!(
        checkpoint.schema == "jope.landmark-diagnostic-checkpoint/v1",
        "unknown diagnostic checkpoint"
    );
    let root = &checkpoint.root;
    let state = State::at(&root.join("state"));
    let digest = jope_runtime::sha256_hex(&bytes);
    ensure!(
        state.read_blob(&digest)?.as_deref() == Some(bytes.as_slice()),
        "checkpoint bytes not retained by local diagnostic store"
    );
    ensure!(
        checkpoint.binary_sha256 == hash(&std::env::current_exe()?)?,
        "reload binary identity changed"
    );
    for (rel, expected) in [
        ("run/receipt.json", &checkpoint.pipeline_receipt_sha256),
        ("run/encoding/cut-000.json", &checkpoint.encoding_sha256),
        ("archive/manifest.json", &checkpoint.archive_manifest_sha256),
        ("input/train.csv", &checkpoint.input_train_sha256),
    ] {
        ensure!(
            hash(&root.join(rel))? == *expected,
            "diagnostic checkpoint artifact changed: {rel}"
        );
    }
    ensure!(
        checkpoint.pointer_rel.starts_with("hwm/")
            && checkpoint
                .pointer_rel
                .split('/')
                .all(jope_learning::state::plain_segment),
        "unsafe diagnostic pointer"
    );
    ensure!(
        hash(&state.root().join(&checkpoint.pointer_rel))? == checkpoint.pointer_sha256,
        "diagnostic pointer changed"
    );
    let native = jope_pipeline::evidence::verify_champion_run(&root.join("run/receipt.json"))?;
    ensure!(
        native.receipt.git_sha.as_deref() == Some(&checkpoint.source_commit),
        "source identity changed"
    );
    let encoding = read(&root.join("run/encoding/cut-000.json"))?;
    ensure!(
        encoding["status"] == "none"
            && encoding["consumer"]["mode"] == "encoding-only"
            && encoding["outcome"] == "complete",
        "expected complete raw zero-landmark encoding"
    );
    let values = encoding["embedding"]
        .as_array()
        .context("missing encoding values")?;
    ensure!(
        values.len() == 512
            && values
                .iter()
                .all(|v| v.as_f64().is_some_and(f64::is_finite)),
        "invalid raw encoding"
    );
    ensure!(
        values
            == encoding["consumer_embedding"]
                .as_array()
                .context("missing direct output")?,
        "direct output changed"
    );
    ensure!(hwm::verify(&state, &[])?.ok(), "local ledger reload failed");
    let archive = read(&root.join("archive/manifest.json"))?;
    ensure!(
        archive["scope"] == "diagnostic_hwm_archive" && archive["teaching_eligible"] == false,
        "archive is not diagnostic-only"
    );
    for entry in archive["entries"]
        .as_array()
        .context("archive entries missing")?
    {
        let rel = entry["blob"]
            .as_str()
            .context("archive blob path missing")?;
        ensure!(
            rel.starts_with("blobs/sha256/")
                && rel.split('/').all(jope_learning::state::plain_segment),
            "unsafe archive blob path"
        );
        ensure!(
            hash(&root.join("archive").join(rel))?
                == entry["record_sha256"]
                    .as_str()
                    .context("archive hash missing")?,
            "archive blob changed"
        );
    }
    ensure!(
        !state.root().join("selfplay/authority/ROOT.json").exists()
            && !state.root().join("replay/adopted-policy.json").exists(),
        "diagnostic execution must not install production authority"
    );
    Ok(
        json!({"verified":true,"pid":std::process::id(),"checkpoint_sha256":digest,"source_commit":checkpoint.source_commit,
        "binary_sha256":checkpoint.binary_sha256,"production_admission":false,"teaching_eligible":false}),
    )
}
fn execute(args: Args) -> Result<Value> {
    let started = now_ms()?;
    let elapsed = Instant::now();
    let output = args.output.context("--output required")?;
    let weights = args.weights.context("--weights required")?;
    let source = args.source_commit.context("--source-commit required")?;
    ensure!(
        source.len() == 40
            && source
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "source must be a full immutable Git commit"
    );
    ensure!(
        args.device == "cpu"
            || args
                .device
                .strip_prefix("cuda:")
                .is_some_and(|s| s.parse::<usize>().is_ok()),
        "device must be cpu or cuda:N"
    );
    // A create-only fresh output isolates this one-shot from configured production stores.
    fs::create_dir(&output).context("one-shot output must be new, with an existing parent")?;
    let root = output.canonicalize()?;
    let input = fixture(&root)?;
    let state = State::at(&root.join("state"));
    let counters_before = hwm::verify(&state, &[])?;
    ensure!(
        counters_before.ok() && counters_before.events == 0,
        "one-shot state must be empty"
    );
    let mut options = RunOptions::new(input.to_string_lossy(), root.join("run"));
    options.budget = Budget {
        evaluations: 2,
        seconds: 15,
    };
    options.cuts = 1;
    options.threads = 1;
    options.cpu_ceiling_millicores = Some(1000);
    options.policy = Policy::Skip;
    options.source_commit = Some(source.clone());
    options.encode_device = args.device.clone();
    options.landmarks.landmark_mode = Some(LandmarkMode::None);
    options.landmarks.landmark_consumer = Some(LandmarkConsumer::Direct);
    options.landmarks.dope_weights = Some(weights);
    let run = jope_pipeline::run::run(&options)?;
    ensure!(
        run.ok && run.search.cuts.len() == 1 && run.search.cuts[0].successful_measurements > 0,
        "bounded native round produced no measurement"
    );
    let path = root.join("run/receipt.json");
    let (candidates, scan) = pipeline::scan(&path, Some("p6-one-shot"))?;
    ensure!(
        candidates.len() == 1
            && !candidates[0].record.benchmark_valid
            && !candidates[0].record.replayable
            && !candidates[0].record.sealed,
        "unexpected qualified/outer observation"
    );
    let source_value = pipeline::scan_source(&path, &scan);
    let ingest = hwm::ingest(&state, None, candidates, source_value)?;
    ensure!(
        ingest.accepted == 1 && ingest.rejected == 0 && ingest.mirror.is_none(),
        "local ingest did not initialize exactly one observation"
    );
    let verification = hwm::verify(&state, &[])?;
    ensure!(verification.ok(), "local ledger verification failed");
    let pointers = state.pointers()?;
    ensure!(pointers.len() == 1, "expected one local HWM pointer");
    let (key, record) = &pointers[0];
    let pointer = state.pointer_path(key)?;
    let pointer_sha = hash(&pointer)?;
    let cas_refusal_preserved = {
        let _lock = state.lock("hwm")?;
        let refused = state
            .write_pointer(key, record, Some(&"0".repeat(64)))
            .is_err();
        refused && hash(&pointer)? == pointer_sha
    };
    ensure!(
        cas_refusal_preserved,
        "stale HWM pointer CAS did not preserve state"
    );
    let archive = hwm::freeze(&state, &root.join("archive"), true)?;
    ensure!(
        archive.included == 1 && !archive.teaching_eligible,
        "diagnostic archive mismatch"
    );
    let executable = std::env::current_exe()?;
    let binary_sha = hash(&executable)?;
    let checkpoint = Checkpoint {
        schema: "jope.landmark-diagnostic-checkpoint/v1".into(),
        root: root.clone(),
        source_commit: source.clone(),
        binary_sha256: binary_sha.clone(),
        pipeline_receipt_sha256: hash(&path)?,
        encoding_sha256: hash(&root.join("run/encoding/cut-000.json"))?,
        archive_manifest_sha256: hash(&root.join("archive/manifest.json"))?,
        pointer_rel: key.rel(),
        pointer_sha256: pointer_sha,
        input_train_sha256: hash(&input.join("train.csv"))?,
    };
    let checkpoint_path = root.join("checkpoint.json");
    write_new(&checkpoint_path, &checkpoint)?;
    let checkpoint_sha = state.put_blob(&fs::read(&checkpoint_path)?)?;
    let child = Command::new(&executable)
        .arg("--verify-checkpoint")
        .arg(&checkpoint_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let child_pid = child.id();
    let result = child.wait_with_output()?;
    ensure!(
        result.status.success(),
        "fresh-process reload failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let reloaded: Value = serde_json::from_slice(&result.stdout)?;
    ensure!(
        reloaded["verified"] == true && reloaded["checkpoint_sha256"] == checkpoint_sha,
        "reload checkpoint identity differs"
    );
    let counters_after = hwm::verify(&State::at(&root.join("state")), &[])?;
    let value = json!({"schema":"jope.landmark-one-shot/v1","bounded_one_shot":true,"rounds":1,
        "continuous_service_demonstrated":false,"production_admission":false,"promotion":false,
        "production_snapshot_publication":"held_missing_authenticated_evidence","remote_global_mutations":false,"teaching_eligible":false,
        "pid":std::process::id(),"source_commit":source,"binary_sha256":binary_sha,"argv":std::env::args_os().map(|a|a.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "started_ms":started,"finished_ms":now_ms()?,"wall_seconds":elapsed.elapsed().as_secs_f64(),"device":args.device,
        "input_grade":"fresh_synthetic_train_only","input_rows":80,"outer_test_rows":0,"budget":{"cuts":1,"evaluations":2,"seconds":15,"workers":1},
        "input_train_sha256":checkpoint.input_train_sha256,"counters_before":counters_before,"ingest":ingest,"verification":verification,
        "archive":archive,"checkpoint_sha256":checkpoint_sha,"reload":{"child_pid":child_pid,"exit":result.status.code(),"verified":true,"receipt":reloaded},
        "counters_after":counters_after,"encoding":read(&root.join("run/encoding/cut-000.json"))?,"cas_refusal_preserved":cas_refusal_preserved,
        "measurement":{"selected_genome":run.search.selected_genome,"best":run.search.best,"cut":run.search.cuts[0]},
        "scope":"fresh local native GP acquisition/ingest/archive proof; no neural optimizer, retention or accepted-policy claim"});
    write_new(&root.join("receipt.json"), &value)?;
    Ok(value)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let value = if let Some(path) = args.verify_checkpoint {
        verify(&path)?
    } else {
        execute(args)?
    };
    serde_json::to_writer_pretty(std::io::stdout(), &value)?;
    println!();
    Ok(())
}
