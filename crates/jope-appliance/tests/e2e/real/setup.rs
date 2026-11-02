use super::*;

pub(super) fn input() -> Result<Vec<u8>> {
    if let Some(path) = std::env::var_os("JOPE_INT01_CSV") {
        let bytes = fs::read(path)?;
        ensure!(
            jope_runtime::sha256_hex(&bytes) == std::env::var("JOPE_INT01_CSV_SHA256")?,
            "real source CSV hash mismatch"
        );
        let root = PathBuf::from(
            std::env::var_os("JOPE_GOLDEN_DIR")
                .context("packet-backed input requires JOPE_GOLDEN_DIR")?,
        );
        ensure!(
            jope_runtime::sha256_hex(&fs::read(root.join("index.json"))?)
                == "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393",
            "golden index pin mismatch"
        );
        let context = std::env::var("JOPE_INT01_CONTEXT_SHA256")?;
        let index: Value = serde_json::from_slice(&fs::read(root.join("index.json"))?)?;
        let relative = format!("contexts/{context}.safetensors");
        let expected = &index["files"][&relative];
        let context_bytes = fs::read(root.join(&relative))?;
        ensure!(
            context_bytes.len() as u64
                == expected["bytes"]
                    .as_u64()
                    .context("context missing from packet")?
                && jope_runtime::sha256_hex(&context_bytes)
                    == expected["sha256"]
                        .as_str()
                        .context("missing context digest")?,
            "context packet mismatch"
        );
        return Ok(bytes);
    }
    // Original-label rows from the existing public TRAIN fixture, no generated y.
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    ensure!(
        jope_runtime::sha256_hex(raw) == SOURCE_SHA,
        "committed TRAIN hash changed"
    );
    let mut names: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
    names.push("target".into());
    let mut csv = names.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw)?.lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    Ok(csv.into_bytes())
}
pub(super) fn directory() -> Result<tempfile::TempDir> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    fs::create_dir_all(&root)?;
    Ok(tempfile::tempdir_in(root)?)
}
pub(super) fn spec(dataset: jope_api::ids::DatasetId) -> RunSpec {
    let mut spec = bounded_synthetic_spec();
    spec.dataset = dataset;
    spec.target = "target".into();
    spec.guidance = jope_api::Guidance::Auto;
    spec.device = "auto".into();
    spec.seed = DecimalU64(73);
    spec.languages = vec![jope_api::Language::Rust, jope_api::Language::Python];
    spec.budget.evaluations = Some(DecimalU64(1));
    spec.budget.wall_seconds = Some(DecimalU64(120));
    spec
}
pub(super) fn initialize(home: &Path, bytes: &[u8]) -> Result<(Arc<InProcessEngine>, RunSpec)> {
    // Other Cargo jobs share a build cache. Pin this executable before starting
    // a long-lived engine or spawning workers from it.
    let pinned = home.join("jopedime");
    let executable = std::env::var_os("JOPE_INT01_EXECUTABLE_PIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(BIN));
    fs::copy(&executable, &pinned)?;
    let digest = jope_runtime::sha256_hex(&fs::read(&pinned)?);
    ensure!(
        digest == jope_runtime::sha256_hex(&fs::read(&executable)?),
        "appliance changed while being pinned"
    );
    let version = Command::new(&pinned)
        .args(["--version", "--json"])
        .output()?;
    ensure!(
        version.status.success(),
        "pinned appliance version check failed"
    );
    let version: Value = serde_json::from_slice(&version.stdout)?;
    if let Ok(expected) = std::env::var("JOPE_INT01_SOURCE_HEAD") {
        ensure!(
            version["source_sha40"] == expected,
            "appliance was built from another source commit"
        );
    }
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root)?;
        fs::write(
            root.join(format!("executable-{digest}.json")),
            serde_json::to_vec_pretty(&version)?,
        )?;
    }
    let engine = Arc::new(InProcessEngine::open_with_worker(home, &pinned)?);
    let dataset = register::register(engine.store(), bytes, &Default::default())?;
    Ok((engine, spec(dataset.id)))
}
pub(super) fn batch_command(home: &Path, spec: &RunSpec) -> Command {
    let mut command = Command::new(home.join("jopedime"));
    command
        .env("JOPEDIME_HOME", home)
        .env_remove("JOPEDIME_ENDPOINT")
        .args([
            "run",
            "--json",
            "--wait",
            "1",
            "--dataset",
            &spec.dataset.to_string(),
            "--target",
            "target",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"1","wall_seconds":"120"}"#,
            "--landmarks",
            "none",
            "--guidance",
            "auto",
            "--language",
            "rust,python",
            "--require-all-languages",
            "--device",
            "auto",
            "--seed",
            "73",
            "--idempotency-key",
            KEY,
        ]);
    if let Some(digest) = &spec.plan_digest {
        command.args(["--plan-digest", digest]);
    }
    command
}
pub(super) fn snapshot(output: std::process::Output) -> Result<RunSnapshot> {
    ensure!(
        output.status.success(),
        "batch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout)?;
    let snapshot: RunSnapshot =
        serde_json::from_value(payload.get("run").unwrap_or(&payload).clone())?;
    ensure!(
        snapshot.state == jope_api::RunState::Completed,
        "batch did not complete: {:?}",
        snapshot.state
    );
    ensure!(
        snapshot
            .best
            .as_ref()
            .is_some_and(|b| b.cv_value.is_finite() && b.model_artifact.is_some()),
        "native champion missing"
    );
    Ok(snapshot)
}

pub(super) fn retain_execution(
    engine: &InProcessEngine,
    output: &std::process::Output,
    label: &str,
) -> Result<()> {
    let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") else {
        return Ok(());
    };
    let root = PathBuf::from(root);
    fs::create_dir_all(&root)?;
    fs::write(root.join(format!("{label}-stdout.json")), &output.stdout)?;
    fs::write(root.join(format!("{label}-stderr.txt")), &output.stderr)?;
    let id = serde_json::from_slice::<Value>(&output.stdout)
        .ok()
        .and_then(|value| {
            serde_json::from_value::<RunSnapshot>(value.get("run").unwrap_or(&value).clone()).ok()
        })
        .map(|s| s.run_id)
        .or_else(|| {
            String::from_utf8_lossy(&output.stderr)
                .split_whitespace()
                .find_map(|s| {
                    s.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .parse::<jope_api::ids::RunId>()
                        .ok()
                })
        });
    if let Some(id) = id {
        if let Some(row) = engine.store().get(&RecordKey::Run(id.clone()))? {
            fs::write(
                root.join(format!("{label}-run-record.json")),
                serde_json::to_vec_pretty(&row)?,
            )?;
            let blobs = root.join(format!("{label}-blobs"));
            fs::create_dir_all(&blobs)?;
            for reference in &row.blobs {
                let bytes = engine.store().read_blob(reference)?;
                fs::write(blobs.join(jope_runtime::sha256_hex(&bytes)), bytes)?;
            }
            let snap: RunSnapshot = serde_json::from_value(row.data["snapshot"].clone())?;
            let events = (1..=snap.event_watermark.0)
                .map(|sequence| {
                    engine
                        .store()
                        .get(&RecordKey::Event {
                            run: id.clone(),
                            sequence,
                        })
                        .map(|r| r.map(|r| r.data))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            fs::write(
                root.join(format!("{label}-durable-events.json")),
                serde_json::to_vec_pretty(&events)?,
            )?;
        }
    }
    Ok(())
}
