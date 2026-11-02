use super::*;

pub(super) fn required_sha256(value: &Value, field: &str) -> Result<String> {
    let digest = value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("missing {field}"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a lowercase SHA-256"
    );
    Ok(digest.to_owned())
}

pub(super) fn verified_path(value: &Value, path_field: &str, hash_field: &str) -> Result<PathBuf> {
    let path = PathBuf::from(
        value
            .get(path_field)
            .and_then(Value::as_str)
            .with_context(|| format!("missing {path_field}"))?,
    );
    verify_file(&path, &required_sha256(value, hash_field)?)?;
    Ok(path)
}

pub(super) fn verify_file(path: &Path, digest: &str) -> Result<()> {
    ensure!(
        path.is_file(),
        "required file is absent: {}",
        path.display()
    );
    ensure!(
        sha256_file(path)? == digest,
        "content hash mismatch: {}",
        path.display()
    );
    Ok(())
}

pub(super) fn ensure_safe_filename(value: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(
        path.components().count() == 1
            && matches!(path.components().next(), Some(Component::Normal(_))),
        "unsafe packed filename {value}"
    );
    Ok(())
}

pub(super) fn continuation_kind(watermark: u64, last_evidence: u64) -> ContinuationKind {
    if watermark >= last_evidence.saturating_add(LABEL_STEP) {
        ContinuationKind::CoalescedLabels
    } else {
        ContinuationKind::RecentReplay
    }
}

pub(super) fn observe_intent_checkpoint(
    intent: &mut SubmissionIntent,
    current_checkpoint_sha256: &str,
) -> IntentCheckpointDisposition {
    if intent.checkpoint_sha256 == current_checkpoint_sha256 {
        intent.checkpoint_mismatch_observations = 0;
        return IntentCheckpointDisposition::Current;
    }
    intent.checkpoint_mismatch_observations =
        intent.checkpoint_mismatch_observations.saturating_add(1);
    if intent.checkpoint_mismatch_observations >= STALE_INTENT_CONFIRMATIONS {
        IntentCheckpointDisposition::Supersede
    } else {
        IntentCheckpointDisposition::Confirming(intent.checkpoint_mismatch_observations)
    }
}

pub(super) fn trainer_job_id(watermark: u64, recipe: &str, sequence: u64) -> String {
    format!("jope-selfplay-train-{watermark:09}-{recipe}-r{sequence:06}")
}

pub(super) fn validate_intent(
    intent: &SubmissionIntent,
    snapshot: &TrainingSnapshot,
    checkpoint: &Checkpoint,
) -> Result<()> {
    ensure!(
        intent.watermark <= snapshot.watermark,
        "pending trainer intent uses a future corpus snapshot"
    );
    ensure!(
        intent.checkpoint_sha256 == checkpoint.sha256,
        "pending trainer intent uses a superseded checkpoint"
    );
    ensure!(
        parse_trainer_id(&intent.job_id) == Some((intent.watermark, intent.sequence)),
        "pending trainer intent identity mismatch"
    );
    Ok(())
}

pub(super) fn render_spec(
    args: &Args,
    job_id: &str,
    seed: u64,
    snapshot: &TrainingSnapshot,
    checkpoint: &Checkpoint,
) -> Result<String> {
    let trainer_source = args.prime_source.join("jope/policy/train_v3.py");
    let mut inputs = vec![
        spec_input(
            "script",
            &args.trainer_script,
            "scripts/test_exact_train.py",
        )?,
        spec_input(
            "common",
            &args.common_script,
            "scripts/test_exact_common.py",
        )?,
        spec_input(
            "packed_source",
            &args.packed_script,
            "scripts/test_exact_packed_corpus.py",
        )?,
        spec_input(
            "manifest",
            &snapshot.manifest,
            "corpus/corpus-manifest.json",
        )?,
        spec_input(
            "checkpoint",
            &checkpoint.path,
            &format!("checkpoint/{}.model.pt", checkpoint.sha256),
        )?,
        spec_input("trainer_source", &trainer_source, "provenance/train_v3.py")?,
        spec_input(
            "packed_manifest",
            &snapshot.packed_manifest,
            "packed/manifest.json",
        )?,
        spec_input(
            "packed_catalog",
            &snapshot.packed_catalog,
            "packed/catalog-data.f32le",
        )?,
    ];
    for (index, (name, path)) in snapshot.packed_files.iter().enumerate() {
        inputs.push(spec_input(
            &format!("packed_{index}"),
            path,
            &format!("packed/{name}"),
        )?);
    }

    let output_name = format!("watermark-{:09}", snapshot.watermark);
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_string(job_id)?),
        format!("project = \"{PROJECT}\""),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        format!(
            "args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \
             \"{{input:manifest}}\", \"--output\", \"{{output:training}}/{output_name}\", \
             \"--device\", \"cuda\", \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \
             {}, \"--seed\", \"{seed}\", \"--packed-manifest\", \"{{input:packed_manifest}}\", \
             \"--packed-catalog-data\", \"{{input:packed_catalog}}\"]",
            toml_string(&args.recipe)?,
        ),
        String::new(),
    ];
    lines.extend(TRAINER.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(TRAINER.render_placement_block().lines().map(str::to_owned));
    lines.extend([
        String::new(),
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = {}",
            toml_string(&args.prime_source.to_string_lossy())?
        ),
        format!("JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\""),
        format!("OMP_NUM_THREADS = \"{}\"", TRAINER.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", TRAINER.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(TRAINER.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    let mut text = lines.join("\n");
    text.push('\n');
    for input in inputs {
        text.push_str(&format!(
            "[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n\n",
            toml_string(&input.name)?,
            toml_string(&input.source.to_string_lossy())?,
            toml_string(&input.target)?,
            toml_string(&input.sha256)?,
        ));
    }
    text.push_str("[[output]]\nname = \"training\"\nsource = \"training\"\ndirectory = true\n");
    Ok(text)
}

pub(super) fn spec_input(name: &str, source: &Path, target: &str) -> Result<SpecInput> {
    ensure!(
        source.is_file(),
        "spec input is absent: {}",
        source.display()
    );
    Ok(SpecInput {
        name: name.to_owned(),
        source: source.to_owned(),
        target: target.to_owned(),
        sha256: sha256_file(source)?,
    })
}

pub(super) fn toml_string(value: &str) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

pub(super) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            sha256_file(path)? == sha256_bytes(bytes),
            "existing trainer spec content drift: {}",
            path.display()
        );
        return Ok(());
    }
    let parent = path.parent().context("trainer spec has no parent")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("trainer spec has no filename")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(super) fn append_event(root: &Path, kind: &str, detail: Value) -> Result<()> {
    let path = root.join("ops/trainer-watchdog-events.v1.jsonl");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "schema": "JopeContinuousTrainerWatchdogEvent.v1",
            "kind": kind,
            "detail": detail,
            "created_at_ms": now_ms(),
        }),
    )?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

pub(super) fn scq_check_submit(scq: &Path, spec: &Path) -> Result<()> {
    let check = Command::new(scq)
        .arg("check")
        .arg(spec)
        .output()
        .with_context(|| format!("run {} check", scq.display()))?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && !temporarily_lacks_launch_capacity(&check_message) {
        bail!("SCQ check rejected trainer: {}", check_message.trim());
    }
    let submit = Command::new(scq)
        .arg("submit")
        .arg(spec)
        .output()
        .with_context(|| format!("run {} submit", scq.display()))?;
    let message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if !submit.status.success() && !message.to_ascii_lowercase().contains("already exists") {
        bail!("SCQ submit rejected trainer: {}", message.trim());
    }
    Ok(())
}

pub(super) fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && message.contains("no eligible launch-ready node currently")
}

pub(super) fn intent_uses_stale_contract(intent: &SubmissionIntent) -> bool {
    fs::read_to_string(&intent.spec)
        .map(|text| {
            !text.contains(&format!(
                "JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\""
            )) || text.contains("--packed-source-train")
                || text.contains("target = \"corpus/corpus.train.jsonl\"")
        })
        .unwrap_or(true)
}
