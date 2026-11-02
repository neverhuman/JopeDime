use super::*;

fn input_path(root: &Path, path: &Path) -> Result<PathBuf> {
    crate::manifest::resolve(
        root,
        if path.is_absolute() {
            path.strip_prefix(root)?
        } else {
            path
        },
    )
}

/// Bind input payloads as well as the manifests naming them. This deliberately
/// pays the verification read cost until a sealed verifier receipt is qualified.
/// It does not qualify the dynamic-library runtime closure.
pub fn input_artifacts(root: &Path, task: &Value) -> Result<BTreeMap<PathBuf, String>> {
    request_artifacts(root, task, &json!({}))
}

pub(crate) fn request_artifacts(
    root: &Path,
    task: &Value,
    environment: &Value,
) -> Result<BTreeMap<PathBuf, String>> {
    let mut paths: Vec<(PathBuf, Option<String>)> = Vec::new();
    let digest = |value: &Value| -> Result<String> {
        let value = value.as_str().context("input content digest absent")?;
        ensure!(
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid input content digest"
        );
        Ok(value.to_owned())
    };
    let corpus = input_path(
        root,
        Path::new(task["corpus"].as_str().context("task corpus absent")?),
    )?;
    let index_bytes = fs::read(corpus.join("index.json"))?;
    let index: Value = serde_json::from_slice(&index_bytes)?;
    if task["kind"] == "Train" {
        crate::corpus_projection::verify_admitted_context_closure(&index)?;
    }
    paths.push((corpus.join("index.json"), Some(hash(&index_bytes))));
    paths.push((
        corpus.join("vocabulary.json"),
        Some(digest(&index["vocabulary_sha256"])?),
    ));
    if index["schema"] == "parallel.teaching_shards.v1" {
        paths.push((corpus.join("provenance.json"), None));
        paths.push((
            corpus.join("groups.bin"),
            Some(digest(&index["groups_sha256"])?),
        ));
        for shard in index["shards"].as_array().context("corpus shards absent")? {
            let path = input_path(
                root,
                Path::new(shard["path"].as_str().context("shard path absent")?),
            )?;
            paths.push((path.join("shard.json"), Some(digest(&shard["sha256"])?)));
            paths.push((
                path.join("records.bin"),
                Some(digest(&shard["records_sha256"])?),
            ));
            if task["kind"] == "Proposal" {
                paths.push((
                    path.join("labels.jsonl"),
                    Some(digest(&shard["labels_sha256"])?),
                ));
            }
        }
    } else {
        ensure!(
            index["schema"] == "parallel.teaching_corpus.v1",
            "unsupported corpus input contract"
        );
        paths.push((
            corpus.join("records.bin"),
            Some(digest(&index["records_sha256"])?),
        ));
        if task["kind"] == "Proposal" {
            let bytes = fs::read(corpus.join("provenance.json"))?;
            let provenance: Value = serde_json::from_slice(&bytes)?;
            paths.push((corpus.join("provenance.json"), Some(hash(&bytes))));
            // W1 round 8: the labels digest the index binds; a legacy index
            // without one keeps the provenance's (its labels stay unbound, and
            // teacher_programs refuses them).
            let labels = match index.get("labels_sha256") {
                Some(bound) => digest(bound)?,
                None => digest(&provenance["labels_sha256"])?,
            };
            paths.push((corpus.join("labels.jsonl"), Some(labels)));
        }
    }
    for context in index["contexts"]
        .as_array()
        .context("corpus contexts absent")?
    {
        paths.push((
            input_path(
                root,
                Path::new(context["path"].as_str().context("context path absent")?),
            )?,
            Some(digest(&context["sha256"])?),
        ));
        if task["kind"] == "Proposal" {
            let prepared = input_path(
                root,
                Path::new(
                    context["prepared"]
                        .as_str()
                        .context("prepared path absent")?,
                ),
            )?;
            paths.push((
                prepared.join("prepared.json"),
                Some(digest(&context["prepared_sha256"])?),
            ));
        }
    }
    let checkpoint = match task["kind"].as_str() {
        Some("Train") => {
            let run = input_path(
                root,
                Path::new(task["run"].as_str().context("task run absent")?),
            )?;
            paths.push((run.join("run.json"), None));
            if task["resume"] == true {
                paths.push((run.join("initialization.json"), None));
                let launch = input_path(
                    root,
                    Path::new(task["launch"].as_str().context("task launch absent")?),
                )?;
                let pointer = launch.join("resume-input.json");
                let bytes = fs::read(&pointer)?;
                let latest: Value = serde_json::from_slice(&bytes)?;
                paths.push((pointer, Some(hash(&bytes))));
                let checkpoint = input_path(
                    root,
                    Path::new(
                        latest["checkpoint"]
                            .as_str()
                            .context("resume checkpoint absent")?,
                    ),
                )?;
                paths.push((
                    checkpoint.join("state.json"),
                    Some(digest(&latest["state_sha256"])?),
                ));
                Some(
                    latest["checkpoint"]
                        .as_str()
                        .context("resume checkpoint absent")?
                        .to_owned(),
                )
            } else {
                task["initialize"].as_str().map(str::to_owned)
            }
        }
        Some("Proposal" | "Monitor") => Some(
            task["checkpoint"]
                .as_str()
                .context("task checkpoint absent")?
                .to_owned(),
        ),
        _ => anyhow::bail!("unsupported resident task input contract"),
    };
    if task["kind"] == "Proposal" {
        let output = input_path(
            root,
            Path::new(task["output"].as_str().context("proposal output absent")?),
        )?;
        let bytes = fs::read(output.join("run.json"))?;
        let run: Value = serde_json::from_slice(&bytes)?;
        paths.push((output.join("run.json"), Some(hash(&bytes))));
        let selection = output
            .parent()
            .context("proposal parent absent")?
            .join("coverage-selection.json");
        if !run["coverage_selection_sha256"].is_null() {
            paths.push((selection, Some(digest(&run["coverage_selection_sha256"])?)));
        } else {
            ensure!(
                !selection.try_exists()?,
                "unbound proposal coverage selection"
            );
        }
        let champion = environment["JOPE_CHAMPION_SNAPSHOT"]
            .as_str()
            .context("proposal champion snapshot absent")?;
        paths.push((
            input_path(root, Path::new(champion))?,
            Some(digest(&run["champion_snapshot_sha256"])?),
        ));
    }
    if let Some(checkpoint) = checkpoint {
        let checkpoint = input_path(root, Path::new(&checkpoint))?;
        let bytes = fs::read(checkpoint.join("state.json"))?;
        let state: Value = serde_json::from_slice(&bytes)?;
        paths.push((checkpoint.join("state.json"), Some(hash(&bytes))));
        for (name, field) in [
            ("config.json", "config_sha256"),
            ("model.ot", "model_sha256"),
        ] {
            paths.push((checkpoint.join(name), Some(digest(&state[field])?)));
        }
        if task["kind"] == "Train" {
            for (name, field) in [
                ("optimizer.ot", "optimizer_sha256"),
                ("optimizer.json", "optimizer_metadata_sha256"),
            ] {
                paths.push((checkpoint.join(name), Some(digest(&state[field])?)));
            }
        }
    }
    for field in ["panel", "replay_policy"] {
        if let Some(path) = task[field].as_str() {
            let path = input_path(root, Path::new(path))?;
            if field == "replay_policy" {
                let bytes = fs::read(&path)?;
                let policy: Value = serde_json::from_slice(&bytes)?;
                if policy["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA {
                    for (reference, sha) in
                        crate::checkpoint_recovery::continuation_references(&policy)?
                    {
                        paths.push((input_path(root, &reference)?, Some(sha)));
                    }
                }
                paths.push((path, Some(hash(&bytes))));
            } else {
                paths.push((path, None));
            }
        }
    }
    paths
        .into_iter()
        .map(|(path, expected)| {
            let checked = input_path(root, &path)?;
            let actual = file_hash(&checked)?;
            ensure!(
                expected.is_none_or(|expected| expected == actual),
                "input payload disagrees with its manifest: {}",
                checked.display()
            );
            Ok((checked.strip_prefix(root)?.to_path_buf(), actual))
        })
        .collect()
}

pub fn verify_request_inputs(request: &Value) -> Result<()> {
    ensure!(
        request["schema"] == REQUEST_SCHEMA,
        "resident request protocol is incompatible"
    );
    let root = Path::new(request["root"].as_str().context("request root absent")?);
    let inputs: BTreeMap<PathBuf, String> =
        serde_json::from_value(request["input_artifacts"].clone())?;
    let required = request_artifacts(root, &request["task"], &request["env"])?;
    ensure!(
        !inputs.is_empty() && inputs == required,
        "resident input artifacts changed or required closure is incomplete"
    );
    Ok(())
}

/// Freeze once before any transfer. A later invocation uses the original input
/// identities and deadline, including when latest.json advanced during the task.
pub fn freeze_request(path: &Path, mut proposed: Value, deadline_ms: u64) -> Result<Value> {
    remaining(deadline_ms)?;
    let root = Path::new(proposed["root"].as_str().context("request root absent")?);
    let path = input_path(root, path)?;
    if path.try_exists()? {
        let saved: Value = read_json(&path)?;
        for field in ["root", "rank", "task", "env", "executable_sha256"] {
            ensure!(
                saved[field] == proposed[field],
                "resident frozen request changed: {field}"
            );
        }
        verify_request_inputs(&saved)?;
        remaining(
            saved["deadline_unix_ms"]
                .as_u64()
                .context("original task deadline absent")?,
        )?;
        return Ok(saved);
    }
    proposed["input_artifacts"] = json!(request_artifacts(
        root,
        &proposed["task"],
        &proposed["env"]
    )?);
    proposed["schema"] = json!(REQUEST_SCHEMA);
    proposed["deadline_unix_ms"] = json!(deadline_ms);
    fs::create_dir_all(path.parent().context("request parent absent")?)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    file.write_all(&serde_json::to_vec_pretty(&proposed)?)?;
    file.sync_all()?;
    fs::File::open(path.parent().context("request parent absent")?)?.sync_all()?;
    remaining(deadline_ms)?;
    Ok(proposed)
}
