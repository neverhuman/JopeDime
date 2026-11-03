use super::outbox::{record_inner, score};
use super::*;

/// Both proposal arms share this immutable cycle-level snapshot. The request
/// binds their common corpus/checkpoint/cut range; changes fail on resume.
pub fn freeze_snapshot(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
) -> Result<PathBuf> {
    freeze_snapshot_inner(root, corpus, checkpoint, output, cuts, cut_start, None)
}

/// Native input freezing uses the actual qualified evaluator identity from installed
/// root-controlled signed policy. It does not authorize or execute numerical work.
pub fn freeze_native_snapshot(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    policy_path: &Path,
) -> Result<PathBuf> {
    let policy = crate::eval_registry::native_scq::policy::InstalledPolicy::load(policy_path)?;
    ensure!(
        policy.signed.body.root == root,
        "native snapshot policy root differs"
    );
    freeze_snapshot_inner(
        root,
        corpus,
        checkpoint,
        output,
        cuts,
        cut_start,
        Some(&policy.signed.body.evaluator_executable_sha256),
    )
}
fn freeze_snapshot_inner(
    root: &Path,
    corpus: &Path,
    checkpoint: &Path,
    output: &Path,
    cuts: usize,
    cut_start: usize,
    native_evaluator: Option<&str>,
) -> Result<PathBuf> {
    recover_outbox(root)?;
    let destination = output
        .parent()
        .context("proposal parent absent")?
        .join("champion-snapshot.json");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(destination.with_extension("lock"))?;
    ensure!(lock.lock().is_ok(), "snapshot lock failed");
    let mut request = json!({"corpus_sha256":file_hash(&corpus.join("index.json"))?,
        "checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,"cuts":cuts,"cut_start":cut_start,
        "vocabulary_sha256":file_hash(&corpus.join("vocabulary.json"))?});
    if let Some(evaluator) = native_evaluator {
        request["native_qualified_evaluator_sha256"] = json!(evaluator);
    }
    let selection_path = destination.with_file_name("coverage-selection.json");
    if selection_path.exists() {
        request["coverage_selection_sha256"] = file_hash(&selection_path)?.into();
        request["coverage_selection_protocol"] = if native_evaluator.is_some() {
            read_json::<Value>(&selection_path)?["request"]["protocol"].clone()
        } else {
            crate::service::coverage::PROTOCOL.into()
        };
    }
    if destination.exists() {
        let snapshot: Snapshot = read_json(&destination)?;
        ensure!(
            snapshot.schema == "parallel.champion_snapshot.v1" && snapshot.request == request,
            "cycle champion snapshot request changed"
        );
        for entry in &snapshot.entries {
            entry.validate()?;
        }
        bind_snapshot(&destination)?;
        return Ok(destination);
    }
    let vocabulary = request["vocabulary_sha256"].as_str().unwrap().to_string();
    let evaluator = if let Some(evaluator) = native_evaluator {
        evaluator.to_owned()
    } else {
        let service: Value = read_json(&root.join("runs/gp-service.json"))?;
        let evaluator = service["executable_sha256"]
            .as_str()
            .context("GP service evaluator identity absent")?;
        let executable = PathBuf::from(
            service["executable"]
                .as_str()
                .context("GP service executable absent")?,
        );
        let executable = if executable.is_absolute() {
            executable
        } else {
            root.join(executable)
        };
        ensure!(
            executable.starts_with(root)
                && file_hash(&executable)? == evaluator
                && crate::telemetry::now()
                    .saturating_sub(service["timestamp"].as_u64().unwrap_or(0))
                    <= 10,
            "GP evaluator receipt is stale or changed"
        );
        ensure!(
            service["vocabulary_sha256"] == vocabulary,
            "GP service vocabulary differs from corpus"
        );
        evaluator.to_owned()
    };
    let evaluator = evaluator.as_str();
    let index = crate::teaching_store::read_index(root, corpus)?;
    let mut available = Vec::<Champion>::new();
    let store = root.join("runs/champions");
    if store.exists() {
        for directory in fs::read_dir(store)? {
            let path = directory?.path().join("current.json");
            if !path.is_file() {
                continue;
            }
            let entry: Champion = read_json(&path)?;
            entry.validate()?;
            if entry.identity.vocabulary_sha256 == vocabulary {
                available.push(entry);
            }
        }
    }
    // Current evaluator first, then a deterministic version order. Scores from
    // different evaluator versions are never compared; every seed is refitted.
    available.sort_by(|a, b| {
        (
            a.identity.evaluator_sha256 != evaluator,
            &a.identity.evaluator_sha256,
        )
            .cmp(&(
                b.identity.evaluator_sha256 != evaluator,
                &b.identity.evaluator_sha256,
            ))
    });
    let mut prepared = BTreeMap::<PathBuf, Prepared>::new();
    let mut entries = Vec::new();
    let selected = if selection_path.exists() {
        crate::service::coverage::selected_indices(&selection_path, corpus, &index.contexts)?
    } else {
        index
            .contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.partition != "meta_test" && c.cut >= cut_start && c.cut - cut_start < cuts
            })
            .map(|(i, _)| i)
            .collect()
    };
    for i in selected {
        let c = &index.contexts[i];
        if !prepared.contains_key(&c.prepared) {
            ensure!(
                file_hash(&resolve(root, &c.prepared)?.join("prepared.json"))? == c.prepared_sha256,
                "snapshot preparation changed"
            );
            prepared.insert(
                c.prepared.clone(),
                read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?,
            );
        }
        let cut = prepared[&c.prepared]
            .cuts
            .iter()
            .find(|p| p.index == c.cut)
            .context("snapshot cut absent")?;
        let identity = Identity {
            dataset: c.dataset.clone(),
            family: c.family.clone(),
            partition: c.partition.clone(),
            cut: c.cut,
            prepared_sha256: c.prepared_sha256.clone(),
            training_sha256: cut.training_sha256.clone(),
            validation_sha256: cut.validation_sha256.clone(),
            evaluator_sha256: evaluator.into(),
            vocabulary_sha256: vocabulary.clone(),
        };
        for entry in &available {
            let mut compatible = entry.identity.clone();
            compatible.evaluator_sha256 = evaluator.into();
            if compatible == identity {
                entries.push(entry.clone());
            }
        }
    }
    write_json(
        &destination,
        &Snapshot {
            schema: "parallel.champion_snapshot.v1".into(),
            request,
            evaluator_sha256: evaluator.into(),
            vocabulary_sha256: vocabulary,
            entries,
        },
    )?;
    bind_snapshot(&destination)?;
    Ok(destination)
}

fn bind_snapshot(path: &Path) -> Result<()> {
    let receipt = path.with_file_name("champion-snapshot-binding.json");
    let expected =
        json!({"schema":"parallel.champion_snapshot_binding.v1","sha256":file_hash(path)?});
    if receipt.exists() {
        ensure!(
            read_json::<Value>(&receipt)? == expected,
            "champion snapshot binding changed"
        );
    } else {
        write_json(&receipt, &expected)?;
    }
    Ok(())
}

pub fn verify_cycle_snapshot(
    output: &Path,
    corpus_sha: &str,
    cuts: usize,
    cut_start: usize,
) -> Result<()> {
    let path = output.join("champion-snapshot.json");
    let snapshot: Snapshot = read_json(&path)?;
    let binding: Value = read_json(&output.join("champion-snapshot-binding.json"))?;
    ensure!(
        binding["sha256"] == file_hash(&path)?
            && snapshot.request["corpus_sha256"] == corpus_sha
            && snapshot.request["cuts"] == cuts
            && snapshot.request["cut_start"] == cut_start,
        "cycle champion snapshot identity changed"
    );
    for entry in &snapshot.entries {
        entry.validate()?;
    }
    Ok(())
}

/// One bounded pass over explicitly declared completed ledgers. Only the best
/// observation for each exact identity survives in RAM; no test CSV is opened.
pub fn import(root: &Path, sources: &[PathBuf]) -> Result<Value> {
    let mut best = BTreeMap::<Identity, (f64, Value, PathBuf)>::new();
    let mut ledgers = 0usize;
    let mut observations = 0usize;
    for source in sources {
        for entry in walkdir::WalkDir::new(resolve(root, source)?).follow_links(false) {
            let entry = entry?;
            if !entry.file_type().is_file() || entry.file_name() != "observations.json" {
                continue;
            }
            let summary = entry.path().parent().unwrap().join("search.json");
            if !summary.is_file() {
                continue;
            }
            ensure!(
                read_json::<Value>(&summary)?["schema"] == "parallel.search.v1",
                "invalid completed teacher search"
            );
            let rows: Vec<Value> = read_json(entry.path())?;
            ledgers += 1;
            for row in rows {
                observations += 1;
                let Some(identity) = Identity::observation(&row)? else {
                    continue;
                };
                let Some(rmse) = score(&row)? else {
                    continue;
                };
                if best.get(&identity).is_none_or(|(old, _, _)| rmse < *old) {
                    best.insert(identity, (rmse, row, entry.path().into()));
                }
            }
        }
    }
    let mut changes = 0;
    for (_, row, path) in best.values() {
        // Historical import preserves its evidence without manufacturing a new
        // live producer event or entering the live winner publication sequence.
        changes += record_inner(root, row, path, false)? as usize;
    }
    Ok(
        json!({"schema":"parallel.champion_import.v1","ledgers":ledgers,"observations":observations,
        "identities":best.len(),"new_or_improved_champions":changes,"source_roots":sources,
        "qualification_only":true,"final_tests_opened":false}),
    )
}
