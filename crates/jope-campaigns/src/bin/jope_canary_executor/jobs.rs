use super::*;

pub(super) fn runtime_bundle() -> Result<PathBuf> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("canary executor is not running from a runtime bundle")
}

pub(super) fn toml_quote(value: &str) -> Result<String> {
    ensure!(
        !value.contains(['"', '\\', '\n', '\r']),
        "unsafe generated TOML value"
    );
    Ok(format!("\"{value}\""))
}

pub(super) fn append_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = {}", toml_quote(name)?),
        format!("source = {}", toml_quote(&source.to_string_lossy())?),
        format!("target = {}", toml_quote(target)?),
        format!("sha256 = {}", toml_quote(&sha256_file(source)?)?),
        String::new(),
    ]);
    Ok(())
}

pub(super) fn write_immutable(path: &Path, body: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(fs::read(path)? == body, "immutable canary artifact drift");
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(body)?;
    file.sync_data()?;
    Ok(())
}

pub(super) fn canary_lane_ancestry(mode: CanaryMode, lane: &str) -> &'static str {
    match lane {
        "gp" => "none",
        "candidate" if mode == CanaryMode::Shadow => "shadow_dl",
        _ => "accepted_dl",
    }
}

pub(super) fn scq_jobs(args: &Args) -> Result<BTreeMap<String, String>> {
    let output = Command::new(&args.scq).args(["--json", "ls"]).output()?;
    ensure!(output.status.success(), "SCQ list failed");
    let value: Value = serde_json::from_slice(&output.stdout)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| {
            Some((
                row.get("id")?.as_str()?.to_owned(),
                row.get("state")?.as_str()?.to_owned(),
            ))
        })
        .collect())
}

pub(super) fn submit_spec(args: &Args, spec: &Path) -> Result<()> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()?;
    ensure!(
        check.status.success(),
        "SCQ check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()?;
    let message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    ensure!(
        submit.status.success() || message.to_ascii_lowercase().contains("already exists"),
        "SCQ submit failed: {}",
        message.trim()
    );
    Ok(())
}

pub(super) fn recursive_find(path: &Path, name: &str, output: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, output);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            output.push(path);
        }
    }
}

pub(super) fn fetch(args: &Args, job_id: &str, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let result = Command::new(&args.scq)
        .args([
            "fetch",
            job_id,
            "--output",
            output.to_string_lossy().as_ref(),
        ])
        .output()?;
    ensure!(
        result.status.success(),
        "SCQ fetch failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

pub(super) fn proposal_job_id(intent_hash: &str, worker_sha256: &str) -> String {
    format!(
        "jope-canary-propose-{}-{}",
        &intent_hash[..12],
        &worker_sha256[..8]
    )
}

pub(super) fn ensure_proposals(
    args: &Args,
    pending: &PendingIntent,
    work: &Path,
    jobs: &BTreeMap<String, String>,
) -> Result<Option<PathBuf>> {
    let proposal_set = work.join("canary-proposals.json");
    if proposal_set.is_file() {
        return Ok(Some(proposal_set));
    }
    let intent_path = work.join("intent.v2.json");
    let mut intent_document = serde_json::to_value(&pending.value)?;
    if pending.promotion.is_some() {
        intent_document["promotion_evaluation_v3"] = json!(true);
    }
    atomic_json(&intent_path, &intent_document)?;
    let (candidate, accepted) = checkpoint_paths(&args.root, pending)?;
    let bundle = runtime_bundle()?;
    let pointer = read_json(&args.root.join("catalog/current.json"))?;
    let catalog = PathBuf::from(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let seed_bank = args.root.join("corpus/robust-gp-seed-bank.v1.json");
    let intent_hash = sha256_file(&intent_path)?;
    let worker = bundle.join("pytorch/proposals/test_exact_canary_propose.py");
    let implementation_sha256 = sha256_bytes(
        format!(
            "{}\0{}",
            sha256_file(&worker)?,
            sha256_file(&std::env::current_exe()?)?
        )
        .as_bytes(),
    );
    let job_id = proposal_job_id(&intent_hash, &implementation_sha256);
    let spec = work.join(format!("{job_id}.toml"));
    if !spec.is_file() {
        let pythonpath = bundle.join("pytorch");
        let mut lines = vec![
            "[job]".to_owned(),
            format!("id = {}", toml_quote(&job_id)?),
            "project = \"jope-selfplay-train\"".to_owned(),
            "tier = \"normal\"".to_owned(),
            "command = \"/usr/bin/python3\"".to_owned(),
            "args = [\"{input:worker}\", \"--intent\", \"{input:intent}\", \"--generator\", \"{input:generator}\", \"--catalog-pointer\", \"{input:pointer}\", \"--catalog\", \"{input:catalog}\", \"--data\", \"{input:data}\", \"--candidate\", \"{input:candidate}\", \"--accepted\", \"{input:accepted}\", \"--seed-bank\", \"{input:seedbank}\", \"--selection-policy\", \"{input:selection_policy}\", \"--output\", \"{output:canary}/canary-proposals.json\"]".to_owned(),
            String::new(),
        ];
        lines.extend(PROPOSE.render_resource_block().lines().map(str::to_owned));
        lines.push(String::new());
        lines.extend(PROPOSE.render_placement_block().lines().map(str::to_owned));
        lines.push(String::new());
        lines.extend([
            "[env]".to_owned(),
            format!(
                "PYTHONPATH = {}",
                toml_quote(&pythonpath.to_string_lossy())?
            ),
            format!("OMP_NUM_THREADS = \"{}\"", PROPOSE.omp_threads),
            String::new(),
            "[preempt]".to_owned(),
            format!("preemptible = {}", PROPOSE.preemptible),
            "checkpoint = \"none\"".to_owned(),
            "resume = \"rerun\"".to_owned(),
            String::new(),
        ]);
        lines.extend(PROPOSE.render_limits_block().lines().map(str::to_owned));
        lines.push(String::new());
        append_input(
            &mut lines,
            "worker",
            &worker,
            "test_exact_canary_propose.py",
        )?;
        append_input(
            &mut lines,
            "generator",
            &bundle.join("pytorch/proposals/test_exact_dl_generate.py"),
            "test_exact_dl_generate.py",
        )?;
        append_input(
            &mut lines,
            "selection_policy",
            &bundle.join("pytorch/proposals/proposal-selection-policy.v1.json"),
            "proposal-selection-policy.v1.json",
        )?;
        append_input(
            &mut lines,
            "common",
            &bundle.join("pytorch/shared/test_exact_common.py"),
            "test_exact_common.py",
        )?;
        append_input(&mut lines, "intent", &intent_path, "intent.json")?;
        append_input(
            &mut lines,
            "pointer",
            &args.root.join("catalog/current.json"),
            "catalog/pointer.json",
        )?;
        append_input(&mut lines, "catalog", &catalog, "catalog/catalog.json")?;
        append_input(&mut lines, "data", &data, "catalog/contexts.f32")?;
        append_input(
            &mut lines,
            "candidate",
            &candidate,
            &format!(
                "{}.model.pt",
                pending.value.candidate_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(
            &mut lines,
            "accepted",
            &accepted,
            &format!(
                "{}.model.pt",
                pending.value.accepted_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(&mut lines, "seedbank", &seed_bank, "seed-bank.json")?;
        lines.extend([
            "[[output]]".to_owned(),
            "name = \"canary\"".to_owned(),
            "source = \"canary\"".to_owned(),
            "directory = true".to_owned(),
            String::new(),
        ]);
        write_immutable(&spec, lines.join("\n").as_bytes())?;
    }
    match jobs.get(&job_id).map(String::as_str) {
        None => {
            submit_spec(args, &spec)?;
            Ok(None)
        }
        Some("completed") => {
            let fetched = work.join("proposal-fetch");
            fetch(args, &job_id, &fetched)?;
            let mut found = Vec::new();
            recursive_find(&fetched, "canary-proposals.json", &mut found);
            let source = found
                .into_iter()
                .next()
                .context("canary proposal output is absent")?;
            let bytes = fs::read(source)?;
            write_immutable(&proposal_set, &bytes)?;
            Ok(Some(proposal_set))
        }
        Some(state) if ACTIVE_STATES.contains(&state) => Ok(None),
        Some(state) => bail!("canary proposal job is terminal in state {state}"),
    }
}

pub(super) type CatalogEntries = BTreeMap<(Task, String), Value>;

pub(super) fn catalog_entries(root: &Path) -> Result<(Value, CatalogEntries)> {
    let pointer = read_json(&root.join("catalog/current.json"))?;
    let catalog = read_json(Path::new(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    ))?;
    let mut entries = BTreeMap::new();
    for row in catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("catalog lacks entries")?
    {
        let task: Task = row
            .get("task")
            .and_then(Value::as_str)
            .context("catalog entry lacks task")?
            .parse()?;
        let dataset = row
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("catalog entry lacks dataset")?
            .to_owned();
        entries.insert((task, dataset), row.clone());
    }
    Ok((pointer, entries))
}

pub(super) fn build_evaluations(
    args: &Args,
    pending: &PendingIntent,
    work: &Path,
    proposal_set: &Path,
    context_limit: usize,
) -> Result<Vec<(String, PathBuf)>> {
    let proposals = read_json(proposal_set)?;
    ensure!(
        proposals.get("schema").and_then(Value::as_str) == Some("CanaryProposalSet.v2"),
        "canary proposal output schema drift"
    );
    let (pointer, catalog) = catalog_entries(&args.root)?;
    let candidate = pending.value.candidate_checkpoint_sha256.as_str();
    let hall = args.root.join("corpus/robust-gp-seed-bank.v1.json");
    let hall_sha = sha256_file(&hall)?;
    let generation = format!("strict-wave-{:06}", 900_000 + pending.value.run_index);
    let mut output = Vec::new();
    for (index, context) in proposals
        .get("contexts")
        .and_then(Value::as_array)
        .context("canary proposal set lacks contexts")?
        .iter()
        .take(context_limit)
        .enumerate()
    {
        let dataset = context
            .get("context_identity")
            .and_then(Value::as_str)
            .context("canary proposal context lacks identity")?;
        let entry = catalog
            .get(&(pending.value.task, dataset.to_owned()))
            .context("canary context is absent from current catalog")?;
        let train = PathBuf::from(
            entry
                .get("train_csv")
                .and_then(Value::as_str)
                .context("canary catalog entry lacks train CSV")?,
        );
        let test = PathBuf::from(
            entry
                .get("test_csv")
                .and_then(Value::as_str)
                .context("canary catalog entry lacks test CSV")?,
        );
        let train_sha = entry
            .get("train_sha256")
            .and_then(Value::as_str)
            .context("canary catalog entry lacks train hash")?;
        let test_sha = entry
            .get("test_sha256")
            .and_then(Value::as_str)
            .context("canary catalog entry lacks test hash")?;
        ensure!(
            sha256_file(&train)? == train_sha && sha256_file(&test)? == test_sha,
            "canary dataset content drift"
        );
        // Ask the protocol's own authority rather than keeping a local map. The previous
        // `if Binary { "auc" } else { "r2" }` silently claimed r2 for timeseries and
        // multiclass; `evaluation_protocol_sha256` validates against
        // `expected_evaluation_metric`, so every MC/TS canary aborted with "evaluation
        // protocol task/metric/tier mismatch" before writing a manifest. Fail-closed, but
        // it left an anchored branch with an unrunnable verification lane. The metric is
        // also bound a second time into `label_authority` below, so a local copy has two
        // chances to drift.
        let metric = expected_evaluation_metric(pending.value.task.as_str())?;
        let protocol = evaluation_protocol_sha256(
            pending.value.task.as_str(),
            metric,
            "test_exact",
            train_sha,
            test_sha,
        )?;
        let mut candidates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut requested = serde_json::Map::new();
        for (lane, source, origin) in [
            ("candidate", "canary_candidate_dl", "dl_direct"),
            ("accepted", "canary_accepted_baseline", "accepted_baseline"),
            ("gp", "canary_zero_dl_gp", "gp_comparator"),
        ] {
            let mut semantics = Vec::new();
            for (slot, row) in context
                .get(lane)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let semantic = row
                    .get("semantic_sha256")
                    .and_then(Value::as_str)
                    .context("canary proposal lacks semantic")?;
                semantics.push(semantic.to_owned());
                if seen.insert(semantic.to_owned()) {
                    candidates.push(json!({
                        "canonical_genome": row.get("canonical_genome"),
                        "semantic_sha256": semantic,
                        "origin": origin,
                        "source": source,
                        "slot_group": lane,
                        "slot_index": slot,
                        "evidence": {
                            "canary_lane": lane,
                            "dl_ancestry_kind": canary_lane_ancestry(pending.value.mode, lane),
                        },
                    }));
                }
            }
            requested.insert(lane.to_owned(), json!(semantics));
        }
        let frozen_seed = sha256_bytes(&serde_json::to_vec(&json!({
            "intent": sha256_file(&pending.path)?,
            "dataset": dataset,
            "requested": requested,
        }))?);
        let manifest_path = work.join(format!("manifest-{index:02}.json"));
        let generated_at_ms = pending.created_at_ms as u128;
        let manifest = json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "receipt_purpose": "canary",
            "task": pending.value.task,
            "dataset_hash": dataset,
            "generation": generation,
            "strict_wave": generation,
            "evaluation_lane": "canary",
            "micro_lane": context.get("class"),
            "catalog_sha256": pointer.get("catalog_sha256"),
            "checkpoint_sha256": candidate,
            "checkpoint_epoch": checkpoint_epoch(candidate)?,
            "train_csv": train,
            "train_sha256": train_sha,
            "test_csv": test,
            "test_sha256": test_sha,
            "protocol_sha256": protocol,
            "generated_at_ms": generated_at_ms,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": metric,
                "test_adaptive": true,
                "benchmark_valid": false,
                "selection_bias": "test-exact-reused",
                "generation": generation,
                "frozen_seed_sha256": frozen_seed,
                "model_checkpoint_sha256": candidate,
                "hall_of_fame_sha256": hall_sha,
            },
            "canary": {
                "mode": pending.value.mode,
                "run_index": pending.value.run_index,
                "accepted_parent_checkpoint_sha256": pending.value.accepted_checkpoint_sha256,
                "candidate_checkpoint_sha256": pending.value.candidate_checkpoint_sha256,
                "requested": requested,
            },
            "candidates": candidates,
        });
        if manifest_path.is_file() {
            let existing = read_json(&manifest_path)?;
            ensure!(
                existing.get("schema").and_then(Value::as_str)
                    == Some("JopeFlywheelCorpusEvalShard.v1")
                    && existing.get("receipt_purpose").and_then(Value::as_str) == Some("canary")
                    && existing.get("task") == manifest.get("task")
                    && existing.get("dataset_hash") == manifest.get("dataset_hash")
                    && existing.get("generation") == manifest.get("generation")
                    && existing.get("checkpoint_sha256") == manifest.get("checkpoint_sha256")
                    && existing.get("train_sha256") == manifest.get("train_sha256")
                    && existing.get("test_sha256") == manifest.get("test_sha256")
                    && existing.pointer("/canary/mode") == manifest.pointer("/canary/mode")
                    && existing.pointer("/canary/run_index")
                        == manifest.pointer("/canary/run_index")
                    && existing.pointer("/canary/accepted_parent_checkpoint_sha256")
                        == manifest.pointer("/canary/accepted_parent_checkpoint_sha256")
                    && existing.pointer("/canary/candidate_checkpoint_sha256")
                        == manifest.pointer("/canary/candidate_checkpoint_sha256")
                    && existing.pointer("/canary/requested")
                        == manifest.pointer("/canary/requested"),
                "existing immutable canary manifest is not bound to the active intent"
            );
        } else {
            write_immutable(&manifest_path, &serde_json::to_vec(&manifest)?)?;
        }
        let intent_hash = sha256_file(&pending.path)?;
        let job_id = format!("jte-canary-{}-{index:02}", &intent_hash[..16]);
        let spec_path = work.join(format!("{job_id}.toml"));
        if !spec_path.is_file() {
            let bundle = runtime_bundle()?;
            let evaluator = bundle.join("runtime/bin/corpus_eval");
            let mut lines = vec![
                "[job]".to_owned(),
                format!("id = {}", toml_quote(&job_id)?),
                "project = \"jope-selfplay-eval\"".to_owned(),
                "tier = \"normal\"".to_owned(),
                "command = \"/usr/bin/env\"".to_owned(),
                "args = [\"{input:runner}\", \"{input:manifest}\", \"{input:train}\", \"{input:test}\", \"{input:evaluator}\", \"{output:receipt}\"]".to_owned(),
                String::new(),
            ];
            lines.extend(EVAL.render_resource_block().lines().map(str::to_owned));
            lines.push(String::new());
            lines.extend(EVAL.render_placement_block().lines().map(str::to_owned));
            lines.push(String::new());
            lines.extend([
                "[env]".to_owned(),
                format!("RAYON_NUM_THREADS = \"{}\"", EVAL.omp_threads),
                format!("OMP_NUM_THREADS = \"{}\"", EVAL.omp_threads),
                String::new(),
                "[preempt]".to_owned(),
                format!("preemptible = {}", EVAL.preemptible),
                "checkpoint = \"none\"".to_owned(),
                "resume = \"rerun\"".to_owned(),
                String::new(),
            ]);
            lines.extend(EVAL.render_limits_block().lines().map(str::to_owned));
            lines.push(String::new());
            append_input(&mut lines, "runner", &args.runner, "runner.sh")?;
            append_input(&mut lines, "manifest", &manifest_path, "manifest.json")?;
            append_input(&mut lines, "train", &train, "train.csv")?;
            append_input(&mut lines, "test", &test, "test.csv")?;
            append_input(&mut lines, "evaluator", &evaluator, "corpus_eval")?;
            lines.extend([
                "[[output]]".to_owned(),
                "name = \"receipt\"".to_owned(),
                "source = \"receipt\"".to_owned(),
                "directory = true".to_owned(),
                String::new(),
            ]);
            write_immutable(&spec_path, lines.join("\n").as_bytes())?;
        }
        output.push((job_id, manifest_path));
    }
    Ok(output)
}

pub(super) fn ensure_evaluations(
    args: &Args,
    work: &Path,
    evaluations: &[(String, PathBuf)],
    jobs: &BTreeMap<String, String>,
) -> Result<Option<Vec<PathBuf>>> {
    let mut fetched_outputs = Vec::new();
    let mut waiting = false;
    for (job_id, manifest) in evaluations {
        let spec = work.join(format!("{job_id}.toml"));
        match jobs.get(job_id).map(String::as_str) {
            None => {
                submit_spec(args, &spec)?;
                waiting = true;
            }
            Some("completed") => {
                let output = work.join("evaluation-fetch").join(job_id);
                if !output.join("FETCHED").is_file() {
                    fetch(args, job_id, &output)?;
                    fs::write(output.join("FETCHED"), b"ok\n")?;
                }
                let mut receipts = Vec::new();
                recursive_find(
                    &output,
                    &format!(
                        "{}.receipts.jsonl",
                        read_json(manifest)?
                            .get("dataset_hash")
                            .and_then(Value::as_str)
                            .context("canary manifest lacks dataset")?
                    ),
                    &mut receipts,
                );
                let receipt = receipts
                    .into_iter()
                    .next()
                    .context("canary receipt is absent")?;
                let receipt_root = receipt
                    .parent()
                    .context("canary receipt has no parent")?
                    .to_path_buf();
                ingest_receipt_with_manifest(
                    &args.root.join("corpus/strict-test-exact.sqlite"),
                    job_id,
                    &receipt_root,
                    manifest,
                )?;
                fetched_outputs.push(receipt);
            }
            Some(state) if ACTIVE_STATES.contains(&state) => waiting = true,
            Some(state @ ("failed" | "cancelled")) => {
                // A failed canary member is evidence of zero validity, not a
                // reason to wedge the executor forever. Keep an explicit empty
                // receipt plus a sidecar; summarize counts all requested slots
                // as invalid and marks replay verification incomplete. No
                // synthetic score is ingested or imputed.
                let failures = work.join("evaluation-failures");
                fs::create_dir_all(&failures)?;
                let receipt = failures.join(format!("{job_id}.receipts.jsonl"));
                write_immutable(&receipt, b"")?;
                atomic_json(
                    &failures.join(format!("{job_id}.v1.json")),
                    &json!({
                        "schema": "CanaryEvaluationFailure.v1",
                        "job_id": job_id,
                        "terminal_state": state,
                        "manifest_sha256": sha256_file(manifest)?,
                        "imputed": false,
                    }),
                )?;
                fetched_outputs.push(receipt);
            }
            Some(state) => bail!("{job_id}: canary evaluator has unknown terminal state {state}"),
        }
    }
    Ok((!waiting && fetched_outputs.len() == evaluations.len()).then_some(fetched_outputs))
}
