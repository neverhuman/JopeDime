use super::*;

// ---------------------------------------------------------------------------
// Native phase handlers
// ---------------------------------------------------------------------------

pub(super) fn watermark_dir(root: &Path, watermark: i64) -> PathBuf {
    root.join("training")
        .join("tournament-work")
        .join(format!("watermark-{watermark:09}"))
}

pub(super) fn handle_launch_checkpoint_tournament(request: &Value, root: &Path) -> Result<Value> {
    let corpus = PathBuf::from(get_str(
        request
            .pointer("/artifacts/freeze_corpus")
            .ok_or_else(|| anyhow!("missing freeze_corpus artifact"))?,
        "manifest",
    )?);
    let packed = PathBuf::from(get_str(
        request
            .pointer("/artifacts/build_packed_corpus")
            .ok_or_else(|| anyhow!("missing build_packed_corpus artifact"))?,
        "manifest",
    )?);
    let recipes = request
        .get("recipes")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("tournament recipes are absent"))?;
    let seed = get_i64(request, "seed")?;
    let ancestor = get_str(request, "accepted_ancestor")?;
    let mut jobs = Vec::new();
    for (offset, recipe) in recipes.iter().enumerate() {
        let recipe = recipe
            .as_str()
            .ok_or_else(|| anyhow!("tournament recipe is not a string"))?;
        let rendered = render_trainer_spec(&corpus, recipe, seed + offset as i64, &packed, root)?;
        if get_str(&rendered, "accepted_ancestor_sha256")? != ancestor {
            bail!("accepted checkpoint changed before tournament dispatch");
        }
        let spec = get_str(&rendered, "spec")?.to_owned();
        check_and_submit(&spec)?;
        let mut job = rendered;
        job["spec_sha256"] = Value::String(sha256_file(Path::new(&spec))?);
        jobs.push(job);
    }
    Ok(json!({"jobs": jobs, "job_count": jobs.len()}))
}

pub(super) fn handle_fetch_candidate_checkpoints(request: &Value, root: &Path) -> Result<Value> {
    let jobs = scheduler_jobs()?;
    let launched = request
        .pointer("/artifacts/launch_checkpoint_tournament/jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing launch_checkpoint_tournament artifact"))?;
    let mut pruned: BTreeMap<String, Option<PathBuf>> = BTreeMap::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        if !jobs.contains_key(job_id) {
            pruned.insert(job_id.to_owned(), pruned_job_outputs(job_id));
        }
    }
    let mut pending_jobs = Vec::new();
    let mut failed = BTreeMap::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        let state = job_state(&jobs, job_id);
        if matches!(state, Some("failed") | Some("cancelled")) {
            failed.insert(job_id.to_owned(), state.unwrap_or("").to_owned());
        }
        let recovered = pruned.get(job_id).map(|p| p.is_some()).unwrap_or(false);
        if state != Some("completed") && !recovered {
            pending_jobs.push(job_id.to_owned());
        }
    }
    if !failed.is_empty() {
        bail!("checkpoint tournament jobs failed: {failed:?}");
    }
    if !pending_jobs.is_empty() {
        return Err(pending(format!(
            "waiting for checkpoint jobs: {}",
            pending_jobs.join(",")
        )));
    }
    let fetch_root = watermark_dir(root, get_i64(request, "watermark")?).join("fetched");
    fs::create_dir_all(&fetch_root)?;
    let mut candidates = Vec::new();
    for item in launched {
        let job_id = get_str(item, "job_id")?;
        let recipe = get_str(item, "recipe")?;
        let destination = fetch_root.join(recipe);
        if !destination.is_dir() {
            let temporary = fetch_root.join(format!(".{recipe}.fetch-{}", std::process::id()));
            fs::create_dir_all(&temporary)?;
            match pruned.get(job_id).and_then(|p| p.as_ref()) {
                Some(source) => copy_tree(source, &temporary.join("outputs"))?,
                None => {
                    let (code, stdout, stderr) = run_command(
                        scq_str(),
                        &["fetch", job_id, "--output", &temporary.to_string_lossy()],
                    )?;
                    if code != 0 {
                        let _ = fs::remove_dir_all(&temporary);
                        bail!(
                            "{}",
                            if stderr.trim().is_empty() {
                                stdout.trim()
                            } else {
                                stderr.trim()
                            }
                        );
                    }
                }
            }
            fs::rename(&temporary, &destination)?;
        }
        let mut results = Vec::new();
        rglob(&destination, "train-result.json", &mut results);
        if results.len() != 1 {
            bail!("{job_id} fetch has {} train-result files", results.len());
        }
        let summary = read_json(&results[0])?;
        if summary
            .get("accepted_ancestor_sha256")
            .and_then(Value::as_str)
            != Some(get_str(request, "accepted_ancestor")?)
            || summary.get("recipe").and_then(Value::as_str) != Some(recipe)
            || get_i64(&summary, "seed")? != get_i64(item, "seed")?
        {
            bail!("{job_id} training provenance drift");
        }
        let digest = summary
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let mut matches = Vec::new();
        rglob(&destination, "*.model.pt", &mut matches);
        let mut verified: Vec<PathBuf> = Vec::new();
        for path in matches {
            if sha256_file(&path)? == digest {
                verified.push(path);
            }
        }
        verified.sort_by_key(|path| {
            (
                path.file_name().map(|n| n.to_string_lossy().into_owned())
                    != Some(format!("{digest}.model.pt")),
                path.to_string_lossy().into_owned(),
            )
        });
        let Some(best) = verified.first() else {
            bail!("{job_id} has no verified checkpoint");
        };
        let immutable = fetch_root
            .join("checkpoints")
            .join(format!("{digest}.model.pt"));
        fs::create_dir_all(immutable.parent().unwrap())?;
        if !immutable.exists() {
            fs::copy(best, &immutable)?;
        }
        if sha256_file(&immutable)? != digest {
            bail!("fetched immutable checkpoint hash mismatch");
        }
        candidates.push(json!({
            "recipe": recipe,
            "seed": item.get("seed").cloned().unwrap_or(Value::Null),
            "checkpoint": immutable.to_string_lossy(),
            "checkpoint_sha256": digest,
            "train_result": results[0].to_string_lossy(),
            "train_result_sha256": sha256_file(&results[0])?,
        }));
    }
    Ok(json!({"candidates": candidates, "candidate_count": candidates.len()}))
}

pub(super) fn handle_generate_gate_proposals(request: &Value, root: &Path) -> Result<Value> {
    let work = watermark_dir(root, get_i64(request, "watermark")?).join("gate-proposals");
    fs::create_dir_all(&work)?;
    let frozen_champions = work.join("frozen-champions.json");
    if !frozen_champions.is_file() {
        fs::copy(root.join("indexes/champions.json"), &frozen_champions)?;
    }
    let (baseline_path, baseline_hash) = accepted_checkpoint()?;
    if baseline_hash != get_str(request, "accepted_ancestor")? {
        bail!("gate proposal baseline changed after corpus freeze");
    }
    let mut checkpoints = vec![json!({
        "role": "baseline",
        "checkpoint": baseline_path.to_string_lossy(),
        "checkpoint_sha256": baseline_hash,
    })];
    for candidate in request
        .pointer("/artifacts/fetch_candidate_checkpoints/candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing fetch_candidate_checkpoints artifact"))?
    {
        let mut row = json!({"role": "candidate"});
        for (key, value) in candidate.as_object().unwrap() {
            row[key] = value.clone();
        }
        checkpoints.push(row);
    }
    let seed = get_i64(request, "seed")?;
    let jobs = scheduler_jobs()?;
    let mut rendered = Vec::new();
    // Raised from 3: attempts r1-r3 were burned before the deficit-tolerant
    // sampler flag existed. Long-term the attempt identity should be
    // content-aware (bind the script sha into the id) so a fixed generator
    // never has to spend the surviving budget of a broken one.
    const MAX_PROPOSAL_ATTEMPTS: u32 = 5;
    for checkpoint in &checkpoints {
        let digest = get_str(checkpoint, "checkpoint_sha256")?;
        let render_attempt = |attempt: u32| {
            let spec_name = if attempt > 1 {
                format!("{digest}-r{attempt}.toml")
            } else {
                format!("{digest}.toml")
            };
            render_checkpoint_proposal_spec(
                &work.join(spec_name),
                Path::new(get_str(checkpoint, "checkpoint")?),
                Some("validation"),
                None,
                seed,
                64,
                root,
                attempt,
            )
        };
        // The scheduler prunes terminal records on its own clock, so decide from
        // the full attempt family: a completed attempt wins, then the highest
        // live one; only then does a failed latest attempt earn a successor.
        let base_spec = render_attempt(1)?;
        let base_spec_value = base_spec.to_value();
        let base_job_id = get_str(&base_spec_value, "job_id")?.to_owned();
        let id_for = |attempt: u32| {
            if attempt > 1 {
                format!("{base_job_id}-r{attempt}")
            } else {
                base_job_id.clone()
            }
        };
        let states: Vec<Option<String>> = (1..=MAX_PROPOSAL_ATTEMPTS)
            .map(|attempt| job_state(&jobs, &id_for(attempt)).map(str::to_owned))
            .collect();
        let completed = (1..=MAX_PROPOSAL_ATTEMPTS)
            .rev()
            .find(|a| states[(*a - 1) as usize].as_deref() == Some("completed"));
        let live = (1..=MAX_PROPOSAL_ATTEMPTS).rev().find(|a| {
            matches!(
                states[(*a - 1) as usize].as_deref(),
                Some("queued" | "starting" | "dispatching" | "running" | "preempting" | "held")
            )
        });
        let highest_known = (1..=MAX_PROPOSAL_ATTEMPTS)
            .rev()
            .find(|a| states[(*a - 1) as usize].is_some());
        let (attempt, needs_submit) = if let Some(attempt) = completed {
            (attempt, false)
        } else if let Some(attempt) = live {
            (attempt, false)
        } else if let Some(highest) = highest_known {
            if highest >= MAX_PROPOSAL_ATTEMPTS {
                bail!(
                    "checkpoint proposal job failed after {highest} attempts: {}",
                    id_for(highest)
                );
            }
            (highest + 1, true)
        } else {
            (1, true)
        };
        let spec = if attempt == 1 {
            base_spec
        } else {
            render_attempt(attempt)?
        };
        let spec_value = spec.to_value();
        let spec_job_id = get_str(&spec_value, "job_id")?;
        if needs_submit && job_state(&jobs, spec_job_id).is_none() {
            check_and_submit(get_str(&spec_value, "spec")?)?;
        }
        if get_str(&spec_value, "checkpoint_sha256")? != digest {
            bail!("proposal spec checkpoint content drift");
        }
        let mut row = checkpoint.clone();
        for (key, value) in spec_value.as_object().unwrap() {
            row[key] = value.clone();
        }
        rendered.push(row);
    }
    let jobs = scheduler_jobs()?;
    let pending_jobs: Vec<String> = rendered
        .iter()
        .filter(|row| {
            job_state(
                &jobs,
                row.get("job_id").and_then(Value::as_str).unwrap_or(""),
            ) != Some("completed")
        })
        .map(|row| {
            row.get("job_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        })
        .collect();
    if !pending_jobs.is_empty() {
        return Err(pending(format!(
            "waiting for gate proposal jobs: {}",
            pending_jobs.join(",")
        )));
    }
    let mut proposal_sets = Vec::new();
    for row in rendered {
        let job_id = get_str(&row, "job_id")?;
        let digest = get_str(&row, "checkpoint_sha256")?;
        let output = fetch_job_output(job_id, &work.join("fetched").join(digest))?;
        let mut proposals = Vec::new();
        rglob(&output, "dl-proposals.jsonl", &mut proposals);
        let mut summaries = Vec::new();
        rglob(&output, "dl-proposals.summary.json", &mut summaries);
        if proposals.len() != 1 || summaries.len() != 1 {
            bail!("{job_id} proposal fetch is incomplete");
        }
        let summary = read_json(&summaries[0])?;
        if summary
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(digest)
            || summary.get("split").and_then(Value::as_str) != Some("validation")
            || get_i64(&summary, "seed")? != seed
            || Some(sha256_file(&proposals[0])?.as_str())
                != summary.get("output_sha256").and_then(Value::as_str)
        {
            bail!("{job_id} proposal provenance drift");
        }
        let mut entry = row.clone();
        entry["proposals"] = Value::String(proposals[0].to_string_lossy().into_owned());
        entry["proposals_sha256"] = Value::String(sha256_file(&proposals[0])?);
        entry["summary"] = Value::String(summaries[0].to_string_lossy().into_owned());
        entry["summary_sha256"] = Value::String(sha256_file(&summaries[0])?);
        entry["proposal_validity"] = summary["proposal_validity"].clone();
        entry["p95_proposal_latency_ms"] = summary["p95_proposal_latency_ms"].clone();
        proposal_sets.push(entry);
    }
    Ok(json!({
        "development_split": "validation",
        "proposal_budget_per_dataset": 64,
        "seed": seed,
        "proposal_sets": proposal_sets,
        "frozen_champions": frozen_champions.to_string_lossy(),
        "frozen_champions_sha256": sha256_file(&frozen_champions)?,
    }))
}

pub(super) fn handle_build_acceptance_evidence(request: &Value, root: &Path) -> Result<Value> {
    let gate_evaluation = request
        .pointer("/artifacts/evaluate_frozen_gate")
        .ok_or_else(|| anyhow!("missing evaluate_frozen_gate artifact"))?;
    let mut evidence = gate_evaluation
        .get("acceptance_evidence")
        .cloned()
        .unwrap_or_else(|| gate_evaluation.clone());
    if !evidence.is_object() {
        bail!("frozen gate did not produce acceptance evidence");
    }
    for key in [
        "baseline",
        "candidate",
        "proposal_validity",
        "replay_verification",
    ] {
        if evidence.get(key).is_none() {
            bail!("frozen gate evidence is missing {key}");
        }
    }
    let mut fetched = BTreeMap::new();
    for candidate in request
        .pointer("/artifacts/fetch_candidate_checkpoints/candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing fetch_candidate_checkpoints artifact"))?
    {
        fetched.insert(
            get_str(candidate, "checkpoint_sha256")?.to_owned(),
            candidate.clone(),
        );
    }
    let checkpoint_hash = evidence
        .pointer("/candidate/checkpoint_sha256")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let Some(candidate) = fetched.get(&checkpoint_hash) else {
        bail!("gate winner is not one of the fetched tournament checkpoints");
    };
    evidence["candidate"]["checkpoint_path"] = candidate["checkpoint"].clone();
    if evidence
        .pointer("/baseline/checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(get_str(request, "accepted_ancestor")?)
    {
        bail!("gate baseline differs from the frozen accepted ancestor");
    }
    let destination =
        watermark_dir(root, get_i64(request, "watermark")?).join("acceptance-evidence.json");
    atomic_json(&destination, &evidence)?;
    let mut result = evidence;
    result["artifact_path"] = Value::String(destination.to_string_lossy().into_owned());
    result["artifact_sha256"] = Value::String(sha256_file(&destination)?);
    Ok(result)
}
