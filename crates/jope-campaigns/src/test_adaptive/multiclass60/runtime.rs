use super::ledger::active_dataset_bindings;
use super::*;

pub(crate) fn run_all(run_id: &str, campaign_root: &Path, resume: bool) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let ready = ready_dataset_bindings(&plan, &root)?;
    let active = active_dataset_bindings(&ready, &root);
    if active.is_empty() {
        let terminal = plan
            .datasets
            .iter()
            .filter(|dataset| dataset_terminal(&root, &dataset.dataset_hash))
            .count();
        if terminal == plan.datasets.len() {
            println!("all 272 multiclass datasets already have terminal champion ledgers");
        } else {
            println!(
                "no unresolved comparator-ready multiclass datasets (ready={}, terminal={}, waiting={})",
                ready.len(),
                terminal,
                plan.datasets.len() - ready.len()
            );
        }
        return Ok(());
    }
    if !Path::new(corpus_eval_str()).is_file() {
        bail!(
            "release corpus_eval is missing: {corpus_eval_str}",
            corpus_eval_str = corpus_eval_str()
        );
    }
    let rows = scq_rows()?;
    let mut specs = Vec::<PathBuf>::new();
    let mut harvested_generations = 0usize;
    for dataset in active {
        let generation = (0..)
            .find(|generation| {
                !checkpoint_path(&root, &dataset.dataset_hash, *generation).is_file()
            })
            .expect("unbounded generation iterator");
        if generation > 0 {
            let previous = load_checkpoint(&root, dataset, generation - 1)?;
            if manage_winner_replay(&plan, &root, dataset, &previous, &rows, &mut specs)? {
                continue;
            }
            if generation == MAX_GENERATIONS {
                write_exhausted_ledger(&root, dataset, &previous)?;
                continue;
            }
            if generation > MAX_GENERATIONS {
                bail!(
                    "{} exceeded the maximum generation budget",
                    dataset.dataset_hash
                );
            }
        }
        let population = if generation == 0 {
            plan.seeds.clone()
        } else {
            load_checkpoint(&root, dataset, generation - 1)?.next_population
        };
        let generation_root = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join(format!("generation-{generation:03}"));
        let mut manifest_paths = Vec::new();
        let mut receipt_paths = Vec::new();
        for shard in 0..SHARD_SIZES.len() {
            let manifest_path = generation_root.join(format!("shard-{shard}.manifest.json"));
            let manifest_bytes = canonical_json(&shard_manifest(
                &plan,
                dataset,
                shard,
                generation,
                &population,
            )?)?;
            if manifest_path.exists() {
                if !resume || fs::read(&manifest_path)? != manifest_bytes {
                    bail!(
                        "immutable multiclass shard collision at {}",
                        manifest_path.display()
                    );
                }
            } else {
                immutable_write(&manifest_path, &manifest_bytes)?;
            }
            manifest_paths.push(manifest_path.clone());
            let mut terminal_failures = 0usize;
            let mut harvested = None;
            for attempt in 0..=(MAX_SCHEDULER_PREEMPTIONS + 2) {
                let job_id = shard_job_id(
                    &dataset.dataset_hash,
                    generation,
                    shard,
                    attempt,
                    &sha256_bytes(&manifest_bytes),
                );
                let Some(row) = rows.get(&job_id) else {
                    let spec_path = generation_root
                        .join(format!("shard-{shard}-a{attempt}-r{SPEC_REVISION}.toml"));
                    let rendered = render_spec(
                        &paths::jope_state(),
                        dataset,
                        shard,
                        generation,
                        attempt,
                        &manifest_path,
                        &sha256_bytes(&manifest_bytes),
                        &plan.corpus_eval_sha256,
                    );
                    if spec_path.exists() {
                        // r1 specs already staged before the admission
                        // tuning retain their original 24 GiB request.
                        // Accept that exact immutable legacy form while
                        // rendering new r1 specs with the leaner request.
                        let existing = fs::read_to_string(&spec_path)?;
                        let legacy_rendered = rendered.replace("ram_gb = 8\n", "ram_gb = 24\n");
                        if existing != rendered && existing != legacy_rendered {
                            bail!(
                                "immutable multiclass SCQ spec collision at {}",
                                spec_path.display()
                            );
                        }
                    } else {
                        immutable_write(&spec_path, rendered.as_bytes())?;
                    }
                    specs.push(spec_path);
                    break;
                };
                let state = row.get("state").and_then(Value::as_str).unwrap_or("");
                if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                    harvested = Some(fetch_shard_receipt(
                        &generation_root,
                        &dataset.dataset_hash,
                        generation,
                        shard,
                        attempt,
                        &job_id,
                    )?);
                    break;
                }
                if terminal_infrastructure_state(state)
                    || (state == "completed"
                        && row.get("exit_code").and_then(Value::as_i64) != Some(0))
                {
                    if scheduler_preemption(row) {
                        continue;
                    }
                    terminal_failures += 1;
                    if terminal_failures == 3 {
                        bail!("multiclass shard {job_id} exhausted infrastructure retries");
                    }
                    continue;
                }
                break;
            }
            if let Some(path) = harvested {
                receipt_paths.push(path);
            }
        }
        if receipt_paths.len() == SHARD_SIZES.len() {
            let mut slots = slots_from_receipts(
                dataset,
                generation,
                &population,
                &manifest_paths,
                &receipt_paths,
                &comparator_protocol_sha256(&plan, &dataset.dataset_hash)?,
            )?;
            slots.sort_by(slot_rank_order);
            let best_slot = slots.iter().find(|slot| slot.valid);
            let best = best_slot.and_then(|slot| slot.score.clone());
            let best_member = best_slot.map(|slot| slot.member.clone());
            let incumbent = dataset
                .strongest_available_incumbent
                .as_ref()
                .context("dataset has no frozen incumbent")?;
            let winner = best.as_ref().is_some_and(|score| {
                qualifies(
                    score,
                    generation,
                    generation == 0,
                    score.macro_f1,
                    incumbent.macro_f1,
                    generation >= 19,
                )
            });
            let next = if winner || generation + 1 == MAX_GENERATIONS {
                Vec::new()
            } else {
                next_population(dataset, &slots, generation + 1)?
            };
            let checkpoint = GenerationCheckpoint {
                schema: CHECKPOINT_SCHEMA.to_string(),
                run_id: run_id.to_string(),
                dataset_hash: dataset.dataset_hash.clone(),
                comparator_sha256: dataset.comparator_sha256.clone(),
                generation,
                test_adaptive: true,
                selection_bias: "test-selected".to_string(),
                benchmark_valid: false,
                slots,
                best,
                best_member,
                initial_budget_complete: generation + 1 == MAX_GENERATIONS,
                continuation_lineage_salt: None,
                next_population: next,
                completed_at_unix_ms: unix_millis(),
            };
            immutable_write(
                &checkpoint_path(&root, &dataset.dataset_hash, generation),
                &canonical_json(&checkpoint)?,
            )?;
            harvested_generations += 1;
        }
    }
    let check_failures = std::sync::Mutex::new(Vec::<String>::new());
    thread::scope(|scope| {
        for spec in &specs {
            let check_failures = &check_failures;
            scope.spawn(move || {
                let failure = match Command::new(scq_str()).arg("check").arg(spec).status() {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(format!(
                        "scq check failed for {} with {status}",
                        spec.display()
                    )),
                    Err(error) => Some(format!(
                        "could not run scq check for {}: {error}",
                        spec.display()
                    )),
                };
                if let Some(failure) = failure {
                    check_failures.lock().unwrap().push(failure);
                }
            });
        }
    });
    let mut check_failures = check_failures.into_inner().unwrap();
    check_failures.sort();
    if !check_failures.is_empty() {
        bail!("{}", check_failures.join("; "));
    }
    if !specs.is_empty() {
        let status = Command::new(scq_str())
            .arg("submit-batch")
            .arg("--json")
            .args(&specs)
            .status()?;
        if !status.success() {
            let reconciled = scq_rows()?;
            let missing = specs
                .iter()
                .map(|spec| spec_job_id(spec))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|job_id| !reconciled.contains_key(job_id))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                bail!(
                    "SCQ multiclass generation submission failed; missing jobs: {}",
                    missing.join(",")
                );
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "submitted_specs": specs.len(),
            "active_datasets": ACTIVE_DATASETS,
            "queue_buffer_datasets": QUEUE_BUFFER_DATASETS,
            "harvested_generations": harvested_generations,
            "resume": resume,
        }))?
    );
    Ok(())
}
