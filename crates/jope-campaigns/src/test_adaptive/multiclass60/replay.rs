use super::*;

pub(crate) fn replay_manifest(
    plan: &Plan,
    dataset: &DatasetBinding,
    generation: usize,
    champion: &SeedMember,
) -> Result<Value> {
    let protocol_sha256 = comparator_protocol_sha256(plan, &dataset.dataset_hash)?;
    Ok(json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "run_id": plan.run_id,
        "release_binary_sha256": plan.release_binary_sha256,
        "multiclass_coverage_final_receipt_sha256": plan.multiclass_coverage_final_receipt_sha256,
        "timeseries_coverage_final_receipt_sha256": plan.timeseries_coverage_final_receipt_sha256,
        "comparator_set_sha256": plan.comparator_set_sha256,
        "comparator_lock_sha256": dataset.comparator_sha256,
        "seed_population_sha256": plan.seed_population_sha256,
        "task": "multiclass",
        "dataset_hash": dataset.dataset_hash,
        "train_csv": dataset.train_path,
        "test_csv": dataset.test_path,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "checkpoint_epoch": format!("checkpoint-{}", champion.semantic_sha256),
        "checkpoint_sha256": champion.semantic_sha256,
        "catalog_sha256": plan.seed_catalog_sha256,
        "protocol_sha256": protocol_sha256,
        "generated_at_ms": 0,
        "meta_json": dataset.meta_path,
        "meta_sha256": dataset.meta_sha256,
        "label_authority": {
            "score_tier": "test",
            "metric": "macro_f1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-selected",
            "generation": format!("multiclass-generation-{generation:03}"),
            "frozen_seed_sha256": plan.seed_population_sha256,
            "model_checkpoint_sha256": champion.semantic_sha256,
            "hall_of_fame_sha256": champion.semantic_sha256,
        },
        "candidates": [{
            "semantic_sha256": champion.semantic_sha256,
            "canonical_genome": champion.canonical_genome,
        }],
    }))
}

pub(crate) fn replay_job_id(
    dataset_hash: &str,
    semantic_sha256: &str,
    node: &str,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    format!(
        "jope-mc272-{dataset_hash}-replay-{}-{node}-a{attempt}-m{}-r{REPLAY_SPEC_REVISION}",
        &semantic_sha256[..12],
        &manifest_sha256[..12]
    )
}

pub(crate) fn replay_role_capability(node: &str) -> Result<&'static str> {
    REPLAY_ROLE_CAPABILITIES
        .iter()
        .find_map(|(candidate, marker)| (*candidate == node).then_some(*marker))
        .with_context(|| format!("no multiclass replay capability role is assigned to {node}"))
}

pub(crate) fn render_replay_spec(
    dataset: &DatasetBinding,
    champion: &SeedMember,
    node: &str,
    attempt: usize,
    manifest: &Path,
    manifest_sha256: &str,
    evaluator_sha256: &str,
) -> Result<String> {
    let id = replay_job_id(
        &dataset.dataset_hash,
        &champion.semantic_sha256,
        node,
        attempt,
        manifest_sha256,
    );
    let timeout = if dataset.class_count == 100 || dataset.n_features >= 1_000 {
        300
    } else {
        180
    };
    let args = [
        "--manifest".to_string(),
        "{input:manifest}".to_string(),
        "--output-dir".to_string(),
        "{output:candidate-receipts}".to_string(),
        "--threads".to_string(),
        "1".to_string(),
        "--timeout-seconds".to_string(),
        timeout.to_string(),
    ]
    .iter()
    .map(|value| serde_json::to_string(value).expect("TOML JSON quoting"))
    .collect::<Vec<_>>()
    .join(", ");
    let role_capability = replay_role_capability(node)?;
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [{evaluator:?}, {args}]\nworkdir = {workdir:?}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = 1\nomp_threads = 1\nram_gb = {ram_gb}\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"multiclass replay role capability available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {role_capability:?}]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"multiclass evaluator compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[env]\nOMP_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\n\n[preempt]\npreemptible = false\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n\n[[input]]\nname = \"evaluator\"\nsource = {evaluator_source:?}\ntarget = \"corpus_eval\"\nsha256 = {evaluator_sha256:?}\n\n[[input]]\nname = \"manifest\"\nsource = {manifest:?}\ntarget = \"manifest.json\"\nsha256 = {manifest_sha256:?}\n\n[[output]]\nname = \"candidate-receipts\"\nsource = \"candidate-receipts\"\ndirectory = true\n",
        ram_gb = if attempt >= 2 { 64 } else { 24 },
        evaluator = "{input:evaluator}",
        evaluator_source = corpus_eval_str(),
        workdir = jope_paths::to_string(&paths::jope_state()),
        manifest = manifest.display().to_string(),
    ))
}

pub(crate) fn validate_replay_receipt(
    path: &Path,
    manifest: &Path,
    dataset: &DatasetBinding,
    champion: &SeedMember,
    generation: usize,
    required_node: &str,
) -> Result<Value> {
    let values = read_receipt_values(path)?;
    if values.len() != 1 {
        bail!("replay must emit exactly one candidate receipt");
    }
    let value = values.into_iter().next().expect("one replay receipt");
    let manifest_value = read_value(manifest)?;
    let protocol_sha256 = required_hash(&manifest_value, "/protocol_sha256")?;
    let score = value.get("score").and_then(Value::as_f64);
    let secondary = value.get("secondary_score").and_then(Value::as_f64);
    if value.get("schema").and_then(Value::as_str) != Some("JopeFlywheelCorpusEvalReceipt.v1")
        || value.get("task").and_then(Value::as_str) != Some("multiclass")
        || value.get("dataset_hash").and_then(Value::as_str) != Some(dataset.dataset_hash.as_str())
        || value.get("semantic_sha256").and_then(Value::as_str)
            != Some(champion.semantic_sha256.as_str())
        || value.get("canonical_genome").and_then(Value::as_str)
            != Some(champion.canonical_genome.as_str())
        || value.get("manifest_sha256").and_then(Value::as_str)
            != Some(sha256_file(manifest)?.as_str())
        || value.get("protocol_sha256").and_then(Value::as_str) != Some(protocol_sha256.as_str())
        || value.get("node").and_then(Value::as_str) != Some(required_node)
        || value.get("generation").and_then(Value::as_str)
            != Some(format!("multiclass-generation-{generation:03}").as_str())
        || value.get("valid").and_then(Value::as_bool) != Some(true)
        || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || !score.is_some_and(f64::is_finite)
        || !secondary.is_some_and(f64::is_finite)
        || value
            .get("predictions_sha256")
            .and_then(Value::as_str)
            .is_none()
        || value
            .get("probabilities_sha256")
            .and_then(Value::as_str)
            .is_none()
    {
        bail!("invalid or drifted replay receipt for {required_node}");
    }
    Ok(value)
}

pub(crate) fn manage_winner_replay(
    plan: &Plan,
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    rows: &BTreeMap<String, Value>,
    specs: &mut Vec<PathBuf>,
) -> Result<bool> {
    let Some(score) = checkpoint.best.as_ref() else {
        return Ok(false);
    };
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset has no frozen incumbent")?;
    if !qualifies(
        score,
        checkpoint.generation,
        checkpoint.generation == 0,
        score.macro_f1,
        incumbent.macro_f1,
        checkpoint.generation >= MAX_GENERATIONS - 1,
    ) {
        return Ok(false);
    }
    let champion = checkpoint
        .best_member
        .as_ref()
        .context("winning checkpoint lacks champion member")?;
    let replay_root = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("replay")
        .join(&champion.semantic_sha256);
    let mut replay_values = Vec::new();
    let mut replay_paths = Vec::new();
    for (node_index, node) in NODES[..3].iter().enumerate() {
        let manifest = replay_root.join(format!("{node}.manifest.json"));
        let manifest_bytes = canonical_json(&replay_manifest(
            plan,
            dataset,
            checkpoint.generation,
            champion,
        )?)?;
        if manifest.is_file() {
            if fs::read(&manifest)? != manifest_bytes {
                bail!(
                    "immutable replay manifest collision at {}",
                    manifest.display()
                );
            }
        } else {
            immutable_write(&manifest, &manifest_bytes)?;
        }
        let mut terminal_failures = 0usize;
        let mut harvested = None;
        for attempt in 0..=(MAX_SCHEDULER_PREEMPTIONS + 2) {
            let job_id = replay_job_id(
                &dataset.dataset_hash,
                &champion.semantic_sha256,
                node,
                attempt,
                &sha256_bytes(&manifest_bytes),
            );
            let Some(row) = rows.get(&job_id) else {
                let spec_path =
                    replay_root.join(format!("{node}-a{attempt}-r{REPLAY_SPEC_REVISION}.toml"));
                let rendered = render_replay_spec(
                    dataset,
                    champion,
                    node,
                    attempt,
                    &manifest,
                    &sha256_bytes(&manifest_bytes),
                    &plan.corpus_eval_sha256,
                )?;
                if spec_path.is_file() {
                    if fs::read_to_string(&spec_path)? != rendered {
                        bail!(
                            "immutable replay SCQ spec collision at {}",
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
                    &replay_root,
                    &dataset.dataset_hash,
                    checkpoint.generation,
                    node_index,
                    attempt,
                    &job_id,
                )?);
                break;
            }
            if terminal_infrastructure_state(state)
                || (state == "completed" && row.get("exit_code").and_then(Value::as_i64) != Some(0))
            {
                if scheduler_preemption(row) {
                    continue;
                }
                terminal_failures += 1;
                if terminal_failures == 3 {
                    bail!("replay job {job_id} exhausted infrastructure retries");
                }
                continue;
            }
            break;
        }
        if let Some(path) = harvested {
            replay_values.push(validate_replay_receipt(
                &path,
                &manifest,
                dataset,
                champion,
                checkpoint.generation,
                node,
            )?);
            replay_paths.push(path);
        }
    }
    if replay_values.len() != 3 {
        return Ok(true);
    }
    let predictions = replay_values
        .iter()
        .filter_map(|value| value.get("predictions_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let probabilities = replay_values
        .iter()
        .filter_map(|value| value.get("probabilities_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let primary = replay_values
        .iter()
        .filter_map(|value| value.get("score").and_then(Value::as_f64).map(f64::to_bits))
        .collect::<BTreeSet<_>>();
    let secondary = replay_values
        .iter()
        .filter_map(|value| {
            value
                .get("secondary_score")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    if predictions.len() != 1
        || probabilities.len() != 1
        || primary.len() != 1
        || secondary.len() != 1
    {
        bail!(
            "three-node replay is not bit-identical for {}",
            dataset.dataset_hash
        );
    }
    let status = if incumbent.macro_f1 == 1.0 && score.macro_f1 == 1.0 {
        "ceiling_matched"
    } else {
        "strict_winner"
    };
    let replays = replay_values
        .iter()
        .zip(&replay_paths)
        .map(|(value, path)| {
            json!({
                "node": value.get("node").and_then(Value::as_str),
                "prediction_sha256": value.get("predictions_sha256").and_then(Value::as_str),
                "probability_sha256": value.get("probabilities_sha256").and_then(Value::as_str),
                "macro_f1": value.get("score").and_then(Value::as_f64),
                "accuracy": value.get("secondary_score").and_then(Value::as_f64),
                "receipt_path": path,
                "receipt_sha256": sha256_file(path).expect("validated replay receipt"),
            })
        })
        .collect::<Vec<_>>();
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, checkpoint.generation);
    let ledger = json!({
        "schema": LEDGER_SCHEMA,
        "run_id": RUN_ID,
        "dataset_hash": dataset.dataset_hash,
        "status": status,
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "comparator_sha256": dataset.comparator_sha256,
        "external_model": incumbent.method,
        "external_report_path": incumbent.report_path,
        "external_report_sha256": incumbent.report_sha256,
        "external_macro_f1": incumbent.macro_f1,
        "external_accuracy": incumbent.accuracy,
        "generation": checkpoint.generation,
        "completed_generations": checkpoint.generation + 1,
        "semantic_sha256": champion.semantic_sha256,
        "canonical_genome": champion.canonical_genome,
        "gene_lineage": champion.source,
        "macro_f1": score.macro_f1,
        "accuracy": score.accuracy,
        "score_margin": score.macro_f1 - incumbent.macro_f1,
        "checkpoint_path": checkpoint_file,
        "checkpoint_sha256": sha256_file(&checkpoint_file)?,
        "final_incumbent_count": dataset.completed_incumbent_count,
        "replays": replays,
        "sotarena_report_path": format!("tracks/multiclass272-test-adaptive/{}/jope.json", dataset.dataset_hash),
        "started_at_unix_ms": checkpoint.completed_at_unix_ms,
        "terminal_at_unix_ms": unix_millis(),
    });
    immutable_write(
        &ledger_path(root, &dataset.dataset_hash),
        &canonical_json(&ledger)?,
    )?;
    write_multiclass_terminal_report(dataset, checkpoint, &ledger)?;
    Ok(true)
}

pub(crate) fn write_exhausted_ledger(
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
) -> Result<()> {
    if checkpoint.generation + 1 != MAX_GENERATIONS || !checkpoint.initial_budget_complete {
        bail!(
            "{} cannot terminate exhausted before generation {}",
            dataset.dataset_hash,
            MAX_GENERATIONS - 1
        );
    }
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset has no frozen incumbent")?;
    if checkpoint.best.as_ref().is_some_and(|score| {
        qualifies(
            score,
            checkpoint.generation,
            false,
            score.macro_f1,
            incumbent.macro_f1,
            true,
        )
    }) {
        bail!(
            "{} has a qualifying candidate and cannot be marked exhausted",
            dataset.dataset_hash
        );
    }
    let best_slot = checkpoint
        .slots
        .iter()
        .filter(|slot| slot.valid)
        .min_by(|left, right| match (&left.score, &right.score) {
            (Some(left), Some(right)) => rank_order(left, right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left.slot.cmp(&right.slot),
        });
    let score = best_slot.and_then(|slot| slot.score.as_ref());
    let member = best_slot.map(|slot| &slot.member);
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, checkpoint.generation);
    let ledger = json!({
        "schema": LEDGER_SCHEMA,
        "run_id": RUN_ID,
        "dataset_hash": dataset.dataset_hash,
        "status": "exhausted",
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "comparator_sha256": dataset.comparator_sha256,
        "external_model": incumbent.method,
        "external_report_path": incumbent.report_path,
        "external_report_sha256": incumbent.report_sha256,
        "external_macro_f1": incumbent.macro_f1,
        "external_accuracy": incumbent.accuracy,
        "generation": checkpoint.generation,
        "completed_generations": MAX_GENERATIONS,
        "semantic_sha256": member.map(|value| value.semantic_sha256.as_str()),
        "canonical_genome": member.map(|value| value.canonical_genome.as_str()),
        "gene_lineage": member.map(|value| value.source.as_str()),
        "macro_f1": score.map(|value| value.macro_f1),
        "accuracy": score.map(|value| value.accuracy),
        "score_margin": score.map(|value| value.macro_f1 - incumbent.macro_f1),
        "prediction_sha256": best_slot.and_then(|slot| slot.predictions_sha256.as_deref()),
        "probability_sha256": best_slot.and_then(|slot| slot.probabilities_sha256.as_deref()),
        "checkpoint_path": checkpoint_file,
        "checkpoint_sha256": sha256_file(&checkpoint_file)?,
        "final_incumbent_count": dataset.completed_incumbent_count,
        "replays": [],
        "sotarena_report_path": format!("tracks/multiclass272-test-adaptive/{}/jope.json", dataset.dataset_hash),
        "started_at_unix_ms": checkpoint.completed_at_unix_ms,
        "terminal_at_unix_ms": unix_millis(),
    });
    immutable_write(
        &ledger_path(root, &dataset.dataset_hash),
        &canonical_json(&ledger)?,
    )?;
    write_multiclass_terminal_report(dataset, checkpoint, &ledger)
}
