use super::*;

pub(super) fn run_eval(
    manifest: &Path,
    output_dir: &Path,
    threads: usize,
    timeout: Duration,
) -> Result<()> {
    let manifest_bytes =
        fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let shard: Shard = serde_json::from_slice(&manifest_bytes).context("parse shard manifest")?;
    resolve_reduction(&shard)?;
    if task_metric(&shard.task).is_none() {
        bail!("unsupported task '{}'", shard.task);
    }
    let authority = Arc::new(resolve_authority(&shard)?);
    let reduction_stamp = shard.reduction.as_ref().map(|spec| spec.version.clone());
    let node = short_hostname();
    let eval_one_root = output_dir.join(format!(".eval-one-{}", std::process::id()));
    fs::create_dir_all(&eval_one_root)?;
    let receipt = |c: &ShardCandidate, sha: String, outcome: Outcome, runtime: f64| {
        let bytes = match &outcome {
            Outcome::Score { predictions, .. } => Some(prediction_bytes(predictions)),
            Outcome::Fail { .. } => None,
        };
        let predictions_sha256 = bytes.as_ref().map(|value| sha256_hex(value));
        let multiclass = match &outcome {
            Outcome::Score { multiclass, .. } => multiclass.as_ref(),
            Outcome::Fail { .. } => None,
        };
        let probability_bytes = multiclass.map(|value| probability_bytes(&value.probabilities));
        let probabilities_sha256 = probability_bytes.as_ref().map(|value| sha256_hex(value));
        Receipt {
            schema: RECEIPT_SCHEMA.to_string(),
            task: shard.task.clone(),
            dataset_hash: shard.dataset_hash.clone(),
            semantic_sha256: sha.clone(),
            canonical_genome: c.canonical_genome.clone(),
            tier: authority.tier.clone(),
            metric: authority.metric.clone(),
            score: match &outcome {
                Outcome::Score { score, .. } => Some(*score),
                Outcome::Fail { .. } => None,
            },
            secondary_metric: secondary_metric(&shard.task).to_string(),
            secondary_score: match &outcome {
                Outcome::Score { secondary, .. } => Some(*secondary),
                Outcome::Fail { .. } => None,
            },
            balanced_accuracy: match &outcome {
                Outcome::Score {
                    balanced_accuracy, ..
                } => *balanced_accuracy,
                Outcome::Fail { .. } => None,
            },
            rmse: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.rmse),
                _ => None,
            },
            mse: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mse),
                _ => None,
            },
            mae: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mae),
                _ => None,
            },
            smape: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.smape),
                _ => None,
            },
            wape: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.wape),
                _ => None,
            },
            prediction_min: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.prediction_min),
                _ => None,
            },
            prediction_max: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.prediction_max),
                _ => None,
            },
            target_min: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.target_min),
                _ => None,
            },
            target_max: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.target_max),
                _ => None,
            },
            mase_denominator: match &outcome {
                Outcome::Score {
                    timeseries_metrics: Some(metrics),
                    ..
                } => Some(metrics.mase_denominator),
                _ => None,
            },
            reduction_version: reduction_stamp.clone(),
            predictions_sha256,
            predictions_path: bytes.as_ref().map(|_| format!("predictions/{sha}.f64le")),
            prediction_count: match &outcome {
                Outcome::Score { predictions, .. } => Some(predictions.len()),
                Outcome::Fail { .. } => None,
            },
            class_labels: multiclass.map(|value| value.classes.clone()),
            probabilities_sha256,
            probabilities_path: probability_bytes
                .as_ref()
                .map(|_| format!("probabilities/{sha}.f64le")),
            probability_shape: multiclass.map(|value| {
                [
                    value.probabilities.len(),
                    value.probabilities.first().map_or(0, Vec::len),
                ]
            }),
            prediction_bytes: bytes,
            probability_bytes,
            valid: matches!(outcome, Outcome::Score { .. }),
            failure: match &outcome {
                Outcome::Score { .. } => None,
                Outcome::Fail { class, .. } => Some(class.to_string()),
            },
            failure_message: match &outcome {
                Outcome::Score { .. } => None,
                Outcome::Fail { message, .. } => Some(message.clone()),
            },
            runtime_seconds: runtime,
            train_sha256: shard.train_sha256.clone(),
            test_sha256: shard.test_sha256.clone(),
            node: node.clone(),
            manifest_sha256: manifest_sha256.clone(),
            test_adaptive: authority.test_adaptive,
            benchmark_valid: authority.benchmark_valid,
            selection_bias: authority.selection_bias.clone(),
            generation: authority.generation.clone(),
            label_authority_sha256: authority.authority_sha256.clone(),
            frozen_seed_sha256: authority.frozen_seed_sha256.clone(),
            model_checkpoint_sha256: authority.model_checkpoint_sha256.clone(),
            hall_of_fame_sha256: authority.hall_of_fame_sha256.clone(),
            checkpoint_epoch: shard.checkpoint_epoch.clone(),
            catalog_sha256: shard.catalog_sha256.clone(),
            protocol_sha256: shard.protocol_sha256.clone(),
            manifest_generated_at_ms: shard.generated_at_ms,
            created_at_ms: unix_millis(),
        }
    };
    // Verified semantic identity: warn + substitute the computed hash on drift.
    let semantic_sha = |c: &ShardCandidate| {
        let computed = sha256_hex(c.canonical_genome.as_bytes());
        if computed != c.semantic_sha256 {
            eprintln!(
                "warning: candidate semantic_sha256 {} does not match sha256(canonical_genome) {computed}; using computed",
                c.semantic_sha256
            );
        }
        computed
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let receipts: Vec<Receipt> = pool.install(|| {
        use rayon::prelude::*;
        shard
            .candidates
            .par_iter()
            .enumerate()
            .map(|(candidate_index, c)| {
                let sha = semantic_sha(c);
                let started = Instant::now();
                let result = eval_one_root.join(format!("{candidate_index:08}.json"));
                let outcome = run_with_timeout(manifest, candidate_index, &result, timeout)?;
                let _ = fs::remove_file(result);
                Ok(receipt(c, sha, outcome, started.elapsed().as_secs_f64()))
            })
            .collect::<Result<Vec<_>>>()
    })?;

    fs::create_dir_all(output_dir)?;
    let prediction_dir = output_dir.join("predictions");
    fs::create_dir_all(&prediction_dir)?;
    let probability_dir = output_dir.join("probabilities");
    fs::create_dir_all(&probability_dir)?;
    for line in &receipts {
        if let (Some(relative), Some(expected), Some(bytes)) = (
            line.predictions_path.as_ref(),
            line.predictions_sha256.as_ref(),
            line.prediction_bytes.as_ref(),
        ) {
            if sha256_hex(bytes) != *expected {
                bail!("prediction evidence hash changed before persistence");
            }
            let path = output_dir.join(relative);
            let temporary = prediction_dir.join(format!(".{}.tmp", line.semantic_sha256));
            fs::write(&temporary, bytes)?;
            fs::rename(&temporary, path)?;
        }
        if let (Some(relative), Some(expected), Some(bytes)) = (
            line.probabilities_path.as_ref(),
            line.probabilities_sha256.as_ref(),
            line.probability_bytes.as_ref(),
        ) {
            if sha256_hex(bytes) != *expected {
                bail!("probability evidence hash changed before persistence");
            }
            let path = output_dir.join(relative);
            let temporary = probability_dir.join(format!(".{}.tmp", line.semantic_sha256));
            fs::write(&temporary, bytes)?;
            fs::rename(&temporary, path)?;
        }
    }
    let final_path = output_dir.join(format!("{}.receipts.jsonl", shard.dataset_hash));
    let tmp_path = output_dir.join(format!(".{}.receipts.jsonl.tmp", shard.dataset_hash));
    let mut buffer = Vec::new();
    for line in &receipts {
        serde_json::to_writer(&mut buffer, line)?;
        buffer.push(b'\n');
    }
    fs::write(&tmp_path, &buffer)?;
    fs::rename(&tmp_path, &final_path)?;
    let done = serde_json::json!({
        "dataset_hash": shard.dataset_hash,
        "manifest_sha256": manifest_sha256,
        "receipts": receipts.len(),
        "valid": receipts.iter().filter(|r| r.valid).count(),
        "node": node,
        "finished_at_ms": unix_millis(),
    });
    fs::write(
        output_dir.join(format!("{}.done", shard.dataset_hash)),
        done.to_string(),
    )?;
    eprintln!(
        "wrote {} receipts ({} valid) to {}",
        receipts.len(),
        receipts.iter().filter(|r| r.valid).count(),
        final_path.display()
    );
    Ok(())
}

pub(super) fn validate_manifest(path: &Path) -> Result<()> {
    let shard: Shard = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .context("parse shard manifest")?;
    let reduction_version = resolve_reduction(&shard)?;
    if task_metric(&shard.task).is_none() {
        bail!("unsupported task '{}'", shard.task);
    }
    if shard.candidates.is_empty() || shard.candidates.len() > 128 {
        bail!("manifest must contain one through 128 candidates");
    }
    resolve_authority(&shard)?;
    let timeseries_schema = if shard.task == "timeseries" {
        let data = load_timeseries_shard_data(&shard)?;
        let feature_names = match reduction_version {
            ReductionVersion::V1 => make_supervised_reduction(&data)?.feature_names,
            ReductionVersion::V2 => make_direct_reduction(&data)?.feature_names,
        };
        Some(DataSchema {
            columns: feature_names
                .iter()
                .map(|name| ColumnSpec::numeric(name.clone()))
                .collect(),
            row_semantics: RowSemantics::Iid,
        })
    } else {
        None
    };
    let multiclass_features = if shard.task == "multiclass" {
        let train = load_pinned_csv(&shard.train_csv, &shard.train_sha256, "train")?;
        let test = load_pinned_csv(&shard.test_csv, &shard.test_sha256, "test")?;
        if train.features != test.features {
            bail!(
                "train/test feature widths differ ({} vs {})",
                train.features,
                test.features
            );
        }
        let meta_path = shard
            .meta_json
            .as_deref()
            .context("multiclass shard manifest must pin meta_json")?;
        let meta_sha256 = shard
            .meta_sha256
            .as_deref()
            .context("multiclass shard manifest must pin meta_sha256")?;
        let meta = load_pinned_json(meta_path, meta_sha256, "meta")?;
        decode_multiclass_splits(&train.y, &test.y, &meta).map_err(anyhow::Error::msg)?;
        Some(train.features)
    } else {
        None
    };
    let mut seen = std::collections::BTreeSet::new();
    for (index, candidate) in shard.candidates.iter().enumerate() {
        let computed = sha256_hex(candidate.canonical_genome.as_bytes());
        if candidate.semantic_sha256 != computed {
            bail!("candidate {index} semantic_sha256 does not match canonical genome");
        }
        if !seen.insert(computed) {
            bail!("candidate {index} duplicates an earlier canonical genome");
        }
        let genome = match parse_canonical(candidate) {
            Ok(genome) => genome,
            Err(Outcome::Fail { class, message }) => {
                bail!("candidate {index} {class}: {message}");
            }
            Err(Outcome::Score { .. }) => unreachable!("parser never returns a score"),
        };
        if shard.task == "binary" {
            let task =
                TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
            validate_v3_genome(&genome, &task, &binary_schema(8))
                .with_context(|| format!("candidate {index} is not binary-task-valid"))?;
        } else if shard.task == "multiclass" {
            let task = TaskSpec::multiclass(Objective::MacroF1)
                .expect("macro_f1 is a valid multiclass objective");
            validate_v3_genome(
                &genome,
                &task,
                &binary_schema(multiclass_features.expect("multiclass data loaded")),
            )
            .with_context(|| format!("candidate {index} is not multiclass-task-valid"))?;
        } else if shard.task == "timeseries" {
            if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
                bail!("candidate {index} uses a binary classification head");
            }
            let task = TaskSpec::regression(Objective::Rmse)
                .expect("rmse is a valid regression objective");
            let schema = timeseries_schema.as_ref().expect("timeseries schema built");
            validate_v3_genome(&genome, &task, schema)
                .with_context(|| format!("candidate {index} is not timeseries-task-valid"))?;
        }
    }
    Ok(())
}
