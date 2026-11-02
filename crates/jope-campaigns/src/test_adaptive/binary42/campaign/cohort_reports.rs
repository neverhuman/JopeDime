use super::*;

pub(crate) fn summarize_dataset(
    run_id: &str,
    spec: &FrozenDataset,
) -> Result<(DatasetValidationSummary, Vec<AllEvaluationIndexRecord>)> {
    let root = campaign_root(run_id, &spec.id);
    let checkpoints = load_checkpoints(&root)?;
    let complete = root.join("RUN_COMPLETE").exists();
    let exhausted = root.join("RUN_EXHAUSTED").exists();
    if complete && exhausted {
        bail!(
            "Binary42 dataset {} has conflicting terminal markers",
            spec.id
        );
    }
    let status = if complete {
        "complete"
    } else if exhausted {
        "exhausted"
    } else {
        "in_progress"
    };
    let winner = checkpoints
        .iter()
        .find_map(|checkpoint| checkpoint.qualifying_winner.as_ref());
    let best_auc = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.best_auc)
        .max_by(f64::total_cmp);
    let unique_genomes = checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .map(|slot| &slot.member.semantic_sha256)
        .collect::<BTreeSet<_>>()
        .len();
    let evaluations = evaluation_records(&checkpoints)
        .into_iter()
        .map(|evaluation| AllEvaluationIndexRecord {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            evaluation,
        })
        .collect();
    Ok((
        DatasetValidationSummary {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            status: status.to_string(),
            completed_generations: checkpoints.len(),
            logical_slots: checkpoints.len() * POPULATION,
            unique_genomes,
            best_auc,
            winner_generation: winner.map(|record| record.generation),
            winner_semantic_sha256: winner.map(|record| record.semantic_sha256.clone()),
        },
        evaluations,
    ))
}

pub(crate) fn rebuild_cohort_indexes(
    run_id: &str,
    cohort: &FrozenCohort,
) -> Result<(Vec<DatasetValidationSummary>, usize)> {
    let cohort_root = run_root(run_id).join("cohort");
    let mut datasets = Vec::with_capacity(cohort.datasets.len());
    let mut evaluations = Vec::new();
    for spec in &cohort.datasets {
        let root = campaign_root(run_id, &spec.id);
        if root.exists() {
            let (summary, mut records) = summarize_dataset(run_id, spec)?;
            datasets.push(summary);
            evaluations.append(&mut records);
        } else {
            datasets.push(DatasetValidationSummary {
                dataset: spec.id.clone(),
                display_name: spec.display_name.clone(),
                frozen_order: spec.order,
                status: "pending".to_string(),
                completed_generations: 0,
                logical_slots: 0,
                unique_genomes: 0,
                best_auc: None,
                winner_generation: None,
                winner_semantic_sha256: None,
            });
        }
    }
    let dataset_index = json!({
        "schema": "JopeBinary42AllDatasetIndex.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "datasets": &datasets,
    });
    atomic_rebuild(
        &cohort_root.join("datasets.json"),
        &canonical_json(&dataset_index)?,
    )?;
    let mut dataset_csv = b"dataset,display_name,frozen_order,status,completed_generations,logical_slots,unique_genomes,best_auc,winner_generation,winner_semantic_sha256\n".to_vec();
    for dataset in &datasets {
        dataset_csv.extend_from_slice(
            format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&dataset.dataset),
                csv_field(&dataset.display_name),
                dataset.frozen_order,
                csv_field(&dataset.status),
                dataset.completed_generations,
                dataset.logical_slots,
                dataset.unique_genomes,
                dataset
                    .best_auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                dataset
                    .winner_generation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(dataset.winner_semantic_sha256.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(&cohort_root.join("datasets.csv"), &dataset_csv)?;
    let mut evaluation_jsonl = Vec::new();
    let mut evaluation_csv = b"dataset,display_name,frozen_order,test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,runtime_seconds,receipt_sha256,prediction_sha256,failure\n".to_vec();
    for record in &evaluations {
        evaluation_jsonl.extend_from_slice(&canonical_json(record)?);
        let evaluation = &record.evaluation;
        evaluation_csv.extend_from_slice(
            format!(
                "{},{},{},true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&record.dataset),
                csv_field(&record.display_name),
                record.frozen_order,
                evaluation.generation,
                evaluation.slot,
                csv_field(&evaluation.semantic_sha256),
                csv_field(&evaluation.ancestry_operation),
                csv_field(&evaluation.ancestry_parents.join(";")),
                csv_field(&evaluation.cache_disposition),
                csv_field(&evaluation.assigned_node),
                csv_field(&evaluation.evaluation_node),
                csv_field(&evaluation.scq_job_id),
                evaluation
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation.complexity,
                evaluation
                    .runtime_seconds
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(&evaluation.receipt_sha256),
                csv_field(&evaluation.prediction_sha256),
                csv_field(evaluation.failure.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(
        &cohort_root.join("all-evaluations.jsonl"),
        &evaluation_jsonl,
    )?;
    atomic_rebuild(&cohort_root.join("all-evaluations.csv"), &evaluation_csv)?;
    let mut winners = b"dataset,display_name,generation,semantic_sha256,best_auc\n".to_vec();
    let mut exhausted = b"dataset,display_name,completed_generations,best_auc\n".to_vec();
    for dataset in &datasets {
        if dataset.status == "complete" {
            winners.extend_from_slice(
                format!(
                    "{},{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.winner_generation.unwrap(),
                    csv_field(dataset.winner_semantic_sha256.as_deref().unwrap()),
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        } else if dataset.status == "exhausted" {
            exhausted.extend_from_slice(
                format!(
                    "{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.completed_generations,
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        }
    }
    atomic_rebuild(&cohort_root.join("winners.csv"), &winners)?;
    atomic_rebuild(&cohort_root.join("exhausted.csv"), &exhausted)?;
    Ok((datasets, evaluations.len()))
}
