use super::*;

pub(super) fn select_panel(request: &Request) -> Result<Vec<Dataset>> {
    let policy_rows = request
        .datasets
        .iter()
        .map(|row| PanelDataset {
            dataset_hash: row.dataset_hash.clone(),
            source_family: row.source_family.clone(),
        })
        .collect::<Vec<_>>();
    let selected = family_diverse_panel(&policy_rows, PANEL_SIZE, request.panel_seed)
        .map_err(anyhow::Error::msg)?;
    let by_hash = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    selected
        .into_iter()
        .map(|row| {
            by_hash
                .get(row.dataset_hash.as_str())
                .map(|value| (*value).clone())
                .context("selected panel dataset disappeared")
        })
        .collect()
}

pub(super) fn qualified_candidates(request: &Request) -> Result<Vec<Qualified>> {
    let rows = request
        .candidates
        .iter()
        .map(|row| CompileCandidate {
            task: request.task.clone(),
            dataset_hash: row.dataset_hash.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            score: row.score,
            valid: row.valid,
        })
        .collect::<Vec<_>>();
    let incumbents = request
        .datasets
        .iter()
        .filter_map(|row| {
            row.incumbent_score
                .map(|score| ((request.task.clone(), row.dataset_hash.clone()), score))
        })
        .collect::<BTreeMap<_, _>>();
    let selected = compiled_finalists(&rows, &incumbents);
    let mut rank_by_identity = BTreeMap::new();
    for dataset in &request.datasets {
        let mut ranked = request
            .candidates
            .iter()
            .filter(|row| {
                row.dataset_hash == dataset.dataset_hash && row.valid && row.score.is_finite()
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        for (index, row) in ranked.into_iter().enumerate() {
            rank_by_identity.insert(
                (row.dataset_hash.clone(), row.semantic_sha256.clone()),
                index + 1,
            );
        }
    }
    let datasets = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::with_capacity(selected.len());
    for row in selected {
        let dataset = datasets
            .get(row.dataset_hash.as_str())
            .context("qualified candidate dataset disappeared")?;
        let rank = *rank_by_identity
            .get(&(row.dataset_hash.clone(), row.semantic_sha256.clone()))
            .context("qualified candidate rank disappeared")?;
        let strict_incumbent_beat = dataset
            .incumbent_score
            .is_some_and(|incumbent| row.score > incumbent);
        ensure!(
            rank <= 2 || strict_incumbent_beat,
            "candidate was qualified without top-two or strict-win evidence"
        );
        output.push(Qualified {
            dataset_hash: row.dataset_hash,
            semantic_sha256: row.semantic_sha256,
            score: row.score,
            rank,
            top_two: rank <= 2,
            strict_incumbent_beat,
            incumbent_score: dataset.incumbent_score,
        });
    }
    let mut per_dataset = BTreeMap::<&str, usize>::new();
    for row in &output {
        *per_dataset.entry(&row.dataset_hash).or_default() += 1;
    }
    ensure!(
        per_dataset.values().all(|count| *count <= 2),
        "more than two finalists selected for a dataset/checkpoint"
    );
    Ok(output)
}

pub(super) fn audit_report(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    let status = cache_status(&cache_root(args))?;
    Ok(json!({
        "schema": "JopeCompileEconomyAudit.v1",
        "mode": "audit",
        "mutated": false,
        "request_sha256": request_sha256,
        "task": request.task,
        "checkpoint_sha256": request.checkpoint_sha256,
        "panel_size": panel.len(),
        "panel_dataset_hashes": panel.iter().map(|row| &row.dataset_hash).collect::<Vec<_>>(),
        "qualified_finalists": qualified,
        "qualified_count": qualified.len(),
        "max_finalists_per_dataset_checkpoint": 2,
        "broad_evaluator": args.evaluator,
        "compile_policy": "explicit_build_only_after_complete_native_replay",
        "fitted_state_scope": "dataset_train_sha256",
        "cache": status,
    }))
}

pub(super) fn candidate_map(request: &Request) -> BTreeMap<(&str, &str), &Candidate> {
    request
        .candidates
        .iter()
        .map(|row| {
            (
                (row.dataset_hash.as_str(), row.semantic_sha256.as_str()),
                row,
            )
        })
        .collect()
}

pub(super) fn verify_dataset_files(dataset: &Dataset) -> Result<()> {
    ensure!(
        sha256_file(&dataset.train_csv)? == dataset.train_sha256,
        "train content hash drift for {}",
        dataset.dataset_hash
    );
    ensure!(
        sha256_file(&dataset.test_csv)? == dataset.test_sha256,
        "test content hash drift for {}",
        dataset.dataset_hash
    );
    Ok(())
}
