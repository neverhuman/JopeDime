use super::*;

fn task_valid(source: &str, task: &str) -> bool {
    let Ok(genome) = GenomeIr::parse(source) else {
        return false;
    };
    if genome.canonical() != source {
        return false;
    }
    let GenomeIr::Pipeline(pipeline) = genome else {
        return false;
    };
    let Some(head) = pipeline.stages.last().map(|stage| stage.op.as_str()) else {
        return false;
    };
    // Mirrors the wave-side allowlist in jope_selfplay_wave.rs exactly: timeseries
    // shares the regression arm because corpus_eval validates a timeseries genome
    // as TaskSpec::regression(Rmse) over the reduced schema, and multiclass admits
    // only the four multiclass_* heads (the same set corpus_eval's V3_EXTRA_HEADS
    // and the multiclass60 campaign pin). If these two allowlists ever disagree,
    // the bridge injects seeds the evaluator rejects -- silent seed loss.
    match task {
        "binary" => matches!(
            head,
            "logit_ridge" | "logistic" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
        ),
        "regression" | "timeseries" => {
            head != "logit_ridge"
                && head != "logistic"
                && (feat_math::heads::is_head_op(head)
                    || matches!(head, "tab_boost" | "extra_trees" | "gam_boost" | "pls"))
        }
        "multiclass" => matches!(
            head,
            "multiclass_ridge"
                | "multiclass_ovr_logistic"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
        ),
        _ => false,
    }
}

pub(super) fn verify_proposals(input: &ProposalInputs) -> Result<VerifiedProposals> {
    ensure!(
        !input.dataset.trim().is_empty(),
        "dataset identity must not be empty"
    );
    let checkpoint_pointer: CheckpointPointer =
        serde_json::from_value(read_json(&input.checkpoint_pointer)?).with_context(|| {
            format!(
                "parse checkpoint pointer {}",
                input.checkpoint_pointer.display()
            )
        })?;
    validate_sha256(
        &checkpoint_pointer.checkpoint_sha256,
        "checkpoint pointer SHA-256",
    )?;
    let checkpoint_artifact =
        resolve_pointer_path(&input.checkpoint_pointer, &checkpoint_pointer.checkpoint);
    ensure!(
        checkpoint_artifact.is_file(),
        "checkpoint artifact is absent: {}",
        checkpoint_artifact.display()
    );
    ensure!(
        sha256_file(&checkpoint_artifact)? == checkpoint_pointer.checkpoint_sha256,
        "checkpoint artifact content drift"
    );

    let proposal_pointer: ProposalPointer =
        serde_json::from_value(read_json(&input.proposals_pointer)?).with_context(|| {
            format!(
                "parse proposal pointer {}",
                input.proposals_pointer.display()
            )
        })?;
    validate_sha256(&proposal_pointer.proposals_sha256, "proposal file SHA-256")?;
    ensure!(
        proposal_pointer.checkpoint_sha256 == checkpoint_pointer.checkpoint_sha256,
        "proposal pointer does not reference the active checkpoint"
    );
    let proposal_file = resolve_pointer_path(&input.proposals_pointer, &proposal_pointer.proposals);
    ensure!(
        proposal_file.is_file(),
        "proposal file is absent: {}",
        proposal_file.display()
    );
    ensure!(
        sha256_file(&proposal_file)? == proposal_pointer.proposals_sha256,
        "proposal pointer content drift"
    );

    let file = File::open(&proposal_file)
        .with_context(|| format!("open proposal file {}", proposal_file.display()))?;
    let mut all = Vec::new();
    let mut task_invalid_skipped = 0usize;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: GpProposal = serde_json::from_str(&line)
            .with_context(|| format!("parse proposal JSONL line {}", index + 1))?;
        ensure!(
            row.model_checkpoint_sha256 == checkpoint_pointer.checkpoint_sha256,
            "proposal file mixes checkpoint epochs at line {}",
            index + 1
        );
        ensure!(
            sha256_bytes(row.canonical_genome.as_bytes()) == row.semantic_sha256,
            "proposal semantic identity drift at line {}",
            index + 1
        );
        // Task-invalid rows are skipped and counted rather than aborting the
        // whole file, matching the sanitize-proposals policy; the count is
        // surfaced in the durable receipt summary.
        if !task_valid(&row.canonical_genome, &row.task) {
            task_invalid_skipped += 1;
            continue;
        }
        all.push(row);
    }
    let proposals = export_gp_seeds(
        &all,
        input.task.as_str(),
        &input.dataset,
        &checkpoint_pointer.checkpoint_sha256,
    )?;
    Ok(VerifiedProposals {
        checkpoint_sha256: checkpoint_pointer.checkpoint_sha256,
        checkpoint_artifact,
        checkpoint_pointer_sha256: sha256_file(&input.checkpoint_pointer)?,
        proposal_file,
        proposal_file_sha256: proposal_pointer.proposals_sha256,
        proposal_pointer_sha256: sha256_file(&input.proposals_pointer)?,
        proposals,
        task_invalid_skipped,
    })
}

pub(super) fn read_population(path: &Path) -> Result<Vec<GpPopulationMember>> {
    let value = read_json(path)?;
    let values = match &value {
        Value::Array(rows) => rows.clone(),
        Value::Object(object) => ["population", "members", "slots"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_array).cloned())
            .context("population JSON requires an array or population/members/slots array")?,
        _ => bail!("population JSON must be an array or object"),
    };
    let mut members = Vec::with_capacity(values.len());
    let mut semantics = HashSet::new();
    for (index, value) in values.into_iter().enumerate() {
        let member: GpPopulationMember = serde_json::from_value(value)
            .with_context(|| format!("parse population member {index}"))?;
        validate_sha256(
            &member.semantic_sha256,
            &format!("population member {index} semantic SHA-256"),
        )?;
        ensure!(
            sha256_bytes(member.canonical_genome.as_bytes()) == member.semantic_sha256,
            "population semantic identity drift at slot {index}"
        );
        ensure!(
            member.score.into_iter().all(f64::is_finite),
            "population contains non-finite score at slot {index}"
        );
        ensure!(
            semantics.insert(member.semantic_sha256.clone()),
            "population repeats semantic identity at slot {index}"
        );
        members.push(member);
    }
    ensure!(!members.is_empty(), "GP population must not be empty");
    Ok(members)
}

pub(super) fn read_dynamic_robust_bank(
    corpus: &Path,
    task: Task,
    protocol_sha256: &str,
    limit: usize,
    minimum_datasets: usize,
) -> Result<(Vec<RobustSeed>, String, Value)> {
    let value = selfplay_ingest::dynamic_seed_bank(
        corpus,
        task.as_str(),
        protocol_sha256,
        limit,
        minimum_datasets,
    )?;
    let bank: RobustBank =
        serde_json::from_value(value.clone()).context("parse dynamic robust seed bank")?;
    ensure!(
        bank.schema == ROBUST_BANK_SCHEMA,
        "unsupported robust seed-bank schema"
    );
    ensure!(
        bank.tasks.keys().all(|key| {
            matches!(
                key.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            )
        }),
        "robust seed bank contains an unsupported task"
    );
    let mut rows = bank
        .tasks
        .get(task.as_str())
        .cloned()
        .with_context(|| format!("robust seed bank lacks {} seeds", task.as_str()))?;
    let mut semantics = HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        ensure!(
            row.task == task.as_str(),
            "robust seed task-map provenance drift at row {index}"
        );
        ensure!(
            row.dataset_count >= 2,
            "robust seed is not repeated across datasets at row {index}"
        );
        ensure!(
            row.mean_within_dataset_percentile.is_finite()
                && row.worst_within_dataset_percentile.is_finite(),
            "robust seed has non-finite rank evidence at row {index}"
        );
        validate_sha256(
            &row.semantic_sha256,
            &format!("robust seed {index} semantic SHA-256"),
        )?;
        ensure!(
            sha256_bytes(row.canonical_genome.as_bytes()) == row.semantic_sha256,
            "robust seed semantic identity drift at row {index}"
        );
        ensure!(
            task_valid(&row.canonical_genome, task.as_str()),
            "robust seed is invalid for {} at row {index}",
            task.as_str()
        );
        ensure!(
            semantics.insert(row.semantic_sha256.clone()),
            "robust seed bank repeats a semantic at row {index}"
        );
    }
    rows.sort_by(|left, right| {
        right
            .mean_within_dataset_percentile
            .total_cmp(&left.mean_within_dataset_percentile)
            .then_with(|| {
                right
                    .worst_within_dataset_percentile
                    .total_cmp(&left.worst_within_dataset_percentile)
            })
            .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
            .then_with(|| right.dataset_count.cmp(&left.dataset_count))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    Ok((rows, digest, value))
}
