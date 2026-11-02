use super::*;

pub(super) fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = q * (sorted.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
}

pub(super) fn parse_initial_population(
    bytes: &[u8],
    expected: usize,
) -> Result<Vec<PopulationMember>> {
    let text = std::str::from_utf8(bytes).context("generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body)
            .with_context(|| format!("generation-zero line {} does not parse", line_number + 1))?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!("generation-zero line {} is not a pipeline", line_number + 1);
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!("generation-zero line {} is not canonical", line_number + 1);
        }
        population.push(PopulationMember {
            semantic_sha256: sha256_bytes(canonical.as_bytes()),
            canonical_genome: canonical,
            ancestry: Ancestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    validate_unique_population(&population, expected)?;
    Ok(population)
}

pub(super) fn validate_unique_population(
    population: &[PopulationMember],
    expected: usize,
) -> Result<()> {
    if population.len() != expected {
        bail!(
            "exact population requires {expected} slots, found {}",
            population.len()
        );
    }
    let unique = population
        .iter()
        .map(|member| member.semantic_sha256.as_str())
        .collect::<HashSet<_>>();
    if unique.len() != expected {
        bail!("exact population is not canonical-unique");
    }
    Ok(())
}

pub(super) fn validate_campaign_config(
    config: &CampaignConfig,
    train: &RegressionData,
    population: &[PopulationMember],
) -> Result<()> {
    if config.generations == 0
        || config.cv_folds < 2
        || config.confirm_folds < 2
        || config.elites == 0
        || config.parent_pool > config.population
        || config.tournament != TOURNAMENT
        || config.final_cv7_candidates == 0
        || config.final_cv7_candidates > config.parent_pool
        || config.rayon_threads < config.population
    {
        bail!("invalid exact-CV campaign configuration");
    }
    if config
        .expected_train_rows
        .is_some_and(|rows| train.y.len() != rows)
    {
        bail!("training row count differs from campaign configuration");
    }
    validate_unique_population(population, config.population)
}

pub(super) fn validate_generation_zero_provenance(
    bytes: &[u8],
    population_sha256: &str,
    population: usize,
) -> Result<serde_json::Value> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("parse generation-zero provenance JSON")?;
    if value.get("schema").and_then(serde_json::Value::as_str)
        != Some("JainExactCvGenerationZeroProvenance.v1")
        || value.get("dataset").and_then(serde_json::Value::as_str) != Some(DATASET)
        || value
            .get("ordered_population_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(population_sha256)
        || value.get("population").and_then(serde_json::Value::as_u64) != Some(population as u64)
    {
        bail!("generation-zero provenance does not bind the requested dataset and population");
    }
    for section in ["report_finalists", "archived_seeds"] {
        let source = value
            .get(section)
            .with_context(|| format!("generation-zero provenance lacks {section}"))?;
        let path = source
            .get("source_path")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("generation-zero provenance {section} lacks source_path"))?;
        let expected_sha256 = source
            .get("source_sha256")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("generation-zero provenance {section} lacks source_sha256"))?;
        validate_sha256("provenance source_sha256", expected_sha256)?;
        validate_file_hash(Path::new(path), expected_sha256, section)?;
    }
    Ok(value)
}

pub(super) fn publish_initial_population(
    artifact_root: &Path,
    population: &[PopulationMember],
    population_sha256: &str,
) -> Result<()> {
    let manifest = serde_json::json!({
        "schema": "JainExactCvInitialPopulation.v1",
        "ordered_file_sha256": population_sha256,
        "slots": population.iter().enumerate().map(|(slot, member)| serde_json::json!({
            "slot": slot,
            "semantic_sha256": member.semantic_sha256,
            "canonical_genome": member.canonical_genome,
        })).collect::<Vec<_>>(),
    });
    immutable_write(
        &artifact_root.join("initial-population-manifest.v1.json"),
        &canonical_json(&manifest)?,
    )
}

pub(super) fn build_fold_manifest(
    dataset: &str,
    train_sha256: &str,
    rows: usize,
    folds: usize,
) -> Result<FoldManifest> {
    let mut assignments = Vec::with_capacity(folds);
    for fold in 0..folds {
        let mut train_rows = Vec::new();
        let mut validation_rows = Vec::new();
        for row in 0..rows {
            if fold_of(dataset, row, folds) == fold {
                validation_rows.push(row);
            } else {
                train_rows.push(row);
            }
        }
        if train_rows.is_empty() || validation_rows.is_empty() {
            bail!("fold manifest contains an empty split");
        }
        assignments.push(FoldAssignment {
            fold,
            train_rows_sha256: sha256_bytes(&canonical_json(&train_rows)?),
            validation_rows_sha256: sha256_bytes(&canonical_json(&validation_rows)?),
            train_rows,
            validation_rows,
        });
    }
    Ok(FoldManifest {
        schema: FOLD_MANIFEST_SCHEMA.to_string(),
        dataset_hash: dataset.to_string(),
        dataset_train_sha256: train_sha256.to_string(),
        fold_authority: FOLD_AUTHORITY.to_string(),
        folds,
        rows,
        assignments,
    })
}

pub(super) fn publish_emitted(
    root: &Path,
    semantic: &str,
    emitted: &Emitted,
) -> Result<ArtifactHashes> {
    let directory = root.join("artifacts").join(semantic);
    let python_sha256 = sha256_bytes(emitted.python.as_bytes());
    let rust_sha256 = sha256_bytes(emitted.rust.as_bytes());
    let manifest_sha256 = sha256_bytes(emitted.manifest.as_bytes());
    immutable_write(
        &directory.join(format!("{python_sha256}.py")),
        emitted.python.as_bytes(),
    )?;
    immutable_write(
        &directory.join(format!("{rust_sha256}.rs")),
        emitted.rust.as_bytes(),
    )?;
    immutable_write(
        &directory.join(format!("{manifest_sha256}.manifest.json")),
        emitted.manifest.as_bytes(),
    )?;
    Ok(ArtifactHashes {
        python_sha256,
        rust_sha256,
        manifest_sha256,
    })
}

pub(super) fn publish_selection_freeze(root: &Path, selection: &SelectionFreeze) -> Result<()> {
    let selection_file = "selection-freeze.v1.json";
    let bytes = canonical_json(selection)?;
    let selection_sha256 = sha256_bytes(&bytes);
    immutable_write(&root.join(selection_file), &bytes)?;
    let complete = SelectionComplete {
        schema: SELECTION_COMPLETE_SCHEMA.to_string(),
        status: "COMPLETE".to_string(),
        selection_file: selection_file.to_string(),
        selection_sha256,
    };
    immutable_write(
        &root.join("SELECTION_COMPLETE"),
        &canonical_json(&complete)?,
    )?;
    sync_directory(root)
}

pub(super) fn validate_selection_freeze(root: &Path) -> Result<(SelectionFreeze, String)> {
    let marker_path = root.join("SELECTION_COMPLETE");
    let marker: SelectionComplete =
        serde_json::from_slice(&fs::read(&marker_path).with_context(|| {
            format!("read immutable selection marker {}", marker_path.display())
        })?)
        .context("parse selection completion marker")?;
    if marker.schema != SELECTION_COMPLETE_SCHEMA
        || marker.status != "COMPLETE"
        || marker.selection_file != "selection-freeze.v1.json"
    {
        bail!("selection completion marker is invalid");
    }
    validate_sha256("selection_sha256", &marker.selection_sha256)?;
    let selection_path = root.join(&marker.selection_file);
    let bytes = fs::read(&selection_path)
        .with_context(|| format!("read immutable selection {}", selection_path.display()))?;
    if sha256_bytes(&bytes) != marker.selection_sha256 {
        bail!("selection freeze differs from its completion hash");
    }
    let selection: SelectionFreeze =
        serde_json::from_slice(&bytes).context("parse selection freeze")?;
    if selection.schema != SELECTION_SCHEMA
        || selection.evaluator != EVALUATOR
        || selection.quick_cv5_fold0.fold_plan_folds < 2
        || selection.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
        || selection.cv7.fold_plan_folds < 2
        || selection.cv7.evaluated_fold_ids
            != (0..selection.cv7.fold_plan_folds).collect::<Vec<_>>()
        || selection.cv7_finalists.is_empty()
        || selection.canonical_genome.trim() != selection.canonical_genome
        || sha256_bytes(selection.canonical_genome.as_bytes()) != selection.semantic_sha256
    {
        bail!("selection freeze structural validation failed");
    }
    let mut evidence = vec![&selection.quick_cv5_fold0, &selection.cv7];
    let mut finalist_semantics = HashSet::new();
    for (expected_rank, finalist) in selection.cv7_finalists.iter().enumerate() {
        if finalist.quick_rank != expected_rank
            || finalist.generation != selection.generation
            || GenomeIr::parse(&finalist.canonical_genome)
                .map(|genome| genome.canonical())
                .ok()
                .as_deref()
                != Some(finalist.canonical_genome.as_str())
            || sha256_bytes(finalist.canonical_genome.as_bytes()) != finalist.semantic_sha256
            || !finalist_semantics.insert(finalist.semantic_sha256.as_str())
            || finalist.quick_cv5_fold0.fold_plan_folds != selection.quick_cv5_fold0.fold_plan_folds
            || finalist.quick_cv5_fold0.evaluated_fold_ids != [QUICK_CV5_HOLDOUT_FOLD]
            || finalist.quick_cv5_fold0.fold_manifest_sha256
                != selection.quick_cv5_fold0.fold_manifest_sha256
            || finalist.cv7.is_some() == finalist.rejection_reason.is_some()
        {
            bail!("selection freeze contains an invalid CV7 finalist row");
        }
        if finalist.cv7.as_ref().is_some_and(|cv7| {
            cv7.fold_plan_folds != selection.cv7.fold_plan_folds
                || cv7.evaluated_fold_ids != (0..cv7.fold_plan_folds).collect::<Vec<_>>()
                || cv7.fold_manifest_sha256 != selection.cv7.fold_manifest_sha256
        }) {
            bail!("selection freeze contains inconsistent full-CV7 evidence");
        }
        evidence.push(&finalist.quick_cv5_fold0);
        if let Some(cv7) = finalist.cv7.as_ref() {
            evidence.push(cv7);
        }
    }
    for evidence in evidence {
        validate_sha256("fold_manifest_sha256", &evidence.fold_manifest_sha256)?;
        if evidence.fold_r2.len() != evidence.evaluated_fold_ids.len()
            || evidence.evaluated_fold_ids.is_empty()
            || evidence
                .evaluated_fold_ids
                .iter()
                .any(|fold| *fold >= evidence.fold_plan_folds)
            || evidence.fold_r2.iter().any(|v| !v.is_finite())
            || !evidence.pooled_oof_r2.is_finite()
            || !evidence.pooled_oof_rmse.is_finite()
            || !evidence.robust_fitness.is_finite()
            || (robust_fitness(evidence.pooled_oof_r2, &evidence.fold_r2) - evidence.robust_fitness)
                .abs()
                > 1e-12
        {
            bail!("selection freeze contains invalid CV evidence");
        }
    }
    let selected_finalist = selection
        .cv7_finalists
        .iter()
        .find(|finalist| finalist.semantic_sha256 == selection.semantic_sha256)
        .context("selected genome is absent from the CV7 finalist set")?;
    if selected_finalist.slot != selection.slot
        || selected_finalist.canonical_genome != selection.canonical_genome
        || selected_finalist.cv7.is_none()
        || canonical_json(selected_finalist.cv7.as_ref().unwrap())?
            != canonical_json(&selection.cv7)?
    {
        bail!("selected CV7 finalist evidence does not match the freeze");
    }
    let mut ranked_finalists = selection
        .cv7_finalists
        .iter()
        .filter(|finalist| finalist.cv7.is_some())
        .collect::<Vec<_>>();
    ranked_finalists.sort_by(rank_finalists);
    if ranked_finalists
        .first()
        .is_none_or(|winner| winner.semantic_sha256 != selection.semantic_sha256)
    {
        bail!("selection freeze did not choose the deterministic CV7 winner");
    }
    let cv5_path = root.join("folds/cv5.v1.json");
    let cv7_path = root.join("folds/cv7.v1.json");
    if sha256_file(&cv5_path)? != selection.quick_cv5_fold0.fold_manifest_sha256
        || sha256_file(&cv7_path)? != selection.cv7.fold_manifest_sha256
    {
        bail!("selection freeze fold manifest hashes do not match");
    }
    if sha256_file(&root.join("slots.v1.jsonl"))? != selection.ledger_sha256 {
        bail!("selection freeze ledger hash does not match");
    }
    Ok((selection, marker.selection_sha256))
}
