fn terminal_op(genome: &GenomeIr) -> Option<&str> {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.last().map(|stage| stage.op.as_str()),
        _ => None,
    }
}

fn is_binary_head(op: &str) -> bool {
    matches!(op, "logit_ridge" | "logistic")
}

fn is_head_like(op: &str) -> bool {
    is_head_op(op)
        || matches!(
            op,
            "tab_boost" | "extra_trees" | "gam_boost" | "pls" | "logit_ridge"
        )
}

fn rewrite_binary_body_to_regression(canonical: &str) -> Result<String> {
    let genome = GenomeIr::parse(canonical).context("parse binary transfer genome")?;
    let GenomeIr::Pipeline(mut pipeline) = genome else {
        bail!("binary transfer genome is not a pipeline");
    };
    while pipeline
        .stages
        .last()
        .is_some_and(|stage| is_head_like(&stage.op) || is_binary_head(&stage.op))
    {
        pipeline.stages.pop();
    }
    pipeline.stages.retain(|stage| !is_target_op(&stage.op));
    let mut text = String::from("(pipe X");
    for stage in pipeline.stages {
        text.push(' ');
        text.push_str(&stage.to_string());
    }
    text.push_str(" (ridge))");
    Ok(GenomeIr::parse(&text)?.canonical())
}

fn timeseries_schema(spec: &DatasetSpec) -> DataSchema {
    let mut names = spec.known_future_covariates.clone();
    if names.is_empty() {
        names.extend((0..spec.csv_features).map(|column| format!("known_future_{column}")));
    }
    names.extend(
        [
            "lag_1",
            "lag_2",
            "lag_3",
            "seasonal_lag_1",
            "seasonal_lag_2",
            "rolling_mean_3",
            "rolling_mean_seasonal",
            "rolling_absdiff_3",
            "rolling_absdiff_seasonal",
            "row_index",
            "horizon_index",
            "seasonal_phase",
            "seasonal_sin",
            "seasonal_cos",
        ]
        .into_iter()
        .map(str::to_string),
    );
    DataSchema {
        columns: names.into_iter().map(ColumnSpec::numeric).collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn timeseries_schema_v2(spec: &DatasetSpec) -> DataSchema {
    DataSchema {
        columns: direct_feature_names_for(&spec.known_future_covariates, spec.csv_features)
            .into_iter()
            .map(ColumnSpec::numeric)
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn canonicalize_for_timeseries(source: &str, spec: &DatasetSpec) -> Result<CanonicalGenome> {
    // v1 keeps the pinned spec.reduction_features count; v2 derives its width
    // from the manifest-level facts so the pinned v1 count stays untouched.
    let (schema, features) = match spec.reduction()? {
        ReductionVersion::V1 => (timeseries_schema(spec), spec.reduction_features),
        ReductionVersion::V2 => {
            let schema = timeseries_schema_v2(spec);
            let features = schema.columns.len();
            (schema, features)
        }
    };
    let canonical = canonicalize_for_features(source, features)?;
    let genome = GenomeIr::parse(&canonical.canonical)?;
    if terminal_op(&genome).is_some_and(is_binary_head) {
        bail!("binary classification head is not valid for timeseries");
    }
    let task = TaskSpec::regression(Objective::Rmse)?;
    validate_v3_genome(&genome, &task, &schema)?;
    Ok(canonical)
}

fn hand_authored_sources() -> Result<Vec<CandidateSource>> {
    let path = PathBuf::from(file!());
    let file_sha256 = sha256_file(&workspace_root().join(file!())).unwrap_or_default();
    let priors = [
        "(pipe X (ridge))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (robust_scale) (ridge))",
        "(pipe X (standardize) (elasticnet alpha=0.001 l1_ratio=0.2))",
        "(pipe X (standardize) (elasticnet alpha=0.01 l1_ratio=0.5))",
        "(pipe X (row_stats append=true) (ridge))",
        "(pipe X (pair_prod append=true topk=8) (ridge))",
        "(pipe X (rff_sin append=true gamma=0.1 n=16 seed=13) (ridge))",
        "(pipe X (rff_cos append=true gamma=0.1 n=16 seed=29) (ridge))",
        "(pipe X (hinge append=true knots=4) (ridge))",
        "(pipe X (winsor_clip append=true q=0.05) (ridge))",
        "(pipe X (quantile_bins append=true bins=8) (ridge))",
        "(pipe X (blom append=true) (ridge))",
        "(pipe X (standardize) (histgbm depth=3 lr=0.05 rounds=80))",
        "(pipe X (robust_scale) (histgbm2 leaves=31 lr=0.04 rounds=300))",
    ];
    Ok(priors
        .iter()
        .enumerate()
        .map(|(record, genome)| CandidateSource {
            genome: (*genome).to_string(),
            kind: "hand_authored_timeseries_prior".to_string(),
            path: path.clone(),
            file_sha256: file_sha256.clone(),
            record,
            rank: record + 1,
            score: None,
            order_score: record as f64,
        })
        .collect())
}

fn binary_rewrite_sources() -> Result<Vec<CandidateSource>> {
    let path = workspace_root().join("seeds/generation-zero-binary.txt");
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let file_sha256 = sha256_bytes(&bytes);
    let text = std::str::from_utf8(&bytes)?;
    let mut sources = Vec::new();
    for (record, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(genome) = rewrite_binary_body_to_regression(line) {
            sources.push(CandidateSource {
                genome,
                kind: "binary_body_to_regression_rewrite".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: None,
                order_score: record as f64,
            });
        }
    }
    Ok(sources)
}

fn sotarena_scan_sources() -> Result<Vec<CandidateSource>> {
    let root = paths::legacy_tree("sotarena_pull").join("out/scans");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path().join("new_candidates.jsonl");
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    let mut sources = Vec::new();
    for path in files {
        let bytes = fs::read(&path)?;
        let file_sha256 = sha256_bytes(&bytes);
        let text = std::str::from_utf8(&bytes)?;
        for (record, line) in text.lines().enumerate() {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let Some(genome) = value.get("canonical_genome").and_then(Value::as_str) else {
                continue;
            };
            let margin = value
                .get("margin_vs_best_comparator")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "sotarena_high_margin_seed".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: Some(margin),
                order_score: -margin,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    sources.truncate(128);
    Ok(sources)
}

fn local_flywheel_elite_sources() -> Result<Vec<CandidateSource>> {
    let roots = [
        paths::flywheel_root().join("receipts"),
        paths::runs_root(),
        paths::legacy_tree("jope-test-adaptive"),
    ];
    let mut receipt_files = Vec::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        let mut stack = vec![(root.to_path_buf(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            if depth > 6 {
                continue;
            }
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push((path, depth + 1));
                } else if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".receipts.jsonl"))
                {
                    receipt_files.push(path);
                }
            }
        }
    }
    receipt_files.sort();
    let mut sources = Vec::new();
    for path in receipt_files {
        let bytes = fs::read(&path)?;
        let file_sha256 = sha256_bytes(&bytes);
        let text = std::str::from_utf8(&bytes)?;
        for (record, line) in text.lines().enumerate() {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if value.get("valid").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let Some(genome) = value.get("canonical_genome").and_then(Value::as_str) else {
                continue;
            };
            let task = value
                .get("task")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let order_score = match task {
                "timeseries" => value
                    .get("secondary_score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::INFINITY),
                "regression" => -value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::NEG_INFINITY),
                "binary" => -value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::NEG_INFINITY),
                _ => continue,
            };
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "local_flywheel_elite".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: order_score.is_finite().then_some(order_score),
                order_score,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    sources.truncate(256);
    Ok(sources)
}

fn cached_sources<F>(
    cell: &'static OnceLock<std::result::Result<Vec<CandidateSource>, String>>,
    build: F,
) -> Result<Vec<CandidateSource>>
where
    F: FnOnce() -> Result<Vec<CandidateSource>>,
{
    match cell.get_or_init(|| build().map_err(|error| format!("{error:#}"))) {
        Ok(sources) => Ok(sources.clone()),
        Err(error) => bail!("{error}"),
    }
}

fn cached_fast30_bank_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, fast30_bank_sources)
}

fn cached_binary_rewrite_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, binary_rewrite_sources)
}

fn cached_sotarena_scan_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, sotarena_scan_sources)
}

fn cached_local_flywheel_elite_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, local_flywheel_elite_sources)
}

fn cached_hand_authored_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, hand_authored_sources)
}

fn admit_seed_sources(
    spec: &DatasetSpec,
    sources: Vec<CandidateSource>,
    seen: &mut BTreeSet<String>,
    seeds: &mut Vec<SeedRecord>,
) {
    for source in sources {
        let canonical = match canonicalize_for_timeseries(&source.genome, spec) {
            Ok(canonical) => canonical,
            Err(_) => continue,
        };
        if !seen.insert(canonical.semantic_sha256.clone()) {
            continue;
        }
        seeds.push(SeedRecord {
            slot: seeds.len(),
            canonical_genome: canonical.canonical,
            semantic_sha256: canonical.semantic_sha256,
            complexity: canonical.complexity,
            rust_source_len: canonical.rust_source_len,
            source_family: source.kind,
            source_file: source.path.to_string_lossy().into_owned(),
            source_file_sha256: source.file_sha256,
            source_record: source.record,
            source_rank: source.rank,
            source_score: source.score,
        });
        if seeds.len() == POPULATION {
            break;
        }
    }
}
