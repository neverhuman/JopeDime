use super::*;

pub(crate) fn lane_metrics(
    requested: &[Value],
    scores: &BTreeMap<String, (f64, f64)>,
) -> LaneMetrics {
    let mut output = LaneMetrics {
        requested: requested.len(),
        ..LaneMetrics::default()
    };
    let mut seen = BTreeSet::new();
    for semantic in requested.iter().filter_map(Value::as_str) {
        if !seen.insert(semantic) {
            continue;
        }
        if let Some((score, runtime)) = scores.get(semantic) {
            output.valid += 1;
            output.scores.push(*score);
            output.runtimes.push(*runtime);
        }
    }
    output
}

pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Registered shadow-screen lower bound over independent context clusters.
/// Candidate rows within one context are correlated, so the uncertainty unit
/// is the context mean rather than the individual proposal comparison.
pub(crate) fn registered_context_lower_bound(values: &[f64]) -> f64 {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return -2.0;
    }
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (center - 1.96 * (variance / values.len() as f64).sqrt()).clamp(-2.0, 2.0)
}

pub(crate) fn dataset_slices(
    train: &Path,
    task: Task,
) -> Result<(usize, usize, String, String, String, String, String)> {
    let mut rows = 0_usize;
    let mut columns = None;
    let mut missing = 0_usize;
    let mut cells = 0_usize;
    let mut target_counts = BTreeMap::<String, usize>::new();
    for line in BufReader::new(fs::File::open(train)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        ensure!(
            fields.len() >= 2,
            "sealed panel CSV has fewer than two columns"
        );
        match columns {
            Some(expected) => ensure!(fields.len() == expected, "sealed panel CSV width drift"),
            None => columns = Some(fields.len()),
        }
        rows += 1;
        cells += fields.len();
        missing += fields
            .iter()
            .filter(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "na" | "nan" | "null" | "none"
                )
            })
            .count();
        if matches!(task, Task::Binary | Task::Multiclass) {
            *target_counts
                .entry(fields.last().unwrap_or(&"").trim().to_owned())
                .or_default() += 1;
        }
    }
    let columns = columns.context("sealed panel CSV is empty")?;
    ensure!(rows > 0 && cells > 0, "sealed panel CSV is empty");
    let missing_fraction = missing as f64 / cells as f64;
    let size_slice = if rows < 2_000 {
        "small"
    } else if rows <= 20_000 {
        "medium"
    } else {
        "large"
    };
    let missingness_slice = if missing == 0 {
        "none"
    } else if missing_fraction <= 0.05 {
        "low"
    } else {
        "high"
    };
    let (imbalance_slice, target_shape_slice) =
        if matches!(task, Task::Binary | Task::Multiclass) && target_counts.len() >= 2 {
            let minimum = target_counts.values().copied().min().unwrap_or_default();
            let maximum = target_counts.values().copied().max().unwrap_or(1);
            let shape = if minimum as f64 / (maximum as f64) < 0.2 {
                "imbalanced"
            } else {
                "balanced"
            };
            (shape, shape)
        } else {
            ("not_applicable", "unknown")
        };
    Ok((
        rows,
        columns,
        format!("{missing_fraction:.12}"),
        size_slice.to_owned(),
        missingness_slice.to_owned(),
        imbalance_slice.to_owned(),
        target_shape_slice.to_owned(),
    ))
}

pub(crate) fn promotion_context_facts(
    args: &Args,
    pending: &PendingIntent,
    context_limit: usize,
) -> Result<BTreeMap<Sha256Digest, SealedPanelContextFactsV1>> {
    let promotion = pending
        .promotion
        .as_ref()
        .context("promotion facts requested for a canary intent")?;
    ensure!(
        context_limit <= promotion.maximum_contexts,
        "promotion context prefix exceeds its schedule"
    );
    let pool: SealedPanelPoolV2 = serde_json::from_value(read_json(
        &args.root.join("training/sealed-panel-pool.v2.json"),
    )?)?;
    pool.validate()?;
    let consumption: SealedPanelConsumptionV1 =
        serde_json::from_value(read_json(&promotion.panel_consumption)?)?;
    let pool_by_context = pool
        .promotion_panel(promotion.task, &consumption.panel)
        .into_iter()
        .map(|row| (&row.context_sha256, row))
        .collect::<BTreeMap<_, _>>();
    let (pointer, catalog) = catalog_entries(&args.root)?;
    let catalog_sha256 = Sha256Digest::parse(
        pointer
            .get("catalog_sha256")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog digest")?
            .to_owned(),
    )?;
    let facts_root = args.root.join("training/sealed-panel-context-facts");
    let mut output = BTreeMap::new();
    for scheduled in promotion.context_schedule.iter().take(context_limit) {
        let sealed = pool_by_context
            .get(&scheduled.context_sha256)
            .context("scheduled promotion context is absent from the sealed panel")?;
        let path = facts_root.join(format!("{}.v1.json", scheduled.context_sha256));
        if path.is_file() {
            let facts: SealedPanelContextFactsV1 = serde_json::from_value(read_json(&path)?)?;
            facts.validate()?;
            ensure!(
                facts.task == promotion.task
                    && facts.context_sha256 == scheduled.context_sha256
                    && facts.catalog_sha256 == catalog_sha256
                    && facts.context_identity == sealed.context_identity,
                "sealed panel context facts scope drift"
            );
            output.insert(scheduled.context_sha256.clone(), facts);
            continue;
        }
        let entry = catalog
            .get(&(promotion.task, sealed.context_identity.clone()))
            .context("sealed context is absent from the current pinned catalog")?;
        ensure!(
            entry.get("context_sha256").and_then(Value::as_str)
                == Some(scheduled.context_sha256.as_str()),
            "sealed context vector changed in the current catalog"
        );
        let train = PathBuf::from(
            entry
                .get("train_csv")
                .and_then(Value::as_str)
                .context("sealed catalog entry lacks train CSV")?,
        );
        let test = PathBuf::from(
            entry
                .get("test_csv")
                .and_then(Value::as_str)
                .context("sealed catalog entry lacks test CSV")?,
        );
        let train_sha256 = Sha256Digest::parse(sha256_file(&train)?)?;
        let test_sha256 = Sha256Digest::parse(sha256_file(&test)?)?;
        ensure!(
            entry.get("train_sha256").and_then(Value::as_str) == Some(train_sha256.as_str())
                && entry.get("test_sha256").and_then(Value::as_str) == Some(test_sha256.as_str()),
            "sealed dataset content drift"
        );
        let (
            rows,
            columns,
            missing_fraction,
            size_slice,
            missingness_slice,
            imbalance_slice,
            target_shape_slice,
        ) = dataset_slices(&train, promotion.task)?;
        let facts = SealedPanelContextFactsV1 {
            schema: SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA.to_owned(),
            task: promotion.task,
            context_identity: sealed.context_identity.clone(),
            context_sha256: scheduled.context_sha256.clone(),
            catalog_sha256: catalog_sha256.clone(),
            train_sha256,
            test_sha256,
            source_family_sha256: sealed.source_family_sha256.clone(),
            near_duplicate_cluster_sha256: sealed.near_duplicate_cluster_sha256.clone(),
            origin: sealed.origin,
            rows,
            columns,
            missing_fraction,
            size_slice,
            missingness_slice,
            imbalance_slice,
            target_shape_slice,
            created_at_ms: now_ms(),
        };
        facts.validate()?;
        write_immutable(&path, &serde_json::to_vec(&facts)?)?;
        output.insert(scheduled.context_sha256.clone(), facts);
    }
    Ok(output)
}
