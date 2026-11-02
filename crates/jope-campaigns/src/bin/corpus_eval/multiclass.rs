use super::*;

/// Bitwise 0.0/1.0 targets with both classes present; any violation is a
/// dataset-level failure shared by every candidate in the shard.
pub(super) fn validate_binary_targets(y: &[f64], split: &str) -> Option<String> {
    if let Some(value) = y.iter().find(|v| **v != 0.0 && **v != 1.0) {
        return Some(format!(
            "{split} target {value} is not a bitwise 0/1 class label"
        ));
    }
    let positives = y.iter().filter(|v| **v == 1.0).count();
    if positives == 0 || positives == y.len() {
        return Some(format!("{split} split does not contain both classes"));
    }
    None
}

pub(super) fn declared_multiclass_codes(meta: &serde_json::Value) -> Result<Vec<f64>, String> {
    if let Some(mapping) = meta.get("class_mapping").and_then(|value| value.as_array()) {
        let mut entries: Vec<&serde_json::Value> =
            mapping.iter().filter(|entry| entry.is_object()).collect();
        entries.sort_by_key(|entry| {
            entry
                .get("code")
                .and_then(|value| value.as_i64())
                .unwrap_or(i64::MAX)
        });
        if entries.len() >= 3 {
            let explicit: Option<Vec<f64>> = entries
                .iter()
                .map(|entry| {
                    entry
                        .get("scaled_code")
                        .or_else(|| entry.get("csv_value"))
                        .and_then(|value| value.as_f64())
                })
                .collect();
            if let Some(codes) = explicit {
                return Ok(codes);
            }
            let denominator = (entries.len() - 1) as f64;
            return Ok((0..entries.len())
                .map(|index| index as f64 / denominator)
                .collect());
        }
    }
    if let Some(mapping) = meta
        .get("class_mapping")
        .and_then(|value| value.as_object())
    {
        if mapping.len() >= 3 {
            let mut ordered: Vec<i64> = mapping
                .values()
                .map(|value| {
                    value
                        .as_i64()
                        .ok_or_else(|| "class_mapping indices must be integers".to_string())
                })
                .collect::<Result<_, _>>()?;
            ordered.sort_unstable();
            ordered.dedup();
            if ordered.len() != mapping.len() {
                return Err("class_mapping indices must be unique".to_string());
            }
            let denominator = (ordered.len() - 1) as f64;
            return Ok((0..ordered.len())
                .map(|index| index as f64 / denominator)
                .collect());
        }
    }
    if let Some(classes) = meta
        .get("target_scaling")
        .and_then(|value| value.get("classes"))
        .and_then(|value| value.as_array())
    {
        if classes.len() >= 3 {
            let denominator = (classes.len() - 1) as f64;
            return Ok((0..classes.len())
                .map(|index| index as f64 / denominator)
                .collect());
        }
    }
    Err("meta.json must declare at least three ordered multiclass classes".to_string())
}

pub(super) fn decode_declared_targets(
    values: &[f64],
    declared_codes: &[f64],
    split: &str,
) -> Result<Vec<i64>, String> {
    const TOLERANCE: f64 = 1e-6;
    values
        .iter()
        .map(|raw| {
            if !raw.is_finite() {
                return Err(format!("{split} target is nonfinite"));
            }
            let matches: Vec<usize> = declared_codes
                .iter()
                .enumerate()
                .filter_map(|(index, code)| ((*raw - *code).abs() <= TOLERANCE).then_some(index))
                .collect();
            if matches.len() != 1 {
                return Err(format!(
                    "{split} target {raw} is ambiguous or outside the declared-code tolerance"
                ));
            }
            Ok(matches[0] as i64)
        })
        .collect()
}

pub(super) fn decode_multiclass_splits(
    train: &[f64],
    test: &[f64],
    meta: &serde_json::Value,
) -> Result<DecodedMulticlassSplits, String> {
    let declared_codes = declared_multiclass_codes(meta)?;
    if declared_codes.iter().any(|code| !code.is_finite()) {
        return Err("declared multiclass codes must be finite".to_string());
    }
    for (index, code) in declared_codes.iter().enumerate() {
        if declared_codes[..index]
            .iter()
            .any(|prior| (*prior - *code).abs() <= f64::EPSILON)
        {
            return Err("declared multiclass codes must be unique".to_string());
        }
    }
    let decoded_train = decode_declared_targets(train, &declared_codes, "train")?;
    let decoded_test = decode_declared_targets(test, &declared_codes, "test")?;
    let observed_declared: Vec<i64> = decoded_train
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if observed_declared.len() < 2 {
        return Err(
            "multiclass training split must contain at least two declared classes".to_string(),
        );
    }
    let fitted_train = decoded_train
        .iter()
        .map(|declared| {
            observed_declared
                .binary_search(declared)
                .expect("observed class came from training targets") as f64
        })
        .collect();
    Ok(DecodedMulticlassSplits {
        train: fitted_train,
        test: decoded_test.into_iter().map(|value| value as f64).collect(),
        context: MulticlassContext {
            observed_declared,
            declared_classes: declared_codes.len(),
        },
    })
}

pub(super) fn declared_macro_f1(y_true: &[f64], y_pred: &[f64], declared_classes: usize) -> f64 {
    let mut total = 0.0;
    for class in 0..declared_classes {
        let label = class as f64;
        let true_positive = y_true
            .iter()
            .zip(y_pred)
            .filter(|(truth, prediction)| **truth == label && **prediction == label)
            .count() as f64;
        let false_positive = y_true
            .iter()
            .zip(y_pred)
            .filter(|(truth, prediction)| **truth != label && **prediction == label)
            .count() as f64;
        let false_negative = y_true
            .iter()
            .zip(y_pred)
            .filter(|(truth, prediction)| **truth == label && **prediction != label)
            .count() as f64;
        let denominator = 2.0 * true_positive + false_positive + false_negative;
        if denominator > 0.0 {
            total += 2.0 * true_positive / denominator;
        }
    }
    total / declared_classes as f64
}

// ---------------------------------------------------------------------------
// rewrite-binary-heads mode
// ---------------------------------------------------------------------------

pub(super) fn rewrite_head(
    canonical: &str,
    head: &str,
) -> std::result::Result<(String, String), String> {
    let genome = GenomeIr::parse(canonical).map_err(|e| format!("parse: {e:#}"))?;
    let GenomeIr::Pipeline(mut dsl) = genome else {
        return Err("not a pipeline genome".to_string());
    };
    dsl.stages.retain(|stage| !is_target_op(&stage.op));
    let source_head = match dsl.stages.last() {
        Some(stage) if is_head_like(&stage.op) => dsl.stages.pop().expect("last stage exists").op,
        _ => "ridge-implicit".to_string(),
    };
    dsl.stages.push(Stage::new(head));
    let rewritten = GenomeIr::Pipeline(dsl).canonical();
    let reparsed = GenomeIr::parse(&rewritten).map_err(|e| format!("reparse: {e:#}"))?;
    if reparsed.canonical() != rewritten {
        return Err("rewritten genome is not canonical-stable".to_string());
    }
    // Width-independent admission check (validate_v3_genome inspects no
    // column counts, so a nominal 8-column all-numeric schema is exact).
    let task = TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
    validate_v3_genome(&reparsed, &task, &binary_schema(8))
        .map_err(|e| format!("admission: {e:#}"))?;
    Ok((rewritten, source_head))
}

pub(super) fn run_rewrite(winners_dir: &Path, heads: &[String], output: &Path) -> Result<()> {
    let mut dataset_dirs = fs::read_dir(winners_dir)
        .with_context(|| format!("read winners dir {}", winners_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join("final-result.v1.json").is_file())
        .collect::<Vec<_>>();
    dataset_dirs.sort();
    if dataset_dirs.is_empty() {
        bail!(
            "no <dataset>/final-result.v1.json winners under {}",
            winners_dir.display()
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut lines = Vec::new();
    let (mut ok, mut rejected) = (0usize, 0usize);
    for dir in &dataset_dirs {
        let dataset = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final-result.v1.json"))?)?;
        let Some(winner) = value.get("canonical_genome").and_then(|v| v.as_str()) else {
            eprintln!("warning: {dataset} final-result has no canonical_genome; skipped");
            continue;
        };
        for head in heads {
            let (validation, genome, source_head) = match rewrite_head(winner, head) {
                Ok((genome, source_head)) => ("ok".to_string(), genome, source_head),
                Err(why) => {
                    rejected += 1;
                    lines.push(serde_json::json!({
                        "canonical_genome": serde_json::Value::Null,
                        "semantic_sha256": serde_json::Value::Null,
                        "source_winner_dataset": dataset,
                        "source_head": serde_json::Value::Null,
                        "binary_head": head,
                        "head_rewritten": true,
                        "validation": format!("rejected:{why}"),
                    }));
                    continue;
                }
            };
            if !seen.insert(genome.clone()) {
                continue; // deduplicate identical rewrites across winners
            }
            ok += 1;
            lines.push(serde_json::json!({
                "canonical_genome": genome,
                "semantic_sha256": sha256_hex(genome.as_bytes()),
                "source_winner_dataset": dataset,
                "source_head": source_head,
                "binary_head": head,
                "head_rewritten": true,
                "validation": validation,
            }));
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buffer = Vec::new();
    for line in &lines {
        serde_json::to_writer(&mut buffer, line)?;
        buffer.push(b'\n');
    }
    fs::write(output, buffer)?;
    eprintln!(
        "rewrote {} winners x {} heads -> {} unique ok, {} rejected ({})",
        dataset_dirs.len(),
        heads.len(),
        ok,
        rejected,
        output.display()
    );
    Ok(())
}
