use super::*;

/// Recover raw numeric rows by replaying the recorded historical transform and
/// matching every original training row. Unknown recipes or ambiguous raw rows
/// are withheld; reversing a scaler alone is never treated as reconstruction.
pub fn reconstruct_numeric(train: &Table, m: &Value) -> Result<(Table, Value)> {
    reconstruct_recorded(train, m, false, None)
}

/// Classification reconstruction replays the recorded source-label encoding
/// and class-to-code mapping before verifying every historical training row.
pub fn reconstruct_classification(train: &Table, m: &Value) -> Result<(Table, Value)> {
    check_classification_recipe(m)?;
    reconstruct_recorded(train, m, true, None)
}

pub(super) fn check_classification_recipe(m: &Value) -> Result<()> {
    ensure!(
        m["target_scaling"]["mode"] == "classification",
        "classification needs a recorded label mapping"
    );
    ensure!(
        m["target_scaling"]["scaled_code_formula"] == "code / max(1, n_classes - 1)",
        "unsupported historical classification formula"
    );
    Ok(())
}

/// [`reconstruct_classification`] with the explicit backing allowlist that
/// may relocate a missing recorded source (route R1).
pub(super) fn reconstruct_classification_with(
    train: &Table,
    m: &Value,
    allowlist: &TransferManifest,
) -> Result<(Table, Value)> {
    check_classification_recipe(m)?;
    reconstruct_recorded(train, m, true, Some(allowlist))
}

pub(super) fn historical_class(target: &str, m: &Value) -> Result<f64> {
    let source = &m["target_source_encoding"];
    let value = match source["kind"].as_str() {
        Some("numeric") => target.parse::<f64>().context("invalid original class")?,
        Some("categorical_codes") => source["mapping"][target]
            .as_f64()
            .context("class absent from original source mapping")?,
        _ => anyhow::bail!("unsupported original class encoding"),
    };
    ensure!(value.is_finite(), "nonfinite original class");
    let classes = m["target_scaling"]["class_to_code"]
        .as_object()
        .context("historical class-to-code mapping absent")?;
    let hits = classes
        .iter()
        .filter(|(label, _)| label.parse::<f64>().is_ok_and(|v| v == value))
        .collect::<Vec<_>>();
    ensure!(
        hits.len() == 1,
        "original class mapping is absent or ambiguous"
    );
    let count = m["target_scaling"]["n_classes"]
        .as_u64()
        .context("historical class count absent")?;
    ensure!(
        count >= 2 && classes.len() == count as usize,
        "invalid historical classes"
    );
    let codes = classes
        .values()
        .map(Value::as_u64)
        .collect::<Option<BTreeSet<_>>>()
        .context("invalid historical class codes")?;
    ensure!(
        codes == (0..count).collect(),
        "class codes must be contiguous and unique"
    );
    Ok(hits[0]
        .1
        .as_f64()
        .context("invalid historical class code")?
        / (count - 1) as f64)
}

/// The part of a recorded backing path after its `hyper/data` components.
/// Recorded backing sources predate ADR-0003 and name a retired host tree
/// (`…/hyper/data/<rel>`); `<rel>` is their relative identity.
pub(super) fn hyper_data_relative(path: &Path) -> Option<std::path::PathBuf> {
    let components: Vec<_> = path.components().collect();
    let index = components
        .windows(2)
        .position(|pair| pair[0].as_os_str() == "hyper" && pair[1].as_os_str() == "data")?;
    let relative: std::path::PathBuf = components[index + 2..].iter().collect();
    (!relative.as_os_str().is_empty()).then_some(relative)
}

/// Where a recorded backing source is read (W1 route R1). The recorded path
/// itself when it exists. Otherwise only a location the explicit, digest-bound
/// backing allowlist names as a `source` with the same `hyper/data`-relative
/// identity, and exactly one such location. The process environment (and so
/// `jope_paths::quant_root`) is never consulted.
pub(super) fn backing_source_location(
    recorded: &Path,
    allowlist: Option<&TransferManifest>,
) -> Result<(std::path::PathBuf, Value)> {
    if recorded.is_file() {
        return Ok((recorded.into(), serde_json::json!({"kind":"recorded_path"})));
    }
    let relative = hyper_data_relative(recorded);
    let allowlist =
        allowlist.context("recorded backing source absent; no allowlist names a relocation")?;
    let relative =
        relative.context("recorded backing source absent and has no relocatable identity")?;
    let candidates = allowlist
        .files
        .iter()
        .filter(|f| hyper_data_relative(&f.source).as_deref() == Some(relative.as_path()))
        .map(|f| f.source.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        candidates.len() == 1,
        "recorded backing source absent; the allowlist names {} relocations of {} (exactly one required)",
        candidates.len(),
        relative.display()
    );
    let location = candidates.into_iter().next().unwrap();
    Ok((
        location.clone(),
        serde_json::json!({"kind":"allowlisted_relocation","recorded":recorded,
            "allowlisted_source":location,"relative_identity":relative}),
    ))
}

pub(super) fn reconstruct_recorded(
    train: &Table,
    m: &Value,
    classification: bool,
    allowlist: Option<&TransferManifest>,
) -> Result<(Table, Value)> {
    ensure!(
        classification || m["target_generation"] == "original_numeric_column",
        "target derivation needs its own reconstruction recipe"
    );
    ensure!(
        m["target_fix"]["changed"] != true,
        "changed legacy target requires a verified transformation recipe"
    );
    let recorded = m["source_path"]
        .as_str()
        .context("missing backing-source path")?;
    let (source, source_resolution) = backing_source_location(Path::new(recorded), allowlist)?;
    // Read once: the parsed rows and the recorded source digest describe the
    // same bytes (route R8).
    let source_bytes = fs::read(&source).context("backing source unavailable")?;
    let source_sha256 = hash(&source_bytes);
    let mut reader = csv::Reader::from_reader(source_bytes.as_slice());
    let header = reader.headers()?.clone();
    let names = m["feature_columns"]
        .as_array()
        .context("missing original feature order")?;
    ensure!(names.len() + 1 == train.width, "raw feature order mismatch");
    let positions = names
        .iter()
        .map(|n| {
            header
                .iter()
                .position(|h| Some(h) == n.as_str())
                .context("backing feature missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let target = m["target_column"].as_str().unwrap_or("target");
    let target_index = header
        .iter()
        .position(|h| h == target)
        .or_else(|| {
            m["source_target_column_original"]
                .as_str()
                .and_then(|t| header.iter().position(|h| h == t))
        })
        .context("backing target missing")?;
    let lo = m["feature_scaling"]["min"]
        .as_array()
        .context("unknown feature scaling")?;
    let hi = m["feature_scaling"]["max"]
        .as_array()
        .context("unknown feature scaling")?;
    ensure!(lo.len() == hi.len(), "scaler dimensions mismatch");
    let scale_positions = if lo.len() == positions.len() {
        (0..positions.len()).collect::<Vec<_>>()
    } else {
        // A recorded feature removal does not reorder the original scaler.
        // Match explicit original header names; never guess positions from data.
        let original = m["original_headers"]
            .as_array()
            .context("original scaler headers absent")?;
        ensure!(
            original.len() == lo.len() + 1
                && original.last().and_then(Value::as_str) == Some("target"),
            "original scaler/header dimensions mismatch"
        );
        ensure!(
            original
                .iter()
                .map(Value::to_string)
                .collect::<BTreeSet<_>>()
                .len()
                == original.len(),
            "ambiguous original scaler headers"
        );
        names
            .iter()
            .map(|name| {
                original[..lo.len()]
                    .iter()
                    .position(|v| v == name)
                    .context("feature absent from original scaler headers")
            })
            .collect::<Result<Vec<_>>>()?
    };
    let ylo = if classification {
        0.0
    } else {
        m["target_scaling"]["min"]
            .as_f64()
            .context("unknown target scaling")?
    };
    let yhi = if classification {
        1.0
    } else {
        m["target_scaling"]["max"]
            .as_f64()
            .context("unknown target scaling")?
    };
    ensure!(yhi > ylo, "constant legacy target");
    fn key(row: &[f64]) -> i64 {
        (row[0] * 10000.0).floor() as i64
    }
    type LegacyCandidate = (usize, Vec<String>, Vec<f64>);
    let mut candidates: BTreeMap<i64, Vec<LegacyCandidate>> = BTreeMap::new();
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        let mut raw = Vec::new();
        let mut encoded = Vec::new();
        let target = &record[target_index];
        let y = if classification {
            historical_class(target.trim(), m)?
        } else {
            match target.parse::<f64>() {
                Ok(y) if y.is_finite() => y,
                _ => continue,
            }
        };
        for (j, &pos) in positions.iter().enumerate() {
            let name = names[j].as_str().context("feature name")?;
            let encoding = &m["feature_encodings"][name];
            let value = record[pos].trim();
            raw.push(value.to_string());
            let x = if encoding["kind"] == "numeric" {
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite())
                    .unwrap_or(
                        encoding["impute_value"]
                            .as_f64()
                            .context("missing historical imputation")?,
                    )
            } else if encoding["kind"] == "categorical_codes" {
                let mapping = encoding["mapping"]
                    .as_object()
                    .context("categorical mapping absent")?;
                let key = if missing(value) { "__MISSING__" } else { value };
                mapping
                    .get(key)
                    .and_then(Value::as_f64)
                    .with_context(|| format!("category outside recorded mapping: {name}={key}"))?
            } else {
                anyhow::bail!("unsupported historical feature encoding for {name}")
            };
            let low = lo[scale_positions[j]].as_f64().context("feature minimum")?;
            let high = hi[scale_positions[j]].as_f64().context("feature maximum")?;
            encoded.push(if high > low {
                (x - low) / (high - low)
            } else {
                0.0
            });
        }
        raw.push(target.to_string());
        encoded.push((y - ylo) / (yhi - ylo));
        candidates
            .entry(key(&encoded))
            .or_default()
            .push((i, raw, encoded));
    }
    let mut raw_rows = Vec::new();
    let mut mapping = Vec::new();
    let mut observed_match_error = 0.0f64;
    let mut equivalent_source_matches = 0usize;
    // W1-T06: equal historical rows take distinct equal backing rows in source
    // order. A historical row with no unused equal backing row left repeats a
    // pointer, which every cut membership check refuses as a duplicate join.
    let mut used = BTreeSet::new();
    for row in &train.rows {
        let numeric = row
            .iter()
            .map(|s| s.parse::<f64>().context("non-numeric original"))
            .collect::<Result<Vec<_>>>()?;
        let k = key(&numeric);
        let hits = candidates
            .range(k - 1..=k + 1)
            .flat_map(|(_, rows)| rows)
            .filter(|(_, _, x)| x.iter().zip(&numeric).all(|(a, b)| (a - b).abs() < 1e-7))
            .collect::<Vec<_>>();
        ensure!(!hits.is_empty(), "raw match outside tolerance");
        ensure!(
            hits.iter().all(|r| r.1 == hits[0].1),
            "ambiguous raw-source row mapping"
        );
        observed_match_error = hits[0]
            .2
            .iter()
            .zip(&numeric)
            .fold(observed_match_error, |maximum, (a, b)| {
                maximum.max((a - b).abs())
            });
        equivalent_source_matches += usize::from(hits.len() > 1);
        let hit = hits
            .iter()
            .find(|hit| !used.contains(&hit.0))
            .unwrap_or(&hits[0]);
        used.insert(hit.0);
        mapping.push(hit.0);
        raw_rows.push(hit.1.clone());
    }
    Ok((
        Table {
            rows: raw_rows,
            width: train.width,
            header: false,
        },
        serde_json::json!({"schema":"parallel.raw_reconstruction.v1","source":source,"source_sha256":source_sha256,"source_resolution":source_resolution,"source_row_indices":mapping,"target_column":header.get(target_index),"recipe":if classification {"replay_recorded_feature_encoding_minmax_and_classification_mapping"} else {"replay_recorded_numeric_or_categorical_encoding_named_minmax_and_original_target_minmax"},"classification_mapping":if classification {m["target_scaling"].clone()} else {Value::Null},"historical_scaler_feature_indices":scale_positions,"recipe_metadata_sha256":hash(&serde_json::to_vec(m)?),"max_match_abs_error":observed_match_error,"match_tolerance":1e-7,"equivalent_source_matches":equivalent_source_matches,"training_rows":train.rows.len()}),
    ))
}

pub(super) fn documented_entity_columns(m: &Value) -> Vec<String> {
    match m["source_dataset_id"].as_str().unwrap_or("") {
        "MASS/epil" => vec!["subject".into()],
        "MASS/OME" => vec!["ID".into()],
        "datasets/DNase" => vec!["Run".into()],
        "MASS/nlschools" => vec!["class".into()],
        "plm/Parity" => vec!["country".into()],
        _ => vec![],
    }
}
pub(super) fn entity_documentation(m: &Value) -> Option<String> {
    if documented_entity_columns(m).is_empty() {
        None
    } else {
        Some(format!(
            "https://vincentarelbundock.github.io/Rdatasets/doc/{}.html",
            m["source_dataset_id"].as_str().unwrap()
        ))
    }
}
pub(super) fn reviewed_entity_columns(
    path: &Path,
    expected_sha256: &str,
    dataset: &Dataset,
    metadata: &Value,
    reconstruction: &Value,
) -> Result<(Vec<String>, Value)> {
    ensure!(
        file_hash(path)? == expected_sha256,
        "entity review differs from frozen family registry"
    );
    let review: Value = read_json(path)?;
    ensure!(
        review["schema"] == "parallel.provenance_review.v2",
        "unsupported entity review schema"
    );
    let entries = review["datasets"]
        .as_array()
        .context("entity review datasets absent")?;
    let matching = entries
        .iter()
        .filter(|entry| entry["id"] == dataset.id)
        .collect::<Vec<_>>();
    ensure!(
        matching.len() == 1,
        "entity review must contain exactly one dataset entry"
    );
    let entry = matching[0];
    ensure!(
        entry["source_key"] == crate::audit::source_key(metadata)
            && entry["source_key"] == dataset.source_key
            && entry["metadata"]["sha256"] == dataset.metadata_sha256
            && entry["reconstruction"]["metadata_sha256"] == dataset.metadata_sha256
            && entry["reconstruction"]["source_sha256"] == reconstruction["source_sha256"]
            && entry["reconstruction"]["training_sha256"]
                == file_hash(&dataset.source_dir.join("train.csv"))?
            && entry["reconstruction"]["row_mapping_sha256"]
                == hash(&serde_json::to_vec(&reconstruction["source_row_indices"])?),
        "entity review source, metadata, training rows, or row mapping identity mismatch"
    );
    ensure!(
        entry["pilot_candidate_status"] == "eligible_for_fresh_cohort_preparation"
            && entry["pilot_hold_reason"].is_null(),
        "entity review withholds this dataset from preparation"
    );
    let documentation: Evidence = serde_json::from_value(entry["documentation"].clone())?;
    documentation.verify()?;
    let columns: Vec<String> = serde_json::from_value(entry["entity_columns"].clone())?;
    ensure!(
        columns.iter().all(|c| !c.is_empty())
            && columns.iter().collect::<BTreeSet<_>>().len() == columns.len(),
        "invalid reviewed entity columns"
    );
    let features: Vec<String> = serde_json::from_value(metadata["feature_columns"].clone())?;
    ensure!(
        columns.iter().all(|c| features.contains(c)),
        "reviewed entity columns must be input features"
    );
    Ok((
        columns.clone(),
        serde_json::json!({"schema":"parallel.entity_grouping_binding.v1","manifest_sha256":expected_sha256,"dataset":dataset.id,"source_key":entry["source_key"],"entity_columns":columns,"documentation":documentation,"training_sha256":entry["reconstruction"]["training_sha256"],"row_mapping_sha256":entry["reconstruction"]["row_mapping_sha256"]}),
    ))
}
pub(super) fn entity_values(
    rows: &[Vec<String>],
    features: &[String],
    columns: &[String],
) -> Result<Option<Vec<String>>> {
    if columns.is_empty() {
        return Ok(None);
    }
    let positions = columns
        .iter()
        .map(|name| {
            features
                .iter()
                .position(|f| f == name)
                .with_context(|| format!("documented entity column is unavailable: {name}"))
        })
        .collect::<Result<Vec<_>>>()?;
    rows.iter()
        .map(|row| {
            let values = positions
                .iter()
                .map(|&i| row[i].as_str())
                .collect::<Vec<_>>();
            ensure!(
                values.iter().all(|v| !missing(v)),
                "documented entity identity is missing"
            );
            Ok(serde_json::to_string(&values)?)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
