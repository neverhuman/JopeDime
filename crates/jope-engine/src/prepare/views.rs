use super::*;

/// A view has no dataset payload: it points to one immutable backing table.
/// Exact row mappings preserve split membership even if generation algorithms change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewSource {
    pub source: std::path::PathBuf,
    pub source_sha256: String,
    pub feature_columns: Vec<String>,
    pub target_column: String,
    pub original_training_row_map: Vec<usize>,
}

/// Receiving-side column semantics of a pointer view (W1-T03). Feature names
/// must be distinct and non-empty, and the target column may never also be a
/// feature: that would be an undocumented target-derived feature.
pub fn validate_view_source(source: &ViewSource) -> Result<()> {
    let mut names = BTreeSet::new();
    ensure!(
        !source.feature_columns.is_empty()
            && source
                .feature_columns
                .iter()
                .all(|name| !name.trim().is_empty() && names.insert(name.as_str())),
        "view feature columns are empty or duplicated"
    );
    ensure!(
        !source.target_column.trim().is_empty(),
        "view target column absent"
    );
    ensure!(
        !names.contains(source.target_column.as_str()),
        "target column {} is also declared as a feature: undocumented target-derived feature",
        source.target_column
    );
    Ok(())
}

/// A feature whose value equals the target on every fitting row is an
/// undocumented target alias. Rows are raw `[features.., target]` strings.
/// Checked only when the fitting target has at least three distinct values,
/// so constant or two-valued targets cannot trigger a coincidental match.
pub fn reject_target_copies(feature_columns: &[String], rows: &[Vec<String>]) -> Result<()> {
    let width = feature_columns.len() + 1;
    ensure!(
        rows.iter().all(|row| row.len() == width),
        "raw view row width differs from its feature columns"
    );
    let distinct = rows
        .iter()
        .map(|row| row[width - 1].as_str())
        .collect::<BTreeSet<_>>();
    if distinct.len() < 3 {
        return Ok(());
    }
    let same = |value: &str, target: &str| {
        value == target
            || matches!(
                (value.parse::<f64>(), target.parse::<f64>()),
                (Ok(a), Ok(b)) if a == b
            )
    };
    for (j, name) in feature_columns.iter().enumerate() {
        ensure!(
            !rows.iter().all(|row| same(&row[j], &row[width - 1])),
            "feature {name} reproduces the target on every fitting row: undocumented target-derived feature"
        );
    }
    Ok(())
}

/// Cut membership lists each original training row once, in original order,
/// with validation disjoint from training both by row index and by backing-row
/// pointer. Duplicate-expanding, reordered or overlapping memberships are
/// refused before any preparation state is fitted or applied.
pub fn check_cut_membership(
    row_map: &[usize],
    train: &[usize],
    validation: &[usize],
) -> Result<()> {
    for (name, indices) in [("training", train), ("validation", validation)] {
        ensure!(!indices.is_empty(), "empty {name} membership");
        ensure!(
            indices.windows(2).all(|pair| pair[0] < pair[1]),
            "{name} rows are duplicated or reordered; a cut lists each original row once in original order"
        );
    }
    let fitting = train.iter().collect::<BTreeSet<_>>();
    ensure!(
        validation.iter().all(|i| !fitting.contains(i)),
        "validation rows overlap training rows"
    );
    let pointer = |i: &usize| {
        row_map
            .get(*i)
            .copied()
            .context("source row pointer outside mapping")
    };
    // W1-T06: two rows of one membership that resolve to one backing row are a
    // duplicate-expanding join, however their indices differ.
    for (name, indices) in [("training", train), ("validation", validation)] {
        let mut seen = BTreeSet::new();
        for i in indices {
            ensure!(
                seen.insert(pointer(i)?),
                "duplicate-expanding join: two {name} rows resolve to one backing row"
            );
        }
    }
    let backing = train.iter().map(pointer).collect::<Result<BTreeSet<_>>>()?;
    for i in validation {
        ensure!(
            !backing.contains(&pointer(i)?),
            "validation and training rows resolve to the same backing row"
        );
    }
    Ok(())
}

/// Learned preparation state is bound to fitting membership: the stored state
/// must equal the state fitted on exactly the cut's training rows. Recoded or
/// reordered categories, changed scaling/imputation or state fitted on held
/// rows are refused, never applied.
pub fn ensure_fitted_on_membership(
    stored: &Preprocessing,
    training: &[Vec<String>],
    task: &str,
) -> Result<()> {
    let refit = Preprocessing::fit(training, task)?;
    ensure!(
        serde_json::to_value(&refit)? == serde_json::to_value(stored)?,
        "cut preprocessing is not the state learned from its training membership \
         (recoded categories, changed scaling/imputation or non-training rows)"
    );
    Ok(())
}
pub fn workspace_root(prepared: &Path) -> Result<std::path::PathBuf> {
    let p = prepared.canonicalize()?;
    p.ancestors()
        .find(|p| p.join("manifests").is_dir() && p.join("Cargo.toml").is_file())
        .map(Path::to_path_buf)
        .context("prepared view must be inside the parallel workspace")
}
pub fn load_view(prepared: &Path, cut: usize, training: bool) -> Result<Vec<Vec<f64>>> {
    load_view_with_root(None, prepared, cut, training)
}
/// Resolve pointer payloads against the explicitly authenticated data root.
/// Relocated historical metadata keeps its original relative source identities.
pub fn load_view_in_root(
    root: &Path,
    prepared: &Path,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    load_view_with_root(Some(root), prepared, cut, training)
}
/// [`load_view_in_root`] for a preparation the caller has already verified
/// and parsed (`p`): `prepared.json` is not reopened, so the view is resolved
/// from exactly the verified preparation (W1 route R8).
pub fn load_bound_view_in_root(
    root: &Path,
    prepared: &Path,
    p: &Prepared,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    view_of(Some(root), prepared, p, cut, training)
}
/// [`load_bound_view_in_root`] without any root: only a contained view, whose
/// rows are the cut's own tables, is resolved. A pointer view names its
/// backing table relative to a data root, which must be declared; it is never
/// discovered by walking the view's ancestors (W1 route R2).
pub fn load_bound_contained_view(
    prepared: &Path,
    p: &Prepared,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    ensure!(
        p.storage.is_none(),
        "pointer view needs its declared data root: its backing table is never \
         resolved from the view's ancestors"
    );
    view_of(None, prepared, p, cut, training)
}
fn load_view_with_root(
    root: Option<&Path>,
    prepared: &Path,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    view_of(root, prepared, &p, cut, training)
}
fn view_of(
    root: Option<&Path>,
    prepared: &Path,
    p: &Prepared,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown view")?;
    let dir = prepared.join(format!("cut-{cut}"));
    let pp_path = dir.join("preprocessing.json");
    // The verified preprocessing bytes are the ones applied (route R8).
    let pp_bytes = read_verified(&pp_path, &c.preprocessing_sha256, "view preprocessing")?;
    let rows = if let Some(source) = &p.storage {
        // Semantics are checked on the fitting membership for both partitions,
        // before any preparation state is applied or any row is returned.
        check_cut_membership(
            &source.original_training_row_map,
            &c.train_indices,
            &c.validation_indices,
        )?;
        let mut indices = c.train_indices.clone();
        if !training {
            indices.extend_from_slice(&c.validation_indices);
        }
        let raw = match root {
            Some(root) => load_raw_view_in_root(root, source, &indices)?,
            None => load_raw_view(prepared, source, &indices)?,
        };
        let (fitting, held) = raw.split_at(c.train_indices.len());
        reject_target_copies(&source.feature_columns, fitting)?;
        let pp: Preprocessing = serde_json::from_slice(&pp_bytes)
            .with_context(|| format!("parse {}", pp_path.display()))?;
        ensure_fitted_on_membership(&pp, fitting, &p.task)?;
        pp.transform(if training { fitting } else { held })?
    } else {
        Table::numeric(&dir.join(if training {
            "train.csv"
        } else {
            "validation.csv"
        }))?
    };
    let expected = if training {
        &c.training_sha256
    } else {
        &c.validation_sha256
    };
    ensure!(
        hash(&csv_bytes(&rows)?) == *expected,
        "view content hash does not match frozen cut"
    );
    Ok(rows)
}
pub fn load_raw_view(
    prepared: &Path,
    source: &ViewSource,
    indices: &[usize],
) -> Result<Vec<Vec<String>>> {
    load_raw_view_in_root(&workspace_root(prepared)?, source, indices)
}
pub fn load_raw_view_in_root(
    root: &Path,
    source: &ViewSource,
    indices: &[usize],
) -> Result<Vec<Vec<String>>> {
    validate_view_source(source)?;
    let path = resolve(root, &source.source)?;
    // Parse exactly the verified bytes; the path is never reopened (route R8).
    let bytes = read_verified(&path, &source.source_sha256, "backing table")?;
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    let header = reader.headers()?.clone();
    // A repeated header name makes the column mapping ambiguous: refuse it
    // rather than silently binding the first occurrence.
    let column = |name: &str, missing: &'static str| -> Result<usize> {
        let mut hits = header.iter().enumerate().filter(|(_, h)| *h == name);
        let (index, _) = hits.next().context(missing)?;
        ensure!(
            hits.next().is_none(),
            "backing header names {name} more than once: ambiguous column"
        );
        Ok(index)
    };
    let features = source
        .feature_columns
        .iter()
        .map(|name| column(name, "backing feature missing"))
        .collect::<Result<Vec<_>>>()?;
    let target = column(&source.target_column, "backing target missing")?;
    let selected = indices
        .iter()
        .map(|&i| {
            source
                .original_training_row_map
                .get(i)
                .copied()
                .context("source row pointer outside mapping")
        })
        .collect::<Result<Vec<_>>>()?;
    let wanted = selected.iter().copied().collect::<BTreeSet<_>>();
    let mut raw = BTreeMap::new();
    for (i, row) in reader.records().enumerate() {
        let row = row?;
        if wanted.contains(&i) {
            let mut values = features
                .iter()
                .map(|&j| row[j].trim().to_string())
                .collect::<Vec<_>>();
            values.push(row[target].trim().to_string());
            raw.insert(i, values);
        }
    }
    selected
        .iter()
        .map(|i| raw.get(i).cloned().context("backing row missing"))
        .collect::<Result<Vec<_>>>()
}
