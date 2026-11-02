use super::*;

pub(super) struct NativePreprocessing {
    schema: String,
    columns: Vec<NativeColumn>,
    labels: Vec<String>,
    target_mean: f64,
    target_scale: f64,
}
#[derive(Deserialize)]
pub(super) struct NativeColumn {
    kind: String,
    impute: f64,
    mean: f64,
    scale: f64,
    categories: Vec<String>,
}
pub(super) fn load_training_rows(
    sources: &ReferenceSources,
    meta: &Value,
    train_indices: &[usize],
    pp: &NativePreprocessing,
    custody: &mut Vec<Artifact>,
) -> Result<Vec<Vec<f64>>> {
    ensure!(
        pp.schema == "parallel.preprocessing.v1"
            && pp.labels.is_empty()
            && pp.target_mean.is_finite()
            && pp.target_scale.is_finite()
            && pp.target_scale > 0.0,
        "native regression preprocessing invalid"
    );
    let storage = &meta["storage"];
    ensure!(
        storage.is_object(),
        "reference preparation requires explicit original row mapping"
    );
    let file = Artifact {
        path: resolve(&sources.source_root, string(storage, "source")?)?,
        sha256: string(storage, "source_sha256")?.into(),
    };
    // Hash the full source, then stream only selected records into numerical input.
    context_producer::worker::verify_file(&policy_artifact(&file), 2u64 << 30)?;
    let map = indices(&storage["original_training_row_map"])?;
    let selected = train_indices
        .iter()
        .map(|&i| {
            map.get(i)
                .copied()
                .context("native row mapping out of range")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        selected.iter().collect::<BTreeSet<_>>().len() == selected.len(),
        "duplicate original source fitting rows"
    );
    let wanted = selected.iter().copied().collect::<BTreeSet<_>>();
    let mut reader = csv::Reader::from_path(&file.path)?;
    let headers = reader.headers()?.clone();
    let feature_names = storage["feature_columns"]
        .as_array()
        .context("native features absent")?
        .iter()
        .map(|v| v.as_str().context("invalid feature name"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        feature_names.len() == pp.columns.len()
            && feature_names.iter().collect::<BTreeSet<_>>().len() == feature_names.len(),
        "native feature schema width/duplicates"
    );
    let position = |name: &str| {
        let found = headers
            .iter()
            .enumerate()
            .filter(|(_, h)| *h == name)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ensure!(found.len() == 1, "source column absent/duplicate: {name}");
        Ok(found[0])
    };
    let features = feature_names
        .iter()
        .map(|name| position(name))
        .collect::<Result<Vec<_>>>()?;
    let target = position(string(storage, "target_column")?)?;
    let mut fitted = BTreeMap::new();
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        if !wanted.contains(&i) {
            continue;
        }
        let mut row = Vec::new();
        for (&column, p) in features.iter().zip(&pp.columns) {
            let value = record.get(column).context("short source record")?.trim();
            let encoded = match p.kind.as_str() {
                "numeric" => {
                    ensure!(
                        p.impute.is_finite()
                            && p.mean.is_finite()
                            && p.scale.is_finite()
                            && p.scale > 0.0,
                        "invalid native numeric preprocessor"
                    );
                    (value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .unwrap_or(p.impute)
                        - p.mean)
                        / p.scale
                }
                "categorical" => {
                    ensure!(
                        p.categories.windows(2).all(|w| w[0] < w[1]),
                        "native categories are not sorted unique"
                    );
                    p.categories
                        .binary_search_by(|c| c.as_str().cmp(value))
                        .map(|i| i as f64)
                        .unwrap_or(-1.0)
                }
                _ => anyhow::bail!("unsupported native preprocessing kind"),
            };
            row.push(encoded);
        }
        let y = record
            .get(target)
            .context("short target record")?
            .trim()
            .parse::<f64>()?;
        row.push((y - pp.target_mean) / pp.target_scale);
        ensure!(
            row.iter().all(|v| v.is_finite()),
            "nonfinite native prepared row"
        );
        fitted.insert(i, row);
    }
    // Detect source mutation during selection without parsing unselected targets.
    context_producer::worker::verify_file(&policy_artifact(&file), 2u64 << 30)?;
    custody.push(file);
    selected
        .iter()
        .map(|i| {
            fitted
                .get(i)
                .cloned()
                .context("selected original training row missing")
        })
        .collect()
}
pub(super) fn numeric_csv(rows: &[Vec<f64>]) -> Vec<u8> {
    rows.iter()
        .map(|r| {
            let mut s = r
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            s.push('\n');
            s
        })
        .collect::<String>()
        .into_bytes()
}
pub(super) fn indices(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .context("row index list missing")?
        .iter()
        .map(|v| {
            usize::try_from(v.as_u64().context("invalid row index")?).context("row index overflow")
        })
        .collect()
}
pub(super) fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing metadata string {key}"))
}
pub(super) fn resolve(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = Path::new(relative);
    ensure!(
        !relative.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "custody reference path escapes root"
    );
    let p = root.join(p);
    ensure!(p.canonicalize()? == p, "custody reference must be physical");
    Ok(p)
}
