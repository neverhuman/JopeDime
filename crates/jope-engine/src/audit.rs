use crate::{manifest::*, prepare::Table};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use walkdir::{DirEntry, WalkDir};

fn text<'a>(m: &'a Value, key: &str) -> &'a str {
    m.get(key).and_then(Value::as_str).unwrap_or("")
}
fn synthetic(m: &Value, dir: &Path) -> bool {
    m.get("synthetic_origin") == Some(&Value::Bool(true))
        || m.get("source_real_data") == Some(&Value::Bool(false))
        || ["synthetic", "generated", "simulation"].contains(&text(m, "origin"))
        || dir.components().any(|x| {
            x.as_os_str().to_string_lossy().starts_with("synth_")
                || x.as_os_str().to_string_lossy().starts_with("synthetic")
        })
}
fn include(entry: &DirEntry) -> bool {
    ![
        ".git",
        ".venv",
        "venv",
        "environments",
        "__pycache__",
        "target",
        "node_modules",
        ".ssh",
    ]
    .contains(&entry.file_name().to_string_lossy().as_ref())
}
pub fn source_key(m: &Value) -> String {
    if !text(m, "source_dataset_id").is_empty() {
        return format!(
            "{}:{}",
            text(m, "source_platform"),
            text(m, "source_dataset_id")
        );
    }
    if let Some(id) = m.get("source_openml_data_id") {
        return format!("openml:{}", id.to_string().trim_matches('"'));
    }
    if !text(m, "source_dataset_config").is_empty() {
        return format!(
            "{}:{}",
            text(m, "source_dataset"),
            text(m, "source_dataset_config")
        );
    }
    text(m, "source_file").to_string()
}
fn stats(table: &Table) -> (TableStats, BTreeSet<String>) {
    let mut row_set = BTreeSet::new();
    let mut features = BTreeMap::new();
    let mut labels = BTreeSet::new();
    let mut missing = 0;
    let mut invalid = 0;
    let mut duplicate = 0;
    let mut conflicts = 0;
    for row in &table.rows {
        let y = row.last().unwrap();
        labels.insert(y.clone());
        if crate::prepare::missing(y) || y.parse::<f64>().is_ok_and(|x| !x.is_finite()) {
            invalid += 1;
        }
        missing += row[..row.len() - 1]
            .iter()
            .filter(|x| crate::prepare::missing(x))
            .count();
        let x = hash(&serde_json::to_vec(&row[..row.len() - 1]).unwrap());
        if let Some(old) = features.insert(x, y.clone()) {
            if old != *y {
                conflicts += 1;
            }
        }
        if !row_set.insert(hash(&serde_json::to_vec(row).unwrap())) {
            duplicate += 1;
        }
    }
    (
        TableStats {
            rows: table.rows.len(),
            columns: table.width,
            target_index: table.width - 1,
            header: table.header,
            missing_features: missing,
            invalid_targets: invalid,
            duplicate_rows: duplicate,
            conflicting_features: conflicts,
            target_values: if labels.len() <= 256 {
                labels.into_iter().collect()
            } else {
                Vec::new()
            },
        },
        features.into_keys().collect(),
    )
}
/// Label-derived statistics the audit never measures on the sealed split.
const WITHHELD_SEALED_STATISTICS: [&str; 4] = [
    "invalid_targets",
    "duplicate_rows",
    "conflicting_features",
    "target_values",
];

/// How the audit read each dataset's sealed final-test split (W1-T04). It is
/// published beside the report so a zero in a withheld field is never read as
/// a measurement.
fn sealed_test_access() -> Value {
    json!({"schema":"parallel.audit_sealed_test_access.v1","split":"test.csv",
        "columns_parsed":"features_only","labels_parsed":false,
        "withheld_label_statistics":WITHHELD_SEALED_STATISTICS,
        "withheld_value":"0 or [] without measurement",
        "reason":"sealed final-test labels are released only to an authorized outer scorer"})
}

/// Feature-only projection of the sealed final-test split. The final (target)
/// cell of every record is dropped as it is parsed: sealed labels are never
/// decoded, retained, summarized or written, so no audit output can depend on
/// them. Width, row count, header detection and missing features — what the
/// preparation contract consumes — and the feature identities used for
/// original-split overlap are reported; label-derived statistics are withheld.
fn sealed_feature_stats(path: &Path, m: &Value) -> Result<(TableStats, BTreeSet<String>)> {
    let bytes = fs::read(path)?;
    let mut records = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes.as_slice());
    let names = m
        .get("output_headers")
        .or_else(|| m.get("original_headers"))
        .and_then(Value::as_array);
    let mut width = None;
    let mut rows = Vec::new();
    // `Table::read`'s header rule needs the first record's final cell compared
    // with the declared target *name*; that one equality is the only use of a
    // final cell, and its value is not retained.
    let mut first_is_header = false;
    for record in records.records() {
        let record = record?;
        let width = *width.get_or_insert(record.len());
        anyhow::ensure!(width >= 2, "expected features and a target");
        if rows.is_empty() {
            first_is_header = names.is_some_and(|h| {
                h.len() == width
                    && h.iter()
                        .zip(record.iter())
                        .all(|(a, b)| a.as_str() == Some(b.trim()))
            });
        }
        rows.push(
            record
                .iter()
                .take(width - 1)
                .map(|cell| cell.trim().to_string())
                .collect::<Vec<_>>(),
        );
    }
    anyhow::ensure!(!rows.is_empty(), "empty table");
    let width = width.unwrap_or(0);
    let header = first_is_header || m.get("headers_removed_from_csv") == Some(&Value::Bool(false));
    if header {
        rows.remove(0);
    }
    anyhow::ensure!(!rows.is_empty(), "no data rows");
    if let Some(n) = m.get("n_features").and_then(Value::as_u64) {
        anyhow::ensure!(n as usize + 1 == width, "feature count mismatch");
    }
    anyhow::ensure!(
        m.get("target_is_final_column") != Some(&Value::Bool(false)),
        "target must be explicitly relocated to final column"
    );
    let features = rows
        .iter()
        .map(|row| hash(&serde_json::to_vec(row).unwrap()))
        .collect();
    let missing = rows
        .iter()
        .flatten()
        .filter(|cell| crate::prepare::missing(cell))
        .count();
    Ok((
        TableStats {
            rows: rows.len(),
            columns: width,
            target_index: width - 1,
            header,
            missing_features: missing,
            invalid_targets: 0,
            duplicate_rows: 0,
            conflicting_features: 0,
            target_values: Vec::new(),
        },
        features,
    ))
}
fn inspect(dir: &Path, m: &Value) -> (Option<TableStats>, Option<TableStats>, usize, Vec<String>) {
    let mut issues = Vec::new();
    let mut tables = Vec::new();
    match Table::read(&dir.join("train.csv"), m) {
        Ok(t) => tables.push(Some(stats(&t))),
        Err(e) => {
            issues.push(format!("train.csv: {e:#}"));
            tables.push(None);
        }
    }
    match sealed_feature_stats(&dir.join("test.csv"), m) {
        Ok(t) => tables.push(Some(t)),
        Err(e) => {
            issues.push(format!("test.csv: {e:#}"));
            tables.push(None);
        }
    }
    let overlap = match (&tables[0], &tables[1]) {
        (Some(a), Some(b)) => a.1.intersection(&b.1).count(),
        _ => 0,
    };
    if overlap > 0 {
        issues.push(format!(
            "{overlap} feature-identical rows overlap original splits"
        ));
    }
    if m.get("feature_scaling").is_some() || m.get("feature_encodings").is_some() {
        issues.push(
            "historical preprocessing requires backing-source reconstruction before clean use"
                .into(),
        );
    }
    (
        tables[0].take().map(|x| x.0),
        tables[1].take().map(|x| x.0),
        overlap,
        issues,
    )
}

pub fn run(roots: &[PathBuf], decisions_path: &Path, output: &Path) -> Result<AuditReport> {
    fs::create_dir_all(output)?;
    let decisions: Vec<Decision> = read_json(decisions_path)?;
    let mut reviews = BTreeMap::new();
    for d in decisions {
        if d.reason.is_empty()
            || (d.admission == Admission::Approved
                && (d.family.is_empty() || d.evidence.is_empty()))
        {
            bail!("incomplete review for {}", d.key);
        }
        for e in &d.evidence {
            e.verify()?;
        }
        if reviews.insert(d.key.clone(), d).is_some() {
            bail!("duplicate provenance decision");
        }
    }
    let roots: Vec<_> = roots
        .iter()
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
        .collect();
    for (i, a) in roots.iter().enumerate() {
        for b in &roots[i + 1..] {
            if a.starts_with(b) || b.starts_with(a) {
                bail!("audit roots overlap");
            }
        }
    }
    let missing_roots = roots
        .iter()
        .filter(|p| !p.is_dir())
        .cloned()
        .collect::<Vec<_>>();
    let mut datasets = Vec::new();
    let mut dirs = HashMap::new();
    let mut discovery_errors = Vec::new();
    for root in &roots {
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(include)
        {
            let e = match entry {
                Ok(e) => e,
                Err(e) => {
                    discovery_errors.push(e.to_string());
                    continue;
                }
            };
            if !e.file_type().is_file() || e.file_name() != "meta.json" {
                continue;
            }
            let m: Value = match read_json(e.path()) {
                Ok(m) => m,
                Err(e) => {
                    discovery_errors.push(e.to_string());
                    continue;
                }
            };
            let dir = e.path().parent().unwrap();
            let id = text(&m, "dataset_hash");
            if id.is_empty() {
                continue;
            }
            let task = match text(&m, "task_type") {
                "multi_label" => "multiclass",
                t => t,
            }
            .to_string();
            let key = source_key(&m);
            let parents = ["derived_from", "parent_dataset", "parent"]
                .iter()
                .filter_map(|k| m.get(k).and_then(Value::as_str))
                .map(String::from)
                .collect();
            let review = reviews
                .get(&format!("dataset:{id}"))
                .or_else(|| reviews.get(&key));
            let (admission, reason, family, evidence) = if synthetic(&m, dir) {
                (
                    Admission::Excluded,
                    "explicit generator/synthetic origin".into(),
                    String::new(),
                    Vec::new(),
                )
            } else if let Some(d) = review {
                (
                    d.admission,
                    d.reason.clone(),
                    d.family.clone(),
                    d.evidence.clone(),
                )
            } else {
                (Admission::Unresolved,"observational origin needs source evidence; legacy real flags are insufficient".into(),text(&m,"source_family").into(),Vec::new())
            };
            let (train, test, overlap, issues) = (None, None, 0, vec![]);
            dirs.insert(dir.to_path_buf(), datasets.len());
            if datasets.len() % 1000 == 0 {
                write_json(
                    &output.join("progress.json"),
                    &json!({"phase":"metadata inventory","datasets":datasets.len(),"timestamp":crate::telemetry::now()}),
                )?;
            }
            datasets.push(Dataset {
                id: id.into(),
                task,
                source_dir: dir.into(),
                source_key: key,
                family,
                parents,
                admission,
                reason,
                evidence,
                metadata_sha256: file_hash(e.path())?,
                train,
                test,
                split_feature_overlap: overlap,
                issues,
                meta_partition: None,
            });
        }
    }
    write_json(&output.join("datasets-discovered.json"), &datasets)?;
    write_json(
        &output.join("progress.json"),
        &json!({"phase":"table statistics","datasets":datasets.len(),"timestamp":crate::telemetry::now()}),
    )?;
    use rayon::prelude::*;
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()?
        .install(|| {
            datasets
                .par_iter_mut()
                .filter(|d| d.admission != Admission::Excluded)
                .try_for_each(|d| -> Result<()> {
                    let m: Value = read_json(&d.source_dir.join("meta.json"))?;
                    let (train, test, overlap, issues) = inspect(&d.source_dir, &m);
                    d.train = train;
                    d.test = test;
                    d.split_feature_overlap = overlap;
                    d.issues = issues;
                    Ok(())
                })
        })?;
    // Resolve descendants to a fixed point; ambiguous parents never admit a child.
    for _ in 0..datasets.len().min(64) {
        let mut changed = false;
        let mut parents: HashMap<String, Vec<Dataset>> = HashMap::new();
        for d in &datasets {
            parents.entry(d.id.clone()).or_default().push(d.clone());
        }
        for d in &mut datasets {
            if d.admission == Admission::Excluded || d.parents.is_empty() {
                continue;
            }
            let ps = d
                .parents
                .iter()
                .flat_map(|p| parents.get(p).into_iter().flatten())
                .collect::<Vec<_>>();
            let excluded = ps.iter().any(|p| p.admission == Admission::Excluded);
            let approved = !ps.is_empty()
                && d.parents.iter().all(|p| parents.contains_key(p))
                && ps.iter().all(|p| p.admission == Admission::Approved)
                && ps.iter().map(|p| &p.family).collect::<BTreeSet<_>>().len() == 1;
            let next = if excluded {
                Admission::Excluded
            } else if approved {
                Admission::Approved
            } else {
                Admission::Unresolved
            };
            if next != d.admission {
                changed = true;
                d.admission = next;
            }
            if approved {
                d.family = ps[0].family.clone();
                d.evidence = ps[0].evidence.clone();
                d.reason = "deterministic descendant of documented observational parent".into();
            } else {
                d.reason = if excluded {
                    "synthetic ancestor"
                } else {
                    "unresolved or ambiguous parent provenance"
                }
                .into();
            }
        }
        if !changed {
            break;
        }
    }
    // Conservative family union: source identity, declared family and exact training
    // bytes merge alternate targets/duplicates before any meta split is allocated.
    let mut uf: Vec<usize> = (0..datasets.len()).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            p[x] = find(p, p[x]);
        }
        p[x]
    }
    let mut family_keys = HashMap::new();
    for (i, d) in datasets
        .iter()
        .enumerate()
        .filter(|(_, d)| d.admission == Admission::Approved)
    {
        let mut keys = vec![format!("family:{}", d.family)];
        if !d.source_key.is_empty() {
            keys.push(format!("source:{}", d.source_key));
        }
        if d.source_dir.join("train.csv").exists() {
            keys.push(format!(
                "train:{}",
                file_hash(&d.source_dir.join("train.csv"))?
            ));
        }
        for key in keys {
            if let Some(j) = family_keys.insert(key, i) {
                let a = find(&mut uf, i);
                let b = find(&mut uf, j);
                uf[a] = b;
            }
        }
    }
    let mut canonical: BTreeMap<usize, String> = BTreeMap::new();
    for (i, d) in datasets
        .iter()
        .enumerate()
        .filter(|(_, d)| d.admission == Admission::Approved)
    {
        let k = find(&mut uf, i);
        canonical
            .entry(k)
            .and_modify(|f| *f = f.clone().min(d.family.clone()))
            .or_insert(d.family.clone());
    }
    for (i, d) in datasets
        .iter_mut()
        .enumerate()
        .filter(|(_, d)| d.admission == Admission::Approved)
    {
        d.family = canonical[&find(&mut uf, i)].clone();
    }
    let mut families = datasets
        .iter()
        .filter(|d| d.admission == Admission::Approved)
        .map(|d| d.family.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    families.sort_by_key(|f| hash(format!("{SEED}:family:{f}").as_bytes()));
    let n = families.len();
    let partitions: BTreeMap<_, _> = families
        .into_iter()
        .enumerate()
        .map(|(i, f)| {
            (
                f,
                if i < n * 8 / 10 {
                    "meta_train"
                } else if i < n * 9 / 10 {
                    "meta_validation"
                } else {
                    "meta_test"
                }
                .to_string(),
            )
        })
        .collect();
    for d in &mut datasets {
        d.meta_partition = partitions.get(&d.family).cloned();
    }

    let artifact_path = output.join("artifacts.jsonl.partial");
    let mut log = BufWriter::new(File::create(&artifact_path)?);
    let mut counts = BTreeMap::new();
    let mut transfer = TransferManifest {
        schema: "parallel.transfer.v1".into(),
        files: vec![],
    };
    for (root_index, root) in roots.iter().enumerate() {
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(include)
        {
            let e = match entry {
                Ok(e) => e,
                Err(e) => {
                    discovery_errors.push(e.to_string());
                    continue;
                }
            };
            if e.file_type().is_dir() {
                continue;
            }
            let mut ancestor = e.path().parent();
            let mut dataset = None;
            while let Some(a) = ancestor {
                if let Some(i) = dirs.get(a) {
                    dataset = Some(&datasets[*i]);
                    break;
                }
                if a == root {
                    break;
                }
                ancestor = a.parent();
            }
            let packed = ["zip", "npz", "npy", "tar", "gz", "sqlite", "db", "bin"].contains(
                &e.path()
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .as_ref(),
            );
            let (admission, reason) = if e.file_type().is_symlink() {
                (
                    Admission::Unresolved,
                    "symlink requires explicit resolved dependency",
                )
            } else if let Some(d) = dataset {
                if packed && d.admission == Admission::Approved {
                    (
                        Admission::Unresolved,
                        "packed artifact requires member/index reconciliation",
                    )
                } else {
                    (d.admission, d.reason.as_str())
                }
            } else if e.path().components().any(|p| {
                p.as_os_str().to_string_lossy().starts_with("synthetic")
                    || p.as_os_str().to_string_lossy().starts_with("synth_")
            }) {
                (Admission::Excluded, "explicit synthetic container")
            } else {
                (
                    Admission::Unresolved,
                    if packed {
                        "mixed packed artifact requires filtered rebuild"
                    } else {
                        "loose artifact requires dependency/provenance classification"
                    },
                )
            };
            let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
            let sha = if admission == Admission::Approved {
                Some(file_hash(e.path())?)
            } else {
                None
            };
            let disposition = format!("{admission:?}").to_lowercase();
            *counts.entry(disposition.clone()).or_insert(0) += 1;
            serde_json::to_writer(
                &mut log,
                &json!({"source":e.path(),"bytes":bytes,"dataset":dataset.map(|d|&d.id),"admission":disposition,"reason":reason,"sha256":sha}),
            )?;
            log.write_all(b"\n")?;
            if let Some(sha256) = sha {
                let rel = e.path().strip_prefix(root)?;
                transfer.files.push(FileEntry {
                    source: e.path().into(),
                    destination: PathBuf::from(format!("data/originals/root-{root_index}"))
                        .join(rel),
                    bytes,
                    sha256,
                    dataset: dataset.map(|d| d.id.clone()),
                    role: "original_with_legacy_sidecars".into(),
                });
            }
        }
    }
    log.flush()?;
    log.get_ref().sync_all()?;
    fs::rename(artifact_path, output.join("artifacts.jsonl"))?;
    transfer
        .files
        .sort_by(|a, b| a.destination.cmp(&b.destination));
    transfer.validate()?;
    write_json(&output.join("transfer.json"), &transfer)?;
    write_json(&output.join("discovery-errors.json"), &discovery_errors)?;
    let complete = missing_roots.is_empty()
        && discovery_errors.is_empty()
        && counts.get("unresolved").copied().unwrap_or(0) == 0;
    let report = AuditReport {
        schema: "parallel.audit.v1".into(),
        seed: SEED,
        roots,
        datasets,
        artifact_counts: counts,
        missing_roots,
        complete,
    };
    let mut published = serde_json::to_value(&report)?;
    published["sealed_test_access"] = sealed_test_access();
    write_json(&output.join("audit.json"), &published)?;
    Ok(report)
}
