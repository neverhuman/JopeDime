use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone)]
pub struct Table {
    pub rows: Vec<Vec<String>>,
    pub width: usize,
    pub header: bool,
}
pub fn missing(s: &str) -> bool {
    s.trim().is_empty()
        || [
            "nan",
            "na",
            "null",
            "none",
            "?",
            "inf",
            "-inf",
            "infinity",
            "-infinity",
        ]
        .contains(&s.trim().to_lowercase().as_str())
}
impl Table {
    pub fn read(path: &Path, meta: &Value) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut records = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(false)
            .from_reader(bytes.as_slice());
        let mut rows = records
            .records()
            .map(|r| r.map(|r| r.iter().map(|s| s.trim().to_string()).collect::<Vec<_>>()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(!rows.is_empty(), "empty table");
        let width = rows[0].len();
        ensure!(width >= 2, "expected features and a target");
        let headers = meta
            .get("output_headers")
            .or_else(|| meta.get("original_headers"))
            .and_then(Value::as_array);
        let header = headers.is_some_and(|h| {
            h.len() == width && h.iter().zip(&rows[0]).all(|(a, b)| a.as_str() == Some(b))
        }) || meta.get("headers_removed_from_csv") == Some(&Value::Bool(false));
        if header {
            rows.remove(0);
        }
        ensure!(!rows.is_empty(), "no data rows");
        if let Some(n) = meta.get("n_features").and_then(Value::as_u64) {
            ensure!(
                n as usize + 1 == width,
                "feature count mismatch: metadata {}, CSV {}",
                n,
                width - 1
            );
        }
        ensure!(
            meta.get("target_is_final_column") != Some(&Value::Bool(false)),
            "target must be explicitly relocated to final column"
        );
        Ok(Self {
            rows,
            width,
            header,
        })
    }
    pub fn numeric(path: &Path) -> Result<Vec<Vec<f64>>> {
        Self::read(path, &Value::Null)?
            .rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|s| s.parse::<f64>().context("expected numeric CSV"))
                    .collect()
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub kind: String,
    pub impute: f64,
    pub mean: f64,
    pub scale: f64,
    pub categories: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preprocessing {
    pub schema: String,
    pub columns: Vec<Column>,
    pub labels: Vec<String>,
    pub target_mean: f64,
    pub target_scale: f64,
}
impl Preprocessing {
    pub fn fit(rows: &[Vec<String>], task: &str) -> Result<Self> {
        ensure!(!rows.is_empty(), "empty training cut");
        let p = rows[0].len() - 1;
        let mut columns = Vec::new();
        for j in 0..p {
            let present = rows
                .iter()
                .map(|r| r[j].as_str())
                .filter(|s| !missing(s))
                .collect::<Vec<_>>();
            let numeric = present
                .iter()
                .all(|s| s.parse::<f64>().is_ok_and(f64::is_finite));
            if numeric {
                let mut values = present
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let impute = if values.is_empty() {
                    0.0
                } else {
                    let n = values.len();
                    (values[(n - 1) / 2] + values[n / 2]) / 2.0
                };
                let values = rows
                    .iter()
                    .map(|r| {
                        r[j].parse::<f64>()
                            .ok()
                            .filter(|v| v.is_finite())
                            .unwrap_or(impute)
                    })
                    .collect::<Vec<_>>();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let scale = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64)
                    .sqrt()
                    .max(1e-12);
                columns.push(Column {
                    kind: "numeric".into(),
                    impute,
                    mean,
                    scale,
                    categories: vec![],
                });
            } else {
                columns.push(Column {
                    kind: "categorical".into(),
                    impute: -1.0,
                    mean: 0.0,
                    scale: 1.0,
                    categories: present
                        .into_iter()
                        .map(String::from)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                });
            }
        }
        let classification = matches!(task, "binary" | "multiclass");
        let labels = if classification {
            rows.iter()
                .map(|r| r[p].clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        if classification {
            ensure!(labels.len() >= 2, "training cut has fewer than two classes");
            if task == "binary" {
                ensure!(labels.len() == 2, "binary cut has more than two classes");
            }
        }
        let targets = if classification {
            vec![0.0]
        } else {
            rows.iter()
                .map(|r| r[p].parse::<f64>().context("invalid regression target"))
                .collect::<Result<Vec<_>>>()?
        };
        ensure!(targets.iter().all(|x| x.is_finite()), "nonfinite targets");
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let target_scale = if classification {
            1.0
        } else {
            (targets
                .iter()
                .map(|v| (v - target_mean).powi(2))
                .sum::<f64>()
                / targets.len() as f64)
                .sqrt()
                .max(1e-12)
        };
        Ok(Self {
            schema: "parallel.preprocessing.v1".into(),
            columns,
            labels,
            target_mean,
            target_scale,
        })
    }
    pub fn features(&self, row: &[String]) -> Result<Vec<f64>> {
        ensure!(row.len() == self.columns.len(), "feature width mismatch");
        row.iter()
            .zip(&self.columns)
            .map(|(s, c)| {
                if c.kind == "numeric" {
                    Ok((s
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .unwrap_or(c.impute)
                        - c.mean)
                        / c.scale)
                } else {
                    Ok(c.categories
                        .binary_search(s)
                        .map(|i| i as f64)
                        .unwrap_or(-1.0))
                }
            })
            .collect()
    }
    pub fn transform(&self, rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        rows.iter()
            .map(|row| {
                let mut r = self.features(&row[..row.len() - 1])?;
                let target = row.last().unwrap();
                r.push(if self.labels.is_empty() {
                    (target.parse::<f64>()? - self.target_mean) / self.target_scale
                } else {
                    self.labels
                        .binary_search(target)
                        .map(|i| i as f64)
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "evaluation target absent from training label mapping: {target}"
                            )
                        })?
                });
                ensure!(r.iter().all(|x| x.is_finite()), "nonfinite prepared row");
                Ok(r)
            })
            .collect()
    }
}
pub fn csv_bytes(rows: &[Vec<f64>]) -> Result<Vec<u8>> {
    let mut w = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    for r in rows {
        w.write_record(r.iter().map(|x| x.to_string()))?;
    }
    Ok(w.into_inner()?)
}

/// Evaluation row selection uses only original position, split and seed.
pub fn cap_indices(
    n_train: usize,
    n_test: usize,
    task: &str,
    train: &[Vec<String>],
) -> Result<(Vec<usize>, Vec<usize>)> {
    let total = n_train + n_test;
    ensure!(
        n_train > 0 && n_test > 0,
        "both original partitions are required"
    );
    let (nt, ne) = if total <= ROW_LIMIT {
        (n_train, n_test)
    } else {
        let nt = ((ROW_LIMIT as u128 * n_train as u128) / total as u128) as usize;
        (
            nt.max(1).min(ROW_LIMIT - 1),
            ROW_LIMIT - nt.max(1).min(ROW_LIMIT - 1),
        )
    };
    if task == "timeseries" {
        ensure!(n_test < ROW_LIMIT, "sealed horizon exceeds row allowance");
        let nt = n_train.min(ROW_LIMIT - n_test);
        return Ok(((n_train - nt..n_train).collect(), (0..n_test).collect()));
    }
    let rank = |split: &str, n: usize| {
        let mut x = (0..n).collect::<Vec<_>>();
        x.sort_by_key(|i| hash(format!("{SEED}:cap:{split}:{i}").as_bytes()));
        x
    };
    let mut ti = rank("train", n_train);
    let mut ei = rank("evaluation", n_test);
    ei.truncate(ne);
    ei.sort_unstable();
    if matches!(task, "binary" | "multiclass") {
        let mut coverage = BTreeMap::new();
        for &i in &ti {
            coverage.entry(train[i].last().unwrap()).or_insert(i);
        }
        ensure!(coverage.len() <= nt, "cap cannot preserve training classes");
        let selected = coverage.into_values().collect::<BTreeSet<_>>();
        ti.retain(|i| !selected.contains(i));
        ti.truncate(nt - selected.len());
        ti.extend(selected);
    } else {
        ti.truncate(nt);
    }
    ti.sort_unstable();
    Ok((ti, ei))
}
pub fn folds(
    rows: &[Vec<String>],
    task: &str,
    horizon: usize,
    entity_groups: Option<&[String]>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    folds_with_seed(rows, task, horizon, entity_groups, SEED)
}
pub fn folds_with_seed(
    rows: &[Vec<String>],
    task: &str,
    horizon: usize,
    entity_groups: Option<&[String]>,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    ensure!(rows.len() >= 6, "insufficient history for five cuts");
    if task == "timeseries" {
        ensure!(
            horizon > 0 && rows.len() > 5 * horizon,
            "insufficient history for five rolling origins"
        );
        return Ok((0..5)
            .map(|cut| {
                let end = rows.len() - (4 - cut) * horizon;
                ((0..end - horizon).collect(), (end - horizon..end).collect())
            })
            .collect());
    }
    if let Some(g) = entity_groups {
        ensure!(g.len() == rows.len(), "entity group length mismatch");
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    // Union explicit entity IDs with feature duplicates, including conflicting labels.
    let mut uf: Vec<usize> = (0..rows.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        if p[i] != i {
            p[i] = find(p, p[i]);
        }
        p[i]
    }
    let mut seen = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let mut keys = vec![format!(
            "x:{}",
            hash(&serde_json::to_vec(&r[..r.len() - 1])?)
        )];
        if let Some(g) = entity_groups {
            keys.push(format!("entity:{}", g[i]));
        }
        for key in keys {
            if let Some(j) = seen.insert(key, i) {
                let a = find(&mut uf, i);
                let b = find(&mut uf, j);
                uf[a] = b;
            }
        }
    }
    for i in 0..rows.len() {
        let k = find(&mut uf, i);
        groups.entry(k.to_string()).or_default().push(i);
    }
    ensure!(
        groups.len() >= 5,
        "fewer than five independent row/entity groups"
    );
    let mut strata: BTreeMap<String, Vec<Vec<usize>>> = BTreeMap::new();
    for g in groups.into_values() {
        let labels = g
            .iter()
            .map(|&i| rows[i].last().unwrap().clone())
            .collect::<BTreeSet<_>>();
        let stratum = if matches!(task, "binary" | "multiclass") {
            serde_json::to_string(&labels)?
        } else {
            String::new()
        };
        strata.entry(stratum).or_default().push(g);
    }
    let mut assignment = vec![0; rows.len()];
    let mut loads = [0usize; 5];
    for mut gs in strata.into_values() {
        gs.sort_by_key(|g| hash(format!("{seed}:fold:{:?}", g).as_bytes()));
        let mut local = [0usize; 5];
        for g in gs {
            let fold = (0..5).min_by_key(|&f| (local[f], loads[f], f)).unwrap();
            local[fold] += g.len();
            loads[fold] += g.len();
            for i in g {
                assignment[i] = fold;
            }
        }
    }
    (0..5)
        .map(|cut| {
            let (val, train): (Vec<_>, Vec<_>) =
                (0..rows.len()).partition(|&i| assignment[i] == cut);
            ensure!(
                !train.is_empty() && !val.is_empty(),
                "empty fold after grouping"
            );
            if matches!(task, "binary" | "multiclass") {
                let all = rows
                    .iter()
                    .map(|r| r.last().unwrap())
                    .collect::<BTreeSet<_>>();
                let labels = train
                    .iter()
                    .map(|&i| rows[i].last().unwrap())
                    .collect::<BTreeSet<_>>();
                ensure!(
                    all == labels,
                    "five class-covering folds impossible with these groups"
                );
                let heldout_labels = val
                    .iter()
                    .map(|&i| rows[i].last().unwrap())
                    .collect::<BTreeSet<_>>();
                ensure!(
                    all == heldout_labels,
                    "insufficient_class_groups: every held-out fold must contain every class"
                );
            }
            Ok((train, val))
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    pub index: usize,
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub preprocessing_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prepared {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<ViewSource>,
    pub schema: String,
    pub dataset: String,
    pub family: String,
    pub task: String,
    pub meta_partition: String,
    pub seed: u64,
    #[serde(default)]
    pub family_partition_frozen: bool,
    pub clean_evaluation: bool,
    pub clean_status: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qualification_only: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluation_limitations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_registry_binding: Option<crate::families::RegistryBinding>,
    pub original_train_sha256: String,
    pub original_test_sha256: String,
    pub selected_train_indices: Vec<usize>,
    pub selected_test_indices: Vec<usize>,
    pub target_last: bool,
    pub rows_total: usize,
    pub cuts: Vec<Cut>,
}

/// Recover raw numeric rows by replaying the recorded historical transform and
/// matching every original training row. Unknown recipes or ambiguous raw rows
/// are withheld; reversing a scaler alone is never treated as reconstruction.
pub fn reconstruct_numeric(train: &Table, m: &Value) -> Result<(Table, Value)> {
    reconstruct_recorded(train, m, false)
}

/// Classification reconstruction replays the recorded source-label encoding
/// and class-to-code mapping before verifying every historical training row.
pub fn reconstruct_classification(train: &Table, m: &Value) -> Result<(Table, Value)> {
    ensure!(
        m["target_scaling"]["mode"] == "classification",
        "classification needs a recorded label mapping"
    );
    ensure!(
        m["target_scaling"]["scaled_code_formula"] == "code / max(1, n_classes - 1)",
        "unsupported historical classification formula"
    );
    reconstruct_recorded(train, m, true)
}

fn historical_class(target: &str, m: &Value) -> Result<f64> {
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

/// Recorded backing sources predate ADR-0003: they name a retired host tree
/// (`…/hyper/data/<rel>`). The mirror of that tree sits beside the dataset
/// tree, `<quant_root>/../hyper/data` (historically `remote_super/hyper/data`
/// next to `remote_super/quant`), so a recorded path that no longer exists is
/// re-rooted there by the part after its `hyper/data` components.
fn mirrored_backing_source(recorded: &std::path::Path) -> Option<std::path::PathBuf> {
    let components: Vec<_> = recorded.components().collect();
    let index = components
        .windows(2)
        .position(|pair| pair[0].as_os_str() == "hyper" && pair[1].as_os_str() == "data")?;
    let relative: std::path::PathBuf = components[index + 2..].iter().collect();
    let mirror = jope_paths::quant_root().parent()?.join("hyper/data");
    Some(mirror.join(relative))
}

fn reconstruct_recorded(train: &Table, m: &Value, classification: bool) -> Result<(Table, Value)> {
    ensure!(
        classification || m["target_generation"] == "original_numeric_column",
        "target derivation needs its own reconstruction recipe"
    );
    ensure!(
        m["target_fix"]["changed"] != true,
        "changed legacy target requires a verified transformation recipe"
    );
    let source = m["source_path"]
        .as_str()
        .context("missing backing-source path")?;
    let mut source = std::path::PathBuf::from(source);
    if !source.is_file() {
        if let Some(mirrored) = mirrored_backing_source(&source) {
            source = mirrored;
        }
    }
    let mut reader = csv::Reader::from_path(&source).context("backing source unavailable")?;
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
    let mut candidates: BTreeMap<i64, Vec<(usize, Vec<String>, Vec<f64>)>> = BTreeMap::new();
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
        mapping.push(hits[0].0);
        raw_rows.push(hits[0].1.clone());
    }
    Ok((
        Table {
            rows: raw_rows,
            width: train.width,
            header: false,
        },
        serde_json::json!({"schema":"parallel.raw_reconstruction.v1","source":source,"source_sha256":file_hash(&source)?,"source_row_indices":mapping,"target_column":header.get(target_index),"recipe":if classification {"replay_recorded_feature_encoding_minmax_and_classification_mapping"} else {"replay_recorded_numeric_or_categorical_encoding_named_minmax_and_original_target_minmax"},"classification_mapping":if classification {m["target_scaling"].clone()} else {Value::Null},"historical_scaler_feature_indices":scale_positions,"recipe_metadata_sha256":hash(&serde_json::to_vec(m)?),"max_match_abs_error":observed_match_error,"match_tolerance":1e-7,"equivalent_source_matches":equivalent_source_matches,"training_rows":train.rows.len()}),
    ))
}

fn documented_entity_columns(m: &Value) -> Vec<String> {
    match m["source_dataset_id"].as_str().unwrap_or("") {
        "MASS/epil" => vec!["subject".into()],
        "MASS/OME" => vec!["ID".into()],
        "datasets/DNase" => vec!["Run".into()],
        "MASS/nlschools" => vec!["class".into()],
        "plm/Parity" => vec!["country".into()],
        _ => vec![],
    }
}
fn entity_documentation(m: &Value) -> Option<String> {
    if documented_entity_columns(m).is_empty() {
        None
    } else {
        Some(format!(
            "https://vincentarelbundock.github.io/Rdatasets/doc/{}.html",
            m["source_dataset_id"].as_str().unwrap()
        ))
    }
}
fn reviewed_entity_columns(
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
fn entity_values(
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

pub fn run(
    audit: &Path,
    dataset: &str,
    output: &Path,
    allow_legacy: bool,
    qualification: bool,
    backing_manifest: &Path,
) -> Result<Prepared> {
    run_with_admission(
        audit,
        dataset,
        output,
        allow_legacy,
        qualification,
        backing_manifest,
        None,
        None,
    )
}

pub fn run_with_admission(
    audit: &Path,
    dataset: &str,
    output: &Path,
    allow_legacy: bool,
    qualification: bool,
    backing_manifest: &Path,
    family_registry: Option<&Path>,
    entity_review: Option<&Path>,
) -> Result<Prepared> {
    let report: AuditReport = read_json(audit)?;
    let candidates = report
        .datasets
        .iter()
        .filter(|d| d.id == dataset)
        .collect::<Vec<_>>();
    let mut matching = Vec::new();
    for d in candidates {
        if d.admission == Admission::Excluded {
            continue;
        }
        if let Some(path) = family_registry {
            if let Ok(binding) = crate::families::binding(path, &d.id, &d.source_key, &d.family) {
                if binding.metadata_sha256 == d.metadata_sha256 {
                    matching.push((d, Some(binding)));
                }
            }
        } else if d.admission == Admission::Approved {
            matching.push((d, None));
        }
    }
    // The full inventory includes acquisition stubs and release copies. Prefer
    // the canonical task/id directory, after any registry metadata constraint.
    let canonical = |d: &Dataset| {
        d.source_dir.file_name().and_then(|s| s.to_str()) == Some(d.id.as_str())
            && d.source_dir
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                == Some(if d.task == "multiclass" {
                    "multi_label"
                } else {
                    d.task.as_str()
                })
    };
    if matching.iter().any(|(d, _)| canonical(d)) {
        matching.retain(|(d, _)| canonical(d));
    }
    ensure!(
        matching.len() == 1,
        "dataset needs exactly one approved or registry-bound canonical audit identity"
    );
    let (d, family_binding) = matching.pop().unwrap();
    ensure!(
        d.admission == Admission::Approved
            || family_binding
                .as_ref()
                .is_some_and(|b| b.entity_review_sha256.is_some()),
        "dataset needs an approved audit or frozen reviewed preparation admission"
    );
    if let Some(binding) = &family_binding {
        ensure!(
            !binding.qualification || qualification,
            "family registry requires explicit qualification admission"
        );
        ensure!(
            binding.entity_review_sha256.is_some() == entity_review.is_some(),
            "provide exactly the entity review required by the frozen registry"
        );
    } else {
        ensure!(
            entity_review.is_none(),
            "entity review requires a frozen family registry"
        );
    }
    let m: Value = read_json(&d.source_dir.join("meta.json"))?;
    ensure!(
        file_hash(&d.source_dir.join("meta.json"))? == d.metadata_sha256,
        "metadata changed since audit"
    );
    if let Some(binding) = &family_binding {
        ensure!(
            binding.metadata_sha256 == d.metadata_sha256,
            "metadata differs from frozen family registry"
        );
    }
    let original = Table::read(&d.source_dir.join("train.csv"), &m)?;
    let test = d
        .test
        .as_ref()
        .context("audited final-test dimensions absent")?;
    ensure!(original.width == test.columns, "split schema mismatch");
    let audited_transfer: TransferManifest = read_json(
        &audit
            .parent()
            .context("audit parent absent")?
            .join("transfer.json"),
    )?;
    let test_identity = audited_transfer
        .files
        .iter()
        .find(|f| f.source == d.source_dir.join("test.csv"));
    ensure!(
        test_identity.is_some() || (qualification && d.task == "binary"),
        "audited final-test hash absent; final targets remain sealed"
    );
    let reconstruction = if matches!(d.task.as_str(), "binary" | "multiclass") {
        reconstruct_classification(&original, &m)
    } else {
        reconstruct_numeric(&original, &m)
    };
    let (train, reconstruction_receipt, reconstruction_error) = match reconstruction {
        Ok((table, receipt)) => (table, Some(receipt), None),
        Err(e) => (original, None, Some(format!("{e:#}"))),
    };
    // The reconstructed cut can be used for training. Final-test preprocessing
    // must be reconstructed and qualified separately before a score is claimed.
    let clean =
        reconstruction_receipt.is_some() && d.split_feature_overlap == 0 && test_identity.is_some();
    ensure!(clean||allow_legacy||(qualification && reconstruction_receipt.is_some()),"backing-source row mapping/preprocessing reconstruction required; original-test overlap requires explicit qualification");
    let (ti, ei) = cap_indices(train.rows.len(), test.rows, &d.task, &train.rows)?;
    let receipt = reconstruction_receipt
        .as_ref()
        .context("pointer cuts require reconstructed source rows")?;
    let transfer: TransferManifest = read_json(backing_manifest)?;
    let source_hash = receipt["source_sha256"]
        .as_str()
        .context("source hash absent")?;
    let backing = transfer
        .files
        .iter()
        .find(|f| f.sha256 == source_hash)
        .context("backing source is not in the transfer allowlist")?;
    let storage = ViewSource {
        source: backing.destination.clone(),
        source_sha256: source_hash.into(),
        feature_columns: m["feature_columns"]
            .as_array()
            .context("feature order absent")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect(),
        target_column: receipt["target_column"]
            .as_str()
            .context("target column absent")?
            .into(),
        original_training_row_map: serde_json::from_value(receipt["source_row_indices"].clone())?,
    };
    validate_view_source(&storage)?;
    let rows = ti
        .iter()
        .map(|&i| train.rows[i].clone())
        .collect::<Vec<_>>();
    reject_target_copies(&storage.feature_columns, &rows)?;
    let horizon = m
        .get("horizon")
        .and_then(Value::as_u64)
        .unwrap_or(test.rows as u64) as usize;
    let (entity_columns, grouping_binding) = if let Some(path) = entity_review {
        let expected = family_binding
            .as_ref()
            .and_then(|b| b.entity_review_sha256.as_deref())
            .context("frozen entity review identity absent")?;
        let (columns, binding) = reviewed_entity_columns(path, expected, d, &m, receipt)?;
        (columns, Some(binding))
    } else {
        (documented_entity_columns(&m), None)
    };
    let entities = entity_values(&rows, &storage.feature_columns, &entity_columns)?;
    let indices = folds(&rows, &d.task, horizon, entities.as_deref())?;
    for (tr, va) in &indices {
        check_cut_membership(
            &storage.original_training_row_map,
            &tr.iter().map(|&i| ti[i]).collect::<Vec<_>>(),
            &va.iter().map(|&i| ti[i]).collect::<Vec<_>>(),
        )?;
    }
    ensure!(
        !output.exists(),
        "prepared destination already exists; use a new immutable version"
    );
    fs::create_dir_all(output)?;
    write_json(
        &output.join("reconstruction.json"),
        &serde_json::json!({"receipt":reconstruction_receipt,"error":reconstruction_error,"sealed_test_reconstruction":"not_yet_qualified","original_test_feature_overlap":d.split_feature_overlap,"original_test_targets_opened":false,"test_identity_source":if test_identity.is_some(){"existing audited transfer manifest"}else{"unavailable_unopened_binary_qualification_only"},"entity_columns":entity_columns,"grouping_source_dataset_id":m["source_dataset_id"],"grouping_documentation":entity_documentation(&m),"grouping_binding":grouping_binding,"family_registry_binding":family_binding}),
    )?;
    let mut cuts = Vec::new();
    for (index, (tr, va)) in indices.into_iter().enumerate() {
        let training = tr.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let validation = va.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let pp = Preprocessing::fit(&training, &d.task)?;
        let dir = output.join(format!("cut-{index}"));
        fs::create_dir_all(&dir)?;
        let tb = csv_bytes(&pp.transform(&training)?)?;
        let vb = csv_bytes(&pp.transform(&validation)?)?;
        write_json(&dir.join("preprocessing.json"), &pp)?;
        cuts.push(Cut {
            index,
            train_indices: tr.iter().map(|&i| ti[i]).collect(),
            validation_indices: va.iter().map(|&i| ti[i]).collect(),
            training_sha256: hash(&tb),
            validation_sha256: hash(&vb),
            preprocessing_sha256: file_hash(&dir.join("preprocessing.json"))?,
        });
    }
    let p = Prepared {
        storage: Some(storage),
        schema: "parallel.prepared.v1".into(),
        dataset: d.id.clone(),
        family: family_binding
            .as_ref()
            .map(|b| b.family.clone())
            .unwrap_or_else(|| d.family.clone()),
        task: d.task.clone(),
        meta_partition: family_binding
            .as_ref()
            .map(|b| b.partition.clone())
            .or_else(|| d.meta_partition.clone())
            .context("missing family partition")?,
        seed: SEED,
        family_partition_frozen: family_binding
            .as_ref()
            .map(|b| b.full_inventory_complete)
            .unwrap_or_else(|| {
                // The inventory is frozen only when it was taken over the
                // authoritative dataset tree (JOPE_REGISTRY's parent).
                let authoritative = jope_paths::quant_root();
                report.complete && report.roots.iter().any(|r| *r == authoritative)
            }),
        clean_evaluation: clean,
        clean_status: if clean {
            "raw_training_reconstructed_final_test_unqualified"
        } else if qualification && reconstruction_receipt.is_some() {
            "raw_training_reconstructed_qualification_only"
        } else {
            "legacy_preprocessing_not_reconstructed"
        }
        .into(),
        qualification_only: qualification && !clean,
        evaluation_limitations: if d.split_feature_overlap > 0 {
            vec![format!(
                "original_test_feature_overlap:{}; final evaluation remains ineligible",
                d.split_feature_overlap
            )]
        } else {
            vec![]
        },
        entity_columns,
        family_registry_binding: family_binding,
        original_train_sha256: file_hash(&d.source_dir.join("train.csv"))?,
        original_test_sha256: test_identity.map(|f| f.sha256.clone()).unwrap_or_default(),
        selected_train_indices: ti,
        selected_test_indices: ei.clone(),
        target_last: true,
        rows_total: rows.len() + ei.len(),
        cuts,
    };
    // Seal metadata binds original evaluation bytes; no final targets enter cut files.
    write_json(&output.join("prepared.json"), &p)?;
    Ok(p)
}

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
fn load_view_with_root(
    root: Option<&Path>,
    prepared: &Path,
    cut: usize,
    training: bool,
) -> Result<Vec<Vec<f64>>> {
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown view")?;
    let dir = prepared.join(format!("cut-{cut}"));
    let pp_path = dir.join("preprocessing.json");
    ensure!(
        file_hash(&pp_path)? == c.preprocessing_sha256,
        "view preprocessing changed"
    );
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
        let pp: Preprocessing = read_json(&pp_path)?;
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
    ensure!(
        file_hash(&path)? == source.source_sha256,
        "backing table changed"
    );
    let mut reader = csv::Reader::from_path(path)?;
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

/// Repeated, seeded five-fold partitions add teacher views without copying data.
/// Duplicate training memberships are skipped; source families do not multiply.
pub fn expand_cuts(prepared: &Path, output: &Path, rounds: usize) -> Result<Prepared> {
    ensure!(rounds > 0 && rounds <= 1000, "rounds must be in 1..=1000");
    ensure!(
        !output.exists(),
        "expanded destination exists; use an immutable new version"
    );
    let mut p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.task != "timeseries",
        "time-series expansion requires distinct rolling origins"
    );
    let source = p
        .storage
        .as_ref()
        .context("expansion requires pointer views")?;
    let raw = load_raw_view(prepared, source, &p.selected_train_indices)?;
    reject_target_copies(&source.feature_columns, &raw)?;
    let entities = entity_values(&raw, &source.feature_columns, &p.entity_columns)?;
    fs::create_dir_all(output)?;
    let parent_sha = file_hash(&prepared.join("prepared.json"))?;
    let mut cuts = Vec::new();
    let mut seen = BTreeSet::new();
    let mut recipes = Vec::new();
    let mut skipped = 0;
    for round in 0..rounds {
        let seed = if round == 0 {
            p.seed
        } else {
            u64::from_str_radix(
                &hash(format!("{}:repeated-folds:{round}", p.seed).as_bytes())[..16],
                16,
            )?
        };
        for (fold, (tr, va)) in folds_with_seed(&raw, &p.task, 1, entities.as_deref(), seed)?
            .into_iter()
            .enumerate()
        {
            let membership = hash(&serde_json::to_vec(&tr)?);
            if !seen.insert(membership) {
                skipped += 1;
                continue;
            }
            let index = round * 5 + fold;
            let train_indices: Vec<usize> =
                tr.iter().map(|&i| p.selected_train_indices[i]).collect();
            let validation_indices: Vec<usize> =
                va.iter().map(|&i| p.selected_train_indices[i]).collect();
            check_cut_membership(
                &source.original_training_row_map,
                &train_indices,
                &validation_indices,
            )?;
            let training = tr.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
            let heldout = va.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
            let pp = Preprocessing::fit(&training, &p.task)?;
            let dir = output.join(format!("cut-{index}"));
            fs::create_dir_all(&dir)?;
            write_json(&dir.join("preprocessing.json"), &pp)?;
            let cut = Cut {
                index,
                train_indices,
                validation_indices,
                training_sha256: hash(&csv_bytes(&pp.transform(&training)?)?),
                validation_sha256: hash(&csv_bytes(&pp.transform(&heldout)?)?),
                preprocessing_sha256: file_hash(&dir.join("preprocessing.json"))?,
            };
            if round == 0 {
                let old = p
                    .cuts
                    .iter()
                    .find(|c| c.index == fold)
                    .context("base fold absent")?;
                ensure!(
                    old.training_sha256 == cut.training_sha256
                        && old.validation_sha256 == cut.validation_sha256,
                    "base fold changed during expansion"
                );
            }
            recipes.push(
                serde_json::json!({"cut":index,"round":round,"fold":fold,"partition_seed":seed}),
            );
            cuts.push(cut);
        }
    }
    p.cuts = cuts;
    write_json(&output.join("prepared.json"), &p)?;
    write_json(
        &output.join("expansion.json"),
        &serde_json::json!({"schema":"parallel.repeated_folds.v1","parent_prepared_sha256":parent_sha,"rounds":rounds,"cuts":recipes,"duplicate_cuts_skipped":skipped,"source_family":p.family,"independent_families":1,"copied_dataset_bytes":0,"heldout_role":"adaptive GP teacher objective; not an unbiased final test","final_test_targets_used":false}),
    )?;
    Ok(p)
}

/// Adapt an already verified cut to pointers and prove exact byte parity before
/// it is used. Original cut manifests remain as historical evidence.
pub fn index_existing(
    prepared: &Path,
    backing_manifest: &Path,
    metadata: &Path,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "indexed destination exists");
    let mut p: Prepared = read_json(&prepared.join("prepared.json"))?;
    let reconstruction: Value = read_json(&prepared.join("reconstruction.json"))?;
    let receipt = &reconstruction["receipt"];
    let m: Value = read_json(metadata)?;
    let transfer: TransferManifest = read_json(backing_manifest)?;
    let sha = receipt["source_sha256"]
        .as_str()
        .context("raw source identity missing")?;
    let entry = transfer
        .files
        .iter()
        .find(|f| f.sha256 == sha)
        .context("backing source not allowlisted")?;
    p.storage = Some(ViewSource {
        source: entry.destination.clone(),
        source_sha256: sha.into(),
        feature_columns: serde_json::from_value(m["feature_columns"].clone())?,
        target_column: m["target_column"]
            .as_str()
            .or_else(|| m["source_target_column_original"].as_str())
            .context("target name missing")?
            .into(),
        original_training_row_map: serde_json::from_value(receipt["source_row_indices"].clone())?,
    });
    validate_view_source(p.storage.as_ref().context("indexed storage absent")?)?;
    fs::create_dir_all(output)?;
    for c in &p.cuts {
        let dir = output.join(format!("cut-{}", c.index));
        fs::create_dir_all(&dir)?;
        atomic_write(
            &dir.join("preprocessing.json"),
            &fs::read(prepared.join(format!("cut-{}/preprocessing.json", c.index)))?,
        )?;
    }
    write_json(&output.join("prepared.json"), &p)?;
    write_json(&output.join("reconstruction.json"), &reconstruction)?;
    for c in &p.cuts {
        load_view(output, c.index, true)?;
        load_view(output, c.index, false)?;
    }
    write_json(
        &output.join("pointer-parity.json"),
        &serde_json::json!({"passed":true,"source_prepared_sha256":file_hash(&prepared.join("prepared.json"))?,"views_checked":p.cuts.len()*2,"method":"exact SHA-256 of on-demand transformed view bytes","copied_dataset_bytes":0}),
    )
}

#[cfg(test)]
mod grouping_review_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reviewed_grouping_binds_frozen_identity_and_rejects_holds_and_target_columns() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join("train.csv"), "0,0\n1,1\n").unwrap();
        fs::write(root.join("evidence.html"), "Documented repeated subjects").unwrap();
        let metadata = json!({"source_platform":"rdatasets","source_dataset_id":"example/repeated","feature_columns":["subject","x"]});
        let dataset: Dataset = serde_json::from_value(json!({"id":"example","task":"regression","source_dir":root,"source_key":"rdatasets:example/repeated","family":"example","parents":[],"admission":"approved","reason":"test fixture","evidence":[],"metadata_sha256":"a".repeat(64),"train":null,"test":null,"split_feature_overlap":0,"issues":[],"meta_partition":"meta_train"})).unwrap();
        let reconstruction = json!({"source_sha256":"b".repeat(64),"source_row_indices":[1,2]});
        let mut review = json!({"schema":"parallel.provenance_review.v2","datasets":[{"id":"example","source_key":"rdatasets:example/repeated","metadata":{"sha256":dataset.metadata_sha256},"reconstruction":{"metadata_sha256":dataset.metadata_sha256,"source_sha256":reconstruction["source_sha256"],"training_sha256":file_hash(&root.join("train.csv")).unwrap(),"row_mapping_sha256":hash(&serde_json::to_vec(&reconstruction["source_row_indices"]).unwrap())},"pilot_candidate_status":"eligible_for_fresh_cohort_preparation","pilot_hold_reason":null,"entity_columns":["subject"],"documentation":{"path":root.join("evidence.html"),"sha256":file_hash(&root.join("evidence.html")).unwrap(),"source_url":"https://example.test/source","finding":"Documented repeated subjects"}}]});
        let path = root.join("review.json");
        write_json(&path, &review).unwrap();
        let sha = file_hash(&path).unwrap();
        let (columns, binding) =
            reviewed_entity_columns(&path, &sha, &dataset, &metadata, &reconstruction).unwrap();
        assert_eq!(columns, ["subject"]);
        assert_eq!(binding["manifest_sha256"], sha);
        assert!(reviewed_entity_columns(
            &path,
            &"0".repeat(64),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("frozen family registry"));
        let changed_mapping = json!({"source_sha256":"b".repeat(64),"source_row_indices":[2,1]});
        assert!(
            reviewed_entity_columns(&path, &sha, &dataset, &metadata, &changed_mapping)
                .unwrap_err()
                .to_string()
                .contains("identity mismatch")
        );
        review["datasets"][0]["pilot_hold_reason"] = json!("unresolved_entity_mapping");
        write_json(&path, &review).unwrap();
        assert!(reviewed_entity_columns(
            &path,
            &file_hash(&path).unwrap(),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("withholds"));
        review["datasets"][0]["pilot_hold_reason"] = Value::Null;
        review["datasets"][0]["entity_columns"] = json!(["target"]);
        write_json(&path, &review).unwrap();
        assert!(reviewed_entity_columns(
            &path,
            &file_hash(&path).unwrap(),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("input features"));
    }
    #[test]
    fn explicit_backing_root_preserves_original_rows_and_refuses_fallback() {
        let parent = tempfile::tempdir().unwrap();
        let source = parent.path().join("source");
        let restored = source.join("recovery/restored");
        fs::create_dir_all(source.join("manifests")).unwrap();
        fs::write(source.join("Cargo.toml"), b"source marker").unwrap();
        fs::create_dir_all(&restored).unwrap();
        let original = b"x,target\n1,10\n2,20\n3,30\n";
        fs::write(source.join("table.csv"), original).unwrap();
        let view = ViewSource {
            source: "table.csv".into(),
            source_sha256: hash(original),
            feature_columns: vec!["x".into()],
            target_column: "target".into(),
            original_training_row_map: vec![2, 0],
        };
        // An outer checkout has valid bytes, but cannot satisfy the explicit relocated root.
        assert!(load_raw_view_in_root(&restored, &view, &[0, 1]).is_err());
        fs::write(restored.join("table.csv"), original).unwrap();
        assert_eq!(
            load_raw_view_in_root(&restored, &view, &[0, 1]).unwrap(),
            vec![
                vec!["3".to_string(), "30".to_string()],
                vec!["1".to_string(), "10".to_string()]
            ]
        );
        fs::write(restored.join("table.csv"), b"x,target\n1,99\n").unwrap();
        assert!(load_raw_view_in_root(&restored, &view, &[0]).is_err());
        fs::remove_file(restored.join("table.csv")).unwrap();
        std::os::unix::fs::symlink(source.join("table.csv"), restored.join("table.csv")).unwrap();
        assert!(load_raw_view_in_root(&restored, &view, &[0]).is_err());
    }
}
