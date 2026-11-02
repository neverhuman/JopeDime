use super::*;

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
        Self::parse(&bytes, meta)
    }
    /// Parse already-read bytes, so a caller that verified their digest parses
    /// exactly those bytes and never reopens the path.
    pub fn parse(bytes: &[u8], meta: &Value) -> Result<Self> {
        let mut records = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(false)
            .from_reader(bytes);
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
