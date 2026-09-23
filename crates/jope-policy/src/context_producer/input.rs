use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingIdentity {
    pub dataset_sha256: String,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub fitting_rows_sha256: String,
    pub cut: u8,
}
impl TrainingIdentity {
    pub fn validate(&self) -> Result<()> {
        for s in [
            &self.dataset_sha256,
            &self.prepared_sha256,
            &self.training_sha256,
            &self.fitting_rows_sha256,
        ] {
            validate_hex(s, 64)?;
        }
        ensure!(self.cut < 5, "cut outside frozen development protocol");
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Numeric,
    Categorical,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feature {
    pub name: String,
    pub kind: FeatureKind,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRow {
    pub id: String,
    pub values: Vec<Option<String>>,
    /// Original regression label. This is not a scoring or outer-heldout label.
    pub label: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingInput {
    pub schema: String,
    pub identity: TrainingIdentity,
    pub features: Vec<Feature>,
    pub rows: Vec<TrainingRow>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnPreprocessing {
    pub name: String,
    pub kind: FeatureKind,
    pub impute: f64,
    pub mean: f64,
    pub scale: f64,
    pub categories: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingPreprocessing {
    pub schema: String,
    pub columns: Vec<ColumnPreprocessing>,
    pub target_mean: f64,
    pub target_scale: f64,
}
pub struct PreparedTraining {
    pub preprocessing: TrainingPreprocessing,
    pub rows: Vec<Vec<f64>>,
}

/// CPU payload identity over original rows, including labels and null/category
/// strings. Historical native references use their own numeric CSV identity.
pub fn training_rows_sha256(rows: &[TrainingRow]) -> Result<String> {
    Ok(sha(&serde_json::to_vec(rows)?))
}

impl TrainingInput {
    pub fn prepare(&self) -> Result<PreparedTraining> {
        ensure!(
            self.schema == "jope.cpu-context-training/regression-v1",
            "only explicit regression training payloads are supported"
        );
        self.identity.validate()?;
        ensure!(
            self.identity.training_sha256 == training_rows_sha256(&self.rows)?,
            "original CPU training row/label content differs from identity"
        );
        ensure!(
            !self.features.is_empty() && self.rows.len() >= 8,
            "encoding requires features and at least8 training rows"
        );
        let mut names = BTreeSet::new();
        ensure!(
            self.features
                .iter()
                .all(|f| !f.name.is_empty() && names.insert(&f.name)),
            "feature names empty or duplicate"
        );
        let ids = self.rows.iter().map(|r| &r.id).collect::<Vec<_>>();
        let mut unique = BTreeSet::new();
        ensure!(
            ids.iter().all(|id| !id.is_empty() && unique.insert(*id)),
            "training IDs empty or duplicate"
        );
        ensure!(
            sha(&serde_json::to_vec(&ids)?) == self.identity.fitting_rows_sha256,
            "training row assignments differ from frozen fitting cut"
        );
        ensure!(
            self.rows
                .iter()
                .all(|r| r.values.len() == self.features.len()),
            "training row width mismatch"
        );
        let mut columns = Vec::new();
        let mut encoded = vec![Vec::with_capacity(self.features.len() + 1); self.rows.len()];
        for (j, feature) in self.features.iter().enumerate() {
            let (column, values) = match feature.kind {
                FeatureKind::Numeric => {
                    let raw = self
                        .rows
                        .iter()
                        .map(|r| r.values[j].as_deref().map(parse_finite).transpose())
                        .collect::<Result<Vec<_>>>()?;
                    let mut present = raw.iter().flatten().copied().collect::<Vec<_>>();
                    present.sort_by(f64::total_cmp);
                    // Match native preprocessing, fail explicitly if its arithmetic overflows.
                    let impute = if present.is_empty() {
                        0.0
                    } else {
                        (present[(present.len() - 1) / 2] + present[present.len() / 2]) / 2.0
                    };
                    let values = raw.iter().map(|v| v.unwrap_or(impute)).collect::<Vec<_>>();
                    let (mean, scale) = mean_scale(&values)?;
                    (
                        ColumnPreprocessing {
                            name: feature.name.clone(),
                            kind: feature.kind.clone(),
                            impute,
                            mean,
                            scale,
                            categories: vec![],
                        },
                        values,
                    )
                }
                FeatureKind::Categorical => {
                    // Every Some string, including empty/NA/null, remains a real category.
                    // Only None is missing; declared kinds prevent accidental numeric inference.
                    let categories = self
                        .rows
                        .iter()
                        .filter_map(|r| r.values[j].clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    let values = self
                        .rows
                        .iter()
                        .map(|r| {
                            r.values[j]
                                .as_ref()
                                .map(|s| {
                                    categories
                                        .binary_search(s)
                                        .expect("category fitted from same training rows")
                                        as f64
                                })
                                .unwrap_or(-1.0)
                        })
                        .collect();
                    (
                        ColumnPreprocessing {
                            name: feature.name.clone(),
                            kind: feature.kind.clone(),
                            impute: -1.0,
                            mean: 0.0,
                            scale: 1.0,
                            categories,
                        },
                        values,
                    )
                }
            };
            for (row, value) in encoded.iter_mut().zip(values) {
                row.push((value - column.mean) / column.scale);
            }
            columns.push(column);
        }
        let targets = self
            .rows
            .iter()
            .map(|r| parse_finite(&r.label))
            .collect::<Result<Vec<_>>>()?;
        let (target_mean, target_scale) = mean_scale(&targets)?;
        for (row, value) in encoded.iter_mut().zip(targets) {
            row.push((value - target_mean) / target_scale);
        }
        validate_rows(&encoded)?;
        Ok(PreparedTraining {
            preprocessing: TrainingPreprocessing {
                schema:
                    "jope.cpu-context-preprocessing/regression-v1:declared-kinds:only-null-missing"
                        .into(),
                columns,
                target_mean,
                target_scale,
            },
            rows: encoded,
        })
    }
}
fn parse_finite(value: &str) -> Result<f64> {
    let value = value
        .parse::<f64>()
        .context("invalid numeric training value")?;
    ensure!(value.is_finite(), "nonfinite numeric training value");
    Ok(value)
}
fn mean_scale(values: &[f64]) -> Result<(f64, f64)> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "empty/nonfinite training column"
    );
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let scale = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64)
        .sqrt()
        .max(1e-12);
    ensure!(
        mean.is_finite() && scale.is_finite(),
        "native preprocessing overflow"
    );
    Ok((mean, scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> TrainingInput {
        let rows = (0..8)
            .map(|i| TrainingRow {
                id: format!("row{i}"),
                values: vec![
                    Some(i.to_string()),
                    [
                        None,
                        Some("".into()),
                        Some("NA".into()),
                        Some("null".into()),
                    ][i % 4]
                        .clone(),
                ],
                label: (i * 2).to_string(),
            })
            .collect::<Vec<_>>();
        TrainingInput {
            schema: "jope.cpu-context-training/regression-v1".into(),
            identity: TrainingIdentity {
                dataset_sha256: "a".repeat(64),
                prepared_sha256: "b".repeat(64),
                training_sha256: training_rows_sha256(&rows).unwrap(),
                fitting_rows_sha256: sha(&serde_json::to_vec(
                    &rows.iter().map(|r| &r.id).collect::<Vec<_>>(),
                )
                .unwrap()),
                cut: 0,
            },
            features: vec![
                Feature {
                    name: "number".into(),
                    kind: FeatureKind::Numeric,
                },
                Feature {
                    name: "category".into(),
                    kind: FeatureKind::Categorical,
                },
            ],
            rows,
        }
    }
    #[test]
    fn retains_original_missing_categories_and_exact_ids() {
        let i = input();
        let p = i.prepare().unwrap();
        assert_eq!(
            p.preprocessing.columns[1].categories,
            vec!["", "NA", "null"]
        );
        assert_eq!(p.rows[0][1], -1.0);
        assert_eq!(p.rows[1][1], 0.0);
        let mut wrong = i.clone();
        wrong.rows.swap(0, 1);
        assert!(wrong.prepare().is_err());
    }
    #[test]
    fn refuses_heldout_fields_and_classification() {
        let i = input();
        let mut json = serde_json::to_value(&i).unwrap();
        json["hold_targets"] = serde_json::json!([1, 2]);
        assert!(serde_json::from_value::<TrainingInput>(json).is_err());
        let mut i = i;
        i.schema = "classification".into();
        assert!(i.prepare().is_err());
    }
    #[test]
    fn rejects_nonfinite_numeric_without_reclassifying() {
        let mut i = input();
        i.rows[0].values[0] = Some("null".into());
        i.identity.training_sha256 = training_rows_sha256(&i.rows).unwrap();
        assert!(i.prepare().is_err());
        i.rows[0].values[0] = None;
        i.identity.training_sha256 = training_rows_sha256(&i.rows).unwrap();
        assert!(i.prepare().is_ok());
        i.rows[0].label = "NaN".into();
        i.identity.training_sha256 = training_rows_sha256(&i.rows).unwrap();
        assert!(i.prepare().is_err());
    }
}
