//! Experimental classification components. Class names are never numeric targets.
//! Foundation is explicitly conditioned on zero, an experimental feature-only
//! probe of existing weights, not a native classification parity claim.
use super::*;

pub const CLASSIFICATION_SCHEMA: &str = "jope.cpu-classification-components/v1";
pub const CLASSIFICATION_INPUT_SCHEMA: &str = "jope.cpu-classification-training/v1";
pub const CLASSIFICATION_FEATURE_POLICY: &str = "asm.experimental-classification-features/v1:training-only-declared-kinds:null-only-missing:hyperion-row128:frozen-dataset-normalizers:foundation-neutral-zero-conditioning:dataset-block-clip32:no-ordinal-targets";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationTask {
    Binary,
    Multiclass,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationInput {
    pub schema: String,
    pub task: ClassificationTask,
    pub classes: Vec<String>,
    /// Original frozen split manifest; distinct from the development blob hash.
    pub split_sha256: String,
    pub identity: TrainingIdentity,
    pub features: Vec<Feature>,
    pub rows: Vec<TrainingRow>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationPreprocessing {
    pub schema: String,
    pub columns: Vec<ColumnPreprocessing>,
}
pub struct PreparedClassification {
    pub preprocessing: ClassificationPreprocessing,
    /// Feature-only numeric rows; no class value has been appended or parsed.
    pub rows: Vec<Vec<f64>>,
}
impl ClassificationInput {
    /// Deterministic train-only preprocessing identity for freezing a comparison
    /// before scheduling its bounded encoder job. This loads no model or tensors.
    pub fn preprocessing_sha256(&self) -> Result<String> {
        Ok(sha(&serde_json::to_vec(&self.prepare()?.preprocessing)?))
    }

    pub fn prepare(&self) -> Result<PreparedClassification> {
        ensure!(
            self.schema == CLASSIFICATION_INPUT_SCHEMA,
            "unknown experimental classification input"
        );
        self.identity.validate()?;
        validate_hex(&self.split_sha256, 64)?;
        ensure!(
            self.identity.training_sha256 == training_rows_sha256(&self.rows)?,
            "classification training content changed"
        );
        ensure!(
            self.rows.len() >= 8 && !self.features.is_empty(),
            "classification requires at least8 rows and features"
        );
        let classes = self.classes.iter().collect::<BTreeSet<_>>();
        ensure!(
            classes.len() == self.classes.len()
                && classes.iter().all(|s| !s.is_empty())
                && match self.task {
                    ClassificationTask::Binary => classes.len() == 2,
                    ClassificationTask::Multiclass => classes.len() >= 3,
                },
            "invalid classification task/vocabulary"
        );
        ensure!(
            self.rows.iter().map(|r| &r.label).collect::<BTreeSet<_>>() == classes,
            "classification train labels differ from declared classes"
        );
        let ids = self.rows.iter().map(|r| &r.id).collect::<Vec<_>>();
        ensure!(
            ids.iter().all(|s| !s.is_empty())
                && ids.iter().copied().collect::<BTreeSet<_>>().len() == ids.len()
                && sha(&serde_json::to_vec(&ids)?) == self.identity.fitting_rows_sha256,
            "classification row assignments changed"
        );
        let names = self
            .features
            .iter()
            .map(|f| &f.name)
            .collect::<BTreeSet<_>>();
        ensure!(
            names.len() == self.features.len()
                && names.iter().all(|s| !s.is_empty())
                && self
                    .rows
                    .iter()
                    .all(|r| r.values.len() == self.features.len()),
            "classification feature schema/width differs"
        );
        let mut columns = Vec::new();
        let mut encoded = vec![Vec::with_capacity(self.features.len()); self.rows.len()];
        for (j, feature) in self.features.iter().enumerate() {
            let (column, values) = match feature.kind {
                FeatureKind::Numeric => {
                    let raw = self
                        .rows
                        .iter()
                        .map(|r| {
                            r.values[j]
                                .as_deref()
                                .map(|v| -> Result<f64> {
                                    let v = v
                                        .parse::<f64>()
                                        .context("invalid classification feature")?;
                                    ensure!(v.is_finite(), "nonfinite classification feature");
                                    Ok(v)
                                })
                                .transpose()
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut present = raw.iter().flatten().copied().collect::<Vec<_>>();
                    present.sort_by(f64::total_cmp);
                    let impute = if present.is_empty() {
                        0.
                    } else {
                        (present[(present.len() - 1) / 2] + present[present.len() / 2]) / 2.
                    };
                    let values = raw
                        .into_iter()
                        .map(|v| v.unwrap_or(impute))
                        .collect::<Vec<_>>();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let scale = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64)
                        .sqrt()
                        .max(1e-12);
                    ensure!(
                        impute.is_finite() && mean.is_finite() && scale.is_finite(),
                        "classification feature preprocessing overflow"
                    );
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
                                    categories.binary_search(s).expect("fitted category") as f64
                                })
                                .unwrap_or(-1.)
                        })
                        .collect::<Vec<_>>();
                    (
                        ColumnPreprocessing {
                            name: feature.name.clone(),
                            kind: feature.kind.clone(),
                            impute: -1.,
                            mean: 0.,
                            scale: 1.,
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
        ensure!(
            encoded.iter().flatten().all(|v| v.is_finite()),
            "classification preprocessing overflow"
        );
        Ok(PreparedClassification {
            preprocessing: ClassificationPreprocessing {
                schema: CLASSIFICATION_FEATURE_POLICY.into(),
                columns,
            },
            rows: encoded,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationArtifact {
    pub schema: String,
    pub feature_policy: String,
    pub backend: String,
    pub task: ClassificationTask,
    pub classes: Vec<String>,
    pub source_sha40: String,
    pub producer_sha256: String,
    pub input_sha256: String,
    pub training: TrainingIdentity,
    pub original_labels_sha256: String,
    pub preprocessing: ClassificationPreprocessing,
    pub preprocessing_sha256: String,
    pub feature_rows_sha256: String,
    /// Identity of components whose native regression-mode qualification is required.
    /// Neutral foundation conditioning itself remains explicitly experimental.
    pub reference_encoder: EncoderIdentity,
    pub hyperion: Vec<f32>,
    pub projected_foundation: Vec<f32>,
    pub row_encodings: Vec<Vec<f32>>,
    pub native_classification_parity_claimed: bool,
    pub accepted_regression_policy_compatible: bool,
    pub final_test_targets_used: bool,
}
impl ClassificationArtifact {
    pub fn validate_for_input(&self, input: &ClassificationInput) -> Result<()> {
        let prepared = input.prepare()?;
        ensure!(
            self.schema == CLASSIFICATION_SCHEMA
                && self.feature_policy == CLASSIFICATION_FEATURE_POLICY
                && self.backend == BACKEND
                && !self.native_classification_parity_claimed
                && !self.accepted_regression_policy_compatible
                && !self.final_test_targets_used,
            "unknown/promoted classification component contract"
        );
        validate_hex(&self.source_sha40, 40)?;
        validate_hex(&self.producer_sha256, 64)?;
        self.reference_encoder.require_accepted()?;
        ensure!(
            self.task == input.task
                && self.classes == input.classes
                && self.training == input.identity
                && self.input_sha256 == sha(&serde_json::to_vec(input)?)
                && self.original_labels_sha256
                    == sha(&serde_json::to_vec(
                        &input.rows.iter().map(|r| &r.label).collect::<Vec<_>>()
                    )?),
            "classification input/labels/identity changed"
        );
        ensure!(
            self.preprocessing_sha256 == sha(&serde_json::to_vec(&prepared.preprocessing)?)
                && self.preprocessing_sha256 == sha(&serde_json::to_vec(&self.preprocessing)?)
                && self.feature_rows_sha256 == sha(&serde_json::to_vec(&prepared.rows)?),
            "classification fitted feature preprocessing changed"
        );
        vector(&self.hyperion, 512)?;
        vector(&self.projected_foundation, 1024)?;
        ensure!(
            self.hyperion
                .iter()
                .chain(&self.projected_foundation)
                .all(|v| v.abs() <= 32.),
            "classification dataset block exceeds pinned clip"
        );
        ensure!(
            self.row_encodings.len() == input.rows.len(),
            "classification row vectors lost rows"
        );
        for row in &self.row_encodings {
            vector(row, 128)?;
        }
        Ok(())
    }
}
/// An explicit neutral-conditioning feature probe. This function never receives
/// labels: class names remain in the original input and the parent equality pool.
fn neutral_foundation_rows(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|row| {
            let mut v = row.clone();
            v.push(0.);
            v
        })
        .collect()
}
impl CpuProducer {
    pub fn produce_classification(
        &self,
        input_bytes: &[u8],
        source_sha40: &str,
        producer_sha256: &str,
    ) -> Result<ClassificationArtifact> {
        ensure!(
            input_bytes.len() <= 128 << 20,
            "classification input exceeds128MiB"
        );
        validate_hex(source_sha40, 40)?;
        validate_hex(producer_sha256, 64)?;
        let input: ClassificationInput = serde_json::from_slice(input_bytes)?;
        // One canonical input encoding makes worker/parent replay exact.
        ensure!(
            serde_json::to_vec(&input)? == input_bytes,
            "classification input must be canonical typed JSON"
        );
        let prepared = input.prepare()?;
        // Input scope of this encode's products: the feature rows' JSON digest.
        let scope = input_scope(&prepared.rows)?;
        let n = prepared.rows.len();
        let width = input.features.len();
        let mut columns = vec![0f32; n * width];
        for (i, row) in prepared.rows.iter().enumerate() {
            for j in 0..width {
                columns[j * n + i] = row[j] as f32;
            }
        }
        ensure!(
            columns.iter().all(|v| v.is_finite()),
            "classification feature conversion exceeds FP32"
        );
        let matrix = Matrix::new(columns, n, width);
        let bounds = fit_quantile_bounds(&matrix, self.settings.q_low, self.settings.q_high);
        let grid = quantize_u8_with_bounds(&matrix, &bounds, self.settings.pad_u8);
        let sweep = run_composite_embedding_sweep(
            &self.hyperion,
            &grid,
            self.settings.pad_u8,
            self.settings.quant_scale,
            self.settings.max_patch,
            &SweepPlan {
                sweeps: 3,
                scales: vec![4, 7, 11, 15],
                batch: 1024,
                seed: self.settings.seed,
                target_chunks_per_sweep: 1,
                fixed_perms: None,
            },
            &NullSink,
        )?;
        ensure!(
            !sweep.cancelled && sweep.row_emb.len() == n * RowEmbeddings::WIDTH,
            "incomplete classification feature sweep"
        );
        let row_embeddings = RowEmbeddings::from_hyperion_sweep(sweep.row_emb, n, &scope)?;
        let pooled = HyperionSignature::pool(&row_embeddings)?;
        let row_encodings = row_embeddings.rows().map(<[f32]>::to_vec).collect();
        let neutral = prepare_jope_foundation_rows(&neutral_foundation_rows(&prepared.rows))?;
        ensure!(
            neutral.input.context_y.iter().all(|v| *v == 0.),
            "neutral foundation labels changed"
        );
        let foundation = RawFoundation::from_foundation_encoder(
            self.foundation.encode_standardized(&neutral.input)?.vector,
            &scope,
        )?;
        // Same frozen named transformations and arithmetic as the regression assembly's first
        // two blocks; the Target block is not produced here.
        let hyperion = self.transforms.signature_block(&pooled)?;
        let projected_foundation = self
            .transforms
            .projection_block(&self.transforms.project(&foundation)?)?;
        let artifact = ClassificationArtifact {
            schema: CLASSIFICATION_SCHEMA.into(),
            feature_policy: CLASSIFICATION_FEATURE_POLICY.into(),
            backend: BACKEND.into(),
            task: input.task,
            classes: input.classes.clone(),
            source_sha40: source_sha40.into(),
            producer_sha256: producer_sha256.into(),
            input_sha256: sha(input_bytes),
            training: input.identity.clone(),
            original_labels_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.label).collect::<Vec<_>>(),
            )?),
            preprocessing_sha256: sha(&serde_json::to_vec(&prepared.preprocessing)?),
            preprocessing: prepared.preprocessing,
            feature_rows_sha256: sha(&serde_json::to_vec(&prepared.rows)?),
            reference_encoder: self.identity.clone(),
            hyperion,
            projected_foundation,
            row_encodings,
            native_classification_parity_claimed: false,
            accepted_regression_policy_compatible: false,
            final_test_targets_used: false,
        };
        artifact.validate_for_input(&input)?;
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ClassificationInput {
        let rows = (0..8)
            .map(|i| TrainingRow {
                id: format!("row{i}"),
                values: vec![
                    if i == 0 { None } else { Some(i.to_string()) },
                    [
                        None,
                        Some("".into()),
                        Some("NA".into()),
                        Some("null".into()),
                    ][i % 4]
                        .clone(),
                ],
                label: if i % 2 == 0 { "red" } else { "blue" }.into(),
            })
            .collect::<Vec<_>>();
        ClassificationInput {
            schema: CLASSIFICATION_INPUT_SCHEMA.into(),
            task: ClassificationTask::Binary,
            classes: vec!["blue".into(), "red".into()],
            split_sha256: "c".repeat(64),
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
                    name: "amount".into(),
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
    fn label_renaming_never_changes_feature_or_neutral_conditioning_bytes() {
        let input = fixture();
        let before = input.prepare().unwrap();
        let mut renamed = input.clone();
        renamed.classes = vec!["999-not-a-number".into(), "null".into()];
        for row in &mut renamed.rows {
            row.label = if row.label == "blue" {
                "999-not-a-number"
            } else {
                "null"
            }
            .into();
        }
        renamed.identity.training_sha256 = training_rows_sha256(&renamed.rows).unwrap();
        let after = renamed.prepare().unwrap();
        assert_eq!(
            serde_json::to_vec(&before.preprocessing).unwrap(),
            serde_json::to_vec(&after.preprocessing).unwrap()
        );
        assert_eq!(
            input.preprocessing_sha256().unwrap(),
            renamed.preprocessing_sha256().unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&before.rows).unwrap(),
            serde_json::to_vec(&after.rows).unwrap()
        );
        assert_eq!(
            neutral_foundation_rows(&before.rows),
            neutral_foundation_rows(&after.rows)
        );
        assert!(neutral_foundation_rows(&before.rows)
            .iter()
            .all(|r| r.last() == Some(&0.)));
        assert_ne!(
            input.identity.training_sha256,
            renamed.identity.training_sha256
        );
        assert_eq!(
            before.preprocessing.columns[1].categories,
            vec!["", "NA", "null"]
        );
        assert!(before.rows.iter().any(|r| r[1] == -1.));
    }
    #[test]
    fn classification_input_cannot_enter_accepted_regression_parser() {
        let input = fixture();
        assert!(
            serde_json::from_slice::<TrainingInput>(&serde_json::to_vec(&input).unwrap()).is_err()
        );
        assert_eq!(input.task, ClassificationTask::Binary);
    }
    #[test]
    fn classification_rejects_changed_labels_ids_or_declared_task() {
        let mut v = fixture();
        v.rows[0].label = "scoring-label".into();
        assert!(v.prepare().is_err());
        let mut v = fixture();
        v.identity.fitting_rows_sha256 = "e".repeat(64);
        assert!(v.prepare().is_err());
        let mut v = fixture();
        v.task = ClassificationTask::Multiclass;
        assert!(v.prepare().is_err());
        let mut v = fixture();
        v.classes[0].clear();
        assert!(v.prepare().is_err());
    }
}
