//! Frozen-panel methods. Historical same-cut v1 methods and adapters are unchanged.
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    validate_v3_genome, BoundModel, ColumnSpec, DataSchema, RawTable, TaskSpec,
};
use feat_math::GenomeIr;
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::data::{fitting_input_sha256, raw_table, RAW_CELL_ENCODING};
use super::score::{predict_bound, Predictions};

/// Unbound: names no fitting input. Every artifact written before v2; still loadable.
pub const RUST_ARTIFACT: &str = "jope.panel-fitted-rust/v1";
/// W3-T04: also names the exact fitting input it was fitted on
/// (`prepared_data::fitting_input_sha256` of its training table and targets), which the fit
/// worker computes from its own input without receiving any row identity.
pub const RUST_ARTIFACT_V2: &str = "jope.panel-fitted-rust/v2";
pub const ADAPTER_CONTRACT: &str = "jope.bench-adapter/v2";
use jope_runtime::methods::registry;
/// The roster, input policy and recipes live in the appliance runtime's registry
/// (`jope_runtime::methods::registry`), so the panel and `jopedime` run one definition.
pub use jope_runtime::methods::registry::{RustMethod, INPUT_POLICY, INPUT_STAGE};
use jope_runtime::task::TaskKind;

/// The catalog task as the runtime's task contract; the panel admits three tasks.
fn kind(task: Task) -> Result<TaskKind> {
    match task {
        Task::Regression => Ok(TaskKind::Regression),
        Task::Binary => Ok(TaskKind::Binary),
        Task::Multiclass => Ok(TaskKind::Multiclass),
        other => anyhow::bail!("{} is outside the frozen panel", other.as_str()),
    }
}

/// Every executable product head compatible with the task, through the raw-table boundary.
/// Regression has 16 families and 17 recipes: non-RBF GPR has a distinct fitted implementation.
/// Binary has five families; multiclass has four. Engine defaults stay defaults (in particular,
/// multiclass's 64 transformed-feature budget); arbitrary hyperparameter sweeps are not implied.
pub fn rust_methods(task: Task) -> Result<Vec<RustMethod>> {
    registry::rust_methods(kind(task)?)
}

/// The fitted engine task is separate from the controller's primary metric. The engine's
/// multiclass task contract admits MacroF1; the panel scores its probability matrix with log loss.
pub fn task_spec(task: Task) -> Result<TaskSpec> {
    registry::task_spec(kind(task)?)
}

fn validate_classes(task: Task, classes: &[String]) -> Result<()> {
    // Outside the panel's tasks the runtime's timeseries rule gives the same two refusals.
    registry::validate_classes(kind(task).unwrap_or(TaskKind::Timeseries), classes)
}

/// This is the entire Rust fit-worker input. Scoring labels, row identities, panel roots, sealed
/// paths and controller state are deliberately absent. Class names are metadata, not row labels.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustFitRequest {
    pub task: Task,
    pub schema: DataSchema,
    pub classes: Vec<String>,
    pub train: RawTable,
    pub targets: Vec<f64>,
    pub predict: RawTable,
    pub method: RustMethod,
}

impl RustFitRequest {
    pub fn validate(&self) -> Result<()> {
        validate_classes(self.task, &self.classes)?;
        self.train.validate_against(&self.schema)?;
        self.predict.validate_against(&self.schema)?;
        ensure!(
            !self.train.rows.is_empty() && self.train.rows.len() == self.targets.len(),
            "fit targets differ from nonempty training rows"
        );
        ensure!(
            self.targets.iter().all(|v| v.is_finite()),
            "nonfinite target"
        );
        if self.task != Task::Regression {
            let mut seen = BTreeSet::new();
            for value in &self.targets {
                ensure!(
                    *value >= 0.0 && value.fract() == 0.0 && *value < self.classes.len() as f64,
                    "target is outside the declared class vocabulary"
                );
                seen.insert(*value as usize);
            }
            ensure!(
                seen.len() == self.classes.len(),
                "a declared class is absent from training"
            );
        }
        ensure!(
            rust_methods(self.task)?.contains(&self.method),
            "unregistered or altered Rust method"
        );
        validate_v3_genome(
            &GenomeIr::parse(&self.method.dsl)?,
            &task_spec(self.task)?,
            &self.schema,
        )
    }
}

/// Private fitted state, including preprocessing vocabulary, imputation, transformations,
/// calibration and every head's parameters. The caller binds these complete bytes by SHA256.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedRustArtifact {
    pub schema: String,
    pub raw_cell_encoding: String,
    pub input_policy: String,
    pub task: Task,
    pub classes: Vec<String>,
    pub method: RustMethod,
    pub model: BoundModel,
    /// Present exactly on `jope.panel-fitted-rust/v2`; absent (and unserialized) on v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitting_input_sha256: Option<String>,
}

impl FittedRustArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (self.schema == RUST_ARTIFACT && self.fitting_input_sha256.is_none())
                || (self.schema == RUST_ARTIFACT_V2
                    && self.fitting_input_sha256.as_deref().is_some_and(|sha| {
                        sha.len() == 64
                            && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                    })),
            "unsupported fitted Rust artifact"
        );
        ensure!(
            self.raw_cell_encoding == RAW_CELL_ENCODING,
            "raw-cell encoding changed"
        );
        ensure!(self.input_policy == INPUT_POLICY, "input policy changed");
        validate_classes(self.task, &self.classes)?;
        ensure!(
            self.model.task == task_spec(self.task)?,
            "fitted task differs from envelope"
        );
        ensure!(
            rust_methods(self.task)?.contains(&self.method),
            "fitted recipe is unregistered"
        );
        let canonical = GenomeIr::parse(&self.method.dsl)?.canonical();
        ensure!(canonical == self.method.dsl, "recipe is not canonical");
        ensure!(
            self.model.genome_sha256 == digest(canonical.as_bytes()),
            "fitted recipe differs from envelope"
        );
        self.model.validate_binding()
    }

    pub fn predict(&self, table: &RawTable) -> Result<Predictions> {
        self.validate()?;
        predict_bound(&self.model, table, &self.classes)
    }

    /// Replay the exact category/null boundary on new original input values.
    pub fn predict_raw(
        &self,
        features: &[jope_pipeline::experiment::Column],
        values: &[Vec<Option<String>>],
    ) -> Result<Predictions> {
        ensure!(
            features.len() == self.model.schema.columns.len()
                && features
                    .iter()
                    .zip(&self.model.schema.columns)
                    .all(|(actual, expected)| {
                        actual.name == expected.name
                            && match actual.kind {
                                jope_pipeline::experiment::ColumnKind::Numeric => {
                                    expected.kind == feat_math::advanced_v3::ColumnKind::Numeric
                                }
                                jope_pipeline::experiment::ColumnKind::Categorical => {
                                    expected.kind == feat_math::advanced_v3::ColumnKind::Categorical
                                }
                            }
                    }),
            "prediction feature schema differs from fitted schema"
        );
        self.predict(&raw_table(features, values)?)
    }

    pub fn bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    /// Validate the whole artifact before accepting its internal bindings. BoundModel's own
    /// binding covers preprocessing identities; this digest also covers learned predictor state.
    pub fn from_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self> {
        ensure!(
            digest(bytes) == expected_sha256,
            "fitted artifact checksum mismatch"
        );
        let artifact: Self =
            serde_json::from_slice(bytes).context("decode fitted Rust artifact")?;
        artifact.validate()?;
        Ok(artifact)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustFitResult {
    pub predictions: Predictions,
    pub artifact: FittedRustArtifact,
}

pub fn fit_rust(request: &RustFitRequest) -> Result<RustFitResult> {
    request.validate()?;
    let model = jope_runtime::methods::native::fit_native(
        &request.method,
        kind(request.task)?,
        &request.schema,
        &request.classes,
        &request.train,
        &request.targets,
    )?;
    let artifact = FittedRustArtifact {
        schema: RUST_ARTIFACT_V2.to_owned(),
        raw_cell_encoding: RAW_CELL_ENCODING.to_owned(),
        input_policy: INPUT_POLICY.to_owned(),
        task: request.task,
        classes: request.classes.clone(),
        method: request.method.clone(),
        model,
        // The exact input fitted above, digested by the fitter itself.
        fitting_input_sha256: Some(fitting_input_sha256(&request.train, &request.targets)?),
    };
    let predictions = artifact.predict(&request.predict)?;
    Ok(RustFitResult {
        predictions,
        artifact,
    })
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Raw nullable strings preserve literal categories such as "NA", "null" and whitespace.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterTraining {
    pub features: Vec<Vec<Option<String>>>,
    pub targets: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterPredict {
    pub features: Vec<Vec<Option<String>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterRequest {
    pub contract: String,
    pub task: Task,
    pub columns: Vec<ColumnSpec>,
    pub classes: Vec<String>,
    pub train: AdapterTraining,
    pub predict: AdapterPredict,
}

/// Relative path under the invocation's private output directory. A process exit or this claim
/// alone does not prove an artifact: the controller must read its bytes and verify this digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterArtifact {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub format: String,
    pub preprocessing_policy: String,
    pub reload_verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterResult {
    pub contract: String,
    pub engine_version: String,
    pub input_sha256: String,
    pub predictions: Predictions,
    pub fitted: AdapterArtifact,
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::advanced_v3::RowSemantics;

    fn request(task: Task, head: &str) -> RustFitRequest {
        let classes: Vec<String> = match task {
            Task::Regression => Vec::new(),
            Task::Binary => vec!["yes".into(), "no".into()],
            _ => vec!["zebra".into(), "alpha".into(), "middle".into()],
        };
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("value")],
            row_semantics: RowSemantics::Iid,
        };
        let rows: Vec<Vec<String>> = (0..18).map(|i| vec![i.to_string()]).collect();
        let targets = (0..18)
            .map(|i| {
                if classes.is_empty() {
                    (2 * i + 1) as f64
                } else {
                    (i % classes.len()) as f64
                }
            })
            .collect();
        RustFitRequest {
            task,
            schema,
            classes,
            train: RawTable::new(vec!["value".into()], rows).unwrap(),
            targets,
            predict: RawTable::new(
                vec!["value".into()],
                vec![vec!["1.5".into()], vec!["".into()]],
            )
            .unwrap(),
            method: rust_methods(task)
                .unwrap()
                .into_iter()
                .find(|m| m.family == head)
                .unwrap(),
        }
    }

    /// W3-T04: a fit names the exact fitting input it was fitted on (v2). A v1 artifact, as
    /// every artifact before v2, names none and still loads and predicts unchanged. The two
    /// forms cannot be mixed.
    #[test]
    fn fitted_artifacts_name_their_fitting_input_and_v1_stays_loadable() {
        for (task, head) in [
            (Task::Regression, "ridge"),
            (Task::Binary, "logit_ridge"),
            (Task::Multiclass, "multiclass_ridge"),
        ] {
            let request = request(task, head);
            let fitted = fit_rust(&request).unwrap();
            let artifact = fitted.artifact;
            let named = fitting_input_sha256(&request.train, &request.targets).unwrap();
            assert_eq!(artifact.schema, RUST_ARTIFACT_V2);
            assert_eq!(
                artifact.fitting_input_sha256.as_deref(),
                Some(named.as_str())
            );
            // Another fitting input (the same rows' targets in another order) is another digest.
            let mut targets = request.targets.clone();
            targets.swap(0, 1);
            assert_ne!(
                fitting_input_sha256(&request.train, &targets).unwrap(),
                named
            );
            let bytes = artifact.bytes().unwrap();
            let restored = FittedRustArtifact::from_bytes(&bytes, &digest(&bytes)).unwrap();
            assert_eq!(restored.fitting_input_sha256, artifact.fitting_input_sha256);

            // v1: unbound, byte-compatible with artifacts written before v2.
            let mut v1 = artifact.clone();
            v1.schema = RUST_ARTIFACT.into();
            v1.fitting_input_sha256 = None;
            let v1_bytes = v1.bytes().unwrap();
            assert!(!String::from_utf8_lossy(&v1_bytes).contains("fitting_input"));
            let v1 = FittedRustArtifact::from_bytes(&v1_bytes, &digest(&v1_bytes)).unwrap();
            assert_eq!(
                serde_json::to_vec(&v1.predict(&request.predict).unwrap()).unwrap(),
                serde_json::to_vec(&fitted.predictions).unwrap()
            );

            // Mixed forms are refused.
            let mut named_v1 = v1.clone();
            named_v1.fitting_input_sha256 = Some(named.clone());
            assert!(named_v1.validate().is_err());
            let mut unnamed_v2 = artifact.clone();
            unnamed_v2.fitting_input_sha256 = None;
            assert!(unnamed_v2.validate().is_err());
            let mut malformed = artifact.clone();
            malformed.fitting_input_sha256 = Some("../fitting-input".into());
            assert!(malformed.validate().is_err());
        }
    }

    /// The panel offers exactly the appliance's registry (`jope_runtime`), not a copy of it.
    #[test]
    fn the_roster_is_the_appliance_registry() {
        for (task, kind) in [
            (Task::Regression, TaskKind::Regression),
            (Task::Binary, TaskKind::Binary),
            (Task::Multiclass, TaskKind::Multiclass),
        ] {
            assert_eq!(
                rust_methods(task).unwrap(),
                registry::rust_methods(kind).unwrap()
            );
            assert_eq!(task_spec(task).unwrap(), registry::task_spec(kind).unwrap());
        }
        for task in [Task::MultiLabel, Task::Timeseries] {
            assert_eq!(
                rust_methods(task).unwrap_err().to_string(),
                format!("{} is outside the frozen panel", task.as_str())
            );
            assert_eq!(
                validate_classes(task, &[]).unwrap_err().to_string(),
                "class vocabulary is incompatible with task"
            );
        }
    }

    #[test]
    fn roster_covers_heads_and_records_aliases_without_duplicate_algorithms() {
        let regression = rust_methods(Task::Regression).unwrap();
        assert_eq!(regression.len(), 17);
        assert_eq!(
            regression
                .iter()
                .map(|m| &m.family)
                .collect::<BTreeSet<_>>()
                .len(),
            16
        );
        for head in [
            "ridge",
            "logistic",
            "ridge_gcv",
            "elasticnet",
            "histgbm",
            "gbm",
            "ensemble",
            "ensemble_ridge_gbm",
            "histgbm2",
            "krr",
            "gpr",
            "ksmooth",
            "hurdle2",
            "stack",
            "portfolio",
        ] {
            assert!(feat_math::heads::is_head_op(head));
            assert!(
                regression
                    .iter()
                    .any(|m| m.family == head || m.aliases.iter().any(|a| a == head)),
                "{head}"
            );
        }
        assert_eq!(rust_methods(Task::Binary).unwrap().len(), 5);
        assert_eq!(rust_methods(Task::Multiclass).unwrap().len(), 4);
        assert!(rust_methods(Task::Timeseries).is_err());
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            let mut req = request(
                task,
                match task {
                    Task::Regression => "ridge",
                    Task::Binary => "logit_ridge",
                    _ => "multiclass_ridge",
                },
            );
            for method in rust_methods(task).unwrap() {
                req.method = method;
                req.validate().unwrap();
            }
        }
    }

    #[test]
    fn fitted_state_reloads_with_prediction_and_original_class_order_parity() {
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            for method in rust_methods(task).unwrap() {
                let mut req = request(task, &method.family);
                req.method = method;
                // Every registered recipe executes; the compact portfolio uses its documented
                // n<20 ridge fallback on this bounded fixture. Portfolio composition has its
                // own engine proofs and is not qualified by this small-input smoke test.
                let result = fit_rust(&req)
                    .unwrap_or_else(|error| panic!("{} fit failed: {error:#}", req.method.id));
                let bytes = result.artifact.bytes().unwrap();
                let restored = FittedRustArtifact::from_bytes(&bytes, &digest(&bytes))
                    .unwrap_or_else(|error| panic!("{} reload failed: {error:#}", req.method.id));
                assert_eq!(
                    restored.predict(&req.predict).unwrap(),
                    result.predictions,
                    "{}",
                    req.method.id
                );
                if let Predictions::Probabilities { classes, .. } = result.predictions {
                    assert_eq!(classes, req.classes);
                }
                assert!(FittedRustArtifact::from_bytes(&bytes, &"0".repeat(64)).is_err());
                let mut stale = serde_json::to_value(restored).unwrap();
                stale["raw_cell_encoding"] = serde_json::json!("changed");
                let stale = serde_json::to_vec(&stale).unwrap();
                assert!(FittedRustArtifact::from_bytes(&stale, &digest(&stale)).is_err());
            }
        }
    }

    #[test]
    fn raw_prediction_refuses_changed_feature_kind_and_empty_class_labels() {
        use jope_pipeline::experiment::{Column, ColumnKind};
        let mut req = request(Task::Regression, "ridge");
        req.schema.columns[0].kind = feat_math::advanced_v3::ColumnKind::Categorical;
        let features = vec![Column {
            name: "value".into(),
            kind: ColumnKind::Categorical,
        }];
        let rows = vec![vec![Some("1".into())], vec![Some("2".into())], vec![None]];
        req.train = raw_table(&features, &rows).unwrap();
        req.targets = vec![1.0, 2.0, 3.0];
        req.predict = raw_table(&features, &rows).unwrap();
        let fit = fit_rust(&req).unwrap();
        assert_eq!(
            fit.artifact.predict_raw(&features, &rows).unwrap(),
            fit.predictions
        );
        let changed = vec![Column {
            name: "value".into(),
            kind: ColumnKind::Numeric,
        }];
        assert!(fit
            .artifact
            .predict_raw(&changed, &rows)
            .unwrap_err()
            .to_string()
            .contains("schema"));
        let mut binary = request(Task::Binary, "logit_ridge");
        binary.classes[0].clear();
        assert!(binary.validate().is_err());
    }

    #[test]
    fn worker_request_rejects_missing_class_and_has_no_scoring_labels() {
        let mut req = request(Task::Binary, "logit_ridge");
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value["predict"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            ["headers", "rows"]
        );
        let mut leaked = value;
        leaked["heldout_targets"] = serde_json::json!([0, 1]);
        assert!(serde_json::from_value::<RustFitRequest>(leaked).is_err());
        req.targets.fill(0.0);
        assert!(fit_rust(&req)
            .unwrap_err()
            .to_string()
            .contains("absent from training"));
    }
}
