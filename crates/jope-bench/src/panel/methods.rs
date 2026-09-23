//! Frozen-panel methods. Historical same-cut v1 methods and adapters are unchanged.
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    fit_table, validate_v3_genome, BoundModel, ColumnSpec, DataSchema, Objective, RawTable,
    TaskSpec,
};
use feat_math::GenomeIr;
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use super::data::{raw_table, RAW_CELL_ENCODING};
use super::score::{predict_bound, Predictions};

pub const RUST_ARTIFACT: &str = "jope.panel-fitted-rust/v1";
pub const ADAPTER_CONTRACT: &str = "jope.bench-adapter/v2";
pub const INPUT_POLICY: &str = "median-onehot256-indicator/v1";
/// No target-aware encodings: groups cannot leak through ordered target statistics.
pub const INPUT_STAGE: &str =
    "(input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256)";

/// A recipe is distinct from an algorithm family. Aliases never add screening slots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustMethod {
    pub id: String,
    pub family: String,
    pub configuration: String,
    pub aliases: Vec<String>,
    pub dsl: String,
}

fn method(
    head: &str,
    configuration: &str,
    parameters: &str,
    aliases: &[&str],
) -> Result<RustMethod> {
    let suffix = if configuration == "default" || configuration == "core_v1" {
        String::new()
    } else {
        format!(".{configuration}")
    };
    let stage = if parameters.is_empty() {
        head.to_owned()
    } else {
        format!("{head} {parameters}")
    };
    let genome = GenomeIr::parse(&format!("(pipe X {INPUT_STAGE} ({stage}))"))?;
    Ok(RustMethod {
        id: format!("rust.{head}{suffix}"),
        family: head.to_owned(),
        configuration: configuration.to_owned(),
        aliases: aliases.iter().map(|name| (*name).to_owned()).collect(),
        dsl: genome.canonical(),
    })
}

/// Every executable product head compatible with the task, through the raw-table boundary.
/// Regression has 16 families and 17 recipes: non-RBF GPR has a distinct fitted implementation.
/// Binary has five families; multiclass has four. Engine defaults stay defaults (in particular,
/// multiclass's 64 transformed-feature budget); arbitrary hyperparameter sweeps are not implied.
pub fn rust_methods(task: Task) -> Result<Vec<RustMethod>> {
    let mut out = Vec::new();
    match task {
        Task::Regression => {
            for (head, aliases) in [
                ("ridge", &["logistic"][..]),
                ("ridge_gcv", &[][..]),
                ("elasticnet", &[][..]),
                ("histgbm", &["gbm"][..]),
                ("histgbm2", &[][..]),
                ("ksmooth", &[][..]),
                ("ensemble", &["ensemble_ridge_gbm"][..]),
                ("krr", &[][..]),
                ("gpr", &[][..]),
                ("hurdle2", &[][..]),
                ("stack", &[][..]),
                ("tab_boost", &[][..]),
                ("extra_trees", &[][..]),
                ("gam_boost", &[][..]),
                ("pls", &[][..]),
            ] {
                out.push(method(head, "default", "", aliases)?);
            }
            out.push(method("portfolio", "core_v1", "preset=core", &[])?);
            out.push(method("gpr", "rbf_linear", "kernel=rbf+linear", &[])?);
        }
        Task::Binary => {
            out.push(method("logit_ridge", "default", "", &["logistic"])?);
            for head in ["tab_boost", "extra_trees", "gam_boost", "portfolio"] {
                out.push(method(head, "default", "", &[])?);
            }
        }
        Task::Multiclass => {
            for head in [
                "multiclass_ridge",
                "multiclass_ovr_logistic",
                "multiclass_diagonal_lda",
                "multiclass_nearest_centroid",
            ] {
                out.push(method(head, "default", "", &[])?);
            }
        }
        other => anyhow::bail!("{} is outside the frozen panel", other.as_str()),
    }
    Ok(out)
}

/// The fitted engine task is separate from the controller's primary metric. The engine's
/// multiclass task contract admits MacroF1; the panel scores its probability matrix with log loss.
pub fn task_spec(task: Task) -> Result<TaskSpec> {
    match task {
        Task::Regression => TaskSpec::regression(Objective::Rmse),
        Task::Binary => TaskSpec::binary(Objective::RocAuc),
        Task::Multiclass => TaskSpec::multiclass(Objective::MacroF1),
        other => anyhow::bail!("{} is outside the frozen panel", other.as_str()),
    }
}

fn validate_classes(task: Task, classes: &[String]) -> Result<()> {
    ensure!(
        classes.iter().collect::<BTreeSet<_>>().len() == classes.len()
            && classes.iter().all(|class| !class.is_empty()),
        "empty or duplicate class labels"
    );
    ensure!(
        match task {
            Task::Regression => classes.is_empty(),
            Task::Binary => classes.len() == 2,
            Task::Multiclass => classes.len() >= 3,
            _ => false,
        },
        "class vocabulary is incompatible with task"
    );
    Ok(())
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
}

impl FittedRustArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RUST_ARTIFACT,
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
    let model = fit_table(
        &GenomeIr::parse(&request.method.dsl)?,
        &task_spec(request.task)?,
        &request.schema,
        &request.train,
        &request.targets,
    )?;
    let artifact = FittedRustArtifact {
        schema: RUST_ARTIFACT.to_owned(),
        raw_cell_encoding: RAW_CELL_ENCODING.to_owned(),
        input_policy: INPUT_POLICY.to_owned(),
        task: request.task,
        classes: request.classes.clone(),
        method: request.method.clone(),
        model,
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
