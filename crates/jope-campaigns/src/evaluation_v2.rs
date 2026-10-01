//! Additive V2 evidence boundary for one explicit regression fold.
//!
//! This module deliberately owns no paths or artifact handles. Scientific
//! identity is separated from runtime observations so the same evaluation can
//! be replayed, rescored, cached, or moved to another executor without making
//! timing and topology part of the proof.

use anyhow::{bail, Context, Result};
use feat_math::{fit, GenomeIr, Rows};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;
use std::time::Instant;

const RUN_SCHEMA: &str = "JopeDimeRunContract.v2";
const PREDICTION_CONTEXT_SCHEMA: &str = "JopeDimePredictionContext.v2";
const FOLD_SCHEMA: &str = "JopeDimeExplicitFold.v1";
const TAPE_SCHEMA: &str = "JopeDimePredictionTape.v1";
const FITNESS_SCHEMA: &str = "JopeDimeRegressionFitness.v1";
const RECEIPT_SCHEMA: &str = "JopeDimeSemanticEvaluationReceipt.v2";
const OBSERVATION_SCHEMA: &str = "JopeDimeEvaluationObservation.v1";
const COMPATIBILITY_SHADOW_SCHEMA: &str = "JopeDimePredictionCompatibilityShadow.v1";
const NO_FORGETTING_ARCHIVE_SCHEMA: &str = "JopeDimeNoForgettingArchiveEntry.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluatorIdentityV1 {
    pub(crate) schema: String,
    pub(crate) evaluator_abi: String,
    pub(crate) implementation_sha256: String,
    pub(crate) dsl_abi_sha256: String,
    pub(crate) cargo_lock_sha256: String,
}

impl EvaluatorIdentityV1 {
    pub(crate) fn current() -> Self {
        let value = Self {
            schema: "JopeDimeEvaluatorIdentity.v1".into(),
            evaluator_abi: "prime-jain.native-feat-math-f64.evaluation-v2.v1".into(),
            implementation_sha256: current_implementation_sha256(),
            dsl_abi_sha256: digest_literal(
                "jope-dime/dsl-abi/v1",
                b"vendored-feat-math.GenomeIr+fit+Model::predict.v1",
            ),
            cargo_lock_sha256: hex::encode(Sha256::digest(include_bytes!("../../../Cargo.lock"))),
        };
        value
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_schema(&self.schema, "JopeDimeEvaluatorIdentity.v1", "evaluator")?;
        require_schema(
            &self.evaluator_abi,
            "prime-jain.native-feat-math-f64.evaluation-v2.v1",
            "evaluator ABI",
        )?;
        validate_sha256("evaluator implementation", &self.implementation_sha256)?;
        validate_sha256("DSL ABI", &self.dsl_abi_sha256)?;
        validate_sha256("Cargo.lock", &self.cargo_lock_sha256)?;
        let expected_dsl = digest_literal(
            "jope-dime/dsl-abi/v1",
            b"vendored-feat-math.GenomeIr+fit+Model::predict.v1",
        );
        let expected_lock = hex::encode(Sha256::digest(include_bytes!("../../../Cargo.lock")));
        if self.dsl_abi_sha256 != expected_dsl {
            bail!("evaluator DSL ABI identity does not match this build");
        }
        let is_current = self.implementation_sha256 == current_implementation_sha256()
            && self.cargo_lock_sha256 == expected_lock;
        // ADR-0004: a receipt written by an earlier build stays valid under the identity it recorded,
        // provided that identity is in the committed table of known evaluator implementations. Only
        // *producing* a receipt requires the current identity (see `current()` callers).
        if !is_current
            && !known_evaluator_identity(&self.implementation_sha256, &self.cargo_lock_sha256)
        {
            bail!(
                "evaluator implementation/Cargo.lock identity matches neither this build nor a known historical \
                 evaluator (crates/jope-campaigns/known-evaluator-identities.json)"
            );
        }
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/evaluator-identity/v1");
        hash.str("schema", &self.schema);
        hash.str("evaluator_abi", &self.evaluator_abi);
        hash.str("implementation_sha256", &self.implementation_sha256);
        hash.str("dsl_abi_sha256", &self.dsl_abi_sha256);
        hash.str("cargo_lock_sha256", &self.cargo_lock_sha256);
        Ok(hash.finish())
    }
}

/// Known historical evaluator identities (ADR-0004). Each entry is an
/// `{implementation_sha256, cargo_lock_sha256}` pair a shipped build reported in
/// `JopeDimeEvaluatorIdentity.v1` receipts; the table is append-only and reviewed.
fn known_evaluator_identity(implementation_sha256: &str, cargo_lock_sha256: &str) -> bool {
    static TABLE: OnceLock<Vec<(String, String)>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let raw: serde_json::Value =
            serde_json::from_str(include_str!("../known-evaluator-identities.json"))
                .expect("known-evaluator-identities.json parses");
        raw["identities"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        Some((
                            row["implementation_sha256"].as_str()?.to_owned(),
                            row["cargo_lock_sha256"].as_str()?.to_owned(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    table.iter().any(|(implementation, lock)| {
        implementation == implementation_sha256 && lock == cargo_lock_sha256
    })
}

fn current_implementation_sha256() -> String {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(compute_implementation_sha256).clone()
}

fn compute_implementation_sha256() -> String {
    let mut hash = SemanticHasher::new("jope-dime/evaluator-implementation/v1");
    for (name, bytes) in [
        (
            "src/evaluation_v2.rs",
            include_bytes!("evaluation_v2.rs").as_slice(),
        ),
        (
            "src/commands/exact_cv_v2.rs",
            include_bytes!("commands/exact_cv_v2.rs").as_slice(),
        ),
        ("src/data.rs", include_bytes!("data.rs").as_slice()),
        (
            "src/commands/exact_cv.rs",
            include_bytes!("commands/exact_cv.rs").as_slice(),
        ),
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        (
            "Cargo.lock",
            include_bytes!("../../../Cargo.lock").as_slice(),
        ),
        (
            "vendor/feat-math/Cargo.toml",
            include_bytes!("../../feat-math/Cargo.toml").as_slice(),
        ),
        (
            "vendor/feat-math/src/advanced_v3.rs",
            include_bytes!("../../feat-math/src/advanced_v3.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/boost2.rs",
            include_bytes!("../../feat-math/src/boost2.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/credit.rs",
            include_bytes!("../../feat-math/src/credit.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/dsl.rs",
            include_bytes!("../../feat-math/src/dsl.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/emit_py.rs",
            include_bytes!("../../feat-math/src/emit_py.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/emit_rs.rs",
            include_bytes!("../../feat-math/src/emit_rs.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/evolve.rs",
            include_bytes!("../../feat-math/src/evolve.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/explain.rs",
            include_bytes!("../../feat-math/src/explain.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/folds.rs",
            include_bytes!("../../feat-math/src/folds.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/full_rust.rs",
            include_bytes!("../../feat-math/src/full_rust.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/genome.rs",
            include_bytes!("../../feat-math/src/genome.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/gpr.rs",
            include_bytes!("../../feat-math/src/gpr.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/heads.rs",
            include_bytes!("../../feat-math/src/heads.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/krr.rs",
            include_bytes!("../../feat-math/src/krr.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/ksmooth.rs",
            include_bytes!("../../feat-math/src/ksmooth.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/lib.rs",
            include_bytes!("../../feat-math/src/lib.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/math_expr.rs",
            include_bytes!("../../feat-math/src/math_expr.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/mathexpr.rs",
            include_bytes!("../../feat-math/src/mathexpr.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/members.rs",
            include_bytes!("../../feat-math/src/members.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/metrics.rs",
            include_bytes!("../../feat-math/src/metrics.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/model.rs",
            include_bytes!("../../feat-math/src/model.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/ops.rs",
            include_bytes!("../../feat-math/src/ops.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/portfolio.rs",
            include_bytes!("../../feat-math/src/portfolio.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/rng.rs",
            include_bytes!("../../feat-math/src/rng.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/router_policy.rs",
            include_bytes!("../../feat-math/src/router_policy.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/spec.rs",
            include_bytes!("../../feat-math/src/spec.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/stack.rs",
            include_bytes!("../../feat-math/src/stack.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/stage_cache.rs",
            include_bytes!("../../feat-math/src/stage_cache.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/target_transform.rs",
            include_bytes!("../../feat-math/src/target_transform.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/tokens.rs",
            include_bytes!("../../feat-math/src/tokens.rs").as_slice(),
        ),
        (
            "vendor/feat-math/src/transpile.rs",
            include_bytes!("../../feat-math/src/transpile.rs").as_slice(),
        ),
    ] {
        hash.bytes(name, bytes);
    }
    hash.finish()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FitSpecV1 {
    pub(crate) schema: String,
    pub(crate) fit_abi: String,
}

impl FitSpecV1 {
    pub(crate) fn legacy_strict() -> Self {
        Self {
            schema: "JopeDimeFitSpec.v1".into(),
            fit_abi: "vendored-feat-math.fit.v1".into(),
        }
    }

    fn validate(&self) -> Result<()> {
        require_schema(&self.schema, "JopeDimeFitSpec.v1", "fit spec")?;
        require_schema(&self.fit_abi, "vendored-feat-math.fit.v1", "fit ABI")
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/fit-spec/v1");
        hash.str("schema", &self.schema);
        hash.str("fit_abi", &self.fit_abi);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConditioningIdentityV1 {
    pub(crate) schema: String,
    pub(crate) kind: String,
    pub(crate) identity_sha256: String,
}

impl ConditioningIdentityV1 {
    pub(crate) fn none() -> Self {
        Self {
            schema: "JopeDimeConditioningIdentity.v1".into(),
            kind: "none-v1".into(),
            identity_sha256: digest_literal("jope-dime/conditioning/v1", b"none-v1"),
        }
    }

    fn validate(&self) -> Result<()> {
        require_schema(
            &self.schema,
            "JopeDimeConditioningIdentity.v1",
            "conditioning",
        )?;
        require_schema(&self.kind, "none-v1", "conditioning kind")?;
        validate_sha256("conditioning identity", &self.identity_sha256)?;
        if *self != Self::none() {
            bail!("this proof slice admits only the canonical none-v1 conditioning identity");
        }
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/conditioning-identity/v1");
        hash.str("schema", &self.schema);
        hash.str("kind", &self.kind);
        hash.str("identity_sha256", &self.identity_sha256);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MathPolicyIdV1 {
    pub(crate) schema: String,
    pub(crate) dtype: String,
    pub(crate) arithmetic: String,
}

impl MathPolicyIdV1 {
    pub(crate) fn strict_scalar_f64() -> Self {
        Self {
            schema: "JopeDimeMathPolicy.v1".into(),
            dtype: "f64".into(),
            arithmetic: "strict-scalar-no-fast-math-v1".into(),
        }
    }

    fn validate(&self) -> Result<()> {
        require_schema(&self.schema, "JopeDimeMathPolicy.v1", "math policy")?;
        require_schema(&self.dtype, "f64", "math dtype")?;
        require_schema(
            &self.arithmetic,
            "strict-scalar-no-fast-math-v1",
            "arithmetic policy",
        )
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/math-policy/v1");
        hash.str("schema", &self.schema);
        hash.str("dtype", &self.dtype);
        hash.str("arithmetic", &self.arithmetic);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ObjectiveIdentityV1 {
    pub(crate) schema: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) config_sha256: String,
}

impl ObjectiveIdentityV1 {
    pub(crate) fn legacy_robust_r2() -> Self {
        Self::known("legacy-robust-r2", "1")
    }

    pub(crate) fn negative_rmse() -> Self {
        Self::known("negative-rmse", "1")
    }

    pub(crate) fn from_policy(policy: &str) -> Result<Self> {
        match policy {
            "legacy-robust-r2-v1" => Ok(Self::legacy_robust_r2()),
            "negative-rmse-v1" => Ok(Self::negative_rmse()),
            _ => bail!("unsupported regression objective policy {policy}"),
        }
    }

    fn known(name: &str, version: &str) -> Self {
        Self {
            schema: "JopeDimeRegressionObjective.v1".into(),
            name: name.into(),
            version: version.into(),
            config_sha256: digest_literal(
                "jope-dime/regression-objective-config/v1",
                format!("{name}-v{version}").as_bytes(),
            ),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_schema(&self.schema, "JopeDimeRegressionObjective.v1", "objective")?;
        validate_sha256("objective configuration", &self.config_sha256)?;
        if *self != Self::legacy_robust_r2() && *self != Self::negative_rmse() {
            bail!("objective identity is not registered in this evaluator build");
        }
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/objective-identity/v1");
        hash.str("schema", &self.schema);
        hash.str("name", &self.name);
        hash.str("version", &self.version);
        hash.str("config_sha256", &self.config_sha256);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PredictionContextV2 {
    pub(crate) schema: String,
    pub(crate) task: String,
    pub(crate) training_feature_view_sha256: String,
    pub(crate) fit_spec: FitSpecV1,
    pub(crate) conditioning: ConditioningIdentityV1,
    pub(crate) evaluator: EvaluatorIdentityV1,
    pub(crate) math_policy: MathPolicyIdV1,
}

impl PredictionContextV2 {
    fn validate(&self) -> Result<()> {
        require_schema(
            &self.schema,
            PREDICTION_CONTEXT_SCHEMA,
            "prediction context",
        )?;
        require_schema(&self.task, "regression", "task")?;
        validate_sha256("training feature view", &self.training_feature_view_sha256)?;
        self.fit_spec.validate()?;
        self.conditioning.validate()?;
        self.evaluator.validate()?;
        self.math_policy.validate()
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/prediction-context/v2");
        hash.str("schema", &self.schema);
        hash.str("task", &self.task);
        hash.str(
            "training_feature_view_sha256",
            &self.training_feature_view_sha256,
        );
        hash.str("fit_spec_sha256", &self.fit_spec.digest()?);
        hash.str("conditioning_sha256", &self.conditioning.digest()?);
        hash.str("evaluator_sha256", &self.evaluator.digest()?);
        hash.str("math_policy_sha256", &self.math_policy.digest()?);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunContractV2 {
    pub(crate) schema: String,
    pub(crate) prediction: PredictionContextV2,
    pub(crate) objective: ObjectiveIdentityV1,
}

impl RunContractV2 {
    pub(crate) fn strict_legacy(
        training_feature_view_sha256: String,
        evaluator: EvaluatorIdentityV1,
        objective: ObjectiveIdentityV1,
    ) -> Result<Self> {
        let value = Self {
            schema: RUN_SCHEMA.into(),
            prediction: PredictionContextV2 {
                schema: PREDICTION_CONTEXT_SCHEMA.into(),
                task: "regression".into(),
                training_feature_view_sha256,
                fit_spec: FitSpecV1::legacy_strict(),
                conditioning: ConditioningIdentityV1::none(),
                evaluator,
                math_policy: MathPolicyIdV1::strict_scalar_f64(),
            },
            objective,
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_schema(&self.schema, RUN_SCHEMA, "run contract")?;
        self.prediction.validate()?;
        self.objective.validate()
    }

    fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut hash = SemanticHasher::new("jope-dime/run-contract/v2");
        hash.str("schema", &self.schema);
        hash.str("prediction_sha256", &self.prediction.digest()?);
        hash.str("objective_sha256", &self.objective.digest()?);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrainOnlyRegressionView<'a> {
    rows: &'a Rows,
    targets: &'a [f64],
    stable_row_ids: &'a [String],
    feature_view_sha256: String,
}

impl<'a> TrainOnlyRegressionView<'a> {
    pub(crate) fn new(
        rows: &'a Rows,
        targets: &'a [f64],
        stable_row_ids: &'a [String],
        feature_width: usize,
    ) -> Result<Self> {
        if rows.is_empty() || rows.len() != targets.len() || rows.len() != stable_row_ids.len() {
            bail!("training rows, targets, and stable row IDs must be equal and non-empty");
        }
        if feature_width == 0
            || rows.iter().any(|row| row.len() != feature_width)
            || rows
                .iter()
                .flatten()
                .chain(targets)
                .any(|value| !value.is_finite())
        {
            bail!("training view must be rectangular, non-empty-width, and finite");
        }
        let mut seen = HashSet::with_capacity(stable_row_ids.len());
        for row_id in stable_row_ids {
            validate_stable_row_id(row_id)?;
            if !seen.insert(row_id) {
                bail!("stable row IDs must be unique");
            }
        }
        let feature_view_sha256 = hash_feature_view(rows, stable_row_ids, feature_width);
        Ok(Self {
            rows,
            targets,
            stable_row_ids,
            feature_view_sha256,
        })
    }

    pub(crate) fn feature_view_sha256(&self) -> &str {
        &self.feature_view_sha256
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExplicitFoldV1 {
    pub(crate) schema: String,
    pub(crate) fold_id: u32,
    pub(crate) train_rows: Vec<usize>,
    pub(crate) validation_rows: Vec<usize>,
    pub(crate) train_row_ids: Vec<String>,
    pub(crate) validation_row_ids: Vec<String>,
    pub(crate) train_rows_sha256: String,
    pub(crate) validation_rows_sha256: String,
}

impl ExplicitFoldV1 {
    pub(crate) fn new(
        fold_id: u32,
        train_rows: Vec<usize>,
        validation_rows: Vec<usize>,
        view: &TrainOnlyRegressionView<'_>,
    ) -> Result<Self> {
        let train_row_ids = row_ids_for(&train_rows, view)?;
        let validation_row_ids = row_ids_for(&validation_rows, view)?;
        let value = Self {
            schema: FOLD_SCHEMA.into(),
            fold_id,
            train_rows_sha256: hash_bound_rows(
                "jope-dime/explicit-fold/train-rows/v1",
                &train_rows,
                &train_row_ids,
            ),
            validation_rows_sha256: hash_bound_rows(
                "jope-dime/explicit-fold/validation-rows/v1",
                &validation_rows,
                &validation_row_ids,
            ),
            train_rows,
            validation_rows,
            train_row_ids,
            validation_row_ids,
        };
        value.validate(view)?;
        Ok(value)
    }

    fn validate(&self, view: &TrainOnlyRegressionView<'_>) -> Result<()> {
        require_schema(&self.schema, FOLD_SCHEMA, "explicit fold")?;
        if self.train_rows.len() < 8 || self.validation_rows.is_empty() {
            bail!("explicit fold needs at least eight training rows and one validation row");
        }
        validate_ordered_unique_rows("training", &self.train_rows, view.len())?;
        validate_ordered_unique_rows("validation", &self.validation_rows, view.len())?;
        let train_set = self.train_rows.iter().copied().collect::<HashSet<_>>();
        if self
            .validation_rows
            .iter()
            .any(|row| train_set.contains(row))
        {
            bail!("training and validation rows overlap");
        }
        let expected_train_ids = row_ids_for(&self.train_rows, view)?;
        let expected_validation_ids = row_ids_for(&self.validation_rows, view)?;
        if self.train_row_ids != expected_train_ids
            || self.validation_row_ids != expected_validation_ids
        {
            bail!("fold row IDs do not match the training view at the declared indices");
        }
        let expected_train_sha = hash_bound_rows(
            "jope-dime/explicit-fold/train-rows/v1",
            &self.train_rows,
            &self.train_row_ids,
        );
        let expected_validation_sha = hash_bound_rows(
            "jope-dime/explicit-fold/validation-rows/v1",
            &self.validation_rows,
            &self.validation_row_ids,
        );
        if self.train_rows_sha256 != expected_train_sha
            || self.validation_rows_sha256 != expected_validation_sha
        {
            bail!("fold row digest mismatch");
        }
        Ok(())
    }

    fn digest(&self, view: &TrainOnlyRegressionView<'_>) -> Result<String> {
        self.validate(view)?;
        let mut hash = SemanticHasher::new("jope-dime/explicit-fold/v1");
        hash.str("schema", &self.schema);
        hash.u64("fold_id", self.fold_id as u64);
        hash.str("train_rows_sha256", &self.train_rows_sha256);
        hash.str("validation_rows_sha256", &self.validation_rows_sha256);
        Ok(hash.finish())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct EvaluationKeyV2(pub(crate) String);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ScoreKeyV1(pub(crate) String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PredictionTapeV1 {
    pub(crate) schema: String,
    pub(crate) fold_id: u32,
    pub(crate) evaluation_key: EvaluationKeyV2,
    pub(crate) validation_row_ids: Vec<String>,
    #[serde(with = "hex_u64_vec")]
    pub(crate) truth_bits: Vec<u64>,
    #[serde(with = "hex_u64_vec")]
    pub(crate) prediction_bits: Vec<u64>,
    pub(crate) validity: Vec<bool>,
    pub(crate) fit_binding_sha256: String,
    pub(crate) tape_sha256: String,
}

impl PredictionTapeV1 {
    fn new(
        fold: &ExplicitFoldV1,
        evaluation_key: EvaluationKeyV2,
        truth: &[f64],
        predictions: &[f64],
        fit_binding_sha256: String,
    ) -> Result<Self> {
        if truth.is_empty()
            || truth.len() != predictions.len()
            || truth.len() != fold.validation_row_ids.len()
            || truth
                .iter()
                .chain(predictions)
                .any(|value| !value.is_finite())
        {
            bail!("prediction tape requires equal, non-empty, finite rows");
        }
        validate_sha256("fit binding", &fit_binding_sha256)?;
        let truth_bits = truth
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let prediction_bits = predictions
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let validity = vec![true; truth.len()];
        let tape_sha256 = hash_tape_parts(
            fold.fold_id,
            &evaluation_key,
            &fold.validation_row_ids,
            &truth_bits,
            &prediction_bits,
            &validity,
            &fit_binding_sha256,
        );
        let tape = Self {
            schema: TAPE_SCHEMA.into(),
            fold_id: fold.fold_id,
            evaluation_key,
            validation_row_ids: fold.validation_row_ids.clone(),
            truth_bits,
            prediction_bits,
            validity,
            fit_binding_sha256,
            tape_sha256,
        };
        tape.validate()?;
        Ok(tape)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_schema(&self.schema, TAPE_SCHEMA, "prediction tape")?;
        let n = self.validation_row_ids.len();
        if n == 0
            || self.truth_bits.len() != n
            || self.prediction_bits.len() != n
            || self.validity.len() != n
            || self.validity.iter().any(|valid| !valid)
        {
            bail!("prediction tape arrays must have equal nonzero length and all rows valid");
        }
        let mut seen = HashSet::with_capacity(n);
        for row in &self.validation_row_ids {
            validate_stable_row_id(row)?;
            if !seen.insert(row) {
                bail!("prediction tape row IDs must be unique");
            }
        }
        if self
            .truth_bits
            .iter()
            .chain(&self.prediction_bits)
            .map(|bits| f64::from_bits(*bits))
            .any(|value| !value.is_finite())
        {
            bail!("prediction tape contains non-finite values");
        }
        validate_sha256("evaluation key", &self.evaluation_key.0)?;
        validate_sha256("fit binding", &self.fit_binding_sha256)?;
        let expected = hash_tape_parts(
            self.fold_id,
            &self.evaluation_key,
            &self.validation_row_ids,
            &self.truth_bits,
            &self.prediction_bits,
            &self.validity,
            &self.fit_binding_sha256,
        );
        if self.tape_sha256 != expected {
            bail!("prediction tape digest mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RegressionFitnessV1 {
    pub(crate) schema: String,
    #[serde(with = "hex_u64")]
    pub(crate) pooled_rmse_bits: u64,
    #[serde(with = "hex_u64")]
    pub(crate) pooled_r2_bits: u64,
    #[serde(with = "hex_u64")]
    pub(crate) robust_fitness_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticEvaluationReceiptV2 {
    pub(crate) schema: String,
    pub(crate) run_contract_sha256: String,
    pub(crate) candidate_semantic_sha256: String,
    pub(crate) fold_sha256: String,
    pub(crate) evaluation_key: EvaluationKeyV2,
    pub(crate) prediction_tape: PredictionTapeV1,
    pub(crate) objective_sha256: String,
    pub(crate) score_key: ScoreKeyV1,
    pub(crate) fitness: RegressionFitnessV1,
    pub(crate) semantic_receipt_sha256: String,
}

impl SemanticEvaluationReceiptV2 {
    pub(crate) fn validate_against(&self, contract: &RunContractV2) -> Result<()> {
        require_schema(&self.schema, RECEIPT_SCHEMA, "semantic receipt")?;
        contract.validate()?;
        self.prediction_tape.validate()?;
        require_schema(&self.fitness.schema, FITNESS_SCHEMA, "regression fitness")?;
        for (label, bits) in [
            ("pooled RMSE", self.fitness.pooled_rmse_bits),
            ("pooled R2", self.fitness.pooled_r2_bits),
            ("robust fitness", self.fitness.robust_fitness_bits),
        ] {
            if !f64::from_bits(bits).is_finite() {
                bail!("{label} is non-finite");
            }
        }
        validate_sha256("run contract", &self.run_contract_sha256)?;
        validate_sha256("candidate semantic", &self.candidate_semantic_sha256)?;
        validate_sha256("fold", &self.fold_sha256)?;
        validate_sha256("evaluation key", &self.evaluation_key.0)?;
        validate_sha256("objective", &self.objective_sha256)?;
        validate_sha256("score key", &self.score_key.0)?;
        validate_sha256("semantic receipt", &self.semantic_receipt_sha256)?;
        if self.run_contract_sha256 != contract.digest()? {
            bail!("semantic receipt run contract mismatch");
        }
        if self.evaluation_key != self.prediction_tape.evaluation_key {
            bail!("outer and tape evaluation keys differ");
        }
        let expected_objective = contract.objective.digest()?;
        if self.objective_sha256 != expected_objective {
            bail!("semantic receipt objective mismatch");
        }
        let expected_score = score_key(&self.prediction_tape.tape_sha256, &expected_objective);
        if self.score_key != expected_score {
            bail!("semantic receipt score key mismatch");
        }
        let expected_receipt = hash_receipt_parts(
            &self.run_contract_sha256,
            &self.candidate_semantic_sha256,
            &self.fold_sha256,
            &self.evaluation_key,
            &self.prediction_tape.tape_sha256,
            &self.objective_sha256,
            &self.score_key,
            &self.fitness,
        );
        if self.semantic_receipt_sha256 != expected_receipt {
            bail!("semantic receipt digest mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationObservationV1 {
    pub(crate) schema: String,
    pub(crate) semantic_receipt_sha256: String,
    pub(crate) cache_hit: bool,
    pub(crate) fit_nanos: u64,
    pub(crate) predict_nanos: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EvaluationOutcomeV2 {
    pub(crate) semantic: SemanticEvaluationReceiptV2,
    pub(crate) observation: EvaluationObservationV1,
}

/// Scientific compatibility around the frozen V1 tape. Runtime observations
/// are deliberately absent: hardware, worker count, and elapsed time cannot
/// affect selection or archive replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PredictionCompatibilityShadowV1 {
    pub(crate) schema: String,
    pub(crate) frozen_evaluator_sha256: String,
    pub(crate) baseline_semantic_receipt_sha256: String,
    pub(crate) challenger_semantic_receipt_sha256: String,
    pub(crate) fold_sha256: String,
    pub(crate) common_rows_sha256: String,
    pub(crate) common_rows: usize,
    pub(crate) objective_rescore_only: bool,
    pub(crate) shadow_sha256: String,
}

impl PredictionCompatibilityShadowV1 {
    pub(crate) fn build(
        baseline: &SemanticEvaluationReceiptV2,
        baseline_contract: &RunContractV2,
        challenger: &SemanticEvaluationReceiptV2,
        challenger_contract: &RunContractV2,
    ) -> Result<Self> {
        baseline.validate_against(baseline_contract)?;
        challenger.validate_against(challenger_contract)?;
        let baseline_evaluator = baseline_contract.prediction.evaluator.digest()?;
        let challenger_evaluator = challenger_contract.prediction.evaluator.digest()?;
        if baseline_evaluator != challenger_evaluator {
            bail!("prediction compatibility shadow evaluator drift");
        }
        if baseline.fold_sha256 != challenger.fold_sha256
            || baseline.prediction_tape.validation_row_ids
                != challenger.prediction_tape.validation_row_ids
            || baseline.prediction_tape.truth_bits != challenger.prediction_tape.truth_bits
        {
            bail!("prediction compatibility shadow requires common fold rows and truth");
        }
        let mut common = SemanticHasher::new("jope-dime/prediction-compatibility-rows/v1");
        common.str("fold_sha256", &baseline.fold_sha256);
        for (row, truth) in baseline
            .prediction_tape
            .validation_row_ids
            .iter()
            .zip(&baseline.prediction_tape.truth_bits)
        {
            common.str("row", row);
            common.u64("truth_bits", *truth);
        }
        let common_rows_sha256 = common.finish();
        let objective_rescore_only = baseline.candidate_semantic_sha256
            == challenger.candidate_semantic_sha256
            && baseline.prediction_tape.prediction_bits
                == challenger.prediction_tape.prediction_bits
            && baseline.evaluation_key == challenger.evaluation_key
            && baseline.objective_sha256 != challenger.objective_sha256;
        let mut identity = SemanticHasher::new("jope-dime/prediction-compatibility-shadow/v1");
        identity.str("frozen_evaluator_sha256", &baseline_evaluator);
        identity.str(
            "baseline_semantic_receipt_sha256",
            &baseline.semantic_receipt_sha256,
        );
        identity.str(
            "challenger_semantic_receipt_sha256",
            &challenger.semantic_receipt_sha256,
        );
        identity.str("fold_sha256", &baseline.fold_sha256);
        identity.str("common_rows_sha256", &common_rows_sha256);
        identity.u64(
            "common_rows",
            baseline.prediction_tape.validation_row_ids.len() as u64,
        );
        identity.u64("objective_rescore_only", objective_rescore_only as u64);
        Ok(Self {
            schema: COMPATIBILITY_SHADOW_SCHEMA.into(),
            frozen_evaluator_sha256: baseline_evaluator,
            baseline_semantic_receipt_sha256: baseline.semantic_receipt_sha256.clone(),
            challenger_semantic_receipt_sha256: challenger.semantic_receipt_sha256.clone(),
            fold_sha256: baseline.fold_sha256.clone(),
            common_rows_sha256,
            common_rows: baseline.prediction_tape.validation_row_ids.len(),
            objective_rescore_only,
            shadow_sha256: identity.finish(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NoForgettingArchiveEntryV1 {
    pub(crate) schema: String,
    pub(crate) candidate_semantic_sha256: String,
    /// Content-addressed common-evidence slice -> bounded scientific score bits.
    pub(crate) confirmed_evidence: BTreeMap<String, u64>,
    pub(crate) compatibility_shadow_sha256: String,
}

impl NoForgettingArchiveEntryV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        require_schema(
            &self.schema,
            NO_FORGETTING_ARCHIVE_SCHEMA,
            "no-forgetting archive",
        )?;
        validate_sha256("archive candidate", &self.candidate_semantic_sha256)?;
        validate_sha256(
            "archive compatibility shadow",
            &self.compatibility_shadow_sha256,
        )?;
        if self.confirmed_evidence.is_empty() {
            bail!("no-forgetting archive requires confirmed evidence");
        }
        for (evidence, score) in &self.confirmed_evidence {
            validate_sha256("archive evidence", evidence)?;
            if !f64::from_bits(*score).is_finite() {
                bail!("no-forgetting archive contains non-finite scientific score");
            }
        }
        Ok(())
    }
}

/// Replace only on a strict evidence superset with no regression on any
/// previously confirmed slice. Equal evidence, partial overlap, and hardware
/// diagnostics can never evict an archive member.
pub(crate) fn replace_no_forgetting_archive(
    current: &NoForgettingArchiveEntryV1,
    challenger: NoForgettingArchiveEntryV1,
) -> Result<NoForgettingArchiveEntryV1> {
    current.validate()?;
    challenger.validate()?;
    if challenger.confirmed_evidence.len() <= current.confirmed_evidence.len()
        || !current
            .confirmed_evidence
            .keys()
            .all(|evidence| challenger.confirmed_evidence.contains_key(evidence))
    {
        bail!("archive replacement requires a strict confirmed-evidence superset");
    }
    for (evidence, current_score) in &current.confirmed_evidence {
        let challenger_score = challenger
            .confirmed_evidence
            .get(evidence)
            .context("strict evidence superset changed during validation")?;
        if f64::from_bits(*challenger_score) < f64::from_bits(*current_score) {
            bail!("archive replacement regresses previously confirmed evidence");
        }
    }
    Ok(challenger)
}

/// Campaign-scoped prediction reuse for the next V2 orchestration slice.
/// The one-shot proof command is intentionally stateless.
#[allow(dead_code)]
pub(crate) struct EvaluationMemoV2 {
    entries: BTreeMap<EvaluationKeyV2, CachedPredictionV2>,
    max_entries: usize,
}

impl Default for EvaluationMemoV2 {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            max_entries: 1_024,
        }
    }
}

#[allow(dead_code)]
impl EvaluationMemoV2 {
    pub(crate) fn evaluate_regression_fold(
        &mut self,
        view: &TrainOnlyRegressionView<'_>,
        contract: &RunContractV2,
        fold: &ExplicitFoldV1,
        canonical_genome: &str,
    ) -> Result<EvaluationOutcomeV2> {
        let prepared = prepare_evaluation(view, contract, fold, canonical_genome)?;
        if let Some(cached) = self.entries.get(&prepared.evaluation_key) {
            let predictions = cached
                .prediction_bits
                .iter()
                .map(|bits| f64::from_bits(*bits))
                .collect::<Vec<_>>();
            let semantic = build_semantic_receipt(
                view,
                contract,
                fold,
                &cached.candidate_semantic_sha256,
                &cached.fold_sha256,
                cached.evaluation_key.clone(),
                &predictions,
                cached.fit_binding_sha256.clone(),
            )?;
            return Ok(EvaluationOutcomeV2 {
                observation: EvaluationObservationV1 {
                    schema: OBSERVATION_SCHEMA.into(),
                    semantic_receipt_sha256: semantic.semantic_receipt_sha256.clone(),
                    cache_hit: true,
                    fit_nanos: 0,
                    predict_nanos: 0,
                },
                semantic,
            });
        }
        let outcome = execute_prepared(view, contract, fold, prepared)?;
        if self.entries.len() == self.max_entries {
            let smallest_key = self.entries.keys().next().cloned();
            if let Some(key) = smallest_key {
                self.entries.remove(&key);
            }
        }
        self.entries.insert(
            outcome.semantic.evaluation_key.clone(),
            CachedPredictionV2 {
                candidate_semantic_sha256: outcome.semantic.candidate_semantic_sha256.clone(),
                fold_sha256: outcome.semantic.fold_sha256.clone(),
                evaluation_key: outcome.semantic.evaluation_key.clone(),
                prediction_bits: outcome.semantic.prediction_tape.prediction_bits.clone(),
                fit_binding_sha256: outcome.semantic.prediction_tape.fit_binding_sha256.clone(),
            },
        );
        Ok(outcome)
    }
}

#[allow(dead_code)]
struct CachedPredictionV2 {
    candidate_semantic_sha256: String,
    fold_sha256: String,
    evaluation_key: EvaluationKeyV2,
    prediction_bits: Vec<u64>,
    fit_binding_sha256: String,
}

pub(crate) fn evaluate_regression_fold_v2(
    view: &TrainOnlyRegressionView<'_>,
    contract: &RunContractV2,
    fold: &ExplicitFoldV1,
    canonical_genome: &str,
) -> Result<EvaluationOutcomeV2> {
    let prepared = prepare_evaluation(view, contract, fold, canonical_genome)?;
    execute_prepared(view, contract, fold, prepared)
}

struct PreparedEvaluation {
    genome: GenomeIr,
    candidate_semantic_sha256: String,
    evaluation_key: EvaluationKeyV2,
    fold_sha256: String,
    training_target_sha256: String,
}

fn prepare_evaluation(
    view: &TrainOnlyRegressionView<'_>,
    contract: &RunContractV2,
    fold: &ExplicitFoldV1,
    canonical_genome: &str,
) -> Result<PreparedEvaluation> {
    contract.validate()?;
    if contract.prediction.training_feature_view_sha256 != view.feature_view_sha256 {
        bail!("run contract training feature digest does not match the supplied view");
    }
    fold.validate(view)?;
    if canonical_genome.len() > 64 * 1024 {
        bail!("candidate genome exceeds the 64 KiB proof-slice limit");
    }
    let genome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GenomeIr::parse(canonical_genome)
    })) {
        Ok(parsed) => parsed.context("parse canonical genome")?,
        Err(_) => bail!("candidate parser panicked"),
    };
    match &genome {
        GenomeIr::Pipeline(pipeline)
            if !pipeline.stages.is_empty() && pipeline.stages.len() <= 8 => {}
        GenomeIr::Pipeline(_) => {
            bail!("V2 compatibility proof accepts at most six transforms plus target/head")
        }
        _ => bail!("V2 compatibility proof accepts pipeline genomes only"),
    }
    if genome.canonical() != canonical_genome {
        bail!("candidate genome is not canonical");
    }
    let candidate_semantic_sha256 = hex::encode(Sha256::digest(canonical_genome.as_bytes()));
    let fold_sha256 = fold.digest(view)?;
    let training_target_sha256 = hash_targets(view.targets, &fold.train_rows);
    let mut hash = SemanticHasher::new("jope-dime/evaluation-key/v2");
    hash.str("prediction_context_sha256", &contract.prediction.digest()?);
    hash.str("candidate_semantic_sha256", &candidate_semantic_sha256);
    hash.str("fold_sha256", &fold_sha256);
    hash.str("training_target_sha256", &training_target_sha256);
    let evaluation_key = EvaluationKeyV2(hash.finish());
    Ok(PreparedEvaluation {
        genome,
        candidate_semantic_sha256,
        evaluation_key,
        fold_sha256,
        training_target_sha256,
    })
}

fn execute_prepared(
    view: &TrainOnlyRegressionView<'_>,
    contract: &RunContractV2,
    fold: &ExplicitFoldV1,
    prepared: PreparedEvaluation,
) -> Result<EvaluationOutcomeV2> {
    let train_x = select_rows(view.rows, &fold.train_rows);
    let train_y = fold
        .train_rows
        .iter()
        .map(|row| view.targets[*row])
        .collect::<Vec<_>>();
    let validation_x = select_rows(view.rows, &fold.validation_rows);
    let validation_y = fold
        .validation_rows
        .iter()
        .map(|row| view.targets[*row])
        .collect::<Vec<_>>();

    let fit_started = Instant::now();
    let model = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit(&prepared.genome, &train_x, &train_y)
    })) {
        Ok(fitted) => fitted.context("fit explicit V2 fold")?,
        Err(_) => bail!("explicit V2 fold fit panicked"),
    };
    let fit_nanos = elapsed_nanos(fit_started);
    let predict_started = Instant::now();
    let predictions = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.predict(&validation_x)
    })) {
        Ok(predictions) => predictions.context("predict explicit V2 fold")?,
        Err(_) => bail!("explicit V2 fold prediction panicked"),
    };
    let predict_nanos = elapsed_nanos(predict_started);
    if predictions.len() != validation_y.len() || predictions.iter().any(|v| !v.is_finite()) {
        bail!("explicit V2 fold produced non-finite or wrong-length predictions");
    }

    let mut fit_binding = SemanticHasher::new("jope-dime/fit-binding/v1");
    fit_binding.str(
        "candidate_semantic_sha256",
        &prepared.candidate_semantic_sha256,
    );
    fit_binding.str("evaluation_key", &prepared.evaluation_key.0);
    fit_binding.str("fold_sha256", &prepared.fold_sha256);
    fit_binding.str("training_target_sha256", &prepared.training_target_sha256);
    fit_binding.str("fit_spec_sha256", &contract.prediction.fit_spec.digest()?);
    fit_binding.str(
        "conditioning_sha256",
        &contract.prediction.conditioning.digest()?,
    );
    fit_binding.str("evaluator_sha256", &contract.prediction.evaluator.digest()?);
    fit_binding.str(
        "math_policy_sha256",
        &contract.prediction.math_policy.digest()?,
    );
    let semantic = build_semantic_receipt(
        view,
        contract,
        fold,
        &prepared.candidate_semantic_sha256,
        &prepared.fold_sha256,
        prepared.evaluation_key,
        &predictions,
        fit_binding.finish(),
    )?;
    let semantic_receipt_sha256 = semantic.semantic_receipt_sha256.clone();
    Ok(EvaluationOutcomeV2 {
        semantic,
        observation: EvaluationObservationV1 {
            schema: OBSERVATION_SCHEMA.into(),
            semantic_receipt_sha256,
            cache_hit: false,
            fit_nanos,
            predict_nanos,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn build_semantic_receipt(
    view: &TrainOnlyRegressionView<'_>,
    contract: &RunContractV2,
    fold: &ExplicitFoldV1,
    candidate_semantic_sha256: &str,
    fold_sha256: &str,
    evaluation_key: EvaluationKeyV2,
    predictions: &[f64],
    fit_binding_sha256: String,
) -> Result<SemanticEvaluationReceiptV2> {
    let validation_y = fold
        .validation_rows
        .iter()
        .map(|row| view.targets[*row])
        .collect::<Vec<_>>();
    let tape = PredictionTapeV1::new(
        fold,
        evaluation_key.clone(),
        &validation_y,
        predictions,
        fit_binding_sha256,
    )?;
    let fitness = score_regression_tape(&tape)?;
    let objective_sha256 = contract.objective.digest()?;
    let score_key = score_key(&tape.tape_sha256, &objective_sha256);
    let run_contract_sha256 = contract.digest()?;
    let semantic_receipt_sha256 = hash_receipt_parts(
        &run_contract_sha256,
        candidate_semantic_sha256,
        fold_sha256,
        &evaluation_key,
        &tape.tape_sha256,
        &objective_sha256,
        &score_key,
        &fitness,
    );
    let receipt = SemanticEvaluationReceiptV2 {
        schema: RECEIPT_SCHEMA.into(),
        run_contract_sha256,
        candidate_semantic_sha256: candidate_semantic_sha256.into(),
        fold_sha256: fold_sha256.into(),
        evaluation_key,
        prediction_tape: tape,
        objective_sha256,
        score_key,
        fitness,
        semantic_receipt_sha256,
    };
    receipt.validate_against(contract)?;
    Ok(receipt)
}

fn score_regression_tape(tape: &PredictionTapeV1) -> Result<RegressionFitnessV1> {
    tape.validate()?;
    let truth = tape
        .truth_bits
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect::<Vec<_>>();
    let predictions = tape
        .prediction_bits
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect::<Vec<_>>();
    let (rmse, r2) = independent_regression_metrics(&truth, &predictions)?;
    let robust = crate::commands::exact_cv_v2::legacy_robust_fitness(r2, &[r2]);
    if !robust.is_finite() {
        bail!("robust regression fitness is non-finite");
    }
    Ok(RegressionFitnessV1 {
        schema: FITNESS_SCHEMA.into(),
        pooled_rmse_bits: rmse.to_bits(),
        pooled_r2_bits: r2.to_bits(),
        robust_fitness_bits: robust.to_bits(),
    })
}

fn independent_regression_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64)> {
    if truth.is_empty()
        || truth.len() != predictions.len()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("metrics require equal, non-empty, finite inputs");
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let mut squared_error = 0.0;
    let mut total_sum_squares = 0.0;
    for (&actual, &predicted) in truth.iter().zip(predictions) {
        squared_error += (actual - predicted).powi(2);
        total_sum_squares += (actual - mean).powi(2);
    }
    let rmse = (squared_error / n).sqrt();
    let r2 = if total_sum_squares > 1e-24 {
        1.0 - squared_error / total_sum_squares
    } else {
        0.0
    };
    if !rmse.is_finite() || !r2.is_finite() {
        bail!("regression metrics are non-finite");
    }
    Ok((rmse, r2))
}

fn hash_feature_view(rows: &Rows, row_ids: &[String], width: usize) -> String {
    let mut hash = SemanticHasher::new("jope-dime/training-feature-view/v1");
    hash.u64("rows", rows.len() as u64);
    hash.u64("width", width as u64);
    for (index, (row_id, row)) in row_ids.iter().zip(rows).enumerate() {
        hash.u64("row_ordinal", index as u64);
        hash.str("stable_row_id", row_id);
        for value in row {
            hash.u64("feature_bits", value.to_bits());
        }
    }
    hash.finish()
}

fn hash_targets(targets: &[f64], rows: &[usize]) -> String {
    let mut hash = SemanticHasher::new("jope-dime/training-target-view/v1");
    for row in rows {
        hash.u64("row", *row as u64);
        hash.u64("target_bits", targets[*row].to_bits());
    }
    hash.finish()
}

fn hash_bound_rows(domain: &str, rows: &[usize], row_ids: &[String]) -> String {
    let mut hash = SemanticHasher::new(domain);
    for (row, row_id) in rows.iter().zip(row_ids) {
        hash.u64("row", *row as u64);
        hash.str("stable_row_id", row_id);
    }
    hash.finish()
}

fn hash_tape_parts(
    fold_id: u32,
    evaluation_key: &EvaluationKeyV2,
    row_ids: &[String],
    truth_bits: &[u64],
    prediction_bits: &[u64],
    validity: &[bool],
    fit_binding_sha256: &str,
) -> String {
    let mut hash = SemanticHasher::new("jope-dime/prediction-tape/v1");
    hash.u64("fold_id", fold_id as u64);
    hash.str("evaluation_key", &evaluation_key.0);
    hash.str("fit_binding_sha256", fit_binding_sha256);
    for (((row_id, truth), prediction), valid) in row_ids
        .iter()
        .zip(truth_bits)
        .zip(prediction_bits)
        .zip(validity)
    {
        hash.str("validation_row_id", row_id);
        hash.u64("truth_bits", *truth);
        hash.u64("prediction_bits", *prediction);
        hash.u64("valid", u64::from(*valid));
    }
    hash.finish()
}

fn hash_receipt_parts(
    run_contract_sha256: &str,
    candidate_semantic_sha256: &str,
    fold_sha256: &str,
    evaluation_key: &EvaluationKeyV2,
    tape_sha256: &str,
    objective_sha256: &str,
    score_key: &ScoreKeyV1,
    fitness: &RegressionFitnessV1,
) -> String {
    let mut hash = SemanticHasher::new("jope-dime/semantic-evaluation-receipt/v2");
    hash.str("run_contract_sha256", run_contract_sha256);
    hash.str("candidate_semantic_sha256", candidate_semantic_sha256);
    hash.str("fold_sha256", fold_sha256);
    hash.str("evaluation_key", &evaluation_key.0);
    hash.str("tape_sha256", tape_sha256);
    hash.str("objective_sha256", objective_sha256);
    hash.str("score_key", &score_key.0);
    hash.str("fitness_schema", &fitness.schema);
    hash.u64("pooled_rmse_bits", fitness.pooled_rmse_bits);
    hash.u64("pooled_r2_bits", fitness.pooled_r2_bits);
    hash.u64("robust_fitness_bits", fitness.robust_fitness_bits);
    hash.finish()
}

fn score_key(tape_sha256: &str, objective_sha256: &str) -> ScoreKeyV1 {
    let mut hash = SemanticHasher::new("jope-dime/score-key/v1");
    hash.str("tape_sha256", tape_sha256);
    hash.str("objective_sha256", objective_sha256);
    ScoreKeyV1(hash.finish())
}

fn row_ids_for(rows: &[usize], view: &TrainOnlyRegressionView<'_>) -> Result<Vec<String>> {
    rows.iter()
        .map(|row| {
            view.stable_row_ids
                .get(*row)
                .cloned()
                .with_context(|| format!("fold row {row} is outside training view"))
        })
        .collect()
}

fn validate_ordered_unique_rows(label: &str, rows: &[usize], len: usize) -> Result<()> {
    let mut seen = HashSet::with_capacity(rows.len());
    for row in rows {
        if *row >= len {
            bail!("{label} row {row} is outside training view");
        }
        if !seen.insert(*row) {
            bail!("{label} rows contain duplicate {row}");
        }
    }
    Ok(())
}

fn select_rows(rows: &Rows, selected: &[usize]) -> Rows {
    selected.iter().map(|row| rows[*row].clone()).collect()
}

fn elapsed_nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn require_schema(actual: &str, expected: &str, label: &str) -> Result<()> {
    if actual != expected {
        bail!("{label} must be {expected}, found {actual}");
    }
    Ok(())
}

fn validate_stable_row_id(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b':'))
    {
        bail!(
            "stable row IDs must be 1..=128 opaque ASCII alphanumeric, '-', '_', or ':' characters"
        );
    }
    Ok(())
}

pub(crate) fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

pub(crate) fn digest_literal(domain: &str, bytes: &[u8]) -> String {
    let mut hash = SemanticHasher::new(domain);
    hash.bytes("value", bytes);
    hash.finish()
}

struct SemanticHasher(Sha256);

impl SemanticHasher {
    fn new(domain: &str) -> Self {
        let mut value = Self(Sha256::new());
        value.bytes("domain", domain.as_bytes());
        value
    }

    fn bytes(&mut self, label: &str, value: &[u8]) {
        self.0.update((label.len() as u64).to_le_bytes());
        self.0.update(label.as_bytes());
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }

    fn str(&mut self, label: &str, value: &str) {
        self.bytes(label, value.as_bytes());
    }

    fn u64(&mut self, label: &str, value: u64) {
        self.bytes(label, &value.to_le_bytes());
    }

    fn finish(self) -> String {
        hex::encode(self.0.finalize())
    }
}

mod hex_u64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{value:016x}"))
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() != 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom(
                "f64 bits must be 16 hexadecimal characters",
            ));
        }
        u64::from_str_radix(&value, 16).map_err(serde::de::Error::custom)
    }
}

mod hex_u64_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(values: &[u64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values
            .iter()
            .map(|value| format!("{value:016x}"))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<String>::deserialize(deserializer)?
            .into_iter()
            .map(|value| {
                if value.len() != 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(serde::de::Error::custom(
                        "f64 bits must be 16 hexadecimal characters",
                    ));
                }
                u64::from_str_radix(&value, 16).map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robust(r2: f64, folds: &[f64]) -> f64 {
        crate::commands::exact_cv_v2::legacy_robust_fitness(r2, folds)
    }

    fn fixture_data() -> (Rows, Vec<f64>, Vec<String>) {
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        let mut ids = Vec::new();
        for row in 0..12 {
            let a = row as f64 / 12.0;
            rows.push(vec![a, 1.0 - a, a * a]);
            targets.push(2.0 * a - 0.25 * (1.0 - a));
            ids.push(format!("row-{row:02}"));
        }
        (rows, targets, ids)
    }

    #[test]
    fn zero_width_roll_view_is_refused_before_fold_execution() {
        let (_, targets, ids) = fixture_data();
        let empty_rows = vec![Vec::new(); targets.len()];
        let error = TrainOnlyRegressionView::new(&empty_rows, &targets, &ids, 0).unwrap_err();
        assert!(error.to_string().contains("non-empty-width"));
        let error = TrainOnlyRegressionView::new(&empty_rows, &targets, &ids, 3).unwrap_err();
        assert!(error.to_string().contains("rectangular"));
    }

    fn evaluator() -> EvaluatorIdentityV1 {
        EvaluatorIdentityV1::current()
    }

    fn objective(name: &str) -> ObjectiveIdentityV1 {
        match name {
            "robust-r2" => ObjectiveIdentityV1::legacy_robust_r2(),
            "negative-rmse" => ObjectiveIdentityV1::negative_rmse(),
            _ => unreachable!(),
        }
    }

    fn fixture<'a>(
        rows: &'a Rows,
        targets: &'a [f64],
        ids: &'a [String],
    ) -> (TrainOnlyRegressionView<'a>, ExplicitFoldV1, RunContractV2) {
        let view = TrainOnlyRegressionView::new(rows, targets, ids, 3).unwrap();
        let fold = ExplicitFoldV1::new(0, (0..8).collect(), (8..12).collect(), &view).unwrap();
        let contract = RunContractV2::strict_legacy(
            view.feature_view_sha256().into(),
            evaluator(),
            objective("robust-r2"),
        )
        .unwrap();
        (view, fold, contract)
    }

    #[test]
    fn prediction_key_invalidates_each_scientific_field() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let baseline = prepare_evaluation(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap()
            .evaluation_key;

        let mut changed_rows = rows.clone();
        changed_rows[0][0] = f64::from_bits(changed_rows[0][0].to_bits() + 1);
        let (changed_view, changed_fold, mut changed_contract) =
            fixture(&changed_rows, &targets, &ids);
        let changed = prepare_evaluation(
            &changed_view,
            &changed_contract,
            &changed_fold,
            "(pipe X (ridge))",
        )
        .unwrap()
        .evaluation_key;
        assert_ne!(baseline, changed);

        let mut changed_targets = targets.clone();
        changed_targets[0] = f64::from_bits(changed_targets[0].to_bits() + 1);
        let (target_view, target_fold, target_contract) = fixture(&rows, &changed_targets, &ids);
        assert_ne!(
            baseline,
            prepare_evaluation(
                &target_view,
                &target_contract,
                &target_fold,
                "(pipe X (ridge))"
            )
            .unwrap()
            .evaluation_key
        );

        let reordered_fold =
            ExplicitFoldV1::new(0, (0..8).rev().collect(), (8..12).collect(), &view).unwrap();
        assert_ne!(
            baseline,
            prepare_evaluation(&view, &contract, &reordered_fold, "(pipe X (ridge))")
                .unwrap()
                .evaluation_key
        );
        assert_ne!(
            baseline,
            prepare_evaluation(
                &view,
                &contract,
                &fold,
                "(pipe X (winsor_clip q=0.1) (ridge))"
            )
            .unwrap()
            .evaluation_key
        );

        changed_contract.prediction.evaluator.implementation_sha256 = "4".repeat(64);
        assert!(prepare_evaluation(
            &changed_view,
            &changed_contract,
            &changed_fold,
            "(pipe X (ridge))"
        )
        .is_err());
    }

    #[test]
    fn build_owned_identities_reject_false_attestation() {
        let mut evaluator = EvaluatorIdentityV1::current();
        evaluator.implementation_sha256 = "4".repeat(64);
        assert!(evaluator.validate().is_err());
        assert!(ObjectiveIdentityV1::from_policy("unregistered-v1").is_err());
        let mut objective = ObjectiveIdentityV1::legacy_robust_r2();
        objective.config_sha256 = "5".repeat(64);
        assert!(objective.validate().is_err());

        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let mut bad_fit = contract.clone();
        bad_fit.prediction.fit_spec.fit_abi = "other-fit".into();
        assert!(prepare_evaluation(&view, &bad_fit, &fold, "(pipe X (ridge))").is_err());
        let mut bad_conditioning = contract.clone();
        bad_conditioning.prediction.conditioning.kind = "ambient-hyperion".into();
        assert!(prepare_evaluation(&view, &bad_conditioning, &fold, "(pipe X (ridge))").is_err());
        let mut bad_math = contract;
        bad_math.prediction.math_policy.arithmetic = "fast-math".into();
        assert!(prepare_evaluation(&view, &bad_math, &fold, "(pipe X (ridge))").is_err());
    }

    #[test]
    fn training_feature_change_rebinds_evaluation_and_fit() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let first =
            evaluate_regression_fold_v2(&view, &contract, &fold, "(pipe X (ridge))").unwrap();

        let mut changed_rows = rows.clone();
        changed_rows[0][0] += 0.5;
        let (changed_view, changed_fold, changed_contract) = fixture(&changed_rows, &targets, &ids);
        let second = evaluate_regression_fold_v2(
            &changed_view,
            &changed_contract,
            &changed_fold,
            "(pipe X (ridge))",
        )
        .unwrap();
        assert_ne!(
            first.semantic.evaluation_key,
            second.semantic.evaluation_key
        );
        assert_ne!(
            first.semantic.prediction_tape.fit_binding_sha256,
            second.semantic.prediction_tape.fit_binding_sha256
        );
    }

    #[test]
    fn objective_rescore_reuses_prediction_tape() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let mut memo = EvaluationMemoV2::default();
        let first = memo
            .evaluate_regression_fold(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap();
        let changed = RunContractV2::strict_legacy(
            view.feature_view_sha256().into(),
            evaluator(),
            objective("negative-rmse"),
        )
        .unwrap();
        let second = memo
            .evaluate_regression_fold(&view, &changed, &fold, "(pipe X (ridge))")
            .unwrap();
        assert!(second.observation.cache_hit);
        assert_eq!(
            first.semantic.evaluation_key,
            second.semantic.evaluation_key
        );
        assert_eq!(
            first.semantic.prediction_tape.tape_sha256,
            second.semantic.prediction_tape.tape_sha256
        );
        assert_ne!(first.semantic.score_key, second.semantic.score_key);
    }

    #[test]
    fn compatibility_shadow_uses_common_tape_evidence_only() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let mut memo = EvaluationMemoV2::default();
        let baseline = memo
            .evaluate_regression_fold(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap();
        let rescored_contract = RunContractV2::strict_legacy(
            view.feature_view_sha256().into(),
            evaluator(),
            objective("negative-rmse"),
        )
        .unwrap();
        let challenger = memo
            .evaluate_regression_fold(&view, &rescored_contract, &fold, "(pipe X (ridge))")
            .unwrap();
        let shadow = PredictionCompatibilityShadowV1::build(
            &baseline.semantic,
            &contract,
            &challenger.semantic,
            &rescored_contract,
        )
        .unwrap();
        assert!(shadow.objective_rescore_only);
        assert_eq!(shadow.common_rows, fold.validation_rows.len());

        let mut changed_truth = challenger.semantic.clone();
        changed_truth.prediction_tape.truth_bits[0] ^= 1;
        assert!(PredictionCompatibilityShadowV1::build(
            &baseline.semantic,
            &contract,
            &changed_truth,
            &rescored_contract,
        )
        .is_err());
    }

    #[test]
    fn no_forgetting_archive_replaces_only_on_confirmed_superset_evidence() {
        let first_evidence = "1".repeat(64);
        let second_evidence = "2".repeat(64);
        let current = NoForgettingArchiveEntryV1 {
            schema: NO_FORGETTING_ARCHIVE_SCHEMA.into(),
            candidate_semantic_sha256: "a".repeat(64),
            confirmed_evidence: BTreeMap::from([(first_evidence.clone(), 0.1_f64.to_bits())]),
            compatibility_shadow_sha256: "c".repeat(64),
        };
        let challenger = NoForgettingArchiveEntryV1 {
            schema: NO_FORGETTING_ARCHIVE_SCHEMA.into(),
            candidate_semantic_sha256: "b".repeat(64),
            confirmed_evidence: BTreeMap::from([
                (first_evidence.clone(), 0.1_f64.to_bits()),
                (second_evidence, 0.2_f64.to_bits()),
            ]),
            compatibility_shadow_sha256: "d".repeat(64),
        };
        assert_eq!(
            replace_no_forgetting_archive(&current, challenger.clone()).unwrap(),
            challenger
        );

        let mut equal_evidence = challenger.clone();
        equal_evidence.confirmed_evidence.remove(&"2".repeat(64));
        assert!(replace_no_forgetting_archive(&current, equal_evidence).is_err());
        let mut regressed = challenger;
        regressed
            .confirmed_evidence
            .insert(first_evidence, 0.09_f64.to_bits());
        assert!(replace_no_forgetting_archive(&current, regressed).is_err());
    }

    #[test]
    fn validation_label_rescore_reuses_predictions_but_rebinds_tape() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let mut memo = EvaluationMemoV2::default();
        let first = memo
            .evaluate_regression_fold(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap();

        let mut changed_targets = targets.clone();
        changed_targets[8] += 1.0;
        let changed_view = TrainOnlyRegressionView::new(&rows, &changed_targets, &ids, 3).unwrap();
        let changed_fold =
            ExplicitFoldV1::new(0, (0..8).collect(), (8..12).collect(), &changed_view).unwrap();
        let second = memo
            .evaluate_regression_fold(&changed_view, &contract, &changed_fold, "(pipe X (ridge))")
            .unwrap();

        assert!(second.observation.cache_hit);
        assert_eq!(
            first.semantic.evaluation_key,
            second.semantic.evaluation_key
        );
        assert_eq!(
            first.semantic.prediction_tape.prediction_bits,
            second.semantic.prediction_tape.prediction_bits
        );
        assert_ne!(
            first.semantic.prediction_tape.truth_bits,
            second.semantic.prediction_tape.truth_bits
        );
        assert_ne!(
            first.semantic.prediction_tape.tape_sha256,
            second.semantic.prediction_tape.tape_sha256
        );
        assert_ne!(first.semantic.score_key, second.semantic.score_key);
        assert_ne!(first.semantic.fitness, second.semantic.fitness);
        assert_ne!(
            first.semantic.semantic_receipt_sha256,
            second.semantic.semantic_receipt_sha256
        );
    }

    #[test]
    fn observation_noise_cannot_change_semantic_receipt_and_memo_hits() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let mut memo = EvaluationMemoV2::default();
        let first = memo
            .evaluate_regression_fold(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap();
        let second = memo
            .evaluate_regression_fold(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap();
        assert!(!first.observation.cache_hit);
        assert!(second.observation.cache_hit);
        assert_eq!(first.semantic, second.semantic);
        let mut noisy = first.observation.clone();
        noisy.fit_nanos = u64::MAX;
        noisy.predict_nanos = 17;
        assert_eq!(first.semantic, second.semantic);
        assert_ne!(first.observation, noisy);
    }

    #[test]
    fn explicit_fold_rejects_overlap_duplicates_and_reordering_drift() {
        let (rows, targets, ids) = fixture_data();
        let view = TrainOnlyRegressionView::new(&rows, &targets, &ids, 3).unwrap();
        assert!(ExplicitFoldV1::new(0, vec![0; 8], vec![8], &view).is_err());
        assert!(ExplicitFoldV1::new(0, (0..8).collect(), vec![7, 8], &view).is_err());
        assert!(ExplicitFoldV1::new(0, (0..8).collect(), vec![12], &view).is_err());
        let mut fold = ExplicitFoldV1::new(0, (0..8).collect(), (8..12).collect(), &view).unwrap();
        fold.train_rows.swap(0, 1);
        assert!(fold.validate(&view).is_err());
    }

    #[test]
    fn prediction_tape_rejects_wrong_length_nonfinite_and_row_mismatch() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let key = prepare_evaluation(&view, &contract, &fold, "(pipe X (ridge))")
            .unwrap()
            .evaluation_key;
        assert!(
            PredictionTapeV1::new(&fold, key.clone(), &[1.0], &[1.0, 2.0], "a".repeat(64)).is_err()
        );
        assert!(PredictionTapeV1::new(
            &fold,
            key,
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, f64::NAN, 4.0],
            "a".repeat(64)
        )
        .is_err());
    }

    #[test]
    fn receipt_cross_links_and_json_bits_fail_closed() {
        let (rows, targets, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let outcome =
            evaluate_regression_fold_v2(&view, &contract, &fold, "(pipe X (ridge))").unwrap();
        outcome.semantic.validate_against(&contract).unwrap();

        let json = serde_json::to_value(&outcome.semantic).unwrap();
        assert!(json["prediction_tape"]["truth_bits"][0].is_string());
        assert!(json["fitness"]["pooled_r2_bits"].is_string());

        let mut bad_score = outcome.semantic.clone();
        bad_score.score_key = ScoreKeyV1("a".repeat(64));
        assert!(bad_score.validate_against(&contract).is_err());
        let mut bad_tape = outcome.semantic.clone();
        bad_tape.prediction_tape.prediction_bits[0] ^= 1;
        assert!(bad_tape.validate_against(&contract).is_err());
    }

    #[test]
    fn compatibility_slice_rejects_non_pipeline_and_path_like_row_ids() {
        let (rows, targets, mut ids) = fixture_data();
        ids[0] = "/private/train/0".into();
        assert!(TrainOnlyRegressionView::new(&rows, &targets, &ids, 3).is_err());

        let (_, _, ids) = fixture_data();
        let (view, fold, contract) = fixture(&rows, &targets, &ids);
        let symbolic = GenomeIr::parse("(sym raw \"col(0)\")").unwrap().canonical();
        assert!(evaluate_regression_fold_v2(&view, &contract, &fold, &symbolic).is_err());

        let oversized = format!("(pipe X {} (ridge))", "(standardize) ".repeat(8));
        let oversized = GenomeIr::parse(&oversized).unwrap().canonical();
        assert!(evaluate_regression_fold_v2(&view, &contract, &fold, &oversized).is_err());
    }

    #[test]
    fn regression_objective_matches_v1_authority_bits() {
        let truth = [1.0, 2.0, 4.0, 8.0];
        let predictions = [1.25, 1.5, 4.5, 7.0];
        let (rmse, r2) = independent_regression_metrics(&truth, &predictions).unwrap();
        assert_eq!(robust(r2, &[r2]).to_bits(), r2.to_bits());
        assert!(rmse > 0.0);
    }
}
