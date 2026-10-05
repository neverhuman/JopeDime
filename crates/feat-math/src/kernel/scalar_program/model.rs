//! Train-only scalar features and the existing ridge head. No v1 coercion.
use super::{ScalarExpressionProgram, ScalarNode};
use crate::heads::RidgeModel;
use crate::kernel::codec::MAX_BYTES;
use crate::kernel::wire_v2::{VersionError, VersionResult};
use crate::kernel::{Error, Result};
use crate::math_expr::{FaultSet, Var};
use crate::Rows;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_SCHEMA: &str = "feat-math.scalar-model";
pub const MODEL_PROFILE: &str = "feat-math.scalar-ridge.experimental.v2";
const CONTRACT: &str = "scalar-ridge-v2;func=0..22;policy=v1|v2;carrier=row|elementwise;nodes<=64;depth<=16;width=1..512;lambda=1e-5..2;head=ridge;score-clamp=20;prediction-clamp=60;fit=train-only;state=ieee754-bits";
const EMITTER_IDENTITY: &str = "feat-math.scalar.ridge.v2";
const MAX_INPUT_WIDTH: usize = 512;

/// A row expression yields one feature; an elementwise expression yields one
/// feature per input column, with `x` bound to that column's original value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarCarrier {
    Row,
    Elementwise,
}

impl ScalarCarrier {
    fn output_width(self, input_width: usize) -> usize {
        match self {
            Self::Row => 1,
            Self::Elementwise => input_width,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarModel {
    program: ScalarExpressionProgram,
    carrier: ScalarCarrier,
    input_width: usize,
    lambda: f64,
    ridge: RidgeModel,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWire {
    schema: String,
    format_version: u8,
    min_reader_version: u8,
    required_features: u64,
    numerical_policy: String,
    profile: String,
    grammar_sha256: String,
    emitter_feature_identity: String,
    output_schema: Vec<String>,
    program: Vec<u8>,
    carrier: ScalarCarrier,
    input_width: usize,
    lambda_bits: u64,
    ridge_bits: [Vec<u64>; 3],
}

impl ScalarExpressionProgram {
    /// Fit only the rows and labels supplied by the training caller. Existing
    /// ridge standardization, solve order and prediction clamps are retained.
    /// Raw input cells reach the scalar evaluator so v2 large/tiny/nonfinite
    /// branches and fault telemetry are not erased by an input sanitization.
    pub fn fit_training(
        &self,
        training: &Rows,
        labels: &[f64],
        carrier: ScalarCarrier,
        lambda: f64,
    ) -> Result<ScalarModel> {
        let input_width = training
            .first()
            .ok_or_else(|| Error::Shape("scalar fitting requires nonempty training rows".into()))?
            .len();
        validate_config(self, carrier, input_width, lambda)?;
        if training.len() != labels.len() {
            return Err(Error::Shape("scalar training row/label count".into()));
        }
        if labels.iter().any(|y| !y.is_finite() || y.abs() > 1e12) {
            return Err(Error::NonFinite(
                "scalar labels must be finite and bounded to 1e12".into(),
            ));
        }
        let (features, _) = transform(self, carrier, input_width, training)?;
        let model = ScalarModel {
            program: self.clone(),
            carrier,
            input_width,
            lambda,
            ridge: RidgeModel::fit(&features, labels, lambda),
        };
        model.validate()?;
        Ok(model)
    }
}

impl ScalarModel {
    pub(super) fn ridge(&self) -> &RidgeModel {
        &self.ridge
    }

    pub fn program(&self) -> &ScalarExpressionProgram {
        &self.program
    }
    pub fn carrier(&self) -> ScalarCarrier {
        self.carrier
    }
    pub fn input_width(&self) -> usize {
        self.input_width
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn transform(&self, rows: &Rows) -> Result<(Rows, Vec<u32>)> {
        self.validate()?;
        transform(&self.program, self.carrier, self.input_width, rows)
    }

    pub fn predict(&self, rows: &Rows) -> Result<(Vec<f64>, Vec<u32>)> {
        let (features, faults) = self.transform(rows)?;
        Ok((self.ridge.predict(&features), faults))
    }

    pub(super) fn validate(&self) -> Result<()> {
        validate_config(&self.program, self.carrier, self.input_width, self.lambda)?;
        let width = self.carrier.output_width(self.input_width);
        // Prevent even forged finite weights overflowing the bounded-score dot
        // product. Include its intercept and a factor-of-two safety margin.
        let weight_limit = f64::MAX / (1.0 + 20.0 * width as f64) / 2.0;
        if self.ridge.means.len() != width
            || self.ridge.scales.len() != width
            || self.ridge.w.len() != width + 1
            || self
                .ridge
                .means
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 1e12)
            || self
                .ridge
                .scales
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .ridge
                .w
                .iter()
                .any(|v| !v.is_finite() || v.abs() > weight_limit)
        {
            return Err(Error::Domain("invalid scalar ridge state".into()));
        }
        Ok(())
    }

    pub fn to_json(&self) -> VersionResult<String> {
        self.validate()?;
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect();
        let wire = ModelWire {
            schema: MODEL_SCHEMA.into(),
            format_version: 2,
            min_reader_version: 2,
            required_features: self.program.required_features()?,
            numerical_policy: self.program.numerical_policy().name().into(),
            profile: MODEL_PROFILE.into(),
            grammar_sha256: grammar_sha256(),
            emitter_feature_identity: EMITTER_IDENTITY.into(),
            output_schema: vec!["prediction:f64".into()],
            program: self.program.to_bytes()?,
            carrier: self.carrier,
            input_width: self.input_width,
            lambda_bits: self.lambda.to_bits(),
            ridge_bits: [
                bits(&self.ridge.means),
                bits(&self.ridge.scales),
                bits(&self.ridge.w),
            ],
        };
        let text =
            serde_json::to_string(&wire).map_err(|e| VersionError::Malformed(e.to_string()))?;
        if text.len() > MAX_BYTES {
            return Err(VersionError::Malformed("scalar model size".into()));
        }
        Ok(text)
    }

    pub fn from_json(text: &str) -> VersionResult<Self> {
        if text.len() > MAX_BYTES {
            return Err(VersionError::Malformed("scalar model size".into()));
        }
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| VersionError::Malformed(e.to_string()))?;
        for name in ["format_version", "min_reader_version"] {
            if value.get(name).is_some_and(|v| v.as_u64() != Some(2)) {
                return Err(VersionError::UnsupportedVersion(format!(
                    "scalar model {name}"
                )));
            }
        }
        // Future header identities refuse before looking at an unknown payload.
        let supported = super::INVERSE_LINK_FEATURE
            | super::STABLE_EXPONENTIAL_FEATURE
            | super::NUMERICAL_POLICY_FEATURE;
        if value
            .get("schema")
            .is_some_and(|v| v.as_str() != Some(MODEL_SCHEMA))
            || value
                .get("required_features")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|bits| bits & !supported != 0)
            || value
                .get("numerical_policy")
                .is_some_and(|v| !matches!(v.as_str(), Some("v1" | "v2")))
        {
            return Err(VersionError::UnsupportedVersion(
                "scalar model header".into(),
            ));
        }
        let wire: ModelWire =
            serde_json::from_value(value).map_err(|e| VersionError::Malformed(e.to_string()))?;
        let program = ScalarExpressionProgram::from_bytes(&wire.program)?;
        if wire.schema != MODEL_SCHEMA
            || wire.required_features != program.required_features()?
            || wire.numerical_policy != program.numerical_policy().name()
            || wire.profile != MODEL_PROFILE
            || wire.grammar_sha256 != grammar_sha256()
            || wire.emitter_feature_identity != EMITTER_IDENTITY
            || wire.output_schema != ["prediction:f64"]
        {
            return Err(VersionError::UnsupportedVersion(
                "scalar model envelope".into(),
            ));
        }
        let floats = |values: &[u64]| values.iter().map(|&v| f64::from_bits(v)).collect();
        let model = Self {
            program,
            carrier: wire.carrier,
            input_width: wire.input_width,
            lambda: f64::from_bits(wire.lambda_bits),
            ridge: RidgeModel {
                means: floats(&wire.ridge_bits[0]),
                scales: floats(&wire.ridge_bits[1]),
                w: floats(&wire.ridge_bits[2]),
            },
        };
        model.validate()?;
        Ok(model)
    }

    pub fn identity(&self) -> VersionResult<String> {
        Ok(format!("{:x}", Sha256::digest(self.to_json()?.as_bytes())))
    }
}

pub(super) fn validate_config(
    program: &ScalarExpressionProgram,
    carrier: ScalarCarrier,
    width: usize,
    lambda: f64,
) -> Result<()> {
    if !(1..=MAX_INPUT_WIDTH).contains(&width) {
        return Err(Error::Bounds(
            "scalar model input width must be in 1..512".into(),
        ));
    }
    if !lambda.is_finite() || !(1e-5..=2.0).contains(&lambda) {
        return Err(Error::Domain(
            "scalar ridge lambda must be in [1e-5,2]".into(),
        ));
    }
    check_scope(program.ast(), carrier, width)
}

fn check_scope(node: &ScalarNode, carrier: ScalarCarrier, width: usize) -> Result<()> {
    match node {
        ScalarNode::Var(Var::X) if carrier == ScalarCarrier::Row => Err(Error::Domain(
            "x requires the scalar elementwise carrier".into(),
        )),
        ScalarNode::Var(Var::Col(col)) if *col >= width => Err(Error::Bounds(
            "scalar column outside declared input width".into(),
        )),
        ScalarNode::Unary(_, child) => check_scope(child, carrier, width),
        ScalarNode::Binary(_, left, right) => {
            check_scope(left, carrier, width)?;
            check_scope(right, carrier, width)
        }
        ScalarNode::Call(_, args) => {
            for arg in args {
                check_scope(arg, carrier, width)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn transform(
    program: &ScalarExpressionProgram,
    carrier: ScalarCarrier,
    width: usize,
    rows: &Rows,
) -> Result<(Rows, Vec<u32>)> {
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::Shape("scalar model input width mismatch".into()));
    }
    let compiled = program.compile()?;
    let mut features = Vec::with_capacity(rows.len());
    let mut faults = Vec::with_capacity(rows.len());
    for row in rows {
        let mut fault = FaultSet::EMPTY;
        let mut output = Vec::with_capacity(carrier.output_width(width));
        for column in 0..carrier.output_width(width) {
            let evaluated = match carrier {
                ScalarCarrier::Row => compiled.evaluate_row(row),
                ScalarCarrier::Elementwise => compiled.evaluate_element(row[column], row),
            };
            fault = fault.union(evaluated.faults);
            output.push(evaluated.value?);
        }
        features.push(output);
        faults.push(fault.bits());
    }
    Ok((features, faults))
}

pub(super) fn grammar_sha256() -> String {
    format!("{:x}", Sha256::digest(CONTRACT.as_bytes()))
}
