//! Explicit v2 FeatureWithinRow pipelines. The legacy generator and grammar
//! are unchanged. This bounded Experimental profile admits at most eight
//! row stages followed by the existing ridge learner, with train-only fit.
use super::codec::{Reader, Writer, MAX_BYTES};
use super::savgol;
use super::wire_v2::{VersionError, VersionResult, MAGIC_V2};
use super::{Error, Result};
use crate::heads::RidgeModel;
use crate::spec::{Plan, PlannedHead, PlannedOp, PlannedStage};
use crate::transpile::{lint_shipped_text, EmitOptions};
use crate::{Pcg32, Rows};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod emit;
pub const PROFILE: &str = "feat-math.feature-row.experimental.v2";
pub const REQUIRED_FEATURES: u64 = 32;
const PROFILE_CONTRACT: &str = "feat-math.feature-row.experimental.v2;axis=FeatureWithinRow;max_stages=8;prior_max=0.25;head=ridge-v1";
const MAX_STAGES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "wire", content = "parameters", deny_unknown_fields)]
pub enum RowStage {
    #[serde(rename = "58")]
    Savgol(savgol::Config),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Program {
    input_width: usize,
    stages: Vec<RowStage>,
    lambda_bits: u64,
}

impl<'de> Deserialize<'de> for Program {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Parameters {
            input_width: usize,
            stages: Vec<RowStage>,
            lambda_bits: u64,
        }
        let raw = Parameters::deserialize(deserializer)?;
        Self::new(
            2,
            raw.input_width,
            raw.stages,
            f64::from_bits(raw.lambda_bits),
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "wire", content = "state", deny_unknown_fields)]
pub enum FittedStage {
    #[serde(rename = "58")]
    Savgol(savgol::Fitted),
}

impl RowStage {
    fn fit(&self, width: usize) -> Result<FittedStage> {
        match self {
            Self::Savgol(config) => Ok(FittedStage::Savgol(config.fit(width)?)),
        }
    }
}

impl FittedStage {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Savgol(state) => state.validate(),
        }
    }
    pub fn input_width(&self) -> usize {
        match self {
            Self::Savgol(state) => state.width(),
        }
    }
    pub fn output_width(&self) -> usize {
        self.input_width()
    }
    pub(crate) fn apply(&self, row: &[f64]) -> Result<savgol::Output> {
        match self {
            Self::Savgol(state) => state.apply_validated(row),
        }
    }
}

impl Program {
    pub fn new(
        version: u8,
        input_width: usize,
        stages: Vec<RowStage>,
        lambda: f64,
    ) -> Result<Self> {
        if version != 2 {
            return Err(Error::Unsupported("row-vector program requires v2".into()));
        }
        if input_width == 0 || input_width > crate::MAX_WIDTH || stages.len() > MAX_STAGES {
            return Err(Error::Shape(
                "row-vector width=1..MAX_WIDTH and at most eight stages".into(),
            ));
        }
        if !lambda.is_finite() || !(1e-5..=2.0).contains(&lambda) {
            return Err(Error::Domain(
                "ridge lambda_frac must be finite in [1e-5,2]".into(),
            ));
        }
        let program = Self {
            input_width,
            stages,
            lambda_bits: lambda.to_bits(),
        };
        program.fitted_stages()?;
        Ok(program)
    }

    pub fn stages(&self) -> &[RowStage] {
        &self.stages
    }
    pub fn input_width(&self) -> usize {
        self.input_width
    }
    pub fn lambda(&self) -> f64 {
        f64::from_bits(self.lambda_bits)
    }

    pub(crate) fn fitted_stages(&self) -> Result<Vec<FittedStage>> {
        let mut width = self.input_width;
        self.stages
            .iter()
            .map(|stage| {
                let state = stage.fit(width)?;
                width = state.output_width();
                Ok(state)
            })
            .collect()
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        Self::new(2, self.input_width, self.stages.clone(), self.lambda())?;
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(3)?;
        writer.byte(2)?;
        writer.word(REQUIRED_FEATURES)?;
        writer.text(
            &serde_json::to_string(self).map_err(|e| VersionError::Malformed(e.to_string()))?,
        )?;
        Ok(writer.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(MAGIC_V2.len())? != MAGIC_V2
            || reader.byte()? != 3
            || reader.byte()? != 2
            || reader.word()? != REQUIRED_FEATURES
        {
            return Err(VersionError::UnsupportedVersion(
                "row-vector envelope/feature set".into(),
            ));
        }
        let parameters: serde_json::Value = serde_json::from_str(&reader.text()?)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        reader.finish()?;
        if let Some(stages) = parameters
            .get("stages")
            .and_then(serde_json::Value::as_array)
        {
            for stage in stages {
                if let Some(wire) = stage.get("wire").and_then(serde_json::Value::as_str) {
                    if wire != "58" {
                        return Err(VersionError::UnsupportedVersion(format!(
                            "row-vector wire {wire} is unavailable"
                        )));
                    }
                }
            }
        }
        let value: Self = serde_json::from_value(parameters)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let lambda = value.lambda();
        Ok(Self::new(2, value.input_width, value.stages, lambda)?)
    }

    pub fn canonical(&self) -> VersionResult<String> {
        Ok(format!(
            "(program codec=2 hex=\"{}\")",
            self.to_bytes()?
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ))
    }

    pub(crate) fn plan(&self) -> Result<Plan> {
        Ok(Plan::Pipeline {
            target: None,
            stages: std::iter::once(PlannedStage {
                op: PlannedOp::RowVectorInputV2 {
                    width: self.input_width,
                },
                append: false,
            })
            .chain(self.fitted_stages()?.into_iter().map(|state| PlannedStage {
                op: PlannedOp::RowVectorV2 { state },
                append: false,
            }))
            .collect(),
            head: PlannedHead::Ridge {
                lambda_frac: self.lambda(),
            },
        })
    }

    /// The geometry-only FIR setup is frozen before either training or export;
    /// recipe constants contain no observations, labels or fitted head state.
    pub fn emit_recipe(&self, language: emit::RowRecipeLanguage) -> Result<String> {
        let plan = self.plan()?;
        let opts = EmitOptions::default();
        let source = match language {
            emit::RowRecipeLanguage::Rust => crate::emit_rs::emit_rust(&plan, &opts),
            emit::RowRecipeLanguage::Python => crate::emit_py::emit_python(&plan, &opts),
            emit::RowRecipeLanguage::Cpp17 => opts
                .cpp_expression_recipe(&plan)
                .map_err(|e| Error::Unsupported(e.join(", ")))?,
        };
        if !lint_shipped_text(&source).is_empty() {
            return Err(Error::Unsupported("row-vector artifact vocabulary".into()));
        }
        Ok(source)
    }

    pub fn emit_recipe_counted(
        &self,
        language: emit::RowRecipeLanguage,
    ) -> Result<(String, crate::transpile::TokenCount)> {
        let source = self.emit_recipe(language)?;
        let kind = match language {
            emit::RowRecipeLanguage::Rust => crate::tokens::Lang::Rust,
            emit::RowRecipeLanguage::Python => crate::tokens::Lang::Python,
            emit::RowRecipeLanguage::Cpp17 => crate::tokens::Lang::Cpp,
        };
        let count = crate::tokens::count_tokens(&source, kind);
        Ok((source, count))
    }

    pub fn fit(&self, x: &Rows, y: &[f64]) -> Result<Model> {
        validate_rows(x, self.input_width)?;
        if x.is_empty() || x.len() != y.len() || y.iter().any(|y| !y.is_finite() || y.abs() > 1e12)
        {
            return Err(Error::Shape("row-vector fit requires equally long nonempty features and bounded finite training labels".into()));
        }
        let stages = self.fitted_stages()?;
        let (features, _) = transform(&stages, x)?;
        let ridge = RidgeModel::fit(&features, y, self.lambda());
        let model = Model {
            program: self.clone(),
            stages,
            ridge,
        };
        model.validate()?;
        Ok(model)
    }

    /// Seeded GP insert/parameter-edit/remove, under a removable profile switch.
    /// Search prior is explicitly bounded at 0.25; Baseline admission is absent.
    pub fn edit(&self, seed: u64, arm: Edit, profile: &ExperimentalProfile) -> Result<Self> {
        profile.validate()?;
        let mut stages = self.stages.clone();
        let mut rng = Pcg32::new(seed, 0x726f777632);
        match arm {
            Edit::Grow if profile.savgol && stages.len() < MAX_STAGES => {
                let window = [3, 5, 7, 9, 17][rng.next_u32() as usize % 5];
                let degree = (rng.next_u32() % 3) as u8;
                let derivative = (rng.next_u32() % (u32::from(degree) + 1)) as u8;
                let stage = RowStage::Savgol(savgol::Config::new(
                    2,
                    58,
                    super::ordered_axis::Axis::FeatureWithinRow,
                    window,
                    degree,
                    derivative,
                    "shrink",
                )?);
                let at = rng.next_u32() as usize % (stages.len() + 1);
                stages.insert(at, stage);
            }
            Edit::Change if profile.savgol && !stages.is_empty() => {
                let at = rng.next_u32() as usize % stages.len();
                let RowStage::Savgol(config) = &stages[at];
                let derivative = (config.derivative() + 1) % (config.degree() + 1);
                let window = if config.window() == 129 {
                    (config.degree() + 1).max(3) | 1
                } else {
                    config.window() + 2
                };
                stages[at] = RowStage::Savgol(savgol::Config::new(
                    2,
                    58,
                    super::ordered_axis::Axis::FeatureWithinRow,
                    window,
                    config.degree(),
                    derivative,
                    "shrink",
                )?);
            }
            Edit::Drop if !stages.is_empty() => {
                let at = rng.next_u32() as usize % stages.len();
                stages.remove(at);
            }
            _ => {
                return Err(Error::Unsupported(
                    "row-vector GP arm/profile unavailable".into(),
                ))
            }
        }
        Self::new(2, self.input_width, stages, self.lambda())
    }

    /// Opt-in prior gate for a search scheduler; an explicit arm remains useful
    /// for EEC probes. Neither API enters the frozen shipping generator.
    pub fn mutate(&self, seed: u64, profile: &ExperimentalProfile) -> Result<Self> {
        profile.validate()?;
        let mut rng = Pcg32::new(seed, 0x726f777072696f72);
        if !profile.savgol || rng.next_f64() >= profile.prior {
            return Ok(self.clone());
        }
        let arm = if self.stages.is_empty() {
            Edit::Grow
        } else {
            match rng.next_u32() % 3 {
                0 if self.stages.len() < MAX_STAGES => Edit::Grow,
                1 => Edit::Change,
                _ => Edit::Drop,
            }
        };
        self.edit(rng.next_u32() as u64, arm, profile)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Edit {
    Grow,
    Change,
    Drop,
}

#[derive(Clone, Debug)]
pub struct ExperimentalProfile {
    pub savgol: bool,
    pub prior: f64,
}
impl ExperimentalProfile {
    pub fn validate(&self) -> Result<()> {
        if !self.prior.is_finite() || self.prior <= 0.0 || self.prior > 0.25 {
            return Err(Error::Domain(
                "row-vector Experimental prior must be in (0,0.25]".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    program: Program,
    stages: Vec<FittedStage>,
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
    stages: Vec<FittedStage>,
    ridge_bits: [Vec<u64>; 3],
}

impl Model {
    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn fitted_stages(&self) -> &[FittedStage] {
        &self.stages
    }
    pub(crate) fn ridge(&self) -> &RidgeModel {
        &self.ridge
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.stages != self.program.fitted_stages()? {
            return Err(Error::Domain(
                "row-vector model stage state differs from program".into(),
            ));
        }
        for stage in &self.stages {
            stage.validate()?;
        }
        let width = self
            .stages
            .last()
            .map_or(self.program.input_width, FittedStage::output_width);
        // Ridge scores are bounded to ±20. Include the intercept and retain
        // a factor-of-two margin so even a forged finite state cannot overflow
        // the fixed-order prediction dot product before the final clamp.
        let weight_limit = f64::MAX / (1.0 + 20.0 * width as f64) / 2.0;
        if self.ridge.means.len() != width
            || self.ridge.scales.len() != width
            || self.ridge.w.len() != width + 1
            || self
                .ridge
                .means
                .iter()
                .chain(&self.ridge.scales)
                .chain(&self.ridge.w)
                .any(|x| !x.is_finite())
            || self.ridge.scales.iter().any(|x| *x <= 0.0)
            || self.ridge.w.iter().any(|w| w.abs() > weight_limit)
        {
            return Err(Error::Domain("invalid row-vector ridge state".into()));
        }
        Ok(())
    }

    pub fn predict(&self, x: &Rows) -> Result<(Vec<f64>, Vec<u32>)> {
        self.validate()?;
        validate_rows(x, self.program.input_width)?;
        let (features, faults) = transform(&self.stages, x)?;
        Ok((self.ridge.predict(&features), faults))
    }

    pub fn to_json(&self) -> VersionResult<String> {
        self.validate()?;
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect();
        let wire = ModelWire {
            schema: "feat-math.feature-row-model".into(),
            format_version: 2,
            min_reader_version: 2,
            required_features: REQUIRED_FEATURES,
            numerical_policy: "row-protected-v2".into(),
            profile: PROFILE.into(),
            grammar_sha256: format!("{:x}", Sha256::digest(PROFILE_CONTRACT.as_bytes())),
            emitter_feature_identity: "feat-math.feature-row.recipes.v2+wire58".into(),
            output_schema: vec!["prediction:f64".into()],
            program: self.program.to_bytes()?,
            stages: self.stages.clone(),
            ridge_bits: [
                bits(&self.ridge.means),
                bits(&self.ridge.scales),
                bits(&self.ridge.w),
            ],
        };
        let text =
            serde_json::to_string(&wire).map_err(|e| VersionError::Malformed(e.to_string()))?;
        if text.len() > MAX_BYTES {
            return Err(VersionError::Malformed("row-vector model size".into()));
        }
        Ok(text)
    }

    pub fn from_json(text: &str) -> VersionResult<Self> {
        if text.len() > MAX_BYTES {
            return Err(VersionError::Malformed("row-vector model size".into()));
        }
        let wire: ModelWire =
            serde_json::from_str(text).map_err(|e| VersionError::Malformed(e.to_string()))?;
        if wire.schema != "feat-math.feature-row-model"
            || wire.format_version != 2
            || wire.min_reader_version != 2
            || wire.required_features != REQUIRED_FEATURES
            || wire.numerical_policy != "row-protected-v2"
            || wire.profile != PROFILE
            || wire.grammar_sha256 != format!("{:x}", Sha256::digest(PROFILE_CONTRACT.as_bytes()))
            || wire.emitter_feature_identity != "feat-math.feature-row.recipes.v2+wire58"
            || wire.output_schema != ["prediction:f64"]
        {
            return Err(VersionError::UnsupportedVersion(
                "row-vector model envelope".into(),
            ));
        }
        let floats = |v: &[u64]| v.iter().map(|&x| f64::from_bits(x)).collect();
        let model = Self {
            program: Program::from_bytes(&wire.program)?,
            stages: wire.stages,
            ridge: RidgeModel {
                means: floats(&wire.ridge_bits[0]),
                scales: floats(&wire.ridge_bits[1]),
                w: floats(&wire.ridge_bits[2]),
            },
        };
        model.validate()?;
        Ok(model)
    }

    pub fn emit_fitted_cpp17(&self) -> Result<String> {
        crate::advanced_v3::emit_row_vector_cpp17(self)
    }
    pub fn identity(&self) -> VersionResult<String> {
        Ok(format!("{:x}", Sha256::digest(self.to_json()?.as_bytes())))
    }
}

fn validate_rows(rows: &Rows, width: usize) -> Result<()> {
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::Shape(
            "row-vector rows must have the declared input width".into(),
        ));
    }
    Ok(())
}

fn transform(stages: &[FittedStage], rows: &Rows) -> Result<(Rows, Vec<u32>)> {
    let mut result = Vec::with_capacity(rows.len());
    let mut faults = Vec::with_capacity(rows.len());
    for row in rows {
        let mut values = row.clone();
        let mut fault = 0;
        for stage in stages {
            let output = crate::ops::apply_row_vector_v2(stage, &values)?;
            values = output.values;
            fault |= output.faults.iter().fold(0, |a, b| a | b);
        }
        // Also protect the removable-stage/no-transform program.
        for value in &mut values {
            let (x, bits) = savgol::normalize(*value);
            *value = x;
            fault |= bits;
        }
        result.push(values);
        faults.push(fault);
    }
    Ok((result, faults))
}
