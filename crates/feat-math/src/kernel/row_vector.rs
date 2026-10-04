//! Explicit v2 FeatureWithinRow pipelines. The legacy generator and grammar
//! are unchanged. This bounded Experimental profile admits at most eight
//! row stages followed by the existing ridge learner, with train-only fit.
use super::codec::{Reader, Writer, MAX_BYTES};
use super::goertzel;
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
// B13: a program that holds a goertzel_bank stage also declares this bit and
// the bank's grammar/emitter identities; savgol-only artifacts are unchanged.
pub const GOERTZEL_FEATURE: u64 = 128;
const GOERTZEL_PROFILE_CONTRACT: &str = "feat-math.feature-row.experimental.v2;axis=FeatureWithinRow;max_stages=8;prior_max=0.25;head=ridge-v1;wire61=goertzel_bank;bins=1..32;width<=512";
const EMITTER_IDENTITY: &str = "feat-math.feature-row.recipes.v2+wire58";
const GOERTZEL_EMITTER_IDENTITY: &str = "feat-math.feature-row.recipes.v2+wire58+wire61";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "wire", content = "parameters", deny_unknown_fields)]
pub enum RowStage {
    #[serde(rename = "58")]
    Savgol(savgol::Config),
    // B13: sparse Goertzel bank; its output width is the bank length.
    #[serde(rename = "61")]
    Goertzel(goertzel::Config),
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
    #[serde(rename = "61")]
    Goertzel(goertzel::Fitted),
}

impl RowStage {
    fn fit(&self, width: usize) -> Result<FittedStage> {
        match self {
            Self::Savgol(config) => Ok(FittedStage::Savgol(config.fit(width)?)),
            Self::Goertzel(config) => Ok(FittedStage::Goertzel(config.fit(width)?)),
        }
    }
}

impl FittedStage {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Savgol(state) => state.validate(),
            Self::Goertzel(state) => state.validate(),
        }
    }
    pub fn input_width(&self) -> usize {
        match self {
            Self::Savgol(state) => state.width(),
            Self::Goertzel(state) => state.width(),
        }
    }
    pub fn output_width(&self) -> usize {
        match self {
            Self::Savgol(_) => self.input_width(),
            Self::Goertzel(state) => state.output_width(),
        }
    }
    pub(crate) fn apply(&self, row: &[f64]) -> Result<savgol::Output> {
        match self {
            Self::Savgol(state) => state.apply_validated(row),
            Self::Goertzel(state) => state.apply_validated(row),
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

    /// B12's feature 32, plus [`GOERTZEL_FEATURE`] when a goertzel_bank stage is present.
    pub fn required_features(&self) -> u64 {
        if self.has_goertzel() {
            REQUIRED_FEATURES | GOERTZEL_FEATURE
        } else {
            REQUIRED_FEATURES
        }
    }

    fn has_goertzel(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| matches!(stage, RowStage::Goertzel(_)))
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
        writer.word(self.required_features())?;
        writer.text(
            &serde_json::to_string(self).map_err(|e| VersionError::Malformed(e.to_string()))?,
        )?;
        Ok(writer.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        let envelope =
            reader.take(MAGIC_V2.len())? == MAGIC_V2 && reader.byte()? == 3 && reader.byte()? == 2;
        let features = if envelope { reader.word()? } else { 0 };
        if !envelope
            || (features != REQUIRED_FEATURES && features != REQUIRED_FEATURES | GOERTZEL_FEATURE)
        {
            return Err(VersionError::UnsupportedVersion(
                "row-vector envelope/feature set".into(),
            ));
        }
        let parameters: serde_json::Value = serde_json::from_str(&reader.text()?)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        reader.finish()?;
        check_stage_wires(&parameters)?;
        let value: Self = serde_json::from_value(parameters)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let lambda = value.lambda();
        let program = Self::new(2, value.input_width, value.stages, lambda)?;
        // B13: the declared feature set must be exactly what the stages require.
        if program.required_features() != features {
            return Err(VersionError::UnsupportedVersion(
                "row-vector feature set differs from its stages".into(),
            ));
        }
        Ok(program)
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
            Edit::Change if profile.savgol && !savgol_stages(&stages).is_empty() => {
                // B13: only savgol stages are eligible; a savgol-only program
                // draws the same stage as before.
                let eligible = savgol_stages(&stages);
                let at = eligible[rng.next_u32() as usize % eligible.len()];
                let RowStage::Savgol(config) = &stages[at] else {
                    return Err(Error::Unsupported("row-vector savgol edit".into()));
                };
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
            Edit::Drop if !savgol_stages(&stages).is_empty() => {
                let eligible = savgol_stages(&stages);
                let at = eligible[rng.next_u32() as usize % eligible.len()];
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
        let arm = if savgol_stages(&self.stages).is_empty() {
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

    /// B13: seeded GP insert/edit/remove of goertzel_bank stages under their own
    /// removable switch. Grow draws one to four frequencies from the exact grid
    /// k/128 (k = 0..=64); Change inserts, removes or replaces one bin and always
    /// alters the bank; Drop stays available when generation is disabled.
    pub fn edit_goertzel(&self, seed: u64, arm: Edit, profile: &GoertzelProfile) -> Result<Self> {
        profile.validate()?;
        let mut stages = self.stages.clone();
        let mut rng = Pcg32::new(seed, 0x726f77676f727a32);
        let banks = goertzel_stages(&stages);
        match arm {
            Edit::Grow if profile.goertzel && stages.len() < MAX_STAGES => {
                let bins = 1 + rng.next_u32() as usize % 4;
                let frequencies = (0..bins)
                    .map(|_| grid_frequency(rng.next_u32()))
                    .collect::<Vec<_>>();
                let at = rng.next_u32() as usize % (stages.len() + 1);
                stages.insert(at, RowStage::Goertzel(goertzel_config(&frequencies)?));
            }
            Edit::Change if profile.goertzel && !banks.is_empty() => {
                let at = banks[rng.next_u32() as usize % banks.len()];
                let RowStage::Goertzel(config) = &stages[at] else {
                    return Err(Error::Unsupported("row-vector goertzel edit".into()));
                };
                let mut frequencies = config.frequencies();
                let slot = rng.next_u32() as usize % frequencies.len();
                match rng.next_u32() % 3 {
                    0 if frequencies.len() < goertzel::MAX_BINS => {
                        frequencies.insert(slot, grid_frequency(rng.next_u32()));
                    }
                    1 if frequencies.len() > 1 => {
                        frequencies.remove(slot);
                    }
                    _ => {
                        let draw = rng.next_u32();
                        let next = grid_frequency(draw);
                        frequencies[slot] = if next == frequencies[slot] {
                            grid_frequency(draw.wrapping_add(1))
                        } else {
                            next
                        };
                    }
                }
                stages[at] = RowStage::Goertzel(goertzel_config(&frequencies)?);
            }
            Edit::Drop if !banks.is_empty() => {
                let at = banks[rng.next_u32() as usize % banks.len()];
                stages.remove(at);
            }
            _ => {
                return Err(Error::Unsupported(
                    "goertzel_bank GP arm/profile unavailable".into(),
                ))
            }
        }
        Self::new(2, self.input_width, stages, self.lambda())
    }

    /// B13: the bank's own prior gate, on a stream separate from B12's.
    pub fn mutate_goertzel(&self, seed: u64, profile: &GoertzelProfile) -> Result<Self> {
        profile.validate()?;
        let mut rng = Pcg32::new(seed, 0x726f77676f707231);
        if !profile.goertzel || rng.next_f64() >= profile.prior {
            return Ok(self.clone());
        }
        let arm = if goertzel_stages(&self.stages).is_empty() {
            Edit::Grow
        } else {
            match rng.next_u32() % 3 {
                0 if self.stages.len() < MAX_STAGES => Edit::Grow,
                1 => Edit::Change,
                _ => Edit::Drop,
            }
        };
        self.edit_goertzel(u64::from(rng.next_u32()), arm, profile)
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

/// B13: the goertzel_bank search switch, separate from B12's savgol profile.
#[derive(Clone, Debug)]
pub struct GoertzelProfile {
    pub goertzel: bool,
    pub prior: f64,
}
impl GoertzelProfile {
    pub fn validate(&self) -> Result<()> {
        if !self.prior.is_finite() || self.prior <= 0.0 || self.prior > 0.25 {
            return Err(Error::Domain(
                "goertzel_bank Experimental prior must be in (0,0.25]".into(),
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
        let bank = self.program.has_goertzel();
        let wire = ModelWire {
            schema: "feat-math.feature-row-model".into(),
            format_version: 2,
            min_reader_version: 2,
            required_features: self.program.required_features(),
            numerical_policy: "row-protected-v2".into(),
            profile: PROFILE.into(),
            grammar_sha256: grammar_sha256(bank),
            emitter_feature_identity: emitter_identity(bank).into(),
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
        let parameters: serde_json::Value =
            serde_json::from_str(text).map_err(|e| VersionError::Malformed(e.to_string()))?;
        check_stage_wires(&parameters)?;
        let wire: ModelWire = serde_json::from_value(parameters)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let bank = wire.required_features == REQUIRED_FEATURES | GOERTZEL_FEATURE;
        if wire.schema != "feat-math.feature-row-model"
            || wire.format_version != 2
            || wire.min_reader_version != 2
            || (wire.required_features != REQUIRED_FEATURES && !bank)
            || wire.numerical_policy != "row-protected-v2"
            || wire.profile != PROFILE
            || wire.grammar_sha256 != grammar_sha256(bank)
            || wire.emitter_feature_identity != emitter_identity(bank)
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
        // B13: the envelope's feature set must be the one its program requires.
        if model.program.required_features() != wire.required_features {
            return Err(VersionError::UnsupportedVersion(
                "row-vector model feature set differs from its program".into(),
            ));
        }
        Ok(model)
    }

    pub fn emit_fitted_cpp17(&self) -> Result<String> {
        crate::advanced_v3::emit_row_vector_cpp17(self)
    }
    pub fn identity(&self) -> VersionResult<String> {
        Ok(format!("{:x}", Sha256::digest(self.to_json()?.as_bytes())))
    }
}

fn check_stage_wires(parameters: &serde_json::Value) -> VersionResult<()> {
    if let Some(stages) = parameters
        .get("stages")
        .and_then(serde_json::Value::as_array)
    {
        for stage in stages {
            if let Some(wire) = stage.get("wire").and_then(serde_json::Value::as_str) {
                if wire != "58" && wire != "61" {
                    return Err(VersionError::UnsupportedVersion(format!(
                        "row-vector wire {wire} is unavailable"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn grammar_sha256(bank: bool) -> String {
    let contract = if bank {
        GOERTZEL_PROFILE_CONTRACT
    } else {
        PROFILE_CONTRACT
    };
    format!("{:x}", Sha256::digest(contract.as_bytes()))
}

fn emitter_identity(bank: bool) -> &'static str {
    if bank {
        GOERTZEL_EMITTER_IDENTITY
    } else {
        EMITTER_IDENTITY
    }
}

fn savgol_stages(stages: &[RowStage]) -> Vec<usize> {
    (0..stages.len())
        .filter(|&i| matches!(stages[i], RowStage::Savgol(_)))
        .collect()
}

fn goertzel_stages(stages: &[RowStage]) -> Vec<usize> {
    (0..stages.len())
        .filter(|&i| matches!(stages[i], RowStage::Goertzel(_)))
        .collect()
}

/// The exact grid k/128 for k = 0..=64 (dividing by a power of two is exact).
fn grid_frequency(draw: u32) -> f64 {
    f64::from(draw % 65) / 128.0
}

fn goertzel_config(frequencies: &[f64]) -> Result<goertzel::Config> {
    goertzel::Config::new(
        2,
        61,
        super::ordered_axis::Axis::FeatureWithinRow,
        frequencies,
    )
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
