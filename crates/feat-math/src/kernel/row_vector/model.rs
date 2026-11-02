use super::*;

#[derive(Clone, Debug)]
pub struct Model {
    pub(super) program: Program,
    pub(super) stages: Vec<FittedStage>,
    pub(super) ridge: RidgeModel,
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

pub(super) fn savgol_stages(stages: &[RowStage]) -> Vec<usize> {
    (0..stages.len())
        .filter(|&i| matches!(stages[i], RowStage::Savgol(_)))
        .collect()
}

pub(super) fn goertzel_stages(stages: &[RowStage]) -> Vec<usize> {
    (0..stages.len())
        .filter(|&i| matches!(stages[i], RowStage::Goertzel(_)))
        .collect()
}

/// The exact grid k/128 for k = 0..=64 (dividing by a power of two is exact).
pub(super) fn grid_frequency(draw: u32) -> f64 {
    f64::from(draw % 65) / 128.0
}

pub(super) fn goertzel_config(frequencies: &[f64]) -> Result<goertzel::Config> {
    goertzel::Config::new(
        2,
        61,
        super::ordered_axis::Axis::FeatureWithinRow,
        frequencies,
    )
}

pub(super) fn validate_rows(rows: &Rows, width: usize) -> Result<()> {
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::Shape(
            "row-vector rows must have the declared input width".into(),
        ));
    }
    Ok(())
}

pub(super) fn transform(stages: &[FittedStage], rows: &Rows) -> Result<(Rows, Vec<u32>)> {
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
