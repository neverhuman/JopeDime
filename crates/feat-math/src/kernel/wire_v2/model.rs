use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWireV2 {
    schema: String,
    format_version: u8,
    min_reader_version: u8,
    required_features: u64,
    numerical_policy: String,
    grammar_sha256: String,
    emitter_feature_identity: String,
    output_schema: Vec<String>,
    program: Vec<u8>,
    model: Value,
}

#[derive(Clone, Debug)]
pub struct ReservedModel {
    wire: ModelWireV2,
}

impl ReservedModel {
    pub fn new(
        model: &Model,
        program: &ReservedProgram,
        grammar_sha256: String,
        emitter_feature_identity: String,
        output_schema: Vec<String>,
    ) -> VersionResult<Self> {
        let wire = ModelWireV2 {
            schema: "feat-math.model-envelope".into(),
            format_version: 2,
            min_reader_version: READER_VERSION,
            required_features: RESERVATION_FEATURE,
            numerical_policy: "v2".into(),
            grammar_sha256,
            emitter_feature_identity,
            output_schema,
            program: program.to_bytes()?,
            model: serde_json::to_value(model)
                .map_err(|error| VersionError::Malformed(error.to_string()))?,
        };
        validate_model_wire(&wire)?;
        Ok(Self { wire })
    }

    pub fn to_json(&self) -> VersionResult<String> {
        validate_model_wire(&self.wire)?;
        let json = serde_json::to_string(&self.wire)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("model envelope bytes".into()));
        }
        Ok(json)
    }
}

fn validate_model_wire(wire: &ModelWireV2) -> VersionResult<()> {
    if wire.schema != "feat-math.model-envelope" || wire.format_version != 2 {
        return Err(VersionError::UnsupportedVersion("model envelope".into()));
    }
    if wire.min_reader_version != READER_VERSION {
        return Err(VersionError::UnsupportedVersion(
            "model minimum reader version".into(),
        ));
    }
    if wire.required_features != RESERVATION_FEATURE || wire.numerical_policy != "v2" {
        return Err(VersionError::UnsupportedVersion(
            "model required feature set or numerical policy".into(),
        ));
    }
    if wire.grammar_sha256.len() != 64
        || !wire
            .grammar_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || wire.emitter_feature_identity.is_empty()
        || wire.output_schema.is_empty()
    {
        return Err(VersionError::Malformed("model identity metadata".into()));
    }
    if !matches!(
        VersionedProgram::from_bytes(&wire.program)?,
        VersionedProgram::ReservedV2(_)
    ) {
        return Err(VersionError::UnsupportedVersion(
            "model requires a v2 program".into(),
        ));
    }
    serde_json::from_value::<Model>(wire.model.clone())
        .map_err(|error| VersionError::Malformed(format!("model payload: {error}")))?;
    Ok(())
}

#[derive(Clone, Debug)]
pub enum VersionedModel {
    LegacyV1(Model),
    ReservedV2(ReservedModel),
    RowVectorV2(super::row_vector::Model),
    DifferenceV2(Box<DifferenceModel>),
    IntegrationV2(Box<super::cumulative_integration::IntegrationModel>),
    ScalarV2(Box<ScalarModel>),
    TimeGradientV2(Box<super::timescale_nabla::TimeGradientModel>),
    PathStatsV2(Box<super::realized_path_stats::PathStatsModel>),
}

impl VersionedModel {
    /// Scalar predictions retain per-row faults and do not update fit state.
    pub fn predict_scalar(&self, rows: &crate::Rows) -> VersionResult<(Vec<f64>, Vec<u32>)> {
        match self {
            Self::ScalarV2(model) => Ok(model.predict(rows)?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no scalar prediction route".into(),
            )),
        }
    }

    /// Fitted ordered export starts each caller from the captured continuation.
    pub fn emit_ordered_fitted_cpp17(&self) -> VersionResult<DifferenceFittedCpp> {
        match self {
            Self::DifferenceV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no ordered fitted export route".into(),
            )),
        }
    }

    /// B14 fitted integration export preserves its typed checkpoint codec.
    pub fn emit_integral_fitted_cpp17(
        &self,
    ) -> VersionResult<super::cumulative_integration::IntegrationFittedCpp> {
        match self {
            Self::IntegrationV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no integral fitted export route".into(),
            )),
        }
    }

    pub fn emit_time_gradient_fitted_cpp17(
        &self,
    ) -> VersionResult<super::timescale_nabla::TimeGradientFittedCpp> {
        match self {
            Self::TimeGradientV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no time_gradient fitted export route".into(),
            )),
        }
    }

    pub fn emit_path_stats_fitted_cpp17(
        &self,
    ) -> VersionResult<super::realized_path_stats::PathStatsFittedCpp> {
        match self {
            Self::PathStatsV2(model) => model.emit_fitted_cpp17(),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no path_stats fitted export route".into(),
            )),
        }
    }

    /// This is the ingress for tagged model/bundle JSON. The old direct serde
    /// Model loader remains byte-for-byte unchanged for legacy callers.
    pub fn from_json(json: &str) -> VersionResult<Self> {
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("model envelope bytes".into()));
        }
        let value: Value = serde_json::from_str(json)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        // B05–B07: scalar envelope reader runs before any legacy dispatch.
        if value.get("schema").and_then(Value::as_str)
            == Some(crate::kernel::scalar_program::MODEL_SCHEMA)
        {
            return ScalarModel::from_json(json)
                .map(Box::new)
                .map(Self::ScalarV2);
        }
        // B12: tagged model ingress, with no v1 deserialization fallback.
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.feature-row-model") {
            return super::row_vector::Model::from_json(json).map(Self::RowVectorV2);
        }
        if value.get("schema").is_none() && value.get("format_version").is_none() {
            return serde_json::from_value(value)
                .map(Self::LegacyV1)
                .map_err(|error| VersionError::Malformed(error.to_string()));
        }
        if value.get("format_version").and_then(Value::as_u64) != Some(2) {
            return Err(VersionError::UnsupportedVersion("model envelope".into()));
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-difference-model")
        {
            return DifferenceModel::from_json(json)
                .map(Box::new)
                .map(Self::DifferenceV2);
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-integral-model") {
            return super::cumulative_integration::IntegrationModel::from_json(json)
                .map(Box::new)
                .map(Self::IntegrationV2);
        }
        if value.get("schema").and_then(Value::as_str)
            == Some("feat-math.causal-time-gradient-model")
        {
            return super::timescale_nabla::TimeGradientModel::from_json(json)
                .map(Box::new)
                .map(Self::TimeGradientV2);
        }
        if value.get("schema").and_then(Value::as_str) == Some("feat-math.causal-path-stats-model")
        {
            return super::realized_path_stats::PathStatsModel::from_json(json)
                .map(Box::new)
                .map(Self::PathStatsV2);
        }
        let wire: ModelWireV2 = serde_json::from_value(value)
            .map_err(|error| VersionError::Malformed(error.to_string()))?;
        validate_model_wire(&wire)?;
        Ok(Self::ReservedV2(ReservedModel { wire }))
    }

    pub fn predict_ordered(&mut self, rows: &[OrderedRow]) -> VersionResult<Vec<f64>> {
        match self {
            Self::DifferenceV2(model) => Ok(model.predict(rows)?),
            Self::IntegrationV2(model) => Ok(model.predict(rows)?),
            Self::TimeGradientV2(model) => Ok(model.predict(rows)?),
            Self::PathStatsV2(model) => Ok(model.predict(rows)?),
            _ => Err(VersionError::UnsupportedVersion(
                "model has no ordered prediction route".into(),
            )),
        }
    }

    pub fn legacy_model(&self) -> VersionResult<&Model> {
        match self {
            Self::LegacyV1(model) => Ok(model),
            Self::ScalarV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 scalar model cannot decode as a v1 model".into(),
            )),
            Self::RowVectorV2(_) => Err(VersionError::UnsupportedVersion(
                "v2 row-vector model cannot decode as a v1 model".into(),
            )),
            Self::DifferenceV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for difference prediction".into(),
            )),
            Self::IntegrationV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for integral prediction".into(),
            )),
            Self::TimeGradientV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for time_gradient prediction".into(),
            )),
            Self::PathStatsV2(_) => Err(VersionError::UnsupportedVersion(
                "ordered input required for path_stats prediction".into(),
            )),
            Self::ReservedV2(_) => Err(VersionError::UnsupportedVersion(
                "version 2 models are not executable".into(),
            )),
        }
    }
}
