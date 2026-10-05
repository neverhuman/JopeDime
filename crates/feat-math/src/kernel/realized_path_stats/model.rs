//! Train-only downstream fit and transactional causal prediction.
use super::model_validation::{validate_compat_ridge, validate_fit_closure};
use super::program::{PROFILE_ID, REQUIRED_FEATURE};
use super::*;
use crate::advanced_v3::{BoundModel, RawTable};
use crate::kernel::codec::MAX_BYTES;
use crate::kernel::wire_v2::{VersionError, VersionResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct PathStatsModel {
    pub(super) program: PathStatsProgram,
    pub(super) task: TaskMode,
    pub(super) fold: FoldBoundary,
    pub(super) state: PathState,
    pub(super) predictor: BoundModel,
}

impl std::fmt::Debug for PathStatsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathStatsModel")
            .field("program", &self.program)
            .field("task", &self.task)
            .field("fold", &self.fold)
            .finish_non_exhaustive()
    }
}

pub struct PathStatsPrediction {
    pub values: Vec<f64>,
    /// Ordered-stage faults, one word per output statistic and observation.
    pub faults: Vec<Vec<u32>>,
}

impl PathStatsProgram {
    /// The caller supplies exactly the training prefix and its labels. Held-out
    /// rows and labels cannot enter this API; total_rows declares future carry.
    pub fn fit_training(
        &self,
        rows: &[OrderedRow],
        labels: &[f64],
        task: TaskMode,
        fold: FoldBoundary,
    ) -> Result<PathStatsModel> {
        if rows.len() != fold.train_end
            || labels.len() != fold.train_end
            || labels.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Shape(
                "path-stats fit needs exactly the train prefix and finite labels".into(),
            ));
        }
        validate_fit_closure(self.downstream())?;
        let mut state = PathState::new(self.config, task, &self.input, fold.clone())?;
        let mut transformed = Vec::with_capacity(rows.len());
        for row in rows {
            transformed.push(
                state
                    .push(row.group.clone(), row.timestamp, &row.values)?
                    .values
                    .to_vec(),
            );
        }
        let predictor = crate::fit_legacy_numeric_bound(&self.downstream, &transformed, labels)
            .map_err(|e| Error::Domain(e.to_string()))?;
        let model = PathStatsModel {
            program: self.clone(),
            task,
            fold,
            state,
            predictor,
        };
        // A successful fit must produce finite, losslessly replayable state.
        model.to_json().map_err(|e| Error::Domain(e.to_string()))?;
        Ok(model)
    }
}

impl PathStatsModel {
    pub fn program(&self) -> &PathStatsProgram {
        &self.program
    }
    /// Commit cursor/history only after the whole batch and predictor succeed.
    pub fn predict(&mut self, rows: &[OrderedRow]) -> Result<Vec<f64>> {
        Ok(self.predict_with_faults(rows)?.values)
    }

    pub fn predict_with_faults(&mut self, rows: &[OrderedRow]) -> Result<PathStatsPrediction> {
        if rows.is_empty() {
            return Ok(PathStatsPrediction {
                values: Vec::new(),
                faults: Vec::new(),
            });
        }
        let mut next = self.state.clone();
        let mut transformed = Vec::with_capacity(rows.len());
        let mut faults = Vec::with_capacity(rows.len());
        for row in rows {
            let output = next.push(row.group.clone(), row.timestamp, &row.values)?;
            transformed.push(output.values.to_vec());
            faults.push(output.faults.to_vec());
        }
        let table = RawTable {
            headers: self
                .predictor
                .schema
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect(),
            rows: transformed
                .iter()
                .map(|r| r.iter().map(|v| v.to_string()).collect())
                .collect(),
        };
        let values = self
            .predictor
            .predict_table(&table)
            .map_err(|e| Error::Domain(e.to_string()))?;
        next.to_bytes()?;
        let mut candidate = self.clone();
        candidate.state = next;
        candidate
            .to_json()
            .map_err(|e| Error::Resource(e.to_string()))?;
        self.state = candidate.state;
        Ok(PathStatsPrediction { values, faults })
    }

    pub fn finish(self) -> Result<()> {
        self.state.finish()
    }

    pub fn to_json(&self) -> VersionResult<String> {
        validate_fit_closure(self.program.downstream())?;
        let predictor = serde_json::to_value(&self.predictor)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        validate_compat_ridge(&predictor, self.program.width())?;
        let wire = ModelWire {
            schema: "feat-math.causal-path-stats-model".into(),
            format_version: 2,
            min_reader_version: 2,
            required_features: REQUIRED_FEATURE,
            numerical_policy: "v2".into(),
            grammar_sha256: self.program.identity()?,
            emitter_feature_identity: PROFILE_ID.into(),
            output_schema: vec!["prediction".into()],
            program: self.program.to_bytes()?,
            task: match self.task {
                TaskMode::Ordered => 1,
                TaskMode::Forecasting => 2,
                TaskMode::IID => 0,
            },
            fold: self.fold.clone(),
            state: self.state.to_bytes()?,
            predictor: pack_floats(predictor),
        };
        let json =
            serde_json::to_string(&wire).map_err(|e| VersionError::Malformed(e.to_string()))?;
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("path-stats model bytes".into()));
        }
        Ok(json)
    }

    pub fn from_json(json: &str) -> VersionResult<Self> {
        if json.len() > MAX_BYTES {
            return Err(VersionError::Malformed("path-stats model bytes".into()));
        }
        let wire: ModelWire =
            serde_json::from_str(json).map_err(|e| VersionError::Malformed(e.to_string()))?;
        if wire.schema != "feat-math.causal-path-stats-model"
            || wire.format_version != 2
            || wire.min_reader_version != 2
            || wire.required_features != REQUIRED_FEATURE
            || wire.numerical_policy != "v2"
            || wire.emitter_feature_identity != PROFILE_ID
        {
            return Err(VersionError::UnsupportedVersion(
                "path-stats model envelope/features".into(),
            ));
        }
        let task = match wire.task {
            1 => TaskMode::Ordered,
            2 => TaskMode::Forecasting,
            _ => {
                return Err(VersionError::UnsupportedVersion(
                    "path-stats model task".into(),
                ))
            }
        };
        let program = PathStatsProgram::from_bytes(&wire.program)?;
        validate_fit_closure(program.downstream())?;
        if wire.grammar_sha256 != program.identity()? || wire.output_schema != ["prediction"] {
            return Err(VersionError::Malformed("path-stats model identity".into()));
        }
        let state = PathState::from_bytes(&wire.state, task, &program.input, wire.fold.clone())?;
        if state.config.window != program.config.window
            || state.cursor.ordinal() < wire.fold.train_end
        {
            return Err(VersionError::Malformed(
                "path-stats model state binding".into(),
            ));
        }
        let predictor_value = unpack_floats(wire.predictor)?;
        validate_compat_ridge(&predictor_value, program.width())?;
        let predictor: BoundModel = serde_json::from_value(predictor_value)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        predictor
            .validate_binding()
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        if predictor.task.kind != crate::advanced_v3::TaskKind::Regression
            || predictor.schema.row_semantics != crate::advanced_v3::RowSemantics::Iid
            || predictor
                .schema
                .columns
                .iter()
                .enumerate()
                .any(|(i, column)| {
                    column.kind != crate::advanced_v3::ColumnKind::Numeric
                        || column.name != format!("c{i}")
                })
        {
            return Err(VersionError::Malformed(
                "path-stats numeric downstream schema/task".into(),
            ));
        }
        let genome_sha = Sha256::digest(program.downstream.canonical().as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if predictor.schema.columns.len() != program.width()
            || predictor.genome_sha256 != genome_sha
        {
            return Err(VersionError::Malformed(
                "path-stats downstream binding".into(),
            ));
        }
        let model = Self {
            program,
            task,
            fold: wire.fold,
            state,
            predictor,
        };
        if model.to_json()? != json {
            return Err(VersionError::Malformed(
                "noncanonical path-stats model".into(),
            ));
        }
        Ok(model)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelWire {
    schema: String,
    format_version: u8,
    min_reader_version: u8,
    required_features: u64,
    numerical_policy: String,
    grammar_sha256: String,
    emitter_feature_identity: String,
    output_schema: Vec<String>,
    program: Vec<u8>,
    task: u8,
    fold: FoldBoundary,
    state: Vec<u8>,
    predictor: Value,
}

// JSON numbers carrying f64 are replaced by integer bit words. serde's Value
// decoder restores f64 directly, avoiding decimal parser rounding in fit state.
fn pack_floats(value: Value) -> Value {
    match value {
        Value::Number(ref n) if n.is_f64() => {
            serde_json::json!({"f64_bits":n.as_f64().unwrap().to_bits()})
        }
        Value::Array(values) => Value::Array(values.into_iter().map(pack_floats).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(k, v)| (k, pack_floats(v)))
                .collect(),
        ),
        other => other,
    }
}
fn unpack_floats(value: Value) -> VersionResult<Value> {
    Ok(match value {
        Value::Object(mut values) if values.len() == 1 && values.contains_key("f64_bits") => {
            let bits = values
                .remove("f64_bits")
                .unwrap()
                .as_u64()
                .ok_or_else(|| VersionError::Malformed("model f64 word".into()))?;
            Value::Number(
                serde_json::Number::from_f64(f64::from_bits(bits))
                    .ok_or_else(|| VersionError::Malformed("nonfinite fitted word".into()))?,
            )
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(unpack_floats)
                .collect::<VersionResult<_>>()?,
        ),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(k, v)| Ok((k, unpack_floats(v)?)))
                .collect::<VersionResult<_>>()?,
        ),
        other => other,
    })
}
