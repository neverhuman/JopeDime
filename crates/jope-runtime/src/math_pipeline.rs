//! Decoder `org.jopedime/math-pipeline/v1`: a fitted `feat_math::model::Model` plus the numeric
//! input contract it was fitted behind. See `docs/model-object-format.md#math-pipeline-v1`.
//!
//! Unversioned/`prediction_version: 1` sections reproduce the producer's f32 arithmetic.
//! New producers use [`InputContract::new`] / `prediction_version: 2` for f64 encoding,
//! prediction and target inversion, with numerical failures instead of non-finite replacement.
//! No stored object is rewritten; legacy JavaScript and narration bindings are preserved.
//! Both versions match columns by name, scale/clamp cells to `[0,1]`, select the recorded columns
//! and map predictions back to target units. Missing-cell and constant-target policies stay intact.

use crate::error::Failure;
use crate::table::CsvTable;
use anyhow::Result;
use feat_math::genome::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PIPELINE_PART: &str = "pipeline.json";
pub const INPUT_PART: &str = "input.json";
pub const F64_PREDICTION_VERSION: u32 = 2;
/// More input columns than any table this runtime reads (`Limits::max_columns` is lower).
const MAX_COLUMNS: usize = 100_000;

/// `pipeline.json`: the fitted pipeline and the genome it was fitted from.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineState {
    pub dsl: String,
    pub model: feat_math::model::Model,
}

/// `input.json`: how raw cells become the pipeline's input and how its output becomes a prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputContract<N = f64> {
    /// Only `regression` in v1.
    pub task: String,
    /// Every encoded column, in encoded order.
    pub columns: Vec<NumericColumn<N>>,
    /// Indices into `columns`, in the order the pipeline reads them.
    pub selected: Vec<usize>,
    pub target: TargetScale<N>,
    /// Absent/1 preserves legacy f32 semantics; new producers must explicitly write 2.
    #[serde(default = "legacy_version", skip_serializing_if = "is_legacy_version")]
    pub prediction_version: u32,
}

fn legacy_version() -> u32 {
    1
}
fn is_legacy_version(version: &u32) -> bool {
    *version == 1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericColumn<N = f64> {
    pub name: String,
    pub min: N,
    pub max: N,
    /// A constant column encodes every cell as 0.
    pub constant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetScale<N = f64> {
    pub name: String,
    pub min: N,
    /// At or below 1e-12 every prediction is `min`.
    pub scale: N,
}

impl<N> InputContract<N> {
    fn map_numbers<M>(self, convert: impl Fn(N) -> M) -> InputContract<M> {
        InputContract {
            task: self.task,
            columns: self
                .columns
                .into_iter()
                .map(|c| NumericColumn {
                    name: c.name,
                    min: convert(c.min),
                    max: convert(c.max),
                    constant: c.constant,
                })
                .collect(),
            selected: self.selected,
            target: TargetScale {
                name: self.target.name,
                min: convert(self.target.min),
                scale: convert(self.target.scale),
            },
            prediction_version: self.prediction_version,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MathPipeline {
    pub input: InputContract,
    pub state: PipelineState,
    genome: GenomeIr,
}

impl MathPipeline {
    /// Decode and validate both parts. A model whose width differs from `selected` is refused here,
    /// before any table is read.
    pub fn decode(parts: &BTreeMap<String, Vec<u8>>, primary: &str) -> Result<Self> {
        let names: BTreeSet<&str> = parts.keys().map(String::as_str).collect();
        if primary != PIPELINE_PART || names != BTreeSet::from([PIPELINE_PART, INPUT_PART]) {
            return Err(Failure::integrity(format!(
                "math-pipeline/v1 has exactly the parts {PIPELINE_PART} (primary) and {INPUT_PART}"
            )));
        }
        let state: PipelineState = crate::strict_json::decode(&parts[PIPELINE_PART])
            .map_err(|e| Failure::integrity(format!("{PIPELINE_PART}: {e}")))?;
        let input = decode_input(&parts[INPUT_PART])?;
        input.validate()?;
        let genome = GenomeIr::parse(&state.dsl)
            .map_err(|e| Failure::integrity(format!("{PIPELINE_PART} dsl: {e:#}")))?;
        feat_math::kernel::width::check_input_width(&state.model, input.selected.len()).map_err(
            |e| {
                Failure::integrity(format!(
                    "fitted pipeline does not read {} selected columns: {e}",
                    input.selected.len()
                ))
            },
        )?;
        Ok(Self {
            input,
            state,
            genome,
        })
    }

    /// Predict every row of a table. Columns are matched by name in any order; the target and any
    /// other column the contract does not name are ignored and returned.
    pub fn predict_table(&self, table: &CsvTable) -> Result<(Vec<f64>, Vec<String>)> {
        crate::prediction_narration::from_env(|tap| self.predict_table_with_narration(table, tap))
    }

    /// Explicit observer for the same shipped GP prediction path.
    pub fn predict_table_with_narration(
        &self,
        table: &CsvTable,
        tap: Option<&jope_narrator::Tap>,
    ) -> Result<(Vec<f64>, Vec<String>)> {
        crate::prediction_narration::observe(
            tap,
            || {
                Ok((
                    self.narration_model_identity()?,
                    crate::prediction_narration::identity(
                        b"jope.math-prediction-table/v2",
                        &(&table.headers, &table.rows),
                    )?,
                ))
            },
            table.rows.len(),
            || self.predict_table_plain(table),
        )
    }

    fn predict_table_plain(&self, table: &CsvTable) -> Result<(Vec<f64>, Vec<String>)> {
        let mut at = Vec::with_capacity(self.input.columns.len());
        for column in &self.input.columns {
            let index = table
                .headers
                .iter()
                .position(|h| h == &column.name)
                .ok_or_else(|| Failure::input(format!("input lacks column {:?}", column.name)))?;
            at.push(index);
        }
        let ignored = table
            .headers
            .iter()
            .filter(|h| !self.input.columns.iter().any(|c| &c.name == *h))
            .cloned()
            .collect();
        let rows: Vec<Vec<Option<&str>>> = table
            .rows
            .iter()
            .map(|row| at.iter().map(|&i| row[i].as_deref()).collect())
            .collect();
        Ok((self.predict_cells_plain(&rows)?, ignored))
    }

    /// Predict rows of raw cells given in `columns` order; `None` is a missing cell.
    pub fn predict_cells(&self, rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>> {
        crate::prediction_narration::from_env(|tap| {
            crate::prediction_narration::observe(
                tap,
                || {
                    Ok((
                        self.narration_model_identity()?,
                        crate::prediction_narration::identity(
                            b"jope.math-prediction-cells/v2",
                            &rows,
                        )?,
                    ))
                },
                rows.len(),
                || self.predict_cells_plain(rows),
            )
        })
    }

    fn predict_cells_plain(&self, rows: &[Vec<Option<&str>>]) -> Result<Vec<f64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let width = self.input.columns.len();
        let x: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| {
                if row.len() != width {
                    return Err(Failure::input(format!(
                        "row has {} cells; the model reads {width}",
                        row.len()
                    )));
                }
                self.input
                    .selected
                    .iter()
                    .map(|&j| {
                        if self.input.prediction_version == 1 {
                            Ok(f64::from(encode(row[j], &self.input.columns[j])))
                        } else {
                            encode_f64(row[j], &self.input.columns[j])
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        let raw = self
            .state
            .model
            .predict(&x)
            .map_err(|e| Failure::integrity(format!("fitted pipeline: {e}")))?;
        raw.into_iter()
            .enumerate()
            .map(|(row, v)| {
                if self.input.prediction_version == 1 {
                    let v = if v.is_finite() { v as f32 } else { 0.0 };
                    return Ok(f64::from(self.input.target.inverse_legacy(v)));
                }
                if !v.is_finite() {
                    return Err(undefined(format!(
                        "row {row}: non-finite fitted prediction"
                    )));
                }
                let value = self.input.target.inverse_f64(v);
                if !value.is_finite() {
                    return Err(undefined(format!(
                        "row {row}: non-finite target prediction"
                    )));
                }
                Ok(value)
            })
            .collect()
    }

    fn narration_model_identity(&self) -> Result<String> {
        // Preserve legacy f32 JSON rounding and omit the default section version.
        if self.input.prediction_version == 1 {
            crate::prediction_narration::identity(
                b"jope.math-prediction-model/v2",
                &(self.input.clone().map_numbers(|v| v as f32), &self.state),
            )
        } else {
            crate::prediction_narration::identity(
                b"jope.math-prediction-model/v2",
                &(&self.input, &self.state),
            )
        }
    }

    /// Standalone ES2020: the fitted pipeline, this contract's encoding, and `predictRows`.
    pub fn export_javascript(&self) -> Result<Vec<u8>> {
        let mut source =
            crate::fitted_js::emit(&self.genome, &self.state.model, self.input.selected.len())?;
        let columns: Vec<_> = self.input.columns.iter().map(|c| &c.name).collect();
        let scalers: Vec<_> = self
            .input
            .columns
            .iter()
            .map(|c| serde_json::json!([c.min, c.max, c.constant]))
            .collect();
        let (header, contract) = if self.input.prediction_version == 1 {
            (JS_CONTRACT_HEADER, JS_CONTRACT)
        } else {
            ("\n// Input contract prediction_version=2: f64 arithmetic, finite results required.\n", JS_CONTRACT_F64)
        };
        source.push_str(&format!(
            "{header}const MODEL_COLUMNS=Object.freeze({});\n\
             const INPUT_SCALERS=Object.freeze({});\n\
             const INPUT_SELECTED=Object.freeze({});\n\
             const TARGET_SCALE=Object.freeze({{min:{},scale:{}}});\n{contract}",
            serde_json::to_string(&columns)?,
            serde_json::to_string(&scalers)?,
            serde_json::to_string(&self.input.selected)?,
            serde_json::to_string(&self.input.target.min)?,
            serde_json::to_string(&self.input.target.scale)?,
        ));
        let hits = feat_math::transpile::lint_shipped_text(&source);
        if !hits.is_empty() {
            return Err(Failure::integrity(format!(
                "banned vocabulary in JavaScript artifact: {}",
                hits.join(", ")
            )));
        }
        Ok(source.into_bytes())
    }
}

mod contract;
use contract::undefined;
pub(super) use contract::{
    decode_input, encode, encode_f64, JS_CONTRACT, JS_CONTRACT_F64, JS_CONTRACT_HEADER,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_narration_keeps_f32_serialization_and_identity() {
        let input = include_bytes!("../tests/fixtures/math-pipeline/input.json");
        let state = include_bytes!("../tests/fixtures/math-pipeline/pipeline.json");
        let parts = BTreeMap::from([
            (INPUT_PART.into(), input.to_vec()),
            (PIPELINE_PART.into(), state.to_vec()),
        ]);
        let model = MathPipeline::decode(&parts, PIPELINE_PART).unwrap();
        let old_input: InputContract<f32> = crate::strict_json::decode(input).unwrap();
        let old_state: PipelineState = crate::strict_json::decode(state).unwrap();
        let expected = crate::prediction_narration::identity(
            b"jope.math-prediction-model/v2",
            &(old_input, old_state),
        )
        .unwrap();
        assert_eq!(model.narration_model_identity().unwrap(), expected);
        assert!(
            !serde_json::to_string(&model.input.clone().map_numbers(|v| v as f32))
                .unwrap()
                .contains("prediction_version")
        );
    }
}
