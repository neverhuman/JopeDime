//! Decoder `org.jopedime/math-pipeline/v1`: a fitted `feat_math::model::Model` plus the numeric
//! input contract it was fitted behind. See `docs/model-object-format.md#math-pipeline-v1`.
//!
//! The contract reproduces one external producer's column encoding exactly, including its f32
//! arithmetic, so a model fitted there predicts here (and in the exported JavaScript) with the
//! same numbers: parse each cell, scale it to [0, 1] with the column's min and max and clamp,
//! pass the selected columns to the fitted pipeline, then map the output back to target units.

use crate::error::Failure;
use crate::table::CsvTable;
use anyhow::Result;
use feat_math::genome::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PIPELINE_PART: &str = "pipeline.json";
pub const INPUT_PART: &str = "input.json";
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
pub struct InputContract {
    /// Only `regression` in v1.
    pub task: String,
    /// Every encoded column, in encoded order.
    pub columns: Vec<NumericColumn>,
    /// Indices into `columns`, in the order the pipeline reads them.
    pub selected: Vec<usize>,
    pub target: TargetScale,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericColumn {
    pub name: String,
    pub min: f32,
    pub max: f32,
    /// A constant column encodes every cell as 0.
    pub constant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetScale {
    pub name: String,
    pub min: f32,
    /// At or below 1e-12 every prediction is `min`.
    pub scale: f32,
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
        let input: InputContract = crate::strict_json::decode(&parts[INPUT_PART])
            .map_err(|e| Failure::integrity(format!("{INPUT_PART}: {e}")))?;
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
                    crate::prediction_narration::identity(
                        b"jope.math-prediction-model/v1",
                        &(&self.input, &self.state),
                    )?,
                    crate::prediction_narration::identity(
                        b"jope.math-prediction-table/v1",
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
                        crate::prediction_narration::identity(
                            b"jope.math-prediction-model/v1",
                            &(&self.input, &self.state),
                        )?,
                        crate::prediction_narration::identity(
                            b"jope.math-prediction-cells/v1",
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
                Ok(self
                    .input
                    .selected
                    .iter()
                    .map(|&j| f64::from(encode(row[j], &self.input.columns[j])))
                    .collect())
            })
            .collect::<Result<_>>()?;
        let raw = self
            .state
            .model
            .predict(&x)
            .map_err(|e| Failure::integrity(format!("fitted pipeline: {e}")))?;
        Ok(raw
            .into_iter()
            .map(|v| {
                let v = if v.is_finite() { v as f32 } else { 0.0 };
                f64::from(self.input.target.inverse(v))
            })
            .collect())
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
            .map(|c| serde_json::json!([f64::from(c.min), f64::from(c.max), c.constant]))
            .collect();
        source.push_str(&format!(
            "{JS_CONTRACT_HEADER}const MODEL_COLUMNS=Object.freeze({});\n\
             const INPUT_SCALERS=Object.freeze({});\n\
             const INPUT_SELECTED=Object.freeze({});\n\
             const TARGET_SCALE=Object.freeze({{min:{},scale:{}}});\n{JS_CONTRACT}",
            serde_json::to_string(&columns)?,
            serde_json::to_string(&scalers)?,
            serde_json::to_string(&self.input.selected)?,
            serde_json::to_string(&f64::from(self.input.target.min))?,
            serde_json::to_string(&f64::from(self.input.target.scale))?,
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

impl InputContract {
    fn validate(&self) -> Result<()> {
        let bad = |message: String| Err(Failure::integrity(format!("{INPUT_PART}: {message}")));
        if self.task != "regression" {
            return bad(format!("task {:?}; v1 decodes regression only", self.task));
        }
        if self.columns.is_empty() || self.columns.len() > MAX_COLUMNS {
            return bad(format!("{} columns", self.columns.len()));
        }
        let mut names = BTreeSet::new();
        for column in &self.columns {
            if column.name.is_empty() || !names.insert(column.name.as_str()) {
                return bad(format!("empty or duplicate column name {:?}", column.name));
            }
            if !column.min.is_finite() || !column.max.is_finite() {
                return bad(format!("column {:?} has a non-finite range", column.name));
            }
            if !column.constant && column.max <= column.min {
                return bad(format!(
                    "column {:?} is not constant but max <= min",
                    column.name
                ));
            }
        }
        let mut seen = BTreeSet::new();
        if self.selected.is_empty()
            || self
                .selected
                .iter()
                .any(|&j| j >= self.columns.len() || !seen.insert(j))
        {
            return bad("selected must be distinct indices into columns".into());
        }
        if !self.target.min.is_finite() || !self.target.scale.is_finite() {
            return bad("target scale is non-finite".into());
        }
        Ok(())
    }
}

impl TargetScale {
    fn inverse(&self, v: f32) -> f32 {
        if self.scale <= 1e-12 {
            self.min
        } else {
            self.min + v * self.scale
        }
    }
}

/// Encode one cell. Missing, non-finite, out-of-f32-range and unparseable cells encode as 0,
/// the scaled column minimum; `true`/`yes` read as 1 and `false`/`no` as 0 before scaling.
fn encode(cell: Option<&str>, column: &NumericColumn) -> f32 {
    let Some(v) = cell.and_then(parse_cell) else {
        return 0.0;
    };
    if column.constant {
        0.0
    } else {
        ((v - column.min) / (column.max - column.min)).clamp(0.0, 1.0)
    }
}

fn parse_cell(cell: &str) -> Option<f32> {
    let t = cell.trim();
    let lower = t.to_ascii_lowercase();
    match lower.as_str() {
        "" | "na" | "n/a" | "null" | "none" | "nan" | "?" | "#n/a" => return None,
        "true" | "yes" => return Some(1.0),
        "false" | "no" => return Some(0.0),
        _ => {}
    }
    let v = t.parse::<f64>().ok()?;
    (v.is_finite() && v >= f64::from(f32::MIN) && v <= f64::from(f32::MAX)).then_some(v as f32)
}

const JS_CONTRACT_HEADER: &str =
    "\n// Input contract (org.jopedime/math-pipeline/v1). Arithmetic is f32, as in the decoder.\n";

/// The decoder's `encode`/`parse_cell`/`inverse` in JavaScript. `Math.fround` after each f32
/// operation reproduces Rust's f32 results exactly; `RUST_SPACE` is Rust's `char::is_whitespace`
/// set, which differs from `String.prototype.trim` (that also strips U+FEFF).
const JS_CONTRACT: &str = r##"const RUST_SPACE="[\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const TRIM=new RegExp("^"+RUST_SPACE+"+|"+RUST_SPACE+"+$","g");
const MISSING=new Set(["","na","n/a","null","none","nan","?","#n/a"]);
const DECIMAL=/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const F32_MAX=3.4028234663852886e38;
function parseCell(cell){
  let v;
  if(cell===null||cell===undefined)return null;
  if(typeof cell==="number"){v=cell;}
  else{
    const t=String(cell).replace(TRIM,""),lower=t.toLowerCase();
    if(MISSING.has(lower))return null;
    if(lower==="true"||lower==="yes")return 1;
    if(lower==="false"||lower==="no")return 0;
    if(!DECIMAL.test(t))return null;
    v=Number(t);
  }
  return Number.isFinite(v)&&v>=-F32_MAX&&v<=F32_MAX?Math.fround(v):null;
}
function encodeCell(cell,j){
  const v=parseCell(cell),[min,max,constant]=INPUT_SCALERS[j];
  if(v===null||constant)return 0;
  const scaled=Math.fround(Math.fround(v-min)/Math.fround(max-min));
  return scaled<0?0:(scaled>1?1:scaled);
}
/** Predict rows of raw cells in MODEL_COLUMNS order. A cell is a string, a number or null. */
function predictRows(rows){
  if(!Array.isArray(rows))throw new TypeError("rows must be an array");
  const n=rows.length,width=INPUT_SELECTED.length,x=new Float64Array(n*width),out=new Float64Array(n);
  rows.forEach((row,r)=>{
    if(!Array.isArray(row)||row.length!==MODEL_COLUMNS.length)throw new RangeError("row "+r+" needs "+MODEL_COLUMNS.length+" cells");
    INPUT_SELECTED.forEach((j,k)=>{x[r*width+k]=encodeCell(row[j],j);});
  });
  const status=modelPredictNumericBatch(modelFittedHandle(),x,x.length,n,width,null,0,out,n);
  if(status!==0)throw new Error("prediction failed with status "+status);
  return Array.from(out,v=>{
    const p=Number.isFinite(v)?Math.fround(v):0;
    return TARGET_SCALE.scale<=1e-12?TARGET_SCALE.min:Math.fround(TARGET_SCALE.min+Math.fround(p*TARGET_SCALE.scale));
  });
}
if(typeof module!=="undefined"&&module.exports)module.exports={MODEL_COLUMNS,predictRows};
"##;
