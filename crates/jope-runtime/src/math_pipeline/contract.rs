use super::*;

impl InputContract {
    /// New regression objects explicitly record the f64 section version. Only decoding an
    /// absent/legacy marker selects compatibility behavior; construction never defaults to f32.
    pub fn new(
        columns: Vec<NumericColumn>,
        selected: Vec<usize>,
        target: TargetScale,
    ) -> Result<Self> {
        let input = Self {
            task: "regression".into(),
            columns,
            selected,
            target,
            prediction_version: F64_PREDICTION_VERSION,
        };
        input.validate()?;
        Ok(input)
    }

    pub(crate) fn validate(&self) -> Result<()> {
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
            if self.prediction_version == F64_PREDICTION_VERSION
                && !column.constant
                && !(column.max - column.min).is_finite()
            {
                return bad(format!("column {:?} range overflows f64", column.name));
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
    pub(crate) fn inverse_legacy(&self, v: f32) -> f32 {
        let min = self.min as f32;
        let scale = self.scale as f32;
        if scale <= 1e-12 {
            min
        } else {
            min + v * scale
        }
    }

    pub(crate) fn inverse_f64(&self, v: f64) -> f64 {
        if self.scale <= 1e-12 {
            self.min
        } else {
            self.min + v * self.scale
        }
    }
}

pub(crate) fn decode_input(bytes: &[u8]) -> Result<InputContract> {
    // Inspect only the section version, then run the existing strict complete decoder. A
    // streaming header avoids retaining a second full copy of a large column contract.
    #[derive(Deserialize)]
    struct PredictionVersion {
        #[serde(default = "legacy_version")]
        prediction_version: u32,
    }
    let version: PredictionVersion = serde_json::from_slice(bytes)
        .map_err(|e| Failure::integrity(format!("{INPUT_PART}: {e}")))?;
    match version.prediction_version {
        1 => crate::strict_json::decode::<InputContract<f32>>(bytes)
            .map(|input| input.map_numbers(f64::from)),
        F64_PREDICTION_VERSION => crate::strict_json::decode::<InputContract>(bytes),
        _ => {
            return Err(Failure::integrity(format!(
                "{INPUT_PART}: unsupported prediction_version"
            )))
        }
    }
    .map_err(|e| Failure::integrity(format!("{INPUT_PART}: {e}")))
}

pub(super) fn undefined(message: String) -> anyhow::Error {
    Failure::new(
        crate::error::FailureKind::Numerical,
        format!("FITNESS_UNDEFINED: {message}"),
    )
    .into()
}

pub(crate) fn encode_f64(cell: Option<&str>, column: &NumericColumn) -> Result<f64> {
    let Some(v) = cell.and_then(parse_cell_f64) else {
        return Ok(0.0);
    };
    if column.constant {
        return Ok(0.0);
    }
    let encoded = ((v - column.min) / (column.max - column.min)).clamp(0.0, 1.0);
    if encoded.is_finite() {
        Ok(encoded)
    } else {
        Err(undefined(format!(
            "non-finite normalized input {:?}",
            column.name
        )))
    }
}

/// Encode one cell. Missing, non-finite, out-of-f32-range and unparseable cells encode as 0,
/// the scaled column minimum; `true`/`yes` read as 1 and `false`/`no` as 0 before scaling.
pub(crate) fn encode(cell: Option<&str>, column: &NumericColumn) -> f32 {
    let Some(v) = cell.and_then(parse_cell) else {
        return 0.0;
    };
    if column.constant {
        0.0
    } else {
        ((v - column.min as f32) / (column.max as f32 - column.min as f32)).clamp(0.0, 1.0)
    }
}

fn parse_cell(cell: &str) -> Option<f32> {
    let v = parse_cell_f64(cell)?;
    (v >= f64::from(f32::MIN) && v <= f64::from(f32::MAX)).then_some(v as f32)
}

fn parse_cell_f64(cell: &str) -> Option<f64> {
    let t = cell.trim();
    let lower = t.to_ascii_lowercase();
    match lower.as_str() {
        "" | "na" | "n/a" | "null" | "none" | "nan" | "?" | "#n/a" => return None,
        "true" | "yes" => return Some(1.0),
        "false" | "no" => return Some(0.0),
        _ => {}
    }
    let v = t.parse::<f64>().ok()?;
    v.is_finite().then_some(v)
}

pub(crate) const JS_CONTRACT_HEADER: &str =
    "\n// Input contract (org.jopedime/math-pipeline/v1). Arithmetic is f32, as in the decoder.\n";

/// The decoder's `encode`/`parse_cell`/`inverse` in JavaScript. `Math.fround` after each f32
/// operation reproduces Rust's f32 results exactly; `RUST_SPACE` is Rust's `char::is_whitespace`
/// set, which differs from `String.prototype.trim` (that also strips U+FEFF).
pub(crate) const JS_CONTRACT: &str = r##"const RUST_SPACE="[\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
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

// Version 2 deliberately keeps its own contract: legacy f32 arithmetic above is frozen.
pub(crate) const JS_CONTRACT_F64: &str = r##"const RUST_SPACE="[\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const TRIM=new RegExp("^"+RUST_SPACE+"+|"+RUST_SPACE+"+$","g");
const MISSING=new Set(["","na","n/a","null","none","nan","?","#n/a"]);
const DECIMAL=/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
function parseCell(cell){
  if(cell===null||cell===undefined)return null;
  if(typeof cell==="number")return Number.isFinite(cell)?cell:null;
  const t=String(cell).replace(TRIM,""),lower=t.toLowerCase();
  if(MISSING.has(lower))return null;
  if(lower==="true"||lower==="yes")return 1;
  if(lower==="false"||lower==="no")return 0;
  if(!DECIMAL.test(t))return null;
  const v=Number(t);return Number.isFinite(v)?v:null;
}
function encodeCell(cell,j){
  const v=parseCell(cell),[min,max,constant]=INPUT_SCALERS[j];
  if(v===null||constant)return 0;
  const scaled=(v-min)/(max-min),result=scaled<0?0:(scaled>1?1:scaled);
  if(!Number.isFinite(result))throw new Error("prediction undefined: non-finite normalized input");
  return result;
}
function predictRows(rows){
  if(!Array.isArray(rows))throw new TypeError("rows must be an array");
  const n=rows.length,width=INPUT_SELECTED.length,x=new Float64Array(n*width),out=new Float64Array(n);
  rows.forEach((row,r)=>{
    if(!Array.isArray(row)||row.length!==MODEL_COLUMNS.length)throw new RangeError("row "+r+" needs "+MODEL_COLUMNS.length+" cells");
    INPUT_SELECTED.forEach((j,k)=>{x[r*width+k]=encodeCell(row[j],j);});
  });
  if(n===0)return [];
  const status=modelPredictNumericBatch(modelFittedHandle(),x,x.length,n,width,null,0,out,n);
  if(status!==0)throw new Error("prediction undefined: prediction failed with status "+status);
  return Array.from(out,(v,r)=>{
    if(!Number.isFinite(v))throw new Error("prediction undefined: row "+r+": non-finite fitted prediction");
    const result=TARGET_SCALE.scale<=1e-12?TARGET_SCALE.min:TARGET_SCALE.min+v*TARGET_SCALE.scale;
    if(!Number.isFinite(result))throw new Error("prediction undefined: row "+r+": non-finite target prediction");
    return result;
  });
}
if(typeof module!=="undefined"&&module.exports)module.exports={MODEL_COLUMNS,predictRows};
"##;
