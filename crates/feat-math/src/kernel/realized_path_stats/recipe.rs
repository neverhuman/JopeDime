//! Complete ordered training recipes and caller-owned fitted continuation.
use super::model_validation::validate_fit_closure;
use super::*;
use crate::kernel::wire_v2::VersionResult;
use crate::transpile::{lint_shipped_text, EmitOptions};
use sha2::{Digest, Sha256};

pub const RECIPE_PROFILE: &str = "feat-math.causal-path-stats-recipe.v2";
pub const FITTED_PROFILE: &str = "feat-math.causal-path-stats-fitted-cpp.v2";
const FITTED_SCHEMA: &str = "feat-math.causal-path-stats-fitted-codec.v2";

#[derive(Clone, Debug)]
pub struct PathStatsRecipe {
    pub profile: &'static str,
    pub program_identity: String,
    pub language: PathStatsLanguage,
    pub source: String,
    pub source_sha256: String,
}

#[derive(Clone, Debug)]
pub struct PathStatsFittedCpp {
    pub profile: &'static str,
    pub source: String,
    pub source_sha256: String,
    pub codec: String,
}

impl PathStatsFittedCpp {
    /// Recreate the source from its validated native checkpoint and bind every
    /// codec field to the current feature renderer; no legacy reader fallback.
    pub fn from_codec_json(json: &str) -> VersionResult<Self> {
        if json.len() > crate::kernel::codec::MAX_BYTES {
            return Err(Error::Resource("ordered fitted codec bytes".into()).into());
        }
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| Error::Syntax(e.to_string()))?;
        if value["schema"].as_str() != Some(FITTED_SCHEMA)
            || value["format_version"].as_u64() != Some(2)
            || value["min_reader_version"].as_u64() != Some(2)
            || value["required_features"].as_u64() != Some(super::program::REQUIRED_FEATURE)
            || value["profile"].as_str() != Some(FITTED_PROFILE)
        {
            return Err(Error::Unsupported("ordered fitted codec version".into()).into());
        }
        let model_json = value["model_json"]
            .as_str()
            .ok_or_else(|| Error::Syntax("ordered fitted checkpoint".into()))?;
        let artifact = PathStatsModel::from_json(model_json)?.emit_fitted_cpp17()?;
        let expected: serde_json::Value =
            serde_json::from_str(&artifact.codec).map_err(|e| Error::Syntax(e.to_string()))?;
        if value != expected {
            return Err(Error::Domain("ordered fitted codec binding".into()).into());
        }
        Ok(artifact)
    }
}

impl PathStatsProgram {
    /// The generation profile admits only whole shapes covered by the recipes.
    /// Decoding remains wider so unsupported stored programs can still round-trip.
    pub fn validate_recipe_closure(&self) -> VersionResult<()> {
        self.model_size_base().map(|_| ())
    }

    pub fn emit_recipe(&self, language: PathStatsLanguage) -> VersionResult<PathStatsRecipe> {
        let model_base_bytes = self.model_size_base()?;
        let opts = EmitOptions::default();
        let input = serde_json::to_string(self.input_identity())
            .map_err(|e| Error::Syntax(e.to_string()))?;
        let (downstream, state, model, literal) = match language {
            PathStatsLanguage::Rust => {
                let emitted = crate::transpile::emit(self.downstream(), &opts)
                    .map_err(|e| Error::Unsupported(e.to_string()))?;
                (
                    format!("mod downstream {{\n{}\n{RUST_FIT}\n}}\n", emitted.rust),
                    super::emit::emit_state(PathStatsLanguage::Rust),
                    include_str!("recipes/path_stats_model_rust.txt"),
                    format!(
                        "std::str::from_utf8(&[{}]).map_err(|_|\"syntax\")?",
                        byte_list(&input)
                    ),
                )
            }
            PathStatsLanguage::Python => {
                let emitted = crate::transpile::emit(self.downstream(), &opts)
                    .map_err(|e| Error::Unsupported(e.to_string()))?;
                // The ordered API owns entry; do not run the child CSV CLI.
                let source = emitted
                    .python
                    .replace("if __name__ == \"__main__\":", "if False:");
                (
                    source,
                    super::emit::emit_state(PathStatsLanguage::Python),
                    include_str!("recipes/path_stats_model_python.txt"),
                    format!("json.loads(bytes([{}]).decode('utf-8'))", byte_list(&input)),
                )
            }
            PathStatsLanguage::Cpp17 => (
                crate::kernel::expression_recipe::emit_cpp17(self.downstream(), &opts)?.source,
                super::emit::emit_state(PathStatsLanguage::Cpp17),
                include_str!("recipes/path_stats_model_cpp.txt"),
                cpp_string(&input),
            ),
        };
        let model = model
            .replace("@MODEL_BASE_BYTES@", &model_base_bytes.to_string())
            .replace("@WINDOW@", &self.config.window().to_string())
            .replace("@INPUT@", &literal);
        let source = format!("{downstream}\n{state}\n{model}");
        check_artifact(&source)?;
        Ok(PathStatsRecipe {
            profile: RECIPE_PROFILE,
            program_identity: self.identity()?,
            language,
            source_sha256: hash(&source),
            source,
        })
    }

    // All variable-size fields are counted by the generated train API. Use
    // the native envelope for the fixed portion, including profile/bindings.
    fn model_size_base(&self) -> VersionResult<usize> {
        validate_fit_closure(self.downstream())?;
        let model = self.fit_training(
            &[OrderedRow {
                group: GroupKey::NumericBits(0),
                timestamp: 0.0,
                values: vec![0.0; self.input_width()],
            }],
            &[0.0],
            TaskMode::Ordered,
            FoldBoundary::new("c", 1, 1)?,
        )?;
        let json = model.to_json()?;
        // Fixed metadata must also fit the fitted codec before a whole shape
        // gains export support. A source digest always has 64 ASCII bytes.
        fitted_codec(&json, &self.identity()?, &"0".repeat(64))?;
        let value: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| Error::Syntax(e.to_string()))?;
        let head = &value["predictor"]["predictor"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()["Pipeline"]["head"]["Ridge"];
        Ok(json.len()
            - value["fold"].to_string().len()
            - value["state"].to_string().len()
            - ["means", "scales", "w"]
                .iter()
                .map(|key| head[key].to_string().len())
                .sum::<usize>())
    }
}

impl PathStatsModel {
    pub fn emit_fitted_cpp17(&self) -> VersionResult<PathStatsFittedCpp> {
        let model_json = self.to_json()?;
        let downstream = crate::advanced_v3::emit_fitted_cpp17(&self.predictor)
            .map_err(|e| Error::Unsupported(e.to_string()))?
            .source
            .replace("jope_fitted_predict", "jope_path_stats_downstream_predict")
            .replace("jope_fitted_columns", "jope_path_stats_downstream_columns");
        let task = match self.task {
            TaskMode::Ordered => 1,
            TaskMode::Forecasting => 2,
            TaskMode::IID => {
                return Err(Error::Unsupported("ordered fitted task required".into()).into())
            }
        };
        let input = serde_json::to_string(self.program.input_identity())
            .map_err(|e| Error::Syntax(e.to_string()))?;
        let bytes = self
            .state
            .to_bytes()?
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let factory = format!(
            "static PathStatsFittedModel make_path_stats_model() {{return {{PathState::from_bytes(OrderedBytes{{{bytes}}},{task},{},{},{},{}),{task}}};}}\n",
            cpp_string(&input), cpp_string(&self.fold.cut_id), self.fold.train_end, self.fold.total_rows
        );
        let source = format!(
            "{downstream}\n{}\n{}\n{factory}",
            super::emit::emit_state(PathStatsLanguage::Cpp17),
            include_str!("recipes/path_stats_fitted_cpp.txt")
                .replace(
                    "@CHECKPOINT_FIXED_BYTES@",
                    &(model_json.len()
                        - serde_json::to_value(&self.state.to_bytes()?)
                            .map_err(|e| Error::Syntax(e.to_string()))?
                            .to_string()
                            .len())
                    .to_string()
                )
                .replace("jope_fitted_predict", "jope_path_stats_downstream_predict")
        );
        check_artifact(&source)?;
        let source_sha256 = hash(&source);
        let codec = fitted_codec(&model_json, &self.program.identity()?, &source_sha256)?;
        Ok(PathStatsFittedCpp {
            profile: FITTED_PROFILE,
            source,
            source_sha256,
            codec,
        })
    }
}

fn fitted_codec(
    model_json: &str,
    program_identity: &str,
    source_sha256: &str,
) -> VersionResult<String> {
    let codec = serde_json::json!({
        "schema":FITTED_SCHEMA,"format_version":2,"min_reader_version":2,
        "required_features":super::program::REQUIRED_FEATURE,
        "profile":FITTED_PROFILE,"source_sha256":source_sha256,
        "program_identity":program_identity,"model_json":model_json
    })
    .to_string();
    if codec.len() > crate::kernel::codec::MAX_BYTES {
        return Err(Error::Resource("ordered fitted codec bytes".into()).into());
    }
    Ok(codec)
}

fn check_artifact(source: &str) -> Result<()> {
    let hits = lint_shipped_text(source);
    if hits.is_empty() {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "artifact vocabulary: {}",
            hits.join(", ")
        )))
    }
}

fn hash(source: &str) -> String {
    Sha256::digest(source.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Byte constructors preserve UTF-8 and cannot close a source-code delimiter.
fn byte_list(value: &str) -> String {
    value
        .bytes()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn cpp_string(value: &str) -> String {
    format!(
        "std::string({{{}}})",
        value
            .bytes()
            .map(|b| format!("char({b})"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

const RUST_FIT: &str = r#"
pub struct Predictor(Model);
pub fn fit(x:&[Vec<f64>],y:&[f64])->Result<Predictor,&'static str> {
    let mut model=Model::fit(x,y);
    if y.iter().all(|v|v.to_bits()==(-0.0f64).to_bits()) {model.head.w[0]=-0.0;}
    let h=&model.head;let width=x[0].len();
    if h.means.len()!=width || h.scales.len()!=width || h.w.len()!=width+1
        || h.means.iter().chain(h.w.iter()).any(|v|!v.is_finite())
        || h.scales.iter().any(|v|!v.is_finite() || *v<=0.0) {return Err("domain");}
    Ok(Predictor(model))
}
impl Predictor {
    pub fn predict(&self,x:&[Vec<f64>])->Vec<f64> {self.0.predict(x)}
    pub fn parameters(&self)->(&[f64],&[f64],&[f64]) {(&self.0.head.means,&self.0.head.scales,&self.0.head.w)}
}
"#;
