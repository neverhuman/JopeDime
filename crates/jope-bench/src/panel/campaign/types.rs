use super::*;

pub const PLAN_SCHEMA: &str = "jope.panel-campaign/v2";

/// Every byte affecting a fit is either an executable/input artifact or a pinned
/// runtime inventory. Runtime inventories are verified before admission and after
/// execution; per-job inputs pin their immutable receipt as well as the adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub program: Artifact,
    pub argument_prefix: Vec<String>,
    pub inputs: Vec<Artifact>,
    /// Complete installed package inventory, authenticated once before/after a run.
    pub package_files: Vec<Artifact>,
    pub environment: BTreeMap<String, String>,
    pub version: String,
    pub unavailable: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Method {
    Synthesized {
        id: String,
        recipe: jope_pipeline::synthesized_recipe::SynthesizedRecipe,
        checkpoint_sha256: Option<String>,
    },
    Rust {
        recipe: RustMethod,
    },
    External {
        id: String,
        engine: String,
        runtime: String,
    },
}
impl Method {
    pub fn id(&self) -> &str {
        match self {
            Self::Rust { recipe } => &recipe.id,
            Self::External { id, .. } | Self::Synthesized { id, .. } => id,
        }
    }
    pub(crate) fn runtime(&self) -> &str {
        match self {
            Self::Rust { .. } | Self::Synthesized { .. } => "rust",
            Self::External { runtime, .. } => runtime,
        }
    }
    pub(crate) fn identity(&self) -> Result<String> {
        match self {
            Self::Rust { recipe } => Ok(jope_pipeline::sha256_hex(recipe.dsl.as_bytes())),
            Self::External { .. } => digest(self),
            Self::Synthesized { recipe, .. } => {
                recipe.validate()?;
                Ok(recipe.executable_sha256.clone())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignPlan {
    pub schema: String,
    pub experiment: Experiment,
    pub panel: Panel,
    pub methods: BTreeMap<String, Vec<Method>>,
    pub runtimes: BTreeMap<String, Runtime>,
    /// A complete panel requires the neural comparison to be attached before
    /// execution. This controller's comparator qualification is named explicitly.
    pub scope: String,
}
impl CampaignPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == PLAN_SCHEMA, "unknown campaign plan");
        self.experiment.validate()?;
        self.panel.validate()?;
        ensure!(
            self.experiment.panel_sha256 == super::super::FROZEN_PANEL_SHA256,
            "panel identity differs from approved manifest"
        );
        // The original approved bytes are checked by read_panel at construction;
        // retain the semantic panel identity as part of the immutable plan.
        ensure!(
            self.experiment.methods_sha256 == digest(&(&self.methods, &self.runtimes))?,
            "method/runtime roster changed"
        );
        ensure!(
            (self.experiment.protocol == super::super::PROTOCOL
                || (self.scope == "asm-origin-proof"
                    && self.experiment.protocol == "jope.asm-origin-proof/v1"))
                && self.experiment.numerical_policy == score::NUMERICAL_POLICY,
            "campaign protocol or numerical policy changed"
        );
        ensure!(
            self.scope == "comparator-qualification"
                || ((self.scope == "asm-panel" || self.scope == "asm-origin-proof")
                    && cfg!(feature = "asm")
                    && self.runtimes.contains_key("asm-configuration")),
            "unsupported campaign scope"
        );
        let native_runtime = self.runtimes.get("rust").context("native runtime absent")?;
        ensure!(
            native_runtime.argument_prefix == ["benchmark"]
                || native_runtime.argument_prefix == ["panel"],
            "native runtime must pin its benchmark or panel command"
        );
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            let methods = self
                .methods
                .get(task.as_str())
                .context("missing task roster")?;
            let mut ids = BTreeSet::new();
            let native = super::super::methods::rust_methods(task)?;
            for method in methods {
                ensure!(
                    ids.insert(method.id()) && !method.id().is_empty(),
                    "duplicate method"
                );
                let runtime = self
                    .runtimes
                    .get(method.runtime())
                    .context("runtime absent")?;
                ensure!(!runtime.version.is_empty(), "runtime version/reason absent");
                match method {
                    Method::Synthesized { .. } => anyhow::bail!(
                        "generated roster entries must come from retained train-side selection"
                    ),
                    Method::Rust { recipe } => {
                        ensure!(native.contains(recipe), "changed native roster")
                    }
                    Method::External { engine, .. } => ensure!(
                        ["xgboost", "lightgbm", "catboost", "sklearn"].contains(&engine.as_str()),
                        "unknown external engine"
                    ),
                }
            }
            ensure!(
                native.iter().all(|r| methods.iter().any(|m| matches!(m,
                Method::Rust {recipe} if recipe == r))),
                "incomplete native roster"
            );
            for engine in ["xgboost", "lightgbm", "catboost", "sklearn"] {
                ensure!(
                    methods
                        .iter()
                        .filter(|m| matches!(m,
                    Method::External {engine: e, ..} if e == engine))
                        .count()
                        == 1,
                    "each external engine requires one explicit outcome slot"
                );
            }
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}

pub struct PendingFit {
    pub method: Method,
    pub request: EvaluationRequest,
    pub job: WorkerJob,
    pub input_sha256: String,
    pub expected_engine_version: String,
}
