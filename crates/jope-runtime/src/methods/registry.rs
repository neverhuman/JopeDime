//! The native method roster, moved unchanged from the frozen-panel adapter
//! (`crates/jope-bench/src/panel/methods.rs`, which re-exports it). Stable ids, configurations,
//! aliases and canonical recipes are the panel's; the appliance adds no method of its own.
use crate::task::{self, TaskKind};
use anyhow::{ensure, Result};
use feat_math::advanced_v3::{Objective, TaskSpec};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INPUT_POLICY: &str = "median-onehot256-indicator/v1";
/// No target-aware encodings: groups cannot leak through ordered target statistics.
pub const INPUT_STAGE: &str =
    "(input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256)";

/// A recipe is distinct from an algorithm family. Aliases never add screening slots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustMethod {
    pub id: String,
    pub family: String,
    pub configuration: String,
    pub aliases: Vec<String>,
    pub dsl: String,
}

fn method(
    head: &str,
    configuration: &str,
    parameters: &str,
    aliases: &[&str],
) -> Result<RustMethod> {
    let suffix = if configuration == "default" || configuration == "core_v1" {
        String::new()
    } else {
        format!(".{configuration}")
    };
    let stage = if parameters.is_empty() {
        head.to_owned()
    } else {
        format!("{head} {parameters}")
    };
    let genome = GenomeIr::parse(&format!("(pipe X {INPUT_STAGE} ({stage}))"))?;
    Ok(RustMethod {
        id: format!("rust.{head}{suffix}"),
        family: head.to_owned(),
        configuration: configuration.to_owned(),
        aliases: aliases.iter().map(|name| (*name).to_owned()).collect(),
        dsl: genome.canonical(),
    })
}

/// Every executable product head compatible with the task, through the raw-table boundary.
/// Regression has 16 families and 17 recipes: non-RBF GPR has a distinct fitted implementation.
/// Binary has five families; multiclass has four. Engine defaults stay defaults (in particular,
/// multiclass's 64 transformed-feature budget); arbitrary hyperparameter sweeps are not implied.
pub fn rust_methods(kind: TaskKind) -> Result<Vec<RustMethod>> {
    let mut out = Vec::new();
    match kind {
        TaskKind::Regression => {
            for (head, aliases) in [
                ("ridge", &["logistic"][..]),
                ("ridge_gcv", &[][..]),
                ("elasticnet", &[][..]),
                ("histgbm", &["gbm"][..]),
                ("histgbm2", &[][..]),
                ("ksmooth", &[][..]),
                ("ensemble", &["ensemble_ridge_gbm"][..]),
                ("krr", &[][..]),
                ("gpr", &[][..]),
                ("hurdle2", &[][..]),
                ("stack", &[][..]),
                ("tab_boost", &[][..]),
                ("extra_trees", &[][..]),
                ("gam_boost", &[][..]),
                ("pls", &[][..]),
            ] {
                out.push(method(head, "default", "", aliases)?);
            }
            out.push(method("portfolio", "core_v1", "preset=core", &[])?);
            out.push(method("gpr", "rbf_linear", "kernel=rbf+linear", &[])?);
        }
        TaskKind::Binary => {
            out.push(method("logit_ridge", "default", "", &["logistic"])?);
            for head in ["tab_boost", "extra_trees", "gam_boost", "portfolio"] {
                out.push(method(head, "default", "", &[])?);
            }
        }
        TaskKind::Multiclass => {
            for head in [
                "multiclass_ridge",
                "multiclass_ovr_logistic",
                "multiclass_diagonal_lda",
                "multiclass_nearest_centroid",
            ] {
                out.push(method(head, "default", "", &[])?);
            }
        }
        other => anyhow::bail!("{} is outside the frozen panel", task::name(other)),
    }
    Ok(out)
}

/// The fitted engine task is separate from the controller's primary metric. The engine's
/// multiclass task contract admits MacroF1; the panel scores its probability matrix with log loss.
pub fn task_spec(kind: TaskKind) -> Result<TaskSpec> {
    match kind {
        TaskKind::Regression => TaskSpec::regression(Objective::Rmse),
        TaskKind::Binary => TaskSpec::binary(Objective::RocAuc),
        TaskKind::Multiclass => TaskSpec::multiclass(Objective::MacroF1),
        other => anyhow::bail!("{} is outside the frozen panel", task::name(other)),
    }
}

pub fn validate_classes(kind: TaskKind, classes: &[String]) -> Result<()> {
    ensure!(
        classes.iter().collect::<BTreeSet<_>>().len() == classes.len()
            && classes.iter().all(|class| !class.is_empty()),
        "empty or duplicate class labels"
    );
    ensure!(
        match kind {
            TaskKind::Regression => classes.is_empty(),
            TaskKind::Binary => classes.len() == 2,
            TaskKind::Multiclass => classes.len() >= 3,
            TaskKind::Timeseries => false,
        },
        "class vocabulary is incompatible with task"
    );
    Ok(())
}

/// Resolve an id (`rust.gpr.rbf_linear`), the id without its `rust.` prefix (`gpr.rbf_linear`),
/// a bare family (`ridge`, the default configuration) or an alias (`gbm`) for one task.
pub fn find(kind: TaskKind, name: &str) -> Result<Option<RustMethod>> {
    let roster = rust_methods(kind)?;
    let exact = roster
        .iter()
        .find(|m| m.id == name || m.id.strip_prefix("rust.") == Some(name));
    let family = roster.iter().find(|m| {
        m.family == name && (m.configuration == "default" || m.configuration == "core_v1")
    });
    let alias = roster.iter().find(|m| m.aliases.iter().any(|a| a == name));
    Ok(exact.or(family).or(alias).cloned())
}
