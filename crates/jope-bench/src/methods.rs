//! What gets benchmarked: tier-1 baselines (every head as a single-head program), programs the
//! caller names, and the search run on a cut's training rows only.

use anyhow::{bail, ensure, Context, Result};
use feat_math::evolve::{evolve_lite_race, EvolveConfig, FitnessMode, GenStats, Observer, Scored};
use feat_math::heads::is_head_op;
use feat_math::transpile::Emitted;
use feat_math::{GenomeIr, Rows};
use serde::{Deserialize, Serialize};

/// Where a method came from.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MethodKind {
    /// A single-head program `(pipe X (<head>))`.
    Baseline,
    /// A program the caller supplied.
    Program,
    /// A program the search produced on the screen cut's training rows.
    Search,
}

/// One method: an id and the program it runs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Method {
    /// Ladder id (`baseline.ridge`, `program.<name>`, `afm.lite-race.v1`).
    pub id: String,
    /// Origin.
    pub kind: MethodKind,
    /// Canonical program text.
    pub dsl: String,
}

/// Every distinct head of the registry (aliases `logistic`, `gbm` and `ensemble_ridge_gbm` are
/// spellings of `ridge`, `histgbm` and `ensemble`; `ridge_gcv` is accepted by the interpreter
/// though absent from the registry).
pub const DEFAULT_HEADS: [&str; 10] = [
    "ridge",
    "ridge_gcv",
    "elasticnet",
    "histgbm",
    "histgbm2",
    "ksmooth",
    "ensemble",
    "krr",
    "gpr",
    "stack",
];

fn canonical(dsl: &str) -> Result<String> {
    let genome = GenomeIr::parse(dsl).with_context(|| format!("program {dsl:?}"))?;
    Ok(genome.canonical())
}

/// `(pipe X (<head>))` for every head, as `baseline.<head>`. The parser accepts any stage
/// name and defers to `fit`, so head names are checked against the interpreter's head set here.
pub fn baselines(heads: &[String]) -> Result<Vec<Method>> {
    heads
        .iter()
        .map(|head| {
            ensure!(
                is_head_op(head),
                "{head:?} is not a head the interpreter knows"
            );
            let dsl = canonical(&format!("(pipe X ({head}))"))?;
            Ok(Method {
                id: format!("baseline.{head}"),
                kind: MethodKind::Baseline,
                dsl,
            })
        })
        .collect()
}

/// `NAME=<dsl>` or `NAME=@<file>`.
pub fn program(spec: &str) -> Result<Method> {
    let Some((name, source)) = spec.split_once('=') else {
        bail!("--program wants NAME=<dsl> or NAME=@<file>, got {spec:?}");
    };
    let name = name.trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("program name {name:?}: use letters, digits, '-' and '_'");
    }
    let text = match source.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("read {path}"))?,
        None => source.to_owned(),
    };
    Ok(Method {
        id: format!("program.{name}"),
        kind: MethodKind::Program,
        dsl: canonical(text.trim())?,
    })
}

/// Search budget.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SearchConfig {
    /// Population size.
    pub pop: usize,
    /// Generations.
    pub gens: usize,
    /// Search seed.
    pub seed: u64,
    /// Wall-clock cap in seconds (0 = none).
    pub seconds: f64,
}

/// Method id of the searched program.
pub const SEARCH_METHOD_ID: &str = "afm.lite-race.v1";

struct Quiet;

impl Observer for Quiet {
    fn on_generation(&self, _: &GenStats) {}
    fn on_improvement(&self, _: &Scored, _: &Emitted, _: &GenStats) {}
}

/// Run the lite race on `x`/`y` (a cut's training rows, never more) and return the best
/// program as a [`MethodKind::Search`] method. The race's own inner folds live inside these rows.
pub fn search(x: &Rows, y: &[f64], config: &SearchConfig) -> Result<Method> {
    let cfg = EvolveConfig {
        pop: config.pop,
        gens: config.gens,
        seed: config.seed,
        time_budget_secs: config.seconds,
        fitness: FitnessMode::LiteRace,
        rank_robust: true,
        ..EvolveConfig::default()
    };
    let outcome = evolve_lite_race(x, y, &feat_math::r2, &cfg, &Quiet);
    let best = outcome
        .ranked
        .first()
        .context("search produced no program")?;
    Ok(Method {
        id: SEARCH_METHOD_ID.to_owned(),
        kind: MethodKind::Search,
        dsl: canonical(&best.dsl)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baselines_are_valid_single_head_programs() {
        let heads: Vec<String> = DEFAULT_HEADS.iter().map(|h| (*h).to_owned()).collect();
        let methods = baselines(&heads).unwrap();
        assert_eq!(methods.len(), DEFAULT_HEADS.len());
        assert_eq!(methods[0].id, "baseline.ridge");
        assert!(methods.iter().all(|m| m.kind == MethodKind::Baseline));
        assert!(baselines(&["no_such_head".to_owned()]).is_err());
        let p = program("champ=(pipe X (robust_scale) (ridge))").unwrap();
        assert_eq!(p.id, "program.champ");
        assert!(program("bad name=(pipe X (ridge))").is_err());
        assert!(program("x").is_err());
    }
}
