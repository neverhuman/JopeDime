//! What gets benchmarked: tier-1 baselines (every head of the task's grammar as a single-head
//! program), programs the caller names, and the search run on a cut's training rows only.

use anyhow::{bail, Context, Result};
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::evolve::{evolve_lite_race, EvolveConfig, FitnessMode, GenStats, Observer, Scored};
use feat_math::heads::is_head_op;
use feat_math::transpile::Emitted;
use feat_math::{GenomeIr, Rows};
use serde::{Deserialize, Serialize};

use crate::eval::numeric_schema;
use crate::task::Task;

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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Method {
    /// Ladder id (`baseline.ridge`, `program.<name>`, `afm.lite-race.v1`,
    /// `afm.lite-race.v1.resampled`).
    pub id: String,
    /// Origin.
    pub kind: MethodKind,
    /// Canonical program text (for a search method: the program found on the screen cut).
    pub dsl: String,
    /// Wall seconds it took to produce the program (0 for baselines and caller programs; the
    /// search's wall time for a search method). The cost-per-program number starts here.
    #[serde(default)]
    pub produce_seconds: f64,
    /// When set, the deep stage re-runs the search on every cut's training rows and scores
    /// that cut's own program, so the deep number measures the *procedure*, not the one program
    /// found on the screen cut (whose selection saw rows the other cuts hold out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resample: Option<SearchConfig>,
}

fn canonical(dsl: &str) -> Result<String> {
    let genome = GenomeIr::parse(dsl).with_context(|| format!("program {dsl:?}"))?;
    Ok(genome.canonical())
}

/// Whether `head` is a head the task's grammar accepts as the final stage of a single-head
/// program. Regression: the frozen interpreter's head set (the parser accepts any stage name
/// and defers to `fit`). Binary and multiclass: the schema-bound v3 grammar, probed on a
/// one-column numeric schema (head compatibility does not depend on the column count).
pub fn check_head(head: &str, task: Task) -> Result<()> {
    match task.v3_spec() {
        None => {
            if !is_head_op(head) {
                bail!("{head:?} is not a head the interpreter knows");
            }
            Ok(())
        }
        Some(spec) => {
            let genome = GenomeIr::parse(&format!("(pipe X ({head}))"))
                .with_context(|| format!("head {head:?}"))?;
            validate_v3_genome(&genome, &spec, &numeric_schema(1))
                .with_context(|| format!("{head:?} is not a {} head", task.as_str()))
        }
    }
}

/// `(pipe X (<head>))` for every head, as `baseline.<head>`, each checked against the task's
/// grammar ([`check_head`]).
pub fn baselines(heads: &[String], task: Task) -> Result<Vec<Method>> {
    heads
        .iter()
        .map(|head| {
            check_head(head, task)?;
            let dsl = canonical(&format!("(pipe X ({head}))"))?;
            Ok(Method {
                id: format!("baseline.{head}"),
                kind: MethodKind::Baseline,
                dsl,
                produce_seconds: 0.0,
                resample: None,
            })
        })
        .collect()
}

/// The task's default baselines ([`Task::default_heads`]).
pub fn default_baselines(task: Task) -> Result<Vec<Method>> {
    let heads: Vec<String> = task
        .default_heads()
        .iter()
        .map(|h| (*h).to_owned())
        .collect();
    baselines(&heads, task)
}

/// `NAME=<dsl>` or `NAME=@<file>`. The program is parsed here; whether its head suits the
/// dataset's task is the evaluator's call (a mismatch is a rejection with the grammar's reason).
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
        produce_seconds: 0.0,
        resample: None,
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

/// Method id of the searched program (one program, found on the screen cut, scored everywhere).
pub const SEARCH_METHOD_ID: &str = "afm.lite-race.v1";
/// Method id of the searched program when the deep stage re-runs the search per cut. A
/// different id, so a ladder built from mixed reports never merges the two numbers.
pub const RESAMPLED_SEARCH_METHOD_ID: &str = "afm.lite-race.v1.resampled";

struct Quiet;

impl Observer for Quiet {
    fn on_generation(&self, _: &GenStats) {}
    fn on_improvement(&self, _: &Scored, _: &Emitted, _: &GenStats) {}
}

/// Run the lite race on `x`/`y` (a cut's training rows, never more) and return the best
/// program's canonical text. The race's own inner folds live inside these rows. Regression only:
/// the race fits and scores on the frozen kernel.
pub fn search_program(x: &Rows, y: &[f64], config: &SearchConfig) -> Result<String> {
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
    canonical(&best.dsl)
}

/// [`search_program`] as a [`MethodKind::Search`] method, with the wall time recorded.
pub fn search(x: &Rows, y: &[f64], config: &SearchConfig) -> Result<Method> {
    let started = std::time::Instant::now();
    let dsl = search_program(x, y, config)?;
    Ok(Method {
        id: SEARCH_METHOD_ID.to_owned(),
        kind: MethodKind::Search,
        dsl,
        produce_seconds: started.elapsed().as_secs_f64(),
        resample: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baselines_are_valid_single_head_programs_for_their_task() {
        let methods = default_baselines(Task::Regression).unwrap();
        assert_eq!(methods.len(), Task::Regression.default_heads().len());
        assert_eq!(methods[0].id, "baseline.ridge");
        assert!(methods.iter().all(|m| m.kind == MethodKind::Baseline));
        assert!(baselines(&["no_such_head".to_owned()], Task::Regression).is_err());
        let p = program("champ=(pipe X (robust_scale) (ridge))").unwrap();
        assert_eq!(p.id, "program.champ");
        assert_eq!(p.produce_seconds, 0.0);
        assert!(program("bad name=(pipe X (ridge))").is_err());
        assert!(program("x").is_err());

        for task in [Task::Binary, Task::Multiclass] {
            let methods = default_baselines(task).unwrap();
            assert_eq!(methods.len(), task.default_heads().len());
            assert!(methods.iter().all(|m| m.dsl.starts_with("(pipe X (")));
        }
        assert_eq!(Task::Binary.default_heads().len(), 6);
        assert!(Task::Binary.default_heads().contains(&"portfolio"));
        // A regression head is not a binary head and vice versa; the grammar says so.
        let error = baselines(&["ridge".to_owned()], Task::Binary).unwrap_err();
        assert!(error.to_string().contains("not a binary head"), "{error:#}");
        assert!(baselines(&["logit_ridge".to_owned()], Task::Regression).is_err());
        assert!(baselines(&["logit_ridge".to_owned()], Task::Multiclass).is_err());
        assert!(baselines(&["multiclass_ridge".to_owned()], Task::Binary).is_err());
    }
}
