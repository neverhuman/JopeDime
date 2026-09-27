//! Stage 4: the genetic-programming search, one run per cut, on feat-math.
//!
//! # Why this is not the engine's search
//!
//! An earlier draft of this stage drove `jope_parallel::gp::search_seeded_with_rng_in_root`.
//! That path cannot measure anything in a shipped binary, for two independent
//! reasons on this SHA:
//!
//! * `crates/jope-engine/src/fit_boundary.rs` refuses every fit unless
//!   `$JOPE_FIT_SERVICE_SOCKET` names a reachable fit authority service
//!   ("mandatory fit authority service unavailable: fitting is fenced"); and
//! * behind it, `eval_registry/authority.rs::activation_block` reads the
//!   activation and coverage tables into `active`, then does `let _ = active;`
//!   and returns `Some("namespace capability inactive: explicit verified
//!   activation required")` *unconditionally*. The only bypass is a
//!   `#[cfg(test)]` block whose own comment reads "Production capabilities have
//!   no verifier yet."
//!
//! So every candidate's fit is refused, `successful_measurements` is 0 on every
//! cut, and the run emits its bootstrap genome. That is a property of the engine
//! path, not of the environment: no amount of configuration reaches past
//! `let _ = active;`.
//!
//! # What this does instead
//!
//! `feat_math` is unfenced and real. This is a plain generational GP over
//! canonical pipeline genomes:
//!
//! * generation 0 is the bootstrap genome plus [`POPULATION`]`-1` grammar-random
//!   pipelines from [`evolve::random_canonical_pipeline`];
//! * a candidate is scored by `feat_math::fit` on the cut's **training** rows and
//!   `feat_math::r2`/RMSE on the cut's **held-out** rows — the fit never sees a
//!   row it is scored on (see [`crate::prepare`]);
//! * the next generation keeps the [`ELITES`] best by held-out RMSE *with their
//!   existing scores* — an elite is never re-fitted, so it never spends budget
//!   twice — and fills the rest with [`evolve::mutate_canonical_pipeline`] and
//!   [`evolve::crossover_canonical_pipelines`] children;
//! * every seed is derived from `--rng-seed`, the cut index, the generation and
//!   the slot, so the whole search replays from the command line alone.
//!
//! The bootstrap genome is in generation 0 on purpose: the search can report a
//! champion worse than it only if the bootstrap itself failed to fit.
//!
//! # Failure is counted, not fatal — except when it is total
//!
//! A candidate that fails to parse, fit, score or that panics inside feat-math is
//! recorded with its reason and the search continues. A cut on which *nothing*
//! measured fails the run loudly, naming the distinct reasons: silently emitting
//! a bootstrap genome is what the engine path did, and is the behaviour this work
//! package exists to remove.

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    evolve,
    genome::GenomeIr,
    metrics::{r2, rmse},
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{prepare::Cut, receipt::Phase, sha256_hex};

/// The search implementation, named in the receipt.
pub const SEARCH_ENGINE: &str = "feat-math canonical-pipeline GP (in-process)";

/// The bounds a budget may ask for. Kept identical to the engine's own guard so
/// a budget written for one is legal for the other.
pub const MAX_EVALUATIONS: usize = 1000;
pub const MAX_SECONDS: u64 = 600;

/// Generation 0 always contains this, so a measured champion is never worse than
/// plain ridge on the raw features.
pub const BOOTSTRAP_GENOME: &str = "(pipe X (ridge))";

/// Candidates alive in one generation.
pub const POPULATION: usize = 12;
/// Scored survivors carried into the next generation without being re-fitted.
pub const ELITES: usize = 4;
/// Transform stages a generated pipeline may carry.
pub const MAX_TRANSFORMS: usize = 4;
/// Attempts to draw an offspring this cut has not already measured.
const DISTINCT_CHILD_ATTEMPTS: usize = 8;

/// Which bound ended the search.
pub const STOPPED_BY_EVALUATIONS: &str = "evaluations";
pub const STOPPED_BY_SECONDS: &str = "seconds";
/// Neither budget bound was reached: the grammar stopped yielding candidates
/// this cut had not already measured. Honest, and it guarantees termination.
pub const STOPPED_BY_EXHAUSTED: &str = "search_space_exhausted";
pub const STOPPED_BY_POOL: &str = "comparison_pool_exhausted";
pub const NOT_REQUESTED: &str = "not_requested";

/// Time is cumulative active wall time on this cut, excluding other cuts and
/// publication. An in-process fit may overrun; no new fit is admitted afterward.
pub const BUDGET_POLICY: &str = "per_cut_total_attempts_and_active_seconds_nonpreemptive_v1";

/// Consecutive generations that may spend no evaluation before the search calls
/// the space exhausted. Without this a run whose offspring all collide would spin
/// until the seconds bound with nothing to show for it.
const BARREN_GENERATION_LIMIT: usize = 8;

/// What happened to one candidate.
pub const MEASURED: &str = "measured";
pub const PARSE_FAILED: &str = "parse_failed";
pub const FIT_FAILED: &str = "fit_failed";
pub const SCORE_FAILED: &str = "score_failed";
pub const PANICKED: &str = "panicked";
/// Every status an observation can carry.
pub const STATUSES: [&str; 5] = [MEASURED, PARSE_FAILED, FIT_FAILED, SCORE_FAILED, PANICKED];

/// How a candidate was produced. A seed enters generation 0 right after the bootstrap as
/// [`INCUMBENT`]; comparison re-measurements are [`REMEASURE`].
pub const BOOTSTRAP: &str = "bootstrap";
pub const INCUMBENT: &str = "incumbent";
pub const REMEASURE: &str = "remeasure";
pub const ORIGINS: [&str; 6] = [
    BOOTSTRAP,
    INCUMBENT,
    "random",
    "mutate",
    "crossover",
    REMEASURE,
];

/// Everything a search is asked to do besides which cuts to search.
#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub budget: Budget,
    pub rng_seed: u64,
    /// Canonical pipeline genomes that enter generation 0 after the bootstrap with origin
    /// [`INCUMBENT`] (already admitted by [`crate::seeds::admit`]). They are fitted on this
    /// search's own cuts like every other candidate and spend the same budget.
    pub seeds: Vec<String>,
}

impl SearchOptions {
    pub fn new(budget: Budget, rng_seed: u64) -> Self {
        SearchOptions {
            budget,
            rng_seed,
            seeds: Vec::new(),
        }
    }
}

/// Evaluation slots a search on `cuts` cuts gives exploration: all of them for one cut, half
/// (rounded up) when the rest is reserved for cross-cut comparison.
pub fn exploration_limit(budget: Budget, cuts: usize) -> usize {
    if cuts > 1 {
        budget.evaluations.div_ceil(2)
    } else {
        budget.evaluations
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub evaluations: usize,
    pub seconds: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            evaluations: 64,
            seconds: 300,
        }
    }
}

impl Budget {
    /// `evaluations=N,seconds=S`, in either order, both optional.
    pub fn parse(value: &str) -> Result<Budget> {
        let mut budget = Budget::default();
        for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, raw) = part
                .split_once('=')
                .with_context(|| format!("budget term {part:?} is not key=value"))?;
            match key.trim() {
                "evaluations" => {
                    budget.evaluations = raw
                        .trim()
                        .parse()
                        .with_context(|| format!("budget evaluations {raw:?}"))?;
                }
                "seconds" => {
                    budget.seconds = raw
                        .trim()
                        .parse()
                        .with_context(|| format!("budget seconds {raw:?}"))?;
                }
                other => bail!("unknown budget term {other:?}; expected evaluations or seconds"),
            }
        }
        budget.check()?;
        Ok(budget)
    }

    pub fn check(&self) -> Result<()> {
        ensure!(
            self.evaluations > 0 && self.evaluations <= MAX_EVALUATIONS,
            "budget evaluations must be 1..={MAX_EVALUATIONS}, got {}",
            self.evaluations
        );
        ensure!(
            self.seconds > 0 && self.seconds <= MAX_SECONDS,
            "budget seconds must be 1..={MAX_SECONDS}, got {}",
            self.seconds
        );
        Ok(())
    }
}

/// One candidate's evidence: what was tried, on which cut, and what came back.
///
/// A later work package pushes these to the learning ledger, so every field a
/// ledger row needs is here rather than reconstructed from the champion.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Observation {
    pub genome: String,
    pub program_sha256: String,
    pub cut: usize,
    pub generation: usize,
    /// How the candidate was produced: `bootstrap`, `random`, `mutate`, `crossover`.
    pub origin: &'static str,
    /// In-sample R^2 on the fit rows; present only when the candidate measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_r2: Option<f64>,
    /// Held-out R^2 on rows the fit never saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hold_r2: Option<f64>,
    /// Held-out RMSE, in the dataset's own target units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rmse: Option<f64>,
    pub seconds: f64,
    /// [`MEASURED`], or one of the failure statuses.
    pub status: &'static str,
    /// Why, whenever `status` is not [`MEASURED`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

impl Observation {
    pub fn measured(&self) -> bool {
        self.status == MEASURED
    }
}

/// An [`Observation`] as `observations.json` publishes it, read back for re-derivation.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRecord {
    pub genome: String,
    pub program_sha256: String,
    pub cut: usize,
    pub generation: usize,
    pub origin: String,
    #[serde(default)]
    pub train_r2: Option<f64>,
    #[serde(default)]
    pub hold_r2: Option<f64>,
    #[serde(default)]
    pub rmse: Option<f64>,
    pub seconds: f64,
    pub status: String,
    #[serde(default)]
    pub failure: Option<String>,
}

impl ObservationRecord {
    /// The observation this record publishes, refusing an unknown origin or status, a digest
    /// that is not its genome's, or a measured row without its scores.
    pub fn into_observation(self) -> Result<Observation> {
        let origin = ORIGINS
            .iter()
            .copied()
            .find(|o| *o == self.origin)
            .with_context(|| format!("unknown observation origin {:?}", self.origin))?;
        let status = STATUSES
            .iter()
            .copied()
            .find(|s| *s == self.status)
            .with_context(|| format!("unknown observation status {:?}", self.status))?;
        ensure!(
            self.program_sha256 == sha256_hex(self.genome.as_bytes()),
            "observation program_sha256 is not the digest of its genome"
        );
        ensure!(
            (status == MEASURED)
                == (self.rmse.is_some() && self.hold_r2.is_some() && self.train_r2.is_some())
                && (status == MEASURED) == self.failure.is_none(),
            "observation status {status} disagrees with its scores"
        );
        Ok(Observation {
            genome: self.genome,
            program_sha256: self.program_sha256,
            cut: self.cut,
            generation: self.generation,
            origin,
            train_r2: self.train_r2,
            hold_r2: self.hold_r2,
            rmse: self.rmse,
            seconds: self.seconds,
            status,
            failure: self.failure,
        })
    }
}

/// Everything one cut's search produced.
#[derive(Clone, Debug, Serialize)]
pub struct CutSearch {
    pub cut: usize,
    #[serde(skip)]
    pub output: PathBuf,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub generations: u64,
    /// Fits attempted. One fit is one evaluation, whatever its outcome.
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_or_unavailable_measurements: u64,
    /// Single cut: exploration's evaluations/seconds/search-space reason.
    /// Multiple cuts: comparison's seconds, evaluations, or pool-exhausted reason
    /// in that precedence; exploration retains its separate reason.
    pub stopped_by: &'static str,
    pub seconds: f64,
    pub exploration: Phase,
    pub comparison: Phase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_rmse: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_r2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_program: Option<String>,
    #[serde(skip)]
    pub observations: Vec<Observation>,
    #[serde(skip)]
    pub search_sha256: String,
    #[serde(skip)]
    pub observations_sha256: String,
    #[serde(skip)]
    pub rng_seed: u64,
    #[serde(skip)]
    pub budget: Budget,
}

/// A scored survivor carried between generations.
#[derive(Clone, Debug)]
struct Scored {
    genome: String,
    rmse: f64,
}

/// Derive one deterministic stream seed from the whole coordinate of a draw.
///
/// SplitMix64 finalisation over the mixed coordinate: distinct `(rng_seed, cut,
/// generation, slot, salt)` give uncorrelated streams, and the same coordinate
/// always gives the same one.
fn stream(rng_seed: u64, cut: usize, generation: usize, slot: usize, salt: u64) -> u64 {
    let mut z = rng_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((cut as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((generation as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add((slot as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
        .wrapping_add(salt);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fit one candidate on the cut's training rows and score it on the held-out
/// rows. Never returns `Err`: a failure is an [`Observation`], not an abort.
fn evaluate(genome: &str, cut: &Cut, generation: usize, origin: &'static str) -> Observation {
    let started = Instant::now();
    let record = |status, failure: Option<String>, seconds| Observation {
        genome: genome.to_owned(),
        program_sha256: sha256_hex(genome.as_bytes()),
        cut: cut.index,
        generation,
        origin,
        train_r2: None,
        hold_r2: None,
        rmse: None,
        seconds,
        status,
        failure,
    };

    let parsed = match GenomeIr::parse(genome) {
        Ok(parsed) => parsed,
        Err(error) => {
            return record(
                PARSE_FAILED,
                Some(error.to_string()),
                started.elapsed().as_secs_f64(),
            )
        }
    };

    // feat-math is a large numerical surface and the search reaches parameter
    // combinations no hand-written call does. A panic in one candidate must not
    // take the run down with it.
    let fitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = feat_math::fit(&parsed, &cut.train_x, &cut.train_y)?;
        let train = model.predict(&cut.train_x);
        let hold = model.predict(&cut.hold_x);
        anyhow::Ok((train, hold))
    }));
    let seconds = started.elapsed().as_secs_f64();

    let (train, hold) = match fitted {
        Ok(Ok(predictions)) => predictions,
        Ok(Err(error)) => return record(FIT_FAILED, Some(format!("{error:#}")), seconds),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "non-string panic payload".to_owned());
            return record(PANICKED, Some(detail), seconds);
        }
    };

    if train.len() != cut.train_y.len() || hold.len() != cut.hold_y.len() {
        return record(
            SCORE_FAILED,
            Some(format!(
                "predicted {} train and {} held-out values for {} and {} rows",
                train.len(),
                hold.len(),
                cut.train_y.len(),
                cut.hold_y.len()
            )),
            seconds,
        );
    }
    let train_r2 = r2(&cut.train_y, &train);
    let hold_r2 = r2(&cut.hold_y, &hold);
    let hold_rmse = rmse(&cut.hold_y, &hold);
    if !hold_rmse.is_finite() || !hold_r2.is_finite() || !train_r2.is_finite() {
        return record(
            SCORE_FAILED,
            Some(format!(
                "nonfinite score: train_r2 {train_r2}, hold_r2 {hold_r2}, rmse {hold_rmse}"
            )),
            seconds,
        );
    }
    Observation {
        train_r2: Some(train_r2),
        hold_r2: Some(hold_r2),
        rmse: Some(hold_rmse),
        ..record(MEASURED, None, seconds)
    }
}

/// Draw one offspring from the scored survivors, preferring one this cut has not
/// already measured. Falls back to a mutation of the best, then to the bootstrap.
fn offspring(
    parents: &[Scored],
    seen: &BTreeSet<String>,
    rng_seed: u64,
    cut: usize,
    generation: usize,
    slot: usize,
) -> (String, &'static str) {
    // Nothing measured yet on this cut, so there is nothing to mutate or cross:
    // every slot is a fresh grammar-random draw. (Reachable when a whole
    // generation failed to fit but the budget still allows more candidates.)
    if parents.is_empty() {
        let seed = stream(rng_seed, cut, generation, slot, 0);
        return (
            evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS),
            "random",
        );
    }
    for attempt in 0..DISTINCT_CHILD_ATTEMPTS {
        let seed = stream(rng_seed, cut, generation, slot, attempt as u64);
        // Crossover whenever there are two parents to cross, mutation otherwise;
        // alternating on the slot keeps both operators in every generation.
        let (child, origin) = if parents.len() >= 2 && slot.is_multiple_of(2) {
            let left = &parents[(seed as usize) % parents.len()].genome;
            let right = &parents[((seed >> 17) as usize) % parents.len()].genome;
            (
                evolve::crossover_canonical_pipelines(left, right, seed, MAX_TRANSFORMS),
                "crossover",
            )
        } else {
            let parent = &parents[(seed as usize) % parents.len()].genome;
            (
                evolve::mutate_canonical_pipeline(parent, seed, MAX_TRANSFORMS),
                "mutate",
            )
        };
        match child {
            Ok(child) if !seen.contains(&child) => return (child, origin),
            // A drawn duplicate or an operator that could not apply: try another
            // stream rather than spending a fit on a genome already measured.
            _ => continue,
        }
    }
    // Every attempt collided. Spend the slot on a fresh grammar-random pipeline
    // rather than re-measuring: it is the one draw that does not depend on the
    // population at all.
    let seed = stream(rng_seed, cut, generation, slot, 0xFFFF);
    (
        evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS),
        "random",
    )
}

/// Run one cut's search and publish its evidence under `output`.
pub fn run_cut(cut: &Cut, budget: Budget, rng_seed: u64, output: &Path) -> Result<CutSearch> {
    run_cut_allocated(cut, &SearchOptions::new(budget, rng_seed), output, false)
}

/// Multiple cuts reserve half the evaluation slots (rounded down) and half the
/// active seconds for comparison. Comparison may use any unspent exploration
/// allowance; the total requested budget never grows.
pub fn run_cuts(
    cuts: &[Cut],
    budget: Budget,
    rng_seed: u64,
    output: &Path,
) -> Result<Vec<CutSearch>> {
    run_cuts_with(cuts, &SearchOptions::new(budget, rng_seed), output)
}

/// [`run_cuts`] with seeds and the other [`SearchOptions`].
pub fn run_cuts_with(
    cuts: &[Cut],
    options: &SearchOptions,
    output: &Path,
) -> Result<Vec<CutSearch>> {
    let mut searches = Vec::new();
    for cut in cuts {
        searches.push(run_cut_allocated(
            cut,
            options,
            &output.join(format!("cut-{}", cut.index)),
            cuts.len() > 1,
        )?);
    }
    remeasure_topk_on_all_cuts(cuts, &mut searches)?;
    Ok(searches)
}

fn run_cut_allocated(
    cut: &Cut,
    options: &SearchOptions,
    output: &Path,
    reserve: bool,
) -> Result<CutSearch> {
    let (budget, rng_seed) = (options.budget, options.rng_seed);
    budget.check()?;
    ensure!(
        !output.join("search.json").exists(),
        "{}: a completed search is already published here",
        output.display()
    );
    let started = Instant::now();
    let evaluation_limit = exploration_limit(budget, if reserve { 2 } else { 1 });
    let seconds_limit = budget.seconds as f64 / if reserve { 2.0 } else { 1.0 };
    let deadline = Duration::from_secs_f64(seconds_limit);

    let mut observations: Vec<Observation> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut survivors: Vec<Scored> = Vec::new();
    let mut evaluations = 0usize;
    let mut generation = 0usize;
    let mut barren = 0usize;

    let stopped_by: &'static str = 'search: loop {
        // Generation 0 is the bootstrap plus grammar-random draws; later
        // generations are the elites' offspring. Elites carry their scores and
        // are never re-fitted, so a slot here is always an unmeasured genome.
        let candidates: Vec<(String, &'static str)> = if generation == 0 {
            let mut initial = vec![(BOOTSTRAP_GENOME.to_owned(), BOOTSTRAP)];
            // Seeds follow the bootstrap and precede every random draw, so a budget that admits
            // generation 0 at all re-measures them first; the random slots are the unseeded
            // search's own draws, unchanged.
            initial.extend(options.seeds.iter().map(|g| (g.clone(), INCUMBENT)));
            for slot in 1..POPULATION {
                let seed = stream(rng_seed, cut.index, 0, slot, 0);
                initial.push((
                    evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS),
                    "random",
                ));
            }
            initial
        } else {
            (survivors.len()..POPULATION)
                .map(|slot| offspring(&survivors, &seen, rng_seed, cut.index, generation, slot))
                .collect()
        };

        let spent_before = evaluations;
        for (genome, origin) in candidates {
            if evaluations >= evaluation_limit {
                break 'search STOPPED_BY_EVALUATIONS;
            }
            if started.elapsed() >= deadline {
                break 'search STOPPED_BY_SECONDS;
            }
            if !seen.insert(genome.clone()) {
                continue;
            }
            evaluations += 1;
            let observation = evaluate(&genome, cut, generation, origin);
            if let Some(rmse) = observation.rmse {
                survivors.push(Scored {
                    genome: genome.clone(),
                    rmse,
                });
            }
            observations.push(observation);
        }

        // A generation that spent nothing drew only genomes already measured.
        // A few in a row means the reachable space is exhausted, not that the
        // search is working; stop rather than spin out the seconds bound.
        barren = if evaluations == spent_before {
            barren + 1
        } else {
            0
        };
        if barren >= BARREN_GENERATION_LIMIT {
            break 'search STOPPED_BY_EXHAUSTED;
        }

        // Keep the best ELITES by held-out RMSE; ties on the canonical genome so
        // the surviving set is a function of the measurements alone.
        survivors.sort_by(|a, b| a.rmse.total_cmp(&b.rmse).then(a.genome.cmp(&b.genome)));
        survivors.truncate(ELITES);
        generation += 1;
    };

    let seconds = started.elapsed().as_secs_f64();
    let successful = observations.iter().filter(|o| o.measured()).count() as u64;
    let failed = observations.len() as u64 - successful;

    // Total failure is not a result. Emitting a bootstrap genome here is exactly
    // the silent non-measurement this stage was rewritten to remove.
    if successful == 0 {
        let mut reasons: Vec<String> = observations
            .iter()
            .filter_map(|o| o.failure.as_ref().map(|f| format!("{}: {f}", o.status)))
            .collect();
        reasons.sort();
        reasons.dedup();
        reasons.truncate(5);
        bail!(
            "cut {}: all {} candidate(s) failed to measure in {seconds:.1}s; \
             distinct reasons: {}",
            cut.index,
            observations.len(),
            if reasons.is_empty() {
                "none recorded (the budget ended before any candidate was fitted)".to_owned()
            } else {
                reasons.join(" | ")
            }
        );
    }

    let best = observations
        .iter()
        .filter(|o| o.measured())
        .min_by(|a, b| {
            a.rmse
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.rmse.unwrap_or(f64::INFINITY))
                .then(a.genome.cmp(&b.genome))
        })
        .expect("a successful measurement exists");

    let mut search = CutSearch {
        cut: cut.index,
        output: output.to_path_buf(),
        train_rows: cut.train_rows(),
        hold_rows: cut.hold_rows(),
        generations: generation as u64 + 1,
        completed_evaluations: evaluations as u64,
        successful_measurements: successful,
        failed_or_unavailable_measurements: failed,
        stopped_by,
        seconds,
        exploration: Phase {
            evaluation_limit: evaluation_limit as u64,
            seconds_limit,
            completed_evaluations: evaluations as u64,
            successful_measurements: successful,
            failed_measurements: failed,
            seconds,
            stopped_by: stopped_by.to_owned(),
        },
        comparison: Phase::not_requested(),
        best_validation_rmse: best.rmse,
        best_validation_r2: best.hold_r2,
        best_program: Some(best.genome.clone()),
        observations,
        search_sha256: String::new(),
        observations_sha256: String::new(),
        rng_seed,
        budget,
    };
    publish_cut_search(&mut search)?;
    Ok(search)
}

/// Up to this many of each cut's best genomes enter the bounded comparison pool.
pub const REMEASURE_TOP_K: usize = ELITES;

/// Try each cut's top-k in rank/cut order, across the other cuts. All attempts,
/// including failures, use the same total allowance as exploration. Partial
/// coverage is retained as evidence and is ineligible for selection.
pub fn remeasure_topk_on_all_cuts(cuts: &[Cut], searches: &mut [CutSearch]) -> Result<()> {
    remeasure_with(cuts, searches, evaluate)
}

fn remeasure_with(
    cuts: &[Cut],
    searches: &mut [CutSearch],
    mut evaluate_candidate: impl FnMut(&str, &Cut, usize, &'static str) -> Observation,
) -> Result<()> {
    ensure!(
        cuts.len() == searches.len(),
        "remeasure: cuts/searches length mismatch"
    );
    for (cut, search) in cuts.iter().zip(searches.iter()) {
        ensure!(cut.index == search.cut, "remeasure: cut identity mismatch");
    }
    if cuts.len() <= 1 {
        return Ok(());
    }
    // Round-robin ranks avoid comparing scores measured on different cuts when
    // ordering this bounded pool. Canonical genome breaks within-cut score ties.
    let ranked: Vec<Vec<String>> = searches
        .iter()
        .map(|search| {
            let mut measured: Vec<_> = search
                .observations
                .iter()
                .filter(|o| o.measured())
                .collect();
            measured.sort_by(|a, b| {
                a.rmse
                    .unwrap()
                    .total_cmp(&b.rmse.unwrap())
                    .then(a.genome.cmp(&b.genome))
            });
            measured
                .into_iter()
                .take(REMEASURE_TOP_K)
                .map(|o| o.genome.clone())
                .collect()
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut pool = Vec::new();
    for rank in 0..REMEASURE_TOP_K {
        for candidates in &ranked {
            if let Some(genome) = candidates.get(rank) {
                if seen.insert(genome.clone()) {
                    pool.push(genome.clone());
                }
            }
        }
    }
    for search in searches.iter_mut() {
        // Do not reset phase history if an already compared search is supplied.
        ensure!(
            search.comparison.stopped_by == NOT_REQUESTED,
            "comparison already performed"
        );
        search.comparison = Phase {
            evaluation_limit: (search.budget.evaluations as u64)
                .saturating_sub(search.completed_evaluations),
            seconds_limit: (search.budget.seconds as f64 - search.seconds).max(0.0),
            stopped_by: STOPPED_BY_POOL.to_owned(),
            ..Phase::not_requested()
        };
    }
    for genome in &pool {
        for (cut, search) in cuts.iter().zip(searches.iter_mut()) {
            let started = Instant::now();
            // Failed attempts count and are never retried on the same cut.
            let already = search.observations.iter().any(|o| &o.genome == genome);
            let elapsed = search.seconds + started.elapsed().as_secs_f64();
            if !already
                && search.completed_evaluations < search.budget.evaluations as u64
                && elapsed < search.budget.seconds as f64
            {
                let observation =
                    evaluate_candidate(genome, cut, search.generations as usize, "remeasure");
                search.completed_evaluations += 1;
                search.comparison.completed_evaluations += 1;
                if observation.measured() {
                    search.successful_measurements += 1;
                    search.comparison.successful_measurements += 1;
                } else {
                    search.failed_or_unavailable_measurements += 1;
                    search.comparison.failed_measurements += 1;
                }
                search.observations.push(observation);
            }
            let seconds = started.elapsed().as_secs_f64();
            search.seconds += seconds;
            search.comparison.seconds += seconds;
        }
    }
    for search in searches.iter_mut() {
        search.stopped_by = if search.seconds >= search.budget.seconds as f64 {
            STOPPED_BY_SECONDS
        } else if search.completed_evaluations >= search.budget.evaluations as u64 {
            STOPPED_BY_EVALUATIONS
        } else {
            STOPPED_BY_POOL
        };
        search.comparison.stopped_by = search.stopped_by.to_owned();
        if let Some(best) = search
            .observations
            .iter()
            .filter(|o| o.measured())
            .min_by(|a, b| {
                a.rmse
                    .unwrap()
                    .total_cmp(&b.rmse.unwrap())
                    .then(a.genome.cmp(&b.genome))
            })
        {
            search.best_validation_rmse = best.rmse;
            search.best_validation_r2 = best.hold_r2;
            search.best_program = Some(best.genome.clone());
        }
        publish_cut_search(search)?;
    }
    Ok(())
}

fn publish_cut_search(search: &mut CutSearch) -> Result<()> {
    let output = &search.output;
    std::fs::create_dir_all(output).with_context(|| format!("create {}", output.display()))?;
    let observations_bytes = serde_json::to_vec_pretty(&search.observations)?;
    let observations_path = output.join("observations.json");
    std::fs::write(&observations_path, &observations_bytes)
        .with_context(|| format!("write {}", observations_path.display()))?;
    search.observations_sha256 = sha256_hex(&observations_bytes);

    // `search.json` names the observations it summarises, so the pair cannot be
    // mixed up after the fact.
    let mut summary = serde_json::to_value(&search)?;
    summary["schema"] = "jope.pipeline-search/v2".into();
    summary["engine"] = SEARCH_ENGINE.into();
    summary["budget_policy"] = BUDGET_POLICY.into();
    summary["deadline_overrun"] = (search.seconds > search.budget.seconds as f64).into();
    summary["rng_seed"] = search.rng_seed.into();
    summary["budget_evaluations"] = search.budget.evaluations.into();
    summary["budget_seconds"] = search.budget.seconds.into();
    summary["observations_sha256"] = search.observations_sha256.clone().into();
    let summary_bytes = serde_json::to_vec_pretty(&summary)?;
    let search_path = output.join("search.json");
    std::fs::write(&search_path, &summary_bytes)
        .with_context(|| format!("write {}", search_path.display()))?;
    search.search_sha256 = sha256_hex(&summary_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare;

    /// `y = 3*a - 2*b + 1`: learnable, so a real search must find something.
    fn linear_rows(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let (a, b) = ((i % 13) as f64, (i % 7) as f64);
                vec![a, b, 3.0 * a - 2.0 * b + 1.0]
            })
            .collect()
    }

    fn one_cut(temp: &Path, dataset: &str) -> prepare::Cut {
        prepare::build(&linear_rows(150), dataset, 3, 1, temp)
            .unwrap()
            .cuts
            .remove(0)
    }

    #[test]
    fn budgets_parse_and_are_bounded() {
        assert_eq!(
            Budget::parse("evaluations=8,seconds=120").unwrap(),
            Budget {
                evaluations: 8,
                seconds: 120
            }
        );
        assert_eq!(
            Budget::parse(" seconds = 5 , evaluations = 2 ").unwrap(),
            Budget {
                evaluations: 2,
                seconds: 5
            }
        );
        assert_eq!(Budget::parse("seconds=9").unwrap().evaluations, 64);
        assert!(Budget::parse("evaluations=0,seconds=1").is_err());
        assert!(Budget::parse("evaluations=1001,seconds=1").is_err());
        assert!(Budget::parse("evaluations=1,seconds=601").is_err());
        assert!(Budget::parse("evaluations").is_err());
        assert!(Budget::parse("rounds=3").is_err());
    }

    #[test]
    fn the_search_measures_and_records_per_candidate_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "measures");
        let budget = Budget {
            evaluations: 20,
            seconds: 120,
        };
        let search = run_cut(&cut, budget, 7, &temp.path().join("search")).unwrap();

        assert!(
            search.successful_measurements > 0,
            "nothing measured: {:?}",
            search.observations
        );
        assert_eq!(
            search.completed_evaluations,
            search.successful_measurements + search.failed_or_unavailable_measurements
        );
        let best = search.best_validation_rmse.unwrap();
        assert!(best.is_finite() && best >= 0.0, "{best}");
        assert!(search.best_program.is_some());

        for observation in &search.observations {
            assert_eq!(observation.cut, cut.index);
            assert_eq!(observation.program_sha256.len(), 64);
            if observation.measured() {
                assert!(observation.rmse.unwrap().is_finite());
                assert!(observation.hold_r2.unwrap().is_finite());
                assert!(observation.train_r2.unwrap().is_finite());
                assert!(observation.failure.is_none());
            } else {
                assert!(observation.failure.is_some(), "{observation:?}");
            }
        }
        // The published evidence is on disk and hashed.
        assert!(temp.path().join("search/observations.json").is_file());
        assert_eq!(search.observations_sha256.len(), 64);
        assert_eq!(search.search_sha256.len(), 64);
    }

    #[test]
    fn the_bootstrap_genome_is_always_measured_first() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "bootstrap");
        let search = run_cut(
            &cut,
            Budget {
                evaluations: 3,
                seconds: 60,
            },
            7,
            &temp.path().join("search"),
        )
        .unwrap();
        assert_eq!(search.observations[0].genome, BOOTSTRAP_GENOME);
        assert_eq!(search.observations[0].origin, "bootstrap");
        // The champion can only be at least as good as the bootstrap.
        let bootstrap = search.observations[0].rmse.unwrap();
        assert!(search.best_validation_rmse.unwrap() <= bootstrap);
    }

    #[test]
    fn comparison_stays_within_total_budget_and_keeps_a_common_pool() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "bounded",
            7,
            3,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let searches = run_cuts(
            &preparation.cuts,
            Budget {
                evaluations: 4,
                seconds: 60,
            },
            7,
            &temp.path().join("search"),
        )
        .unwrap();
        let mut common: Option<BTreeSet<String>> = None;
        for search in &searches {
            assert!(search.completed_evaluations <= 4, "overspent: {search:?}");
            assert_eq!(
                search.completed_evaluations,
                search.observations.len() as u64
            );
            assert_eq!(
                search.completed_evaluations,
                search.exploration.completed_evaluations + search.comparison.completed_evaluations
            );
            assert_eq!(search.exploration.evaluation_limit, 2);
            assert_eq!(search.comparison.evaluation_limit, 2);
            let measured: BTreeSet<String> = search
                .observations
                .iter()
                .filter(|o| o.measured())
                .map(|o| o.genome.clone())
                .collect();
            common = Some(match common {
                Some(c) => c.intersection(&measured).cloned().collect(),
                None => measured,
            });
        }
        assert!(common.unwrap().iter().any(|g| g != BOOTSTRAP_GENOME));
        assert_eq!(
            crate::select::select(&searches)
                .unwrap()
                .best
                .unwrap()
                .cuts_measured,
            3
        );
    }

    fn comparison_fixture(temp: &Path, reserve: bool) -> (Vec<Cut>, Vec<CutSearch>) {
        let cuts = prepare::build(
            &linear_rows(150),
            "comparison-limits",
            7,
            2,
            &temp.join("prepared"),
        )
        .unwrap()
        .cuts;
        let searches = cuts
            .iter()
            .map(|cut| {
                run_cut_allocated(
                    cut,
                    &SearchOptions::new(
                        Budget {
                            evaluations: 8,
                            seconds: 60,
                        },
                        7,
                    ),
                    &temp.join(format!("search-{}", cut.index)),
                    reserve,
                )
                .unwrap()
            })
            .collect();
        (cuts, searches)
    }

    #[test]
    fn comparison_never_admits_a_fit_after_the_total_evaluation_limit() {
        let temp = tempfile::tempdir().unwrap();
        let (cuts, mut searches) = comparison_fixture(temp.path(), false);
        remeasure_with(&cuts, &mut searches, |_, _, _, _| {
            panic!("admitted an over-budget fit")
        })
        .unwrap();
        assert!(searches
            .iter()
            .all(|s| s.completed_evaluations == 8 && s.comparison.completed_evaluations == 0));
    }

    #[test]
    fn comparison_never_admits_a_fit_after_the_total_seconds_limit() {
        let temp = tempfile::tempdir().unwrap();
        let (cuts, mut searches) = comparison_fixture(temp.path(), true);
        for search in &mut searches {
            search.seconds = 60.0;
            search.exploration.seconds = 60.0;
        }
        remeasure_with(&cuts, &mut searches, |_, _, _, _| {
            panic!("admitted a late fit")
        })
        .unwrap();
        assert!(
            searches
                .iter()
                .all(|s| s.stopped_by == STOPPED_BY_SECONDS
                    && s.comparison.completed_evaluations == 0)
        );
    }

    #[test]
    fn a_nonpreemptible_overrun_prevents_the_next_fit() {
        let temp = tempfile::tempdir().unwrap();
        let (cuts, mut searches) = comparison_fixture(temp.path(), true);
        searches[0].seconds = 59.99;
        searches[1].seconds = 60.0;
        let mut calls = 0;
        remeasure_with(&cuts, &mut searches, |genome, cut, generation, origin| {
            calls += 1;
            std::thread::sleep(Duration::from_millis(25));
            evaluate(genome, cut, generation, origin)
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert!(searches[0].seconds > 60.0);
        assert_eq!(searches[0].stopped_by, STOPPED_BY_SECONDS);
    }

    #[test]
    fn comparison_failures_spend_budget_and_are_not_retried() {
        let temp = tempfile::tempdir().unwrap();
        let (cuts, mut searches) = comparison_fixture(temp.path(), true);
        let mut calls = 0;
        remeasure_with(&cuts, &mut searches, |genome, cut, generation, origin| {
            calls += 1;
            Observation {
                genome: genome.to_owned(),
                program_sha256: sha256_hex(genome.as_bytes()),
                cut: cut.index,
                generation,
                origin,
                train_r2: None,
                hold_r2: None,
                rmse: None,
                seconds: 0.0,
                status: FIT_FAILED,
                failure: Some("injected failure".to_owned()),
            }
        })
        .unwrap();
        assert!(calls > 0);
        for search in searches {
            assert!(search.completed_evaluations <= 8);
            assert_eq!(
                search.comparison.completed_evaluations,
                search.comparison.failed_measurements
            );
            assert_eq!(
                search
                    .observations
                    .iter()
                    .map(|o| &o.genome)
                    .collect::<BTreeSet<_>>()
                    .len(),
                search.observations.len()
            );
        }
    }

    #[test]
    fn a_one_evaluation_budget_has_explicit_bootstrap_only_coverage() {
        let temp = tempfile::tempdir().unwrap();
        let cuts = prepare::build(
            &linear_rows(150),
            "tiny",
            7,
            3,
            &temp.path().join("prepared"),
        )
        .unwrap()
        .cuts;
        let searches = run_cuts(
            &cuts,
            Budget {
                evaluations: 1,
                seconds: 60,
            },
            7,
            &temp.path().join("search"),
        )
        .unwrap();
        assert!(searches
            .iter()
            .all(|s| s.completed_evaluations == 1 && s.comparison.completed_evaluations == 0));
        let selected = crate::select::select(&searches).unwrap();
        assert_eq!(selected.genome, BOOTSTRAP_GENOME);
        assert_eq!(selected.coverage.len(), 1);
        assert_eq!(selected.compared_cuts, vec![0, 1, 2]);
    }

    #[test]
    fn the_evaluation_bound_stops_the_search_and_is_recorded() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "evalbound");
        let search = run_cut(
            &cut,
            Budget {
                evaluations: 5,
                seconds: 600,
            },
            7,
            &temp.path().join("search"),
        )
        .unwrap();
        assert_eq!(search.completed_evaluations, 5);
        assert_eq!(search.stopped_by, STOPPED_BY_EVALUATIONS);
    }

    #[test]
    fn the_seconds_bound_stops_the_search_and_is_recorded() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "secondsbound");
        // One second against a 1000-evaluation allowance: time must bind first.
        let search = run_cut(
            &cut,
            Budget {
                evaluations: 1000,
                seconds: 1,
            },
            7,
            &temp.path().join("search"),
        )
        .unwrap();
        assert_eq!(search.stopped_by, STOPPED_BY_SECONDS);
        assert!(search.completed_evaluations < 1000);
    }

    /// Everything the search decides replays; only the wall clock does not.
    ///
    /// `Observation::seconds` is a real measurement of this machine, so
    /// `observations.json` — and therefore `observations_sha256` — is not
    /// byte-stable across runs by design. The genomes, the order they were
    /// drawn in, their statuses and every metric are.
    #[test]
    fn the_same_seed_replays_the_same_search() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "replay");
        let budget = Budget {
            evaluations: 16,
            seconds: 300,
        };
        let a = run_cut(&cut, budget, 21, &temp.path().join("a")).unwrap();
        let b = run_cut(&cut, budget, 21, &temp.path().join("b")).unwrap();
        assert_eq!(a.best_program, b.best_program);
        assert_eq!(a.best_validation_rmse, b.best_validation_rmse);
        assert_eq!(a.best_validation_r2, b.best_validation_r2);
        assert_eq!(a.completed_evaluations, b.completed_evaluations);
        assert_eq!(a.successful_measurements, b.successful_measurements);
        assert_eq!(a.generations, b.generations);
        let decided = |s: &CutSearch| {
            s.observations
                .iter()
                .map(|o| {
                    (
                        o.genome.clone(),
                        o.generation,
                        o.origin,
                        o.status,
                        o.train_r2,
                        o.hold_r2,
                        o.rmse,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(decided(&a), decided(&b));
    }

    #[test]
    fn a_different_seed_searches_a_different_population() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "seeds");
        let budget = Budget {
            evaluations: 16,
            seconds: 300,
        };
        let a = run_cut(&cut, budget, 21, &temp.path().join("a")).unwrap();
        let b = run_cut(&cut, budget, 22, &temp.path().join("b")).unwrap();
        let genomes = |s: &CutSearch| {
            s.observations
                .iter()
                .map(|o| o.genome.clone())
                .collect::<Vec<_>>()
        };
        assert_ne!(genomes(&a), genomes(&b));
    }

    #[test]
    fn no_candidate_is_fitted_twice_on_one_cut() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "distinct");
        let search = run_cut(
            &cut,
            Budget {
                evaluations: 30,
                seconds: 300,
            },
            5,
            &temp.path().join("search"),
        )
        .unwrap();
        let mut genomes: Vec<&str> = search
            .observations
            .iter()
            .map(|o| o.genome.as_str())
            .collect();
        let total = genomes.len();
        genomes.sort_unstable();
        genomes.dedup();
        assert_eq!(genomes.len(), total, "a genome was measured twice");
    }

    /// Generation 1 with nothing measured must draw fresh genomes, not index an
    /// empty parent set. `offspring` is called directly because reaching this
    /// state through `run_cut` needs a dataset every candidate fails on.
    #[test]
    fn an_empty_parent_set_draws_a_fresh_candidate_instead_of_panicking() {
        let seen = BTreeSet::new();
        let (genome, origin) = offspring(&[], &seen, 7, 0, 1, 3);
        assert_eq!(origin, "random");
        assert!(GenomeIr::parse(&genome).is_ok(), "{genome}");
    }

    /// One parent is enough to mutate from; crossover needs two.
    #[test]
    fn a_single_parent_still_produces_a_child() {
        let seen = BTreeSet::new();
        let parents = [Scored {
            genome: BOOTSTRAP_GENOME.to_owned(),
            rmse: 1.0,
        }];
        for slot in 0..4 {
            let (genome, origin) = offspring(&parents, &seen, 7, 0, 1, slot);
            assert!(GenomeIr::parse(&genome).is_ok(), "{genome}");
            assert!(matches!(origin, "mutate" | "random"), "{origin}");
        }
    }

    /// The guarantee this whole stage exists for: a search that measured nothing
    /// must say so and fail, not quietly hand a bootstrap genome to the emitters
    /// the way the engine-backed search did on every run.
    #[test]
    fn a_search_that_measures_nothing_fails_loudly_instead_of_falling_back() {
        // A NaN target: every candidate fits or not, but nothing can be scored.
        let cut = Cut {
            index: 0,
            train_x: (0..40).map(|i| vec![i as f64, (i % 5) as f64]).collect(),
            train_y: vec![f64::NAN; 40],
            hold_x: (0..10).map(|i| vec![i as f64, (i % 3) as f64]).collect(),
            hold_y: vec![f64::NAN; 10],
            hold_indices: (0..10).collect(),
        };
        let temp = tempfile::tempdir().unwrap();
        let error = format!(
            "{:#}",
            run_cut(
                &cut,
                Budget {
                    evaluations: 6,
                    seconds: 60,
                },
                7,
                &temp.path().join("search"),
            )
            .unwrap_err()
        );
        assert!(error.contains("failed to measure"), "{error}");
        assert!(error.contains("cut 0"), "{error}");
        // The reasons are named, not swallowed.
        assert!(
            error.contains("score_failed") || error.contains("fit_failed"),
            "{error}"
        );
    }

    #[test]
    fn a_republished_search_is_refused_rather_than_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "republish");
        let budget = Budget {
            evaluations: 4,
            seconds: 60,
        };
        let output = temp.path().join("search");
        run_cut(&cut, budget, 7, &output).unwrap();
        let error = format!("{:#}", run_cut(&cut, budget, 7, &output).unwrap_err());
        assert!(error.contains("already published"), "{error}");
    }
}
