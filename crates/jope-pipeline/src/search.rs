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
//!   the slot, so a search that the evaluation budget ends replays from the
//!   command line alone. When the seconds deadline binds, how many candidates
//!   start before it depends on timing (even at one thread), so the same command
//!   can admit and select differently; that phase records `seconds` as its stop
//!   reason (see `docs/pipeline.md`).
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
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use crate::{
    clock::{Clock, Stamp},
    islands::{island_seed, next_seeds, Champion},
    prepare::Cut,
    receipt::Phase,
    search_events::{
        self, BehaviorLaw, EditTransition, Effort, EvidenceRef, Recorder, SearchBinding,
        SearchDecision, SearchEvent, SearchEventLog,
    },
    sha256_hex,
};

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
    /// search's own cuts like every other candidate and spend the same evaluation budget; the
    /// seconds bound never skips them ([`seconds_exempt`]).
    pub seeds: Vec<String>,
    /// Worker threads evaluating one generation's admitted candidates. `1` evaluates one at a
    /// time, exactly as the sequential search always has (on the evaluation pool's single
    /// thread, so on the same stack as any other count). While the evaluation budget ends every
    /// phase, any count gives the same decisions, because candidates are admitted and folded
    /// back in slot order. A binding seconds bound is checked again before each fit starts, on
    /// its worker (`evaluate_admitted`), so how many candidates start before it depends on
    /// timing and so on this count; the phase then records [`STOPPED_BY_SECONDS`].
    pub threads: usize,
    /// Independent random streams over the same cuts, sharing each cut's budget and its
    /// "never fit twice" rule. Island 0 is the `--rng-seed` stream itself, so one island is the
    /// plain search. Each island is evaluated on `threads` workers.
    pub islands: usize,
    /// Every this many generations each island's top-1 is carried into every other island's
    /// elites (with its existing measurement: a migrant is never re-fitted).
    pub migration_interval: usize,
    pub clock: Clock,
    /// Set by SIGTERM/SIGINT ([`crate::parallel::termination_flag`]). Checked before every fit:
    /// once set, nothing new starts, in-flight fits finish, and the search returns
    /// [`crate::parallel::Cancelled`] without publishing the interrupted cut.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Additive sidecar recording. Disable only for observational parity/control runs.
    pub record_events: bool,
    /// Original run/role/profile bindings, when supplied by the admitted caller.
    pub event_binding: Option<SearchBinding>,
    /// Live `jope.invent.progress.v1` lines ([`crate::progress`]); `None` reports nothing.
    /// Advisory only: it does not mutate decisions or observations. Reporter overhead can
    /// affect wall timings and admission when the seconds deadline binds.
    pub progress: Option<Arc<crate::progress::Progress>>,
}

/// Generations between migrations when `--migration-interval` is not given.
pub const DEFAULT_MIGRATION_INTERVAL: usize = 4;

impl SearchOptions {
    pub fn new(budget: Budget, rng_seed: u64) -> Self {
        SearchOptions {
            budget,
            rng_seed,
            seeds: Vec::new(),
            threads: 1,
            islands: 1,
            migration_interval: DEFAULT_MIGRATION_INTERVAL,
            clock: Clock::Wall,
            cancel: None,
            record_events: true,
            event_binding: None,
            progress: None,
        }
    }

    /// `islands x threads`: the evaluation pool's size.
    pub fn workers(&self) -> usize {
        self.threads.max(1).saturating_mul(self.islands.max(1))
    }

    pub fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    }

    fn check_cancelled(&self) -> Result<()> {
        if self.cancelled() {
            return Err(crate::parallel::Cancelled.into());
        }
        Ok(())
    }
}

/// Generation 0's head — the bootstrap and every seed — is exempt from the seconds bound, both
/// when it is admitted and when its fit starts; the evaluation bound and a stop request still
/// apply. Re-measuring the seeds on this search's own cuts is what the receipt's seeding record,
/// its evidence verification ([`crate::evidence::verify_run`]: an [`INCUMBENT`] observation in
/// generation 0 on every cut) and a lineage's "never worse than the round before" rest on, so a
/// slow seed fit (or a loaded host) must not keep a later seed from being measured. They precede
/// every random draw, the run refuses up front a budget whose evaluations cannot cover them, and
/// the random draws after them are stopped by the seconds bound exactly as before. An overrun is
/// recorded as `deadline_overrun`, like any in-flight fit's.
pub fn seconds_exempt(generation: usize, origin: &str) -> bool {
    generation == 0 && (origin == BOOTSTRAP || origin == INCUMBENT)
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
    /// Opt-in I16 resource vector. This does not change legacy search admission
    /// or make an in-process fit preemptible. Callers reserve all other stages
    /// and terminal headroom separately in the original experiment window.
    pub fn resource_cap_v1(
        &self,
        cpu_slots: u64,
        memory_bytes: u64,
    ) -> Result<crate::resource_ledger::Amounts> {
        use crate::resource_ledger::{Amounts, Unit};
        self.check()?;
        ensure!(
            cpu_slots > 0 && memory_bytes > 0,
            "empty resource allocation"
        );
        let wall = self
            .seconds
            .checked_mul(1_000_000)
            .context("wall cap overflow")?;
        Ok(Amounts::new([
            (Unit::AttemptWallMicros, wall),
            (
                Unit::CpuCoreMicros,
                wall.checked_mul(cpu_slots).context("CPU cap overflow")?,
            ),
            (Unit::PeakMemoryBytes, memory_bytes),
            (Unit::Evaluations, u64::try_from(self.evaluations)?),
        ]))
    }

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
    /// The island that drew this candidate, when the search ran more than one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub island: Option<usize>,
}

impl Observation {
    /// Physical in-process observation adapter; wall is measured, CPU and
    /// memory remain explicitly unknown. This cannot qualify a hard fit bound.
    pub fn resource_cost_v1(
        &self,
        reservation: &crate::resource_ledger::Reservation,
        ended_unix_us: u64,
    ) -> Result<crate::resource_ledger::AttemptCost> {
        use crate::resource_ledger::{
            AttemptCost, Disposition, Measurement, Reading, Unit, SCHEMA,
        };
        ensure!(
            matches!(
                reservation.request.stage,
                crate::resource_ledger::Stage::Fit | crate::resource_ledger::Stage::Evaluate
            ),
            "search observation requires fit/evaluation reservation"
        );
        let mut measurements = std::collections::BTreeMap::new();
        for &unit in reservation.request.requested.0.keys() {
            let reading = match unit {
                Unit::AttemptWallMicros => {
                    Reading::exact(crate::resource_ledger::seconds_to_micros(self.seconds)?)
                }
                Unit::Evaluations => Reading::exact(1),
                Unit::Retries => Reading::exact(u64::from(reservation.request.retry_of.is_some())),
                _ => Reading::unknown(
                    "legacy in-process search has no physical collector for this dimension",
                ),
            };
            measurements.insert(
                unit,
                Measurement {
                    reading,
                    collector: "search-observation:nonpreemptive-v1".into(),
                    clock: "active-wall:search-clock".into(),
                    allocation: reservation.request.ids.experiment_sha256.clone(),
                    scope: vec![
                        "experiment".into(),
                        reservation.request.ids.attempt_id.clone(),
                    ],
                    from_unix_us: reservation.reserved_unix_us,
                    to_unix_us: ended_unix_us,
                },
            );
        }
        let digest = crate::sha256_hex(&serde_json::to_vec(self)?);
        Ok(AttemptCost {
            schema: SCHEMA.into(),
            reservation_sha256: reservation.identity()?,
            ids: reservation.request.ids.clone(),
            disposition: if self.measured() {
                Disposition::Completed
            } else {
                Disposition::Failed
            },
            ended_unix_us,
            exit_code: None,
            signal: None,
            measurements,
            evidence_sha256: [digest.clone()].into(),
            retained_bytes: Reading::unknown(
                "search observation has no retained storage byte collector",
            ),
            retained_sha256: if self.measured() {
                [digest].into()
            } else {
                Default::default()
            },
            reason: self.failure.clone().unwrap_or_else(|| {
                "measured in-process; nonpreemptive; CPU/memory unavailable".into()
            }),
        })
    }

    pub fn measured(&self) -> bool {
        self.status == MEASURED
    }
}

/// An [`Observation`] as `observations.json` publishes it, read back for re-derivation.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
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
    #[serde(default)]
    pub island: Option<usize>,
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
            island: self.island,
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
    /// Every champion migration between islands on this cut (none with one island).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub migrations: Vec<crate::receipt::Migration>,
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
type Scored = Champion;

/// One island: its own random stream and its own elites.
struct Island {
    rng_seed: u64,
    survivors: Vec<Scored>,
}

/// Carry each island's top-1 into every other island's elites. The receiving island keeps the
/// best [`ELITES`] of its own elites and the migrants ([`next_seeds`]), so a migrant enters only
/// when it beats one of them; either way the event is recorded. Pure in the islands' state:
/// deterministic given the seed.
fn migrate(
    islands: &mut [Island],
    cut: usize,
    after_generation: usize,
    events: &mut Vec<crate::receipt::Migration>,
    recorder: Option<&Recorder>,
) {
    let tops: Vec<(usize, Champion)> = islands
        .iter()
        .enumerate()
        .filter_map(|(index, island)| island.survivors.first().map(|c| (index, c.clone())))
        .collect();
    for (to, island) in islands.iter_mut().enumerate() {
        let incoming: Vec<&(usize, Champion)> =
            tops.iter().filter(|(from, _)| *from != to).collect();
        if incoming.is_empty() {
            continue;
        }
        let before: BTreeSet<String> = island.survivors.iter().map(|c| c.genome.clone()).collect();
        let mut pool = island.survivors.clone();
        pool.extend(incoming.iter().map(|(_, c)| c.clone()));
        island.survivors = next_seeds(&pool, ELITES);
        let mut counted = BTreeSet::new();
        for (from, champion) in incoming {
            let admitted = !before.contains(&champion.genome)
                && counted.insert(champion.genome.clone())
                && island.survivors.iter().any(|c| c.genome == champion.genome);
            record_draw(
                recorder,
                format!("migration/{after_generation}/{from}/{to}"),
                "migration",
                Some(&champion.genome),
                None,
                &[],
                if admitted {
                    "migration_admitted"
                } else {
                    "migration_rejected"
                },
                None,
            );
            events.push(crate::receipt::Migration {
                cut: cut as u64,
                after_generation: after_generation as u64,
                from_island: *from as u64,
                to_island: to as u64,
                program_sha256: sha256_hex(champion.genome.as_bytes()),
                rmse: champion.rmse,
                admitted,
            });
        }
    }
}

/// Derive one deterministic stream seed from the whole coordinate of a draw.
///
/// SplitMix64 finalisation over the mixed coordinate: distinct `(rng_seed, cut,
/// generation, slot, salt)` give uncorrelated streams, and the same coordinate
/// always gives the same one.
pub(crate) fn stream(rng_seed: u64, cut: usize, generation: usize, slot: usize, salt: u64) -> u64 {
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
pub(crate) fn evaluate(
    genome: &str,
    cut: &Cut,
    generation: usize,
    origin: &'static str,
    clock: Clock,
) -> Observation {
    let started = clock.start();
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
        island: None,
    };

    let parsed = match GenomeIr::parse(genome) {
        Ok(parsed) => parsed,
        Err(error) => {
            return record(
                PARSE_FAILED,
                Some(error.to_string()),
                started.elapsed().as_secs_f64(),
            );
        }
    };

    // feat-math is a large numerical surface and the search reaches parameter
    // combinations no hand-written call does. A panic in one candidate must not
    // take the run down with it.
    let fitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = feat_math::fit(&parsed, &cut.train_x, &cut.train_y)?;
        let train = model.predict(&cut.train_x)?;
        let hold = model.predict(&cut.hold_x)?;
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
#[cfg(test)]
fn offspring(
    parents: &[Scored],
    seen: &BTreeSet<String>,
    rng_seed: u64,
    cut: usize,
    generation: usize,
    slot: usize,
) -> (String, &'static str) {
    let (genome, origin, _) =
        offspring_recorded(parents, seen, rng_seed, cut, generation, slot, 0, None);
    (genome, origin)
}

#[allow(clippy::too_many_arguments)]
fn offspring_recorded(
    parents: &[Scored],
    seen: &BTreeSet<String>,
    rng_seed: u64,
    cut: usize,
    generation: usize,
    slot: usize,
    island: usize,
    recorder: Option<&Recorder>,
) -> (String, &'static str, Option<String>) {
    let coordinate = |attempt| format!("exploration/{generation}/{island}/{slot}/{attempt}");
    if parents.is_empty() {
        let seed = stream(rng_seed, cut, generation, slot, 0);
        let genome = evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS);
        let draw = record_draw(
            recorder,
            coordinate(0),
            "random",
            Some(&genome),
            Some(seed),
            &[],
            "candidate",
            None,
        );
        return (genome, "random", draw);
    }
    for attempt in 0..DISTINCT_CHILD_ATTEMPTS {
        let seed = stream(rng_seed, cut, generation, slot, attempt as u64);
        let (child, origin, selected) = if parents.len() >= 2 && slot.is_multiple_of(2) {
            let left = &parents[(seed as usize) % parents.len()].genome;
            let right = &parents[((seed >> 17) as usize) % parents.len()].genome;
            (
                evolve::crossover_canonical_pipelines(left, right, seed, MAX_TRANSFORMS),
                "crossover",
                vec![left.as_str(), right.as_str()],
            )
        } else {
            let parent = &parents[(seed as usize) % parents.len()].genome;
            (
                evolve::mutate_canonical_pipeline(parent, seed, MAX_TRANSFORMS),
                "mutate",
                vec![parent.as_str()],
            )
        };
        match child {
            Ok(child) => {
                let duplicate = seen.contains(&child);
                let draw = record_draw(
                    recorder,
                    coordinate(attempt),
                    origin,
                    Some(&child),
                    Some(seed),
                    &selected,
                    if duplicate { "duplicate" } else { "candidate" },
                    None,
                );
                if !duplicate {
                    return (child, origin, draw);
                }
            }
            Err(error) => {
                record_draw(
                    recorder,
                    coordinate(attempt),
                    origin,
                    None,
                    Some(seed),
                    &selected,
                    "operator_failed",
                    Some(error.to_string()),
                );
            }
        }
    }
    let seed = stream(rng_seed, cut, generation, slot, 0xFFFF);
    let genome = evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS);
    let draw = record_draw(
        recorder,
        coordinate(DISTINCT_CHILD_ATTEMPTS),
        "random_fallback",
        Some(&genome),
        Some(seed),
        &[],
        "candidate",
        None,
    );
    (genome, "random", draw)
}

#[allow(clippy::too_many_arguments)]
fn record_draw(
    recorder: Option<&Recorder>,
    coordinate: String,
    source: &str,
    genome: Option<&str>,
    seed: Option<u64>,
    parents: &[&str],
    outcome: &str,
    detail: Option<String>,
) -> Option<String> {
    if let Some(program) = genome {
        feat_math::capture_gp_source!(
            "pipeline",
            &format!("{coordinate}/{source}/{outcome}"),
            program
        );
    }
    recorder.and_then(|recorder| recorder.update(|log| {
        let canonical = genome.and_then(search_events::canonical_recipe);
        let program = genome.map(|g| sha256_hex(g.as_bytes()));
        let request_id = program.as_ref().map(|g| log.request_id(g, &canonical)).transpose()?;
        let behavior_law = if matches!(source, BOOTSTRAP | INCUMBENT | REMEASURE | "migration") {
            BehaviorLaw::NotApplicable { reason: "deterministic population insertion/reuse; no sampled child probability".into() }
        } else { BehaviorLaw::Unknown { reason: "actual marginal program law has not been reconstructed; seed/parent likelihood is not child density".into() } };
        let draw = log.append(SearchEvent::Draw { decision: SearchDecision {
            schema: search_events::DECISION_SCHEMA.into(), coordinate: coordinate.clone(), source: source.into(), seed_hex: seed.map(|s| format!("{s:016x}")),
            program_sha256: program, canonical_sha256: canonical.clone(), request_id, outcome: outcome.into(), detail: detail.clone(), behavior_law,
            proposal_cost: Effort::unknown("proposal stage not timed independently; no evaluation cost implied"),
        } })?;
        if matches!(source, "mutate" | "crossover" | "random_fallback") {
            let parent_hashes: Vec<_> = parents.iter().map(|p| sha256_hex(p.as_bytes())).collect();
            log.append(SearchEvent::Edit { transition: EditTransition {
                schema: search_events::EDIT_SCHEMA.into(), draw_id: draw.clone(), donor: if source == "crossover" { parent_hashes.get(1).cloned() } else { None },
                parents: parent_hashes, parent_canonical_sha256: parents.iter().map(|p| search_events::canonical_recipe(p)).collect(),
                sibling_group: coordinate.rsplit_once('/').map(|(group,_)| group).unwrap_or(&coordinate).into(), operator: source.into(),
                parameters: [("max_transforms".into(), MAX_TRANSFORMS.to_string()), ("stream_seed_hex".into(), seed.map(|s| format!("{s:016x}")).unwrap_or_default())].into(),
                post_canonical_sha256: canonical.clone(), legal: canonical.is_some(), refusal: if canonical.is_none() { Some(detail.unwrap_or_else(|| "parser refused operator output".into())) } else { None },
            } })?;
        }
        Ok(draw)
    }))
}

fn admission(recorder: Option<&Recorder>, draw: &Option<String>, disposition: &str) {
    if let (Some(recorder), Some(draw)) = (recorder, draw) {
        recorder.update(|log| {
            log.append(SearchEvent::Admission {
                draw_id: draw.clone(),
                disposition: disposition.into(),
            })
        });
    }
}

fn record_evaluation(
    recorder: Option<&Recorder>,
    draw: &Option<String>,
    clock: Clock,
    evaluate: impl FnOnce() -> Observation,
) -> Observation {
    let attempt = recorder.and_then(|r| {
        draw.as_ref()
            .and_then(|d| r.update(|log| log.start(d, 1, None)))
    });
    let started = recorder.map(|_| clock.start());
    let observation = evaluate();
    let elapsed = started.map(|s| s.secs());
    record_evaluation_result(recorder, attempt, observation, elapsed)
}

fn record_evaluation_result(
    recorder: Option<&Recorder>,
    attempt: Option<String>,
    observation: Observation,
    elapsed: Option<f64>,
) -> Observation {
    if let (Some(recorder), Some(attempt_id)) = (recorder, attempt) {
        recorder.update(|log| {
            let mut cost = Effort::wall(elapsed.context("missing attempt clock")?)?;
            cost.reason = "full parse/fit/predict/score wall span; observer publication, CPU/GPU/memory not instrumented".into();
            log.append(SearchEvent::Finished {
                attempt_id,
                status: observation.status.into(),
                reason: observation.failure.clone(),
                observation_sha256: Some(sha256_hex(&serde_json::to_vec(&observation)?)),
                cost,
            })
        });
    }
    observation
}

fn search_recorder(cut: &Cut, options: &SearchOptions) -> Result<Option<Recorder>> {
    if !options.record_events {
        return Ok(None);
    }
    let context = search_events::cut_context_sha256(cut)?;
    let binding = options.event_binding.clone().unwrap_or_else(|| SearchBinding {
        run_id: sha256_hex(
            format!("cut:{}:seed:{}:context:{context}", cut.index, options.rng_seed).as_bytes(),
        ), task: "regression".into(),
        experiment: EvidenceRef::unknown("legacy search caller supplied no experiment manifest"),
        role_exposure: EvidenceRef::unknown("legacy search caller supplied no authenticated I04 exposure receipt; observation grants no replay eligibility"),
        context_sha256: context.clone(), source_sha256: search_events::source_sha256(),
        evaluator: EvidenceRef::unknown("legacy evaluator identity not provided by caller"),
        profile: EvidenceRef::unknown("legacy GP does not select a registered SearchProfile ID/revision"),
        legal_set: EvidenceRef::unknown("legacy canonical pipeline grammar has no adopted legal-set receipt"),
        fidelity: "full-cut-fit-and-heldout-score".into(),
    });
    // A caller binding must identify these exact numerical inputs, never a different cut.
    ensure!(
        binding.context_sha256 == context,
        "search-event context differs from actual cut inputs"
    );
    ensure!(
        binding.source_sha256 == search_events::source_sha256(),
        "search-event source differs from actual implementation"
    );
    Ok(Some(Recorder::new(SearchEventLog::new(binding)?)))
}

/// Run one cut's search and publish its evidence under `output` (sequentially, on the
/// evaluation stack).
pub fn run_cut(cut: &Cut, budget: Budget, rng_seed: u64, output: &Path) -> Result<CutSearch> {
    crate::parallel::on_evaluation_stack(|| {
        run_cut_allocated(
            cut,
            &SearchOptions::new(budget, rng_seed),
            None,
            output,
            false,
        )
    })?
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

/// [`run_cuts`] with seeds, worker threads and the other [`SearchOptions`].
///
/// The worker count bounds concurrency only. Admission (which candidates are fitted, up to the
/// evaluation bound) and the fold of results back into the search (survivors, observations,
/// counters) both happen in slot order on the calling thread, so while the evaluation budget,
/// not the seconds bound, ends every phase, any thread count yields the same observations, the
/// same selection and the same published bytes; only wall-clock readings differ, as they do
/// between any two runs. The seconds bound is checked again before each fit starts, on
/// whichever worker runs it, so when it binds, how many candidates start before it depends on
/// how fast the fits run, and with it the observations and possibly the selection. Such a phase
/// records [`STOPPED_BY_SECONDS`] as its stop reason.
///
/// Every fit runs on the pool's threads, which have
/// [`crate::parallel::EVALUATION_STACK_BYTES`] of stack: one worker keeps the sequential
/// slot-by-slot loop, but on the pool's single thread rather than the caller's, so no worker
/// count gives a fit less stack than another.
pub fn run_cuts_with(
    cuts: &[Cut],
    options: &SearchOptions,
    output: &Path,
) -> Result<Vec<CutSearch>> {
    let pool = crate::parallel::pool(options.workers())?;
    let concurrent = (options.workers() > 1).then_some(&pool);
    pool.install(|| {
        let mut searches = Vec::new();
        for (position, cut) in cuts.iter().enumerate() {
            if let Some(progress) = &options.progress {
                progress.begin_cut(position);
            }
            searches.push(run_cut_allocated(
                cut,
                options,
                concurrent,
                &output.join(format!("cut-{}", cut.index)),
                cuts.len() > 1,
            )?);
        }
        remeasure_in(cuts, &mut searches, options, concurrent)?;
        if let Some(progress) = &options.progress {
            progress.finish();
        }
        Ok(searches)
    })
}

/// Evaluate `admitted` in order. Each candidate starts only while the cut's seconds bound
/// allows it ([`seconds_exempt`] candidates always, unless stopped); one that could not start is
/// `None`. Sequentially (no pool) the first `None` ends the batch, exactly like the sequential
/// loop's per-candidate deadline check (the exempt candidates are the batch's head); with a
/// pool, every result lands back in its slot, whatever order the threads finished in. Each
/// worker reads the clock for its own candidate, so once the deadline passes mid-batch, which
/// slots started depends on timing and on the worker count.
#[allow(clippy::too_many_arguments)]
fn evaluate_admitted(
    admitted: &[(usize, String, &'static str, Option<String>)],
    cut: &Cut,
    generation: usize,
    options: &SearchOptions,
    pool: Option<&rayon::ThreadPool>,
    started: Stamp,
    deadline: Duration,
    recorder: Option<&Recorder>,
) -> Vec<Option<Observation>> {
    let clock = options.clock;
    // A stop request is honoured exactly where the seconds bound is: before a fit starts.
    let may_start = |origin: &str| {
        !options.cancelled() && (seconds_exempt(generation, origin) || started.elapsed() < deadline)
    };
    match pool {
        None => {
            let mut results = Vec::with_capacity(admitted.len());
            for (island, genome, origin, draw) in admitted {
                if !may_start(origin) {
                    break;
                }
                results.push(Some(record_evaluation(recorder, draw, clock, || {
                    let mut observation = evaluate(genome, cut, generation, origin, clock);
                    observation.island = (options.islands > 1).then_some(*island);
                    observation
                })));
            }
            results.resize(admitted.len(), None);
            results
        }
        Some(pool) => {
            let evaluated: Vec<Option<(Observation, Option<f64>)>> = pool.install(|| {
                admitted
                    .par_iter()
                    .map(|(island, genome, origin, _)| {
                        may_start(origin).then(|| {
                            let started = recorder.map(|_| clock.start());
                            let mut observation = evaluate(genome, cut, generation, origin, clock);
                            observation.island = (options.islands > 1).then_some(*island);
                            let elapsed = started.map(|stamp| stamp.secs());
                            (observation, elapsed)
                        })
                    })
                    .collect()
            });
            evaluated
                .into_iter()
                .zip(admitted)
                .map(|(result, (_, _, _, draw))| {
                    result.map(|(observation, elapsed)| {
                        let attempt = recorder.and_then(|recorder| {
                            draw.as_ref()
                                .and_then(|draw| recorder.update(|log| log.start(draw, 1, None)))
                        });
                        record_evaluation_result(recorder, attempt, observation, elapsed)
                    })
                })
                .collect()
        }
    }
}

fn run_cut_allocated(
    cut: &Cut,
    options: &SearchOptions,
    pool: Option<&rayon::ThreadPool>,
    output: &Path,
    reserve: bool,
) -> Result<CutSearch> {
    let (budget, rng_seed, clock) = (options.budget, options.rng_seed, options.clock);
    budget.check()?;
    ensure!(
        !output.join("search.json").exists(),
        "{}: a completed search is already published here",
        output.display()
    );
    ensure!(!output.join("search-events.json").exists(), "{}: previous search-event evidence already exists; preserve it and use a new run destination", output.display());
    let recorder = search_recorder(cut, options)?;
    let recorder = recorder.as_ref();
    let started = clock.start();
    let evaluation_limit = exploration_limit(budget, if reserve { 2 } else { 1 });
    let seconds_limit = budget.seconds as f64 / if reserve { 2.0 } else { 1.0 };
    let deadline = Duration::from_secs_f64(seconds_limit);

    let mut observations: Vec<Observation> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let island_count = options.islands.max(1);
    let multi = island_count > 1;
    let mut islands: Vec<Island> = (0..island_count)
        .map(|index| Island {
            rng_seed: island_seed(rng_seed, index),
            survivors: Vec::new(),
        })
        .collect();
    let mut migrations = Vec::new();
    let mut evaluations = 0usize;
    let mut generation = 0usize;
    let mut barren = 0usize;

    let stopped_by: &'static str = 'search: loop {
        // Generation 0 is the bootstrap plus grammar-random draws; later
        // generations are the elites' offspring. Elites carry their scores and
        // are never re-fitted, so a slot here is always an unmeasured genome.
        // With islands, each island draws its own generation from its own stream,
        // in island order.
        let mut candidates: Vec<(usize, String, &'static str, Option<String>)> = Vec::new();
        for (index, island) in islands.iter().enumerate() {
            if generation == 0 {
                let draw = record_draw(
                    recorder,
                    format!("exploration/0/{index}/bootstrap"),
                    BOOTSTRAP,
                    Some(BOOTSTRAP_GENOME),
                    None,
                    &[],
                    "candidate",
                    None,
                );
                candidates.push((index, BOOTSTRAP_GENOME.to_owned(), BOOTSTRAP, draw));
                // Seeds follow the bootstrap and precede every random draw, so a budget that
                // admits generation 0 at all re-measures them first; the random slots are the
                // unseeded search's own draws, unchanged.
                candidates.extend(options.seeds.iter().enumerate().map(|(slot, g)| {
                    let draw = record_draw(
                        recorder,
                        format!("exploration/0/{index}/incumbent/{slot}"),
                        INCUMBENT,
                        Some(g),
                        None,
                        &[],
                        "candidate",
                        None,
                    );
                    (index, g.clone(), INCUMBENT, draw)
                }));
                for slot in 1..POPULATION {
                    let seed = stream(island.rng_seed, cut.index, 0, slot, 0);
                    let genome = evolve::random_canonical_pipeline(seed, MAX_TRANSFORMS);
                    let draw = record_draw(
                        recorder,
                        format!("exploration/0/{index}/{slot}/0"),
                        "random",
                        Some(&genome),
                        Some(seed),
                        &[],
                        "candidate",
                        None,
                    );
                    candidates.push((index, genome, "random", draw));
                }
            } else {
                candidates.extend((island.survivors.len()..POPULATION).map(|slot| {
                    let (genome, origin, draw) = offspring_recorded(
                        &island.survivors,
                        &seen,
                        island.rng_seed,
                        cut.index,
                        generation,
                        slot,
                        index,
                        recorder,
                    );
                    (index, genome, origin, draw)
                }));
            }
        }

        let spent_before = evaluations;
        // Admission, in slot order: exactly the candidates the one-at-a-time loop would fit,
        // with its checks in its order — the evaluation bound, then the seconds bound, then
        // "never fit a genome twice on one cut". The seconds bound is checked again before each
        // fit starts (`evaluate_admitted`).
        let mut admitted: Vec<(usize, String, &'static str, Option<String>)> = Vec::new();
        // Another island's draw of a genome this cut already measured: it joins that island's
        // candidates with its existing score instead of being fitted again.
        let mut shared: Vec<(usize, String)> = Vec::new();
        let mut bound = None;
        let mut candidates = candidates.into_iter();
        for (island, genome, origin, draw) in candidates.by_ref() {
            if evaluations + admitted.len() >= evaluation_limit {
                bound = Some(STOPPED_BY_EVALUATIONS);
                admission(recorder, &draw, STOPPED_BY_EVALUATIONS);
                break;
            }
            if !seconds_exempt(generation, origin) && started.elapsed() >= deadline {
                bound = Some(STOPPED_BY_SECONDS);
                admission(recorder, &draw, STOPPED_BY_SECONDS);
                break;
            }
            if !seen.insert(genome.clone()) {
                admission(recorder, &draw, "duplicate");
                if multi {
                    shared.push((island, genome));
                }
                continue;
            }
            admission(recorder, &draw, "admitted");
            admitted.push((island, genome, origin, draw));
        }
        for (_, _, _, draw) in candidates {
            admission(recorder, &draw, bound.unwrap_or("cancelled"));
        }
        if options.cancelled() {
            if let Some(recorder) = recorder {
                recorder.publish(output)?;
            }
        }
        options.check_cancelled()?;
        let results = evaluate_admitted(
            &admitted, cut, generation, options, pool, started, deadline, recorder,
        );
        // The fold, in slot order: survivors and observations never depend on which worker
        // finished first.
        let mut late = false;
        for ((island, genome, _, _), result) in admitted.into_iter().zip(results) {
            let Some(mut observation) = result else {
                late = true;
                continue;
            };
            evaluations += 1;
            if let Some(rmse) = observation.rmse {
                islands[island].survivors.push(Scored { genome, rmse });
            }
            observation.island = multi.then_some(island);
            if let Some(progress) = &options.progress {
                progress.observe_current(&observation);
            }
            observations.push(observation);
        }
        if let Some(progress) = &options.progress {
            progress.generation_done(generation as u64 + 1);
        }
        // Stopped between evaluations: nothing of this cut is published, and the run that
        // asked for it writes no receipt.
        if options.cancelled() {
            if let Some(recorder) = recorder {
                recorder.publish(output)?;
            }
        }
        options.check_cancelled()?;
        if late {
            break 'search STOPPED_BY_SECONDS;
        }
        if let Some(bound) = bound {
            break 'search bound;
        }
        for (island, genome) in shared {
            let known = observations
                .iter()
                .find(|o| o.genome == genome && o.measured())
                .and_then(|o| o.rmse);
            if let Some(rmse) = known {
                if !islands[island].survivors.iter().any(|s| s.genome == genome) {
                    islands[island].survivors.push(Scored { genome, rmse });
                }
            }
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
        for island in &mut islands {
            island.survivors = next_seeds(&island.survivors, ELITES);
        }
        if multi
            && options.migration_interval > 0
            && (generation + 1).is_multiple_of(options.migration_interval)
        {
            migrate(
                &mut islands,
                cut.index,
                generation,
                &mut migrations,
                recorder,
            );
        }
        generation += 1;
    };

    let seconds = started.elapsed().as_secs_f64();
    if let Some(progress) = &options.progress {
        progress.cut_done(generation as u64 + 1);
    }
    if let Some(recorder) = recorder {
        recorder.publish(output)?;
    }
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
        migrations,
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
    crate::parallel::on_evaluation_stack(|| {
        remeasure_in(
            cuts,
            searches,
            &SearchOptions::new(Budget::default(), 0),
            None,
        )
    })?
}

/// The comparison phase, one worker per cut when a pool is given. Each cut's admissions depend
/// only on that cut's own state (what it already measured, its evaluation count, its active
/// seconds), and the pool is visited in the same order on every cut, so unless the seconds
/// bound ends a cut's comparison, running the cuts concurrently decides exactly what running
/// them one after another does.
fn remeasure_in(
    cuts: &[Cut],
    searches: &mut [CutSearch],
    options: &SearchOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    let Some(genomes) = open_comparison(cuts, searches)? else {
        return Ok(());
    };
    let recorders = comparison_recorders(searches)?;
    let clock = options.clock;
    let stop = || options.cancelled();
    let progress = options.progress.as_deref();
    if let Some(progress) = progress {
        progress.begin_comparison();
    }
    // Every comparison fit, counted on the cut (by position) it was made on.
    let evaluate_counted = |position: usize, g: &str, c: &Cut, n: usize, o: &'static str| {
        let observation = evaluate(g, c, n, o, clock);
        if let Some(progress) = progress {
            progress.observe(position, &observation);
            progress.tick();
        }
        observation
    };
    match pool {
        None => {
            for (position, ((cut, search), recorder)) in cuts
                .iter()
                .zip(searches.iter_mut())
                .zip(&recorders)
                .enumerate()
            {
                remeasure_cut(
                    cut,
                    search,
                    &genomes,
                    clock,
                    &stop,
                    recorder.as_ref(),
                    &mut |g, c, n, o| evaluate_counted(position, g, c, n, o),
                );
            }
        }
        Some(pool) => pool.install(|| {
            cuts.par_iter()
                .zip(searches.par_iter_mut())
                .zip(recorders.par_iter())
                .enumerate()
                .for_each(|(position, ((cut, search), recorder))| {
                    remeasure_cut(
                        cut,
                        search,
                        &genomes,
                        clock,
                        &stop,
                        recorder.as_ref(),
                        &mut |g, c, n, o| evaluate_counted(position, g, c, n, o),
                    )
                })
        }),
    }
    for (search, recorder) in searches.iter().zip(&recorders) {
        if let Some(recorder) = recorder {
            recorder.publish(&search.output)?;
        }
    }
    // Stopped mid-comparison: preserve attempt evidence without publishing a completed cut.
    options.check_cancelled()?;
    close_comparison(searches)
}

#[cfg(test)]
fn remeasure_with(
    cuts: &[Cut],
    searches: &mut [CutSearch],
    mut evaluate_candidate: impl FnMut(&str, &Cut, usize, &'static str) -> Observation,
) -> Result<()> {
    let Some(genomes) = open_comparison(cuts, searches)? else {
        return Ok(());
    };
    let recorders = comparison_recorders(searches)?;
    for ((cut, search), recorder) in cuts.iter().zip(searches.iter_mut()).zip(&recorders) {
        remeasure_cut(
            cut,
            search,
            &genomes,
            Clock::Wall,
            &|| false,
            recorder.as_ref(),
            &mut evaluate_candidate,
        );
        if let Some(recorder) = recorder {
            recorder.publish(&search.output)?;
        }
    }
    close_comparison(searches)
}

/// Check the cut/search pairing, build the bounded comparison pool and open each cut's
/// comparison phase. `None` when there is nothing to compare (a single cut).
fn open_comparison(cuts: &[Cut], searches: &mut [CutSearch]) -> Result<Option<Vec<String>>> {
    ensure!(
        cuts.len() == searches.len(),
        "remeasure: cuts/searches length mismatch"
    );
    for (cut, search) in cuts.iter().zip(searches.iter()) {
        ensure!(cut.index == search.cut, "remeasure: cut identity mismatch");
    }
    if cuts.len() <= 1 {
        return Ok(None);
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
    Ok(Some(pool))
}

fn comparison_recorders(searches: &[CutSearch]) -> Result<Vec<Option<Recorder>>> {
    searches
        .iter()
        .map(|s| {
            let path = s.output.join("search-events.json");
            if !path.exists() {
                return Ok(None);
            } // Explicit legacy/no-logging route.
            let log = SearchEventLog::from_bytes(&std::fs::read(path)?)?;
            log.reconcile()?;
            Ok(Some(Recorder::new(log)))
        })
        .collect()
}

/// One cut's comparison: the pool in order, each genome this cut has not attempted, while its
/// total evaluation and seconds allowances last.
#[allow(clippy::too_many_arguments)]
fn remeasure_cut(
    cut: &Cut,
    search: &mut CutSearch,
    pool: &[String],
    clock: Clock,
    stop: &(dyn Fn() -> bool + Sync),
    recorder: Option<&Recorder>,
    evaluate_candidate: &mut dyn FnMut(&str, &Cut, usize, &'static str) -> Observation,
) {
    for (slot, genome) in pool.iter().enumerate() {
        if stop() {
            return;
        }
        let started = clock.start();
        // Failed attempts count and are never retried on the same cut.
        let already = search.observations.iter().any(|o| &o.genome == genome);
        let elapsed = search.seconds + started.secs();
        let draw = record_draw(
            recorder,
            format!("comparison/{slot}"),
            REMEASURE,
            Some(genome),
            None,
            &[],
            if already { "duplicate" } else { "candidate" },
            None,
        );
        let disposition = if already {
            "duplicate"
        } else if search.completed_evaluations >= search.budget.evaluations as u64 {
            STOPPED_BY_EVALUATIONS
        } else if elapsed >= search.budget.seconds as f64 {
            STOPPED_BY_SECONDS
        } else {
            "admitted"
        };
        admission(recorder, &draw, disposition);
        if !already
            && search.completed_evaluations < search.budget.evaluations as u64
            && elapsed < search.budget.seconds as f64
        {
            let observation = record_evaluation(recorder, &draw, clock, || {
                evaluate_candidate(genome, cut, search.generations as usize, REMEASURE)
            });
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
        let seconds = started.secs();
        search.seconds += seconds;
        search.comparison.seconds += seconds;
    }
}

/// Record each cut's comparison outcome and best, and publish it.
fn close_comparison(searches: &mut [CutSearch]) -> Result<()> {
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
                    None,
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
            evaluate(genome, cut, generation, origin, Clock::Wall)
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
                island: None,
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

    /// Everything three cuts' searches publish, in a fixed order.
    fn published(searches: &[CutSearch]) -> Vec<(usize, Vec<u8>, Vec<u8>)> {
        searches
            .iter()
            .map(|s| {
                (
                    s.cut,
                    std::fs::read(s.output.join("search.json")).unwrap(),
                    std::fs::read(s.output.join("observations.json")).unwrap(),
                )
            })
            .collect()
    }

    /// With the wall clock frozen — the one input two runs can never agree on — the seconds bound
    /// never binds and the evaluation budget ends every phase; then any worker count decides
    /// exactly what one worker decides, and every byte a search publishes is identical.
    #[test]
    fn eight_threads_publish_exactly_what_one_thread_publishes() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "threads",
            7,
            3,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let run = |threads: usize, name: &str| {
            let options = SearchOptions {
                threads,
                clock: Clock::Frozen,
                ..SearchOptions::new(
                    Budget {
                        evaluations: 40,
                        seconds: 600,
                    },
                    11,
                )
            };
            run_cuts_with(&preparation.cuts, &options, &temp.path().join(name)).unwrap()
        };
        let one = run(1, "one");
        let eight = run(8, "eight");
        assert_eq!(published(&one), published(&eight));
        // Past generation 0, and through the comparison phase: both paths were exercised.
        assert!(one.iter().all(|s| {
            s.observations
                .iter()
                .any(|o| o.generation >= 1 && o.measured())
        }));
        assert!(one.iter().all(|s| s.comparison.completed_evaluations > 0));
    }

    fn island_options(islands: usize, threads: usize, rng_seed: u64) -> SearchOptions {
        SearchOptions {
            threads,
            islands,
            migration_interval: 2,
            clock: Clock::Frozen,
            ..SearchOptions::new(
                Budget {
                    evaluations: 90,
                    seconds: 600,
                },
                rng_seed,
            )
        }
    }

    /// Islands replay from the seed, record every migration, tag every observation with its
    /// island, and publish the same bytes on any worker count (frozen clock, so the evaluation
    /// budget ends every phase).
    #[test]
    fn islands_migrate_record_and_replay_on_any_worker_count() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "islands",
            7,
            1,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let run = |options: &SearchOptions, name: &str| {
            run_cuts_with(&preparation.cuts, options, &temp.path().join(name)).unwrap()
        };
        let a = run(&island_options(3, 1, 11), "a");
        let again = run(&island_options(3, 1, 11), "again");
        let wide = run(&island_options(3, 4, 11), "wide");
        assert_eq!(published(&a), published(&again));
        assert_eq!(published(&a), published(&wide));
        let search = &a[0];
        assert!(!search.migrations.is_empty(), "no migration was recorded");
        for migration in &search.migrations {
            assert_ne!(migration.from_island, migration.to_island);
            assert!(migration.from_island < 3 && migration.to_island < 3);
            assert_eq!((migration.after_generation + 1) % 2, 0);
            assert!(search
                .observations
                .iter()
                .any(|o| o.program_sha256 == migration.program_sha256
                    && o.rmse == Some(migration.rmse)));
        }
        let islands: BTreeSet<usize> = search
            .observations
            .iter()
            .map(|o| {
                o.island
                    .expect("every exploration observation names its island")
            })
            .collect();
        assert_eq!(islands, BTreeSet::from([0, 1, 2]));
        // No genome is fitted twice on the cut, whichever island drew it.
        let genomes: BTreeSet<&str> = search
            .observations
            .iter()
            .map(|o| o.genome.as_str())
            .collect();
        assert_eq!(genomes.len(), search.observations.len());
        // Another seed searches other islands.
        let other = run(&island_options(3, 1, 12), "other");
        assert_ne!(published(&a), published(&other));
    }

    /// One island is exactly the search without islands: same stream, same bytes.
    #[test]
    fn one_island_is_the_plain_search() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "one-island",
            7,
            2,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let plain = SearchOptions {
            clock: Clock::Frozen,
            ..SearchOptions::new(
                Budget {
                    evaluations: 30,
                    seconds: 600,
                },
                3,
            )
        };
        let one = SearchOptions {
            islands: 1,
            migration_interval: 1,
            ..plain.clone()
        };
        let a = run_cuts_with(&preparation.cuts, &plain, &temp.path().join("plain")).unwrap();
        let b = run_cuts_with(&preparation.cuts, &one, &temp.path().join("one")).unwrap();
        assert_eq!(published(&a), published(&b));
        assert!(a
            .iter()
            .all(|s| s.migrations.is_empty() && s.observations.iter().all(|o| o.island.is_none())));
    }

    #[test]
    fn a_migrant_enters_only_an_island_it_improves() {
        let champion = |genome: &str, rmse: f64| Champion {
            genome: genome.to_owned(),
            rmse,
        };
        let mut islands = vec![
            Island {
                rng_seed: 1,
                survivors: vec![champion("a", 1.0), champion("b", 2.0)],
            },
            Island {
                rng_seed: 2,
                survivors: vec![
                    champion("c", 0.5),
                    champion("d", 0.6),
                    champion("e", 0.7),
                    champion("f", 0.8),
                ],
            },
        ];
        let mut events = Vec::new();
        migrate(&mut islands, 0, 3, &mut events, None);
        assert_eq!(events.len(), 2);
        // `c` (0.5) enters island 0; `a` (1.0) is worse than all of island 1's four elites.
        let into_zero = events.iter().find(|e| e.to_island == 0).unwrap();
        assert!(into_zero.admitted && into_zero.from_island == 1 && into_zero.rmse == 0.5);
        let into_one = events.iter().find(|e| e.to_island == 1).unwrap();
        assert!(!into_one.admitted && into_one.from_island == 0);
        assert_eq!(islands[0].survivors[0].genome, "c");
        assert_eq!(
            islands[1]
                .survivors
                .iter()
                .map(|c| c.genome.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "d", "e", "f"]
        );
        assert!(events.iter().all(|e| e.after_generation == 3 && e.cut == 0));
    }

    /// With the real clock, and a seconds budget far longer than 30 evaluations on these rows
    /// take, the evaluation budget ends every phase: only the recorded seconds differ and every
    /// decision matches. The test checks it ran in that regime. A binding seconds bound is the
    /// timing-dependent case this does not cover.
    #[test]
    fn thread_count_changes_no_decision_when_the_evaluation_budget_binds() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "wall",
            7,
            2,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let decided = |threads: usize, name: &str| {
            let options = SearchOptions {
                threads,
                ..SearchOptions::new(
                    Budget {
                        evaluations: 30,
                        seconds: 600,
                    },
                    5,
                )
            };
            let searches =
                run_cuts_with(&preparation.cuts, &options, &temp.path().join(name)).unwrap();
            assert!(
                searches.iter().all(|s| s.stopped_by != STOPPED_BY_SECONDS
                    && s.exploration.stopped_by != STOPPED_BY_SECONDS
                    && s.comparison.stopped_by != STOPPED_BY_SECONDS),
                "{threads} thread(s): the seconds bound ended a phase"
            );
            searches
                .iter()
                .flat_map(|s| {
                    s.observations
                        .iter()
                        .map(|o| (o.cut, o.genome.clone(), o.generation, o.origin, o.rmse))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(decided(1, "one"), decided(4, "four"));
    }

    /// Generation 0's bootstrap and seeds are measured on every cut even when the seconds bound
    /// has already passed (a slow seed fit on a loaded host), on any worker count, so the
    /// receipt's seeding record and the evidence verification always agree; the random draws are
    /// what the bound stops. A stop request still stops everything.
    #[test]
    fn seeds_are_remeasured_on_every_cut_even_past_the_seconds_bound() {
        let temp = tempfile::tempdir().unwrap();
        let preparation = prepare::build(
            &linear_rows(150),
            "expired",
            7,
            3,
            &temp.path().join("prepared"),
        )
        .unwrap();
        let seeds: Vec<String> = ["(pipe X (standardize) (ridge))", "(pipe X (rank) (ridge))"]
            .iter()
            .map(|g| crate::seeds::canonical_pipeline(g).unwrap())
            .collect();
        let options = |threads: usize| SearchOptions {
            seeds: seeds.clone(),
            threads,
            clock: Clock::Expired,
            ..SearchOptions::new(
                Budget {
                    evaluations: 40,
                    seconds: 60,
                },
                11,
            )
        };
        for threads in [1, 3] {
            let searches = run_cuts_with(
                &preparation.cuts,
                &options(threads),
                &temp.path().join(format!("threads-{threads}")),
            )
            .unwrap();
            assert_eq!(searches.len(), 3);
            for search in &searches {
                let measured: Vec<(&str, usize, &str)> = search
                    .observations
                    .iter()
                    .map(|o| (o.genome.as_str(), o.generation, o.origin))
                    .collect();
                assert_eq!(
                    measured,
                    vec![
                        (BOOTSTRAP_GENOME, 0, BOOTSTRAP),
                        (seeds[0].as_str(), 0, INCUMBENT),
                        (seeds[1].as_str(), 0, INCUMBENT),
                    ],
                    "cut {} at {threads} thread(s)",
                    search.cut
                );
                assert_eq!(search.exploration.stopped_by, STOPPED_BY_SECONDS);
                assert_eq!(search.comparison.completed_evaluations, 0);
            }
        }
        let stopped = SearchOptions {
            cancel: Some(Arc::new(AtomicBool::new(true))),
            ..options(1)
        };
        let error =
            run_cuts_with(&preparation.cuts, &stopped, &temp.path().join("stopped")).unwrap_err();
        assert!(crate::parallel::is_cancelled(&error), "{error:#}");
        let log = SearchEventLog::from_bytes(
            &std::fs::read(temp.path().join("stopped/cut-0/search-events.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(log.reconcile().unwrap().evaluation_attempts, 0);
        assert!(log.require_complete().is_err());
        assert!(!temp.path().join("stopped/cut-0/search.json").exists());
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
        let log = SearchEventLog::from_bytes(
            &std::fs::read(temp.path().join("search/search-events.json")).unwrap(),
        )
        .unwrap();
        let counts = log.require_complete().unwrap();
        assert!(counts.failed_attempts > 0);
        assert_eq!(counts.failed_attempts, counts.evaluation_attempts);
        assert_eq!(counts.logical_completed, 0);
        assert!(!temp.path().join("search/search.json").exists());
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
    #[test]
    fn event_logging_preserves_frozen_choices_observations_and_migration() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "event-parity");
        let mut options = SearchOptions {
            clock: Clock::Frozen,
            islands: 2,
            threads: 2,
            migration_interval: 1,
            ..SearchOptions::new(
                Budget {
                    evaluations: 28,
                    seconds: 600,
                },
                23,
            )
        };
        let logged = run_cuts_with(
            std::slice::from_ref(&cut),
            &options,
            &temp.path().join("logged"),
        )
        .unwrap();
        options.record_events = false;
        let control = run_cuts_with(
            std::slice::from_ref(&cut),
            &options,
            &temp.path().join("control"),
        )
        .unwrap();
        assert_eq!(logged[0].observations, control[0].observations);
        assert_eq!(logged[0].best_program, control[0].best_program);
        assert_eq!(logged[0].migrations, control[0].migrations);
        assert!(!logged[0].migrations.is_empty());
        for name in ["observations.json", "search.json"] {
            assert_eq!(
                std::fs::read(logged[0].output.join(name)).unwrap(),
                std::fs::read(control[0].output.join(name)).unwrap()
            );
        }
        let log = SearchEventLog::from_bytes(
            &std::fs::read(logged[0].output.join("search-events.json")).unwrap(),
        )
        .unwrap();
        let counts = log.require_complete().unwrap();
        assert_eq!(counts.evaluation_attempts, logged[0].observations.len());
        assert_eq!(
            counts.logical_completed,
            logged[0].successful_measurements as usize
        );
        assert!(counts.draws > counts.evaluation_attempts);
        assert!(counts.migration_decisions > 0);
        assert!(log.events.iter().any(|e| matches!(&e.event, SearchEvent::Edit { transition } if transition.operator == "crossover" && transition.parents.len() == 2)));
        assert!(!control[0].output.join("search-events.json").exists());
        let observed: BTreeSet<_> = log
            .events
            .iter()
            .filter_map(|e| match &e.event {
                SearchEvent::Finished {
                    observation_sha256, ..
                } => observation_sha256.clone(),
                _ => None,
            })
            .collect();
        for observation in &logged[0].observations {
            assert!(observed.contains(&sha256_hex(&serde_json::to_vec(observation).unwrap())));
        }
    }

    #[test]
    fn event_logging_preserves_retry_streams_duplicates_and_random_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "fallback-parity");
        let options = SearchOptions::new(
            Budget {
                evaluations: 1,
                seconds: 1,
            },
            7,
        );
        let recorder = search_recorder(&cut, &options).unwrap().unwrap();
        let parents = vec![Scored {
            genome: BOOTSTRAP_GENOME.into(),
            rmse: 1.,
        }];
        let mut seen = BTreeSet::new();
        for attempt in 0..DISTINCT_CHILD_ATTEMPTS {
            let seed = stream(7, cut.index, 1, 1, attempt as u64);
            if let Ok(child) =
                evolve::mutate_canonical_pipeline(BOOTSTRAP_GENOME, seed, MAX_TRANSFORMS)
            {
                seen.insert(child);
            }
        }
        let expected = offspring(&parents, &seen, 7, cut.index, 1, 1);
        let (genome, origin, _) =
            offspring_recorded(&parents, &seen, 7, cut.index, 1, 1, 0, Some(&recorder));
        assert_eq!((genome.clone(), origin), expected);
        assert_eq!(
            genome,
            evolve::random_canonical_pipeline(stream(7, cut.index, 1, 1, 0xFFFF), MAX_TRANSFORMS)
        );
        recorder.publish(temp.path()).unwrap();
        let log = SearchEventLog::from_bytes(
            &std::fs::read(temp.path().join("search-events.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(log.reconcile().unwrap().draws, DISTINCT_CHILD_ATTEMPTS + 1);
        assert!(log.events.iter().any(|e| matches!(&e.event, SearchEvent::Draw { decision } if decision.source == "random_fallback")));
        let invalid = vec![Scored {
            genome: "invalid-parent".into(),
            rmse: 1.,
        }];
        let recorder = search_recorder(&cut, &options).unwrap().unwrap();
        let expected = offspring(&invalid, &seen, 7, cut.index, 2, 1);
        let (genome, origin, _) =
            offspring_recorded(&invalid, &seen, 7, cut.index, 2, 1, 0, Some(&recorder));
        assert_eq!((genome, origin), expected);
        recorder.publish(&temp.path().join("invalid")).unwrap();
        let log = SearchEventLog::from_bytes(
            &std::fs::read(temp.path().join("invalid/search-events.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(log.reconcile().unwrap().draws, 9);
        assert_eq!(log.events.iter().filter(|e| matches!(&e.event, SearchEvent::Draw { decision } if decision.outcome == "operator_failed")).count(), 8);
    }

    #[test]
    fn comparison_event_joins_preserve_frozen_numerical_results() {
        let temp = tempfile::tempdir().unwrap();
        let cuts = prepare::build(
            &linear_rows(50),
            "comparison-events",
            7,
            2,
            &temp.path().join("prepared"),
        )
        .unwrap()
        .cuts;
        let mut options = SearchOptions {
            clock: Clock::Frozen,
            ..SearchOptions::new(
                Budget {
                    evaluations: 8,
                    seconds: 600,
                },
                9,
            )
        };
        let logged = run_cuts_with(&cuts, &options, &temp.path().join("logged")).unwrap();
        options.record_events = false;
        let control = run_cuts_with(&cuts, &options, &temp.path().join("control")).unwrap();
        let mut comparisons = 0;
        for (logged, control) in logged.iter().zip(&control) {
            assert_eq!(logged.observations, control.observations);
            let log = SearchEventLog::from_bytes(
                &std::fs::read(logged.output.join("search-events.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                log.require_complete().unwrap().evaluation_attempts,
                logged.observations.len()
            );
            comparisons += log.events.iter().filter(|r| matches!(&r.event, SearchEvent::Draw { decision } if decision.source == REMEASURE)).count();
        }
        assert!(comparisons > 0);
    }
    #[test]
    fn event_persistence_failure_refuses_success_publication() {
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "event-write-failure");
        let blocked = temp.path().join("blocked");
        std::fs::write(&blocked, b"existing bytes").unwrap();
        let result = run_cut(
            &cut,
            Budget {
                evaluations: 1,
                seconds: 60,
            },
            7,
            &blocked,
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(&blocked).unwrap(), b"existing bytes");
    }

    #[test]
    fn progress_search_refuses_previous_source_binding_before_publication() {
        // Literal pre-progress source digest from the four include_bytes inputs at91a94c3.
        // This is a compatibility refusal check, not an evaluator/lineage fixture rebaseline.
        const PRE_PROGRESS_SOURCE: &str =
            "8becb8f1e3119d06736adb1957769626cb108e87b27913bbf36cdffe92e0be0f";
        let temp = tempfile::tempdir().unwrap();
        let cut = one_cut(&temp.path().join("prepared"), "progress-source-binding");
        let current = search_events::source_sha256();
        assert_ne!(
            current, PRE_PROGRESS_SOURCE,
            "progress source identity did not move"
        );
        let mut binding = SearchBinding {
            run_id: sha256_hex(b"progress-source-binding"),
            task: "regression".into(),
            experiment: EvidenceRef::unknown("test supplies no experiment authority"),
            role_exposure: EvidenceRef::unknown("test supplies no authenticated role exposure"),
            context_sha256: search_events::cut_context_sha256(&cut).unwrap(),
            source_sha256: PRE_PROGRESS_SOURCE.into(),
            evaluator: EvidenceRef::unknown("test supplies no evaluator authority"),
            profile: EvidenceRef::unknown("test supplies no registered profile"),
            legal_set: EvidenceRef::unknown("test supplies no legal-set authority"),
            fidelity: "full-cut-fit-and-heldout-score".into(),
        };
        let mut options = SearchOptions {
            clock: Clock::Frozen,
            event_binding: Some(binding.clone()),
            ..SearchOptions::new(
                Budget {
                    evaluations: 1,
                    seconds: 60,
                },
                7,
            )
        };
        let stale = temp.path().join("stale");
        let error = run_cuts_with(std::slice::from_ref(&cut), &options, &stale).unwrap_err();
        assert!(
            format!("{error:#}").contains("search-event source differs from actual implementation"),
            "{error:#}"
        );
        assert!(
            !stale.exists(),
            "stale binding must be refused before publication or fitting"
        );

        // Change only the binding's source identity: the same actual run path now measures
        // and publishes current-source evidence, with unknown authority still unknown.
        binding.source_sha256 = current.clone();
        options.event_binding = Some(binding);
        let searches = run_cuts_with(
            std::slice::from_ref(&cut),
            &options,
            &temp.path().join("current"),
        )
        .unwrap();
        assert_eq!(searches.len(), 1);
        assert_eq!(searches[0].completed_evaluations, 1);
        assert_eq!(searches[0].successful_measurements, 1);
        let path = searches[0].output.join("search-events.json");
        let log = SearchEventLog::from_bytes(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(log.binding.source_sha256, current);
        assert_eq!(log.require_complete().unwrap().evaluation_attempts, 1);
    }
}
