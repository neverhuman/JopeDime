//! `jope pipeline evolve`: rounds of seeded search on one dataset's fixed cuts, and the
//! `jope.pipeline-lineage/v1` record that chains them.
//!
//! Round 1 runs from `--seeds` (a `jope.pipeline-seeds/v1` file, typically the champion bank's
//! export) or unseeded. Every later round is seeded from the previous round's champion plus its
//! ranked runners-up ([`crate::islands::next_seeds`] over the programs measured on every cut),
//! with the previous receipt as its verified `--incumbent`. The fold seed and cut count never
//! change, so every round measures on the identical prepared cuts and its champion's mean
//! held-out RMSE is directly comparable with the round before; the previous champion is
//! re-measured there, so a round can only match or beat it *on those cuts*. Every round selects
//! on the same held-out folds, so part of any round-over-round gain is selection on reused folds;
//! only the sealed test split (`jope pipeline score`) speaks to generalisation.
//!
//! The optional control is one more run at the final round's configuration (one round's budget,
//! the final round's random stream) with no seeds: what one round of search finds without
//! inherited seeds. It is not compute-matched against the lineage, whose final round inherits
//! every earlier round's search on the same cuts, and one control on one stream cannot separate
//! seeding from that extra compute or from noise; [`verify`] reports both evaluation totals so
//! nobody has to take the comparison on trust.
//!
//! Every refusal a round could make — parallelism, budget, a cut count outside the fold
//! partition, a seed that is not a pipeline, a dataset that does not resolve, is not a regression
//! task or cannot be partitioned into those cuts, a budget too small to re-measure the seeds — is
//! made before the output directory is created, so a refused command can be corrected and re-run
//! into the same `--output`. That `--output` must be new or an empty directory: anything already
//! in it (a stale round, control or lineage, or an unrelated file) is refused before anything is
//! written, rather than when the round that would write over it is reached.
//!
//! `lineage.json` is rewritten atomically and fsynced after every completed round, so a parent
//! process can count completed rounds while evolve runs; a round interrupted by SIGTERM writes no
//! receipt and never enters it. Evolve never touches the learning store: HWM ingest and the
//! champion bank are separate `jope-learning` steps over its outputs.
//! [`verify`] re-derives the whole chain from the round directories.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

use crate::{
    evidence::{self, VerifiedRun},
    islands::{self, Champion},
    receipt::Receipt,
    run::{self, RunOptions},
    search::{self, Observation, REMEASURE},
    seeds, sha256_hex, TOOL_VERSION,
};

pub const LINEAGE_SCHEMA_ID: &str = "jope.pipeline-lineage/v1";
pub const LINEAGE_FILE: &str = "lineage.json";
/// Upper bound on `--rounds`: each round is a full run with its own directory.
pub const MAX_ROUNDS: usize = 100;

pub const SEEDED: &str = "seeded";
pub const UNSEEDED: &str = "unseeded";
pub const CONTROL: &str = "control";

pub const FROM_NONE: &str = "none";
pub const FROM_SEEDS_FILE: &str = "seeds_file";
pub const FROM_PREVIOUS_ROUND: &str = "previous_round";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageDataset {
    pub id: String,
    pub dataset_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundSeed {
    pub genome: String,
    pub program_sha256: String,
}

/// A cut's best held-out RMSE after each exploration generation (`None` until one measured).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutProgress {
    pub cut: u64,
    pub best_so_far: Vec<Option<f64>>,
    pub stopped_by: String,
    pub exploration_stopped_by: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Round {
    /// 1-based; the control carries `rounds + 1`.
    pub round: u64,
    /// `seeded`, `unseeded` or `control`.
    pub kind: String,
    /// Relative to the lineage directory.
    pub directory: String,
    pub rng_seed: u64,
    pub receipt_sha256: String,
    pub parent_receipt_sha256: Option<String>,
    /// `none`, `seeds_file` (round 1 from `--seeds`) or `previous_round`.
    pub seed_origin: String,
    pub seeds_file_sha256: Option<String>,
    pub seeds: Vec<RoundSeed>,
    pub selected_genome: String,
    pub program_sha256: String,
    /// Mean held-out RMSE / R^2 of the champion over the lineage's identical cuts.
    pub validation_rmse: f64,
    pub validation_r2: f64,
    pub train_r2: f64,
    pub cuts_measured: u64,
    pub completed_evaluations: u64,
    /// Successful measurements of candidates bred in generation 1 or later (not seeds, not
    /// generation-0 draws, not comparison re-measurements). It shows the search ran past its
    /// starting population, not that anything it bred was better: see
    /// [`Round::champion_found_in_round`].
    pub generation_ge1_measured: u64,
    pub cuts: Vec<CutProgress>,
    /// Sum of the receipt's stage seconds.
    pub seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lineage {
    pub schema: String,
    pub tool_version: String,
    pub dataset: LineageDataset,
    /// The prepared cuts every round and the control measured on.
    pub prepared_sha256: String,
    pub fold_seed: u64,
    pub cuts: Vec<u64>,
    pub evaluations: u64,
    pub seconds: u64,
    /// The base `--rng-seed`; round `r` uses [`round_rng`] of it.
    pub rng_seed: u64,
    pub islands: u64,
    pub migration_interval: u64,
    pub rounds_requested: u64,
    pub control_requested: bool,
    /// sha256 of the `--seeds` file round 1 started from.
    pub seeds_file_sha256: Option<String>,
    /// Completed rounds only, in order.
    pub rounds: Vec<Round>,
    pub control: Option<Round>,
}

/// Round `round`'s random-stream seed: round 1 is the base seed itself.
pub fn round_rng(base: u64, round: u64) -> u64 {
    if round <= 1 {
        return base;
    }
    let mut z = base
        .wrapping_add(0x7E57_A11E_0000_0000)
        .wrapping_add(round.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn round_directory(round: u64) -> String {
    format!("round-{round:03}")
}

pub const CONTROL_DIRECTORY: &str = "control";

impl Round {
    /// The round's champion is neither the bootstrap nor one of the seeds it started from: this
    /// round's own search produced it. `false` for a seeded round usually means the previous
    /// champion was re-measured and kept.
    pub fn champion_found_in_round(&self) -> bool {
        self.selected_genome != search::BOOTSTRAP_GENOME
            && !self
                .seeds
                .iter()
                .any(|s| s.program_sha256 == self.program_sha256)
    }

    /// Where the champion came from, for people: `found in this round`, `carried from its
    /// seeds` or `the bootstrap`.
    pub fn champion_origin(&self) -> &'static str {
        if self.champion_found_in_round() {
            "found in this round"
        } else if self.selected_genome == search::BOOTSTRAP_GENOME {
            "the bootstrap"
        } else {
            "carried from its seeds"
        }
    }
}

/// The next round's seeds: the champion and its ranked runners-up, at most `ELITES`.
pub fn seeds_after(run: &VerifiedRun) -> Result<Vec<Champion>> {
    let ranking = run.ranking()?;
    let champions: Vec<Champion> = ranking
        .into_iter()
        .map(|best| Champion::by_rmse(best.genome, best.validation_rmse))
        .collect();
    Ok(islands::next_seeds(&champions, search::ELITES))
}

fn progress(cut: usize, observations: &[Observation], receipt: &Receipt) -> Result<CutProgress> {
    let recorded = receipt
        .search
        .cuts
        .iter()
        .find(|c| c.cut == cut as u64)
        .context("cut missing from the receipt")?;
    let explored: Vec<&Observation> = observations
        .iter()
        .filter(|o| o.origin != REMEASURE)
        .collect();
    let generations = explored.iter().map(|o| o.generation + 1).max().unwrap_or(0);
    let mut best: Option<f64> = None;
    let mut best_so_far = Vec::with_capacity(generations);
    for generation in 0..generations {
        for o in explored.iter().filter(|o| o.generation == generation) {
            if let Some(rmse) = o.rmse {
                best = Some(best.map_or(rmse, |b: f64| b.min(rmse)));
            }
        }
        best_so_far.push(best);
    }
    Ok(CutProgress {
        cut: cut as u64,
        best_so_far,
        stopped_by: recorded.stopped_by.clone(),
        exploration_stopped_by: recorded.accounting.as_ref().map_or_else(
            || recorded.stopped_by.clone(),
            |a| a.exploration.stopped_by.clone(),
        ),
    })
}

/// A lineage round, derived entirely from its verified run (and the seeds it recorded).
pub fn round_record(
    round: u64,
    kind: &str,
    directory: &str,
    seed_origin: &str,
    run: &VerifiedRun,
) -> Result<Round> {
    let receipt = &run.receipt;
    let best = receipt
        .search
        .best
        .as_ref()
        .context("the round's receipt names no measured champion")?;
    let seeding = receipt
        .search
        .seeding
        .as_ref()
        .context("the round's receipt predates seeding (not v3)")?;
    let mut cuts = Vec::new();
    let mut generation_ge1_measured = 0u64;
    for (cut, observations) in &run.cuts {
        generation_ge1_measured += observations
            .iter()
            .filter(|o| o.measured() && o.generation >= 1 && o.origin != REMEASURE)
            .count() as u64;
        cuts.push(progress(*cut, observations, receipt)?);
    }
    Ok(Round {
        round,
        kind: kind.to_owned(),
        directory: directory.to_owned(),
        rng_seed: receipt.search.rng_seed,
        receipt_sha256: run.receipt_sha256.clone(),
        parent_receipt_sha256: seeding.parent_receipt_sha256.clone(),
        seed_origin: seed_origin.to_owned(),
        seeds_file_sha256: seeding.seeds_file_sha256.clone(),
        seeds: seeding
            .seeds
            .iter()
            .map(|s| RoundSeed {
                genome: s.genome.clone(),
                program_sha256: s.program_sha256.clone(),
            })
            .collect(),
        selected_genome: receipt.search.selected_genome.clone(),
        program_sha256: receipt.search.selected_program_sha256.clone(),
        validation_rmse: best.validation_rmse,
        validation_r2: best.validation_r2,
        train_r2: best.train_r2,
        cuts_measured: best.cuts_measured,
        completed_evaluations: receipt
            .search
            .cuts
            .iter()
            .map(|c| c.completed_evaluations)
            .sum(),
        generation_ge1_measured,
        cuts,
        seconds: receipt.stages.iter().map(|s| s.seconds).sum(),
    })
}

/// Write `lineage.json` through a sibling temporary file: fsync the file, rename, fsync the
/// directory. A reader sees the previous complete lineage or this one, never a torn write.
pub fn write_lineage(directory: &Path, lineage: &Lineage) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(lineage)?;
    bytes.push(b'\n');
    let path = directory.join(LINEAGE_FILE);
    let temporary = directory.join(format!(".{LINEAGE_FILE}.tmp-{}", std::process::id()));
    {
        let mut file = std::fs::File::create(&temporary)
            .with_context(|| format!("create {}", temporary.display()))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, &path).with_context(|| format!("publish {}", path.display()))?;
    std::fs::File::open(directory)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("fsync {}", directory.display()))?;
    Ok(())
}

pub fn read_lineage(path: &Path) -> Result<Lineage> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let lineage: Lineage = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {} as {LINEAGE_SCHEMA_ID}", path.display()))?;
    ensure!(
        lineage.schema == LINEAGE_SCHEMA_ID,
        "{}: schema is {:?}, expected {LINEAGE_SCHEMA_ID:?}",
        path.display(),
        lineage.schema
    );
    Ok(lineage)
}

#[derive(Clone, Debug)]
pub struct EvolveOptions {
    /// The per-round run: dataset, budget, fold seed, base rng seed, cuts, emit, policy,
    /// parallelism, source commit. Its `output`, `seed_genomes` and `incumbent` are set per
    /// round and must be left empty here.
    pub run: RunOptions,
    pub output: PathBuf,
    pub rounds: usize,
    pub control: bool,
    /// Round 1's `jope.pipeline-seeds/v1` file (copied into the round directory first).
    pub seeds: Option<PathBuf>,
}

/// `--output` must not exist yet or be an empty directory. Anything already in it is refused
/// before a byte is written: a stale `control/` or `round-002/` would otherwise be refused only
/// when that run is reached, after the earlier rounds had written `lineage.json` (and the round
/// its `seeds.json`), and the corrected retry would then be refused as "already holds a
/// lineage"; an entry no round writes would sit in the lineage directory as if it belonged there.
fn check_output(output: &Path) -> Result<()> {
    let entries = match std::fs::read_dir(output) {
        Ok(entries) => entries,
        // Not there at all (a dangling symbolic link is there, and is refused below).
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                && std::fs::symlink_metadata(output).is_err() =>
        {
            return Ok(())
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "{}: not a directory to write a lineage into; choose a new --output",
                    output.display()
                )
            })
        }
    };
    let mut names = entries
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("read {}", output.display()))?;
    if names.is_empty() {
        return Ok(());
    }
    ensure!(
        !names
            .iter()
            .any(|name| name == LINEAGE_FILE || *name == round_directory(1)),
        "{}: already holds a lineage; choose a new --output",
        output.display()
    );
    names.sort();
    let shown = names.len().min(3);
    let more = match names.len() - shown {
        0 => String::new(),
        n => format!(" and {n} more"),
    };
    bail!(
        "{}: is not empty ({}{more}); evolve writes only into a new or empty directory, so \
         choose a new --output or empty this one",
        output.display(),
        names[..shown].join(", ")
    )
}

/// Run the lineage. Returns it as written after the last completed round.
pub fn evolve(options: &EvolveOptions) -> Result<Lineage> {
    ensure!(options.run.landmarks.landmark_events.is_none() && options.run.landmarks.landmark_receipt.is_none(),
        "evolve uses per-round encoding output directories; explicit landmark output files would be reused");
    ensure!(
        (1..=MAX_ROUNDS).contains(&options.rounds),
        "--rounds must be 1..={MAX_ROUNDS}, got {}",
        options.rounds
    );
    ensure!(
        options.run.seed_genomes.is_none() && options.run.incumbent.is_none(),
        "evolve seeds its own rounds; pass round 1's seeds with --seeds"
    );
    // A lineage carries its champions by RMSE and verifies that RMSE never regresses between
    // rounds, so an objective-ranked round could not be carried or compared.
    ensure!(
        options.run.objective.is_none(),
        "a lineage ranks and verifies its rounds by held-out RMSE; run an objective-ranked \
         search with jope pipeline run --fitness"
    );
    let output = &options.output;
    check_output(output)?;
    // Everything that can be refused is refused before the first byte is written: a refusal
    // inside round 1 would otherwise leave `round-001/` behind, and the guard above would then
    // turn the corrected retry into "already holds a lineage".
    if let Some(path) = &options.seeds {
        seeds::read(path).with_context(|| format!("--seeds {}", path.display()))?;
    }
    // Round 1 exactly as it will run: parallelism, budget, the cut count, its seeds, the dataset
    // and its partition into those cuts.
    run::check(&RunOptions {
        seed_genomes: options.seeds.clone(),
        ..options.run.clone()
    })?;
    // Every later round carries up to ELITES seeds, which generation 0 re-measures after the
    // bootstrap on each cut; a budget that cannot would refuse round 2 after round 1 ran.
    let limit = search::exploration_limit(options.run.budget, options.run.cuts);
    ensure!(
        options.rounds == 1 || search::ELITES < limit,
        "--rounds {}: every round after the first re-measures up to {} seeds plus the bootstrap \
         in generation 0 on each cut, and this budget's exploration allows {limit} evaluation(s) \
         per cut; raise --budget evaluations",
        options.rounds,
        search::ELITES
    );
    std::fs::create_dir_all(output).with_context(|| format!("create {}", output.display()))?;

    let mut lineage: Option<Lineage> = None;
    let mut previous: Option<(VerifiedRun, PathBuf)> = None;
    for round in 1..=options.rounds as u64 {
        run::check_cancelled(&options.run)?;
        let directory = round_directory(round);
        let root = output.join(&directory);
        std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
        let mut round_options = RunOptions {
            output: root.clone(),
            rng_seed: round_rng(options.run.rng_seed, round),
            ..options.run.clone()
        };
        let (kind, origin) = match (&previous, &options.seeds) {
            (Some((parent, parent_receipt)), _) => {
                let next = seeds_after(parent)?;
                let file = seeds::SeedsFile::new(
                    Some(parent.receipt.dataset.dataset_hash.clone()),
                    next.iter()
                        .map(|c| seeds::SeedGenome {
                            genome: c.genome.clone(),
                            program_sha256: sha256_hex(c.genome.as_bytes()),
                            bank_entry_sha256: None,
                            receipt_sha256: Some(parent.receipt_sha256.clone()),
                        })
                        .collect(),
                );
                let path = root.join("seeds.json");
                seeds::write(&path, &file)?;
                round_options.seed_genomes = Some(path);
                round_options.incumbent = Some(parent_receipt.clone());
                (SEEDED, FROM_PREVIOUS_ROUND)
            }
            (None, Some(given)) => {
                let path = root.join("seeds.json");
                std::fs::copy(given, &path)
                    .with_context(|| format!("copy {} into {}", given.display(), root.display()))?;
                round_options.seed_genomes = Some(path);
                (SEEDED, FROM_SEEDS_FILE)
            }
            (None, None) => (UNSEEDED, FROM_NONE),
        };
        run::run(&round_options)?;
        let receipt_path = root.join("receipt.json");
        let verified = evidence::verify_champion_run(&receipt_path)?;
        let record = round_record(round, kind, &directory, origin, &verified)?;
        let lineage = lineage.get_or_insert_with(|| Lineage {
            schema: LINEAGE_SCHEMA_ID.to_owned(),
            tool_version: TOOL_VERSION.to_owned(),
            dataset: LineageDataset {
                id: verified.receipt.dataset.id.clone(),
                dataset_hash: verified.receipt.dataset.dataset_hash.clone(),
            },
            prepared_sha256: verified.receipt.prepared.sha256.clone(),
            fold_seed: verified.receipt.prepared.seed,
            cuts: verified.receipt.prepared.cuts.clone(),
            evaluations: verified.receipt.search.evaluations,
            seconds: verified.receipt.search.seconds,
            rng_seed: options.run.rng_seed,
            islands: options.run.islands as u64,
            migration_interval: options.run.migration_interval as u64,
            rounds_requested: options.rounds as u64,
            control_requested: options.control,
            seeds_file_sha256: verified
                .receipt
                .search
                .seeding
                .as_ref()
                .and_then(|s| s.seeds_file_sha256.clone())
                .filter(|_| options.seeds.is_some()),
            rounds: Vec::new(),
            control: None,
        });
        ensure!(
            verified.receipt.prepared.sha256 == lineage.prepared_sha256,
            "round {round} measured on different cuts ({}) than round 1 ({})",
            verified.receipt.prepared.sha256,
            lineage.prepared_sha256
        );
        lineage.rounds.push(record);
        write_lineage(output, lineage)?;
        previous = Some((verified, receipt_path));
    }
    let mut lineage = lineage.context("no round completed")?;
    if options.control {
        run::check_cancelled(&options.run)?;
        let root = output.join(CONTROL_DIRECTORY);
        std::fs::create_dir_all(&root)?;
        let last = lineage
            .rounds
            .last()
            .context("no round to control")?
            .rng_seed;
        run::run(&RunOptions {
            output: root.clone(),
            rng_seed: last,
            ..options.run.clone()
        })?;
        let verified = evidence::verify_champion_run(&root.join("receipt.json"))?;
        ensure!(
            verified.receipt.prepared.sha256 == lineage.prepared_sha256,
            "the control measured on different cuts than the lineage"
        );
        lineage.control = Some(round_record(
            options.rounds as u64 + 1,
            CONTROL,
            CONTROL_DIRECTORY,
            FROM_NONE,
            &verified,
        )?);
        write_lineage(output, &lineage)?;
    }
    Ok(lineage)
}

/// What [`verify`] established.
#[derive(Clone, Debug, Serialize)]
pub struct VerifyReport {
    pub schema: &'static str,
    pub ok: bool,
    pub lineage: String,
    pub dataset_hash: String,
    pub prepared_sha256: String,
    pub rounds: u64,
    /// Round champions' mean held-out RMSE on the identical cuts, in round order.
    pub validation_rmse: Vec<f64>,
    pub control_validation_rmse: Option<f64>,
    /// Every round measured at least one candidate bred in generation 1 or later: the search ran
    /// past its starting population. That is not a discovery; the next two fields say what the
    /// rounds actually found.
    pub bred_candidates_in_every_round: bool,
    pub generation_ge1_measured: Vec<u64>,
    /// Per round, [`Round::champion_found_in_round`]: the champion is neither the bootstrap nor
    /// one of the round's seeds.
    pub champion_found_in_round: Vec<bool>,
    /// Per round after the first: its champion's mean held-out RMSE is strictly lower than the
    /// round before's on the same cuts.
    pub improved_on_previous_round: Vec<bool>,
    /// Completed evaluations of every round, summed over their cuts: the search the final
    /// round's champion inherits.
    pub rounds_completed_evaluations: u64,
    /// The control's completed evaluations: one round's budget. The control is not
    /// compute-matched against the lineage; compare the two totals before reading a difference
    /// in RMSE as the effect of seeding.
    pub control_completed_evaluations: Option<u64>,
}

/// Re-derive every round (and the control) from its directory and check the chain: identical
/// cuts and budget, each round's parent is the previous round's receipt, its seeds are exactly
/// [`seeds_after`] the previous round, and no round's champion is worse than the previous
/// round's on those cuts. `require_bred_candidates` also demands a measured candidate bred in
/// generation 1 or later in every round (the search ran past its seeds; not that it improved).
pub fn verify(path: &Path, require_bred_candidates: bool) -> Result<VerifyReport> {
    let lineage = read_lineage(path)?;
    let root = path.parent().map(Path::to_path_buf).unwrap_or_default();
    ensure!(
        !lineage.rounds.is_empty(),
        "the lineage records no completed round"
    );
    ensure!(
        lineage.rounds.len() as u64 <= lineage.rounds_requested,
        "more rounds than were requested"
    );
    ensure!(
        lineage.control.is_none() || lineage.control_requested,
        "a control that was not requested"
    );
    let mut previous: Option<VerifiedRun> = None;
    for (index, recorded) in lineage.rounds.iter().enumerate() {
        let round = index as u64 + 1;
        let label = format!("round {round}");
        ensure!(
            recorded.round == round && recorded.directory == round_directory(round),
            "{label}: numbered {} in {}",
            recorded.round,
            recorded.directory
        );
        let run =
            evidence::verify_champion_run(&root.join(&recorded.directory).join("receipt.json"))
                .with_context(|| label.clone())?;
        check_common(&lineage, &run, &label)?;
        ensure!(
            run.receipt.search.rng_seed == round_rng(lineage.rng_seed, round),
            "{label}: rng seed {} is not the lineage's round seed",
            run.receipt.search.rng_seed
        );
        let seeding = run.receipt.search.seeding.as_ref().context("v3 required")?;
        let (kind, origin) = match &previous {
            Some(parent) => {
                ensure!(
                    seeding.parent_receipt_sha256.as_deref()
                        == Some(parent.receipt_sha256.as_str()),
                    "{label}: its parent is not round {}'s receipt",
                    round - 1
                );
                let expected: Vec<String> = seeds_after(parent)?
                    .into_iter()
                    .map(|c| c.genome)
                    .filter(|g| g != search::BOOTSTRAP_GENOME)
                    .collect();
                let actual: Vec<String> = seeding.seeds.iter().map(|s| s.genome.clone()).collect();
                ensure!(
                    actual == expected,
                    "{label}: seeded with {actual:?}, not round {}'s champion and runners-up {expected:?}",
                    round - 1
                );
                let parent_rmse = parent.receipt.search.best.as_ref().unwrap().validation_rmse;
                let rmse = run.receipt.search.best.as_ref().unwrap().validation_rmse;
                ensure!(
                    rmse <= parent_rmse,
                    "{label}: champion mean held-out RMSE {rmse} is worse than round {}'s {parent_rmse} \
                     on the same cuts",
                    round - 1
                );
                (SEEDED, FROM_PREVIOUS_ROUND)
            }
            None if lineage.seeds_file_sha256.is_some() => {
                ensure!(
                    seeding.seeds_file_sha256 == lineage.seeds_file_sha256
                        && seeding.parent_receipt_sha256.is_none(),
                    "{label}: not seeded from the lineage's --seeds file"
                );
                (SEEDED, FROM_SEEDS_FILE)
            }
            None => {
                ensure!(
                    seeding.seeds.is_empty()
                        && seeding.parent_receipt_sha256.is_none()
                        && seeding.seeds_file_sha256.is_none(),
                    "{label}: an unseeded lineage's first round carries seeds"
                );
                (UNSEEDED, FROM_NONE)
            }
        };
        if kind == SEEDED {
            let file = root.join(&recorded.directory).join("seeds.json");
            let sha = crate::sha256_file(&file).with_context(|| format!("{label}: seeds.json"))?;
            ensure!(
                seeding.seeds_file_sha256.as_deref() == Some(sha.as_str()),
                "{label}: seeds.json is not the file the round was seeded from"
            );
        }
        let derived = round_record(round, kind, &recorded.directory, origin, &run)?;
        ensure!(
            derived == *recorded,
            "{label}: lineage.json does not match what its receipt and evidence show"
        );
        if require_bred_candidates {
            ensure!(
                derived.generation_ge1_measured > 0,
                "{label}: no candidate bred in generation 1 or later was measured"
            );
        }
        previous = Some(run);
    }
    let mut control_rmse = None;
    if let Some(recorded) = &lineage.control {
        let label = "control".to_owned();
        ensure!(
            recorded.directory == CONTROL_DIRECTORY
                && recorded.round == lineage.rounds_requested + 1
                && lineage.rounds.len() as u64 == lineage.rounds_requested,
            "control: misnumbered or recorded before every round completed"
        );
        let run = evidence::verify_champion_run(&root.join(CONTROL_DIRECTORY).join("receipt.json"))
            .context("control")?;
        check_common(&lineage, &run, &label)?;
        let seeding = run.receipt.search.seeding.as_ref().context("v3 required")?;
        ensure!(
            seeding.seeds.is_empty()
                && seeding.parent_receipt_sha256.is_none()
                && seeding.seeds_file_sha256.is_none(),
            "control: the control must be unseeded"
        );
        ensure!(
            run.receipt.search.rng_seed == lineage.rounds.last().unwrap().rng_seed,
            "control: not the final round's random stream"
        );
        let derived = round_record(recorded.round, CONTROL, CONTROL_DIRECTORY, FROM_NONE, &run)?;
        ensure!(
            derived == *recorded,
            "control: lineage.json does not match what its receipt and evidence show"
        );
        control_rmse = Some(derived.validation_rmse);
    }
    Ok(VerifyReport {
        schema: LINEAGE_SCHEMA_ID,
        ok: true,
        lineage: path.display().to_string(),
        dataset_hash: lineage.dataset.dataset_hash.clone(),
        prepared_sha256: lineage.prepared_sha256.clone(),
        rounds: lineage.rounds.len() as u64,
        validation_rmse: lineage.rounds.iter().map(|r| r.validation_rmse).collect(),
        control_validation_rmse: control_rmse,
        bred_candidates_in_every_round: lineage
            .rounds
            .iter()
            .all(|r| r.generation_ge1_measured > 0),
        generation_ge1_measured: lineage
            .rounds
            .iter()
            .map(|r| r.generation_ge1_measured)
            .collect(),
        champion_found_in_round: lineage
            .rounds
            .iter()
            .map(Round::champion_found_in_round)
            .collect(),
        improved_on_previous_round: lineage
            .rounds
            .windows(2)
            .map(|pair| pair[1].validation_rmse < pair[0].validation_rmse)
            .collect(),
        rounds_completed_evaluations: lineage.rounds.iter().map(|r| r.completed_evaluations).sum(),
        control_completed_evaluations: lineage.control.as_ref().map(|c| c.completed_evaluations),
    })
}

fn check_common(lineage: &Lineage, run: &VerifiedRun, label: &str) -> Result<()> {
    let receipt = &run.receipt;
    ensure!(
        receipt.dataset.dataset_hash == lineage.dataset.dataset_hash
            && receipt.dataset.id == lineage.dataset.id,
        "{label}: another dataset"
    );
    ensure!(
        receipt.prepared.sha256 == lineage.prepared_sha256
            && receipt.prepared.seed == lineage.fold_seed
            && receipt.prepared.cuts == lineage.cuts,
        "{label}: measured on other cuts than the lineage's"
    );
    ensure!(
        receipt.search.evaluations == lineage.evaluations
            && receipt.search.seconds == lineage.seconds,
        "{label}: another budget than the lineage's"
    );
    if receipt.schema != crate::receipt::RECEIPT_SCHEMA_ID {
        bail!(
            "{label}: a {} receipt cannot be part of a lineage",
            receipt.schema
        );
    }
    let islands = receipt
        .search
        .islands
        .as_ref()
        .with_context(|| format!("{label}: a v3 receipt records its islands"))?;
    ensure!(
        islands.count == lineage.islands.max(1)
            && islands.migration_interval == lineage.migration_interval,
        "{label}: another island model than the lineage's ({} island(s), migration every {} \
         generation(s), where the lineage records {} and {})",
        islands.count,
        islands.migration_interval,
        lineage.islands,
        lineage.migration_interval
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{clock::Clock, emit::Language, run::Policy, search::Budget};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    fn dataset() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../datasets/test/pmlb-229-pwlinear"
        )
        .to_owned()
    }

    fn template(evaluations: usize) -> RunOptions {
        RunOptions {
            budget: Budget {
                evaluations,
                seconds: 600,
            },
            rng_seed: 7,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(dataset(), PathBuf::new())
        }
    }

    fn evolve_into(
        output: &Path,
        rounds: usize,
        control: bool,
        run: RunOptions,
    ) -> Result<Lineage> {
        evolve(&EvolveOptions {
            run,
            output: output.to_path_buf(),
            rounds,
            control,
            seeds: None,
        })
    }

    /// Every file under `root`, by relative path.
    fn tree(root: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
        fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else {
                    out.insert(
                        path.strip_prefix(root).unwrap().display().to_string(),
                        std::fs::read(&path).unwrap(),
                    );
                }
            }
        }
        let mut out = std::collections::BTreeMap::new();
        walk(root, root, &mut out);
        out
    }

    /// Every entry under `root`, directories included, by relative path (a file with its bytes).
    fn entries(root: &Path) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let bytes = if path.is_dir() {
                    pending.push(path.clone());
                    Vec::new()
                } else {
                    std::fs::read(&path).unwrap()
                };
                out.push((
                    path.strip_prefix(root).unwrap().display().to_string(),
                    bytes,
                ));
            }
        }
        out.sort();
        out
    }

    #[test]
    fn rounds_chain_on_identical_cuts_never_get_worse_and_verify() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        let lineage = evolve_into(&out, 3, true, template(24)).unwrap();
        assert_eq!(lineage.rounds.len(), 3);
        assert_eq!(lineage.rounds[0].kind, UNSEEDED);
        assert_eq!(lineage.rounds[0].seed_origin, FROM_NONE);
        for pair in lineage.rounds.windows(2) {
            assert!(pair[1].validation_rmse <= pair[0].validation_rmse);
            assert_eq!(
                pair[1].parent_receipt_sha256.as_deref(),
                Some(pair[0].receipt_sha256.as_str())
            );
            assert_eq!(pair[1].seed_origin, FROM_PREVIOUS_ROUND);
            assert_ne!(pair[1].rng_seed, pair[0].rng_seed);
        }
        let control = lineage.control.as_ref().unwrap();
        assert_eq!(control.kind, CONTROL);
        assert!(control.seeds.is_empty() && control.parent_receipt_sha256.is_none());
        assert_eq!(control.rng_seed, lineage.rounds[2].rng_seed);
        // Round 2 starts from round 1's champion and ranked runners-up, nothing else.
        let first = evidence::verify_run(&out.join("round-001/receipt.json")).unwrap();
        let expected: Vec<String> = seeds_after(&first)
            .unwrap()
            .into_iter()
            .map(|c| c.genome)
            .filter(|g| g != search::BOOTSTRAP_GENOME)
            .collect();
        assert_eq!(
            lineage.rounds[1]
                .seeds
                .iter()
                .map(|s| s.genome.clone())
                .collect::<Vec<_>>(),
            expected
        );
        // Per-generation best-so-far never rises, and every round measured bred candidates.
        for round in &lineage.rounds {
            for cut in &round.cuts {
                let values: Vec<f64> = cut.best_so_far.iter().flatten().copied().collect();
                assert!(values.windows(2).all(|w| w[1] <= w[0]), "{values:?}");
            }
            assert!(round.generation_ge1_measured > 0);
        }
        assert_eq!(read_lineage(&out.join(LINEAGE_FILE)).unwrap(), lineage);
        let report = verify(&out.join(LINEAGE_FILE), true).unwrap();
        assert_eq!(report.rounds, 3);
        assert!(report.bred_candidates_in_every_round);
        // What the rounds found is reported per round, and agrees with the RMSEs.
        assert_eq!(
            report.champion_found_in_round,
            lineage
                .rounds
                .iter()
                .map(Round::champion_found_in_round)
                .collect::<Vec<_>>()
        );
        assert_eq!(report.improved_on_previous_round.len(), 2);
        for (index, improved) in report.improved_on_previous_round.iter().enumerate() {
            let (before, after) = (&lineage.rounds[index], &lineage.rounds[index + 1]);
            assert_eq!(*improved, after.validation_rmse < before.validation_rmse);
            // Never worse, so a round that did not improve tied its parent exactly.
            if !*improved {
                assert_eq!(after.validation_rmse, before.validation_rmse);
            }
        }
        assert_eq!(
            report.control_validation_rmse,
            Some(control.validation_rmse)
        );
        // The compute the control comparison rests on: the rounds' total against one round's.
        assert_eq!(
            report.rounds_completed_evaluations,
            lineage
                .rounds
                .iter()
                .map(|r| r.completed_evaluations)
                .sum::<u64>()
        );
        assert_eq!(
            report.control_completed_evaluations,
            Some(control.completed_evaluations)
        );
        assert!(report.rounds_completed_evaluations > control.completed_evaluations);
    }

    #[test]
    fn a_tampered_lineage_round_receipt_or_seeds_file_fails_verification() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        evolve_into(&out, 2, false, template(16)).unwrap();
        let path = out.join(LINEAGE_FILE);
        verify(&path, false).unwrap();
        let expect_refused = |needle: &str| {
            let error = format!("{:#}", verify(&path, false).unwrap_err());
            assert!(error.contains(needle), "wanted {needle:?}: {error}");
        };

        let pristine = std::fs::read(&path).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
        let rmse = value["rounds"][1]["validation_rmse"].as_f64().unwrap();
        value["rounds"][1]["validation_rmse"] = (rmse / 2.0).into();
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        expect_refused("does not match what its receipt and evidence show");
        std::fs::write(&path, &pristine).unwrap();

        // The island model is lineage-wide: a lineage that claims another one than its rounds
        // were searched with is refused, not reported.
        for (field, claimed) in [("islands", 3u64), ("migration_interval", 1)] {
            let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
            assert_ne!(value[field].as_u64(), Some(claimed));
            value[field] = claimed.into();
            std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            expect_refused("another island model than the lineage's");
            std::fs::write(&path, &pristine).unwrap();
        }

        let seeds = out.join("round-002/seeds.json");
        let original = std::fs::read(&seeds).unwrap();
        std::fs::write(&seeds, [original.as_slice(), b" "].concat()).unwrap();
        expect_refused("seeds.json is not the file");
        std::fs::write(&seeds, &original).unwrap();

        let observations = out.join("round-001/search/cut-0/observations.json");
        let original = std::fs::read(&observations).unwrap();
        std::fs::write(&observations, [original.as_slice(), b" "].concat()).unwrap();
        expect_refused("round 1");
        std::fs::write(&observations, &original).unwrap();

        verify(&path, false).unwrap();
    }

    /// `--threads 1` and `--threads 8` write the same lineage: every round directory and
    /// `lineage.json`, byte for byte, with the wall clock frozen (so the evaluation budget, never
    /// the seconds bound, ends every phase; a binding seconds bound is timing-dependent).
    #[test]
    fn one_and_eight_threads_write_byte_identical_lineages() {
        let temp = tempfile::tempdir().unwrap();
        let write = |threads: usize, name: &str| {
            let out = temp.path().join(name);
            evolve_into(
                &out,
                2,
                true,
                RunOptions {
                    threads,
                    clock: Clock::Frozen,
                    ..template(20)
                },
            )
            .unwrap();
            tree(&out)
        };
        let workers = crate::parallel::affinity_cpus()
            .unwrap()
            .min(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
            .min(8);
        if workers < 2 {
            eprintln!("skipped: this host admits one worker");
            return;
        }
        let one = write(1, "one");
        let many = write(workers, "many");
        assert_eq!(
            one.keys().collect::<Vec<_>>(),
            many.keys().collect::<Vec<_>>()
        );
        for (path, bytes) in &one {
            assert!(
                many[path] == *bytes,
                "{path} differs between 1 and {workers} threads"
            );
        }
        assert!(one.contains_key(LINEAGE_FILE) && one.contains_key("control/receipt.json"));
    }

    /// A stop request during round 2 (SIGTERM in the CLI) leaves round 2 without a receipt and
    /// the lineage with round 1 only, which still verifies.
    #[test]
    fn a_stop_request_during_a_round_keeps_only_completed_rounds() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        let flag = Arc::new(AtomicBool::new(false));
        let watcher = {
            let flag = flag.clone();
            let prepared = out.join("round-002/prepared/prepared.json");
            std::thread::spawn(move || {
                while !prepared.exists() {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                flag.store(true, Ordering::SeqCst);
            })
        };
        let error = evolve_into(
            &out,
            3,
            true,
            RunOptions {
                cancel: Some(flag),
                ..template(30)
            },
        )
        .unwrap_err();
        watcher.join().unwrap();
        assert!(crate::parallel::is_cancelled(&error), "{error:#}");
        let lineage = read_lineage(&out.join(LINEAGE_FILE)).unwrap();
        assert_eq!(lineage.rounds.len(), 1);
        assert!(lineage.control.is_none());
        assert!(out.join("round-001/receipt.json").exists());
        assert!(!out.join("round-002/receipt.json").exists());
        assert!(!out.join("round-003").exists() && !out.join(CONTROL_DIRECTORY).exists());
        verify(&out.join(LINEAGE_FILE), false).unwrap();
    }

    /// A round whose champion is one of its own seeds found nothing better, however many bred
    /// candidates it measured; the report must say so rather than call it a discovery.
    #[test]
    fn a_round_whose_champion_is_its_seed_is_not_reported_as_a_discovery() {
        let temp = tempfile::tempdir().unwrap();
        // A strong pipeline for this dataset, against 3 bred candidates in generation 1.
        let genome =
            seeds::canonical_pipeline("(pipe X (rank append=true) (gpr m=400 steps=60))").unwrap();
        let file = temp.path().join("strong.json");
        seeds::write(
            &file,
            &seeds::SeedsFile::new(
                None,
                vec![seeds::SeedGenome {
                    program_sha256: sha256_hex(genome.as_bytes()),
                    genome: genome.clone(),
                    bank_entry_sha256: None,
                    receipt_sha256: None,
                }],
            ),
        )
        .unwrap();
        let out = temp.path().join("lineage");
        let lineage = evolve(&EvolveOptions {
            run: template(16),
            output: out.clone(),
            rounds: 1,
            control: false,
            seeds: Some(file),
        })
        .unwrap();
        let round = &lineage.rounds[0];
        assert_eq!(
            round.selected_genome, genome,
            "the seed was beaten after all"
        );
        assert!(round.generation_ge1_measured > 0);
        assert!(!round.champion_found_in_round());
        assert_eq!(round.champion_origin(), "carried from its seeds");
        // The bred-candidates check still holds (the search ran past generation 0), and the
        // report says, separately, that nothing it bred won.
        let report = serde_json::to_value(verify(&out.join(LINEAGE_FILE), true).unwrap()).unwrap();
        assert_eq!(report["bred_candidates_in_every_round"], true, "{report}");
        assert_eq!(
            report["champion_found_in_round"],
            serde_json::json!([false]),
            "{report}"
        );
        assert_eq!(report["improved_on_previous_round"], serde_json::json!([]));
        assert!(report.get("discovery_in_every_round").is_none(), "{report}");
    }

    #[test]
    fn round_one_can_start_from_a_seeds_file() {
        let temp = tempfile::tempdir().unwrap();
        let genome = seeds::canonical_pipeline("(pipe X (standardize) (ridge_gcv))").unwrap();
        let file = temp.path().join("bank-export.json");
        let sha = seeds::write(
            &file,
            &seeds::SeedsFile::new(
                None,
                vec![seeds::SeedGenome {
                    program_sha256: sha256_hex(genome.as_bytes()),
                    genome: genome.clone(),
                    bank_entry_sha256: None,
                    receipt_sha256: None,
                }],
            ),
        )
        .unwrap();
        let out = temp.path().join("lineage");
        let lineage = evolve(&EvolveOptions {
            run: template(16),
            output: out.clone(),
            rounds: 1,
            control: false,
            seeds: Some(file),
        })
        .unwrap();
        assert_eq!(lineage.seeds_file_sha256.as_deref(), Some(sha.as_str()));
        let round = &lineage.rounds[0];
        assert_eq!(
            (round.kind.as_str(), round.seed_origin.as_str()),
            (SEEDED, FROM_SEEDS_FILE)
        );
        assert_eq!(round.seeds[0].genome, genome);
        verify(&out.join(LINEAGE_FILE), false).unwrap();
    }

    /// A copy of the committed regression fixture whose catalog row says `binary`.
    fn binary_dataset(root: &Path) -> String {
        let source = Path::new(&dataset()).to_path_buf();
        let target = root.join("binary");
        std::fs::create_dir_all(&target).unwrap();
        for name in ["train.csv", "test.csv"] {
            std::fs::copy(source.join(name), target.join(name)).unwrap();
        }
        let mut row: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("metadata.json")).unwrap()).unwrap();
        row["task"] = "binary".into();
        row["target"]["type"] = "binary".into();
        std::fs::write(target.join("metadata.json"), row.to_string()).unwrap();
        target.display().to_string()
    }

    /// Every refusal a round could make is made before evolve writes anything: a seed that is
    /// not a pipeline, an unresolvable or non-regression dataset, a budget that cannot re-measure
    /// round 1's seeds or the seeds a later round carries. Nothing is left behind, so the
    /// corrected command runs into the same --output.
    #[test]
    fn a_refused_lineage_writes_nothing_and_the_corrected_retry_runs() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        let seeds_of = |name: &str, genomes: &[&str]| {
            let path = temp.path().join(name);
            seeds::write(
                &path,
                &seeds::SeedsFile::new(
                    None,
                    genomes
                        .iter()
                        .map(|g| seeds::SeedGenome {
                            genome: (*g).to_owned(),
                            program_sha256: sha256_hex(g.as_bytes()),
                            bank_entry_sha256: None,
                            receipt_sha256: None,
                        })
                        .collect(),
                ),
            )
            .unwrap();
            path
        };
        let not_a_pipeline = seeds_of("sym.json", &["(sym affine \"x0\")"]);
        let four = seeds_of(
            "four.json",
            &[
                "(pipe X (standardize) (ridge))",
                "(pipe X (rank) (ridge))",
                "(pipe X (blom) (ridge))",
                "(pipe X (robust_scale) (ridge))",
            ],
        );
        let refused = |options: EvolveOptions, needle: &str| {
            let error = format!("{:#}", evolve(&options).unwrap_err());
            assert!(error.contains(needle), "wanted {needle:?}: {error}");
            assert!(
                !out.exists(),
                "a refusal ({needle}) left {} behind",
                out.display()
            );
        };
        let options = |run: RunOptions, rounds: usize, seeds: Option<PathBuf>| EvolveOptions {
            run,
            output: out.clone(),
            rounds,
            control: false,
            seeds,
        };
        refused(
            options(template(16), 2, Some(not_a_pipeline)),
            "seed \"(sym affine",
        );
        refused(
            options(
                RunOptions {
                    dataset: "0000000000000000".to_owned(),
                    ..template(16)
                },
                2,
                None,
            ),
            "not in the local dataset cache",
        );
        refused(
            options(
                RunOptions {
                    dataset: binary_dataset(temp.path()),
                    ..template(16)
                },
                2,
                None,
            ),
            "task binary is unsupported",
        );
        refused(
            options(template(4), 1, Some(four.clone())),
            "cannot re-measure 4 seed(s)",
        );
        refused(
            options(
                RunOptions {
                    cuts: 2,
                    ..template(8)
                },
                2,
                None,
            ),
            "every round after the first",
        );
        // The corrected command runs into the same directory.
        let lineage = evolve(&options(template(16), 2, Some(four))).unwrap();
        assert_eq!(lineage.rounds.len(), 2);
        verify(&out.join(LINEAGE_FILE), false).unwrap();
    }

    /// The PR #118 review's reproduction: `--rounds 2 --cuts 6` was refused by the preparation
    /// inside round 1, after `round-001/` existed, so the corrected `--cuts 3` into the same
    /// `--output` failed with "already holds a lineage". A cut count outside the partition (0 or
    /// 6) and a dataset that resolves and parses but cannot be cut are now refused before
    /// anything is written, and the corrected command runs.
    #[test]
    fn a_refused_cut_count_leaves_nothing_and_the_corrected_retry_runs() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        let refused = |run: RunOptions, needle: &str| {
            let error = format!("{:#}", evolve_into(&out, 2, false, run).unwrap_err());
            assert!(error.contains(needle), "wanted {needle:?}: {error}");
            assert!(
                !out.exists(),
                "a refusal ({needle}) left {} behind",
                out.display()
            );
        };
        for cuts in [0, crate::prepare::FOLDS + 1] {
            refused(
                RunOptions {
                    cuts,
                    ..template(16)
                },
                &format!("--cuts must be 1..={}, got {cuts}", crate::prepare::FOLDS),
            );
        }
        refused(
            RunOptions {
                dataset: crate::run::tests::constant_target_dataset(temp.path()),
                cuts: 3,
                ..template(16)
            },
            "held-out target is constant",
        );
        let lineage = evolve_into(
            &out,
            2,
            false,
            RunOptions {
                cuts: 3,
                ..template(16)
            },
        )
        .unwrap();
        assert_eq!(lineage.rounds.len(), 2);
        assert_eq!(lineage.cuts, vec![0, 1, 2]);
        verify(&out.join(LINEAGE_FILE), false).unwrap();
    }

    #[test]
    fn an_existing_lineage_or_a_bad_round_count_is_refused_before_writing() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("lineage");
        for rounds in [0, MAX_ROUNDS + 1] {
            assert!(evolve_into(&out, rounds, false, template(4)).is_err());
            assert!(!out.exists());
        }
        let error = format!(
            "{:#}",
            evolve_into(
                &out,
                1,
                false,
                RunOptions {
                    cpu_ceiling_millicores: Some(10),
                    ..template(4)
                }
            )
            .unwrap_err()
        );
        assert!(error.contains("--cpu-ceiling"), "{error}");
        assert!(!out.exists());
        evolve_into(&out, 1, false, template(4)).unwrap();
        let error = format!(
            "{:#}",
            evolve_into(&out, 1, false, template(4)).unwrap_err()
        );
        assert!(error.contains("already holds a lineage"), "{error}");
    }

    /// The PR #118 verifier's reproduction: an `--output` holding a stale `control/` or
    /// `round-002/` was refused only when that run was reached, after the earlier rounds had
    /// written `lineage.json` (and round 2 its `seeds.json`), so the corrected retry was refused
    /// as "already holds a lineage". Anything in `--output`, an unrelated file included, is now
    /// refused before a byte is written, and so is an `--output` that is not a directory. An
    /// empty directory is accepted: the same command runs once the stale entry is removed.
    #[test]
    fn an_objective_ranked_lineage_is_refused_before_writing() {
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("objective");
        let run = RunOptions {
            objective: Some(jope_runtime::fitness::Fitness::Mae {}),
            ..template(10)
        };
        let error = format!("{:#}", evolve_into(&out, 2, false, run).unwrap_err());
        assert!(
            error.contains("ranks and verifies its rounds by held-out RMSE"),
            "{error}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn a_non_empty_output_is_refused_before_writing_and_the_emptied_one_runs() {
        let temp = tempfile::tempdir().unwrap();
        for (name, stale, control, needle) in [
            (
                "stale-control",
                "control/receipt.json",
                true,
                "is not empty (control)",
            ),
            (
                "stale-round",
                "round-002/receipt.json",
                false,
                "is not empty (round-002)",
            ),
            ("unrelated", "notes.txt", false, "is not empty (notes.txt)"),
            (
                "old-lineage",
                LINEAGE_FILE,
                false,
                "already holds a lineage",
            ),
        ] {
            let out = temp.path().join(name);
            let path = out.join(stale);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"{}\n").unwrap();
            let before = entries(&out);
            let error = format!(
                "{:#}",
                evolve_into(&out, 2, control, template(10)).unwrap_err()
            );
            assert!(error.contains(needle), "{name}: wanted {needle:?}: {error}");
            assert_eq!(entries(&out), before, "{name}: the refusal wrote into it");
        }
        let file = temp.path().join("a-file");
        std::fs::write(&file, b"x").unwrap();
        let dangling = temp.path().join("dangling");
        let missing = temp.path().join("missing");
        std::os::unix::fs::symlink(&missing, &dangling).unwrap();
        for out in [&file, &dangling] {
            let error = format!("{:#}", evolve_into(out, 1, false, template(4)).unwrap_err());
            assert!(
                error.contains("not a directory to write a lineage into"),
                "{error}"
            );
        }
        assert_eq!(std::fs::read(&file).unwrap(), b"x");
        assert!(
            !missing.exists(),
            "evolve created the dangling link's target"
        );
        // Positive control: the stale control/ removed, the same command runs into the directory.
        let out = temp.path().join("stale-control");
        std::fs::remove_dir_all(out.join(CONTROL_DIRECTORY)).unwrap();
        assert!(entries(&out).is_empty());
        let lineage = evolve_into(&out, 2, true, template(10)).unwrap();
        assert_eq!(lineage.rounds.len(), 2);
        assert!(lineage.control.is_some());
        verify(&out.join(LINEAGE_FILE), false).unwrap();
    }
}
