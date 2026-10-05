//! The exact rewrite pass at the end of each search round (laneD:B20-rewrite-simplify).
use super::super::{Refinement, State};
use super::*;
use crate::{
    cli::{Cli, Cmd},
    clock::Clock,
    emit::Language,
    kernel2::{
        archive, publication,
        receipt::Receipt,
        run,
        tests::{data, evaluator},
        ArchivePolicy, EngineSelection, Kernel2Observation, MovesRecord, RefineMethod,
    },
    run::{Policy, RunOptions},
    search::{Budget, SearchOptions},
};
use clap::Parser;
use feat_math::{evolve::FaultPolicy, kernel::expression_search::LmOptions};
use std::path::PathBuf;

/// `1*c0`, which the pass reduces to `c0`.
const ONE_TIMES: &str = "(pipe X (derive expr=\"1*c0\") (ridge))";
/// `c0*1.0`, the same program spelled the other way round.
const TIMES_ONE: &str = "(pipe X (derive expr=\"c0*1.0\") (ridge))";
/// A division by one and a foldable product, before a scaling stage.
const FOLDED: &str = "(pipe X (derive expr=\"(c0/1.0)+(2.0*3.0)\") (standardize) (ridge))";
/// An elementwise carrier.
const MAPPED: &str = "(pipe X (map expr=\"abs(abs(x))\") (ridge))";
/// `c0*(-1.0)` reduces to `-c0`, a negation of a variable the profile refuses.
const NEGATED: &str = "(pipe X (derive expr=\"c0*(-1.0)\") (ridge))";
/// A refused stage beside an admitted one.
const MIXED: &str = "(pipe X (derive expr=\"c0*(-1.0)\") (derive expr=\"1*c0\") (ridge))";
/// Already as small as its class.
const SINE: &str = "(pipe X (derive expr=\"sin(0.5*c0)\") (ridge))";

fn penalize() -> FaultPolicy {
    FaultPolicy::Penalize {
        lambda_domain: 0.05,
        lambda_saturation: 0.0,
        tie_break: true,
    }
}

fn simplify() -> EngineSelection {
    EngineSelection {
        simplify: true,
        ..EngineSelection::default()
    }
}

/// Search options with the pass selected, `top_k` parents per round and no refinement trials.
fn options(top_k: usize, evaluations: usize, policy: FaultPolicy) -> Options {
    let mut search = SearchOptions::new(
        Budget {
            evaluations,
            seconds: 120,
        },
        7,
    );
    search.clock = Clock::Frozen;
    Options::new(
        search,
        "kernel2-baseline",
        policy,
        Refinement {
            top_k,
            trials_per_program: 0,
        },
    )
    .unwrap()
    .with_engine(simplify())
    .unwrap()
}

/// A search state holding `genomes`, each measured on `cut` and charged.
fn parents(cut: &Cut, options: &Options, genomes: &[&str]) -> State {
    let mut state = State::new(options.search.budget.evaluations);
    for genome in genomes {
        let o = options
            .evaluator
            .measure(
                genome,
                cut,
                0,
                "random",
                options.search.clock,
                &mut state.budget,
            )
            .unwrap();
        assert!(o.observation.measured(), "{genome}: {:?}", o.observation);
        state.seen.insert(o.observation.genome.clone());
        state.observations.push(o);
    }
    state
}

/// `parent` with stage `index`'s expression replaced by `expression`, as the pass prints it.
fn with_expression(parent: &str, index: usize, expression: &str) -> String {
    let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse(parent).unwrap() else {
        panic!("{parent} is not a pipeline");
    };
    let source = MathExpr::parse(expression).unwrap().to_source().unwrap();
    pipe.stages[index]
        .params
        .insert("expr".into(), ParamValue::Str(source));
    GenomeIr::Pipeline(pipe).canonical()
}

/// Everything a measurement scores, as bits.
type Scores = (
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    u64,
    u64,
    u64,
);

fn scores(o: &Kernel2Observation) -> Scores {
    (
        o.observation.train_r2.map(f64::to_bits),
        o.observation.hold_r2.map(f64::to_bits),
        o.observation.rmse.map(f64::to_bits),
        o.selection_score.map(f64::to_bits),
        o.faults.evaluated_cells,
        o.faults.domain_cells,
        o.faults.saturation_cells,
    )
}

fn parent_of<'a>(state: &'a State, record: &MoveRecord) -> &'a Kernel2Observation {
    state
        .observations
        .iter()
        .find(|o| o.observation.program_sha256 == record.parent_sha256)
        .unwrap()
}

fn all_genomes_distinct(state: &State) -> bool {
    let genomes: BTreeSet<&str> = state
        .observations
        .iter()
        .map(|o| o.observation.genome.as_str())
        .collect();
    genomes.len() == state.observations.len()
}

#[test]
fn rewritten_children_score_bit_for_bit_as_their_parents_and_cost_one_each() {
    let cut = data();
    for policy in [FaultPolicy::RejectHard, penalize()] {
        let opt = options(4, 64, policy);
        let mut state = parents(&cut, &opt, &[ONE_TIMES, TIMES_ONE, FOLDED, MAPPED]);
        let before = state.budget.used();
        simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
        assert_eq!(state.moves.len(), 4, "{policy:?}");
        let measured: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == MEASURED)
            .collect();
        // `1*c0` and `c0*1.0` reduce to one program: the first ranked is measured, the other is a
        // duplicate and spends nothing.
        assert_eq!(measured.len(), 3, "{:?}", state.moves);
        let duplicate: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == DUPLICATE)
            .collect();
        assert_eq!(duplicate.len(), 1);
        assert!(duplicate[0].measurement.is_none());
        // One evaluation per measured child; the shared identity holds; no program twice.
        assert_eq!(state.budget.used() - before, measured.len());
        assert_eq!(state.budget.used(), state.observations.len());
        assert!(all_genomes_distinct(&state));
        let expected: BTreeSet<String> = [
            with_expression(ONE_TIMES, 0, "c0"),
            with_expression(FOLDED, 0, "c0+6.0"),
            with_expression(MAPPED, 0, "abs(x)"),
        ]
        .into_iter()
        .collect();
        let programs: BTreeSet<String> = measured
            .iter()
            .map(|m| m.program.clone().unwrap())
            .collect();
        assert_eq!(programs, expected);
        assert_eq!(
            duplicate[0].program,
            Some(with_expression(ONE_TIMES, 0, "c0"))
        );
        for m in &measured {
            let child = &state.observations[m.measurement.unwrap()];
            let parent = parent_of(&state, m);
            assert_eq!(child.observation.origin, crate::kernel2::SIMPLIFY);
            assert_eq!(child.observation.generation, state.generation);
            assert_eq!(Some(&child.observation.genome), m.program.as_ref());
            assert!(child.observation.measured(), "{:?}", child.observation);
            assert_eq!(
                scores(child),
                scores(parent),
                "{} vs {}",
                child.observation.genome,
                parent.observation.genome
            );
            assert_eq!(m.rewrites.len(), 1);
            let rewrite = &m.rewrites[0];
            assert!(rewrite.expression.is_some() && rewrite.refusal.is_none());
            assert_eq!(rewrite.stop, "saturated");
            assert!(rewrite.iterations > 0 && rewrite.applications > 0);
            assert!(rewrite.enodes > 0 && rewrite.peak >= rewrite.enodes);
        }
    }
}

#[test]
fn refused_outputs_are_discarded_with_their_reason_and_admitted_stages_kept() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &[NEGATED, MIXED, SINE]);
    let before = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let record = |genome: &str| {
        let sha = crate::sha256_hex(genome.as_bytes());
        state
            .moves
            .iter()
            .find(|m| m.parent_sha256 == sha)
            .unwrap()
            .clone()
    };
    // The whole proposal refused: nothing assembled, nothing charged, the reason kept.
    let negated = record(NEGATED);
    assert_eq!(negated.outcome, REFUSED);
    assert!(negated.program.is_none() && negated.measurement.is_none());
    assert!(negated.reason.is_none());
    let refusal = negated.rewrites[0].refusal.as_deref().unwrap();
    assert!(refusal.contains("negative constant"), "{refusal}");
    assert!(negated.rewrites[0].expression.is_none());
    // One stage refused, the other admitted: the refused stage keeps its expression.
    let mixed = record(MIXED);
    assert_eq!(mixed.outcome, MEASURED);
    assert!(mixed.rewrites[0].refusal.is_some() && mixed.rewrites[0].expression.is_none());
    assert!(mixed.rewrites[1].refusal.is_none() && mixed.rewrites[1].expression.is_some());
    assert_eq!(mixed.program, Some(with_expression(MIXED, 1, "c0")));
    let child = &state.observations[mixed.measurement.unwrap()];
    assert_eq!(scores(child), scores(parent_of(&state, &mixed)));
    // Nothing smaller: unchanged, with the saturation counters still recorded.
    let sine = record(SINE);
    assert_eq!(sine.outcome, UNCHANGED);
    assert!(sine.program.is_none() && sine.measurement.is_none());
    assert!(sine.rewrites[0].expression.is_none() && sine.rewrites[0].refusal.is_none());
    assert!(sine.rewrites[0].iterations > 0);
    assert_eq!(state.budget.used() - before, 1);
    assert_eq!(state.budget.used(), state.observations.len());
    // The refusal is the profile's own: the same program with `-c0` is refused by admission.
    let refused = with_expression(NEGATED, 0, "-c0");
    assert!(admit_child(&refused, opt.evaluator.profile(), 1).is_err());
}

#[test]
fn the_allowance_and_the_start_guard_bound_the_pass() {
    let cut = data();
    let genomes = [ONE_TIMES, FOLDED, MAPPED];
    // One evaluation left: one child, then the pass stops before the next proposal.
    let opt = options(4, genomes.len() + 1, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, MEASURED);
    assert_eq!(state.budget.remaining(), 0);
    assert_eq!(state.budget.used(), state.observations.len());
    // None left: no proposal at all.
    let opt = options(4, genomes.len(), FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert!(state.moves.is_empty());
    // The guard refuses the first proposal: nothing is recorded or spent.
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    let used = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || false).unwrap();
    assert!(state.moves.is_empty());
    assert_eq!(state.budget.used(), used);
    // The guard admits the proposal and refuses its measurement: recorded as not started, and
    // the pass ends.
    let mut state = parents(&cut, &opt, &genomes);
    let mut calls = 0;
    simplify_round(&mut state, &cut, &opt, &mut || {
        calls += 1;
        calls == 1
    })
    .unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, NOT_STARTED);
    assert!(state.moves[0].program.is_some() && state.moves[0].measurement.is_none());
    assert_eq!(state.budget.used(), used);
    // Only `top_k` parents are read.
    let opt = options(1, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
}

#[test]
fn proposals_are_identical_with_a_poisoned_holdout() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut poisoned = cut.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    // The parents are fixed: measured on the clean rows.
    let fixed = parents(&cut, &opt, &[ONE_TIMES, FOLDED, NEGATED, MAPPED]);
    let run = |c: &Cut| {
        let mut state = State::new(opt.search.budget.evaluations);
        state.budget = fixed.budget.clone();
        state.observations = fixed.observations.clone();
        state.seen = fixed.seen.clone();
        simplify_round(&mut state, c, &opt, &mut || true).unwrap();
        state
    };
    let (clean, dirty) = (run(&cut), run(&poisoned));
    assert_eq!(clean.moves, dirty.moves);
    assert_eq!(clean.budget.used(), dirty.budget.used());
    // The poison reached the children's measurements, not the proposals.
    let children = |s: &State| -> Vec<bool> {
        s.moves
            .iter()
            .filter_map(|m| m.measurement)
            .map(|i| s.observations[i].observation.measured())
            .collect()
    };
    assert_eq!(children(&clean), vec![true; 3]);
    assert_eq!(children(&dirty), vec![false; 3]);
    // A proposal reads the parent's text and the input width only.
    for genome in [ONE_TIMES, FOLDED, NEGATED, MAPPED] {
        let a = propose(&opt.evaluator, genome, 1).unwrap();
        let b = propose(&opt.evaluator, genome, 1).unwrap();
        assert_eq!(
            (a.rewrites, a.program, a.reason, a.admitted),
            (b.rewrites, b.program, b.reason, b.admitted)
        );
    }
}

// Runs through the published route.

/// The `cli_tests` seed program, which the pass reduces to `c0`.
const SEED: &str = ONE_TIMES;

/// The files every run publishes.
const SIDECARS: [&str; 7] = [
    "receipt.json",
    "prepared/prepared.json",
    "search/cut-0/search.json",
    "search/cut-0/observations.json",
    "search/cut-0/refinements.json",
    "emit/model.rs",
    "emit/manifest.json",
];

fn dataset() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear")
        .display()
        .to_string()
}

/// The `cli_tests` run (pmlb-229-pwlinear, rng seed 7, a frozen clock, Rust only, [`SEED`] as the
/// seeds file beside the output) in `dir/run`. The default selection is left unapplied, as the
/// CLI does.
fn run_options(
    dir: &Path,
    evaluations: usize,
    refinement: Refinement,
    engine: EngineSelection,
) -> run::Options {
    std::fs::create_dir_all(dir).unwrap();
    let mut base = RunOptions::new(dataset(), dir.join("run"));
    base.budget = Budget {
        evaluations,
        seconds: 300,
    };
    base.policy = Policy::Skip;
    base.languages = vec![Language::Rust];
    base.rng_seed = 7;
    base.clock = Clock::Frozen;
    base.source_commit = Some("a".repeat(40));
    let seeds = dir.join("seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        vec![crate::seeds::SeedGenome {
            genome: SEED.into(),
            program_sha256: crate::sha256_hex(SEED.as_bytes()),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }],
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    base.seed_genomes = Some(seeds);
    let options = run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        refinement,
    )
    .unwrap();
    if engine == EngineSelection::default() {
        options
    } else {
        options.with_engine(engine).unwrap()
    }
}

fn four_parents() -> Refinement {
    Refinement {
        top_k: 4,
        trials_per_program: 4,
    }
}

fn read(opt: &run::Options, relative: &str) -> Vec<u8> {
    std::fs::read(opt.run.output.join(relative)).unwrap()
}

fn published_moves(opt: &run::Options) -> Vec<MoveRecord> {
    serde_json::from_slice(&read(opt, PATH)).unwrap()
}

fn published_measurements(opt: &run::Options) -> Vec<Measurement> {
    serde_json::from_slice(&read(opt, "search/cut-0/observations.json")).unwrap()
}

fn lint(text: &str) {
    let hits = feat_math::transpile::lint_shipped_text(text);
    assert!(hits.is_empty(), "{text}: {hits:?}");
}

/// Every measured move's child scores bit for bit as its parent, in the published records.
fn check_exact(opt: &run::Options) -> usize {
    let measurements = published_measurements(opt);
    let moves = published_moves(opt);
    let bits = |m: &Measurement| {
        (
            m.observation.train_r2.map(f64::to_bits),
            m.observation.hold_r2.map(f64::to_bits),
            m.observation.rmse.map(f64::to_bits),
            m.selection_score.map(f64::to_bits),
            m.faults.clone(),
        )
    };
    let mut count = 0;
    for m in moves.iter().filter(|m| m.outcome == MEASURED) {
        let child = &measurements[m.measurement.unwrap()];
        let parent = measurements
            .iter()
            .find(|p| {
                p.observation.program_sha256 == m.parent_sha256
                    && p.observation.status == crate::search::MEASURED
            })
            .unwrap();
        assert_eq!(child.observation.status, crate::search::MEASURED);
        assert_eq!(bits(child), bits(parent), "{}", child.observation.genome);
        count += 1;
    }
    assert_eq!(
        count,
        measurements
            .iter()
            .filter(|m| m.observation.origin == crate::kernel2::SIMPLIFY)
            .count()
    );
    count
}

#[test]
fn rewrite_runs_replay_every_file_with_a_frozen_clock_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let lm = RefineMethod::Lm(LmOptions::default());
    // The damped least-squares run refines one parent per round. With four parents and 80 or more
    // evaluations, two parents refine to one program in one round on this fixture, and
    // `verify_run` refuses that run with the pass off too, under either refinement method:
    // publication indexes each refinement child by the first equal observation. It gets a larger
    // allowance, since each refinement spends up to 12 trials, to leave the pass measurements.
    for (label, engine, refinement, evaluations) in [
        ("coordinate", simplify(), four_parents(), 40),
        (
            "lm-archive",
            EngineSelection {
                refine: lm,
                archive: Some(ArchivePolicy::ArchiveOnly),
                simplify: true,
            },
            Refinement {
                top_k: 1,
                trials_per_program: 12,
            },
            60,
        ),
    ] {
        let make = |name: &str| {
            run_options(
                &temp.path().join(format!("{label}-{name}")),
                evaluations,
                refinement,
                engine.clone(),
            )
        };
        let (a, b) = (make("a"), make("b"));
        let receipt = run::run(&a).unwrap();
        assert_eq!(receipt, run::run(&b).unwrap(), "{label}");
        for relative in SIDECARS.into_iter().chain([PATH]) {
            assert_eq!(read(&a, relative), read(&b, relative), "{label} {relative}");
        }
        if engine.archive.is_some() {
            assert_eq!(read(&a, archive::PATH), read(&b, archive::PATH), "{label}");
        }
        assert_eq!(
            publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
            receipt,
            "{label}"
        );
        // The pass measured children, each scoring exactly as its parent, and the shared
        // identity holds.
        assert!(
            check_exact(&a) > 0,
            "{label}: no child was measured: {:?}",
            published_moves(&a)
                .iter()
                .map(|m| (m.round, m.outcome.as_str()))
                .collect::<Vec<_>>()
        );
        let s = &receipt.search;
        assert_eq!(
            s.completed_evaluations,
            s.candidate_measurements + s.objective_trials
        );
        let record = s.engine_selection.as_ref().unwrap().moves.as_ref().unwrap();
        assert_eq!(record.ids, [REWRITE_SATURATE]);
        assert_eq!(
            record.sha256.as_deref(),
            Some(crate::sha256_hex(&read(&a, PATH)).as_str())
        );
        lint(&String::from_utf8(read(&a, PATH)).unwrap());
        lint(&serde_json::to_string(record).unwrap());
    }
}

/// The seeded `cli_tests` run's search sidecars (prepared, search, observations, refinements),
/// pinned by `default_runs_keep_the_pre_wiring_search_sidecars` at the commit before the first
/// engine method was wired, and repeated here for this flag.
const SEEDED_PINS: [&str; 4] = [
    "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
    "b2444c5c34dee80c91aa52d19a4a34aeaa358bf68dd3919723f52463fb2c5420",
    "ad1e6ab3e218f237120668e39321daf2a60cc43b5a503011f1a05ecbbb907767",
    "a6e3af0c1a0b0e85bfe64e3d8bd5c869e6a58b890403ba3438efba144e12a4d7",
];

#[test]
fn with_the_flag_off_nothing_is_selected_called_or_written() {
    let temp = tempfile::tempdir().unwrap();
    let (_, profile) = flags(&[
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
    ]);
    assert!(!profile.rewrite_simplify);
    assert!(!EngineSelection::default().simplify);
    assert_eq!(EngineSelection::default().record(), None);
    // The seeded `cli_tests` run, without the flag, keeps its pinned bytes and writes nothing new.
    let seeded = run_options(
        &temp.path().join("seeded"),
        28,
        Refinement {
            top_k: 1,
            trials_per_program: 4,
        },
        EngineSelection::default(),
    );
    let receipt = run::run(&seeded).unwrap();
    for (relative, pin) in [
        "prepared/prepared.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        "search/cut-0/refinements.json",
    ]
    .into_iter()
    .zip(SEEDED_PINS)
    {
        assert_eq!(
            crate::sha256_hex(&read(&seeded, relative)),
            pin,
            "{relative}"
        );
    }
    assert!(receipt.search.engine_selection.is_none());
    let plain = run_options(
        &temp.path().join("plain"),
        40,
        four_parents(),
        EngineSelection::default(),
    );
    run::run(&plain).unwrap();
    for opt in [&seeded, &plain] {
        assert!(!opt.run.output.join(PATH).exists());
        for relative in ["receipt.json", "search/cut-0/search.json"] {
            assert!(!String::from_utf8(read(opt, relative))
                .unwrap()
                .contains("moves"));
        }
        assert!(published_measurements(opt)
            .iter()
            .all(|m| m.observation.origin != crate::kernel2::SIMPLIFY));
        publication::verify_run(&opt.run.output.join("receipt.json")).unwrap();
    }
    // With the flag on, every measurement before the first rewrite child is the same as off:
    // the pass consumes no random stream, and nothing runs before it.
    let on = run_options(&temp.path().join("on"), 40, four_parents(), simplify());
    run::run(&on).unwrap();
    let (off, with) = (published_measurements(&plain), published_measurements(&on));
    let first = with
        .iter()
        .position(|m| m.observation.origin == crate::kernel2::SIMPLIFY)
        .unwrap();
    assert!(first > 0);
    assert_eq!(off[..first], with[..first]);
    assert_eq!(
        read(&plain, "prepared/prepared.json"),
        read(&on, "prepared/prepared.json")
    );
}

/// The run's CLI flags, parsed as `jope pipeline run` parses them.
fn flags(extra: &[&str]) -> (crate::cli::SearchArgs, crate::kernel2::cli::ProfileArgs) {
    let mut argv = vec![
        "jope-pipeline",
        "run",
        "--dataset",
        "unresolved",
        "--output",
        "unused",
    ];
    argv.extend_from_slice(extra);
    let Cmd::Run {
        search, profile, ..
    } = Cli::try_parse_from(argv).unwrap().cmd
    else {
        panic!("run expected");
    };
    (search, profile)
}

/// Rewrites a run's move records (when given) and its selection record, rehashing search.json
/// and the receipt, so that only the move checks can refuse the change.
fn rewrite(
    output: &Path,
    receipt: &Receipt,
    moves: Option<&[u8]>,
    engine: Option<EngineRecord>,
) -> PathBuf {
    if let Some(bytes) = moves {
        std::fs::write(output.join(PATH), bytes).unwrap();
    }
    let mut edited = receipt.clone();
    edited.search.engine_selection = engine;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    path
}

fn digested(engine: &EngineRecord, bytes: &[u8]) -> EngineRecord {
    let mut engine = engine.clone();
    engine.moves.as_mut().unwrap().sha256 = Some(crate::sha256_hex(bytes));
    engine
}

/// One named edit of the measured record at `first`; `other` is another measured program's
/// sha256.
fn tamper(moves: &mut Vec<MoveRecord>, first: usize, label: &str, other: &str) {
    if label == "record removed" {
        moves.remove(first);
        return;
    }
    let m = &mut moves[first];
    match label {
        "child program" => m.program = Some(SINE.into()),
        "stage expression" => m.rewrites[0].expression = Some("c1".into()),
        "stop reason" => m.rewrites[0].stop = "node_limit".into(),
        "iterations" => m.rewrites[0].iterations += 1,
        "enodes" => m.rewrites[0].enodes += 1,
        "applications" => m.rewrites[0].applications += 1,
        "peak" => m.rewrites[0].peak += 1,
        "refusal" => m.rewrites[0].refusal = Some("refused".into()),
        "outcome" => {
            m.outcome = DUPLICATE.into();
            m.measurement = None;
        }
        "measurement index" => m.measurement = Some(0),
        "round" => m.round += 1,
        "parent" => m.parent_sha256 = other.into(),
        "method id" => m.id = "feat-math.rewrite-saturate.v2".into(),
        _ => unreachable!("{label}"),
    }
}

#[test]
fn verify_refuses_every_tampered_move_record_digest_or_selection() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = ["receipt.json", "search/cut-0/search.json", PATH]
        .iter()
        .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
        .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    // The rewrite itself keeps a valid run.
    rewrite(
        output,
        &receipt,
        Some(originals[2].1.as_slice()),
        Some(engine.clone()),
    );
    publication::verify_run(&path).unwrap();
    restore();

    let first = moves.iter().position(|m| m.outcome == MEASURED).unwrap();
    // The bootstrap program: measured, with no expression stage to rewrite.
    let other = crate::sha256_hex(crate::search::BOOTSTRAP_GENOME.as_bytes());
    assert!(published_measurements(&opt)
        .iter()
        .any(|m| m.observation.program_sha256 == other
            && m.observation.status == crate::search::MEASURED));
    let mut errors = Vec::new();
    for label in [
        "child program",
        "stage expression",
        "stop reason",
        "iterations",
        "enodes",
        "applications",
        "peak",
        "refusal",
        "outcome",
        "measurement index",
        "round",
        "parent",
        "record removed",
        "method id",
    ] {
        let mut edited = moves.clone();
        tamper(&mut edited, first, label, &other);
        assert_ne!(edited, moves, "{label}: the edit was not applied");
        let bytes = serde_json::to_vec_pretty(&edited).unwrap();
        let path = rewrite(
            output,
            &receipt,
            Some(bytes.as_slice()),
            Some(digested(&engine, &bytes)),
        );
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    // Record edits, the file unchanged.
    let mut records: Vec<(&str, Option<EngineRecord>)> = Vec::new();
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = Some("0".repeat(64));
    records.push(("digest", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = None;
    records.push(("digest dropped", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = Some("XYZ".into());
    records.push(("malformed digest", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids = vec!["feat-math.rewrite-saturate.v2".into()];
    records.push(("record id", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids.clear();
    records.push(("no ids", Some(record)));
    let mut record = engine.clone();
    record
        .moves
        .as_mut()
        .unwrap()
        .ids
        .push(REWRITE_SATURATE.into());
    records.push(("repeated id", Some(record)));
    records.push(("record dropped, file kept", None));
    for (label, record) in records {
        assert_ne!(record.as_ref(), Some(&engine), "{label}");
        let path = rewrite(output, &receipt, None, record);
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    // The file edited under its unchanged digest, then removed.
    let mut bytes = originals[2].1.clone();
    bytes.push(b'\n');
    std::fs::write(output.join(PATH), &bytes).unwrap();
    errors.push(format!("{:#}", publication::verify_run(&path).unwrap_err()));
    restore();
    std::fs::remove_file(output.join(PATH)).unwrap();
    errors.push(format!("{:#}", publication::verify_run(&path).unwrap_err()));
    restore();
    publication::verify_run(&path).unwrap();
    // The refusals pass the shipped-text lint, with the run's random temporary path left out.
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn the_selection_record_round_trips_and_the_flag_is_checked_and_lint_clean() {
    let record = simplify().record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"moves": {"ids": [REWRITE_SATURATE]}})
    );
    assert_eq!(record.selection().unwrap(), simplify());
    let mut published = record.clone();
    published.moves.as_mut().unwrap().sha256 = Some("a".repeat(64));
    assert_eq!(published.selection().unwrap(), simplify());
    let all = EngineSelection {
        refine: RefineMethod::Lm(LmOptions::default()),
        archive: Some(ArchivePolicy::ArchiveOnly),
        simplify: true,
    };
    let record = all.record().unwrap();
    assert!(record.damped_ls().is_some() && record.archive.is_some() && record.moves.is_some());
    assert_eq!(record.selection().unwrap(), all);
    for bad in [
        serde_json::json!({"moves": {"ids": ["other"]}}),
        serde_json::json!({"moves": {"ids": []}}),
        serde_json::json!({"moves": {"ids": [REWRITE_SATURATE, REWRITE_SATURATE]}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<MovesRecord>(
        serde_json::json!({"ids": [REWRITE_SATURATE], "rounds": 3})
    )
    .is_err());
    // The pass reads `--refine-top-k` parents, so it needs at least one.
    let none = Refinement {
        top_k: 0,
        trials_per_program: 4,
    };
    assert!(simplify().validate(none).is_err());
    assert!(simplify().validate(four_parents()).is_ok());
    let e = evaluator().with_engine(simplify()).unwrap();
    assert_eq!(e.engine(), &simplify());
    assert_eq!(e.engine_record(), simplify().record());
    // The CLI: kernel2 only, and refused without a parent.
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let parse = |top_k: &str| {
        let (mut search, profile) = flags(&[
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            top_k,
            "--refine-trials",
            "4",
            "--rewrite-simplify",
            "--source-commit",
            source.as_str(),
        ]);
        assert!(profile.rewrite_simplify);
        search.dataset = dataset();
        (search, profile)
    };
    let (search, profile) = parse("1");
    assert!(profile
        .options(&search, temp.path().join("one"), false)
        .unwrap()
        .is_some());
    let (search, profile) = parse("0");
    let error = profile
        .options(&search, temp.path().join("zero"), false)
        .unwrap_err();
    assert!(format!("{error:#}").contains("--refine-top-k"), "{error:#}");
    lint(&format!("{error:#}"));
    let (mut legacy_search, legacy) = flags(&["--rewrite-simplify"]);
    legacy_search.dataset = dataset();
    let output = temp.path().join("refused");
    let error = legacy
        .options(&legacy_search, output.clone(), false)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("legacy search refuses kernel2"),
        "{error:#}"
    );
    assert!(!output.exists());
    // Shipped text: the id, the path, the outcomes, the flag's help and the internal note.
    for text in [
        REWRITE_SATURATE,
        PATH,
        UNCHANGED,
        REFUSED,
        DUPLICATE,
        NOT_STARTED,
        MEASURED,
        crate::kernel2::SIMPLIFY,
    ] {
        lint(text);
    }
    lint(include_str!(
        "../../../../../../docs/internal/kernel2/b20-rewrite-simplify.md"
    ));
    use clap::CommandFactory;
    let mut command = Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "rewrite_simplify")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(help.contains("moves.json"), "{help}");
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
}
