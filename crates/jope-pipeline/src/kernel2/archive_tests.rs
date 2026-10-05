//! The opt-in loss × cost front, published as evidence only (laneD:B18-pareto-archive).
use super::{
    archive::{self, program_cost},
    cli::ProfileArgs,
    publication,
    receipt::{self, Receipt},
    records::Measurement,
    run,
    search::Refinement,
    tests::{data, evaluator, measured},
    ArchivePolicy, ArchiveRecord, EngineRecord, EngineSelection, Kernel2Evaluator,
    Kernel2Observation, RefineMethod, FRONT_LOSS_COST,
};
use crate::{
    cli::{Cli, Cmd},
    clock::Clock,
    emit::Language,
    run::{Policy, RunOptions},
    search::Budget,
};
use clap::Parser;
use feat_math::{
    dsl::ParamValue,
    evolve::FaultPolicy,
    kernel::{
        expression_search::LmOptions,
        pareto::{self, Archive, Candidate, Member},
        Program,
    },
    math_expr::{Expr, MathExpr},
    GenomeIr,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The seed program of the `cli_tests` runs.
const SEED: &str = "(pipe X (derive expr=\"1*c0\") (ridge))";
/// A program with one expression stage, of cost 6.
const SEED_SINE: &str = "(pipe X (derive expr=\"sin(0.5*c0)\") (ridge))";

/// The published files of a run with the archive selected.
const SIDECARS: [&str; 8] = [
    "receipt.json",
    "prepared/prepared.json",
    "search/cut-0/search.json",
    "search/cut-0/observations.json",
    "search/cut-0/refinements.json",
    "search/cut-0/archive.json",
    "emit/model.rs",
    "emit/manifest.json",
];

fn penalize(tie_break: bool) -> FaultPolicy {
    FaultPolicy::Penalize {
        lambda_domain: 0.05,
        lambda_saturation: 0.0,
        tie_break,
    }
}

fn selection(refine: RefineMethod, archive: bool) -> EngineSelection {
    EngineSelection {
        refine,
        archive: archive.then_some(ArchivePolicy::ArchiveOnly),
    }
}

fn lm() -> RefineMethod {
    RefineMethod::Lm(LmOptions::default())
}

/// The `cli_tests` dataset.
fn dataset() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear")
        .display()
        .to_string()
}

/// The `cli_tests` run (pmlb-229-pwlinear, rng seed 7, a frozen clock, Rust only, [`SEED`] as
/// the seeds file beside the output) in `dir/run`, with the given budget, profile, policy, trials
/// per program and engine selection. The default selection is left unapplied, as the CLI does.
fn opts(
    dir: &Path,
    evaluations: usize,
    profile: &str,
    policy: FaultPolicy,
    trials: usize,
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
        profile,
        policy,
        Refinement {
            top_k: 1,
            trials_per_program: trials,
        },
    )
    .unwrap();
    if engine == EngineSelection::default() {
        options
    } else {
        options.with_engine(engine).unwrap()
    }
}

/// The same run on another committed test dataset, without the seeds file. The
/// `pmlb-229-pwlinear` runs probed had one-program fronts (a single-stage head won); at rng seed 7
/// and 24 evaluations `pmlb-first-principles-supernovae-zg` has a two-program front whose winner
/// is the costlier member, with an expression stage, and `rdatasets-wooldridge-barium` a
/// three-program front.
fn on_dataset(mut opt: run::Options, name: &str) -> run::Options {
    opt.run.dataset = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test")
        .join(name)
        .display()
        .to_string();
    opt.run.seed_genomes = None;
    opt
}

/// A baseline run with the archive selected, at 24 evaluations on `dataset`.
fn archived_on(dir: &Path, dataset: &str) -> run::Options {
    on_dataset(
        opts(
            dir,
            24,
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            4,
            selection(RefineMethod::Coordinate, true),
        ),
        dataset,
    )
}

/// The two-program front with an expression stage.
const SUPERNOVAE: &str = "pmlb-first-principles-supernovae-zg";
/// The three-program front.
const BARIUM: &str = "rdatasets-wooldridge-barium";

fn read(opt: &run::Options, relative: &str) -> Vec<u8> {
    std::fs::read(opt.run.output.join(relative)).unwrap()
}

/// The published archive, read back.
fn published(opt: &run::Options) -> Archive {
    Archive::from_json(std::str::from_utf8(&read(opt, archive::PATH)).unwrap()).unwrap()
}

/// The published observations, re-derived under `policy` as `verify_run` does.
fn derived(opt: &run::Options, policy: FaultPolicy) -> Vec<Kernel2Observation> {
    let records: Vec<Measurement> =
        serde_json::from_slice(&read(opt, "search/cut-0/observations.json")).unwrap();
    records.iter().map(|r| r.derive(policy).unwrap()).collect()
}

/// The sha256 of a program's exact codec bytes.
fn key(genome: &str) -> [u8; 32] {
    let bytes = Program::import_legacy(&GenomeIr::parse(genome).unwrap())
        .to_bytes()
        .unwrap();
    Sha256::digest(&bytes).into()
}

/// The pipeline cost, counted recursively here, independently of `archive::program_cost`.
fn cost(genome: &str) -> u32 {
    fn count(expr: &Expr) -> u32 {
        1 + match expr {
            Expr::Const(_) | Expr::Var(_) => 0,
            Expr::Unary(_, child) => count(child),
            Expr::Binary(_, a, b) => count(a) + count(b),
            Expr::Call(_, args) => args.iter().map(count).sum(),
        }
    }
    let GenomeIr::Pipeline(pipe) = GenomeIr::parse(genome).unwrap() else {
        panic!("{genome} is not a pipeline");
    };
    pipe.stages
        .iter()
        .map(|stage| match stage.params.get("expr") {
            Some(ParamValue::Str(source)) if matches!(stage.op.as_str(), "map" | "derive") => {
                1 + count(MathExpr::parse(source).unwrap().ast())
            }
            _ => 1,
        })
        .sum()
}

/// A ranked observation as a member, re-derived with the test's own cost.
fn member_of(o: &Kernel2Observation) -> Member {
    Member::admit(&Candidate {
        loss: -o.selection_score.unwrap(),
        complexity: cost(&o.observation.genome),
        age: None,
        faulted_cells: o.faults.faulted_cells(),
        hard_fault: o.faults.is_hard(),
        key: key(&o.observation.genome),
    })
    .unwrap()
}

/// Every member re-derives from a ranked observation under the recorded cost version, no member
/// dominates another, the members are in front order, and every ranked observation is covered by
/// a member at least as good in both objectives.
fn check_front(archive: &Archive, observations: &[Kernel2Observation]) {
    assert_eq!(archive.cost_version(), pareto::PIPELINE_COST_VERSION);
    let entries: Vec<(Member, &str)> = observations
        .iter()
        .filter(|o| archive::ranked(o))
        .map(|o| (member_of(o), o.observation.genome.as_str()))
        .collect();
    let members = archive.members();
    assert!(!members.is_empty());
    for (i, m) in members.iter().enumerate() {
        assert!(
            entries.iter().any(|(e, _)| e == m),
            "member {m:?} re-derives from no ranked observation"
        );
        for other in members {
            assert!(
                !pareto::dominates(&other.objectives(false), &m.objectives(false)),
                "{other:?} dominates {m:?}"
            );
        }
        if let Some(next) = members.get(i + 1) {
            assert!(next.complexity > m.complexity && next.loss() < m.loss());
        }
    }
    for (e, genome) in &entries {
        assert!(
            members
                .iter()
                .any(|m| m.loss() <= e.loss() && m.complexity <= e.complexity),
            "{genome} is not covered"
        );
    }
}

/// Ranking's winner is on the front, or the front holds its loss at a simpler program, or at the
/// same cost with fewer faulted cells or a lower key; nothing on the front has a lower loss.
fn check_winner(archive: &Archive, best: &Kernel2Observation) {
    let winner = member_of(best);
    assert!(archive.members().iter().all(|m| m.loss() >= winner.loss()));
    let holder = archive
        .members()
        .iter()
        .find(|m| m.loss() == winner.loss())
        .expect("the winner's loss is on the front");
    assert!(holder.complexity <= winner.complexity);
    if holder.key != winner.key {
        assert!(
            holder.complexity < winner.complexity
                || (holder.faulted_cells, holder.key) < (winner.faulted_cells, winner.key),
            "{holder:?} holds the winner's place {winner:?}"
        );
    }
}

/// The search summary without the archive record: the record, and the selection record when
/// nothing else is left in it.
fn without_archive(summary: &[u8]) -> Value {
    let mut value: Value = serde_json::from_slice(summary).unwrap();
    let fields = value.as_object_mut().unwrap();
    if let Some(engine) = fields.get_mut("engine_selection") {
        engine.as_object_mut().unwrap().remove("archive");
        if engine.as_object().unwrap().is_empty() {
            fields.remove("engine_selection");
        }
    }
    value
}

#[test]
fn archive_on_and_off_keep_every_measurement_refinement_and_the_winner_bit_for_bit() {
    let temp = tempfile::tempdir().unwrap();
    for (label, refine, trials) in [
        ("coordinate", RefineMethod::Coordinate, 4),
        ("lm", lm(), 12),
    ] {
        let run_with = |archive: bool| {
            let dir = temp.path().join(format!("{label}-{archive}"));
            let opt = opts(
                &dir,
                40,
                "kernel2-baseline",
                FaultPolicy::RejectHard,
                trials,
                selection(refine, archive),
            );
            let receipt = run::run(&opt).unwrap();
            assert_eq!(
                publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
                receipt,
                "{label} {archive}"
            );
            (opt, receipt)
        };
        let (off, plain) = run_with(false);
        let (on, archived) = run_with(true);
        for relative in [
            "prepared/prepared.json",
            "search/cut-0/observations.json",
            "search/cut-0/refinements.json",
            "emit/model.rs",
            "emit/manifest.json",
        ] {
            assert_eq!(
                read(&off, relative),
                read(&on, relative),
                "{label} {relative}"
            );
        }
        assert_eq!(plain.search.best, archived.search.best, "{label}");
        assert_eq!(plain.artifacts, archived.artifacts, "{label}");
        // The summaries differ only by the archive record.
        assert_eq!(
            without_archive(&read(&off, "search/cut-0/search.json")),
            without_archive(&read(&on, "search/cut-0/search.json")),
            "{label}"
        );
        // So do the receipts: without the archive record and the summary's digest they are equal.
        let mut expected = archived.clone();
        expected.search.engine_selection = expected
            .search
            .engine_selection
            .filter(|e| e.refine.is_some())
            .map(|e| EngineRecord { archive: None, ..e });
        expected.search_sha256 = plain.search_sha256.clone();
        assert_eq!(expected, plain, "{label}");
        // Off: no sidecar and no record. On: the sidecar and its digest.
        assert!(!off.run.output.join(archive::PATH).exists());
        assert!(!String::from_utf8(read(&off, "search/cut-0/search.json"))
            .unwrap()
            .contains("archive"));
        assert!(!String::from_utf8(read(&off, "receipt.json"))
            .unwrap()
            .contains("archive"));
        let record = archived
            .search
            .engine_selection
            .as_ref()
            .and_then(|e| e.archive.as_ref())
            .unwrap();
        assert_eq!(record.id, FRONT_LOSS_COST);
        assert_eq!(record.cost_version, pareto::PIPELINE_COST_VERSION);
        assert_eq!(
            record.sha256.as_deref(),
            Some(crate::sha256_hex(&read(&on, archive::PATH)).as_str())
        );
    }
}

#[test]
fn every_member_is_non_dominated_and_the_winner_is_on_the_front_or_an_equal_loss_tie() {
    let temp = tempfile::tempdir().unwrap();
    for (label, profile, policy, refine, trials) in [
        (
            "baseline",
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            RefineMethod::Coordinate,
            4,
        ),
        (
            "experimental",
            "kernel2-experimental",
            penalize(false),
            RefineMethod::Coordinate,
            4,
        ),
        ("lm", "kernel2-baseline", FaultPolicy::RejectHard, lm(), 12),
    ] {
        let opt = opts(
            &temp.path().join(label),
            40,
            profile,
            policy,
            trials,
            selection(refine, true),
        );
        check_run(label, &opt, policy);
    }
    // Fronts of two and three programs, the winner the costliest member.
    for (dataset, size) in [(SUPERNOVAE, 2), (BARIUM, 3)] {
        let opt = archived_on(&temp.path().join(dataset), dataset);
        let archive = check_run(dataset, &opt, FaultPolicy::RejectHard);
        assert_eq!(archive.members().len(), size, "{dataset}");
    }
}

/// Runs `opt`, then checks its published archive against its re-derived observations and its
/// winner. The ranked observations hold more programs than the front.
fn check_run(label: &str, opt: &run::Options, policy: FaultPolicy) -> Archive {
    let receipt = run::run(opt).unwrap();
    let archive = published(opt);
    let observations = derived(opt, policy);
    check_front(&archive, &observations);
    let best = receipt.search.best.derive(policy).unwrap();
    check_winner(&archive, &best);
    assert!(
        observations.iter().filter(|o| archive::ranked(o)).count() > archive.members().len(),
        "{label}"
    );
    archive
}

/// Measured programs of seven costs with set scores: the front keeps exactly the four that no
/// simpler program matches, the winner among them.
#[test]
fn a_constructed_front_keeps_exactly_the_programs_no_simpler_one_matches() {
    let e = evaluator();
    let c = data();
    let two_terms = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c0)\") (ridge))";
    let scaled = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c0)\") (standardize) (ridge))";
    let mixed =
        "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))";
    let programs = [
        ("(pipe X (ridge))", 1, 0.10),
        ("(pipe X (standardize) (ridge))", 2, 0.05),
        (SEED, 5, 0.30),
        (SEED_SINE, 6, 0.30),
        (two_terms, 10, 0.60),
        (mixed, 11, 0.55),
        (scaled, 11, 0.90),
    ];
    let observations: Vec<Kernel2Observation> = programs
        .iter()
        .map(|&(genome, expected, score)| {
            assert_eq!(cost(genome), expected, "{genome}");
            let mut o = measured(&e, genome, &c);
            assert!(o.observation.measured(), "{genome}: {:?}", o.observation);
            o.selection_score = Some(score);
            o
        })
        .collect();
    let archive = archive::build(&observations).unwrap();
    check_front(&archive, &observations);
    let kept: Vec<(u32, [u8; 32])> = archive
        .members()
        .iter()
        .map(|m| (m.complexity, m.key))
        .collect();
    assert_eq!(
        kept,
        vec![
            (1, key("(pipe X (ridge))")),
            (5, key(SEED)),
            (10, key(two_terms)),
            (11, key(scaled)),
        ]
    );
    let ranked = e.rank(&observations);
    assert_eq!(ranked[0].observation.genome, scaled);
    check_winner(&archive, ranked[0]);
}

#[test]
fn ties_and_duplicates_settle_identically_under_any_observation_order() {
    let e = evaluator();
    let c = data();
    // Equal cost (two stages and three expression nodes) with equal scores: fewer faulted cells,
    // then the lower key, keeps the place.
    let tied = [
        "(pipe X (derive expr=\"1*c0\") (ridge))",
        "(pipe X (derive expr=\"2*c0\") (ridge))",
        "(pipe X (derive expr=\"3*c0\") (ridge))",
    ];
    let mut observations: Vec<Kernel2Observation> = tied
        .iter()
        .map(|g| {
            let mut o = measured(&e, g, &c);
            o.selection_score = Some(0.5);
            o
        })
        .collect();
    assert!(tied.iter().all(|g| cost(g) == 5));
    assert_eq!(
        observations[0].faults.faulted_cells(),
        observations[1].faults.faulted_cells()
    );
    observations[2].faults.domain_cells = 3;
    observations[2].faults.evaluated_cells = observations[2].faults.evaluated_cells.max(3);
    assert!(observations[2].faults.faulted_cells() > 0);
    // A duplicate of each, and two spellings of one program with one codec key.
    observations.extend(observations.clone());
    let spelled = [
        "(pipe X (ridge lambda_frac=0.000000001))",
        "(pipe X (ridge lambda_frac=1e-9))",
    ];
    assert_ne!(spelled[0], spelled[1]);
    assert_eq!(key(spelled[0]), key(spelled[1]));
    for g in spelled {
        let mut o = measured(&e, g, &c);
        o.selection_score = Some(0.25);
        observations.push(o);
    }
    // A better, more complex program.
    let mut best = measured(&e, SEED_SINE, &c);
    best.selection_score = Some(0.75);
    observations.push(best);
    // Every entry is ranked, so each tie and duplicate above reaches the archive.
    assert!(observations.iter().all(archive::ranked));
    let reference = archive::build(&observations).unwrap();
    let json = reference.to_json();
    let keys: Vec<[u8; 32]> = reference.members().iter().map(|m| m.key).collect();
    let low = if key(tied[0]) < key(tied[1]) {
        tied[0]
    } else {
        tied[1]
    };
    assert_eq!(keys, vec![key(spelled[0]), key(low), key(SEED_SINE)]);
    let n = observations.len();
    let mut orders: Vec<Vec<usize>> = vec![(0..n).rev().collect()];
    for shift in 1..n {
        orders.push((0..n).map(|i| (i + shift) % n).collect());
    }
    orders.push((0..n).step_by(2).chain((1..n).step_by(2)).collect());
    orders.push((1..n).step_by(2).chain((0..n).step_by(2)).rev().collect());
    for order in orders {
        let shuffled: Vec<Kernel2Observation> =
            order.iter().map(|&i| observations[i].clone()).collect();
        assert_eq!(
            archive::build(&shuffled).unwrap().to_json(),
            json,
            "{order:?}"
        );
    }
}

#[test]
fn the_archive_input_is_exactly_the_observations_ranking_admits() {
    let c = data();
    for (label, e) in [
        ("reject-hard", evaluator()),
        (
            "penalize",
            evaluator().with_fault_policy(penalize(true)).unwrap(),
        ),
    ] {
        let mut wide = c.clone();
        for row in &mut wide.train_x {
            row.push(row[0]);
        }
        let mut observations = vec![
            measured(&e, "(pipe X (ridge lambda=0.00000001))", &c),
            measured(&e, SEED, &c),
            measured(&e, SEED_SINE, &c),
            // Refused before any score: parse, profile admission, a held-out Hard fault.
            measured(&e, "(broken", &c),
            measured(&e, "(pipe X (map expr=\"exp(x)\") (ridge))", &c),
            measured(
                &e,
                "(pipe X (derive expr=\"c1\" append=false) (ridge))",
                &wide,
            ),
        ];
        assert!(observations[5].faults.is_hard());
        // Measured, but with a score ranking refuses.
        for score in [f64::NAN, f64::INFINITY] {
            let mut o = measured(&e, "(pipe X (derive expr=\"2*c0\") (ridge))", &c);
            o.selection_score = Some(score);
            observations.push(o);
        }
        let ranked: Vec<Kernel2Observation> = e.rank(&observations).into_iter().cloned().collect();
        let filtered: Vec<&Kernel2Observation> =
            observations.iter().filter(|o| archive::ranked(o)).collect();
        let names = |list: Vec<&Kernel2Observation>| {
            let mut names: Vec<String> =
                list.iter().map(|o| o.observation.genome.clone()).collect();
            names.sort();
            names
        };
        assert_eq!(names(filtered), names(ranked.iter().collect()), "{label}");
        assert_eq!(ranked.len(), 3, "{label}");
        let all = archive::build(&observations).unwrap();
        assert_eq!(all, archive::build(&ranked).unwrap(), "{label}");
        // Whatever the refused observations hold, the archive is the same.
        let mut edited = observations.clone();
        for o in edited.iter_mut().skip(3) {
            o.observation.genome = "(pipe X (standardize) (ridge))".into();
            o.faults.domain_cells += 1;
        }
        assert_eq!(archive::build(&edited).unwrap(), all, "{label}");
    }
}

#[test]
fn program_cost_counts_every_stage_and_expression_node() {
    for (genome, expected) in [
        ("(pipe X (ridge))", 1),
        ("(pipe X (standardize) (ridge))", 2),
        (SEED, 5),
        (
            "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))",
            6,
        ),
        (
            "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))",
            11,
        ),
        ("(pipe X (derive expr=\"(-0.5)*c0\") (ridge))", 6),
        (
            // Three stages; nine nodes: +, clip, c0, the negation and its 1.0, 1.0, *, 0.3, c1.
            "(pipe X (derive expr=\"clip(c0,(-1.0),1.0)+(0.3*c1)\") (standardize) (ridge))",
            12,
        ),
    ] {
        let parsed = GenomeIr::parse(genome).unwrap();
        assert_eq!(program_cost(&parsed).unwrap(), expected, "{genome}");
        assert_eq!(cost(genome), expected, "{genome}");
    }
}

#[test]
fn archive_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    for (label, refine, trials) in [
        ("coordinate", RefineMethod::Coordinate, 4),
        ("lm", lm(), 12),
    ] {
        let make = |name: &str| {
            opts(
                &temp.path().join(format!("{label}-{name}")),
                40,
                "kernel2-baseline",
                FaultPolicy::RejectHard,
                trials,
                selection(refine, true),
            )
        };
        let (a, b) = (make("a"), make("b"));
        let receipt = run::run(&a).unwrap();
        assert_eq!(receipt, run::run(&b).unwrap(), "{label}");
        for relative in SIDECARS {
            assert_eq!(read(&a, relative), read(&b, relative), "{label} {relative}");
        }
        assert_eq!(
            publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
            receipt
        );
    }
}

/// Rewrites a run's archive (when given) and its selection record, rehashing search.json and the
/// receipt, so that only the archive checks can refuse the change.
fn rewrite(
    output: &Path,
    receipt: &Receipt,
    archive: Option<&[u8]>,
    engine: Option<EngineRecord>,
) -> PathBuf {
    if let Some(bytes) = archive {
        std::fs::write(output.join(archive::PATH), bytes).unwrap();
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

/// The run's record with the archive digest set to that of `bytes`.
fn digested(engine: &EngineRecord, bytes: &[u8]) -> EngineRecord {
    let mut engine = engine.clone();
    engine.archive.as_mut().unwrap().sha256 = Some(crate::sha256_hex(bytes));
    engine
}

#[test]
fn the_archive_round_trips_and_verify_refuses_every_tampered_archive_or_record() {
    let temp = tempfile::tempdir().unwrap();
    let opt = archived_on(temp.path(), SUPERNOVAE);
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> =
        ["receipt.json", "search/cut-0/search.json", archive::PATH]
            .iter()
            .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
            .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    // Round trip: the reader takes the published bytes back to themselves.
    let bytes = originals[2].1.clone();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let back = Archive::from_json(&text).unwrap();
    assert_eq!(back.to_json().as_bytes(), bytes.as_slice());
    assert!(
        !back.members().is_empty(),
        "the fixture has a front to edit"
    );
    let engine = receipt.search.engine_selection.clone().unwrap();
    // The rewrite itself keeps a valid run.
    rewrite(output, &receipt, Some(&bytes), Some(engine.clone()));
    publication::verify_run(&path).unwrap();
    restore();

    let hex = |bits: u64| format!("{bits:016x}");
    let key_hex = |key: &[u8; 32]| key.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let members = back.members();
    let last = members.last().unwrap();
    let mut edits: Vec<(&str, String)> = Vec::new();
    // A lower loss on the last member keeps the front in order: only re-derivation refuses it.
    edits.push((
        "member loss",
        text.replacen(&hex(last.loss), &hex((last.loss() - 0.001).to_bits()), 1),
    ));
    let mut other = members[0].key;
    other[0] ^= 0x01;
    edits.push((
        "member key",
        text.replacen(&key_hex(&members[0].key), &key_hex(&other), 1),
    ));
    edits.push((
        "member complexity",
        text.replacen(
            &format!("\"complexity\":{}", last.complexity),
            &format!("\"complexity\":{}", last.complexity + 1),
            1,
        ),
    ));
    edits.push((
        "cost version",
        text.replace(pareto::PIPELINE_COST_VERSION, pareto::COST_VERSION),
    ));
    // The front without its simplest member is still a front (possibly empty): only re-derivation
    // refuses it.
    let kept: Vec<Candidate> = members[1..]
        .iter()
        .map(|m| Candidate {
            loss: m.loss(),
            complexity: m.complexity,
            age: m.age,
            faulted_cells: m.faulted_cells,
            hard_fault: false,
            key: m.key,
        })
        .collect();
    let mut fewer = Archive::with_cost_version(pareto::PIPELINE_COST_VERSION).unwrap();
    fewer.update(&kept);
    edits.push(("member removed", fewer.to_json()));
    // A dominated member appended: the reader refuses it.
    edits.push((
        "dominated member",
        text.replacen(
            "]}",
            &format!(
                ",{{\"loss\":\"{}\",\"complexity\":{},\"age\":null,\"faulted_cells\":0,\"key\":\"{}\"}}]}}",
                hex((last.loss() + 1.0).to_bits()),
                last.complexity + 1,
                "0".repeat(64)
            ),
            1,
        ),
    ));
    let mut errors = Vec::new();
    for (label, edited) in &edits {
        assert_ne!(
            edited.as_bytes(),
            bytes.as_slice(),
            "{label}: the edit was not applied"
        );
        let path = rewrite(
            output,
            &receipt,
            Some(edited.as_bytes()),
            Some(digested(&engine, edited.as_bytes())),
        );
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{error:#}"));
        restore();
    }
    // Record edits, the archive bytes unchanged.
    let mut records: Vec<(&str, Option<EngineRecord>)> = Vec::new();
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().sha256 = Some("0".repeat(64));
    records.push(("digest", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().sha256 = None;
    records.push(("digest dropped", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().cost_version = pareto::COST_VERSION.into();
    records.push(("record cost version", Some(record)));
    let mut record = engine.clone();
    record.archive.as_mut().unwrap().id = "feat-math.front-loss-cost.v2".into();
    records.push(("record id", Some(record)));
    records.push(("record dropped, file kept", None));
    for (label, record) in records {
        assert_ne!(
            record.as_ref(),
            Some(&engine),
            "{label}: the edit was not applied"
        );
        let path = rewrite(output, &receipt, None, record);
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{error:#}"));
        restore();
    }
    // The file removed under its record.
    std::fs::remove_file(output.join(archive::PATH)).unwrap();
    assert!(publication::verify_run(&path).is_err());
    restore();
    publication::verify_run(&path).unwrap();
    // The refusals pass the shipped-text lint, with the run's random temporary path left out.
    let root = output.display().to_string();
    for error in errors {
        let text = error.replace(&root, "<run>");
        let hits = feat_math::transpile::lint_shipped_text(&text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    }
}

#[test]
fn the_selection_record_round_trips_and_unknown_records_are_refused() {
    let archived = selection(RefineMethod::Coordinate, true);
    let record = archived.record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": pareto::PIPELINE_COST_VERSION}})
    );
    assert_eq!(record.selection().unwrap(), archived);
    // A published record carries the digest; the selection is the same.
    let mut published = record.clone();
    published.archive.as_mut().unwrap().sha256 = Some("a".repeat(64));
    assert_eq!(published.selection().unwrap(), archived);
    let both = selection(lm(), true);
    let record = both.record().unwrap();
    assert!(record.damped_ls().is_some() && record.archive.is_some());
    assert_eq!(record.selection().unwrap(), both);
    assert_eq!(selection(RefineMethod::Coordinate, false).record(), None);
    for bad in [
        serde_json::json!({"archive": {"id": "other", "cost_version": pareto::PIPELINE_COST_VERSION}}),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": pareto::COST_VERSION}}),
        serde_json::json!({"archive": {"id": FRONT_LOSS_COST, "cost_version": "feat-math.cost.pipeline-nodes.v2"}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<ArchiveRecord>(serde_json::json!(
        {"id": FRONT_LOSS_COST, "cost_version": pareto::PIPELINE_COST_VERSION, "members": 3}
    ))
    .is_err());
    // The selection keeps every other check: an archive alongside an invalid refinement method.
    let invalid = EngineSelection {
        refine: RefineMethod::Lm(LmOptions {
            tau: f64::NAN,
            ..LmOptions::default()
        }),
        archive: Some(ArchivePolicy::ArchiveOnly),
    };
    assert!(evaluator().with_engine(invalid).is_err());
    let e: Kernel2Evaluator = evaluator().with_engine(archived.clone()).unwrap();
    assert_eq!(e.engine(), &archived);
    assert_eq!(e.engine_record(), archived.record());
}

/// The run's CLI flags, parsed as `jope pipeline run` parses them.
fn flags(extra: &[&str]) -> (crate::cli::SearchArgs, ProfileArgs) {
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

#[test]
fn the_flag_selects_the_archive_on_kernel2_only_and_its_text_passes_the_lint() {
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let kernel2 = [
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
        "--budget",
        "evaluations=16,seconds=300",
        "--policy",
        "skip",
        "--emit",
        "rust",
        "--rng-seed",
        "7",
        "--source-commit",
        source.as_str(),
    ];
    let make = |extra: &[&str], output: PathBuf| {
        let mut argv = kernel2.to_vec();
        argv.extend_from_slice(extra);
        let (mut search, profile) = flags(&argv);
        search.dataset = dataset();
        let mut opt = profile.options(&search, output, false).unwrap().unwrap();
        opt.run.clock = Clock::Frozen;
        opt
    };
    let plain = make(&[], temp.path().join("plain"));
    let flagged = make(&["--pareto-archive"], temp.path().join("flagged"));
    let (a, b) = (run::run(&plain).unwrap(), run::run(&flagged).unwrap());
    assert!(a.search.engine_selection.is_none());
    assert!(!plain.run.output.join(archive::PATH).exists());
    let record = b.search.engine_selection.as_ref().unwrap();
    assert!(record.refine.is_none() && record.archive.is_some());
    assert_eq!(
        publication::verify_run(&flagged.run.output.join("receipt.json")).unwrap(),
        b
    );
    assert_eq!(a.search.best, b.search.best);
    // With the method too.
    make(
        &["--pareto-archive", "--refine-method", "lm"],
        temp.path().join("lm"),
    );
    // The legacy profile refuses the flag before anything is written.
    let output = temp.path().join("refused");
    let (mut search, profile) = flags(&["--pareto-archive"]);
    search.dataset = dataset();
    let error = profile.options(&search, output.clone(), false).unwrap_err();
    assert!(
        format!("{error:#}").contains("legacy search refuses kernel2"),
        "{error:#}"
    );
    assert!(!output.exists());
    // Shipped text: ids, the cost version, the record, the archive and the flag's help.
    let lint = |text: &str| {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    };
    lint(FRONT_LOSS_COST);
    lint(pareto::PIPELINE_COST_VERSION);
    lint(archive::PATH);
    lint(&serde_json::to_string(record).unwrap());
    lint(&String::from_utf8(read(&flagged, archive::PATH)).unwrap());
    // The internal note that documents the flag, the ids and the record.
    lint(include_str!(
        "../../../../docs/internal/kernel2/b18-pareto-archive.md"
    ));
    use clap::CommandFactory;
    let mut command = Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "pareto_archive")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(help.contains("archive.json"), "{help}");
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
    let record_check = receipt::validate_value(&{
        let mut value = serde_json::to_value(&b).unwrap();
        value["search"]["engine_selection"]["archive"]["sha256"] = "XYZ".into();
        value
    })
    .unwrap_err();
    lint(&format!("{record_check:#}"));
}
