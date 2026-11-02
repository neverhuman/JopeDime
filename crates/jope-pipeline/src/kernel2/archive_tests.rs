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
        simplify: false,
        linearize: false,
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

#[cfg(test)]
#[path = "archive_tests/behavior.rs"]
mod behavior;

#[cfg(test)]
#[path = "archive_tests/receipts.rs"]
mod receipts;

#[cfg(test)]
#[path = "archive_tests/cli.rs"]
mod cli;
