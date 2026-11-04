use super::{cli::ProfileArgs, publication, receipt, run, search::Refinement};

#[path = "cli_tests/tests_01.rs"]
mod cli_tests_01;
#[path = "cli_tests/tests_02.rs"]
mod cli_tests_02;
#[path = "cli_tests/tests_03.rs"]
mod cli_tests_03;

use crate::{
    cli::{Cli, Cmd},
    clock::Clock,
    emit::Language,
    run::{Policy, RunOptions},
    search::Budget,
};
use clap::Parser;
use feat_math::evolve::FaultPolicy;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn options(output: &Path, evaluations: usize) -> run::Options {
    let dataset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear");
    let mut base = RunOptions::new(dataset.display().to_string(), output);
    base.budget = Budget {
        evaluations,
        seconds: 300,
    };
    base.policy = Policy::Skip;
    base.languages = vec![Language::Rust];
    base.rng_seed = 7;
    base.clock = Clock::Frozen;
    base.source_commit = Some("a".repeat(40));
    run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 1,
            trials_per_program: 4,
        },
    )
    .unwrap()
}

fn args(extra: &[&str]) -> (crate::cli::SearchArgs, ProfileArgs) {
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

fn seed_entry(text: &str) -> crate::seeds::SeedGenome {
    crate::seeds::SeedGenome {
        genome: text.into(),
        program_sha256: crate::sha256_hex(text.as_bytes()),
        bank_entry_sha256: None,
        receipt_sha256: None,
    }
}

fn seeded(output: &Path) -> run::Options {
    let mut opt = options(output, 28);
    let path = output.parent().unwrap().join("seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        vec![seed_entry("(pipe X (derive expr=\"1*c0\") (ridge))")],
    );
    std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(path);
    opt
}

// laneD:B17-lm-refinement begin
fn lm_seeded(output: &Path, evaluations: usize) -> run::Options {
    let mut base = seeded(output).run;
    base.budget.evaluations = evaluations;
    run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 1,
            trials_per_program: 12,
        },
    )
    .unwrap()
    .with_engine(super::EngineSelection {
        refine: super::RefineMethod::Lm(feat_math::kernel::expression_search::LmOptions::default()),
        archive: None,
        simplify: false,
        linearize: false,
    })
    .unwrap()
}

const SIDECARS: [&str; 7] = [
    "receipt.json",
    "prepared/prepared.json",
    "search/cut-0/search.json",
    "search/cut-0/observations.json",
    "search/cut-0/refinements.json",
    "emit/model.rs",
    "emit/manifest.json",
];

/// Rewrites one run's refinements sidecar and selection record, rehashing search.json and the
/// receipt, so that only the evidence checks can refuse the change.
fn rewrite_lm_evidence(
    output: &Path,
    receipt: &receipt::Receipt,
    refinements: &Value,
    engine: Option<super::EngineRecord>,
) -> PathBuf {
    let dir = output.join("search/cut-0");
    let bytes = serde_json::to_vec_pretty(refinements).unwrap();
    std::fs::write(dir.join("refinements.json"), &bytes).unwrap();
    let mut edited = receipt.clone();
    edited.search.refinements_sha256 = crate::sha256_hex(&bytes);
    edited.search.engine_selection = engine;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(dir.join("search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    path
}

/// A run through the CLI flags: `trials` per program with `method`, `evaluations` in all, one
/// seed, a frozen clock, and the seeds file beside the output.
fn flag_run(
    dir: &Path,
    seed: &str,
    evaluations: usize,
    trials: usize,
    method: &str,
) -> run::Options {
    std::fs::create_dir_all(dir).unwrap();
    let source = "a".repeat(40);
    let budget = format!("evaluations={evaluations},seconds=300");
    let trials = trials.to_string();
    let (mut search, profile) = args(&[
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        trials.as_str(),
        "--refine-method",
        method,
        "--budget",
        budget.as_str(),
        "--policy",
        "skip",
        "--emit",
        "rust",
        "--rng-seed",
        "7",
        "--source-commit",
        source.as_str(),
    ]);
    search.dataset = options(&dir.join("unused"), 4).run.dataset;
    let mut opt = profile
        .options(&search, dir.join("run"), false)
        .unwrap()
        .unwrap();
    opt.run.clock = Clock::Frozen;
    let path = dir.join("seeds.json");
    let file = crate::seeds::SeedsFile::new(None, vec![seed_entry(seed)]);
    std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(path);
    opt
}

/// A two-site seed whose first refinement is a surrogate stage whose winner's pipeline pass ran.
const MULTI_SITE_SEED: &str = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c1)\") (ridge))";

// laneD:B17-lm-refinement end
