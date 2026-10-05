use super::{cli::ProfileArgs, publication, receipt, run, search::Refinement};
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

#[test]
fn cli_keeps_legacy_defaults_and_requires_explicit_kernel2_refinement_limits() {
    let (search, profile) = args(&[]);
    assert!(profile
        .options(&search, PathBuf::from("unused"), false)
        .unwrap()
        .is_none());
    assert_eq!(
        search.options("unused".into(), false).unwrap().languages,
        RunOptions::new("unresolved", "unused").languages
    );
    let (search, profile) = args(&["--search-profile", "legacy", "--fault-policy", "legacy"]);
    assert!(profile
        .options(&search, "unused".into(), false)
        .unwrap()
        .is_none());
    let (search, profile) = args(&["--search-profile", "kernel2-baseline"]);
    assert!(profile
        .options(&search, "unused".into(), false)
        .unwrap_err()
        .to_string()
        .contains("--refine-top-k"));
    // Both CLI spellings use the existing run path and write identical bytes.
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let flags = [
        "--budget",
        "evaluations=4,seconds=300",
        "--policy",
        "skip",
        "--emit",
        "rust",
        "--rng-seed",
        "7",
        "--source-commit",
        source.as_str(),
    ];
    let make = |explicit: bool, output: PathBuf| {
        let mut argv = flags.to_vec();
        if explicit {
            argv.extend(["--search-profile", "legacy", "--fault-policy", "legacy"]);
        }
        let (mut search, profile) = args(&argv);
        search.dataset = options(&output, 4).run.dataset;
        assert!(profile
            .options(&search, output.clone(), false)
            .unwrap()
            .is_none());
        let mut options = search.options(output, false).unwrap();
        options.clock = Clock::Frozen;
        options
    };
    let implicit = make(false, temp.path().join("implicit"));
    let explicit = make(true, temp.path().join("explicit"));
    assert_eq!(
        crate::run::run(&implicit).unwrap(),
        crate::run::run(&explicit).unwrap()
    );
    assert_eq!(
        std::fs::read(implicit.output.join("receipt.json")).unwrap(),
        std::fs::read(explicit.output.join("receipt.json")).unwrap()
    );
}

#[test]
fn cli_adds_fitted_cpp_only_to_kernel2_defaults_and_preserves_explicit_emit() {
    let flags = [
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
    ];
    let (search, profile) = args(&flags);
    let opt = profile
        .options(&search, "unused".into(), false)
        .unwrap()
        .unwrap();
    assert!(opt.run.languages.contains(&Language::Cpp17Fitted));
    let mut explicit = flags.to_vec();
    explicit.extend(["--emit", "rust,python,cpp17,cpp20,js"]);
    let (search, profile) = args(&explicit);
    let opt = profile
        .options(&search, "unused".into(), false)
        .unwrap()
        .unwrap();
    assert!(!opt.run.languages.contains(&Language::Cpp17Fitted));
}

#[test]
fn invalid_fault_profile_and_penalty_options_are_refused_before_output() {
    for flags in [
        vec!["--search-profile", "unknown"],
        vec!["--fault-policy", "reject-hard"],
        vec![
            "--search-profile",
            "kernel2-baseline",
            "--fault-policy",
            "legacy",
        ],
        vec![
            "--search-profile",
            "kernel2-baseline",
            "--fault-policy",
            "penalize",
        ],
        vec![
            "--search-profile",
            "kernel2-baseline",
            "--lambda-domain",
            "0.5",
        ],
        vec![
            "--search-profile",
            "kernel2-baseline",
            "--fault-policy",
            "penalize",
            "--lambda-domain=-0.5",
            "--lambda-saturation",
            "0",
            "--fault-tie-break",
            "true",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "4",
        ],
    ] {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("run");
        let (search, profile) = args(&flags);
        assert!(profile.options(&search, output.clone(), false).is_err());
        assert!(!output.exists());
    }
}

#[test]
fn shipping_run_emits_and_rederives_the_profile_receipt() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut opt = options(&output, 4);
    opt.run.languages = vec![
        Language::Rust,
        Language::Python,
        Language::Cpp17,
        Language::Cpp17Fitted,
    ];
    let receipt = run::run(&opt).unwrap();
    assert_eq!(receipt.schema, receipt::SCHEMA_ID);
    assert_eq!(receipt.search.completed_evaluations, 4);
    assert!(receipt
        .artifacts
        .iter()
        .any(|a| a.language == "cpp17-fitted"));
    assert_eq!(
        publication::verify_run(&output.join("receipt.json")).unwrap(),
        receipt
    );
    assert!(crate::receipt::read(&output.join("receipt.json")).is_err());
}

#[test]
fn explicit_experimental_profile_and_penalty_are_bound_and_rederived() {
    let temp = tempfile::tempdir().unwrap();
    let base = options(&temp.path().join("run"), 4).run;
    let opt = run::Options::new(
        base,
        "kernel2-experimental",
        FaultPolicy::Penalize {
            lambda_domain: 0.05,
            lambda_saturation: 0.0,
            tie_break: true,
        },
        Refinement {
            top_k: 1,
            trials_per_program: 4,
        },
    )
    .unwrap();
    let receipt = run::run(&opt).unwrap();
    assert_eq!(receipt.search.profile_id, "kernel2-experimental");
    assert_eq!(
        publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
        receipt
    );
}

#[test]
fn seeded_refinement_trials_and_child_measurements_share_the_published_budget() {
    let temp = tempfile::tempdir().unwrap();
    let opt = seeded(&temp.path().join("run"));
    let receipt = run::run(&opt).unwrap();
    assert!(receipt.search.objective_trials > 0);
    assert_eq!(
        receipt.search.completed_evaluations,
        receipt.search.candidate_measurements + receipt.search.objective_trials
    );
    assert_eq!(
        publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
        receipt
    );
    let refinements: Value = serde_json::from_slice(
        &std::fs::read(opt.run.output.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    assert!(refinements
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["measurement"].is_number()));
}

#[test]
fn receipt_rejects_unknown_profile_digest_budget_and_fault_claims() {
    let temp = tempfile::tempdir().unwrap();
    let receipt = run::run(&options(&temp.path().join("run"), 4)).unwrap();
    let original = serde_json::to_value(receipt).unwrap();
    for (pointer, bad) in [
        ("/search/profile_sha256", json!("0".repeat(64))),
        ("/search/completed_evaluations", json!(0)),
        ("/search/objective_trials", json!(100)),
        ("/search/best/selection_score", json!(100.0)),
        ("/search/best/faulted_cells", json!(100)),
        ("/search/stopped_by", json!("bored")),
    ] {
        let mut value = original.clone();
        *value.pointer_mut(pointer).unwrap() = bad;
        assert!(receipt::validate_value(&value).is_err(), "{pointer}");
    }
}

#[test]
fn changed_observation_bytes_or_selected_winner_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let receipt = run::run(&options(&output, 4)).unwrap();
    let path = output.join("receipt.json");
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["search"]["best"]["observation"]["genome"] = "(pipe X (standardize) (ridge))".into();
    value["search"]["best"]["observation"]["program_sha256"] =
        crate::sha256_hex(b"(pipe X (standardize) (ridge))").into();
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(publication::verify_run(&path).is_err());
    std::fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    std::fs::write(output.join("search/cut-0/observations.json"), b"[]").unwrap();
    assert!(publication::verify_run(&path).is_err());
}

#[test]
fn rehashed_sidecars_cannot_forge_selection_scores_or_refinement_counts() {
    let temp = tempfile::tempdir().unwrap();
    let opt = seeded(&temp.path().join("run"));
    let receipt = run::run(&opt).unwrap();
    let path = opt.run.output.join("receipt.json");
    let dir = opt.run.output.join("search/cut-0");
    for name in ["observations.json", "refinements.json"] {
        let original = std::fs::read(dir.join(name)).unwrap();
        let mut sidecar: Value = serde_json::from_slice(&original).unwrap();
        if name == "observations.json" {
            sidecar[0]["selection_score"] = 123.0.into();
        } else {
            sidecar[0]["refinement"]["evaluations"] = 100.into();
        }
        let bytes = serde_json::to_vec_pretty(&sidecar).unwrap();
        std::fs::write(dir.join(name), &bytes).unwrap();
        let mut edited = receipt.clone();
        if name == "observations.json" {
            edited.search.observations_sha256 = crate::sha256_hex(&bytes);
        } else {
            edited.search.refinements_sha256 = crate::sha256_hex(&bytes);
        }
        let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
        std::fs::write(dir.join("search.json"), &summary).unwrap();
        edited.search_sha256 = crate::sha256_hex(&summary);
        std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
        assert!(publication::verify_run(&path).is_err(), "{name}");
        std::fs::write(dir.join(name), original).unwrap();
    }
}

#[test]
fn rehashed_recipe_artifacts_must_match_the_selected_emission_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut opt = options(&output, 4);
    opt.run.languages = vec![Language::Rust, Language::Python];
    let original = run::run(&opt).unwrap();
    let path = output.join("receipt.json");
    publication::verify_run(&path).unwrap();
    for index in 0..original.artifacts.len() {
        let mut edited = original.clone();
        let artifact = &mut edited.artifacts[index];
        let source = output.join("emit").join(&artifact.path);
        let bytes = std::fs::read(&source).unwrap();
        let substituted = b"unrelated emitted source\n";
        std::fs::write(&source, substituted).unwrap();
        artifact.sha256 = crate::sha256_hex(substituted);
        artifact.bytes = substituted.len() as u64;
        let language = artifact.language.clone();
        std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
        assert!(
            publication::verify_run(&path).is_err(),
            "rehashed {} artifact disagrees with the selected manifest",
            language
        );
        std::fs::write(&source, &bytes).unwrap();
        let mut renamed = original.clone();
        renamed.artifacts[index].path = format!("renamed-{}", original.artifacts[index].path);
        std::fs::write(
            output.join("emit").join(&renamed.artifacts[index].path),
            bytes,
        )
        .unwrap();
        std::fs::write(&path, serde_json::to_vec(&renamed).unwrap()).unwrap();
        assert!(
            publication::verify_run(&path).is_err(),
            "renamed {language} artifact"
        );
    }
    let mut reclassified = original.clone();
    reclassified.artifacts[0].language = "cpp17".into();
    reclassified.artifacts[0].path = "model.cpp".into();
    let substituted = b"unrelated emitted source\n";
    std::fs::write(output.join("emit/model.cpp"), substituted).unwrap();
    reclassified.artifacts[0].sha256 = crate::sha256_hex(substituted);
    reclassified.artifacts[0].bytes = substituted.len() as u64;
    std::fs::write(&path, serde_json::to_vec(&reclassified).unwrap()).unwrap();
    assert!(
        publication::verify_run(&path).is_err(),
        "reclassified recipe"
    );
    let mut omitted = original.clone();
    omitted.artifacts.remove(0);
    std::fs::write(&path, serde_json::to_vec(&omitted).unwrap()).unwrap();
    assert!(publication::verify_run(&path).is_err(), "omitted recipe");
    std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    publication::verify_run(&path).unwrap();
    let all_output = temp.path().join("all");
    let mut all_run = options(&all_output, 4).run;
    all_run.budget.evaluations = 1;
    all_run.languages = vec![
        Language::Rust,
        Language::Python,
        Language::Cpp17,
        Language::Cpp20,
        Language::Cpp17Fitted,
    ];
    let all_options = run::Options::new(
        all_run,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 0,
            trials_per_program: 0,
        },
    )
    .unwrap();
    let all = run::run(&all_options).unwrap();
    let all_path = all_output.join("receipt.json");
    publication::verify_run(&all_path).unwrap();
    for language in [
        "cpp17",
        "cpp20",
        "cpp17-fitted",
        "cpp17-fitted-codec",
        "cpp17-fitted-standalone",
    ] {
        assert!(
            all.artifacts
                .iter()
                .any(|artifact| artifact.language == language),
            "{language}"
        );
    }
    for language in [
        "cpp17-fitted",
        "cpp17-fitted-codec",
        "cpp17-fitted-standalone",
    ] {
        let mut incomplete = all.clone();
        incomplete
            .artifacts
            .retain(|artifact| artifact.language != language);
        std::fs::write(&all_path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
        assert!(
            publication::verify_run(&all_path).is_err(),
            "missing {language}"
        );
    }
    std::fs::write(&all_path, serde_json::to_vec(&all).unwrap()).unwrap();
    publication::verify_run(&all_path).unwrap();
}

#[test]
fn unknown_artifact_traversal_and_symlink_evidence_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let receipt = run::run(&options(&output, 4)).unwrap();
    let path = output.join("receipt.json");
    let mut unknown = receipt.clone();
    unknown.artifacts[0].language = "unknown".into();
    std::fs::write(&path, serde_json::to_vec(&unknown).unwrap()).unwrap();
    assert!(publication::verify_run(&path).is_err());
    for forged in ["../receipt.json", "/etc/passwd"] {
        let mut edited = receipt.clone();
        edited.artifacts[0].path = forged.into();
        std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
        assert!(publication::verify_run(&path).is_err());
    }
    std::fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
    let artifact = output.join("emit").join(&receipt.artifacts[0].path);
    let outside = temp.path().join("outside");
    std::fs::rename(&artifact, &outside).unwrap();
    std::os::unix::fs::symlink(&outside, &artifact).unwrap();
    assert!(publication::verify_run(&path).is_err());
}

#[test]
fn unsupported_cuts_concurrency_seeds_and_existing_output_fail_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    for mode in ["cuts", "threads", "islands", "incumbent"] {
        let output = temp.path().join(mode);
        let mut opt = options(&output, 4);
        match mode {
            "cuts" => opt.run.cuts = 2,
            "threads" => opt.run.threads = 2,
            "islands" => opt.run.islands = 2,
            _ => opt.run.incumbent = Some("missing".into()),
        }
        assert!(run::run(&opt).is_err());
        assert!(!output.exists());
    }
    let output = temp.path().join("seed");
    let mut opt = options(&output, 4);
    let file = temp.path().join("invalid-seed.json");
    let seeds = crate::seeds::SeedsFile::new(
        None,
        vec![seed_entry("(pipe X (derive expr=\"exp(c0)\") (ridge))")],
    );
    std::fs::write(&file, serde_json::to_vec(&seeds).unwrap()).unwrap();
    opt.run.seed_genomes = Some(file);
    assert!(run::run(&opt).is_err());
    assert!(!output.exists());
    let output = temp.path().join("existing");
    std::fs::create_dir(&output).unwrap();
    std::os::unix::fs::symlink(temp.path().join("outside"), output.join("emit")).unwrap();
    assert!(run::run(&options(&output, 4)).is_err());
    assert_eq!(std::fs::read_dir(output).unwrap().count(), 1);
}

#[test]
fn cancelled_run_writes_nothing_and_the_same_uncancelled_command_succeeds() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut opt = options(&output, 4);
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    opt.run.cancel = Some(cancelled.clone());
    assert!(crate::parallel::is_cancelled(&run::run(&opt).unwrap_err()));
    assert!(!output.exists());
    cancelled.store(false, std::sync::atomic::Ordering::SeqCst);
    run::run(&opt).unwrap();
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

#[test]
fn profile_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    let a = options(&temp.path().join("a"), 16);
    let b = options(&temp.path().join("b"), 16);
    assert_eq!(run::run(&a).unwrap(), run::run(&b).unwrap());
    for relative in [
        "receipt.json",
        "prepared/prepared.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        "search/cut-0/refinements.json",
        "emit/model.rs",
        "emit/manifest.json",
    ] {
        assert_eq!(
            std::fs::read(a.run.output.join(relative)).unwrap(),
            std::fs::read(b.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
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

#[test]
fn lm_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    let a = lm_seeded(&temp.path().join("a"), 40);
    let b = lm_seeded(&temp.path().join("b"), 40);
    let receipt = run::run(&a).unwrap();
    assert_eq!(receipt, run::run(&b).unwrap());
    for relative in SIDECARS {
        assert_eq!(
            std::fs::read(a.run.output.join(relative)).unwrap(),
            std::fs::read(b.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
    let engine = receipt.search.engine_selection.as_ref().unwrap();
    assert_eq!(engine.refine.as_ref().unwrap().id, super::DAMPED_LS);
    assert!(receipt.search.objective_trials > 0);
    assert_eq!(
        publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
        receipt
    );
    let refinements: Value = serde_json::from_slice(
        &std::fs::read(a.run.output.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    assert!(refinements
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["refinement"]["trace"]["stages"]
            .as_array()
            .is_some_and(|s| s.iter().any(|s| s["kind"] == "surrogate"))));
}

#[test]
fn explicit_coordinate_method_keeps_receipt_bytes_and_lm_flags_are_checked() {
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let dataset = options(&temp.path().join("unused"), 4).run.dataset;
    let base = [
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
        "--budget",
        "evaluations=6,seconds=300",
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
        let mut argv = base.to_vec();
        argv.extend_from_slice(extra);
        let (mut search, profile) = args(&argv);
        search.dataset = dataset.clone();
        let mut opt = profile.options(&search, output, false).unwrap().unwrap();
        opt.run.clock = Clock::Frozen;
        opt
    };
    let implicit = make(&[], temp.path().join("implicit"));
    let explicit = make(
        &["--refine-method", "coordinate"],
        temp.path().join("explicit"),
    );
    let receipt = run::run(&implicit).unwrap();
    assert_eq!(receipt, run::run(&explicit).unwrap());
    for relative in SIDECARS {
        assert_eq!(
            std::fs::read(implicit.run.output.join(relative)).unwrap(),
            std::fs::read(explicit.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
    assert!(serde_json::to_value(&receipt).unwrap()["search"]
        .get("engine_selection")
        .is_none());
    assert!(!String::from_utf8(
        std::fs::read(implicit.run.output.join("search/cut-0/refinements.json")).unwrap()
    )
    .unwrap()
    .contains("\"trace\""));
    make(&["--refine-method", "lm"], temp.path().join("lm"));
    for (flags, refusal) in [
        (
            vec!["--refine-method", "lm"],
            "legacy search refuses kernel2 fault/refinement options",
        ),
        (
            vec![
                "--search-profile",
                "legacy",
                "--refine-method",
                "coordinate",
            ],
            "legacy search refuses kernel2 fault/refinement options",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "4",
                "--refine-method",
                "fast",
            ],
            "--refine-method must be coordinate or lm",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "2",
                "--refine-method",
                "lm",
            ],
            "--refine-trials of at least 3",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "0",
                "--refine-trials",
                "4",
                "--refine-method",
                "lm",
            ],
            "--refine-top-k of at least 1",
        ),
    ] {
        let output = temp.path().join("refused");
        let (mut search, profile) = args(&flags);
        search.dataset = dataset.clone();
        let error = profile.options(&search, output.clone(), false).unwrap_err();
        assert!(
            format!("{error:#}").contains(refusal),
            "{flags:?}: {error:#}"
        );
        assert!(!output.exists());
    }
}

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

#[test]
fn rehashed_lm_traces_and_selection_records_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let opt = lm_seeded(&temp.path().join("run"), 40);
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/refinements.json",
    ]
    .iter()
    .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
    .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone();
    let refinements: Value = serde_json::from_slice(&originals[2].1).unwrap();
    // Rewriting the untouched evidence through the same path still verifies.
    rewrite_lm_evidence(output, &receipt, &refinements, engine.clone());
    publication::verify_run(&path).unwrap();
    restore();
    let traced = refinements
        .as_array()
        .unwrap()
        .iter()
        .position(|r| r["refinement"]["trace"].is_object())
        .expect("a refinement with a trace");
    let stage = &refinements[traced]["refinement"]["trace"]["stages"][0];
    let mut edits: Vec<(&str, Value)> = Vec::new();
    let mut dropped = refinements.clone();
    dropped[traced]["refinement"]
        .as_object_mut()
        .unwrap()
        .remove("trace");
    edits.push(("dropped trace", dropped));
    let mut reordered = refinements.clone();
    reordered[traced]["refinement"]["trace"]["method"] = "feat-math.other.v1".into();
    edits.push(("unknown method", reordered));
    if stage["kind"] == "surrogate" {
        for field in [
            "trials",
            "jacobians",
            "initial",
            "rescore",
            "rejected_trials",
        ] {
            let mut edited = refinements.clone();
            let value = &mut edited[traced]["refinement"]["trace"]["stages"][0]["surrogate"][field];
            *value = (value.as_u64().unwrap() + 1).into();
            edits.push((field, edited));
        }
        if stage["pipeline"].is_object() {
            let mut flipped = refinements.clone();
            let accepted =
                &mut flipped[traced]["refinement"]["trace"]["stages"][0]["pipeline"]["accepted"];
            *accepted = (!accepted.as_bool().unwrap()).into();
            edits.push(("pipeline decision", flipped));
        }
    } else {
        let mut edited = refinements.clone();
        let value = &mut edited[traced]["refinement"]["trace"]["stages"][0]["evaluations"];
        *value = (value.as_u64().unwrap() + 1).into();
        edits.push(("coordinate evaluations", edited));
    }
    for (label, edited) in &edits {
        let path = rewrite_lm_evidence(output, &receipt, edited, engine.clone());
        assert!(publication::verify_run(&path).is_err(), "{label}");
        restore();
    }
    // A trace without the selection that produces it.
    let path = rewrite_lm_evidence(output, &receipt, &refinements, None);
    assert!(publication::verify_run(&path).is_err(), "unselected trace");
    restore();
    // A selection the receipt cannot name.
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["search"]["engine_selection"]["refine"]["id"] = "feat-math.other.v1".into();
    assert!(receipt::validate_value(&value).is_err());
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["search"]["engine_selection"]["refine"]["jacobian"] = "one".into();
    receipt::validate_value(&value).unwrap();
    // With more than one literal site the two charges differ, so the other charge is refused.
    let multi_site = refinements.as_array().unwrap().iter().any(|r| {
        r["refinement"]["trace"]["stages"]
            .as_array()
            .is_some_and(|s| s.iter().any(|s| s["surrogate"]["sites"].as_u64() > Some(1)))
    });
    if multi_site {
        let mut record = engine.clone().unwrap();
        record.refine.as_mut().unwrap().jacobian = "one".into();
        let path = rewrite_lm_evidence(output, &receipt, &refinements, Some(record));
        assert!(publication::verify_run(&path).is_err(), "jacobian charge");
        restore();
    }
    publication::verify_run(&path).unwrap();
    // A coordinate run's refinements cannot claim the selection after the fact.
    let plain = seeded(&temp.path().join("plain"));
    let plain_receipt = run::run(&plain).unwrap();
    let plain_refinements: Value = serde_json::from_slice(
        &std::fs::read(plain.run.output.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    let path = rewrite_lm_evidence(
        &plain.run.output,
        &plain_receipt,
        &plain_refinements,
        engine,
    );
    assert!(publication::verify_run(&path).is_err(), "claimed selection");
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

/// `--refine-trials 4 --refine-method lm` on a single-site seed: no stage reaches its surrogate
/// minimum (one site plus five), so the run is the coordinate run, measurements and refinements
/// bit for bit, and every trace stage names the small allowance.
#[test]
fn lm_under_the_surrogate_minimum_is_the_coordinate_run_with_its_reason() {
    let temp = tempfile::tempdir().unwrap();
    let seed = "(pipe X (derive expr=\"sin(0.5*c0)\") (ridge))";
    let coordinate = flag_run(&temp.path().join("coordinate"), seed, 40, 4, "coordinate");
    let lm = flag_run(&temp.path().join("lm"), seed, 40, 4, "lm");
    let (plain, traced) = (run::run(&coordinate).unwrap(), run::run(&lm).unwrap());
    for (opt, receipt) in [(&coordinate, &plain), (&lm, &traced)] {
        assert_eq!(
            publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
            *receipt
        );
    }
    let read =
        |opt: &run::Options, relative: &str| std::fs::read(opt.run.output.join(relative)).unwrap();
    assert_eq!(
        read(&coordinate, "search/cut-0/observations.json"),
        read(&lm, "search/cut-0/observations.json")
    );
    let expected: Value =
        serde_json::from_slice(&read(&coordinate, "search/cut-0/refinements.json")).unwrap();
    let mut actual: Value =
        serde_json::from_slice(&read(&lm, "search/cut-0/refinements.json")).unwrap();
    assert!(!expected.as_array().unwrap().is_empty());
    let mut stages = 0;
    for record in actual.as_array_mut().unwrap() {
        let charged = record["refinement"]["evaluations"].as_u64().unwrap();
        let trace = record["refinement"]
            .as_object_mut()
            .unwrap()
            .remove("trace");
        assert_eq!(trace.is_some(), charged > 0);
        for stage in trace.iter().flat_map(|t| t["stages"].as_array().unwrap()) {
            assert_eq!(stage["kind"], "coordinate", "{stage}");
            assert_eq!(
                stage["reason"],
                super::refinement::SMALL_ALLOWANCE,
                "{stage}"
            );
            stages += 1;
        }
    }
    assert!(stages > 0);
    assert_eq!(actual, expected);
    // The summaries differ only by the selection and the refinements digest.
    let summary = |opt: &run::Options| {
        let mut value: Value =
            serde_json::from_slice(&read(opt, "search/cut-0/search.json")).unwrap();
        let fields = value.as_object_mut().unwrap();
        fields.remove("refinements_sha256");
        let selection = fields.remove("engine_selection");
        (value, selection.is_some())
    };
    assert_eq!(summary(&coordinate), (summary(&lm).0, false));
    assert!(summary(&lm).1);
    assert_eq!(
        (
            plain.search.objective_trials,
            plain.search.rejected_trials,
            plain.search.completed_evaluations
        ),
        (
            traced.search.objective_trials,
            traced.search.rejected_trials,
            traced.search.completed_evaluations
        )
    );
}

/// A two-site seed whose first refinement is a surrogate stage whose winner's pipeline pass ran.
const MULTI_SITE_SEED: &str = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c1)\") (ridge))";

/// The rehashed-evidence refusals on a fixture that has every shape they edit: a surrogate stage
/// with two literal sites and a pipeline decision. Each edit is checked to have changed the
/// evidence before its refusal is checked, and the two records the review found accepted (a
/// fallback with equal counts, Jacobians traded for trials) are among them.
#[test]
fn rehashed_multi_site_lm_traces_are_refused_with_every_edit_applied() {
    use super::refinement::{ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED};
    let temp = tempfile::tempdir().unwrap();
    let opt = flag_run(&temp.path().join("multi"), MULTI_SITE_SEED, 40, 12, "lm");
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/refinements.json",
    ]
    .iter()
    .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
    .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone();
    let refinements: Value = serde_json::from_slice(&originals[2].1).unwrap();
    rewrite_lm_evidence(output, &receipt, &refinements, engine.clone());
    publication::verify_run(&path).unwrap();
    restore();
    // The fixture's guarantee, checked rather than assumed.
    let pointer = refinements
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .find_map(|(r, record)| {
            let stages = record["refinement"]["trace"]["stages"].as_array()?;
            let s = stages.iter().position(|s| {
                s["kind"] == "surrogate"
                    && s["surrogate"]["sites"].as_u64() >= Some(2)
                    && s["pipeline"]["evaluations"] == 2
                    && s["pipeline"]["winner"].is_u64()
            })?;
            Some(format!("/{r}/refinement/trace/stages/{s}"))
        })
        .expect("a multi-site surrogate stage with a pipeline decision");
    let stage = refinements.pointer(&pointer).unwrap().clone();
    let count = |value: &Value| value.as_u64().unwrap();
    let charge = |s: &Value| {
        let surrogate = &s["surrogate"];
        count(&surrogate["initial"])
            + count(&surrogate["trials"])
            + count(&surrogate["jacobian_cost"]) * count(&surrogate["jacobians"])
            + count(&surrogate["rescore"])
            + count(&s["pipeline"]["evaluations"])
    };
    let (evaluations, rejected) = (
        charge(&stage),
        count(&stage["surrogate"]["rejected_trials"])
            + count(&stage["pipeline"]["rejected_trials"]),
    );
    let mut edits: Vec<(String, Value)> = Vec::new();
    let mut edit = |label: String, change: &dyn Fn(&mut Value)| {
        let mut edited = refinements.clone();
        change(edited.pointer_mut(&pointer).unwrap());
        assert_ne!(edited, refinements, "{label}: the edit was not applied");
        edits.push((label, edited));
    };
    for field in [
        "trials",
        "jacobians",
        "initial",
        "rescore",
        "rejected_trials",
        "sites",
        "accepted",
    ] {
        edit(field.into(), &|s| {
            let value = &mut s["surrogate"][field];
            *value = (value.as_u64().unwrap() + 1).into();
        });
    }
    edit("pipeline decision".into(), &|s| {
        let accepted = &mut s["pipeline"]["accepted"];
        *accepted = (!accepted.as_bool().unwrap()).into();
    });
    edit("winner's pass dropped".into(), &|s| {
        let pipeline = s["pipeline"].as_object_mut().unwrap();
        pipeline.remove("winner");
        pipeline.insert("evaluations".into(), 1.into());
        pipeline.insert("accepted".into(), false.into());
    });
    edit("start's loss dropped".into(), &|s| {
        s["pipeline"].as_object_mut().unwrap().remove("start");
    });
    edit("pipeline passes dropped".into(), &|s| {
        s.as_object_mut().unwrap().remove("pipeline");
    });
    // The review's first trigger: the stage replaced by a fallback with equal counts.
    for reason in [ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED] {
        edit(format!("fallback {reason}"), &|s| {
            *s = json!({
                "kind": "coordinate",
                "stage": s["stage"],
                "reason": reason,
                "evaluations": evaluations,
                "rejected_trials": rejected,
            });
        });
    }
    // The review's second trigger at this stage's charge: one more Jacobian, and a Jacobian's
    // worth fewer trials with the steps they took, keep the stage's charge and every refusal.
    edit("jacobians for trials".into(), &|s| {
        let before = charge(s);
        let surrogate = &mut s["surrogate"];
        let cost = count(&surrogate["jacobian_cost"]);
        let iterations = surrogate["iterations"].as_array_mut().unwrap();
        let removed: Vec<Value> = iterations
            .drain(iterations.len() - cost as usize..)
            .collect();
        assert!(removed
            .iter()
            .all(|i| i["solved"] == true && i["trial"].is_u64()));
        let steps = removed.iter().filter(|i| i["accepted"] == true).count() as u64;
        surrogate["accepted"] = (count(&surrogate["accepted"]) - steps).into();
        surrogate["trials"] = (count(&surrogate["trials"]) - cost).into();
        surrogate["jacobians"] = (count(&surrogate["jacobians"]) + 1).into();
        assert_eq!(charge(s), before);
    });
    for (label, edited) in &edits {
        let path = rewrite_lm_evidence(output, &receipt, edited, engine.clone());
        assert!(publication::verify_run(&path).is_err(), "{label}");
        restore();
    }
    // A stage with two sites re-derives a different charge under `one`.
    let mut record = engine.clone().unwrap();
    record.refine.as_mut().unwrap().jacobian = "one".into();
    let path = rewrite_lm_evidence(output, &receipt, &refinements, Some(record));
    assert!(publication::verify_run(&path).is_err(), "jacobian charge");
    restore();
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

/// The default runs against the stack before the method was wired: these digests were measured
/// at a8bd39855, the commit before the wiring (fix-01 probe, the same two runs in this lane on
/// that commit's sources). The search sidecars are pinned; emission is not this change's subject.
#[test]
fn default_runs_keep_the_pre_wiring_search_sidecars() {
    let temp = tempfile::tempdir().unwrap();
    for (opt, pins) in [
        (
            seeded(&temp.path().join("seeded")),
            [
                "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
                "b2444c5c34dee80c91aa52d19a4a34aeaa358bf68dd3919723f52463fb2c5420",
                "ad1e6ab3e218f237120668e39321daf2a60cc43b5a503011f1a05ecbbb907767",
                "a6e3af0c1a0b0e85bfe64e3d8bd5c869e6a58b890403ba3438efba144e12a4d7",
            ],
        ),
        (
            options(&temp.path().join("plain"), 16),
            [
                "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
                "fb3f1637f853a232ca9d13d7cc5aa9b8d4644704ce46ad121377a3837c7948fb",
                "fa6fd28f1aa37a516a39719ebd49d476624f3e220af92ec2a67b4104dc48bb96",
                "75b7234a70a005c8ba7533ad9ec200dcbe70bb3537e3d97ffe5f75bc449bd3fa",
            ],
        ),
    ] {
        run::run(&opt).unwrap();
        for (relative, pin) in [
            "prepared/prepared.json",
            "search/cut-0/search.json",
            "search/cut-0/observations.json",
            "search/cut-0/refinements.json",
        ]
        .into_iter()
        .zip(pins)
        {
            let bytes = std::fs::read(opt.run.output.join(relative)).unwrap();
            assert_eq!(crate::sha256_hex(&bytes), pin, "{relative}");
        }
    }
}
// laneD:B17-lm-refinement end

#[test]
fn v4_schema_matches_its_generated_type_and_legacy_schemas_are_unchanged() {
    assert_eq!(receipt::schema(), receipt::derived_schema());
    assert_eq!(
        receipt::schema()["properties"]["schema"]["const"],
        receipt::SCHEMA_ID
    );
    let legacy = crate::receipt::schema();
    assert_eq!(legacy, crate::receipt::derived_schema());
    assert_eq!(
        legacy["properties"]["schema"]["const"],
        crate::receipt::RECEIPT_SCHEMA_ID
    );
}
