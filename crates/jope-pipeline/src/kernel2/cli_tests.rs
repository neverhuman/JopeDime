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
