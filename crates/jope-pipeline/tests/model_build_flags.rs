//! JA-02 executor-level flag tests (SC-DEC-184E). Authored now; executed only in JA-06.
use clap::Parser;
use jope_pipeline::cli::{run, Cli, Cmd};
use jope_pipeline::model_build::{
    disposition, resolve, route, verify_checkpoint, CheckpointClaim, CheckpointRefusal,
    Disposition, ModelBuildFile, NeuralCheckpoint, ReleaseClosure, Route, Source, OPTIONS_SCHEMA,
};
use jope_pipeline::pathway::ThinkingLevel;

const SOURCE: &str = "9c9202aa39357bc64758c0550ad509e7541c4bb4";

/// Parses `run` or `run-legacy-gp`, supplying the required `--dataset` flag.
fn parse(args: &[&str]) -> Cli {
    let mut full = vec!["jope-pipeline", args[0], "--dataset", "fixture"];
    full.extend_from_slice(&args[1..]);
    Cli::try_parse_from(full).unwrap()
}

fn model_of(cli: &Cli) -> jope_pipeline::model_build::ModelBuildArgs {
    match &cli.cmd {
        Cmd::Run { model, .. } => model.clone(),
        _ => panic!("expected the run subcommand"),
    }
}

#[test]
fn cli_thinking_is_parsed_and_resolved_with_cli_over_file_precedence() {
    let cli = parse(&["run", "--output", "/tmp/x", "--thinking", "high"]);
    let flags = model_of(&cli).flags().unwrap();
    let file = ModelBuildFile {
        schema: OPTIONS_SCHEMA.into(),
        automl: None,
        ensemble: None,
        thinking: Some("low".into()),
        neural_checkpoint: None,
    };
    let resolved = resolve(Some(&file), flags).unwrap();
    assert_eq!(resolved.thinking, Some(ThinkingLevel::High));
    assert_eq!(resolved.thinking_source, Source::Flag);
}

#[test]
fn absent_thinking_is_recorded_as_default_and_never_selects_direct() {
    let cli = parse(&["run", "--output", "/tmp/x"]);
    let resolved = resolve(None, model_of(&cli).flags().unwrap()).unwrap();
    assert_eq!(resolved.thinking, None);
    assert_eq!(resolved.thinking_source, Source::Default);
}

#[test]
fn med_and_medium_share_one_identity_through_the_cli() {
    let med = resolve(
        None,
        model_of(&parse(&["run", "--output", "/tmp/x", "--thinking", "med"]))
            .flags()
            .unwrap(),
    )
    .unwrap();
    let medium = resolve(
        None,
        model_of(&parse(&[
            "run",
            "--output",
            "/tmp/x",
            "--thinking",
            "medium",
        ]))
        .flags()
        .unwrap(),
    )
    .unwrap();
    assert_eq!(med.thinking, medium.thinking);
    assert_eq!(
        serde_json::to_string(&med).unwrap(),
        serde_json::to_string(&medium).unwrap()
    );
}

#[test]
fn thinking_source_round_trips_in_the_resolved_record() {
    let cli = parse(&["run", "--output", "/tmp/x", "--thinking", "ultra"]);
    let resolved = resolve(None, model_of(&cli).flags().unwrap()).unwrap();
    let text = serde_json::to_string(&resolved).unwrap();
    assert!(text.contains(r#""thinking_source":"flag""#), "{text}");
    let back: jope_pipeline::model_build::ResolvedModelBuild = serde_json::from_str(&text).unwrap();
    assert_eq!(back, resolved);
}

#[test]
fn all_four_switch_combinations_map_to_their_routes_through_the_cli() {
    let cases = [
        (&[][..], Route::FastNeuralGp),
        (&["--automl"][..], Route::NeuralGpAutomlBakeoff),
        (&["--ensemble"][..], Route::NeuralGpWinnerEnsemble),
        (
            &["--automl", "--ensemble"][..],
            Route::CombinedRosterWinnerEnsemble,
        ),
    ];
    for (switches, expected) in cases {
        let cli = parse(
            &["run", "--output", "/tmp/x"]
                .iter()
                .chain(switches.iter())
                .copied()
                .collect::<Vec<_>>(),
        );
        let resolved = resolve(None, model_of(&cli).flags().unwrap()).unwrap();
        assert_eq!(route(&resolved), expected, "{switches:?}");
    }
}

#[test]
fn ensemble_alone_never_enables_the_automl_sweep() {
    let cli = parse(&["run", "--output", "/tmp/x", "--ensemble"]);
    let resolved = resolve(None, model_of(&cli).flags().unwrap()).unwrap();
    assert!(!resolved.automl);
    assert!(resolved.ensemble);
    let verified = NeuralCheckpoint::Verified(verified_checkpoint());
    match disposition(route(&resolved), &verified) {
        Disposition::Pending { missing, .. } => {
            assert!(missing.iter().any(|m| m.contains("JA-04")), "{missing:?}");
            assert!(!missing.iter().any(|m| m.contains("JA-03")), "{missing:?}");
        }
        other => panic!("ensemble-only must be pending, got {other:?}"),
    }
}

#[test]
fn automl_false_in_the_file_does_not_disable_an_explicit_ensemble_flag() {
    let file = ModelBuildFile {
        schema: OPTIONS_SCHEMA.into(),
        automl: Some(false),
        ensemble: None,
        thinking: None,
        neural_checkpoint: None,
    };
    let cli = parse(&["run", "--output", "/tmp/x", "--ensemble"]);
    let resolved = resolve(Some(&file), model_of(&cli).flags().unwrap()).unwrap();
    assert!(!resolved.automl);
    assert_eq!(resolved.automl_source, Source::File);
    assert!(resolved.ensemble);
    assert_eq!(route(&resolved), Route::NeuralGpWinnerEnsemble);
}

#[test]
fn default_run_refuses_without_a_verified_checkpoint_and_names_no_fallback() {
    let cli = parse(&["run", "--output", "/nonexistent-ja02-out"]);
    let err = run(cli).unwrap_err().to_string();
    assert!(err.contains("no verified P02 release closure"), "{err}");
    assert!(err.contains("no generic CPU GP or ridge fallback"), "{err}");
}

#[test]
fn explicit_legacy_gp_command_is_separate_from_the_model_build_routes() {
    let cli = parse(&["run-legacy-gp", "--output", "/tmp/x"]);
    assert!(matches!(cli.cmd, Cmd::RunLegacyGp { .. }));
}

fn verified_checkpoint() -> jope_pipeline::model_build::VerifiedNeuralCheckpoint {
    let claim = CheckpointClaim {
        release_id: "native-policy-lm-282-r1".into(),
        source_commit: SOURCE.into(),
        checkpoint_sha256: "a".repeat(64),
        runtime_sha256: "b".repeat(64),
    };
    let closure = ReleaseClosure {
        release_id: claim.release_id.clone(),
        source_commit: claim.source_commit.clone(),
        checkpoint_sha256: claim.checkpoint_sha256.clone(),
        runtime_sha256: claim.runtime_sha256.clone(),
    };
    verify_checkpoint(&claim, Some(&closure), SOURCE).unwrap()
}

#[test]
fn unverified_malformed_and_wrong_source_checkpoints_are_refused_explicitly() {
    let claim = CheckpointClaim {
        release_id: "native-policy-lm-282-r1".into(),
        source_commit: SOURCE.into(),
        checkpoint_sha256: "a".repeat(64),
        runtime_sha256: "b".repeat(64),
    };
    assert_eq!(
        verify_checkpoint(&claim, None, SOURCE).unwrap_err(),
        CheckpointRefusal::Absent
    );
    let mut malformed = claim.clone();
    malformed.checkpoint_sha256 = "not-a-hash".into();
    assert!(matches!(
        verify_checkpoint(&malformed, None, SOURCE),
        Err(CheckpointRefusal::Malformed(_))
    ));
    let other = "2".repeat(40);
    assert!(matches!(
        verify_checkpoint(&claim, None, &other),
        Err(CheckpointRefusal::WrongSource { .. })
    ));
    let refused = NeuralCheckpoint::Refused(CheckpointRefusal::Absent);
    for r in [Route::FastNeuralGp, Route::CombinedRosterWinnerEnsemble] {
        assert!(matches!(
            disposition(r, &refused),
            Disposition::Refused { .. }
        ));
    }
}

#[test]
fn a_file_claim_cannot_bypass_the_closure_on_the_public_command() {
    let claim = CheckpointClaim {
        release_id: "native-policy-lm-282-r1".into(),
        source_commit: SOURCE.into(),
        checkpoint_sha256: "a".repeat(64),
        runtime_sha256: "b".repeat(64),
    };
    let file = ModelBuildFile {
        schema: OPTIONS_SCHEMA.into(),
        automl: None,
        ensemble: None,
        thinking: None,
        neural_checkpoint: Some(claim),
    };
    let text = serde_json::to_string(&file).unwrap();
    let dir = tempfile::Builder::new()
        .prefix("ja02-flags-")
        .tempdir()
        .unwrap();
    let path = dir.path().join("options.json");
    std::fs::write(&path, text).unwrap();
    let cli = parse(&[
        "run",
        "--output",
        "/tmp/x",
        "--model-build-config",
        path.to_str().unwrap(),
    ]);
    let err = run(cli).unwrap_err().to_string();
    assert!(
        err.contains("no independently trusted source expectation"),
        "{err}"
    );
}
