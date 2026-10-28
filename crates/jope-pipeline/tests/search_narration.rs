//! Public API qualification for the Rust pipeline narration boundary.
use jope_narrator::{fact_channel, LineageFact, RecordingTap, DEFAULT_MAX_BYTES};
use jope_pipeline::{
    emit::Language,
    run::{run_with_narration, Policy, RunOptions},
    search::Budget,
};

fn options(output: &std::path::Path) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations: 1,
            seconds: 60,
        },
        cuts: 1,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../datasets/test/pmlb-229-pwlinear"
            ),
            output,
        )
    }
}

#[test]
fn actual_single_evaluation_has_draw_start_result_selection_emission_and_terminal() {
    let temp = tempfile::tempdir().unwrap();
    let recording = RecordingTap::default();
    let config = options(&temp.path().join("run"));
    let receipt = run_with_narration(&config, Some(recording.tap())).unwrap();
    assert!(receipt.ok);
    let facts = recording.facts();
    let started = facts
        .iter()
        .position(|fact| matches!(fact, LineageFact::Started { .. }))
        .unwrap();
    let measured = facts
        .iter()
        .position(|fact| matches!(fact, LineageFact::Measured { .. }))
        .unwrap();
    let selected = facts
        .iter()
        .position(|fact| {
            matches!(fact, LineageFact::Selected { program_sha256, .. }
        if program_sha256 == &receipt.search.selected_program_sha256)
        })
        .unwrap();
    let emitted = facts.iter().position(|fact| matches!(fact, LineageFact::Emitted { program_sha256, languages }
        if program_sha256 == &receipt.search.selected_program_sha256 && languages == &["rust".to_owned()])).unwrap();
    assert!(started < measured && measured < selected && selected < emitted);
    assert!(facts[..started]
        .iter()
        .any(|fact| matches!(fact, LineageFact::Drawn { .. })));
    assert_eq!(
        facts
            .iter()
            .filter(|fact| matches!(fact, LineageFact::Measured { .. }))
            .count(),
        1
    );
    assert_eq!(
        facts
            .iter()
            .filter(|fact| matches!(fact, LineageFact::RunStopped { .. }))
            .count(),
        1
    );
    assert!(
        matches!(facts.last(), Some(LineageFact::RunStopped { status }) if status == "completed")
    );
    assert!(config.output.join("receipt.json").is_file());
    assert!(config.output.join("emit/manifest.json").is_file());
}

#[test]
fn preflight_failure_is_terminal_and_does_not_claim_a_measurement() {
    let temp = tempfile::tempdir().unwrap();
    let recording = RecordingTap::default();
    let mut config = options(&temp.path().join("run"));
    config.policy = Policy::Require;
    assert!(run_with_narration(&config, Some(recording.tap())).is_err());
    assert_eq!(
        recording.facts(),
        vec![LineageFact::RunStopped {
            status: "failed".to_owned()
        }]
    );
    assert!(!config.output.exists());
}

#[test]
fn detached_consumer_does_not_fail_or_change_the_pipeline() {
    let temp = tempfile::tempdir().unwrap();
    let (tap, receiver) = fact_channel(1, DEFAULT_MAX_BYTES);
    drop(receiver);
    let receipt = run_with_narration(&options(&temp.path().join("run")), Some(tap)).unwrap();
    assert!(receipt.ok);
    assert_eq!(
        receipt.search.selected_genome,
        jope_pipeline::search::BOOTSTRAP_GENOME
    );
}

#[test]
fn invent_reports_ranked_offers_without_claiming_consumer_selection_or_emission() {
    let temp = tempfile::tempdir().unwrap();
    let train = temp.path().join("train.csv");
    let mut csv = String::from("x,y\n");
    for index in 0..60 {
        csv.push_str(&format!("{index},{}\n", 2 * index + 1));
    }
    std::fs::write(&train, csv).unwrap();
    let recording = RecordingTap::default();
    let mut config =
        jope_pipeline::invent::InventOptions::new(&train, "y", temp.path().join("work"));
    config.cuts = 1;
    config.budget = Budget {
        evaluations: 1,
        seconds: 60,
    };
    let result =
        jope_pipeline::invent::invent_with_narration(&config, Some(recording.tap())).unwrap();
    assert!(result.ok);
    let facts = recording.facts();
    let offered: Vec<_> = result
        .candidates
        .iter()
        .map(|candidate| candidate.program_sha256.clone())
        .collect();
    assert!(facts.iter().any(|fact| matches!(fact, LineageFact::CandidatesOffered { program_sha256s, rule }
        if program_sha256s == &offered && rule.starts_with("rank_then_executable_filter_then_top_k:"))));
    assert!(!facts.iter().any(|fact| matches!(
        fact,
        LineageFact::Selected { .. } | LineageFact::Emitted { .. }
    )));
    assert_eq!(
        facts
            .iter()
            .filter(|fact| matches!(fact, LineageFact::RunStopped { .. }))
            .count(),
        1
    );
    assert!(
        matches!(facts.last(), Some(LineageFact::RunStopped { status }) if status == "completed")
    );
}
