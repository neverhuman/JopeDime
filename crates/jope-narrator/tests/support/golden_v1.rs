//! Synthetic contract fixture, not search measurements or runtime qualification.
use super::*;
use jope_narrator::Survivor;

fn facts() -> Vec<LineageFact> {
    let id = program("(pipe X (ridge lambda_frac=0.04))").sha256;
    let mut facts = vec![
        LineageFact::SearchStarted {
            cut: 0,
            engine: "synthetic renderer fixture".into(),
            train_rows: 10,
            hold_rows: 3,
            features: 2,
            population: 12,
            elites: 4,
            max_transforms: 4,
            islands: 2,
            migration_interval: 2,
            exploration_evaluations: 12,
            exploration_seconds: 60.0,
            rng_seed: u64::MAX,
            objective: Some("rmse".into()),
            incumbents: 0,
        },
        LineageFact::Admitted {
            cut: 0,
            generation: 1,
            island: 0,
            program_sha256: id.clone(),
            disposition: "admitted".into(),
        },
        LineageFact::Started {
            cut: 0,
            phase: "exploration".into(),
            generation: 1,
            island: Some(0),
            program_sha256: id.clone(),
        },
        LineageFact::Measured {
            cut: 0,
            phase: "exploration".into(),
            generation: 1,
            island: Some(0),
            program_sha256: id.clone(),
            origin: "fixture".into(),
            status: "measured".into(),
            rmse: Some(0.25),
            hold_r2: Some(0.8),
            train_r2: Some(0.9),
            objective_loss: Some(0.5),
            seconds: 0.01,
            failure: None,
        },
        LineageFact::GenerationClosed {
            cut: 0,
            generation: 1,
            island: 0,
            survivors: vec![Survivor {
                program_sha256: id.clone(),
                rmse: 0.25,
                score: 0.5,
            }],
            evaluations: 2,
            barren: 0,
        },
        LineageFact::Migrated {
            cut: 0,
            after_generation: 1,
            from_island: 0,
            to_island: 1,
            program_sha256: id.clone(),
            rmse: 0.25,
            admitted: true,
        },
        LineageFact::ExplorationStopped {
            cut: 0,
            reason: "fixture_budget".into(),
            generations: 1,
            evaluations: 2,
        },
        LineageFact::ComparisonStarted {
            cuts: vec![0, 1],
            program_sha256s: vec![id.clone()],
        },
        LineageFact::TapDropped { dropped: 3 },
        LineageFact::Selected {
            program_sha256: id.clone(),
            rule: "synthetic selection record".into(),
        },
        LineageFact::CandidatesOffered {
            program_sha256s: vec![id.clone()],
            rule: "synthetic offer record".into(),
        },
        LineageFact::Emitted {
            program_sha256: id.clone(),
            languages: vec!["rust".into()],
        },
        LineageFact::RunStopped {
            status: "completed".into(),
        },
        LineageFact::Prediction {
            status: "binding_unavailable".into(),
            program_sha256: None,
        },
        LineageFact::PredictionBound {
            status: "started".into(),
            model_sha256: "a".repeat(64),
            input_sha256: "b".repeat(64),
            rows: 3,
            seconds: None,
        },
        LineageFact::PredictionBound {
            status: "completed".into(),
            model_sha256: "a".repeat(64),
            input_sha256: "b".repeat(64),
            rows: 3,
            seconds: Some(0.02),
        },
    ];
    for n in 0..64 {
        let mut fact = draw(
            "(pipe X (ridge lambda_frac=0.02))",
            "(pipe X (ridge lambda_frac=0.04))",
        );
        if let LineageFact::Drawn { coordinate, .. } = &mut fact {
            *coordinate = format!("synthetic/variant/{n}");
        }
        facts.push(fact);
    }
    facts
}

#[test]
fn retained_v1_journal_pins_every_fact_kind_and_grammar_variant() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v1.jsonl");
    if std::env::var_os("JOPE_NARRATION_CREATE_V1_GOLDEN").is_some() {
        let dir = scratch();
        let path = dir.path().join("v1.jsonl");
        let (tap, writer) =
            FactJournal::create(&path, "synthetic-renderer-v1", Limits::default()).unwrap();
        for fact in facts() {
            assert_eq!(tap.emit(fact), EmitStatus::Queued);
        }
        writer.finish().unwrap();
        fs::create_dir_all(fixture.parent().unwrap()).unwrap();
        for (from, to) in [
            (&path, fixture.clone()),
            (
                &jope_narrator::commit_path(&path),
                jope_narrator::commit_path(&fixture),
            ),
        ] {
            let bytes = fs::read(from).unwrap();
            if to.exists() {
                assert_eq!(
                    fs::read(&to).unwrap(),
                    bytes,
                    "v1 golden is immutable; introduce a new renderer revision"
                );
            } else {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(to)
                    .unwrap()
                    .write_all(&bytes)
                    .unwrap();
            }
        }
    }
    let page = read_journal(&fixture, None, 1024, 16 << 20).unwrap();
    assert!(!page.partial_tail);
    let kinds = page
        .events
        .iter()
        .map(|e| e.fact.kind())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        kinds.len(),
        16,
        "all v1 fact kinds must retain their original wording"
    );
    let variants = page
        .events
        .iter()
        .filter_map(|e| e.narration.variant)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(variants, (0..8).collect());
    assert!(page
        .events
        .iter()
        .all(|e| e.renderer == "controlled-search-commentary/1"));
    // The versioned v1 reader verifies every recorded clause without rewriting text.
    assert_eq!(
        jope_narrator::read_commit(&fixture).unwrap().records,
        page.events.len().to_string()
    );
}
