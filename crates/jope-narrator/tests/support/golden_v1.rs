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
    branch_fixture();
    retained_v2_fixture();
    current_fixture();
    assert!(
        std::env::var_os("JOPE_NARRATION_CREATE_V1_GOLDEN").is_none(),
        "v1 goldens are frozen; create a new renderer revision"
    );
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v1.jsonl");
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

fn branch_facts() -> Vec<(LineageFact, Links)> {
    let mut rows = Vec::new();
    let mut add = |name: &str, parent: &str, child: &str| {
        let mut fact = draw(parent, child);
        if let LineageFact::Drawn { coordinate, .. } = &mut fact {
            *coordinate = format!("synthetic/branch/{name}");
        }
        rows.push((fact, Links::default()));
    };
    for (name, parent, child) in [
        ("unchanged", "(pipe X (ridge))", "(pipe X (ridge))"),
        (
            "whole_program",
            "(sym raw \"c0 + 1\")",
            "(sym affine \"c0 * 2\")",
        ),
        ("insert", "(pipe X (ridge))", "(pipe X (rank) (ridge))"),
        ("delete", "(pipe X (rank) (ridge))", "(pipe X (ridge))"),
        (
            "replace",
            "(pipe X (rank) (ridge))",
            "(pipe X (robust_scale) (ridge))",
        ),
        (
            "swap",
            "(pipe X (rank) (robust_scale) (ridge))",
            "(pipe X (robust_scale) (rank) (ridge))",
        ),
        (
            "clamped",
            "(pipe X (hinge knots=40) (ridge))",
            "(pipe X (hinge knots=50) (ridge))",
        ),
        (
            "resolved",
            "(pipe X (cheb degree=2) (ridge))",
            "(pipe X (cheb degree=8) (ridge))",
        ),
        (
            "clipped",
            "(pipe X (winsor_clip q=0.5) (ridge))",
            "(pipe X (winsor_clip q=0.8) (ridge))",
        ),
        (
            "clip_change",
            "(pipe X (winsor_clip q=0.05) (ridge))",
            "(pipe X (winsor_clip q=0.1) (ridge))",
        ),
        (
            "seed",
            "(pipe X (rand_proj k=4 seed=1) (ridge))",
            "(pipe X (rand_proj k=4 seed=2) (ridge))",
        ),
        (
            "recipe",
            "(pipe X (rank) (ridge))",
            "(pipe X (cheb degree=4) (ridge lambda_frac=0.1))",
        ),
        ("logistic", "(pipe X (ridge))", "(pipe X (logistic))"),
        ("gbm", "(pipe X (ridge))", "(pipe X (gbm rounds=10))"),
        ("ensemble", "(pipe X (ridge))", "(pipe X (ensemble))"),
        // Invalid captured syntax is a limitation, never a guessed exact change.
        (
            "analysis_unavailable",
            "recorded unsupported syntax",
            "(pipe X (ridge))",
        ),
    ] {
        add(name, parent, child);
    }
    let before = format!("(pipe X {} (ridge))", "(cheb degree=1) ".repeat(7));
    let after = format!("(pipe X {} (ridge))", "(cheb degree=8) ".repeat(7));
    add("many_edits", &before, &after);
    for equal in [false, true] {
        let mut fact = draw("(pipe X (ridge))", "(pipe X (rank) (ridge))");
        if let LineageFact::Drawn {
            parents,
            origin,
            coordinate,
            ..
        } = &mut fact
        {
            parents.push(if equal {
                parents[0].clone()
            } else {
                program("(pipe X (robust_scale) (ridge))")
            });
            *origin = "crossover".into();
            *coordinate = format!("synthetic/branch/crossover/{equal}");
        }
        rows.push((fact, Links::default()));
    }
    let mut no_parent = draw("(pipe X (ridge))", "(pipe X (ridge))");
    if let LineageFact::Drawn {
        parents, origin, ..
    } = &mut no_parent
    {
        parents.clear();
        *origin = "random".into();
    }
    rows.push((no_parent, Links::default()));
    let mut comparison = draw("(pipe X (ridge))", "(pipe X (ridge lambda_frac=0.1))");
    if let LineageFact::Drawn { phase, .. } = &mut comparison {
        *phase = "comparison".into();
    }
    rows.push((comparison, Links::default()));
    let mut failed = draw("(pipe X (ridge))", "(pipe X (ridge))");
    if let LineageFact::Drawn { child, outcome, .. } = &mut failed {
        *child = None;
        *outcome = "operator_failed".into();
    }
    rows.push((failed, Links::default()));
    for failed in [false, true] {
        rows.push((
            LineageFact::Measured {
                cut: 0,
                phase: "comparison".into(),
                generation: 1,
                island: None,
                program_sha256: program("(pipe X (ridge))").sha256,
                origin: "remeasure".into(),
                status: if failed { "fit_failed" } else { "measured" }.into(),
                rmse: (!failed).then_some(0.25),
                hold_r2: None,
                train_r2: None,
                objective_loss: None,
                seconds: 0.02,
                failure: failed.then(|| "synthetic failure".into()),
            },
            Links {
                observation_sha256: Some("c".repeat(64)),
                ..Links::default()
            },
        ));
    }
    rows.push((
        LineageFact::PredictionBound {
            status: "failed".into(),
            model_sha256: "a".repeat(64),
            input_sha256: "b".repeat(64),
            rows: 3,
            seconds: Some(0.01),
        },
        Links::default(),
    ));
    rows
}

fn branch_fixture() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v1-branches.jsonl");
    let page = read_journal(&fixture, None, 1024, 16 << 20).unwrap();
    assert!(!page.partial_tail);
    assert_eq!(page.events.len(), branch_facts().len());
    let text = page
        .events
        .iter()
        .map(|e| e.narration.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for clause in [
        "comparison pool",
        "unchanged from",
        "semantic analysis is unavailable",
        "additional exact edits",
        "first two ordered parent roles",
        "unique crossover splice",
        "without parent references",
        "construction returned no program",
        "draw outcome",
        "retained boundary data",
        "insert ",
        "remove ",
        "replaces ",
        "swap stage[",
        "seeded construction",
        "changed recipe",
        "both requests resolve",
        "planner resolves",
        "logistic head to ridge",
        "gbm head to histgbm",
        "ensemble head to ensemble_ridge_gbm",
        "exact structural analysis is unavailable",
        "prediction failed",
    ] {
        assert!(text.contains(clause), "branch golden missing {clause}");
    }
    assert_eq!(
        jope_narrator::read_commit(&fixture).unwrap().records,
        page.events.len().to_string()
    );
}

fn retained_v2_fixture() {
    assert!(
        std::env::var_os("JOPE_NARRATION_CREATE_V2_GOLDEN").is_none(),
        "v2 golden is frozen; create a new renderer revision"
    );
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v2.jsonl");
    let page = read_journal(&fixture, None, 1024, 16 << 20).unwrap();
    assert_eq!(page.events.len(), 114);
    assert!(!page.partial_tail);
    assert!(page
        .events
        .iter()
        .all(|e| e.renderer == "controlled-search-commentary/2"));
    assert_eq!(
        jope_narrator::read_commit(&fixture).unwrap().records,
        page.events.len().to_string()
    );
}

fn current_fixture() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/renderer-v3.jsonl");
    let mut rows = facts()
        .into_iter()
        .map(|f| (f, Links::default()))
        .collect::<Vec<_>>();
    rows.extend(branch_facts());
    for disposition in [
        "admitted",
        "duplicate",
        "evaluations",
        "seconds",
        "cancelled",
    ] {
        rows.push((
            LineageFact::Admitted {
                cut: 0,
                generation: 0,
                island: 0,
                program_sha256: program("(pipe X (ridge))").sha256,
                disposition: disposition.into(),
            },
            Links::default(),
        ));
    }
    for reason in [
        "evaluations",
        "seconds",
        "cancelled",
        "search_space_exhausted",
    ] {
        rows.push((
            LineageFact::ExplorationStopped {
                cut: 0,
                reason: reason.into(),
                evaluations: 4,
                generations: 1,
            },
            Links::default(),
        ));
    }
    if std::env::var_os("JOPE_NARRATION_CREATE_V3_GOLDEN").is_some() {
        let dir = scratch();
        let path = dir.path().join("v3.jsonl");
        let (tap, writer) =
            FactJournal::create(&path, "synthetic-renderer-v3", Limits::default()).unwrap();
        for (fact, links) in &rows {
            assert_eq!(
                tap.emit_linked(fact.clone(), links.clone()),
                EmitStatus::Queued
            );
        }
        writer.finish().unwrap();
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
                    "v3 golden is immutable; introduce a new renderer revision"
                );
            } else {
                fs::OpenOptions::new()
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
    assert_eq!(page.events.len(), 114);
    assert_eq!(
        page.events
            .iter()
            .map(|e| e.fact.kind())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        16
    );
    assert_eq!(
        page.events
            .iter()
            .filter_map(|e| e.narration.variant)
            .collect::<std::collections::BTreeSet<_>>(),
        (0..8).collect()
    );
    assert!(page
        .events
        .iter()
        .all(|e| e.renderer == "controlled-search-commentary/3"));
    let text = page
        .events
        .iter()
        .map(|e| e.narration.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for clause in [
        "admitted for evaluation",
        "duplicate; not admitted for a new evaluation",
        "not admitted: the evaluation bound was reached",
        "not admitted: the active-seconds bound was reached",
        "cancelled; this decision does not establish a physical evaluation start",
        "1 exact endpoint edit is retained",
        "A random proposal",
        "Construction returned no program",
        "The captured program is unchanged",
        "The program changed;",
        "The recorded recipe is available;",
        "the evaluation bound was reached",
        "the active-seconds bound was reached",
        "the source run was cancelled",
        "consecutive generations spent no evaluations",
    ] {
        assert!(text.contains(clause), "v3 golden missing {clause}");
    }
    assert!(!text.contains("1 exact endpoint edits"));
    for event in &page.events {
        let (predicate, key, token) = match &event.fact {
            LineageFact::Admitted { disposition, .. } => ("admission", "disposition", disposition),
            LineageFact::ExplorationStopped { reason, .. } => {
                ("exploration_stopped", "reason", reason)
            }
            _ => continue,
        };
        let claim = event
            .narration
            .claims
            .iter()
            .find(|claim| claim.predicate == predicate)
            .unwrap();
        assert_eq!(
            claim.details[key],
            token.as_str(),
            "exact source decision token must remain in claims"
        );
    }
}
