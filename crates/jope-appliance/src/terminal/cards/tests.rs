use super::*;
use jope_api::{
    ids::{AlgorithmId, ArtifactId, RunId},
    AlgorithmAcceptedBody, BestAlgorithm, BudgetUsed, EventBody, ExportCompletedBody, FinalScores,
    Nondeterminism, Plan, PublicEvent, RunFinishedBody,
};

fn test_event(sequence: u64, seconds: u8, body: EventBody) -> PublicEvent {
    PublicEvent {
        run_id: RunId::default(),
        sequence: DecimalU64(sequence),
        time: format!("2026-10-30T10:00:{seconds:02}Z"),
        event: body,
    }
}

fn sample_events() -> Vec<PublicEvent> {
    let algorithm: AlgorithmId = "alg_0007".parse().unwrap();
    let artifact: ArtifactId = "art_0123456789abcdef01234567".parse().unwrap();
    let plan = Plan {
        plan_digest: "plan".to_owned(),
        task: jope_api::ResolvedTask {
            value: jope_api::Task::Regression,
            inferred: false,
            reason: "requested".to_owned(),
        },
        objective: jope_api::ResolvedObjective {
            name: "RMSE".to_owned(),
            digest: "objective".to_owned(),
            direction: Direction::Minimize,
            fitting: "fit".to_owned(),
        },
        rows: jope_api::RowCounts {
            total: DecimalU64(20640),
            fit: DecimalU64(16512),
            holdout: DecimalU64(4128),
            test: None,
        },
        effort: jope_api::ResolvedEffort {
            name: jope_api::Effort::Medium,
            evaluations: DecimalU64(128),
            wall_seconds: DecimalU64(60),
            cv_folds: 5,
            rounds_in_flight: 1,
        },
        guidance: jope_api::ResolvedGuidance {
            value: jope_api::Guidance::Learned,
            prior_only: false,
            reason: "qualified".to_owned(),
        },
        landmarks: jope_api::ResolvedLandmarks {
            mode: jope_api::LandmarkMode::Full,
            sample_rows: 1024,
        },
        languages: jope_api::ResolvedLanguages {
            requested: vec![Language::Python],
            excluded_operations: Vec::new(),
            intersection_empty: false,
        },
        placement: jope_api::Placement {
            dope: "cpu".to_owned(),
            ranker: "cpu".to_owned(),
            evaluation: "cpu".to_owned(),
            workers: 7,
        },
        limits: jope_api::RunLimits {
            eval_timeout_s: 30,
            worker_rss_bytes: DecimalU64(1073741824),
        },
        estimates: jope_api::ResourceEstimates {
            peak_ram_bytes: DecimalU64(1181116006),
            disk_bytes: DecimalU64(0),
        },
        warnings: Vec::new(),
    };
    let first = AlgorithmAcceptedBody {
        algorithm_id: algorithm.clone(),
        metric: "RMSE".to_owned(),
        cv_value: 0.742,
        previous_value: None,
        summary: "linear baseline (ridge)".to_owned(),
        model_artifact: Some(artifact.clone()),
        excerpt: String::new(),
    };
    let improved = AlgorithmAcceptedBody {
            algorithm_id: algorithm.clone(),
            metric: "RMSE".to_owned(),
            cv_value: 0.487,
            previous_value: Some(0.512),
            summary: "log-scaled income, deeper trees".to_owned(),
            model_artifact: Some(artifact.clone()),
            excerpt: "def predict(row):\n    income = row[\"income\"]\n    scaled = math.log1p(income)\n    return scaled\n".to_owned(),
        };
    let warning = Warning {
        code: "GPU_FALLBACK".to_owned(),
        severity: Severity::Warn,
        state: WarningState::Raised,
        host: true,
        value: None,
        limit: None,
        message: "GPU unavailable; continuing on CPU".to_owned(),
        remedy: "check device availability".to_owned(),
    };
    let mut events = vec![
        test_event(1, 0, EventBody::Plan(Box::new(plan))),
        test_event(2, 1, EventBody::AlgorithmAccepted(first)),
        test_event(3, 2, EventBody::AlgorithmAccepted(improved)),
        test_event(4, 3, EventBody::Warning(warning.clone())),
        test_event(
            5,
            4,
            EventBody::Warning(Warning {
                state: WarningState::Cleared,
                ..warning
            }),
        ),
        test_event(
            6,
            5,
            EventBody::Reasoning(jope_api::ReasoningBody {
                stage: "search".to_owned(),
                algorithm_id: Some(algorithm.clone()),
                summary: "Testing a log transform on skewed income features…".to_owned(),
                evidence: None,
            }),
        ),
        test_event(
            7,
            6,
            EventBody::ExportCompleted(ExportCompletedBody {
                artifact_id: artifact,
                kind: ExportKind::Source,
                languages: vec![Language::Python, Language::Rust],
                parity_status: ParityStatus::Passed,
            }),
        ),
        test_event(
            8,
            7,
            EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
                experiment_id: "exp_1".to_owned(),
                outcome: ExperimentOutcome::NotImproved,
                metric: "RMSE".to_owned(),
                value: Some(0.5),
                failure: None,
                attempt_receipt: None,
            }),
        ),
        test_event(
            9,
            8,
            EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
                experiment_id: "exp_2".to_owned(),
                outcome: ExperimentOutcome::NotImproved,
                metric: "RMSE".to_owned(),
                value: Some(0.51),
                failure: None,
                attempt_receipt: None,
            }),
        ),
    ];
    events.push(test_event(
        10,
        9,
        EventBody::RunFinished(RunFinishedBody {
            completion_reason: CompletionReason::Converged,
            champion: Some(BestAlgorithm {
                algorithm_id: algorithm,
                metric: "RMSE".to_owned(),
                cv_value: 0.487,
                model_artifact: Some("art_0123456789abcdef01234567".parse().unwrap()),
            }),
            scores: FinalScores {
                cv: Some(0.487),
                holdout: Some(0.503),
                test: Some(0.512),
            },
            budget_used: BudgetUsed {
                evaluations: DecimalU64(41),
                wall_seconds: "23.4".to_owned(),
            },
            nondeterminism: Nondeterminism {
                count: DecimalU64(0),
                experiment_ids: Vec::new(),
            },
            evaluation_receipt: None,
        }),
    ));
    events
}

fn colorless_theme() -> Theme {
    use crate::terminal::ColorLevel;
    Theme::new(crate::terminal::ThemeMode::Dark, ColorLevel::None, None)
}

fn color_theme() -> Theme {
    use crate::terminal::ColorLevel;
    Theme::new(crate::terminal::ThemeMode::Dark, ColorLevel::Ansi16, None)
}

fn symbols() -> Symbols {
    crate::terminal::symbols_for_locale(Some("en_US.UTF-8"))
}

#[test]
fn event_cards_match_golden_strings_at_80_and_120_columns_with_and_without_color() {
    let events = sample_events();
    for (width, expected) in [(80, golden_80()), (120, golden_120())] {
        let plain = render_transcript(&events, width, colorless_theme(), symbols());
        assert_eq!(plain, expected, "plain width {width}");
        let colored = render_transcript(&events, width, color_theme(), symbols());
        let expected_colored = color_golden(expected);
        assert_eq!(colored, expected_colored, "color width {width}");
        assert!(plain.iter().all(|line| line.chars().count() <= width));
    }
}

#[test]
fn sanitizes_user_text_and_caps_excerpts_at_six_lines() {
    let mut events = sample_events();
    if let EventBody::AlgorithmAccepted(accepted) = &mut events[2].event {
        accepted.summary = "safe\u{1b}[31m text".to_owned();
        accepted.excerpt = "one\ntwo\nthree\nfour\nfive\nsix\nseven".to_owned();
    }
    let cards = render_transcript(&events, 80, colorless_theme(), symbols());
    assert!(cards.iter().all(|line| !line.contains('\u{1b}')));
    let excerpt_lines = cards
        .iter()
        .filter(|line| line.starts_with("   │ "))
        .count();
    assert_eq!(excerpt_lines, 6);
    assert!(cards.iter().any(|line| line.contains("safe text")));
}

#[test]
fn no_improvement_summaries_are_throttled_to_ten_second_windows() {
    let event = |sequence: u64, time: &str| PublicEvent {
        run_id: RunId::default(),
        sequence: DecimalU64(sequence),
        time: time.to_owned(),
        event: EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
            experiment_id: format!("exp_{sequence}"),
            outcome: ExperimentOutcome::NotImproved,
            metric: "RMSE".to_owned(),
            value: Some(0.5),
            failure: None,
            attempt_receipt: None,
        }),
    };
    let events = [
        event(1, "2026-10-30T10:00:00Z"),
        event(2, "2026-10-30T10:00:02Z"),
        event(3, "2026-10-30T10:00:11Z"),
        event(4, "2026-10-30T10:00:12Z"),
        event(5, "2026-10-30T10:00:21Z"),
        event(6, "2026-10-30T10:00:22Z"),
    ];
    let cards = render_transcript(&events, 80, colorless_theme(), symbols());
    assert_eq!(
        cards,
        [
            "· 2 experiments in the last 10 s, no improvement",
            "· 3 experiments in the last 10 s, no improvement"
        ]
    );
}

fn golden_80() -> Vec<String> {
    vec![
        "Plan   regression · RMSE (minimize) · effort medium (≤128 experiments, ≤60 s) ·"
            .to_owned(),
        "       guidance learned".to_owned(),
        "       landmarks full (1,024-row sample) · languages python".to_owned(),
        "       FIT 16,512 rows · HOLDOUT 4,128 rows · 7 workers · est. peak RAM 1.1 GB".to_owned(),
        "✓ First model linear baseline (ridge) · CV RMSE 0.742".to_owned(),
        "✓ Improved log-scaled income, deeper trees · CV RMSE 0.487 · −4.9%".to_owned(),
        "   ┌ python · alg_0007".to_owned(),
        "   │ def predict(row):".to_owned(),
        "   │ income = row[\"income\"]".to_owned(),
        "   │ scaled = math.log1p(income)".to_owned(),
        "   │ return scaled".to_owned(),
        format!("   └{}", "─".repeat(60)),
        "⚠ Warning raised GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
        "✓ Warning cleared GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
        "   Thinking Testing a log transform on skewed income features…".to_owned(),
        "✓ Export art_0123456789abcdef01234567 · source · python, rust · parity passed".to_owned(),
        "· 2 experiments in the last 10 s, no improvement".to_owned(),
        "✓ Run finished converged · champion alg_0007 (RMSE 0.487) · CV 0.487 · HOLDOUT".to_owned(),
        "               0.503 · TEST 0.512".to_owned(),
    ]
}

fn golden_120() -> Vec<String> {
    vec![
            "Plan   regression · RMSE (minimize) · effort medium (≤128 experiments, ≤60 s) · guidance learned".to_owned(),
            "       landmarks full (1,024-row sample) · languages python".to_owned(),
            "       FIT 16,512 rows · HOLDOUT 4,128 rows · 7 workers · est. peak RAM 1.1 GB".to_owned(),
            "✓ First model linear baseline (ridge) · CV RMSE 0.742".to_owned(),
            "✓ Improved log-scaled income, deeper trees · CV RMSE 0.487 · −4.9%".to_owned(),
            "   ┌ python · alg_0007".to_owned(),
            "   │ def predict(row):".to_owned(),
            "   │ income = row[\"income\"]".to_owned(),
            "   │ scaled = math.log1p(income)".to_owned(),
            "   │ return scaled".to_owned(),
            format!("   └{}", "─".repeat(60)),
            "⚠ Warning raised GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "✓ Warning cleared GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "   Thinking Testing a log transform on skewed income features…".to_owned(),
            "✓ Export art_0123456789abcdef01234567 · source · python, rust · parity passed".to_owned(),
            "· 2 experiments in the last 10 s, no improvement".to_owned(),
            "✓ Run finished converged · champion alg_0007 (RMSE 0.487) · CV 0.487 · HOLDOUT 0.503 · TEST 0.512".to_owned(),
        ]
}

fn color_golden(plain: Vec<String>) -> Vec<String> {
    plain
        .into_iter()
        .map(|line| {
            if line.starts_with('✓') {
                line.replacen('✓', "\x1b[32m✓\x1b[0m", 1)
            } else if line.starts_with('⚠') {
                line.replacen('⚠', "\x1b[33m⚠\x1b[0m", 1)
            } else if line.starts_with('·') {
                line.replacen('·', "\x1b[90m·\x1b[0m", 1)
            } else {
                line
            }
        })
        .collect()
}
