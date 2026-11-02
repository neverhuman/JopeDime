use super::*;
use std::collections::BTreeSet;

const RUN: &str = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0";
const PROBLEM: ProblemDigest = ProblemDigest([7; 32]);

fn projector() -> Projector {
    Projector::new(RUN.parse().unwrap(), PROBLEM)
}
fn context() -> ScoreContext {
    ScoreContext {
        problem: PROBLEM,
        folds: FoldsDigest([3; 32]),
        metric: "rmse".into(),
        scope: "cv_fit_5fold".into(),
    }
}
fn score(value: f64) -> Score {
    Score {
        context: context(),
        value,
    }
}
fn entry() -> CatalogEntry<'static> {
    CatalogEntry {
        public_name: "log scaling",
        purpose: "reduce skew in income features",
    }
}
fn proposal(ordinal: u64) -> AdmittedProposal<'static> {
    AdmittedProposal {
        ordinal: EvaluationOrdinal(ordinal),
        template: Template::StageInsert,
        event_digest: [ordinal as u8; 32],
        entry: entry(),
        details: EventDetails::default(),
    }
}
fn champion(committed: bool, value: f64) -> Champion<'static> {
    Champion {
        algorithm_id: "alg_0007".parse().unwrap(),
        head: CatalogEntry {
            public_name: "gradient-boosted trees",
            purpose: "capture interactions",
        },
        committed,
        score: score(value),
    }
}
fn reasoning_of(body: EventBody) -> ReasoningBody {
    match body {
        EventBody::Reasoning(body) => body,
        other => panic!("expected reasoning, got {other:?}"),
    }
}

#[test]
fn every_claim_level_requires_its_evidence() {
    let p = projector();
    let measured = |rung| Claim::Measured {
        evaluation: CompletedEvaluation {
            ordinal: EvaluationOrdinal(1),
            rung,
            score: score(0.493),
        },
        best: Some(score(0.487)),
    };
    assert_eq!(
        p.project(measured(0)).unwrap_err(),
        ProjectionError::MissingEvidence("a completed rung-1 evaluation")
    );
    assert_eq!(
        reasoning_of(p.project(measured(1)).unwrap()).stage,
        "measured"
    );

    let accepted = |committed| Claim::Accepted {
        champion: champion(committed, 0.487),
        previous: Some(score(0.512)),
    };
    assert_eq!(
        p.project(accepted(false)).unwrap_err(),
        ProjectionError::MissingEvidence("a committed champion")
    );
    let body = reasoning_of(p.project(accepted(true)).unwrap());
    assert_eq!(body.stage, "accepted");
    assert_eq!(body.algorithm_id.unwrap().to_string(), "alg_0007");
    assert_eq!(
        body.summary,
        "New best: gradient-boosted trees (CV rmse 0.487, -4.9%)."
    );

    let finalization = |rows, excluded| {
        Claim::Final(Finalization {
            holdout: score(0.491),
            holdout_rows: rows,
            excluded_from_search: excluded,
        })
    };
    for (rows, excluded) in [(0, true), (4128, false)] {
        assert!(matches!(
            p.project(finalization(rows, excluded)).unwrap_err(),
            ProjectionError::MissingEvidence(_)
        ));
    }
    assert_eq!(
        reasoning_of(p.project(finalization(4128, true)).unwrap()).summary,
        "Holdout rmse 0.491 on 4128 rows not used during search."
    );

    static BOTH: [Language; 2] = [Language::Python, Language::Rust];
    let export = |languages: &'static [Language], rows, within| {
        Claim::Export(ParityReceipt {
            languages,
            check_rows: rows,
            within_tolerance: within,
        })
    };
    for claim in [
        export(&BOTH, 64, false),
        export(&[], 64, true),
        export(&BOTH, 0, true),
    ] {
        assert!(matches!(
            p.project(claim).unwrap_err(),
            ProjectionError::MissingEvidence(_)
        ));
    }
    assert_eq!(
        reasoning_of(p.project(export(&BOTH, 64, true)).unwrap()).summary,
        "Python and Rust predictors match the engine within the release tolerance on 64 check rows."
    );

    let hypothesis = reasoning_of(p.project(Claim::Hypothesis(proposal(5))).unwrap());
    assert_eq!(hypothesis.stage, "hypothesis");
    assert!(hypothesis.summary.contains("log scaling"));
    let unknown = reasoning_of(p.project(Claim::Unknown).unwrap());
    assert_eq!(
        (unknown.stage.as_str(), unknown.summary.as_str()),
        ("unknown", "Evaluating a new algorithm variant.")
    );
}

#[test]
fn accepted_percentage_handles_finite_extremes() {
    for (before, after, percentage) in [
        (1e308, -1e308, Some("-200.0%")),
        (-1e308, 1e308, Some("+200.0%")),
        (f64::MIN_POSITIVE, f64::MAX, None),
    ] {
        let body = reasoning_of(
            projector()
                .project(Claim::Accepted {
                    champion: champion(true, after),
                    previous: Some(score(before)),
                })
                .unwrap(),
        );
        assert!(!body.summary.contains("inf") && !body.summary.contains("NaN"));
        match percentage {
            Some(expected) => assert!(body.summary.contains(expected)),
            None => assert!(!body.summary.contains('%')),
        }
        let evidence = body.evidence.unwrap();
        assert!(evidence.comparable);
        assert_eq!(
            (evidence.before, evidence.after),
            (Some(before), Some(after))
        );
    }
}

#[test]
fn mismatched_comparisons_are_unavailable_and_matching_ones_compare() {
    let current = score(0.493);
    let matching = compare(&current, Some(&score(0.487))).unwrap();
    assert!(matching.comparable);
    assert_eq!(
        (matching.before, matching.after),
        (Some(0.487), Some(0.493))
    );

    let mut baselines = Vec::new();
    for change in 0..4 {
        let mut context = context();
        match change {
            0 => context.problem = ProblemDigest([8; 32]),
            1 => context.folds = FoldsDigest([4; 32]),
            2 => context.metric = "mae".into(),
            _ => context.scope = "cv_fit_3fold".into(),
        }
        baselines.push(Score {
            context,
            value: 0.487,
        });
    }
    for baseline in baselines.iter().map(Some).chain([None]) {
        let evidence = compare(&current, baseline).unwrap();
        assert!(!evidence.comparable);
        assert_eq!(evidence.before, None);
        let body = reasoning_of(
            projector()
                .project(Claim::Measured {
                    evaluation: CompletedEvaluation {
                        ordinal: EvaluationOrdinal(2),
                        rung: 1,
                        score: current.clone(),
                    },
                    best: baseline.cloned(),
                })
                .unwrap(),
        );
        assert_eq!(
            body.summary,
            "Cross-validated rmse 0.493; comparison unavailable."
        );
        assert!(!body.evidence.unwrap().comparable);
    }
    let accepted = reasoning_of(
        projector()
            .project(Claim::Accepted {
                champion: champion(true, 0.487),
                previous: Some(baselines[1].clone()),
            })
            .unwrap(),
    );
    assert!(accepted.summary.ends_with("; comparison unavailable)."));
    assert!(compare(&score(f64::NAN), None).is_err());
}

#[test]
fn claims_about_this_run_must_use_its_problem_digest() {
    let mut foreign = score(0.5);
    foreign.context.problem = ProblemDigest([9; 32]);
    let p = projector();
    assert_eq!(
        p.project(Claim::Measured {
            evaluation: CompletedEvaluation {
                ordinal: EvaluationOrdinal(3),
                rung: 1,
                score: foreign.clone(),
            },
            best: None,
        })
        .unwrap_err(),
        ProjectionError::ProblemMismatch
    );
    let mut champion = champion(true, 0.5);
    champion.score = foreign.clone();
    assert_eq!(
        p.project(Claim::Accepted {
            champion,
            previous: None
        })
        .unwrap_err(),
        ProjectionError::ProblemMismatch
    );
    assert_eq!(
        p.project(Claim::Final(Finalization {
            holdout: foreign,
            holdout_rows: 10,
            excluded_from_search: true,
        }))
        .unwrap_err(),
        ProjectionError::ProblemMismatch
    );
}

#[test]
fn experiment_ids_are_opaque_deterministic_and_unique() {
    let p = projector();
    let other = Projector::new("run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap(), PROBLEM);
    let other_problem = Projector::new(RUN.parse().unwrap(), ProblemDigest([8; 32]));
    let mut seen = BTreeSet::new();
    for ordinal in 0..10_000 {
        let id = p.experiment_id(EvaluationOrdinal(ordinal));
        assert_eq!(id.len(), 20);
        assert!(id.starts_with("exp_"));
        assert!(id[4..]
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert_ne!(id[4..], format!("{ordinal:016x}"));
        assert_eq!(id, p.experiment_id(EvaluationOrdinal(ordinal)));
        assert_eq!(id, other.experiment_id(EvaluationOrdinal(ordinal)));
        assert_ne!(id, other_problem.experiment_id(EvaluationOrdinal(ordinal)));
        assert!(seen.insert(id));
    }
    match p.experiment_started(&proposal(9)).unwrap() {
        EventBody::ExperimentStarted(body) => {
            assert_eq!(body.experiment_id, p.experiment_id(EvaluationOrdinal(9)));
            assert!(body.hypothesis.contains("log scaling"));
        }
        other => panic!("expected experiment.started, got {other:?}"),
    }
}

#[test]
fn banned_wording_is_refused() {
    let mut banned = proposal(1);
    banned.entry.public_name = "parent pipeline";
    assert!(matches!(
        projector().project(Claim::Hypothesis(banned)),
        Err(ProjectionError::Wording(_))
    ));
    assert!(projector().experiment_started(&banned).is_err());
}

/// Type inspection over the live `jope_api` schemas: no public payload has a
/// field for programs, operators, lineage or search positions (§8.1 rule 2).
#[test]
fn no_public_payload_schema_has_a_field_for_programs_lineage_or_search_positions() {
    const FORBIDDEN: &[&str] = &[
        "dsl",
        "program",
        "operator",
        "parent",
        "lineage",
        "round",
        "population",
        "elite",
        "genome",
        "generation",
        "slot",
        "attempt",
        "island",
        "offspring",
    ];
    // Planned pipeline depth is a count, not a round index. The Q1 receipt
    // contract also exposes only a hashed candidate identity and its actual
    // charged-attempt index/count; it does not expose source programs or
    // search positions from discovery.
    const ALLOWED: &[&str] = &[
        "rounds_in_flight",
        "attempt",
        "attempts",
        "attempt_receipt",
        "program_sha256",
        "champion_program",
    ];
    fn collect(value: &serde_json::Value, names: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    if key == "properties" {
                        if let serde_json::Value::Object(properties) = child {
                            names.extend(properties.keys().cloned());
                        }
                    }
                    collect(child, names);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| collect(v, names)),
            _ => {}
        }
    }
    let mut names = BTreeSet::new();
    let mut inspected = BTreeSet::new();
    for operation in jope_api::registry::OPERATIONS {
        for payload in operation
            .request_types
            .iter()
            .chain(operation.response_types)
        {
            if let Ok(schema) = payload.resolve_schema() {
                inspected.insert(payload.name);
                collect(&serde_json::to_value(schema).unwrap(), &mut names);
            }
        }
    }
    for required in ["PublicEvent", "TelemetryEvent", "RunSnapshot", "Algorithm"] {
        assert!(
            inspected.contains(required),
            "{required} schema not inspected"
        );
    }
    assert!(names.contains("summary") && names.contains("experiment_id"));
    for name in names.iter().filter(|n| !ALLOWED.contains(&n.as_str())) {
        for token in name.to_ascii_lowercase().split('_') {
            for forbidden in FORBIDDEN {
                assert!(
                    token != *forbidden && token != format!("{forbidden}s"),
                    "public field `{name}` can carry `{forbidden}`"
                );
            }
        }
    }
}
