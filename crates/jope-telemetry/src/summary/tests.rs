use super::*;
use crate::flywheel::TaskType;
use serde_json::json;

#[test]
fn summary_reports_spans_diminishing_deltas_and_missing() {
    let sc = [
        ScalarIn {
            metric: "train.loss",
            group: None,
            step: Some(1),
            value: 2.0,
        },
        ScalarIn {
            metric: "train.loss",
            group: None,
            step: Some(2),
            value: f64::NAN,
        },
        ScalarIn {
            metric: "step.wall",
            group: None,
            step: Some(1),
            value: 10.0,
        },
        ScalarIn {
            metric: "step.span.fwd",
            group: None,
            step: Some(1),
            value: 6.0,
        },
        ScalarIn {
            metric: "step.unattributed",
            group: None,
            step: Some(1),
            value: 4.0,
        },
    ];
    let rounds: Vec<Value> = [(1, -0.179), (2, -0.070), (3, -0.003)]
        .iter()
        .map(|(r, d)| {
            json!({"schema": crate::flywheel::ROUND_SCHEMA, "round": r, "generation": r,
                       "target_deltas": {"mean_strat300_evo_fitness_delta": d, "count.x": 1.0},
                       "arm_contrasts": {}})
        })
        .collect();
    let guard = json!({"hashes_equal": true, "composite_sha256": "abc", "violations": []});
    let mut recs: Vec<(&str, &Value)> = rounds
        .iter()
        .map(|r| (crate::flywheel::ROUND_SCHEMA, r))
        .collect();
    recs.push(("jope.telemetry.dope-guard/v1", &guard));
    let s = summarize(sc, recs);
    assert_eq!(s.training.steps, 2);
    assert_eq!(s.training.loss.nonfinite, 1);
    assert_eq!(
        s.training.host_span_share,
        vec![("step.span.fwd".to_owned(), 0.6)]
    );
    assert_eq!(s.training.unattributed_share, Some(0.4));
    assert_eq!(s.evolution.rounds, vec![1, 2, 3]);
    assert_eq!(
        s.evolution.diminishing,
        vec!["mean_strat300_evo_fitness_delta".to_owned()]
    );
    assert_eq!(s.dope.all_equal, Some(true));
    assert!(s.missing.contains_key("val.loss") && !s.missing.contains_key("dope.freeze"));
    let text = s.render_text();
    assert!(text.contains("DIMINISHING") && text.contains("step.span.fwd"));
    assert!(s.elo.is_none() && s.missing.contains_key("fly.elo"));
    for (metric, label) in TOK_S_LABELS {
        assert!(s.missing.contains_key(metric));
        assert!(text.contains(label), "{label} prints even without values");
    }
    assert!(s.missing.contains_key(TOK_S_REASONING));
    assert!(text.contains("reasoning tok/s            -  unavailable"));
}

#[test]
fn reasoning_tok_s_is_the_last_run_total_with_its_tokenizer() {
    // A periodic running total, then the final one: the last emission is the run's value.
    let sc = [
        (TOK_S_REASONING, 5, 50.0),
        (REASONING_TOKENS, 5, 200.0),
        (REASONING_WALL_S, 5, 4.0),
        (TOK_S_REASONING, 9, 100.0),
        (REASONING_TOKENS, 9, 1_000.0),
        (REASONING_WALL_S, 9, 10.0),
    ]
    .map(|(metric, step, value)| ScalarIn {
        metric,
        group: None,
        step: Some(step),
        value,
    });
    let body = serde_json::json!({"tokenizer": "qwen2", "reasoning_tokens": 1000.0});
    let s = summarize(sc, [(REASONING_KIND, &body)]);
    let k = &s.tok_s;
    assert_eq!(
        (
            k.reasoning_tok_per_s,
            k.reasoning_tokens,
            k.reasoning_wall_s
        ),
        (Some(100.0), Some(1_000.0), Some(10.0))
    );
    assert_eq!(k.reasoning_tokenizer.as_deref(), Some("qwen2"));
    assert!(!s.missing.contains_key(TOK_S_REASONING));
    // Kept apart from the labeled pair, which still reports no values.
    assert!(k.rust_only.p50.is_none() && s.missing.contains_key(TOK_S_RUST));
    let text = s.render_text();
    let line = text.lines().find(|l| l.starts_with(TOK_S_REASONING_LABEL));
    assert!(
        line.is_some_and(|l| l.contains("100.0000  = 1000.0000 tokens / 10.0000 s")
            && l.ends_with("(tokenizer qwen2)"))
    );
}

#[test]
fn tok_s_is_the_labeled_pair_with_its_timing_basis() {
    let sc = [
        (TOK_S_RUST, 1, 100.0),
        (TOK_S_RUST, 2, 300.0),
        (TOK_S_ALL_LANGUAGES, 1, 1_000.0),
        (TOK_S_ALL_LANGUAGES, 2, 3_000.0),
        (TOK_S_RUST_TIMED_ALONE, 1, 1.0),
        (TOK_S_RUST_TIMED_ALONE, 2, 0.0),
        (TOK_S_LEGACY, 1, 7.0),
    ]
    .map(|(metric, step, value)| ScalarIn {
        metric,
        group: None,
        step: Some(step),
        value,
    });
    let s = summarize(sc, std::iter::empty());
    let k = &s.tok_s;
    assert_eq!(
        (k.rust_only.p50, k.all_languages.p50),
        (Some(200.0), Some(2_000.0))
    );
    assert_eq!((k.rust_timed_alone_steps, k.rust_shared_span_steps), (1, 1));
    assert_eq!(k.legacy_unlabeled.as_ref().map(|d| d.n), Some(1));
    assert!(!s.missing.contains_key(TOK_S_RUST) && !s.missing.contains_key(TOK_S_ALL_LANGUAGES));
    let text = s.render_text();
    let rust = text
        .lines()
        .find(|l| l.starts_with("Rust-only tok/s (headline)"));
    assert!(rust.is_some_and(|l| l.contains("200.0000")
        && l.ends_with("(timed on the Rust output alone in 1 of 2 steps)")));
    assert!(text.contains("all-languages tok/s") && text.contains("unlabeled tok/s"));
}

#[test]
fn elo_is_read_against_methods_in_the_latest_snapshot() {
    use crate::flywheel::{EloFitId, EloSolver, Rating};
    let fit = EloFitId {
        solver: EloSolver::NsFamilyBt,
        contender_set_sha256: "c".into(),
        cohort_sha256: "h".into(),
        family_roster_sha256: None,
        fold_map_sha256: None,
        anchor: Some("H38".into()),
        tie_rule: "abs:1e-4".into(),
    };
    let snap = |round: u32, fly: f64| {
        let mut e = EloSnapshot::new(round, fit.clone(), 10, 4);
        let r = |elo| Rating {
            elo,
            ci95: None,
            wins: None,
            ties: None,
            losses: None,
        };
        e.ratings.insert(
            TaskType::Regression,
            [
                ("H38".to_owned(), r(1500.0)),
                ("tabpfn".to_owned(), r(1610.0)),
                ("ROUND_2".to_owned(), r(fly)),
            ]
            .into(),
        );
        serde_json::to_value(e).unwrap()
    };
    let (old, new) = (snap(1, 1400.0), snap(2, 1550.0));
    let s = summarize(std::iter::empty(), [(ELO_SCHEMA, &new), (ELO_SCHEMA, &old)]);
    let e = s.elo.unwrap();
    assert_eq!(
        (e.round, e.snapshots_seen),
        (2, 2),
        "latest round wins regardless of order"
    );
    let t = &e.tasks[&TaskType::Regression];
    assert_eq!(t.flywheel, Some(("ROUND_2".to_owned(), 1550.0, 2)));
    assert_eq!(t.best_method, Some(("tabpfn".to_owned(), 1610.0)));
    assert_eq!(t.delta_vs_best_method, Some(-60.0));
    assert!(e.caveats.contains(&"no_frozen_family_roster".to_owned()));
}
