//! The seal and champion gates are off by default and refuse to seal or promote, whatever the
//! flag says. The evaluators behind them are exercised on synthetic inputs and on the
//! committed G-JUDGE roster proposal (read, never modified).

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;

use jope_learning::canon::sha256_hex;
use jope_learning::qualify::gate::{Gate, FLAG};
use jope_learning::qualify::promotion::{self, Verdict};
use jope_learning::qualify::roster;
use jope_learning::qualify::{request_promotion, request_seal, Action, Refusal};
use serde_json::{json, Value};

const PROPOSAL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../deploy/dl-flywheel/pipelines/rlfw/judge/SEALED_ROSTER_PROPOSAL.json"
);

/// A named edit of a valid fixture document.
type Mutation = Box<dyn Fn(&mut Value)>;

fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn committed_proposal() -> Vec<u8> {
    std::fs::read(PROPOSAL).unwrap()
}

/// A sound proposal: two training, one development, two confirmation and three sealed families.
fn proposal(alpha_budget: Value) -> Value {
    let record = |n: u32| {
        json!({
            "context_ids": [],
            "dataset_hashes": [format!("d{n}")],
            "member_roles": ["sealed_final"],
            "sources": ["sealed_catalog"],
            "view_ids": [format!("d{n}__r0__f0"), format!("d{n}__r0__f1")],
        })
    };
    json!({
        "adopted": false,
        "alpha_budget": alpha_budget,
        "encoder_pretraining_closure": "unmeasured",
        "partition": {
            "confirmation": ["c1", "c2"],
            "development": ["v1"],
            "meta_training": ["t1", "t2"],
            "sealed_final": ["s1", "s2", "s3"],
        },
        "proof": {"adopted": false, "failures": [], "families": 3, "passed": true,
                  "seal_created": false, "seal_spent": false},
        "qualifying": false,
        "roles": ["meta_training", "development", "confirmation", "sealed_final"],
        "schema": "jope.rlfw-sealed-roster-proposal/v1",
        "seal_created": false,
        "seal_spent": false,
        "sealed_family_records": {"s1": record(1), "s2": record(2), "s3": record(3)},
        "sources": {"closure": {"sealed_context_ids_in_closure": 0, "sealed_views_in_closure": 0}},
    })
}

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

const FAMILIES: usize = 12;

fn manifest() -> Value {
    let families: Vec<String> = (1..=FAMILIES).map(|n| format!("fam-{n:02}")).collect();
    json!({
        "schema": "jope.qualify-arena-manifest/v1",
        "arena_id": "fixture-arena",
        "metric": {"name": "outer_rmse_ratio", "lower_is_better": true},
        "incumbent": {"id": "frozen-v1", "sha256": sha(1)},
        "challenger": {"id": "candidate", "sha256": sha(2)},
        "roster_role": "confirmation",
        "families": families,
        "family_weighting": "equal",
        "min_families": 8,
        "practical_margin": 0.005,
        "alpha_total": 0.05,
        "looks": [{"index": 1, "alpha": 0.01}, {"index": 2, "alpha": 0.04}],
        "contenders": 1,
        "bootstrap": {"resamples": 2000, "seed": 20_261_013},
        "max_failure_rate": 0.05,
        "safety": {"panel": "l6_full_val263", "reference": "frozen-v1", "max_regression": 0.0},
    })
}

/// Two contexts per family; `effect(family, context)` is the incumbent minus the challenger.
fn rows(effect: impl Fn(usize, usize) -> f64) -> Vec<Value> {
    let mut rows = Vec::new();
    for family in 1..=FAMILIES {
        for context in 0..2 {
            rows.push(json!({
                "family_id": format!("fam-{family:02}"),
                "context_id": format!("ctx-{family:02}-{context}"),
                "selection": "selected",
                "challenger": 1.0 - effect(family, context),
                "incumbent": 1.0,
                "attempts": 10,
                "failures": 0,
            }));
        }
    }
    rows
}

fn results(manifest: &[u8], rows: Vec<Value>) -> Value {
    json!({
        "schema": "jope.qualify-arena-results/v1",
        "manifest_sha256": sha256_hex(manifest),
        "look": 1,
        "prior_looks": [],
        "rows": rows,
        "holds": [],
        "cost": {"cpu_seconds": 3600.0, "wall_seconds": 600.0},
        "safety": {"challenger": 0.8970, "reference": 0.8973},
    })
}

/// A clear lift: every family gains about 0.05.
#[allow(clippy::cast_precision_loss)]
fn clear_lift(family: usize, context: usize) -> f64 {
    0.05 + 0.001 * ((family + context) % 3) as f64
}

fn eligible() -> (Vec<u8>, Vec<u8>) {
    let manifest = bytes(&manifest());
    let results = bytes(&results(&manifest, rows(clear_lift)));
    (manifest, results)
}

// ---- the flag ----------------------------------------------------------------------------------

#[test]
fn the_flag_is_jope_seal_gate_enabled_and_defaults_off() {
    assert_eq!(FLAG, "JOPE_SEAL_GATE_ENABLED");
    assert_eq!(Gate::default(), Gate::Off);
    assert_eq!(Gate::from_value(None), Gate::Off);
    assert!(!Gate::from_value(None).is_on());
}

#[test]
fn this_test_environment_keeps_the_gate_off() {
    // A tripwire: the gate must never be switched on where these tests run.
    assert!(
        !Gate::from_env().is_on(),
        "{FLAG} is on in the test environment"
    );
}

#[test]
fn only_the_exact_value_1_turns_the_gate_on() {
    assert_eq!(Gate::from_value(Some(OsStr::new("1"))), Gate::On);
    for off in ["", "0"] {
        assert_eq!(
            Gate::from_value(Some(OsStr::new(off))),
            Gate::Off,
            "{off:?}"
        );
    }
    for value in [
        "true", "TRUE", "yes", "on", " 1", "1 ", "01", "2", "-1", "1\n",
    ] {
        let gate = Gate::from_value(Some(OsStr::new(value)));
        assert_eq!(gate, Gate::Unrecognized(value.to_owned()), "{value:?}");
        assert!(!gate.is_on(), "{value:?}");
    }
    let not_utf8 = Gate::from_value(Some(OsStr::from_bytes(b"1\xff")));
    assert!(matches!(not_utf8, Gate::Unrecognized(_)) && !not_utf8.is_on());
}

// ---- seal (G23 / J17) --------------------------------------------------------------------------

#[test]
fn seal_is_refused_by_default_before_the_proposal_is_read() {
    for input in [committed_proposal(), b"not json".to_vec(), Vec::new()] {
        let refusal = request_seal(&Gate::from_env(), &input).unwrap_err();
        assert_eq!(
            refusal,
            Refusal::GateOff {
                action: Action::Seal,
                gate: Gate::Off
            }
        );
    }
    let refusal = request_seal(&Gate::Unrecognized("true".to_owned()), b"not json").unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::GateOff {
            action: Action::Seal,
            ..
        }
    ));
    assert!(refusal.to_string().contains("JOPE_SEAL_GATE_ENABLED"));
}

#[test]
fn the_committed_proposal_is_sound_but_its_alpha_budget_is_undecided() {
    let raw = committed_proposal();
    let report = roster::check(&raw).unwrap();
    // Shown with --nocapture: the report a reviewer can bind to the proposal's SHA-256.
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    assert_eq!(report.proposal_sha256, sha256_hex(&raw));
    assert!(report.defects.is_empty(), "{:?}", report.defects);
    assert_eq!(report.owner_decisions.len(), 1);
    assert!(report.owner_decisions[0].contains("alpha budget (U13)"));
    assert!(report.owner_decisions[0].contains("unspent_jepson_decision"));
    assert!(report
        .limitations
        .iter()
        .any(|item| item.contains("pretraining closure")));
    assert!(!report.adoptable());
    assert!(!report.adopted && !report.seal_created);
    assert!(report.families["sealed_final"] > 0);

    // Even with the gate on, the undecided budget keeps it from being sealed.
    let refusal = request_seal(&Gate::On, &raw).unwrap_err();
    match refusal {
        Refusal::NotReady { action, reasons } => {
            assert_eq!(action, Action::Seal);
            assert!(reasons.iter().any(|item| item.contains("alpha budget")));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_adoptable_proposal_is_still_refused_for_the_pending_g23_ruling() {
    let raw = bytes(&proposal(json!(0.05)));
    let report = roster::check(&raw).unwrap();
    assert!(report.adoptable(), "{:?}", report.reasons());
    assert_eq!(report.families["sealed_final"], 3);
    let refusal = request_seal(&Gate::On, &raw).unwrap_err();
    assert_eq!(
        refusal,
        Refusal::RulingPending {
            action: Action::Seal
        }
    );
    assert!(refusal.to_string().contains("G23"));
    assert!(refusal.to_string().contains("this build has no seal path"));
}

#[test]
fn roster_defects_are_named() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "in both",
            Box::new(|p| p["partition"]["confirmation"] = json!(["c1", "s1"])),
        ),
        (
            "listed twice",
            Box::new(|p| p["partition"]["meta_training"] = json!(["t1", "t1"])),
        ),
        (
            "unknown role",
            Box::new(|p| p["partition"]["holdout"] = json!(["h1"])),
        ),
        (
            "no development list",
            Box::new(|p| {
                p["partition"]
                    .as_object_mut()
                    .unwrap()
                    .remove("development");
            }),
        ),
        (
            "roles",
            Box::new(|p| p["roles"] = json!(["meta_training", "sealed_final"])),
        ),
        (
            "members outside",
            Box::new(|p| {
                p["sealed_family_records"]["s2"]["member_roles"] = json!(["sealed_final", "train"]);
            }),
        ),
        (
            "dataset_hashes shared",
            Box::new(|p| {
                p["sealed_family_records"]["s3"]["dataset_hashes"] = json!(["d1"]);
            }),
        ),
        (
            "no context, dataset or view id",
            Box::new(|p| {
                p["sealed_family_records"]["s1"] = json!({"context_ids": [], "dataset_hashes": [], "member_roles": ["sealed_final"], "view_ids": []});
            }),
        ),
        (
            "without a record",
            Box::new(|p| {
                p["sealed_family_records"]
                    .as_object_mut()
                    .unwrap()
                    .remove("s3");
            }),
        ),
        (
            "outside sealed_final",
            Box::new(|p| {
                p["sealed_family_records"]["c1"] = p["sealed_family_records"]["s1"].clone();
            }),
        ),
        (
            "proof did not pass",
            Box::new(|p| p["proof"]["passed"] = json!(false)),
        ),
        (
            "proof covers",
            Box::new(|p| p["proof"]["families"] = json!(2)),
        ),
        (
            "sealed_views_in_closure is 4",
            Box::new(|p| {
                p["sources"]["closure"]["sealed_views_in_closure"] = json!(4);
            }),
        ),
        (
            "sources.closure is absent",
            Box::new(|p| p["sources"] = json!({})),
        ),
        (
            "cannot claim adopted",
            Box::new(|p| p["adopted"] = json!(true)),
        ),
        (
            "cannot claim seal_spent",
            Box::new(|p| p["seal_spent"] = json!(true)),
        ),
        (
            "sealed_final is empty",
            Box::new(|p| {
                p["partition"]["sealed_final"] = json!([]);
                p["sealed_family_records"] = json!({});
                p["proof"]["families"] = json!(0);
            }),
        ),
    ];
    for (expected, mutate) in cases {
        let mut document = proposal(json!(0.05));
        mutate(&mut document);
        let report = roster::check(&bytes(&document)).unwrap();
        assert!(
            report.defects.iter().any(|item| item.contains(expected)),
            "{expected}: {:?}",
            report.defects
        );
        assert!(!report.adoptable(), "{expected}");
        let refusal = request_seal(&Gate::On, &bytes(&document)).unwrap_err();
        assert!(
            matches!(refusal, Refusal::NotReady { .. }),
            "{expected}: {refusal:?}"
        );
    }
}

#[test]
fn only_a_decided_alpha_level_settles_the_budget() {
    for budget in [
        json!("unspent_jepson_decision"),
        json!(null),
        json!(0),
        json!(1),
        json!(1.5),
    ] {
        let report = roster::check(&bytes(&proposal(budget.clone()))).unwrap();
        assert!(report.defects.is_empty(), "{budget}");
        assert_eq!(report.owner_decisions.len(), 1, "{budget}");
    }
}

#[test]
fn a_document_of_another_schema_is_invalid() {
    let mut document = proposal(json!(0.05));
    document["schema"] = json!("jope.rlfw-judge-seal/v1");
    assert!(roster::check(&bytes(&document))
        .unwrap_err()
        .contains("schema"));
    let refusal = request_seal(&Gate::On, &bytes(&document)).unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::Invalid {
            action: Action::Seal,
            ..
        }
    ));
}

// ---- promotion (G36 / J18) ---------------------------------------------------------------------

#[test]
fn promotion_is_refused_by_default_before_the_evidence_is_read() {
    let (manifest, results) = eligible();
    for (m, r) in [(manifest, results), (b"{".to_vec(), Vec::new())] {
        let refusal = request_promotion(&Gate::from_env(), &m, &r).unwrap_err();
        assert_eq!(
            refusal,
            Refusal::GateOff {
                action: Action::Promote,
                gate: Gate::Off
            }
        );
    }
    let refusal = request_promotion(&Gate::Unrecognized("yes".to_owned()), b"{", b"{").unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::GateOff {
            action: Action::Promote,
            ..
        }
    ));
}

#[test]
fn an_eligible_challenger_is_still_refused_for_the_pending_g36_ruling() {
    let (manifest, results) = eligible();
    let report = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::PromoteEligible,
        "{:?}",
        report.reasons
    );
    assert!(report.lower.unwrap() > report.practical_margin);
    assert_eq!(report.families_complete, FAMILIES);
    assert_eq!((report.wins, report.ties, report.losses), (FAMILIES, 0, 0));
    assert!(!report.promoted);
    assert_eq!(report.generalization_claim, promotion::GENERALIZATION_CLAIM);
    assert_eq!(report.manifest_sha256, sha256_hex(&manifest));
    assert_eq!(report.results_sha256, sha256_hex(&results));

    let refusal = request_promotion(&Gate::On, &manifest, &results).unwrap_err();
    assert_eq!(
        refusal,
        Refusal::RulingPending {
            action: Action::Promote
        }
    );
    assert!(refusal.to_string().contains("G36"));
    assert!(refusal
        .to_string()
        .contains("this build has no promote path"));
}

#[test]
fn an_inconclusive_arena_is_a_verdict_not_an_error() {
    let manifest = bytes(&manifest());
    let noisy = rows(|family, _| if family % 2 == 0 { 0.03 } else { -0.02 });
    let results = bytes(&results(&manifest, noisy));
    let report = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Inconclusive,
        "{:?}",
        report.reasons
    );
    assert!(report.lower.unwrap() <= report.practical_margin);
    assert!(report.upper.unwrap() >= report.practical_margin);
    let refusal = request_promotion(&Gate::On, &manifest, &results).unwrap_err();
    match refusal {
        Refusal::NotReady { reasons, .. } => assert_eq!(reasons[0], "verdict inconclusive"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_lift_below_the_practical_margin_is_no_practical_lift() {
    let manifest = bytes(&manifest());
    let tiny = rows(|family, _| if family % 2 == 0 { 0.0005 } else { 0.0 });
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, tiny))).unwrap();
    assert!(
        report.lower.unwrap() > 0.0,
        "a positive lower bound alone is not enough"
    );
    assert_eq!(
        report.verdict,
        Verdict::NoPracticalLift,
        "{:?}",
        report.reasons
    );
}

#[test]
fn the_effect_direction_follows_the_metric() {
    let mut higher = manifest();
    higher["metric"]["lower_is_better"] = json!(false);
    let higher = bytes(&higher);
    let report = promotion::decide(&higher, &bytes(&results(&higher, rows(clear_lift)))).unwrap();
    // The challenger's metric is lower; on a higher-is-better metric that is a loss.
    assert_eq!((report.wins, report.losses), (0, FAMILIES));
    assert_eq!(report.verdict, Verdict::NoPracticalLift);
}

#[test]
fn an_l6_regression_against_the_frozen_reference_is_refused() {
    let manifest = bytes(&manifest());
    let mut document = results(&manifest, rows(clear_lift));
    // L6 is a loss: above frozen v1 0.8973 is worse.
    document["safety"] = json!({"challenger": 0.8990, "reference": 0.8973});
    let report = promotion::decide(&manifest, &bytes(&document)).unwrap();
    assert_eq!(report.verdict, Verdict::Regression);
    assert!(report.safety_regression > 0.0);
    assert!(report
        .reasons
        .iter()
        .any(|item| item.contains("lower is better")));
    // Below the reference is not a regression.
    document["safety"] = json!({"challenger": 0.8960, "reference": 0.8973});
    let report = promotion::decide(&manifest, &bytes(&document)).unwrap();
    assert_eq!(report.verdict, Verdict::PromoteEligible);
}

#[test]
fn holds_failures_and_missing_attempt_accounting_hold_the_gate() {
    let manifest = bytes(&manifest());
    let mut held = results(&manifest, rows(clear_lift));
    held["holds"] = json!(["custody HOLD 2026-10-13"]);
    let report = promotion::decide(&manifest, &bytes(&held)).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);

    let mut failing = rows(clear_lift);
    for row in &mut failing {
        row["failures"] = json!(1);
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, failing))).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);
    assert!((report.failure_rate.unwrap() - 0.1).abs() < 1e-12);

    let mut unaccounted = rows(clear_lift);
    for row in &mut unaccounted {
        row["attempts"] = json!(0);
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, unaccounted))).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);
    assert!(report.failure_rate.is_none());
}

#[test]
fn unknown_outcomes_drop_their_family_and_are_never_imputed() {
    let manifest = bytes(&manifest());
    let mut partial = rows(clear_lift);
    // Five families lose one context each: seven complete families remain, fewer than eight.
    for row in partial.iter_mut().step_by(2).take(5) {
        row["challenger"] = Value::Null;
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, partial))).unwrap();
    assert_eq!(report.contexts_unknown, 5);
    assert_eq!(report.families_complete, FAMILIES - 5);
    assert_eq!(report.families_incomplete.len(), 5);
    assert_eq!(report.verdict, Verdict::Inconclusive);
    assert!(report.lower.is_none() && report.upper.is_none());

    // A declared family with no rows at all is incomplete too.
    let some: Vec<Value> = rows(clear_lift).into_iter().skip(2).collect();
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, some))).unwrap();
    assert_eq!(report.families_incomplete, vec!["fam-01".to_owned()]);
    assert_eq!(report.verdict, Verdict::Inconclusive);
}

#[test]
fn incomplete_families_cap_the_verdict_unless_predeclared() {
    // Eleven families show a clear lift; the twelfth has one unknown context. Dropping it must
    // not let the remaining families promote: selection on missingness.
    let mut one_unknown = rows(clear_lift);
    one_unknown[0]["incumbent"] = Value::Null;
    let strict = bytes(&manifest());
    let report =
        promotion::decide(&strict, &bytes(&results(&strict, one_unknown.clone()))).unwrap();
    assert_eq!(report.max_incomplete_families, 0);
    assert_eq!(report.families_incomplete, vec!["fam-01".to_owned()]);
    assert!(
        report.lower.unwrap() > report.practical_margin,
        "the bounds alone would promote"
    );
    assert_eq!(report.verdict, Verdict::Inconclusive);
    assert!(report
        .reasons
        .iter()
        .any(|item| item.contains("selection on missingness")));
    let refusal = request_promotion(
        &Gate::On,
        &strict,
        &bytes(&results(&strict, one_unknown.clone())),
    );
    assert!(matches!(refusal, Err(Refusal::NotReady { .. })));

    // A lift below the margin is not decided either while a family is missing.
    let mut tiny = rows(|family, _| if family % 2 == 0 { 0.0005 } else { 0.0 });
    tiny[0]["challenger"] = Value::Null;
    let report = promotion::decide(&strict, &bytes(&results(&strict, tiny))).unwrap();
    assert_eq!(report.verdict, Verdict::Inconclusive);

    // A manifest that predeclares one tolerated incomplete family decides the look.
    let mut tolerant = manifest();
    tolerant["max_incomplete_families"] = json!(1);
    let tolerant = bytes(&tolerant);
    let report = promotion::decide(&tolerant, &bytes(&results(&tolerant, one_unknown))).unwrap();
    assert_eq!(report.max_incomplete_families, 1);
    assert_eq!(report.verdict, Verdict::PromoteEligible);
}

#[test]
fn invalid_evidence_is_refused_not_judged() {
    let manifest_value = manifest();
    let manifest = bytes(&manifest_value);
    let base = results(&manifest, rows(clear_lift));
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "outer-oracle",
            Box::new(|r| r["rows"][3]["selection"] = json!("oracle")),
        ),
        (
            "exact manifest bytes",
            Box::new(|r| r["manifest_sha256"] = json!(sha(9))),
        ),
        (
            "not declared",
            Box::new(|r| r["rows"][0]["family_id"] = json!("fam-99")),
        ),
        (
            "repeated",
            Box::new(|r| r["rows"][1]["context_id"] = json!("ctx-01-0")),
        ),
        (
            "more failures than attempts",
            Box::new(|r| r["rows"][0]["failures"] = json!(11)),
        ),
        ("look 3 is not declared", Box::new(|r| r["look"] = json!(3))),
        ("prior looks", Box::new(|r| r["look"] = json!(2))),
        (
            "stopped at look 1",
            Box::new(|r| {
                r["look"] = json!(2);
                r["prior_looks"] = json!([{"index": 1, "verdict": "no_practical_lift"}]);
            }),
        ),
        (
            "results schema",
            Box::new(|r| r["schema"] = json!("jope.other/v1")),
        ),
        ("unknown field", Box::new(|r| r["oracle_rows"] = json!([]))),
    ];
    for (expected, mutate) in cases {
        let mut document = base.clone();
        mutate(&mut document);
        let error = promotion::decide(&manifest, &bytes(&document)).unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
        let refusal = request_promotion(&Gate::On, &manifest, &bytes(&document)).unwrap_err();
        assert!(matches!(
            refusal,
            Refusal::Invalid {
                action: Action::Promote,
                ..
            }
        ));
    }
    // A second look after an inconclusive first look is in schedule.
    let mut second = base.clone();
    second["look"] = json!(2);
    second["prior_looks"] = json!([{"index": 1, "verdict": "inconclusive"}]);
    let report = promotion::decide(&manifest, &bytes(&second)).unwrap();
    assert!((report.alpha_spent - 0.05).abs() < 1e-12);
    assert!((report.alpha_look - 0.04).abs() < 1e-12);
}

#[test]
fn an_arena_must_be_a_usable_predeclared_confirmation_arena() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "challenger is the incumbent",
            Box::new(|m| m["challenger"]["sha256"] = json!(sha(1))),
        ),
        (
            "adopted sealed roster",
            Box::new(|m| m["roster_role"] = json!("sealed_final")),
        ),
        (
            "not independent confirmation",
            Box::new(|m| m["roster_role"] = json!("development")),
        ),
        (
            "not independent confirmation",
            Box::new(|m| m["roster_role"] = json!("meta_training")),
        ),
        (
            "more than alpha_total",
            Box::new(|m| m["looks"][1]["alpha"] = json!(0.045)),
        ),
        ("numbered", Box::new(|m| m["looks"][1]["index"] = json!(3))),
        ("min_families", Box::new(|m| m["min_families"] = json!(13))),
        ("min_families", Box::new(|m| m["min_families"] = json!(1))),
        (
            "family_weighting",
            Box::new(|m| m["family_weighting"] = json!("context")),
        ),
        (
            "missing field `family_weighting`",
            Box::new(|m| {
                m.as_object_mut().unwrap().remove("family_weighting");
            }),
        ),
        (
            "leaves fewer than min_families",
            Box::new(|m| m["max_incomplete_families"] = json!(5)),
        ),
        ("unique", Box::new(|m| m["families"][1] = json!("fam-01"))),
        (
            "resamples",
            Box::new(|m| m["bootstrap"]["resamples"] = json!(999)),
        ),
        ("finer than", Box::new(|m| m["contenders"] = json!(50))),
        (
            "practical_margin is negative",
            Box::new(|m| m["practical_margin"] = json!(-0.1)),
        ),
        (
            "lowercase hex sha256",
            Box::new(|m| m["incumbent"]["sha256"] = json!("ea078456")),
        ),
        (
            "max_regression is negative",
            Box::new(|m| m["safety"]["max_regression"] = json!(-1)),
        ),
        (
            "manifest schema",
            Box::new(|m| m["schema"] = json!("jope.other/v1")),
        ),
    ];
    for (expected, mutate) in cases {
        let mut document = manifest();
        mutate(&mut document);
        let manifest = bytes(&document);
        let results = bytes(&results(&manifest, rows(clear_lift)));
        let error = promotion::decide(&manifest, &results).unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
    }
}

#[test]
fn contenders_split_the_alpha_of_a_look() {
    let mut shared = manifest();
    shared["contenders"] = json!(4);
    let shared = bytes(&shared);
    let report = promotion::decide(&shared, &bytes(&results(&shared, rows(clear_lift)))).unwrap();
    assert!((report.alpha_per_comparison - 0.0025).abs() < 1e-15);
}

#[test]
fn the_bootstrap_replays_from_its_seed() {
    let manifest = bytes(&manifest());
    let noisy = rows(|family, context| {
        if (family + context) % 3 == 0 {
            0.04
        } else {
            -0.01
        }
    });
    let results = bytes(&results(&manifest, noisy));
    let first = promotion::decide(&manifest, &results).unwrap();
    let second = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
}
