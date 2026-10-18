use jope_records::research::{
    compile, render_capsule, EvidenceLocator, EvidenceResolver, EvidenceValue, ResearchCapsuleV2,
    RoundInput, Stage, CAPSULE_BYTE_LIMIT,
};
use std::collections::BTreeMap;

#[derive(Default)]
struct Resolver {
    value: Option<EvidenceValue>,
    revoked: bool,
}
impl EvidenceResolver for Resolver {
    type Error = &'static str;
    fn is_revoked(&self, _: &EvidenceLocator) -> Result<bool, Self::Error> {
        Ok(self.revoked)
    }
    fn resolve(&self, _: &EvidenceLocator) -> Result<Option<EvidenceValue>, Self::Error> {
        Ok(self.value.clone())
    }
}
fn locator() -> EvidenceLocator {
    EvidenceLocator {
        uri: "blob://fixture/metric".into(),
        sha256: "a".repeat(64),
        version: "v1".into(),
    }
}
fn value(low: f64, high: f64) -> EvidenceValue {
    EvidenceValue {
        locator: locator(),
        unit: "score".into(),
        numerator: 1,
        denominator: 4,
        estimate: Some(0.0),
        ci_low: Some(low),
        ci_high: Some(high),
        status: "complete".into(),
        estimator: "fixture-estimator".into(),
        protocol: "fixture-protocol".into(),
        disclosed: true,
    }
}
fn input() -> RoundInput {
    RoundInput {
        round_id: "round-fail-1".into(),
        source_commit: None,
        stage: Stage::Shadow,
        outcome: "aborted".into(),
        requested: 5,
        attempted: 3,
        computed: 2,
        applied: 0,
        committed: 0,
        exported: 0,
        reloaded: 0,
        next_actor: 0,
        facts: vec!["two computed".into()],
        inference: vec!["unverified".into()],
        limitations: vec![],
        evidence: vec![locator()],
        disclosure_granted: true,
    }
}

#[test]
fn deterministic_compilation_and_rendering() {
    let r = Resolver {
        value: Some(value(1.0, 2.0)),
        revoked: false,
    };
    let a = compile(&input(), &r).unwrap();
    let b = compile(&input(), &r).unwrap();
    assert_eq!(a, b);
    assert_eq!(render_capsule(&a).unwrap(), render_capsule(&b).unwrap());
}
#[test]
fn revoked_or_missing_evidence_cannot_support_assertion() {
    let mut r = Resolver {
        value: Some(value(1.0, 2.0)),
        revoked: true,
    };
    let c = compile(&input(), &r).unwrap();
    assert!(!c.assertion_supported);
    assert!(c.evidence.is_empty());
    assert!(c.limitations.iter().any(|s| s.contains("revoked")));
    r.revoked = false;
    r.value = None;
    let c = compile(&input(), &r).unwrap();
    assert!(!c.assertion_supported);
    assert!(c.limitations.iter().any(|s| s.contains("missing")));
}
#[test]
fn confidence_interval_containing_zero_is_inconclusive() {
    let r = Resolver {
        value: Some(value(-0.2, 0.3)),
        revoked: false,
    };
    let c = compile(&input(), &r).unwrap();
    assert!(c.limitations.iter().any(|s| s.contains("inconclusive")));
    assert_eq!(c.outcome, "aborted");
}
#[test]
fn derived_text_does_not_upgrade_shadow_stage_or_failure_denominators() {
    let r = Resolver {
        value: Some(value(1.0, 2.0)),
        revoked: false,
    };
    let c = compile(&input(), &r).unwrap();
    assert_eq!(c.stage, Stage::Shadow);
    assert_eq!(c.denominator.get("requested"), Some(&5));
    assert_eq!(c.denominator.get("attempted"), Some(&3));
    assert_eq!(c.denominator.get("applied"), Some(&0));
    assert_eq!(c.outcome, "aborted");
}
#[test]
fn renderer_emits_stable_hash_and_unknown_source_commit() {
    let c = ResearchCapsuleV2 {
        round_id: "r".into(),
        source_commit: None,
        stage: Stage::Adaptive,
        outcome: "inconclusive".into(),
        denominator: BTreeMap::new(),
        facts: vec![],
        inference: vec![],
        limitations: vec![],
        evidence: vec![],
        assertion_supported: false,
    };
    let x = render_capsule(&c).unwrap();
    assert!(String::from_utf8(x.canonical_json)
        .unwrap()
        .contains("\"source_commit\":null"));
    assert_eq!(x.sha256.len(), 64);
}

#[test]
fn disclosure_filter_removes_protected_ids_counts_and_prose() {
    let mut i = input();
    i.disclosure_granted = false;
    let r = Resolver {
        value: Some(value(1.0, 2.0)),
        revoked: false,
    };
    let c = compile(&i, &r).unwrap();
    assert_eq!(c.round_id, "withheld");
    assert!(c.denominator.is_empty());
    assert!(c.facts.is_empty() && c.inference.is_empty() && c.evidence.is_empty());
    assert!(!c.assertion_supported);
}

#[test]
fn shuffled_evidence_and_prose_have_identical_bytes() {
    let r = Resolver {
        value: Some(value(1.0, 2.0)),
        revoked: false,
    };
    let mut a = input();
    a.facts = vec!["z".into(), "a".into()];
    a.inference = vec!["second".into(), "first".into()];
    a.evidence.push(locator());
    let mut b = a.clone();
    b.facts.reverse();
    b.inference.reverse();
    b.evidence.reverse();
    assert_eq!(
        render_capsule(&compile(&a, &r).unwrap()).unwrap(),
        render_capsule(&compile(&b, &r).unwrap()).unwrap()
    );
}

#[test]
fn mismatched_evidence_locator_is_limitation_only() {
    let mut wrong = value(1.0, 2.0);
    wrong.locator.sha256 = "b".repeat(64);
    let c = compile(
        &input(),
        &Resolver {
            value: Some(wrong),
            revoked: false,
        },
    )
    .unwrap();
    assert!(!c.assertion_supported && c.evidence.is_empty());
    assert!(c.limitations.iter().any(|x| x.contains("mismatched")));
}

#[test]
fn malformed_locator_fails_closed_before_lookup() {
    let mut i = input();
    i.evidence[0].sha256 = "not-a-digest".into();
    let c = compile(&i, &Resolver::default()).unwrap();
    assert!(!c.assertion_supported && c.evidence.is_empty());
    assert!(c
        .limitations
        .iter()
        .any(|x| x.contains("canonical SHA-256")));
}

#[test]
fn zero_denominator_evidence_is_excluded() {
    let mut v = value(1.0, 2.0);
    v.denominator = 0;
    let c = compile(
        &input(),
        &Resolver {
            value: Some(v),
            revoked: false,
        },
    )
    .unwrap();
    assert!(!c.assertion_supported && c.evidence.is_empty());
}

#[test]
fn compile_is_read_only_and_keeps_aborted_round_outcome() {
    let i = input();
    let before = i.clone();
    let c = compile(
        &i,
        &Resolver {
            value: Some(value(1.0, 2.0)),
            revoked: false,
        },
    )
    .unwrap();
    assert_eq!(i, before);
    assert_eq!(c.outcome, "aborted");
    assert_eq!(c.denominator.get("committed"), Some(&0));
}

#[test]
fn renderer_enforces_size_limit_without_truncating_limitations() {
    let mut c = compile(&input(), &Resolver::default()).unwrap();
    c.limitations.push("x".repeat(CAPSULE_BYTE_LIMIT));
    assert!(render_capsule(&c).is_err());
    assert_eq!(c.limitations.last().unwrap().len(), CAPSULE_BYTE_LIMIT);
}
