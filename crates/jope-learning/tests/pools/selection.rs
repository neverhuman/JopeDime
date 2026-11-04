use std::collections::BTreeSet;

use super::*;

pub(crate) fn freeze_is_deterministic_and_verifiable_body() {
    let rows = synthetic_rows(6, 30, 12);
    let (pools, manifest) = freeze(&rows, &rule(), &h("input")).unwrap();
    assert_eq!(pools.len(), 30);
    assert!(manifest.excluded.is_empty());
    verify(&pools, &manifest).unwrap();
    let mut reversed = rows.clone();
    reversed.reverse();
    let (_, again) = freeze(&reversed, &rule(), &h("input")).unwrap();
    assert_eq!(
        manifest.sha256().unwrap(),
        again.sha256().unwrap(),
        "row order must not change pools"
    );
    assert!(manifest.shadow_only && !manifest.qualified_claim);
}

pub(crate) fn tampered_pool_fails_verification_body() {
    let (mut pools, manifest) = freeze(&synthetic_rows(3, 6, 10), &rule(), &h("input")).unwrap();
    pools[2].members[0].q = Some(99.0);
    pools[2].members[0].status = Status::Complete;
    assert!(verify(&pools, &manifest).is_err());
}

pub(crate) fn held_out_roles_are_refused_body() {
    for role in [
        "val",
        "validation",
        "sealed",
        "test",
        "development",
        "meta_test",
    ] {
        let mut rows = synthetic_rows(2, 2, 10);
        rows[3].role = role.into();
        let err = freeze(&rows, &rule(), &h("input")).unwrap_err();
        assert!(
            format!("{err:#}").contains("may not enter a shadow pool"),
            "{role}: {err:#}"
        );
    }
}

pub(crate) fn unknown_and_conflicting_contexts_are_excluded_not_imputed_body() {
    let mut rows = synthetic_rows(2, 4, 10);
    rows[1].status = Status::Unknown;
    rows[1].q = None;
    // Context 1: the same program measured twice with different q.
    let mut dup = rows[12].clone();
    dup.q = Some(dup.q.unwrap_or(0.0) + 0.25);
    dup.status = Status::Complete;
    rows.push(dup);
    // Context 2: too small after removing members.
    let dropped: BTreeSet<String> = (5..10).map(|k| h(&format!("program-{k}"))).collect();
    rows.retain(|r| !(r.context_id == h("context-2") && dropped.contains(&r.program_sha256)));
    let (pools, manifest) = freeze(&rows, &rule(), &h("input")).unwrap();
    let excluded: BTreeSet<&str> = manifest
        .excluded
        .iter()
        .map(|e| e.context_id.as_str())
        .collect();
    assert!(excluded.contains(h("context-0").as_str()));
    assert!(excluded.contains(h("context-1").as_str()));
    assert!(excluded.contains(h("context-2").as_str()));
    assert_eq!(pools.len() + manifest.excluded.len(), 4);
    assert!(manifest
        .excluded
        .iter()
        .any(|e| e.reason.contains("min_pool")));
    assert!(manifest
        .excluded
        .iter()
        .any(|e| e.reason.contains("unknown outcome")));
    assert!(manifest
        .excluded
        .iter()
        .any(|e| e.reason.contains("conflicting")));
}

pub(crate) fn cap_is_salted_and_selector_independent_body() {
    let rows = synthetic_rows(2, 4, 30);
    let mut capped = rule();
    capped.max_pool = Some(12);
    let (a, _) = freeze(&rows, &capped, &h("input")).unwrap();
    assert!(a.iter().all(|p| p.members.len() == 12));
    capped.salt = "other".into();
    let (b, _) = freeze(&rows, &capped, &h("input")).unwrap();
    assert_ne!(
        a[0].members, b[0].members,
        "a different salt draws a different pool"
    );
}

pub(crate) fn oracle_candidate_is_established_and_constant_candidate_is_not_body() {
    // Budget 2 of 24 members: the prior must actually choose, so the oracle's margin is visible.
    let (pools, manifest) = freeze(&synthetic_rows(24, 120, 24), &rule(), &h("input")).unwrap();
    let cfg = Config { budget: 2, ..cfg() };
    let oracle = external("oracle", Class::Candidate, &pools, |p, i| {
        p.members[i].outcome().unwrap_or(-1.0)
    });
    let constant = external("constant", Class::Candidate, &pools, |_, _| 0.0);
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![&oracle, &constant],
        unavailable: vec![],
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &cfg).unwrap();
    assert!(receipt.shadow_only && !receipt.qualified_claim && !receipt.l6_claim);
    assert_eq!(receipt.contexts.len(), 120);
    let s = &receipt.summaries;
    assert_eq!(s["oracle"].mean_pairwise_accuracy, Some(1.0));
    assert!(s["oracle"].mean_regret.abs() < 1e-12);
    assert_eq!(s["constant"].mean_pairwise_accuracy, Some(0.5));
    assert_eq!(s[UNIFORM].mean_pairwise_accuracy, Some(0.5));
    // The program effect is shared across contexts, so the prior beats uniform here.
    assert!(s["program_prior"].mean_realized_gain > s[UNIFORM].mean_realized_gain);
    assert_eq!(receipt.strongest_baseline, "program_prior");
    let primary = |cand: &str| {
        receipt
            .comparisons
            .iter()
            .find(|c| c.candidate == cand && c.baseline == receipt.strongest_baseline)
            .unwrap()
    };
    assert_eq!(primary("oracle").realized_gain.verdict, Verdict::Greater);
    assert_ne!(primary("constant").realized_gain.verdict, Verdict::Greater);
    assert_eq!(receipt.ranking_learnability, "ESTABLISHED_SHADOW");

    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![&constant],
        unavailable: vec![],
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &cfg).unwrap();
    assert_eq!(receipt.ranking_learnability, "NOT_ESTABLISHED");
}

pub(crate) fn beating_uniform_alone_does_not_establish_body() {
    // A candidate equal to the program prior's own information beats uniform but not the prior.
    let (pools, manifest) = freeze(&synthetic_rows(12, 96, 16), &rule(), &h("input")).unwrap();
    let effect = external("effect_only", Class::Candidate, &pools, |p, i| {
        let program = &p.members[i].program_sha256;
        (0..64)
            .find(|k| &h(&format!("program-{k}")) == program)
            .map(|k| u(&format!("effect-{k}")))
            .unwrap()
    });
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![&effect],
        unavailable: vec![],
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &cfg()).unwrap();
    let vs_uniform = receipt
        .comparisons
        .iter()
        .find(|c| c.baseline == UNIFORM)
        .unwrap();
    assert_eq!(vs_uniform.realized_gain.verdict, Verdict::Greater);
    let vs_strongest = receipt
        .comparisons
        .iter()
        .find(|c| c.baseline == receipt.strongest_baseline)
        .unwrap();
    if vs_strongest.realized_gain.verdict != Verdict::Greater {
        assert_eq!(receipt.ranking_learnability, "NOT_ESTABLISHED");
    }
}
