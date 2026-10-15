//! Synthetic fixtures for freeze → evaluate. No real data, no network, no GPU.

use jope_learning::canon::sha256_hex;
use jope_learning::pools::bootstrap::Verdict;
use jope_learning::pools::evaluate::{evaluate, fold_of, Config, Inputs, Unavailable, UNIFORM};
use jope_learning::pools::pool::{
    export_views, freeze, verify, Exclusion, FreezeRule, Manifest, MeasurementRow, Pool, PoolView,
    Reference, Status,
};
use jope_learning::pools::selectors::{
    Class, External, ExternalDocument, ExternalScore, Features, Selector, EXTERNAL_SCHEMA,
};
use std::cell::RefCell;
use std::collections::BTreeSet;

fn h(label: &str) -> String {
    sha256_hex(label.as_bytes())
}

/// Deterministic pseudo-random value in [0, 1) from a label.
fn u(label: &str) -> f64 {
    let d = h(label);
    u64::from_str_radix(&d[..12], 16).unwrap() as f64 / (1u64 << 48) as f64
}

const PROTOCOL: &str = "protocol";

/// `contexts` contexts in `parents` parents, `programs` programs each. A program's true quality is
/// a program effect plus a context-specific effect, so a context-blind prior is informative but
/// not perfect. Every fifth member failed.
fn synthetic_rows(parents: usize, contexts: usize, programs: usize) -> Vec<MeasurementRow> {
    let mut rows = Vec::new();
    for c in 0..contexts {
        let parent = format!("parent-{}", c % parents);
        let context = h(&format!("context-{c}"));
        for p in 0..programs {
            let program = h(&format!("program-{p}"));
            let failed = (c + p) % 5 == 0;
            let q = 0.6 * u(&format!("effect-{p}")) + 0.4 * u(&format!("noise-{c}-{p}"));
            rows.push(MeasurementRow {
                context_id: context.clone(),
                parent_id: parent.clone(),
                role: "train".into(),
                program_sha256: program,
                status: if failed {
                    Status::ProgramFailed
                } else {
                    Status::Complete
                },
                q: (!failed).then_some(q),
                incumbent_q: Some(0.55),
                protocol_sha256: h(PROTOCOL),
                tokens: Some(vec![1, 3 + (p % 7) as u32, 10 + (p % 3) as u32, 2]),
            });
        }
    }
    rows
}

fn rule() -> FreezeRule {
    FreezeRule {
        min_pool: 9,
        max_pool: None,
        salt: "test".into(),
        min_distinct_measured: 2,
        source_grade: "synthetic_fixture".into(),
    }
}

fn cfg() -> Config {
    Config {
        reference: Reference::Incumbent,
        budget: 8,
        folds: 4,
        fold_salt: "folds".into(),
        prior_weight: 2.0,
        knn_k: 4,
        ridge_lambda: 1.0,
        bootstrap_draws: 400,
        bootstrap_seed: 7,
        alpha: 0.05,
    }
}

fn external(
    name: &str,
    class: Class,
    pools: &[Pool],
    score: impl Fn(&Pool, usize) -> f64,
) -> External {
    let scores = pools
        .iter()
        .flat_map(|p| {
            (0..p.members.len()).map(|i| ExternalScore {
                context_id: p.context_id.clone(),
                program_sha256: p.members[i].program_sha256.clone(),
                score: score(p, i),
            })
        })
        .collect();
    External::from_document(&ExternalDocument {
        schema: EXTERNAL_SCHEMA.into(),
        name: name.into(),
        class,
        rule: format!("{name}/test"),
        provenance: serde_json::json!({"fixture": true}),
        scores,
    })
    .unwrap()
}

#[test]
fn freeze_is_deterministic_and_verifiable() {
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

#[test]
fn tampered_pool_fails_verification() {
    let (mut pools, manifest) = freeze(&synthetic_rows(3, 6, 10), &rule(), &h("input")).unwrap();
    pools[2].members[0].q = Some(99.0);
    pools[2].members[0].status = Status::Complete;
    assert!(verify(&pools, &manifest).is_err());
}

#[test]
fn held_out_roles_are_refused() {
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

#[test]
fn unknown_and_conflicting_contexts_are_excluded_not_imputed() {
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

#[test]
fn cap_is_salted_and_selector_independent() {
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

#[test]
fn oracle_candidate_is_established_and_constant_candidate_is_not() {
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

#[test]
fn beating_uniform_alone_does_not_establish() {
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

/// Records which contexts a fitted selector's training set contained.
struct Spy {
    seen: RefCell<Vec<(String, String)>>,
}

impl Selector for Spy {
    fn name(&self) -> &str {
        "spy"
    }
    fn class(&self) -> Class {
        Class::Baseline
    }
    fn rule(&self) -> String {
        "spy".into()
    }
    fn scores(&self, view: &PoolView<'_>) -> anyhow::Result<Vec<f64>> {
        self.seen
            .borrow_mut()
            .push((view.context_id.to_owned(), view.parent_id.to_owned()));
        Ok(vec![0.0; view.programs.len()])
    }
}

#[test]
fn folds_keep_whole_parents_together_and_every_selector_scores_every_context() {
    let (pools, manifest) = freeze(&synthetic_rows(10, 60, 12), &rule(), &h("input")).unwrap();
    let cfg = cfg();
    for p in &pools {
        let fold = fold_of(&cfg.fold_salt, &p.parent_id, cfg.folds);
        for q in pools.iter().filter(|q| q.parent_id == p.parent_id) {
            assert_eq!(fold, fold_of(&cfg.fold_salt, &q.parent_id, cfg.folds));
        }
    }
    let spy = Spy {
        seen: RefCell::new(Vec::new()),
    };
    let cf = Features {
        dim: 2,
        rows: pools
            .iter()
            .map(|p| {
                (
                    p.context_id.clone(),
                    vec![u(&p.context_id), u(&p.parent_id)],
                )
            })
            .collect(),
    };
    let pf = Features {
        dim: 1,
        rows: (0..12)
            .map(|k| (h(&format!("program-{k}")), vec![u(&format!("effect-{k}"))]))
            .collect(),
    };
    let inputs = Inputs {
        context_features: Some(&cf),
        program_features: Some(&pf),
        external: vec![&spy],
        unavailable: vec![],
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &cfg).unwrap();
    assert_eq!(
        spy.seen.borrow().len(),
        60,
        "external selectors score every pool once"
    );
    let names: BTreeSet<&str> = receipt.selectors.iter().map(|s| s.name.as_str()).collect();
    for n in [
        UNIFORM,
        "program_prior",
        "retrieval_knn",
        "ridge_program",
        "contextual_surrogate_qhat",
        "spy",
    ] {
        assert!(names.contains(n), "{n} missing");
    }
    for c in &receipt.contexts {
        assert_eq!(c.metrics.len(), names.len());
    }
    // The ridge on the true program effect is a strong baseline on this synthetic.
    assert!(
        receipt.summaries["ridge_program"]
            .mean_pairwise_accuracy
            .unwrap()
            > 0.6
    );
    // Same inputs, same receipt.
    let again = evaluate(&pools, &manifest, &inputs, &cfg).unwrap();
    assert_eq!(receipt.sha256().unwrap(), again.sha256().unwrap());
}

#[test]
fn contextual_surrogate_qhat_establishes_h2_on_interaction_pools() {
    // 16 parents, 32 contexts, 10 programs per context drawn from two context-disjoint cohorts
    // (cohort A vs cohort B) so kNN exact program-ID overlap across cohorts is zero, and equal
    // numbers of z=+1 and z=-1 contexts so marginal program_prior and context-blind ridge_program
    // tie around 0.5. Only the bilinear context x program surrogate learns sign(z) * phi(p).
    let mut rows = Vec::new();
    let mut cf_rows = std::collections::BTreeMap::new();
    let mut pf_rows = std::collections::BTreeMap::new();
    for p in 0..20 {
        let prog_sha = h(&format!("inter-prog-{p}"));
        let phi = if p % 2 == 0 { 1.0 } else { -1.0 };
        pf_rows.insert(prog_sha, vec![phi]);
    }
    for c in 0..32 {
        let parent = format!("inter-parent-{}", c / 2);
        let ctx_sha = h(&format!("inter-ctx-{c}"));
        let z = if (c / 2) % 2 == 0 { 1.0 } else { -1.0 };
        cf_rows.insert(ctx_sha.clone(), vec![z]);
        // Alternate cohort (0..10 vs 10..20) by context parity within parent so nearest-z
        // neighbours in other parents include disjoint program IDs.
        let cohort_start = if c % 2 == 0 { 0 } else { 10 };
        for p in cohort_start..(cohort_start + 10) {
            let prog_sha = h(&format!("inter-prog-{p}"));
            let phi = if p % 2 == 0 { 1.0 } else { -1.0 };
            let q = 0.5 + 0.35 * (z * phi) + 0.02 * u(&format!("inter-noise-{c}-{p}"));
            rows.push(MeasurementRow {
                context_id: ctx_sha.clone(),
                parent_id: parent.clone(),
                role: "train".into(),
                program_sha256: prog_sha,
                status: Status::Complete,
                q: Some(q),
                incumbent_q: Some(0.5),
                protocol_sha256: h(PROTOCOL),
                tokens: Some(vec![1, if p % 2 == 0 { 3 } else { 4 }, 2]),
            });
        }
    }
    let (pools, manifest) = freeze(&rows, &rule(), &h("inter-input")).unwrap();
    let cf = Features {
        dim: 1,
        rows: cf_rows,
    };
    let pf = Features {
        dim: 1,
        rows: pf_rows,
    };
    let inputs = Inputs {
        context_features: Some(&cf),
        program_features: Some(&pf),
        external: vec![],
        unavailable: vec![],
    };
    let eval_cfg = Config {
        budget: 2,
        knn_k: 1,
        ..cfg()
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &eval_cfg).unwrap();
    assert_eq!(receipt.ranking_learnability, "ESTABLISHED_SHADOW");
    let primary = receipt
        .comparisons
        .iter()
        .find(|c| {
            c.candidate == "contextual_surrogate_qhat" && c.baseline == receipt.strongest_baseline
        })
        .unwrap();
    assert_eq!(primary.realized_gain.verdict, Verdict::Greater);
    assert_eq!(
        primary.pairwise_accuracy.as_ref().unwrap().verdict,
        Verdict::Greater
    );
    assert!(
        receipt.summaries["contextual_surrogate_qhat"]
            .mean_pairwise_accuracy
            .unwrap()
            > 0.70
    );

    // Caller-supplied Unavailable{contextual_surrogate_qhat} skips fitting the internal surrogate.
    let disabled_inputs = Inputs {
        context_features: Some(&cf),
        program_features: Some(&pf),
        external: vec![],
        unavailable: vec![Unavailable {
            name: "contextual_surrogate_qhat".into(),
            reason: "caller_disabled".into(),
        }],
    };
    let disabled_receipt = evaluate(&pools, &manifest, &disabled_inputs, &eval_cfg).unwrap();
    assert!(!disabled_receipt
        .summaries
        .contains_key("contextual_surrogate_qhat"));
    assert_eq!(disabled_receipt.unavailable_selectors.len(), 1);
    assert_eq!(
        disabled_receipt.unavailable_selectors[0].reason,
        "caller_disabled"
    );
}

#[test]
fn missing_external_score_is_an_error() {
    let (pools, manifest) = freeze(&synthetic_rows(4, 8, 10), &rule(), &h("input")).unwrap();
    let partial = external("partial", Class::Candidate, &pools[..7], |_, _| 1.0);
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![&partial],
        unavailable: vec![],
    };
    let err = evaluate(&pools, &manifest, &inputs, &cfg()).err().unwrap();
    assert!(format!("{err:#}").contains("no score"), "{err:#}");
}

#[test]
fn pool_floor_reference_needs_no_incumbent_and_derives_token_features() {
    let mut rows = synthetic_rows(8, 48, 14);
    for r in &mut rows {
        r.incumbent_q = None;
    }
    let (pools, manifest) = freeze(&rows, &rule(), &h("input")).unwrap();
    let oracle = external("oracle", Class::Candidate, &pools, |p, i| {
        p.members[i].outcome().unwrap_or(-1.0)
    });
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![&oracle],
        unavailable: vec![Unavailable {
            name: "native_gp_rank".into(),
            reason: "fixture".into(),
        }],
    };
    let err = evaluate(&pools, &manifest, &inputs, &cfg()).err().unwrap();
    assert!(format!("{err:#}").contains("no incumbent"), "{err:#}");
    let cfg = Config {
        reference: Reference::PoolFloor,
        ..cfg()
    };
    let receipt = evaluate(&pools, &manifest, &inputs, &cfg).unwrap();
    assert!(
        receipt.program_features.starts_with("token_bag/v1"),
        "{}",
        receipt.program_features
    );
    assert!(receipt.summaries.contains_key("ridge_program"));
    assert_eq!(receipt.unavailable_selectors.len(), 1);
    for c in &receipt.contexts {
        let m = &c.metrics["oracle"];
        assert!(c.oracle_gain > 0.0);
        assert!((m.realized_gain - c.oracle_gain).abs() < 1e-12);
        assert_eq!(m.normalized_regret, Some(0.0));
        let u = &c.metrics[UNIFORM];
        assert!(u.normalized_regret.unwrap() > 0.0 && u.normalized_regret.unwrap() < 1.0);
    }
}

#[test]
fn exported_views_carry_no_outcomes() {
    let (pools, manifest) = freeze(&synthetic_rows(3, 6, 10), &rule(), &h("input")).unwrap();
    let views = export_views(&pools, &manifest).unwrap();
    assert_eq!(views.len(), pools.len());
    for (view, pool) in views.iter().zip(&pools) {
        let text = serde_json::to_string(view).unwrap();
        for forbidden in ["\"q\"", "status", "incumbent", "outcome"] {
            assert!(!text.contains(forbidden), "{forbidden} leaked into {text}");
        }
        assert_eq!(view.programs.len(), pool.members.len());
    }
    let (mut tampered, _) = freeze(&synthetic_rows(3, 6, 10), &rule(), &h("input")).unwrap();
    tampered[0].members.pop();
    assert!(export_views(&tampered, &manifest).is_err());
}

/// Re-derive digest and metadata after a semantic mutation, so only the invariant is wrong.
fn rehash(pools: &[Pool], manifest: &mut Manifest) {
    for (pool, entry) in pools.iter().zip(&mut manifest.pools) {
        entry.parent_id = pool.parent_id.clone();
        entry.members = pool.members.len();
        entry.measured = pool
            .members
            .iter()
            .filter(|m| m.outcome().is_some())
            .count();
        entry.pool_sha256 = pool.sha256().unwrap();
    }
}

fn assert_refused(pools: &[Pool], manifest: &Manifest, needle: &str) {
    let inputs = Inputs {
        context_features: None,
        program_features: None,
        external: vec![],
        unavailable: vec![],
    };
    let checks = [
        ("verify", verify(pools, manifest).err()),
        ("views", export_views(pools, manifest).err()),
        ("evaluate", evaluate(pools, manifest, &inputs, &cfg()).err()),
    ];
    for (path, err) in checks {
        let err = err.unwrap_or_else(|| panic!("{path} accepted an artifact violating {needle:?}"));
        assert!(
            format!("{err:#}").contains(needle),
            "{path}: {err:#} lacks {needle:?}"
        );
    }
}

#[test]
fn reloaded_valid_pools_pass_and_rehashed_invalid_pools_are_refused() {
    let (pools, manifest) = freeze(&synthetic_rows(6, 18, 12), &rule(), &h("input")).unwrap();
    // Valid freeze -> JSON -> reload path.
    let reload: Vec<Pool> = pools
        .iter()
        .map(|p| serde_json::from_str(&serde_json::to_string(p).unwrap()).unwrap())
        .collect();
    let remanifest: Manifest =
        serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
    verify(&reload, &remanifest).unwrap();
    export_views(&reload, &remanifest).unwrap();
    let ok = Inputs {
        context_features: None,
        program_features: None,
        external: vec![],
        unavailable: vec![],
    };
    evaluate(&reload, &remanifest, &ok, &cfg()).unwrap();

    type Mutation = (&'static str, fn(&mut Vec<Pool>, &mut Manifest));
    let cases: [Mutation; 12] = [
        ("may not enter a shadow pool", |p, _| {
            p[1].role = "sealed".into()
        }),
        ("may not enter a shadow pool", |p, _| {
            p[2].role = "validation".into()
        }),
        ("unknown outcome", |p, _| {
            p[0].members[0].status = Status::Unknown;
            p[0].members[0].q = None;
        }),
        ("q present", |p, _| {
            p[0].members[1].status = Status::ProgramFailed;
            p[0].members[1].q = Some(0.4);
        }),
        ("complete row without q", |p, _| {
            p[0].members[2].status = Status::Complete;
            p[0].members[2].q = None;
        }),
        ("min_pool", |p, _| p[3].members.truncate(4)),
        ("parent_id is required", |p, _| p[0].parent_id = " ".into()),
        ("lowercase sha256", |p, m| {
            p[0].context_id = "0".into();
            m.pools[0].context_id = "0".into();
        }),
        ("incumbent_q must be finite", |p, _| {
            p[0].incumbent_q = Some(f64::NAN)
        }),
        ("distinct and sorted", |p, _| {
            let dup = p[0].members[0].clone();
            p[0].members.insert(1, dup);
        }),
        ("both pooled and excluded", |_, m| {
            m.excluded.push(Exclusion {
                context_id: m.pools[0].context_id.clone(),
                reason: "fixture".into(),
            });
        }),
        ("shadow-only", |_, m| m.qualified_claim = true),
    ];
    for (needle, mutate) in cases {
        let (mut bad, mut bad_manifest) = (pools.clone(), manifest.clone());
        mutate(&mut bad, &mut bad_manifest);
        rehash(&bad, &mut bad_manifest);
        assert_refused(&bad, &bad_manifest, needle);
    }
    // Metadata that disagrees with a correctly hashed pool is refused too.
    let mut stale = manifest.clone();
    stale.pools[0].measured += 1;
    assert_refused(&pools, &stale, "metadata differs");
}
