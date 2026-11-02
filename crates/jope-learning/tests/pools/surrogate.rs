use super::*;
use std::cell::RefCell;
use std::collections::BTreeSet;

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

pub(super) fn folds_keep_whole_parents_together_and_every_selector_scores_every_context_body() {
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
    let cf = Features::context_id_descriptors(&pools, 2).unwrap();
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

pub(super) fn contextual_surrogate_qhat_establishes_h2_on_interaction_pools_body() {
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
        let mut ctx_sha = h(&format!("inter-ctx-{c}"));
        let z = if (c / 2) % 2 == 0 { 1.0 } else { -1.0 };
        ctx_sha.replace_range(..2, if z > 0.0 { "ff" } else { "00" });
        cf_rows.insert(
            ctx_sha.clone(),
            Features::context_id_vector(&ctx_sha, 1).unwrap(),
        );
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

pub(super) fn missing_external_score_is_an_error_body() {
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

pub(super) fn pool_floor_reference_needs_no_incumbent_and_derives_token_features_body() {
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

pub(super) fn exported_views_carry_no_outcomes_body() {
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
