use super::*;

#[test]
fn an_epoch_label_cannot_upgrade_unknown_legacy_objective_or_exact_resume() {
    let request = Request::same();
    let mut state = parent();
    legacy(&mut state);
    state["optimizer_epoch_sha256"] = json!("a".repeat(64));
    state["policy_lineage_sha256"] = json!("b".repeat(64));
    for mode in [Mode::ExactResume, Mode::NewCorpusTransition] {
        let error = request.check(mode, &state, None).unwrap_err();
        assert!(error.to_string().contains(LEGACY_REFUSAL));
    }
    let report = request.check(Mode::WeightWarmStart, &state, None).unwrap();
    assert_eq!(report["optimizer"], "reset");
    assert_eq!(report["objective"]["continuity"], "legacy_ids_only");
}

#[test]
fn the_objective_identity_restates_every_declared_literal() {
    // Restated independently: any change to a coefficient, smoothing, weight,
    // temperature, boost, reduction or rule changes this value and its digest.
    let expected = json!({
        "schema": "jope.training_objective.v1",
        "coefficients": {"sequence_nll": 1.0, "ranking_bt": 0.25, "ranking_listwise": 0.10,
            "runtime_huber": 0.10, "contextfree_bt": 0.05, "stability": 0.0, "loss_log_vars": 0.0},
        "sequence": {
            "targets": "tokens_1_to_T_minus_1_bos_never_a_target",
            "legal_mask": "grammar_legal_set_neg_infinity",
            "label_smoothing": 0.05,
            "smoothing_support": "grammar_legal_set",
            "row_reduction": "mean_over_supervised_positions",
            "row_weight": {"floor": 0.05f32, "percentile_exponent": 4i32},
            "batch_reduction": "row_weighted_mean",
            "row_without_supervision": "refused"},
        "ranking": {
            "backward": "dense_pairwise_v2",
            "score": "mean_of_reward_heads",
            "reward_heads": 4i64,
            "group": "context_and_comparison_of_8",
            "pairs": "all_unordered_within_group",
            "pair_weight": "absolute_value_difference",
            "boost_threshold": 0.9f32,
            "boost_factor": 2.0,
            "ties": "skipped",
            "pair_reduction": "weighted_mean_over_microbatch",
            "empty_support": "connected_zero",
            "listwise_temperature": 0.25,
            "listwise_min_members": 3usize,
            "listwise_reduction": "mean_over_eligible_groups"},
        "runtime": {"objective": "huber_log1p_seconds_v2", "huber_beta": 1.0,
            "reduction": "mean_over_rows"},
        "ablation": {"included": true, "construction": "ranking_bt_on_context_free_head"},
        "inactive": ["stability", "loss_log_vars"],
        "reward_head_dropout": 0.05,
        "supervision": "validate_rows_v1",
        "nonfinite_objective": "refused",
        "normalization": "microbatch_weighted_means_rank_mean_v1"
    });
    assert_eq!(objective::identity(), expected);
    assert_eq!(
        objective::identity_sha256(),
        hash(&serde_json::to_vec(&expected).unwrap())
    );
    assert_eq!(
        (
            neural::LABEL_SMOOTHING,
            neural::LISTWISE_TEMPERATURE,
            neural::LISTWISE_MIN_MEMBERS,
            neural::BOOST_THRESHOLD,
            neural::BOOST_FACTOR,
            neural::REWARD_HEAD_DROPOUT
        ),
        (0.05, 0.25, 3, 0.9f32, 2.0, 0.05)
    );
    // The partition dependence is part of the identity's declared normalization
    // and is bound separately by the execution partition.
    assert_eq!(
        training_partition(8, 256, 2048).unwrap()["normalization"],
        expected["normalization"]
    );
}

#[test]
fn every_request_is_exactly_one_mode_and_ambiguity_is_refused() {
    let reset = json!({"optimizer":"reset"});
    let carry = json!({"optimizer":"carry"});
    for (resume, parent, policy, mode) in [
        (false, false, None, Mode::ColdInitialization),
        (false, true, None, Mode::WeightWarmStart),
        (false, true, Some(&reset), Mode::WeightWarmStart),
        (false, true, Some(&carry), Mode::NewCorpusTransition),
        (true, false, None, Mode::ExactResume),
        (true, false, Some(&carry), Mode::ExactResume),
    ] {
        assert_eq!(classify(resume, parent, policy).unwrap(), mode);
    }
    let unknown = json!({"optimizer":"partial"});
    let missing = json!({"replay":"broad"});
    for (resume, parent, policy) in [
        (true, true, None),
        (false, false, Some(&carry)),
        (false, true, Some(&unknown)),
        (false, true, Some(&missing)),
    ] {
        assert!(classify(resume, parent, policy).is_err());
    }
}

#[test]
fn the_two_identifier_guard_accepted_an_objective_change_that_the_full_identity_refuses() {
    // G1 reproduction: carry and resume compared only `ranking_backward` and
    // `runtime_objective`. A parent trained with a different coefficient
    // passes that guard unchanged.
    let changed = parent_with_changed_objective();
    let optimizer = json!({"step":3000,"sha256":"a".repeat(64)});
    let two_ids = json!({"ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2"});
    verify_resume_metadata(&changed, &two_ids, &optimizer)
        .expect("the legacy guard cannot see the objective change");
    // The full identity refuses the same parent for carry and exact resume.
    let request = Request::same();
    for mode in [Mode::NewCorpusTransition, Mode::ExactResume] {
        let error = request.check(mode, &changed, None).unwrap_err();
        assert!(
            error.to_string().contains("same objective"),
            "{mode:?}: {error}"
        );
    }
    // A weight warm start may change the objective, and says so.
    let record = request
        .check(Mode::WeightWarmStart, &changed, None)
        .unwrap();
    assert_eq!(record["objective"]["continuity"], "changed");
    assert_eq!(record["optimizer"], "reset");
}

#[test]
fn exact_resume_refuses_every_undeclared_change() {
    let record = Request::same()
        .check(Mode::ExactResume, &parent(), None)
        .unwrap();
    assert_eq!(record["mode"], "exact_resume");
    assert_eq!(record["exact_resume_claim"], true);
    assert_eq!(record["objective"]["continuity"], "identical");
    assert_eq!(record["execution_partition"]["continuity"], "identical");
    assert_eq!(record["sampler"], "continued");
    type Change = Box<dyn Fn(&mut Request, &mut Value, &mut Option<Value>)>;
    let cases: Vec<(&str, Change)> = vec![
        (
            "objective",
            Box::new(|_, s, _| *s = parent_with_changed_objective()),
        ),
        (
            "objective digest",
            Box::new(|_, s, _| s["objective_sha256"] = json!("b".repeat(64))),
        ),
        ("microbatch", Box::new(|r, _, _| r.micro = 128)),
        ("world", Box::new(|r, _, _| r.world = 4)),
        ("corpus", Box::new(|r, _, _| r.corpus = NEW_CORPUS)),
        ("vocabulary", Box::new(|r, _, _| r.vocabulary = NEW_CORPUS)),
        ("seed", Box::new(|r, _, _| r.seed = 8)),
        ("sampler", Box::new(|r, _, _| r.sampler = "other_sampler")),
        (
            "architecture",
            Box::new(|r, _, _| {
                let mut c = Config::regression(401);
                c.layers = 5;
                r.config = serde_json::to_value(c).unwrap();
            }),
        ),
        (
            "declared change",
            Box::new(|_, _, p| {
                *p = Some(json!({"declared_changes":{
                    "execution_partition":training_partition(8, 256, 2048).unwrap()}}))
            }),
        ),
        (
            "partition absent",
            Box::new(|_, s, _| drop(s.as_object_mut().unwrap().remove("execution_partition"))),
        ),
        (
            "legacy with other runtime objective",
            Box::new(|_, s, _| {
                legacy(s);
                s["runtime_objective"] = json!("huber_seconds_v1");
            }),
        ),
    ];
    for (case, change) in cases {
        let mut request = Request::same();
        let mut state = parent();
        let mut policy = None;
        change(&mut request, &mut state, &mut policy);
        assert!(
            request
                .check(Mode::ExactResume, &state, policy.as_ref())
                .is_err(),
            "exact resume accepted a {case} change"
        );
    }
    // A legacy parent (no objective identity) is refused even when both legacy
    // IDs match: it continues only as a weight warm start (owner decision
    // 2026-09-26, OWNER-DECISIONS.md D12).
    let mut state = parent();
    legacy(&mut state);
    let error = Request::same()
        .check(Mode::ExactResume, &state, None)
        .unwrap_err();
    assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
}
