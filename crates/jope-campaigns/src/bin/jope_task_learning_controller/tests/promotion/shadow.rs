use super::*;

#[test]
fn proposer_jobs_do_not_block_the_independent_trainer_lane() {
    assert!(is_task_trainer_job_id(
        "jope-selfplay-shadow-binary-005263844-s000006"
    ));
    assert!(is_task_trainer_job_id(
        "jope-selfplay-production-regression-005263844-s000007"
    ));
    assert!(is_task_trainer_job_id("jope-selfplay-trainer-legacy"));
    assert!(!is_task_trainer_job_id(
        "jope-selfplay-cohort-propose-fbcd5459f2c1-wave"
    ));
    assert!(!is_task_trainer_job_id(
        "jope-canary-propose-dc0072851563-447531cc"
    ));
}

#[test]
fn shadow_warm_start_requires_a_complete_passing_canary() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let completed = root.join("training/shadow-canaries/completed");
    fs::create_dir_all(&completed).unwrap();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let candidate = Sha256Digest::parse("b".repeat(64)).unwrap();
    let path = completed.join("summary.json");
    let summary = |validity: f64| {
        json!({
            "schema": "ShadowBattleSummary.v1",
            "mode": "shadow",
            "task": "binary",
            "accepted_parent_checkpoint_sha256": parent,
            "candidate_checkpoint_sha256": candidate,
            "created_at_ms": 1,
            "contexts": vec![json!({"context": "x"}); 16],
            "accepted_validity": 1.0,
            "gp_validity": 1.0,
            "replay_verification_complete": true,
            "matched_comparisons_complete": true,
            "comparison_lower_bound_method": SHADOW_COMPARISON_LOWER_BOUND_METHOD,
            "candidate_vs_accepted": {
                "validity": validity,
                "pairwise_lift": 0.0,
                "pairwise_lift_lower_bound": 0.01,
                "latency_ratio": 1.0,
                "stability": 1.0
            },
            "candidate_vs_gp": {"pairwise_lift_lower_bound": 0.01}
        })
    };
    atomic_json(&path, &summary(1.0)).unwrap();
    assert!(shadow_canary_warm_start_eligible(root, Task::Binary, &parent, &candidate).unwrap());
    atomic_json(&path, &summary(0.9375)).unwrap();
    assert!(!shadow_canary_warm_start_eligible(root, Task::Binary, &parent, &candidate).unwrap());
}

#[test]
fn validity_is_a_hard_shadow_gate_even_in_a_degraded_environment() {
    let environment = |accepted: f64, candidate_validity: f64| {
        json!({
            "contexts": vec![json!({"context": "x"}); 16],
            "accepted_validity": accepted,
            "gp_validity": accepted,
            "replay_verification_complete": true,
            "matched_comparisons_complete": true,
            "comparison_lower_bound_method": SHADOW_COMPARISON_LOWER_BOUND_METHOD,
            "candidate_vs_accepted": {
                "validity": candidate_validity,
                "pairwise_lift": 0.01,
                "pairwise_lift_lower_bound": 0.001,
                "latency_ratio": 1.0,
                "stability": 1.0
            },
            "candidate_vs_gp": {"pairwise_lift_lower_bound": 0.001}
        })
    };
    assert!(!shadow_summary_eligible(&environment(0.875, 0.875)));
    assert!(!shadow_summary_eligible(&environment(0.875, 0.8125)));
    assert!(!shadow_summary_eligible(&environment(1.0, 0.9766)));
    assert!(shadow_summary_eligible(&environment(1.0, 0.99)));
    let mut missing = environment(1.0, 1.0);
    missing.as_object_mut().unwrap().remove("gp_validity");
    assert!(!shadow_summary_eligible(&missing));
}

#[test]
fn shadow_gate_hard_blocks_latency_and_nonpositive_lower_bounds() {
    let canary = |accepted_lower: f64, gp_lower: f64, latency_ratio: f64| {
        json!({
            "contexts": vec![json!({"context": "x"}); 16],
            "accepted_validity": 1.0,
            "gp_validity": 1.0,
            "replay_verification_complete": true,
            "matched_comparisons_complete": true,
            "comparison_lower_bound_method": SHADOW_COMPARISON_LOWER_BOUND_METHOD,
            "candidate_vs_accepted": {
                "validity": 1.0,
                "pairwise_lift": accepted_lower,
                "pairwise_lift_lower_bound": accepted_lower,
                "latency_ratio": latency_ratio,
                "stability": 1.0
            },
            "candidate_vs_gp": {"pairwise_lift_lower_bound": gp_lower}
        })
    };
    assert!(shadow_summary_eligible(&canary(0.01, 0.01, 1.25)));
    assert!(!shadow_summary_eligible(&canary(0.01, 0.01, 1.2501)));
    assert!(!shadow_summary_eligible(&canary(0.0, 0.01, 1.0)));
    assert!(!shadow_summary_eligible(&canary(0.01, 0.0, 1.0)));
    assert!(!shadow_summary_eligible(&canary(0.01, 0.01, f64::NAN)));
}

#[test]
fn three_rejected_shadows_publish_and_hold_a_parent_plateau() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let completed = root.join("training/shadow-canaries/completed");
    fs::create_dir_all(&completed).unwrap();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
        let candidate = character.repeat(64);
        atomic_json(
            &completed.join(format!("{candidate}.json")),
            &json!({
                "schema": "ShadowBattleSummary.v1",
                "mode": "shadow",
                "task": "binary",
                "accepted_parent_checkpoint_sha256": parent,
                "candidate_checkpoint_sha256": candidate,
                "created_at_ms": index + 1,
                "contexts": vec![json!({"context": "x"}); 16],
                "accepted_validity": 1.0,
                "gp_validity": 1.0,
                "replay_verification_complete": true,
                "candidate_vs_accepted": {
                    "validity": 0.98,
                    "pairwise_lift": -0.01,
                    "latency_ratio": 1.1,
                    "stability": 1.0
                },
                "candidate_vs_gp": {
                    "pairwise_lift": -0.04
                }
            }),
        )
        .unwrap();
    }
    let accepted = BTreeMap::from([
        (
            Task::Binary,
            AcceptedPointer {
                digest: parent.clone(),
                path: PathBuf::new(),
            },
        ),
        (
            Task::Regression,
            AcceptedPointer {
                digest: Sha256Digest::parse("b".repeat(64)).unwrap(),
                path: PathBuf::new(),
            },
        ),
    ]);
    let plateau = publish_plateau_diagnoses(root, &accepted, &BTreeMap::new()).unwrap();
    assert_eq!(plateau, BTreeSet::from([Task::Binary]));
    let path = root
        .join("training/plateau-diagnoses")
        .join(format!("binary-{}.v1.json", parent.as_str()));
    let diagnosis: PlateauDiagnosisV1 = serde_json::from_value(read_json(&path).unwrap()).unwrap();
    assert_eq!(diagnosis.rejected_candidates.len(), 3);
    assert_eq!(diagnosis.gp_disadvantage, 0.04);
    // The additive v2 receipt records the (here unresolvable) recipe
    // identities and the conservative scoping that still paused.
    let v2: PlateauDiagnosisV2 = serde_json::from_value(
        read_json(
            &root
                .join("training/plateau-diagnoses")
                .join(format!("binary-{}.v2.json", parent.as_str())),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(v2.counted_candidates.len(), 3);
    assert!(v2
        .counted_candidates
        .iter()
        .all(|candidate| candidate.recipe_identity_sha256.is_none()));
    assert!(!v2.recipe_change_directive_present);
}
