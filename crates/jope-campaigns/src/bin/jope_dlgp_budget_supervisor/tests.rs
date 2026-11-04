use super::*;

fn digest(label: &str) -> String {
    sha256_bytes(label.as_bytes())
}

#[test]
fn exact_disk_boundary_blocks_below_two_percent_and_only_warns_below_five() {
    let gate = LaunchGates {
        schema: "JopeDlgpBudgetLaunchGates.v1".to_owned(),
        protocol_sha256: "a".repeat(64),
        proposal_checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.to_owned(),
        fresh_evaluator_evidence: true,
        fresh_supply_evidence: true,
        reviewed_acknowledgement: true,
        real_scq: true,
        observed_project_weight: 10,
        expected_project_weight: 10,
        disk_free_fraction_micros: 20_000,
        generated_at_ms: 100,
    };
    assert!(gate.blockers(&"a".repeat(64), 100).is_empty());
    let mut below = gate;
    below.disk_free_fraction_micros = 19_999;
    assert!(below
        .blockers(&"a".repeat(64), 100)
        .iter()
        .any(|value| value.contains("2%")));
}

#[test]
fn status_counts_partial_evaluations_and_keeps_tracks_separate() {
    let status = JopeDlgpBudgetCampaignStatusV1 {
        schema: JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA.to_owned(),
        generated_at_ms: 10,
        last_advanced_at_ms: 9,
        protocol_sha256: "a".repeat(64),
        stage: "stage1".to_owned(),
        gate_state: DlgpCampaignGateStateV1::Runnable,
        trajectory_count: 20,
        completed_trajectory_count: 2,
        rung_counts: BTreeMap::from([("1000".to_owned(), 3)]),
        evaluations_used: 3_500,
        evaluations_cap: 200_000,
        active_jobs: Vec::new(),
        robust_wins: 3,
        robust_wins_target: 7,
        robust_wins_denominator: 22,
        seed_agreement: "3/22".to_owned(),
        license_track: "separate".to_owned(),
        production_compatible_state: "active".to_owned(),
        research_only_state: "not_started".to_owned(),
        test_opening_state: "sealed".to_owned(),
        promotion_decisions: Vec::new(),
        blockers: Vec::new(),
    };
    status.validate().unwrap();
    assert_eq!(status.evaluations_used, 3_500);
    assert_ne!(
        status.production_compatible_state,
        status.research_only_state
    );
}

#[test]
fn recovery_spec_changes_only_the_infrastructure_identity() {
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join("original.toml");
    fs::write(
        &original,
        "[job]\nid = \"budget-traj-a\"\nproject = \"jope-dlgp-budget-first\"\n\n[env]\nSEED = \"101\"\nDATA = \"frozen\"\n",
    )
    .unwrap();
    let plan = TrajectoryPlan {
        schema: "JopeDlgpBudgetTrajectoryPlan.v1".to_owned(),
        protocol_sha256: digest("protocol"),
        trajectory_identity: "trajectory-a".to_owned(),
        dataset_sha256: digest("dataset"),
        seed: 101,
        license_track: "production_compatible".to_owned(),
        runtime_bundle_sha256: digest("runtime"),
        binary_sha256: digest("binary"),
        configuration_sha256: digest("configuration"),
        train_data_sha256: digest("train"),
        metric_sha256: digest("metric"),
        job_identity: "budget-traj-a".to_owned(),
        spec_path: original.clone(),
        spec_sha256: sha256_file(&original).unwrap(),
    };
    plan.validate(&plan.protocol_sha256).unwrap();
    let retry =
        materialize_attempt_spec(root.path(), &plan, 6, "budget-traj-a-infra000006").unwrap();
    let retry_text = fs::read_to_string(retry).unwrap();
    assert!(retry_text.contains("id = \"budget-traj-a-infra000006\""));
    assert!(retry_text.contains("SEED = \"101\""));
    assert!(retry_text.contains("DATA = \"frozen\""));
    assert!(fs::read_to_string(original)
        .unwrap()
        .contains("id = \"budget-traj-a\""));
}
