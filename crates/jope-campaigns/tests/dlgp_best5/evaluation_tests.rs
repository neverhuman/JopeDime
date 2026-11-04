use super::*;

#[test]
fn paired_log_rmse_comparison_applies_pilot_gates_and_tiebreaks() {
    let datasets = (0..20)
        .map(|index| digest(&format!("comparison-{index}")))
        .collect::<Vec<_>>();
    let mut scores = Vec::new();
    for dataset in &datasets {
        scores.push(ComparisonScoreV1 {
            arm: Best5ArmV1::ConfidenceGpControl,
            dataset_sha256: dataset.clone(),
            rmse: 1.0,
            latency_seconds: 1.0,
            finalist_validity: 1.0,
            overall_validity: 1.0,
            strict_benchmark_win: false,
            preserved_existing_win: true,
            predicted_cpu_seconds: 10.0,
        });
        for arm in Best5ArmV1::EXPERIMENTS {
            scores.push(ComparisonScoreV1 {
                arm,
                dataset_sha256: dataset.clone(),
                rmse: 0.9,
                latency_seconds: 1.0,
                finalist_validity: 1.0,
                overall_validity: 1.0,
                strict_benchmark_win: false,
                preserved_existing_win: true,
                predicted_cpu_seconds: 10.0,
            });
        }
    }
    let comparison = compare_campaign(
        &digest("campaign comparison"),
        DatasetStageV1::Pilot,
        &datasets,
        &scores,
    )
    .unwrap();
    assert_eq!(
        comparison.decision,
        CampaignDecisionV1::PilotWinner {
            arm: Best5ArmV1::EvolvabilityUcbLineages
        }
    );
    assert!(comparison.arms.iter().all(|arm| arm.advances));
}

#[test]
fn scq_plan_has_protected_wave_and_enforces_exact_two_percent_disk_floor() {
    let campaign = campaign();
    let blocked = LaunchEvidenceV1 {
        schema: LaunchEvidenceV1::SCHEMA.into(),
        scq_doctor_success: true,
        evaluator_evidence_age_seconds: 1,
        supply_evidence_age_seconds: 1,
        max_evidence_age_seconds: 60,
        reviewed_action_acknowledged: true,
        observed_project_priority_weight: 10,
        eligible_worker_disk_free_fractions: vec![0.02, 0.019_999],
    };
    let plan = build_scq_plan(&campaign, &blocked, false, None).unwrap();
    assert!(!plan.launch_gate_passed);
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::ProtectedPilot)
            .count(),
        24
    );
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::RemainingPilot)
            .count(),
        216
    );
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::PilotSealedScore)
            .count(),
        0
    );

    let mut ready = blocked;
    ready.eligible_worker_disk_free_fractions = vec![0.02, 0.049];
    assert!(ready.disk_warning().unwrap());
    let released = build_scq_plan(
        &campaign,
        &ready,
        true,
        Some(Best5ArmV1::HeadFamilyMapElites),
    )
    .unwrap();
    assert!(released.launch_gate_passed);
    verify_submission_weight(&released, 10).unwrap();
    assert!(verify_submission_weight(&released, 1)
        .unwrap_err()
        .contains("submission aborted"));
    assert_eq!(
        released
            .jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::PilotSealedScore)
            .count(),
        20
    );
    assert_eq!(
        released
            .jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::ConfirmationSearch)
            .count(),
        88
    );
}

#[test]
fn residual_and_portfolio_predictors_round_trip_and_reconstruct() {
    let residual = FrozenPredictorV1::ResidualComposite {
        base: genome(0),
        residual_pipeline: genome(1),
        eta: 0.25,
    };
    let residual_values = residual
        .reconstruct(&[vec![1.0, 2.0], vec![4.0, -4.0]])
        .unwrap();
    assert_eq!(residual_values, vec![2.0, 1.0]);
    let encoded = serde_json::to_vec(&residual).unwrap();
    let decoded: FrozenPredictorV1 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, residual);
    assert_eq!(decoded.semantic_sha256(), residual.semantic_sha256());

    let portfolio = FrozenPredictorV1::SparsePortfolio {
        pipelines: vec![genome(2), genome(3), genome(4)],
        weights: vec![0.2, 0.3, 0.5],
    };
    assert_eq!(
        portfolio
            .reconstruct(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]])
            .unwrap(),
        vec![2.3, 4.6]
    );
    let mut invalid = portfolio;
    if let FrozenPredictorV1::SparsePortfolio { weights, .. } = &mut invalid {
        weights[0] = -0.1;
    }
    assert!(invalid.validate().is_err());
}

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn training_staging_rebuilds_stable_rows_without_resolving_test() {
    let mut campaign = campaign();
    campaign.sotarena_root = sotarena_fixture_root(&campaign.datasets[0].dataset_id);
    let dataset = &mut campaign.datasets[0];
    dataset.train_relative_path = format!(
        "data/tabular-regression-v1/{}/train.csv",
        dataset.dataset_id
    );
    dataset.metadata_relative_path = format!(
        "splits/tabular-regression-v1/{}/metadata.json",
        dataset.dataset_id
    );
    dataset.test_relative_path = "intentionally/not/resolved/test.csv".into();
    let dataset_id = dataset.dataset_id.clone();
    let train_rows = dataset.train_rows;
    let first = build_training_table(&campaign, &dataset_id).unwrap();
    let replay = build_training_table(&campaign, &dataset_id).unwrap();
    assert_eq!(first.target_rows, replay.target_rows);
    assert_eq!(first.target_rows.len(), train_rows as usize);
    assert_eq!(
        first
            .target_rows
            .iter()
            .map(|row| &row.stable_row_id)
            .collect::<BTreeSet<_>>()
            .len(),
        first.target_rows.len()
    );
    assert!(!first.training_artifact.path.contains("test.csv"));
    assert!(!first.metadata_artifact.path.contains("test.csv"));
}

#[test]
fn dataset_level_sealed_claim_slot_rejects_competing_selection_sets() {
    let directory = tempfile::tempdir().unwrap();
    let store = ArtifactStoreV1::new(directory.path());
    let campaign = digest("claim-campaign");
    let dataset = digest("claim-dataset");
    let slot = sealed_claim_slot_identity(&campaign, &dataset).unwrap();
    let first = SealedClaimV1 {
        schema: SealedClaimV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.clone(),
        dataset_sha256: dataset.clone(),
        selection_freeze_sha256s: vec![digest("selection-a"), digest("selection-b")],
        access_ordinal: 1,
    };
    let second = SealedClaimV1 {
        selection_freeze_sha256s: vec![digest("selection-a"), digest("selection-c")],
        ..first.clone()
    };
    store.publish(&slot, &first).unwrap();
    assert!(store.publish(&slot, &first).is_ok());
    assert!(store
        .publish(&slot, &second)
        .unwrap_err()
        .contains("collision"));
}

#[test]
fn public_finalizer_bundle_shape_rejects_bare_search_result() {
    let bare = serde_json::to_vec(&search_result()).unwrap();
    assert!(serde_json::from_slice::<SearchBundleV1>(&bare).is_err());
}

#[test]
fn scq_renderer_declares_immutable_inputs_and_outputs() {
    let job = ScqJobV1 {
        identity: "best5-render-test".into(),
        stage: ScqStageV1::ProposalAll42,
        dataset_id: None,
        arm: None,
        master_seed: None,
        gpus: 1,
        dependencies: Vec::new(),
    };
    let rendered = render_scq_spec_with_io(
        &job,
        &["worker".into(), "propose-all".into()],
        "/work",
        &[ScqInputV1 {
            name: "worker".into(),
            source: "/immutable/worker".into(),
            target: "worker".into(),
            sha256: digest("worker"),
            executable: true,
        }],
        &[ScqOutputV1 {
            name: "artifacts".into(),
            source: "worker-output".into(),
            directory: true,
        }],
    );
    assert!(rendered.contains("[[input]]"));
    assert!(rendered.contains("sha256 ="));
    assert!(rendered.contains("executable = true"));
    assert!(rendered.contains("[[output]]"));
    assert!(rendered.contains("ram_mb = 8192"));
    assert!(rendered.contains("JOPE_MIN_WORKER_DISK_FREE_FRACTION_MICROS = \"20000\""));
    assert!(rendered.contains("JOPE_EXPECTED_PROJECT_PRIORITY_WEIGHT = \"10\""));
}

#[test]
fn emit_plan_checks_every_immutable_spec_and_never_submits() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("scq.log");
    let fake_scq = directory.path().join("scq");
    fs::write(
        &fake_scq,
        format!(
            "#!/bin/sh\n[ \"$1\" = check ] || exit 91\nprintf '%s\\n' \"$2\" >> '{}'\n",
            log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_scq, fs::Permissions::from_mode(0o755)).unwrap();
    let campaign = campaign();
    let evidence = LaunchEvidenceV1 {
        schema: LaunchEvidenceV1::SCHEMA.into(),
        scq_doctor_success: false,
        evaluator_evidence_age_seconds: 1,
        supply_evidence_age_seconds: 1,
        max_evidence_age_seconds: 60,
        reviewed_action_acknowledged: false,
        observed_project_priority_weight: 10,
        eligible_worker_disk_free_fractions: vec![0.09],
    };
    let plan = build_scq_plan(&campaign, &evidence, false, None).unwrap();
    let input = |name: &str| ScqInputV1 {
        name: name.into(),
        source: format!("/immutable/{name}"),
        target: name.into(),
        sha256: digest(name),
        executable: name == "worker",
    };
    let checked = emit_and_check_plan(
        &plan,
        input("worker"),
        input("campaign"),
        &directory.path().join("specs"),
        "/work",
        &fake_scq,
        |job| {
            Ok((job.stage != ScqStageV1::ProposalAll42)
                .then(|| input("proposal_batch"))
                .into_iter()
                .collect())
        },
    )
    .unwrap();
    assert_eq!(checked.len(), plan.jobs.len());
    assert_eq!(
        fs::read_to_string(log).unwrap().lines().count(),
        plan.jobs.len()
    );
    assert!(checked.iter().all(|path| {
        let spec = fs::read_to_string(path).unwrap();
        spec.contains("[[input]]") && spec.contains("[[output]]")
    }));
}
