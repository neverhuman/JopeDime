use super::*;

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn all_six_arms_complete_eight_generation_synthetic_worker_with_exact_ledgers() {
    let directory = tempfile::tempdir().unwrap();
    let mut campaign = campaign();
    campaign.sotarena_root = sotarena_fixture_root(&campaign.datasets[0].dataset_id);
    let dataset_id = campaign.datasets[0].dataset_id.clone();
    campaign.datasets[0].train_relative_path =
        format!("data/tabular-regression-v1/{dataset_id}/train.csv");
    campaign.datasets[0].metadata_relative_path =
        format!("splits/tabular-regression-v1/{dataset_id}/metadata.json");
    campaign.datasets[0].test_relative_path =
        format!("data/tabular-regression-v1/{dataset_id}/test.csv");
    let evaluator = fake_evaluator(&directory.path().join("corpus-eval"));
    campaign.evaluator_sha256 = evaluator.sha256.clone();
    campaign.evaluator_artifact = evaluator;

    let protocol_id = digest("synthetic-observation-protocol");
    let observations = (64..80)
        .flat_map(|index| {
            let protocol_id = protocol_id.clone();
            (0..3).map(move |support| AnchorObservationV1 {
                observation_snapshot_sha256: protocol_id.clone(),
                dataset_sha256: digest(&format!("anchor-source-{index}-{support}")),
                source_family: format!("anchor-family-{}", support % 2),
                genome: genome(index),
                within_dataset_percentile: 0.9 - index as f64 / 10_000.0,
                strict_win: true,
            })
        })
        .collect::<Vec<_>>();
    let observation_path = directory.path().join("observations.json");
    fs::write(
        &observation_path,
        serde_json::to_vec(&serde_json::json!({
            "snapshot_sha256": protocol_id,
            "observations": observations,
            "diversity_seeds": [],
        }))
        .unwrap(),
    )
    .unwrap();
    let final_bytes = fs::read(&observation_path).unwrap();
    campaign.observation_snapshot_sha256 = digest("synthetic-observation-protocol");
    campaign.observation_snapshot_artifact.path = observation_path.to_string_lossy().into_owned();
    campaign.observation_snapshot_artifact.sha256 = raw_sha256(&final_bytes);
    campaign.observation_snapshot_artifact.bytes = final_bytes.len() as u64;

    let proposal_batch = ProposalBatchFreezeV1 {
        schema: ProposalBatchFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        proposals: campaign
            .datasets
            .iter()
            .map(|dataset| proposal_for(&dataset.dataset_sha256))
            .collect(),
    };
    proposal_batch.validate(&campaign).unwrap();
    let artifact_root = directory.path().join("artifacts");
    let work_root = directory.path().join("work");
    let store = ArtifactStoreV1::new(&artifact_root);
    store
        .publish(
            &proposal_batch.identity(&campaign).unwrap(),
            &proposal_batch,
        )
        .unwrap();
    let mut control_bundle = None;
    let mut selections = Vec::new();
    for arm in Best5ArmV1::ALL {
        let bundle = run_search_one(
            &campaign,
            &proposal_batch,
            &dataset_id,
            arm,
            101,
            &artifact_root,
            &work_root,
        )
        .unwrap();
        assert_eq!(bundle.result.populations.len(), 8);
        assert!(bundle
            .result
            .populations
            .iter()
            .all(|population| population.semantic_sha256s.len() == 80));
        bundle.result.ledger.validate(arm).unwrap();
        assert_eq!(
            bundle.prediction_receipts.len(),
            bundle.result.ledger.logical_operations as usize
        );
        if arm == Best5ArmV1::CrossFittedResidualDlgp {
            assert!(matches!(
                bundle
                    .result
                    .finalists
                    .iter()
                    .find(|finalist| {
                        finalist.semantic_sha256 == bundle.result.winner_semantic_sha256
                    })
                    .unwrap()
                    .predictor,
                FrozenPredictorV1::ResidualComposite { .. }
            ));
        }
        if arm == Best5ArmV1::SparseOofPortfolioGp {
            assert!(bundle.result.finalists.iter().any(|finalist| matches!(
                finalist.predictor,
                FrozenPredictorV1::SparsePortfolio { .. }
            )));
        }
        if arm == Best5ArmV1::ConfidenceGpControl {
            verify_search_bundle(&campaign, &proposal_batch, &bundle, &artifact_root).unwrap();
            control_bundle = Some(bundle);
        } else {
            let winner = bundle
                .result
                .finalists
                .iter()
                .find(|candidate| candidate.semantic_sha256 == bundle.result.winner_semantic_sha256)
                .unwrap();
            let choices = SEARCH_SEEDS
                .into_iter()
                .map(|master_seed| SeedChoiceV1 {
                    master_seed,
                    search_result_sha256: bundle.result.identity().unwrap(),
                    winner_semantic_sha256: winner.semantic_sha256.clone(),
                    winner_predictor: winner.predictor.clone(),
                    pooled_oof_r2: winner.pooled_oof_r2,
                    fold_r2_sd: 0.0,
                    rmse: winner.rmse,
                    runtime_seconds: winner.runtime_seconds,
                    complexity: winner.complexity,
                })
                .collect::<Vec<_>>();
            selections.push(SelectionFreezeV1 {
                schema: SelectionFreezeV1::SCHEMA.into(),
                campaign_freeze_sha256: campaign.identity().unwrap(),
                arm,
                dataset_sha256: campaign.datasets[0].dataset_sha256.clone(),
                selected_seed: 101,
                winner_semantic_sha256: winner.semantic_sha256.clone(),
                winner_predictor: winner.predictor.clone(),
                choices,
            });
        }
    }

    let control_bundle = control_bundle.unwrap();
    let control_winner = control_bundle
        .result
        .finalists
        .iter()
        .find(|candidate| candidate.semantic_sha256 == control_bundle.result.winner_semantic_sha256)
        .unwrap();
    selections.push(SelectionFreezeV1 {
        schema: SelectionFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        arm: Best5ArmV1::ConfidenceGpControl,
        dataset_sha256: campaign.datasets[0].dataset_sha256.clone(),
        choices: SEARCH_SEEDS
            .into_iter()
            .map(|master_seed| SeedChoiceV1 {
                master_seed,
                search_result_sha256: control_bundle.result.identity().unwrap(),
                winner_semantic_sha256: control_winner.semantic_sha256.clone(),
                winner_predictor: control_winner.predictor.clone(),
                pooled_oof_r2: control_winner.pooled_oof_r2,
                fold_r2_sd: 0.0,
                rmse: control_winner.rmse,
                runtime_seconds: control_winner.runtime_seconds,
                complexity: control_winner.complexity,
            })
            .collect(),
        selected_seed: 101,
        winner_semantic_sha256: control_winner.semantic_sha256.clone(),
        winner_predictor: control_winner.predictor.clone(),
    });
    let sealed = run_sealed_one(&campaign, selections, &artifact_root, &work_root).unwrap();
    verify_sealed_bundle(&campaign, &sealed, &artifact_root).unwrap();
    assert_eq!(sealed.scores.len(), 6);

    let warm_replay = run_search_one(
        &campaign,
        &proposal_batch,
        &dataset_id,
        Best5ArmV1::ConfidenceGpControl,
        101,
        &artifact_root,
        &work_root,
    )
    .unwrap();
    assert_eq!(
        control_bundle.identity().unwrap(),
        warm_replay.identity().unwrap()
    );

    let mut missing_receipt = control_bundle.clone();
    missing_receipt.prediction_receipts.pop();
    assert!(missing_receipt.validate().is_err());

    let mut altered_metric = control_bundle.clone();
    altered_metric.result.archive[0].split_evidence[0].rmse += 1.0;
    assert!(
        verify_search_bundle(&campaign, &proposal_batch, &altered_metric, &artifact_root).is_err()
    );

    let mut foreign = control_bundle;
    let foreign_campaign = digest("foreign-campaign");
    foreign.staging.campaign_freeze_sha256 = foreign_campaign.clone();
    foreign.result.campaign_freeze_sha256 = foreign_campaign;
    assert!(verify_search_bundle(&campaign, &proposal_batch, &foreign, &artifact_root).is_err());
}
