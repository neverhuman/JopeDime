use super::*;

fn pilot_results(delta: f64) -> Vec<PilotDatasetResultV1> {
    let mut results = Vec::new();
    for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
        for index in 0..10 {
            let dataset = digest(&format!("pilot-{task:?}-{index}"));
            results.push(PilotDatasetResultV1 {
                arm: CampaignArmV1::LiteRaceControl,
                dataset_sha256: dataset.clone(),
                task,
                sealed_skill: 0.5,
                cpu_seconds: 10.0,
                winner_latency_seconds: 1.0,
                generation_zero_validity: 1.0,
                finalist_validity: 1.0,
                overall_validity: 1.0,
                sealed_winner: true,
            });
            for (arm, bonus, cpu) in [
                (CampaignArmV1::RotatingMean, 0.0, 10.0),
                (CampaignArmV1::ConfidenceGp, 0.001, 11.0),
                (CampaignArmV1::QualityDiversityGp, 0.002, 9.0),
            ] {
                results.push(PilotDatasetResultV1 {
                    arm,
                    dataset_sha256: dataset.clone(),
                    task,
                    sealed_skill: 0.5 + delta + bonus,
                    cpu_seconds: cpu,
                    winner_latency_seconds: 1.0,
                    generation_zero_validity: 1.0,
                    finalist_validity: 1.0,
                    overall_validity: 1.0,
                    sealed_winner: true,
                });
            }
        }
    }
    results
}

#[test]
fn pilot_bootstrap_gates_and_typed_no_improvement_are_deterministic() {
    let campaign = digest("campaign");
    let seed = digest("bootstrap");
    let first = compare_pilot(&campaign, &seed, &pilot_results(0.02)).unwrap();
    let second = compare_pilot(&campaign, &seed, &pilot_results(0.02)).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.decision,
        PilotDecisionV1::Advance {
            arm: CampaignArmV1::QualityDiversityGp
        }
    );
    let stopped = compare_pilot(&campaign, &seed, &pilot_results(-0.01)).unwrap();
    assert!(matches!(
        stopped.decision,
        PilotDecisionV1::NoImprovement { .. }
    ));
}
