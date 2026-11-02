use super::*;
use std::os::unix::net::UnixListener;

fn record(project: &str, state: &str, epoch: &str, wave: Option<&str>) -> JobRecord {
    JobRecord {
        job_id: format!("{project}-{state}-{epoch}-{}", wave.unwrap_or("none")),
        project: project.to_owned(),
        state: state.to_owned(),
        epoch: epoch.to_owned(),
        strict_wave: wave.map(str::to_owned),
        generation: wave.map(str::to_owned),
        checkpoint_sha256: None,
        claim_state: Some("submitted".to_owned()),
        manifest: Some("/manifest".to_owned()),
        manifest_sha256: Some("a".repeat(64)),
        manifest_checked: true,
        manifest_valid: true,
        invalid_reason: None,
    }
}

fn seed(semantic: &str, mean: f64, dl_origin: bool) -> SeedRow {
    SeedRow {
        task: "regression".to_owned(),
        semantic_sha256: semantic.to_owned(),
        canonical_genome: semantic.to_owned(),
        dataset_count: 2,
        mean_within_dataset_percentile: mean,
        worst_within_dataset_percentile: mean,
        strict_win_count: 0,
        dl_origin,
    }
}

mod mutation_tests;
mod queue_tests;
