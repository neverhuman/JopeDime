use super::*;
use tempfile::tempdir;

fn digest(value: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(value.as_bytes())
}

fn write_reject_summary(root: &Path, parent: &Sha256Digest, candidate: &str, created: u64) {
    let completed = root.join("training/shadow-canaries/completed");
    fs::create_dir_all(&completed).unwrap();
    atomic_json(
        &completed.join(format!("{candidate}.json")),
        &json!({
            "schema": "ShadowBattleSummary.v1",
            "mode": "shadow",
            "task": "binary",
            "accepted_parent_checkpoint_sha256": parent,
            "candidate_checkpoint_sha256": candidate,
            "created_at_ms": created,
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

fn write_passing_summary(root: &Path, parent: &Sha256Digest, candidate: &str, created: u64) {
    let completed = root.join("training/shadow-canaries/completed");
    fs::create_dir_all(&completed).unwrap();
    atomic_json(
        &completed.join(format!("{candidate}.json")),
        &json!({
            "schema": "ShadowBattleSummary.v1",
            "mode": "shadow",
            "task": "binary",
            "accepted_parent_checkpoint_sha256": parent,
            "candidate_checkpoint_sha256": candidate,
            "created_at_ms": created,
            "contexts": vec![json!({"context": "x"}); 16],
            "accepted_validity": 1.0,
            "gp_validity": 1.0,
            "replay_verification_complete": true,
            "matched_comparisons_complete": true,
            "comparison_lower_bound_method": SHADOW_COMPARISON_LOWER_BOUND_METHOD,
            "candidate_vs_accepted": {
                "validity": 1.0,
                "pairwise_lift": 0.1748,
                "pairwise_lift_lower_bound": 0.01,
                "latency_ratio": 0.9,
                "stability": 1.0
            },
            "candidate_vs_gp": {
                "pairwise_lift": 0.0698,
                "pairwise_lift_lower_bound": 0.01
            }
        }),
    )
    .unwrap();
}

include!("state.rs");
include!("promotion.rs");
include!("recovery.rs");
include!("recipe.rs");
