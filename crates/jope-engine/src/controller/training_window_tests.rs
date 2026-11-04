use super::training_rank::{training_step_window, verify_training_rank};
use serde_json::json;

#[test]
fn optimizer_carry_requires_new_steps_and_exact_rank_identity() {
    let window = training_step_window(9000, 1000).unwrap();
    let receipt = json!({"rank":0,"world":8,"steps":10000,"local_steps":1000,
        "initial_optimizer_step":9000,"executed_steps":1000,"corpus_sha256":"corpus"});
    verify_training_rank(&receipt, &window, 0, 8, "corpus").unwrap();
    for (key, value) in [
        ("steps", json!(9000)),
        ("local_steps", json!(0)),
        ("executed_steps", json!(0)),
        ("rank", json!(1)),
        ("world", json!(4)),
    ] {
        let mut bad = receipt.clone();
        bad[key] = value;
        assert!(
            verify_training_rank(&bad, &window, 0, 8, "corpus").is_err(),
            "{key}"
        );
    }
}

#[test]
fn exact_resume_counts_only_remaining_attempt_exposure() {
    let window = training_step_window(9000, 1000).unwrap();
    let receipt = json!({"rank":0,"world":8,"steps":10000,"local_steps":1000,
        "initial_optimizer_step":9500,"executed_steps":500,"corpus_sha256":"corpus"});
    verify_training_rank(&receipt, &window, 0, 8, "corpus").unwrap();
    assert!(training_step_window(u64::MAX, 1).is_err());
    assert!(training_step_window(0, 0).is_err());
}
