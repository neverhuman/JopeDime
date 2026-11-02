use super::*;
#[test]
fn new_fence_preserves_original_rendezvous_and_optimizer_window() {
    let t = tempfile::tempdir().unwrap();
    let job = jain_smartcluster::JobId::new();
    let first = jain_smartcluster::FencedAttempt {
        job_id: job,
        number: jain_smartcluster::AttemptNumber::new(1).unwrap(),
        fence: jain_smartcluster::FenceToken::new(24).unwrap(),
    };
    let second = jain_smartcluster::FencedAttempt {
        job_id: job,
        number: jain_smartcluster::AttemptNumber::new(2).unwrap(),
        fence: jain_smartcluster::FenceToken::new(25).unwrap(),
    };
    let window = json!({"initial_optimizer_step":60000,"requested_step_increment":100,"target_optimizer_step":60100});
    let a = native_attempt_launch(t.path(), first, &window).unwrap();
    fs::write(a.join("nccl.id"), b"old rendezvous").unwrap();
    let b = native_attempt_launch(t.path(), second, &window).unwrap();
    assert_ne!(a, b);
    assert!(!b.join("nccl.id").exists());
    assert_eq!(fs::read(a.join("nccl.id")).unwrap(), b"old rendezvous");
    let old: Value = read_json(&a.join("native-attempt.json")).unwrap();
    let new: Value = read_json(&b.join("native-attempt.json")).unwrap();
    assert_eq!(old["original_window_sha256"], new["original_window_sha256"]);
    assert!(native_attempt_launch(t.path(), first, &window).is_err());
}
