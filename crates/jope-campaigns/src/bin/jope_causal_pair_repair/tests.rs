use super::{
    allocate_requests, parse_evaluator_input, CausalLane, PairRepairRequestV1, Sha256Digest, Task,
};
use std::{fs, path::PathBuf};

fn digest(byte: char) -> Sha256Digest {
    Sha256Digest::parse(byte.to_string().repeat(64)).unwrap()
}

fn request(task: Task, index: usize) -> PairRepairRequestV1 {
    PairRepairRequestV1 {
        pair_id: Sha256Digest::of_bytes(format!("pair-{task}-{index}").as_bytes()),
        task,
        missing_lane: CausalLane::NativeDlDirect,
        original_manifest_sha256: digest('f'),
        permanent_failure: false,
    }
}

#[test]
fn deficit_allocation_preserves_the_sixty_forty_bound() {
    let mut requests = Vec::new();
    for index in 0..12 {
        requests.push(request(Task::Regression, index));
    }
    for index in 0..8 {
        requests.push(request(Task::Binary, index));
    }
    let selected = allocate_requests(&requests, 20, Task::Regression);
    assert_eq!(selected.len(), 20);
    assert_eq!(
        selected
            .iter()
            .filter(|row| row.task == Task::Regression)
            .count(),
        12
    );
    assert_eq!(
        selected
            .iter()
            .filter(|row| row.task == Task::Binary)
            .count(),
        8
    );
}

#[test]
fn original_evaluator_input_is_parsed_from_the_immutable_spec() {
    let temporary = tempfile::tempdir().unwrap();
    let spec = temporary.path().join("job.toml");
    fs::write(
        &spec,
        r#"
[[input]]
name = "manifest"
source = "/tmp/manifest"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[[input]]
name = "evaluator"
source = "/tmp/corpus_eval"
target = "corpus_eval"
sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

[[output]]
name = "receipt"
"#,
    )
    .unwrap();
    let (path, sha256) = parse_evaluator_input(&spec).unwrap();
    assert_eq!(path, PathBuf::from("/tmp/corpus_eval"));
    assert_eq!(sha256, "b".repeat(64));
}
