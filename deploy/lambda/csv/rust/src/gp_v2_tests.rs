use super::*;
use super::*;
fn fixture(task: &str) -> (Vec<u8>, Job) {
    let raw = (0..100)
        .map(|i| {
            format!(
                "{},{},{},{}\n",
                i as f64 / 100.0,
                (i * 17 % 101) as f64 / 100.0,
                (i * 31 % 97) as f64 / 100.0,
                if task == "regression" {
                    i as f64 * 0.7
                } else if task == "binary" {
                    (i % 2) as f64
                } else {
                    (i % 3) as f64 / 2.0
                }
            )
        })
        .collect::<String>()
        .into_bytes();
    let job = Job {
        bucket: default_bucket(),
        dataset_hash: "fixture".into(),
        name: "fixture".into(),
        key: "blobs/sha256/fixture".into(),
        sha256: hash(&raw),
        version: "v1".into(),
        rows: 100,
        columns: 4,
        bytes: raw.len(),
        task: task.into(),
        partition: "experiment".into(),
        metadata: BTreeMap::new(),
    };
    (raw, job)
}
#[test]
fn serialized_float_state_preserves_ieee_bits() {
    let mut bits = 0x123456789abcdef0u64;
    for _ in 0..10000 {
        bits = bits
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = f64::from_bits(bits);
        if x.is_finite() {
            let encoded = serde_json::to_vec(&x).unwrap();
            let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(
                x.to_bits(),
                decoded.to_bits(),
                "{}",
                String::from_utf8(encoded).unwrap()
            );
        }
    }
}

fn mock_archive_url(id: &str) -> String {
    format!("https://{OUTPUT_BUCKET}.s3.us-east-2.amazonaws.com/{OUTPUT_PREFIX}{id}.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=TEST%2F20260928%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20260928T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=content-type%3Bhost%3Bif-none-match%3Bx-amz-server-side-encryption&X-Amz-Signature={}","0".repeat(64))
}
#[test]
fn archive_capability_is_bound_to_exact_immutable_attempt() {
    let id = "0123456789abcdef0123456789abcdef";
    let url = mock_archive_url(id);
    assert_eq!(
        validate_output_url(id, &url).unwrap(),
        format!("{OUTPUT_PREFIX}{id}.json")
    );
    for bad in [
        url.replace(OUTPUT_BUCKET, "other-bucket"),
        url.replace("us-east-2", "us-west-2"),
        url.replace("raw-attempts/", "other/"),
        url.replace("Expires=900", "Expires=901"),
        url.replace("if-none-match", "arbitrary-header"),
        url.replace("x-amz-server-side-encryption", "other-header"),
        format!("{url}#fragment"),
    ] {
        assert!(validate_output_url(id, &bad).is_err());
    }
    assert!(validate_output_url("ffffffffffffffffffffffffffffffff", &url).is_err());
    assert!(output_key("INVALID").is_err());
    assert!(output_key("0123456789ABCDEF0123456789ABCDEF").is_err());
}
#[test]
fn lambda_requests_cannot_omit_durable_output_capability() {
    assert!(serde_json::from_value::<Request>(
        json!({"job_ids":[0],"download_urls":["unused"],"seed":42})
    )
    .is_err());
}
#[test]
fn folds_disjoint_complete() {
    for n in 5..500 {
        let f = fold(n, "x", 42, false);
        let union: BTreeSet<usize> = f
            .fit
            .iter()
            .chain(&f.selection)
            .chain(&f.audit)
            .copied()
            .collect();
        assert_eq!(union.len(), n);
        assert_eq!(f.fit.len() + f.selection.len() + f.audit.len(), n);
        assert!(!f.audit.is_empty());
    }
}
#[test]
fn curated_bucket_requires_exact_namespace() {
    let (_, mut job) = fixture("regression");
    assert!(admit(&job).is_ok());
    job.bucket = LEARNING_BUCKET.into();
    assert!(admit(&job).is_err());
    job.key = format!("{}curated-inputs/example.csv", CURATED_PREFIX);
    assert!(admit(&job).is_ok());
    job.bucket = "unrelated-bucket".into();
    assert!(admit(&job).is_err());
}
#[test]
fn sealed_metadata_refused() {
    let (_, mut job) = fixture("regression");
    job.metadata.insert("sealed".into(), json!(true));
    assert!(admit(&job).is_err());
    job.metadata.remove("sealed");
    job.metadata.insert(
        "historical_partitions".into(),
        json!(["meta_train", "meta_test"]),
    );
    assert!(admit(&job).is_err());
}
#[test]
fn malformed_and_nonfinite_cells_refused() {
    for s in ["NaNxxx", "inf", "1\"", "\"1", "\"1\"2"] {
        assert!(number(s).is_err());
    }
    assert_eq!(number("\"1.5\"").unwrap(), Some(1.5));
}
#[test]
fn failure_keeps_stage_timestamps() {
    let (raw, mut job) = fixture("regression");
    job.sha256 = "0".repeat(64);
    let value = one(&raw, &job, 42, 25).unwrap();
    assert_eq!(value["status"], "unsupported_or_failed");
    let t = &value["timestamps"];
    assert!(t["prepare_start_utc_us"].as_u64().is_some());
    assert!(t["prepare_end_utc_us"].as_u64().is_some());
    assert!(t.get("train_start_utc_us").is_none());
}
#[test]
fn corrupted_input_rejected() {
    let (raw, mut job) = fixture("regression");
    job.sha256 = "0".repeat(64);
    assert!(prepare(&raw, &job, 42).is_err());
}
#[test]
fn costly_grammar_rejected() {
    assert!(safe_genome("(pipe X (gpr))").is_err());
    assert!(safe_genome("(pipe X (cheb degree=4) (ridge))").is_err());
    assert!(safe_genome("(pipe X (rank append=true) (ridge))").is_err());
}
classification_and_regression_replay_and_audit_isolation!();
projection_and_vocabulary_ignore_audit_data!(|raw: &[u8]| Job {
    bucket: default_bucket(),
    dataset_hash: "wide".into(),
    name: "wide".into(),
    key: "blobs/sha256/x".into(),
    sha256: hash(raw),
    version: "v".into(),
    rows: 100,
    columns: 71,
    bytes: raw.len(),
    task: "regression".into(),
    partition: "".into(),
    metadata: BTreeMap::new(),
});
