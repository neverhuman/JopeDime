use super::*;

fn corpus_row(
    dataset: &str,
    task: &str,
    metric: &str,
    score: f64,
    tier: &str,
    valid: bool,
    ts: u64,
) -> String {
    json!({
        "dataset_id": dataset, "source": "internal_test_exact_selfplay",
        "embedding": [0.1, 0.2, 0.3], "target_embedding": [0.0],
        "genome": format!("(pipe X (ridge alpha={score}))"), "canonical_genome": format!("(pipe X (ridge alpha={score}))"),
        "valid": valid, "metric": metric, "score": score, "cost_ms": 12.5, "engine": "featmath",
        "metadata": {"task_type": task, "dataset_hash": dataset, "split": "train", "score_tier": tier,
            "selection_bias": "test-exact-reused", "benchmark_valid": false, "generation": "strict-wave-000003",
            "semantic_plan_hash": sha256_hex(format!("{dataset}{score}").as_bytes()),
            "receipt_sha256": sha256_hex(b"r"), "label_created_at_ms": ts}
    })
    .to_string()
        + "\n"
}

#[test]
fn snapshot_maxima_are_bound_to_the_manifest_hashes() {
    let tmp = tempfile::tempdir().unwrap();
    let train = [
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.5,
            "test_exact",
            true,
            1_785_000_000_000,
        ),
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.7,
            "test_exact",
            true,
            1_785_000_001_000,
        ),
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.9,
            "test_exact",
            false,
            0,
        ), // invalid
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.95,
            "cv_robust",
            true,
            0,
        ), // wrong tier
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "auc",
            0.6,
            "test_exact",
            true,
            0,
        ),
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "logloss",
            0.1,
            "test_exact",
            true,
            0,
        ), // wrong metric
    ]
    .concat();
    let val = [
        corpus_row(
            "aaaaaaaaaaaaaaaa",
            "regression",
            "r2",
            0.7,
            "test_exact",
            true,
            1_785_000_002_000,
        ), // tie keeps train
        corpus_row(
            "bbbbbbbbbbbbbbbb",
            "binary",
            "auc",
            0.8,
            "test_exact",
            true,
            1_785_000_003_000,
        ),
    ]
    .concat();
    let train_p = tmp.path().join("corpus.train.jsonl");
    let val_p = tmp.path().join("corpus.val.jsonl");
    fs::write(&train_p, &train).unwrap();
    fs::write(&val_p, &val).unwrap();
    let manifest = json!({
        "schema": "JopeTestOnlyTrainingCorpus.v1", "source_label_watermark": 4754860,
        "counts": {"regression/train": 4, "binary/train": 2, "regression/validation": 1, "binary/validation": 1},
        "strict_filter": {"binary": {"metric": "auc", "score_tier": "test_exact"}, "regression": {"metric": "r2", "score_tier": "test_exact"}, "selection_bias": "test-exact-reused"},
        "train_sha256": sha256_hex(train.as_bytes()), "validation_sha256": sha256_hex(val.as_bytes()),
    });
    let m_p = tmp.path().join("corpus-manifest.json");
    fs::write(&m_p, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let state_dir = tmp.path().join("state");
    let rows_t = format!("train={}", train_p.display());
    let rows_v = format!("validation={}", val_p.display());
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "snapshot",
            m_p.to_str().unwrap(),
            "--rows",
            &rows_t,
            "--rows",
            &rows_v,
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["scan"]["counts_match_manifest"], true);
    assert_eq!(rep["scan"]["splits"]["train"]["sha256_matches"], true);
    assert_eq!(rep["scan"]["splits"]["train"]["rows"], 6);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_invalid"], 1);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_tier"], 1);
    assert_eq!(rep["scan"]["splits"]["train"]["skipped_metric"], 1);
    assert_eq!(rep["scan"]["maxima"], 2);
    assert_eq!(rep["accepted"], 2);
    let state = State::at(&state_dir);
    let cut = snapshot::cut_id("test-exact-reused").unwrap();
    let a = exact_current(&state, "regression", "aaaaaaaaaaaaaaaa", &cut);
    assert_eq!(a.score.value, 0.7);
    assert_eq!(a.provenance.run_id, "strict-wave-000003/train");
    assert_eq!(a.provenance.source, "snapshot-4754860");
    assert_eq!(a.evaluator.sha256, "unbound");
    assert_eq!(a.ts_unix, 1_785_000_001);
    let b = exact_current(&state, "binary", "bbbbbbbbbbbbbbbb", &cut);
    assert_eq!(
        (b.score.value, b.provenance.run_id.as_str()),
        (0.8, "strict-wave-000003/validation")
    );
    // A stream that does not hash to the manifest is refused outright.
    fs::write(&val_p, val.replace("0.8", "0.81")).unwrap();
    let r = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "snapshot",
            m_p.to_str().unwrap(),
            "--rows",
            &rows_v,
        ],
    );
    assert_eq!(r.status, 2);
    assert!(r.stderr.contains("manifest binds"), "{}", r.stderr);
    assert_eq!(jl(&state_dir, &["hwm", "verify"]).ok().json()["events"], 2);
}

pub(crate) const HEADER: &str = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n";
