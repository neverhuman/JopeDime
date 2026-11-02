use super::archive::{aggregate_records, read_and_materialize, read_and_materialize_filtered};
use super::publish::sha256_bytes;
use super::*;
use tempfile::TempDir;

fn record(
    dataset: &str,
    genome: &str,
    valid: bool,
    rmse: Option<f64>,
    failure: Option<&str>,
    runtime: f64,
) -> Value {
    json!({
        "schema": "fixture",
        "cohort": "golden42",
        "dataset": dataset,
        "canonical_genome": genome,
        "semantic_sha256": sha256_bytes(genome.as_bytes()),
        "valid": valid,
        "rmse": rmse,
        "failure": failure.map(|class| json!({"class": class})),
        "runtime_seconds": runtime,
        "run_id": "fixture",
    })
}

fn fixture_archive(root: &Path, rows: &[Value]) {
    fs::create_dir_all(root).unwrap();
    let mut ledger = Vec::new();
    for row in rows {
        ledger.extend(serde_json::to_vec(row).unwrap());
        ledger.push(b'\n');
    }
    fs::write(root.join("all-evaluations.jsonl"), &ledger).unwrap();
    let manifest = json!({
        "schema": "jope-dime.all-regression-analysis-archive.v1",
        "all_evaluations_jsonl": "all-evaluations.jsonl",
        "all_evaluations_jsonl_sha256": sha256_bytes(&ledger),
        "evaluation_records": rows.len(),
    });
    fs::write(
        root.join(ARCHIVE_MANIFEST),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn read_examples(root: &Path) -> Vec<Value> {
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join(ARCHIVE_MANIFEST)).unwrap()).unwrap();
    let archive_sha = manifest["all_evaluations_jsonl_sha256"].as_str().unwrap();
    fs::read_to_string(
        root.parent()
            .unwrap()
            .join("out/dime_score/corpora")
            .join(archive_sha)
            .join("examples.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect()
}

#[test]
fn duplicate_and_mixed_outcomes_are_aggregated() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    let genome = "(pipe X (ridge))";
    fixture_archive(
        &archive,
        &[
            record("d", genome, false, None, Some("timeout"), 30.0),
            record("d", genome, true, Some(1.0), None, 2.0),
            record("d", genome, true, Some(1.0), None, 2.0),
        ],
    );
    read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap();
    let examples = read_examples(&archive);
    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0]["valid_count"], 2);
    assert_eq!(examples[0]["invalid_count"], 1);
    assert_eq!(examples[0]["success_rate"], json!(2.0 / 3.0));
    assert_eq!(examples[0]["failure_classes"]["timeout"], 1);
    assert_eq!(examples[0]["rmse"], 1.0);
}

#[test]
fn tied_percentiles_use_unique_valid_genomes() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[
            record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0),
            record(
                "d",
                "(pipe X (ridge alpha=1.0))",
                true,
                Some(1.0),
                None,
                1.0,
            ),
            record("d", "(pipe X (ridge_gcv))", true, Some(2.0), None, 1.0),
        ],
    );
    read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap();
    let examples = read_examples(&archive);
    let percentiles = examples
        .iter()
        .map(|row| row["percentile"].as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(percentiles, vec![1.0, 1.0, 1.0 / 3.0]);
}

#[test]
fn conflicting_valid_rmse_is_fatal() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    let genome = "(pipe X (ridge))";
    fixture_archive(
        &archive,
        &[
            record("d", genome, true, Some(1.0), None, 1.0),
            record("d", genome, true, Some(2.0), None, 1.0),
        ],
    );
    let error = read_and_materialize(&archive, &temp.path().join("out"), || {})
        .unwrap_err()
        .to_string();
    assert!(error.contains("conflicting valid RMSE"));
}

#[test]
fn malformed_records_and_sha_mismatch_are_fatal_or_retryable() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
    );
    fs::write(archive.join("all-evaluations.jsonl"), b"{bad\n").unwrap();
    let error = read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap_err();
    assert!(error.downcast_ref::<ArchiveChanged>().is_some());
}

#[test]
fn truncated_ledger_after_open_requests_retry_without_publication() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
    );
    let output = temp.path().join("out");
    let error = read_and_materialize(&archive, &output, || {
        fs::write(archive.join("all-evaluations.jsonl"), b"").unwrap();
    })
    .unwrap_err();
    assert!(error.downcast_ref::<ArchiveChanged>().is_some());
    assert!(!output.exists());
}

#[test]
fn same_size_ledger_rewrite_requests_retry_without_publication() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
    );
    let path = archive.join("all-evaluations.jsonl");
    let before = fs::read_to_string(&path).unwrap();
    let changed = before.replace("\"rmse\":1.0", "\"rmse\":9.0");
    assert_ne!(changed, before);
    assert_eq!(changed.len(), before.len());
    let output = temp.path().join("out");
    let error = read_and_materialize(&archive, &output, || {
        fs::write(&path, changed).unwrap();
    })
    .unwrap_err();
    assert!(error.downcast_ref::<ArchiveChanged>().is_some());
    assert!(!output.exists());
}

#[test]
fn buffered_parser_input_stays_bound_to_its_digest_after_file_rewrite() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
    );
    let path = archive.join("all-evaluations.jsonl");
    let before = fs::read_to_string(&path).unwrap();
    let changed = before.replace("\"rmse\":1.0", "\"rmse\":9.0");
    assert_ne!(changed, before);
    let mut ledger = BufReader::new(HashingReader::new(File::open(&path).unwrap()));
    assert_eq!(ledger.fill_buf().unwrap(), before.as_bytes());
    fs::write(&path, changed.as_bytes()).unwrap();
    let (aggregates, _, records) = aggregate_records(&mut ledger).unwrap();
    io::copy(&mut ledger, &mut io::sink()).unwrap();
    assert_eq!(
        ledger.into_inner().into_sha256(),
        sha256_bytes(before.as_bytes())
    );
    assert_eq!(records, 1);
    assert_eq!(aggregates.values().next().unwrap().valid_rmse, Some(1.0));
    assert_eq!(fs::read_to_string(path).unwrap(), changed);
}

#[test]
fn early_parse_error_checks_the_full_ledger_digest_before_deciding_retry() {
    // The unread tail exceeds BufReader's capacity, so only draining the
    // stream can distinguish a matching pin from changed malformed input.
    for pin_matches in [true, false] {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        fixture_archive(
            &archive,
            &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
        );
        let mut bytes = b"{bad\n".to_vec();
        bytes.extend(vec![b' '; 16 * 1024]);
        bytes.push(b'\n');
        fs::write(archive.join("all-evaluations.jsonl"), &bytes).unwrap();
        if pin_matches {
            let path = archive.join(ARCHIVE_MANIFEST);
            let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            manifest["all_evaluations_jsonl_sha256"] = json!(sha256_bytes(&bytes));
            fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        }
        let output = temp.path().join("out");
        let error = read_and_materialize(&archive, &output, || {}).unwrap_err();
        assert_eq!(
            error.downcast_ref::<ArchiveChanged>().is_none(),
            pin_matches
        );
        if pin_matches {
            assert!(error
                .to_string()
                .contains("malformed archive record at line 1"));
        }
        assert!(!output.exists());
    }
}

#[test]
fn manifest_change_during_mapped_read_requests_retry() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    fixture_archive(
        &archive,
        &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
    );
    let path = archive.join(ARCHIVE_MANIFEST);
    let error = read_and_materialize(&archive, &temp.path().join("out"), || {
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["updated_at_unix_ms"] = json!(1);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    })
    .unwrap_err();
    assert!(error.downcast_ref::<ArchiveChanged>().is_some());
}

#[test]
fn repeated_builds_are_byte_identical() {
    let first = TempDir::new().unwrap();
    let second = TempDir::new().unwrap();
    let rows = [
        record(
            "d",
            "(pipe X (nystrom gamma=0.6688638669418179 k=16) (ridge))",
            true,
            Some(1.0),
            None,
            1.25,
        ),
        record(
            "d",
            "(pipe X (derive expr=\"col(0) + 1.25\") (ridge))",
            false,
            None,
            Some("timeout"),
            30.0,
        ),
    ];
    fixture_archive(&first.path().join("archive"), &rows);
    fixture_archive(&second.path().join("archive"), &rows);
    let left = read_and_materialize(
        &first.path().join("archive"),
        &first.path().join("out"),
        || {},
    )
    .unwrap();
    let right = read_and_materialize(
        &second.path().join("archive"),
        &second.path().join("out"),
        || {},
    )
    .unwrap();
    for file in ["examples.jsonl", "gene-schema.v1.json", "manifest.v1.json"] {
        assert_eq!(
            fs::read(left.join(file)).unwrap(),
            fs::read(right.join(file)).unwrap()
        );
    }
}

#[test]
fn cohort_filter_creates_a_distinct_corpus() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("archive");
    let mut golden = record("g", "(pipe X (ridge))", true, Some(1.0), None, 1.0);
    golden["cohort"] = json!("golden42");
    let mut holdout = record("h", "(pipe X (ridge_gcv))", true, Some(2.0), None, 1.0);
    holdout["cohort"] = json!("holdout13");
    fixture_archive(&archive, &[golden, holdout]);

    let root =
        read_and_materialize_filtered(&archive, &temp.path().join("out"), Some("golden42"), || {})
            .unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.v1.json")).unwrap()).unwrap();
    assert_eq!(manifest["cohort_filter"], "golden42");
    assert_eq!(manifest["datasets"].as_object().unwrap().len(), 1);
    assert_eq!(manifest["datasets"]["golden42:g"], 1);
    assert_eq!(
        fs::read_to_string(root.join("examples.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}
