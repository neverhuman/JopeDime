use super::{
    json::parse_unique_json, read_records, record_inventory, summary_counts, validate_record,
    verify_inventory, ReaderOptions,
};
use serde_json::{json, Value};
use sha2::Digest;
use std::{fs, io::Cursor};
use tempfile::tempdir;

fn valid_round() -> Value {
    json!({
        "schema": "jope.research.round/v1",
        "identity": {
            "run_id": "reader-test",
            "task": "regression",
            "evidence": "engine",
            "partition": "train",
            "source_commit": "a".repeat(40),
            "corpus_manifest_sha256": "b".repeat(64),
            "dataset_sha256": "c".repeat(64),
            "family_sha256": "d".repeat(64),
            "problem_sha256": "e".repeat(64),
            "bundle_sha256": "f".repeat(64),
            "conditioning_sha256": "1".repeat(64),
            "evaluator_sha256": "2".repeat(64),
            "dope_weights_sha256": super::DOPE_SHA256,
        },
        "round": 0,
        "seed": 7,
        "tokens": vec![vec![0.0; 32]; 10],
        "mask": vec![false; 10],
        "revealed": [],
        "proposals": [{
            "program_sha256": "3".repeat(64),
            "operator_family": "linear",
            "feat412": vec![0.0; 412],
            "proj16": vec![0.0; 16],
        }],
        "source_indices": [0],
        "policy": {"kind": "heuristic", "explore": false, "slate_costs": null, "admitted": [0]},
        "evaluate_all_proposals": false,
        "dispatched": [0],
        "capture_complete": true,
        "outcomes": [{
            "proposal_index": 0,
            "wall_ms": 12,
            "measurement": {"status": "success", "surrogate": 0.25, "objective": 0.5, "gate": true},
        }],
    })
}

#[test]
fn valid_round_inventory_counts_and_hashes_source_bytes() {
    let root = tempdir().unwrap();
    let path = root.path().join("records.jsonl");
    let mut bytes = serde_json::to_vec(&valid_round()).unwrap();
    bytes.push(b'\n');
    fs::write(&path, &bytes).unwrap();

    let records = read_records(&path, ReaderOptions::default())
        .unwrap()
        .collect::<anyhow::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(records.len(), 1);
    let inventory = record_inventory(&path, ReaderOptions::default()).unwrap();
    assert_eq!(inventory["counts"]["rounds"], 1);
    assert_eq!(inventory["counts"]["proposals"], 1);
    assert_eq!(inventory["counts"]["admissions"], 1);
    assert_eq!(inventory["counts"]["dispatches"], 1);
    assert_eq!(inventory["counts"]["outcomes"], 1);
    assert_eq!(inventory["counts"]["successes"], 1);
    assert_eq!(inventory["counts"]["failures"], 0);
    assert_eq!(
        inventory["records_sha256"],
        format!("{:x}", sha2::Sha256::digest(&bytes))
    );
    assert_eq!(
        summary_counts(&inventory, false).unwrap(),
        json!({"rounds":1,"proposals":1,"outcomes":1,"failures":0,"incomplete_rounds":0})
    );
    assert_eq!(
        summary_counts(&inventory, true).unwrap(),
        inventory["counts"]
    );
    verify_inventory(&inventory, &inventory).unwrap();
    let mut changed = inventory.clone();
    changed["counts"]["outcomes"] = json!(0);
    assert!(verify_inventory(&inventory, &changed).is_err());
}

#[test]
fn synthetic_and_incomplete_records_require_explicit_inspection_opt_ins() {
    let mut synthetic = valid_round();
    synthetic["identity"]["evidence"] = json!("synthetic_control");
    assert!(validate_record(&synthetic, false, false).is_err());
    validate_record(&synthetic, true, false).unwrap();

    let mut incomplete = valid_round();
    incomplete["dispatched"] = json!([]);
    incomplete["outcomes"] = json!([]);
    incomplete["capture_complete"] = json!(false);
    assert!(validate_record(&incomplete, false, false).is_err());
    validate_record(&incomplete, false, true).unwrap();
}

#[test]
fn duplicate_json_keys_and_partial_lines_are_rejected() {
    assert!(parse_unique_json(br#"{"schema":"first","schema":"second"}"#).is_err());

    let mut reader = read_records_from(Cursor::new(b"{}".to_vec()));
    assert!(reader.next().unwrap().is_err());
    assert!(read_records_from(Cursor::new(Vec::<u8>::new()))
        .next()
        .unwrap()
        .is_err());
}

#[test]
fn evaluate_all_requirement_is_checked_before_yielding_a_record() {
    let root = tempdir().unwrap();
    let path = root.path().join("records.jsonl");
    let mut bytes = serde_json::to_vec(&valid_round()).unwrap();
    bytes.push(b'\n');
    fs::write(&path, bytes).unwrap();
    let options = ReaderOptions {
        require_evaluate_all: true,
        ..ReaderOptions::default()
    };
    let error = read_records(&path, options)
        .unwrap()
        .next()
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), "complete all-proposal capture required");
}

fn read_records_from<R: std::io::BufRead>(reader: R) -> super::RecordReader<R> {
    super::RecordReader::new(reader, ReaderOptions::default())
}
