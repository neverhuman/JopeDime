use super::*;

#[test]
fn catalog_metric_map_agrees_with_task_authority() {
    use jope_dime_regressor::task::TaskKind;
    for kind in [
        TaskKind::Regression,
        TaskKind::Binary,
        TaskKind::Timeseries,
        TaskKind::Multiclass,
    ] {
        assert_eq!(
            expected_catalog_metric(kind.as_str()).unwrap(),
            kind.primary_metric()
        );
    }
    assert!(expected_catalog_metric("multi_label").is_err());
}

#[test]
fn promotion_ends_at_pointer_publication_and_never_builds_supply() {
    assert_eq!(PHASES.last(), Some(&"advance_accepted_pointer"));
    assert!(!PHASES.contains(&"seed_next_strict_wave"));
    assert!(!PYTHON_PHASES.contains(&"seed_next_strict_wave"));
}
use tempfile::TempDir;

#[test]
fn python_float_repr_parity() {
    for (value, expected) in [
        (2.5, "2.5"),
        (0.1, "0.1"),
        (1e-7, "1e-07"),
        (-0.0, "-0.0"),
        (12345678901234567890123.0, "1.2345678901234568e+22"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (1.0, "1.0"),
        (-12.75, "-12.75"),
    ] {
        assert_eq!(py_float_repr(value).unwrap(), expected, "value {value}");
    }
}

#[test]
fn python_dumps_parity() {
    let fixture = json!({
        "b": [1, 2.5, "x\"y"],
        "a": {"z": null, "k": true},
        "c": 1e-7,
        "d": 0.1,
        "e": -0.0,
        "f": 12345678901234567890123.0,
        "empty": {},
        "el": [],
    });
    assert_eq!(
            py_dumps(&fixture, PyStyle::Compact).unwrap(),
            "{\"a\":{\"k\":true,\"z\":null},\"b\":[1,2.5,\"x\\\"y\"],\"c\":1e-07,\"d\":0.1,\"e\":-0.0,\"el\":[],\"empty\":{},\"f\":1.2345678901234568e+22}"
        );
    assert_eq!(
            py_dumps(&fixture, PyStyle::Default).unwrap(),
            "{\"a\": {\"k\": true, \"z\": null}, \"b\": [1, 2.5, \"x\\\"y\"], \"c\": 1e-07, \"d\": 0.1, \"e\": -0.0, \"el\": [], \"empty\": {}, \"f\": 1.2345678901234568e+22}"
        );
    let pretty = py_dumps(&fixture, PyStyle::Pretty).unwrap();
    assert!(pretty.starts_with("{\n  \"a\": {\n    \"k\": true,\n    \"z\": null\n  },"));
    assert!(pretty.contains("\n  \"el\": [],\n  \"empty\": {},\n"));
}

#[test]
fn action_key_parity_with_python() {
    let key = action_key(
        "fetch_candidate_checkpoints",
        1_805_437,
        "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
        "tournament",
        20_260_726,
        &"c".repeat(64),
        &"e".repeat(64),
    )
    .unwrap();
    assert_eq!(
        key,
        "93514d51284f8ba404d21e4f4fc7e34729287b7060024140256bf4657c4ca1f6"
    );
}

#[test]
fn pruned_outputs_prefers_newest_attempt_with_result() {
    let temp = TempDir::new().unwrap();
    let runs = temp.path();
    let job = runs.join("jte-test/attempts");
    fs::create_dir_all(job.join("0/outputs/a")).unwrap();
    fs::create_dir_all(job.join("1/outputs/b")).unwrap();
    fs::write(job.join("0/outputs/a/train-result.json"), "{}").unwrap();
    // Only attempt 0 has a result; attempt 1 exists but is empty.
    let attempts = job.clone();
    let mut outputs: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&attempts).unwrap().flatten() {
        let candidate = entry.path().join("outputs");
        if candidate.is_dir() {
            let modified = fs::metadata(&candidate).and_then(|m| m.modified()).unwrap();
            outputs.push((modified, candidate));
        }
    }
    outputs.sort();
    let mut selected = None;
    for (_, candidate) in outputs.into_iter().rev() {
        let mut results = Vec::new();
        rglob(&candidate, "train-result.json", &mut results);
        if !results.is_empty() {
            selected = Some(candidate);
            break;
        }
    }
    assert_eq!(selected.unwrap(), job.join("0/outputs"));
}
