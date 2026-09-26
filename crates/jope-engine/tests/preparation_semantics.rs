//! W1-T03: explicit preparation semantics at the receiving path.
//!
//! `load_view_in_root` is the receiver used before context construction
//! (`encoding.rs`) and fitting (`gp.rs`, `label_admission.rs`). Each negative
//! fixture is internally consistent: every hash in `prepared.json` is recomputed
//! the way a buggy or hostile producer would, so only semantic checks can refuse
//! it. The positive controls prove valid views still round-trip original labels.

use jope_parallel::{manifest::*, prepare::*};
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Clone)]
struct Spec {
    header: Vec<&'static str>,
    rows: Vec<Vec<String>>,
    features: Vec<&'static str>,
    target: &'static str,
    row_map: Vec<usize>,
    train: Vec<usize>,
    validation: Vec<usize>,
    /// Rows the stored preprocessing was fitted on (defaults to `train`).
    fit: Option<Vec<usize>>,
    /// Replace one categorical column's fitted categories (recoding).
    recode: Option<(usize, Vec<String>)>,
}

fn backing_rows(n: usize) -> Vec<Vec<String>> {
    (0..n)
        .map(|i| {
            vec![
                format!("{}", i as f64 * 1.5),
                ["red", "blue", "green", ""][i % 4].to_string(),
                format!("{}", 100.0 + 3.0 * i as f64 + (i % 2) as f64 * 0.5),
            ]
        })
        .collect()
}

fn spec() -> Spec {
    Spec {
        header: vec!["x", "c", "target"],
        rows: backing_rows(12),
        features: vec!["x", "c"],
        target: "target",
        row_map: (0..10).collect(),
        train: (0..8).collect(),
        validation: vec![8, 9],
        fit: None,
        recode: None,
    }
}

/// Independent of the code under test: first-match header lookup, trimmed cells.
fn raw_rows(s: &Spec, indices: &[usize]) -> Vec<Vec<String>> {
    let position = |name: &str| s.header.iter().position(|h| *h == name).unwrap();
    let columns: Vec<usize> = s
        .features
        .iter()
        .map(|f| position(f))
        .chain([position(s.target)])
        .collect();
    indices
        .iter()
        .map(|&i| {
            let row = &s.rows[s.row_map[i]];
            columns.iter().map(|&j| row[j].trim().to_string()).collect()
        })
        .collect()
}

struct View {
    _temp: tempfile::TempDir,
    root: PathBuf,
    prepared: PathBuf,
}

fn build(s: &Spec) -> View {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let mut csv = s.header.join(",") + "\n";
    for row in &s.rows {
        csv.push_str(&row.join(","));
        csv.push('\n');
    }
    let backing = root.join("data/originals/raw.csv");
    fs::write(&backing, csv).unwrap();
    let fit_rows = raw_rows(s, s.fit.as_ref().unwrap_or(&s.train));
    let mut pp = Preprocessing::fit(&fit_rows, "regression").unwrap();
    if let Some((column, categories)) = &s.recode {
        pp.columns[*column].categories = categories.clone();
    }
    let prepared = root.join("data/prepared/example");
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let training = csv_bytes(&pp.transform(&raw_rows(s, &s.train)).unwrap()).unwrap();
    let validation = csv_bytes(&pp.transform(&raw_rows(s, &s.validation)).unwrap()).unwrap();
    let p = json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example",
        "task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,
        "clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),
        "original_test_sha256":"1".repeat(64),"selected_train_indices":(0..s.row_map.len()).collect::<Vec<_>>(),
        "selected_test_indices":[0,1],"target_last":true,"rows_total":s.row_map.len()+2,
        "cuts":[{"index":0,"train_indices":s.train,"validation_indices":s.validation,
            "training_sha256":hash(&training),"validation_sha256":hash(&validation),
            "preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],
        "storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&backing).unwrap(),
            "feature_columns":s.features,"target_column":s.target,"original_training_row_map":s.row_map}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    View {
        _temp: temp,
        root,
        prepared,
    }
}

fn load(v: &View, training: bool) -> anyhow::Result<Vec<Vec<f64>>> {
    load_view_in_root(&v.root, &v.prepared, 0, training)
}

/// Both partitions must be refused before any row reaches a consumer.
fn refused(s: &Spec, needle: &str) {
    let v = build(s);
    for training in [true, false] {
        match load(&v, training) {
            Ok(rows) => panic!(
                "accepted {} view ({} rows) that must fail with {needle:?}",
                if training { "training" } else { "validation" },
                rows.len()
            ),
            Err(e) => {
                let text = format!("{e:#}");
                assert!(text.contains(needle), "expected {needle:?}, got {text}");
            }
        }
    }
}

#[test]
fn valid_pointer_view_round_trips_original_labels() {
    let s = spec();
    let v = build(&s);
    let pp: Preprocessing = read_json(&v.prepared.join("cut-0/preprocessing.json")).unwrap();
    for (training, indices) in [(true, &s.train), (false, &s.validation)] {
        let rows = load(&v, training).unwrap();
        let original = raw_rows(&s, indices);
        assert_eq!(rows.len(), original.len());
        for (row, raw) in rows.iter().zip(&original) {
            let label: f64 = raw.last().unwrap().parse().unwrap();
            let restored = row.last().unwrap() * pp.target_scale + pp.target_mean;
            assert!((restored - label).abs() <= 1e-9 * label.abs().max(1.0));
        }
    }
    // The raw receiver returns the exact original label strings in cut order.
    let p: Prepared = read_json(&v.prepared.join("prepared.json")).unwrap();
    let raw = load_raw_view_in_root(&v.root, p.storage.as_ref().unwrap(), &s.train).unwrap();
    assert_eq!(raw, raw_rows(&s, &s.train));
}

#[test]
fn heldout_only_category_and_missingness_changes_do_not_alter_training_preparation() {
    let base = spec();
    let mut changed = spec();
    for &i in &changed.validation {
        let r = changed.row_map[i];
        changed.rows[r][1] = "violet".into();
        changed.rows[r][0] = "?".into();
    }
    let a = build(&base);
    let b = build(&changed);
    assert_eq!(
        fs::read(a.prepared.join("cut-0/preprocessing.json")).unwrap(),
        fs::read(b.prepared.join("cut-0/preprocessing.json")).unwrap()
    );
    assert_eq!(load(&a, true).unwrap(), load(&b, true).unwrap());
    // Declared inductive handling: unseen category -> -1, missing -> training median.
    let pp: Preprocessing = read_json(&b.prepared.join("cut-0/preprocessing.json")).unwrap();
    let held = load(&b, false).unwrap();
    assert!(held.iter().all(|r| r[1] == -1.0));
    let imputed = (pp.columns[0].impute - pp.columns[0].mean) / pp.columns[0].scale;
    assert!(held.iter().all(|r| r[0] == imputed));
}

#[test]
fn target_column_declared_as_a_feature_is_rejected() {
    let mut s = spec();
    s.features = vec!["x", "c", "target"];
    refused(&s, "undocumented target-derived feature");
}

#[test]
fn renamed_exact_target_copy_is_rejected() {
    let mut s = spec();
    s.header = vec!["x", "y_copy", "c", "target"];
    for row in &mut s.rows {
        let target = row[2].clone();
        row.insert(1, target);
    }
    s.features = vec!["x", "y_copy", "c"];
    refused(&s, "undocumented target-derived feature");
}

#[test]
fn ambiguous_backing_header_is_rejected() {
    let mut s = spec();
    s.header = vec!["x", "x", "c", "target"];
    for (i, row) in s.rows.iter_mut().enumerate() {
        row.insert(1, format!("{}", 1000 + i));
    }
    refused(&s, "ambiguous");
}

#[test]
fn duplicated_feature_names_are_rejected() {
    let mut s = spec();
    s.features = vec!["x", "x", "c"];
    refused(&s, "duplicated");
}

#[test]
fn duplicate_expanding_training_membership_is_rejected() {
    let mut s = spec();
    s.train = vec![0, 1, 1, 2, 3, 4, 5, 6, 7];
    refused(&s, "duplicated or reordered");
}

#[test]
fn reordered_training_membership_is_rejected() {
    let mut s = spec();
    s.train.reverse();
    refused(&s, "duplicated or reordered");
}

#[test]
fn validation_rows_inside_training_are_rejected() {
    let mut s = spec();
    s.validation = vec![7, 8];
    refused(&s, "overlap");
}

#[test]
fn one_backing_row_on_both_sides_of_a_cut_is_rejected() {
    let mut s = spec();
    s.row_map = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 3];
    refused(&s, "same backing row");
}

#[test]
fn preparation_state_fitted_beyond_training_membership_is_rejected() {
    let mut s = spec();
    s.fit = Some((0..10).collect());
    refused(&s, "not the state learned from its training membership");
}

#[test]
fn recoded_categories_are_rejected() {
    let mut s = spec();
    let fitted = Preprocessing::fit(&raw_rows(&s, &s.train), "regression").unwrap();
    let mut recoded = fitted.columns[1].categories.clone();
    assert!(recoded.len() >= 2);
    recoded.reverse();
    s.recode = Some((1, recoded));
    refused(&s, "not the state learned from its training membership");
}

/// Producer-side refusal leaves no prepared output behind (no authority-bearing
/// state change): the same target-as-feature metadata cannot be prepared.
#[test]
fn producer_refuses_target_feature_before_writing_any_view() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let source = root.join("data/originals/raw.csv");
    let mut csv = "x,target\n".to_string();
    for i in 0..12 {
        csv.push_str(&format!("{},{}\n", i, 100 + i * 3));
    }
    fs::write(&source, csv).unwrap();
    let legacy = root.join("data/originals/regression/example");
    fs::create_dir_all(&legacy).unwrap();
    let scaled = (0..10)
        .map(|i| vec![i as f64 / 11.0, i as f64 / 11.0, i as f64 / 11.0])
        .collect::<Vec<_>>();
    fs::write(legacy.join("train.csv"), csv_bytes(&scaled).unwrap()).unwrap();
    let metadata = json!({"source_path":source,"target_generation":"original_numeric_column",
        "feature_columns":["x","target"],"target_column":"target",
        "feature_encodings":{"x":{"kind":"numeric","impute_value":0.0},"target":{"kind":"numeric","impute_value":0.0}},
        "feature_scaling":{"min":[0.0,100.0],"max":[11.0,133.0]},"target_scaling":{"min":100.0,"max":133.0}});
    write_json(&legacy.join("meta.json"), &metadata).unwrap();
    let test_stats = json!({"rows":2,"columns":3,"target_index":2,"header":false,"missing_features":0,
        "invalid_targets":0,"duplicate_rows":0,"conflicting_features":0,"target_values":[]});
    let audit = json!({"schema":"parallel.audit.v1","seed":SEED,"roots":[],"datasets":[{"id":"example",
        "task":"regression","source_dir":legacy,"source_key":"example","family":"example","parents":[],
        "admission":"approved","reason":"test fixture","evidence":[],
        "metadata_sha256":file_hash(&legacy.join("meta.json")).unwrap(),"train":null,"test":test_stats,
        "split_feature_overlap":1,"issues":[],"meta_partition":"meta_train"}],"artifact_counts":{},
        "missing_roots":[],"complete":false});
    let audit_path = root.join("manifests/audit.json");
    write_json(&audit_path, &audit).unwrap();
    write_json(&root.join("manifests/transfer.json"), &json!({"schema":"parallel.transfer.v1","files":[{"source":legacy.join("test.csv"),"destination":"data/originals/legacy/test.csv","bytes":32,"sha256":"1".repeat(64),"dataset":"example","role":"original_test"}]})).unwrap();
    let backing = root.join("manifests/backing.json");
    write_json(&backing, &json!({"schema":"parallel.transfer.v1","files":[{"source":source,"destination":"data/originals/raw.csv","bytes":fs::metadata(&source).unwrap().len(),"sha256":file_hash(&source).unwrap(),"dataset":"example","role":"backing_source"}]})).unwrap();
    let output = root.join("data/prepared/leaky");
    let result = run(&audit_path, "example", &output, false, true, &backing);
    assert!(
        format!("{:#}", result.as_ref().map(|_| ()).unwrap_err())
            .contains("undocumented target-derived feature"),
        "producer accepted the target as a feature"
    );
    assert!(!output.exists(), "a refused preparation left output behind");
}
