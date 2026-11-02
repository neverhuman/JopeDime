use super::*;

pub(super) fn contract() -> NumericalContract {
    NumericalContract {
        protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
        precision: "f64".into(),
        omp_threads: 1,
        rayon_threads: 1,
        blas_threads: 1,
        mkl_threads: 1,
    }
}

pub(super) fn training_hash(rows: &[Vec<f64>]) -> String {
    let x = rows
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect::<Vec<_>>();
    let y = rows
        .iter()
        .map(|row| *row.last().unwrap())
        .collect::<Vec<_>>();
    ordered_training_sha256(&x, &y).unwrap()
}

pub(super) fn cut_training_hash(prepared: &Path) -> String {
    training_hash(&load_view(prepared, 0, true).unwrap())
}

pub(super) fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "fixture CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// The pointer-view preparation of `tests/view_contracts.rs`.
pub(super) fn view_fixture() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let prepared = view_fixture_in(t.path(), "example", &[8, 9]);
    (t, prepared)
}

/// A pointer-view preparation under `root`: cut 0 trains on raw rows 0..8 and
/// validates on `validation`. Every variant shares the training rows and their
/// preprocessing, so only the scored membership differs between variants.
pub(super) fn view_fixture_in(root: &Path, name: &str, validation: &[usize]) -> PathBuf {
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join("data/originals/raw.csv"), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared").join(name);
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let scored: Vec<Vec<String>> = validation.iter().map(|&i| raw[i].clone()).collect();
    let va = csv_bytes(&pp.transform(&scored).unwrap()).unwrap();
    let mapped = validation.iter().max().map_or(10, |&i| (i + 1).max(10));
    let p = json!({"schema":"parallel.prepared.v1","dataset":name,"family":"example","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),"selected_train_indices":(0..mapped).collect::<Vec<_>>(),"selected_test_indices":[0,1],"target_last":true,"rows_total":12,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":validation,"training_sha256":hash(&tr),"validation_sha256":hash(&va),"preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],"storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&root.join("data/originals/raw.csv")).unwrap(),"feature_columns":["x"],"target_column":"target","original_training_row_map":(0..mapped).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

/// The 40-row raw training file of the fitted predictor contract.
pub(super) fn raw_training(root: &Path) -> (PathBuf, String) {
    let train = root.join("train.csv");
    let rows = (0..40)
        .map(|i| vec![i as f64, (i * i) as f64, 100.0 + i as f64 * 2.0])
        .collect::<Vec<_>>();
    fs::write(&train, csv_bytes(&rows).unwrap()).unwrap();
    let raw = Table::read(&train, &Value::Null).unwrap();
    let preprocessing = Preprocessing::fit(&raw.rows, "regression").unwrap();
    let training = training_hash(&preprocessing.transform(&raw.rows).unwrap());
    (train, training)
}

/// A refused public fit publishes nothing and claims nothing.
pub(super) fn assert_fit_refused(
    authority: &Fixture,
    mut client: Command,
    train: &Path,
    message: &str,
) {
    let output = authority.root.join("refused-model.json");
    let result = client
        .args(["fit", "--train"])
        .arg(train)
        .args(["--genome", PROGRAM, "--output"])
        .arg(&output)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "refused fit succeeded");
    assert!(error.contains(message), "expected {message:?}: {error}");
    assert!(!output.exists(), "refused fit published a predictor");
    authority.assert_no_fit_starts();
}
