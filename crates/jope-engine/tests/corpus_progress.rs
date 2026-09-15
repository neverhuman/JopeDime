use jope_parallel::{corpus, manifest::*};
use serde_json::{json, Value};
use std::{fs, path::Path};

#[test]
fn observed_packing_preserves_all_artifact_bytes_and_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let vocabulary = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let mut contexts = Vec::new();
    let mut observations = Vec::new();
    for family in ["family-a", "family-b"] {
        let prepared = root.join(family);
        write_json(&prepared.join("prepared.json"), &json!({
            "schema":"parallel.prepared.v1","dataset":family,"family":family,"task":"regression",
            "meta_partition":"meta_train","seed":17,"clean_evaluation":false,"clean_status":"fixture",
            "original_train_sha256":"original","original_test_sha256":"sealed",
            "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
            "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"train",
                "validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
        let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
        let path = prepared.join("context.bin");
        fs::write(&path, b"frozen context fixture").unwrap();
        write_json(&path.with_extension("json"),&json!({"schema":"parallel.encoding.v1","dataset":family,
            "family":family,"partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,
            "training_sha256":"train","selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing",
            "encoder_spec_sha256":"assembly","numerical_backend":"fixture","dataset_dimensions":1536,
            "target_dimensions":512,"output_sha256":file_hash(&path).unwrap()})).unwrap();
        contexts.push(
            json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
            "path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap(),
            "prepared":family,"prepared_sha256":prepared_sha}),
        );
        let observed = root.join(format!("observations-{family}"));
        let rows = (0..8)
            .map(|n| {
                let transform = [
                    "standardize",
                    "robust_scale",
                    "norm_row",
                    "row_stats",
                    "rank",
                    "blom",
                    "sigmoid",
                    "step",
                ][n];
                let genome = format!("(pipe X ({transform}) (ridge))");
                json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                "training_sha256":"train","validation_sha256":"validation","genome":genome,
                "program_sha256":hash(genome.as_bytes()),"metrics":{"rmse":n as f64+1.},
                "runtime_seconds":0.1,"evaluation_bias":"fixture","failure":null})
            })
            .collect::<Vec<_>>();
        write_json(&observed.join("observations.json"), &rows).unwrap();
        write_json(&observed.join("search.json"), &json!({})).unwrap();
        observations.push(observed);
    }
    let contexts_path = root.join("contexts.json");
    write_json(
        &contexts_path,
        &json!({"schema":"parallel.context_index.v1","qualification":true,
        "assembly_sha256":"assembly","encodings":contexts}),
    )
    .unwrap();
    corpus::build(
        root,
        &contexts_path,
        &vocabulary,
        &observations,
        &root.join("plain"),
        true,
    )
    .unwrap();
    let mut updates = Vec::new();
    corpus::build_observed(
        root,
        &contexts_path,
        &vocabulary,
        &observations,
        &root.join("observed"),
        true,
        |operation, done, total| {
            updates.push((operation.to_string(), done, total));
            Ok(())
        },
    )
    .unwrap();
    for name in [
        "records.bin",
        "labels.jsonl",
        "index.json",
        "provenance.json",
        "vocabulary.json",
    ] {
        assert_eq!(
            fs::read(root.join("plain").join(name)).unwrap(),
            fs::read(root.join("observed").join(name)).unwrap(),
            "changed {name}"
        );
    }
    let index: Value = read_json(&root.join("plain/index.json")).unwrap();
    assert_eq!(index["records"], 16);
    assert!(updates
        .iter()
        .any(|(op, n, _)| op == "observations scanned" && *n == 16));
    assert!(updates
        .iter()
        .any(|(op, n, _)| op.starts_with("records packed") && *n == 16));
    let canceled = root.join("canceled");
    assert!(corpus::build_observed(
        root,
        &contexts_path,
        &vocabulary,
        &observations,
        &canceled,
        true,
        |_, _, _| anyhow::bail!("injected cancellation")
    )
    .is_err());
    assert!(!canceled.exists());
}
