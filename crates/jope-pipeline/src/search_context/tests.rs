use super::*;
/// W2-T02: the controller's duplicated lineage constants are the accepted semantic
/// manifest's values (the release decision).
/// Replace `target` (new inode, via rename) with `bytes` the first time a
/// receiver reports it verified.
fn replace_on_verification(
    target: &std::path::Path,
    bytes: Vec<u8>,
) -> crate::execution::verification_hook::Guard {
    let target = std::fs::canonicalize(target).unwrap();
    let mut done = false;
    crate::execution::verification_hook::install(move |path| {
        if !done && std::fs::canonicalize(path).ok().as_deref() == Some(target.as_path()) {
            done = true;
            let staged = target.with_extension("replacement");
            std::fs::write(&staged, &bytes).unwrap();
            std::fs::rename(&staged, &target).unwrap();
        }
    })
}

/// W2-T04 G-10e: the native context receipt and vector consumed are the
/// bytes their pins verified; a file replaced after its check is never
/// consumed.
#[test]
fn a_native_context_replaced_after_its_check_is_never_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let vector: Vec<u8> = (0..2048)
        .flat_map(|i| ((i % 7) as f32 * 0.1).to_le_bytes())
        .collect();
    let mut meta = native_meta();
    meta["output_sha256"] = serde_json::json!(sha256_hex(&vector));
    let meta_bytes = serde_json::to_vec(&meta).unwrap();
    let pin = |name: &str, bytes: &[u8]| {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        Artifact {
            path,
            sha256: sha256_hex(bytes),
        }
    };
    let files = NativeContextFiles {
        vector: pin("context.bin", &vector),
        receipt: pin("receipt.json", &meta_bytes),
        frozen_transforms: Artifact {
            path: dir.path().join("absent-transforms"),
            sha256: TRANSFORMS_SHA256.into(),
        },
        training: training(),
    };
    // Positive controls.
    assert_eq!(native_vector(&files).unwrap(), vector);
    assert_eq!(native_receipt(&files).unwrap(), meta);
    // A context vector replaced after its check (another finite, clipped context).
    let other: Vec<u8> = (0..2048).flat_map(|_| 0.5f32.to_le_bytes()).collect();
    {
        let _hook = replace_on_verification(&files.vector.path, other);
        assert_eq!(
            native_vector(&files).unwrap(),
            vector,
            "a context vector other than the pinned bytes was consumed"
        );
    }
    std::fs::write(&files.vector.path, &vector).unwrap();
    // A receipt replaced after its check (still a valid receipt).
    let mut replaced = meta.clone();
    replaced["note"] = serde_json::json!("replaced after its check");
    {
        let _hook =
            replace_on_verification(&files.receipt.path, serde_json::to_vec(&replaced).unwrap());
        assert_eq!(
            native_receipt(&files).unwrap(),
            meta,
            "a context receipt other than the pinned bytes was consumed"
        );
    }
    // Negative control: bytes other than the pin are refused outright.
    std::fs::write(&files.vector.path, [0u8; 8192]).unwrap();
    assert!(native_vector(&files).is_err());
}

#[test]
fn duplicated_lineage_constants_are_the_accepted_semantic_manifests() {
    let release = jope_policy::semantic::accepted().unwrap();
    assert_eq!(ASSEMBLY_SHA256, release.encoders.assembly_sha256);
    assert_eq!(TRANSFORMS_SHA256, release.encoders.frozen_transforms_sha256);
}
fn row_ids() -> Vec<String> {
    [3usize, 5, 8, 12]
        .iter()
        .map(|n| sha256_hex(n.to_string().as_bytes()))
        .collect()
}
fn training() -> TrainingIdentity {
    TrainingIdentity {
        prepared_sha256: sha256_hex(b"synthetic prepared"),
        training_sha256: sha256_hex(b"synthetic train"),
        selected_training_rows_sha256: sha256_hex(&serde_json::to_vec(&row_ids()).unwrap()),
        preprocessing_sha256: sha256_hex(b"synthetic preprocessing"),
        cut: 2,
    }
}
fn native_meta() -> serde_json::Value {
    let t = training();
    serde_json::json!({"schema":"parallel.encoding.v1", "dataset_dimensions":1536,"target_dimensions":512,"prepared_sha256":t.prepared_sha256,"training_sha256":t.training_sha256,"selected_training_rows_sha256":t.selected_training_rows_sha256,"preprocessing_sha256":t.preprocessing_sha256,"cut":t.cut,"encoder_spec_sha256":ASSEMBLY_SHA256,"numerical_backend":NATIVE_BACKEND,"output_sha256":sha256_hex(b"synthetic context"),"partition":"meta_train","final_test_targets_used":false})
}
#[test]
fn accepted_receipt_refuses_wrong_rows_old128_and_classification_aliases() -> Result<()> {
    let expected = sha256_hex(b"synthetic context");
    validate_native_receipt(&native_meta(), &training(), &expected)?;
    for (key, value) in [
        (
            "selected_training_rows_sha256",
            serde_json::json!(sha256_hex(b"heldout rows")),
        ),
        ("dataset_dimensions", serde_json::json!(128)),
        ("task", serde_json::json!("binary")),
        ("partition", serde_json::json!("meta_test")),
        ("numerical_backend", serde_json::json!("CPU replacement")),
        ("qualification", serde_json::json!(true)),
        ("final_test_targets_used", serde_json::Value::Null),
    ] {
        let mut m = native_meta();
        m[key] = value;
        assert!(
            validate_native_receipt(&m, &training(), &expected).is_err(),
            "accepted altered {key}"
        );
    }
    Ok(())
}
#[test]
fn experimental_binary_and_multiclass_contexts_are_bitwise_label_rename_invariant() -> Result<()> {
    let rows = (0..4)
        .map(|i| (0..128).map(|j| (i * 128 + j) as f32 / 128.).collect())
        .collect::<Vec<Vec<f32>>>();
    for (labels, renamed) in [
        (["a", "b", "a", "b"], ["z", "-100", "z", "-100"]),
        (["a", "b", "c", "b"], ["third", "first", "second", "first"]),
    ] {
        let before = classification_context(
            training(),
            &row_ids(),
            &[0.; 512],
            &[1.; 1024],
            &rows,
            &labels.map(String::from),
        )?;
        let after = classification_context(
            training(),
            &row_ids(),
            &[0.; 512],
            &[1.; 1024],
            &rows,
            &renamed.map(String::from),
        )?;
        assert_eq!(before.values().len(), 2048);
        assert_eq!(before.sha256(), after.sha256());
        assert_eq!(
            before
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(!before.policy_compatible);
        assert_eq!(before.lineage, CLASSIFICATION_LINEAGE);
    }
    Ok(())
}
#[test]
fn experimental_pool_rejects_changed_training_selection_and_missing_classes() {
    let labels = ["a", "b", "a", "b"].map(String::from);
    let rows = vec![vec![1.; 128]; 4];
    assert!(classification_context(
        training(),
        &[
            row_ids()[0].clone(),
            row_ids()[1].clone(),
            row_ids()[2].clone(),
            sha256_hex(b"other row")
        ],
        &[0.; 512],
        &[1.; 1024],
        &rows,
        &labels
    )
    .is_err());
    assert!(classification_context(
        training(),
        &row_ids(),
        &[0.; 512],
        &[1.; 1024],
        &rows,
        &["a", "a", "a", "a"].map(String::from)
    )
    .is_err());
    assert!(classification_context(
        training(),
        &row_ids(),
        &[0.; 128],
        &[1.; 1024],
        &rows,
        &labels
    )
    .is_err());
}
