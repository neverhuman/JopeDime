use super::*;
use crate::canon::canonical_json;
use crate::record::ComparisonContext;
use serde_json::json;

fn digest(s: &str) -> String {
    sha256_hex(s.as_bytes())
}
fn member(row: &str, group: &str) -> Member {
    Member {
        row: row.into(),
        group: group.into(),
    }
}
fn scope() -> Scope {
    Scope {
        comparison: HwmIdentity {
            task: "regression".into(),
            dataset_hash: digest("dataset"),
            cut_id: digest("cut"),
            evaluator_sha256: digest("evaluator"),
            vocabulary_sha256: digest("vocabulary"),
            protocol: "inner-oof".into(),
            comparison: Some(ComparisonContext {
                numerical_policy: "frozen".into(),
                stage: "train_oof".into(),
                aggregation: "mean_fold_loss".into(),
                label_role: "development".into(),
                sealed: false,
                benchmark_valid: false,
                recipe_encoding: "native".into(),
                class_order: vec![],
                checkpoint_sha256: None,
            }),
        },
        source_family: "fixture".into(),
        outer_fold: Some(0),
        inner_fold: Some(1),
        fitting: vec![member("b", "fit-b"), member("a", "fit-a")],
        forbidden: vec![member("held", "held-group")],
    }
}
fn encoder() -> Encoder {
    Encoder {
        artifact_sha256: digest("encoder"),
        configuration_sha256: digest("config"),
        seed: 42,
        product: Product::DatasetSignature512,
    }
}
fn inputs() -> Inputs {
    Inputs {
        features_sha256: digest("features"),
        labels_sha256: Some(digest("labels")),
        parent_receipts: vec![digest("scaler-receipt")],
    }
}
fn capability() -> EncoderCapability {
    EncoderCapability::bind(scope(), encoder(), inputs()).unwrap()
}
fn output() -> Vec<u8> {
    vec![0; 512 * 4]
}
fn binding(r: &EncoderReceipt) -> TeachingBinding {
    TeachingBinding {
        comparison_sha256: r.comparison_sha256.clone(),
        program_sha256: digest("program"),
        encoder_receipt_sha256: canonical_sha256(r).unwrap(),
        product: r.encoder.product,
    }
}
fn check(c: &EncoderCapability, r: &EncoderReceipt, b: &TeachingBinding) -> Result<()> {
    check_teaching_binding(c, Some(r), b, &digest("program"), b"state", &output())
}

#[test]
fn complete_binding_roundtrips_and_missing_provenance_fails() {
    let c = capability();
    let r = c.receipt(b"state", &output()).unwrap();
    let b = binding(&r);
    let decoded: EncoderReceipt = serde_json::from_slice(&canonical_json(&r).unwrap()).unwrap();
    check(&c, &decoded, &b).unwrap();
    assert!(
        check_teaching_binding(&c, None, &b, &digest("program"), b"state", &output())
            .unwrap_err()
            .to_string()
            .contains("MissingEncoderProvenance")
    );
    let mut unknown = serde_json::to_value(&r).unwrap();
    unknown["pure"] = json!(true);
    assert!(serde_json::from_value::<EncoderReceipt>(unknown).is_err());
    let mut missing = serde_json::to_value(&r).unwrap();
    missing.as_object_mut().unwrap().remove("scope_sha256");
    assert!(serde_json::from_value::<EncoderReceipt>(missing).is_err());
}

#[test]
fn forged_receipt_cannot_rebind_capability_even_when_its_digest_is_recomputed() {
    let c = capability();
    let r = c.receipt(b"state", &output()).unwrap();
    let mutations = [
        ("/schema", json!("jope.encoder_receipt/v2")),
        ("/capability_sha256", json!(digest("other cap"))),
        ("/scope_sha256", json!(digest("other scope"))),
        ("/comparison_sha256", json!(digest("other comparison"))),
        ("/ordered_fitting_rows_sha256", json!(digest("reordered"))),
        ("/permitted_label_rows_sha256", json!(digest("all labels"))),
        ("/encoder/artifact_sha256", json!(digest("other encoder"))),
        (
            "/encoder/configuration_sha256",
            json!(digest("other config")),
        ),
        ("/encoder/seed", json!(43)),
        ("/encoder/product", json!("target_embedding512")),
        ("/inputs/features_sha256", json!(digest("global features"))),
        ("/inputs/labels_sha256", json!(digest("global labels"))),
        ("/inputs/parent_receipts", json!([])),
        ("/fitted_state_sha256", json!(digest("global scaler"))),
        ("/output_sha256", json!(digest("swapped output"))),
    ];
    for (path, value) in mutations {
        let mut wire = serde_json::to_value(&r).unwrap();
        *wire.pointer_mut(path).unwrap() = value;
        let forged: EncoderReceipt = serde_json::from_value(wire).unwrap();
        assert!(
            check(&c, &forged, &binding(&forged)).is_err(),
            "accepted {path}"
        );
    }
}

#[test]
fn different_dataset_fold_task_vocabulary_or_membership_cannot_reuse_receipt() {
    let c = capability();
    let r = c.receipt(b"state", &output()).unwrap();
    let b = binding(&r);
    let variants: Vec<fn(&mut Scope)> = vec![
        |s| s.comparison.dataset_hash = digest("other dataset"),
        |s| s.comparison.task = "binary".into(),
        |s| s.comparison.cut_id = digest("other cut"),
        |s| s.comparison.vocabulary_sha256 = digest("other vocabulary"),
        |s| s.outer_fold = Some(1),
        |s| {
            s.outer_fold = None;
            s.inner_fold = None;
        },
        |s| s.inner_fold = Some(2),
        |s| s.fitting.reverse(),
        |s| s.fitting[0].row = "new-fitting-row".into(),
        |s| s.source_family = "test-adaptive".into(),
    ];
    for mutate in variants {
        let mut s = scope();
        mutate(&mut s);
        let other = EncoderCapability::bind(s, encoder(), inputs()).unwrap();
        assert!(check(&other, &r, &b).is_err());
    }
}

#[test]
fn membership_rejects_overlap_duplicate_group_leak_and_missing_fitting_payload() {
    let variants: Vec<fn(&mut Scope)> = vec![
        |s| s.fitting.push(s.fitting[0].clone()),
        |s| s.fitting.push(s.forbidden[0].clone()),
        |s| s.forbidden[0].group = s.fitting[0].group.clone(),
        |s| s.forbidden.push(s.forbidden[0].clone()),
        |s| s.fitting.clear(),
        |s| s.fitting[0].group.clear(),
        |s| s.outer_fold = Some(5),
        |s| s.outer_fold = None,
        |s| s.inner_fold = Some(5),
        |s| s.comparison.evaluator_sha256 = "unbound".into(),
        |s| s.comparison.comparison = None,
        |s| s.comparison.comparison.as_mut().unwrap().sealed = true,
        |s| s.comparison.comparison.as_mut().unwrap().benchmark_valid = true,
        |s| s.comparison.comparison.as_mut().unwrap().label_role = "meta_test".into(),
    ];
    for mutate in variants {
        let mut s = scope();
        mutate(&mut s);
        assert!(EncoderCapability::bind(s, encoder(), inputs()).is_err());
    }
    assert!(capability()
        .project_fitting(&BTreeMap::<String, f32>::new())
        .is_err());
    let mut i = inputs();
    i.parent_receipts.push(i.parent_receipts[0].clone());
    assert!(EncoderCapability::bind(scope(), encoder(), i).is_err());
}

#[test]
fn output_state_program_and_pinned_receipt_substitutions_fail() {
    let c = capability();
    let r = c.receipt(b"state", &output()).unwrap();
    let b = binding(&r);
    assert!(check_teaching_binding(
        &c,
        Some(&r),
        &b,
        &digest("other program"),
        b"state",
        &output()
    )
    .is_err());
    assert!(check_teaching_binding(
        &c,
        Some(&r),
        &b,
        &digest("program"),
        b"other state",
        &output()
    )
    .is_err());
    let mut swapped = output();
    swapped[..4].copy_from_slice(&1f32.to_le_bytes());
    assert!(
        check_teaching_binding(&c, Some(&r), &b, &digest("program"), b"state", &swapped).is_err()
    );
    let mut wrong_pin = b.clone();
    wrong_pin.encoder_receipt_sha256 = digest("other receipt");
    assert!(check(&c, &r, &wrong_pin).is_err());
    let mut wrong_product = b;
    wrong_product.product = Product::TargetEmbedding512;
    assert!(check(&c, &r, &wrong_product).is_err());
    assert!(c.receipt(b"", &output()).is_err());
    assert!(c.receipt(b"state", &[0; 4]).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = output();
        invalid[..4].copy_from_slice(&value.to_le_bytes());
        assert!(c.receipt(b"state", &invalid).is_err());
    }
}

#[test]
fn reference_projection_is_invariant_to_forbidden_features_and_labels() {
    // Deliberately a tiny in-memory reference encoder, not a production-encoder proof.
    // Two values per row: feature and original label. Only the projected view is encoded.
    fn encode(c: &EncoderCapability, rows: &BTreeMap<String, [f32; 2]>) -> (String, String) {
        let projected = c.project_fitting(rows).unwrap();
        let fitted = projected.iter().map(|r| r[0] + r[1]).sum::<f32>() / projected.len() as f32;
        (
            canonical_sha256(&projected).unwrap(),
            sha256_hex(&fitted.to_le_bytes()),
        )
    }
    let c = capability();
    let mut rows = BTreeMap::from([
        ("a".into(), [1., 2.]),
        ("b".into(), [3., 4.]),
        ("held".into(), [5., 6.]),
    ]);
    let before = encode(&c, &rows);
    rows.insert("held".into(), [f32::NAN, f32::INFINITY]);
    assert_eq!(before, encode(&c, &rows));
    rows.remove("held");
    assert_eq!(before, encode(&c, &rows));
    assert_eq!(*c.project_fitting(&rows).unwrap()[0], [3., 4.]); // frozen order, not map order
    rows.insert("a".into(), [10., 20.]);
    let changed = encode(&c, &rows);
    assert_ne!(before.0, changed.0);
    assert_ne!(before.1, changed.1);
}

#[test]
fn label_free_encoder_binds_empty_label_membership() {
    let mut i = inputs();
    i.labels_sha256 = None;
    let c = EncoderCapability::bind(scope(), encoder(), i).unwrap();
    let r = c.receipt(b"state", &output()).unwrap();
    assert_eq!(
        r.permitted_label_rows_sha256,
        canonical_sha256(&Vec::<String>::new()).unwrap()
    );
    check(&c, &r, &binding(&r)).unwrap();
}
