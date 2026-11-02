use super::*;

fn labels(count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| {
            json!({
                "label": {"dataset": format!("d{index}"), "split": "TRAIN"},
                "committed_at_ms": 100 + index as u64,
            })
        })
        .collect()
}

fn families(count: usize) -> BTreeMap<String, String> {
    (0..count)
        .map(|index| (format!("d{index}"), format!("family-{index}")))
        .collect()
}

fn build(count: usize, audit_fraction: Value, legacy: &[Value]) -> Value {
    build_split_manifest(
        &labels(count),
        &families(count),
        &json!(0.5),
        &audit_fraction,
        &json!(1),
        legacy,
    )
    .unwrap()
}

#[test]
fn creates_complete_family_disjoint_manifest_with_python_rounding() {
    let manifest = build(5, json!(0.5), &[]);
    validate_split_manifest(&manifest).unwrap();
    assert_eq!(manifest["schema"], SPLIT_MANIFEST_V1);
    assert_eq!(manifest["status"], "complete");
    let assignments = manifest["assignments"].as_object().unwrap();
    let mut counts = BTreeMap::<String, usize>::new();
    for record in assignments.values() {
        *counts
            .entry(record["split"].as_str().unwrap().to_owned())
            .or_default() += 1;
    }
    assert_eq!(counts[TRAIN_SPLIT], 1);
    assert_eq!(counts[DEVELOPMENT_SPLIT], 2);
    assert_eq!(counts[TEMPORAL_AUDIT_SPLIT], 2);
    assert_eq!(manifest["temporal_audit_cutoff_ms"], 103);
    assert_eq!(build(5, json!(0.5), &[]), manifest);
}

#[test]
fn filters_authority_and_uses_earliest_integer_commit_time() {
    let mut rows = labels(3);
    rows.push(json!({
        "label": {"dataset": "ignored", "split": "temporal-audit"},
        "committed_at_ms": 1,
    }));
    rows.push(json!({"dataset": "d0", "committed_at_ms": 7}));
    assert!(is_training_authority_label(
        &json!({"label": {"split": "Train"}})
    ));
    assert!(!is_training_authority_label(
        &json!({"split": "validation"})
    ));
    assert!(!is_training_authority_label(
        &json!({"split": "temporal-audit"})
    ));
    let mut manifest = build_split_manifest(
        &rows,
        &families(3),
        &json!(0.5),
        &json!(0.25),
        &json!(7),
        &[],
    )
    .unwrap();
    assert!(manifest["assignments"].get("ignored").is_none());
    assert_eq!(manifest["assignments"]["d0"]["first_committed_at_ms"], 7);
    validate_split_manifest(&manifest).unwrap();
    manifest["seed"] = json!(8);
    assert_eq!(
        validate_split_manifest(&manifest).unwrap_err().to_string(),
        "split manifest content hash differs"
    );
}

#[test]
fn incomplete_inputs_are_reported_as_sorted_unique_reasons() {
    let rows = vec![json!({"dataset": "missing", "committed_at_ms": 1})];
    let manifest = build_split_manifest(
        &rows,
        &BTreeMap::new(),
        &json!(0.2),
        &json!(0.2),
        &json!(3),
        &[],
    )
    .unwrap();
    assert_eq!(manifest["status"], "incomplete");
    assert_eq!(manifest["assignments"], json!({}));
    assert_eq!(
        manifest["incomplete_reasons"],
        json!(["missing_source_family"])
    );
    assert_eq!(manifest["missing_source_family"], json!(["missing"]));
    validate_split_manifest(&manifest).unwrap();
}

#[test]
fn legacy_development_selection_cannot_split_a_family() {
    let mut source_families = families(5);
    source_families.insert("d1".into(), "family-0".into());
    let manifest = build_split_manifest(
        &labels(5),
        &source_families,
        &json!(0.5),
        &json!(0.25),
        &json!(1),
        &[json!("d0")],
    )
    .unwrap();
    assert_eq!(manifest["status"], "incomplete");
    assert_eq!(
        manifest["incomplete_reasons"],
        json!(["legacy_development_family_overlap"])
    );
    validate_split_manifest(&manifest).unwrap();
}

#[test]
fn validator_rejects_family_overlap_after_hash_is_recomputed() {
    let mut manifest = build(6, json!(0.25), &[]);
    let first = manifest["assignments"]["d0"]["split"].clone();
    manifest["assignments"]["d1"]["split"] = json!(if first == TRAIN_SPLIT {
        DEVELOPMENT_SPLIT
    } else {
        TRAIN_SPLIT
    });
    let mut body = manifest.clone();
    body.as_object_mut().unwrap().remove("manifest_sha256");
    manifest["manifest_sha256"] = Value::String(canonical_sha256(&body).unwrap());
    assert_eq!(
        validate_split_manifest(&manifest).unwrap_err().to_string(),
        "source family overlaps multiple splits"
    );
}

#[test]
fn validator_rejects_unknown_fields_and_noncanonical_lists() {
    let mut manifest = build(6, json!(0.25), &[]);
    manifest["unexpected"] = json!(true);
    assert_eq!(
        validate_split_manifest(&manifest).unwrap_err().to_string(),
        "split manifest has unknown or missing fields"
    );
    let mut manifest = build(6, json!(0.25), &[]);
    manifest["missing_source_family"] = json!(["z", "a"]);
    let mut body = manifest.clone();
    body.as_object_mut().unwrap().remove("manifest_sha256");
    manifest["manifest_sha256"] = Value::String(canonical_sha256(&body).unwrap());
    assert_eq!(
        validate_split_manifest(&manifest).unwrap_err().to_string(),
        "split manifest missing_source_family is not canonical"
    );
}
