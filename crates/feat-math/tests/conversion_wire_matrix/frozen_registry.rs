#[test]
fn all_frozen_wire_ids_have_current_conversion_witnesses_or_explicit_refusals() {
    let historical: Value = serde_json::from_str(HISTORICAL).unwrap();
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = historical["registry"].as_array().unwrap();
    let programs = rows_by_name(emitted["programs"].as_array().unwrap());
    let model_rows = rows_by_name(models["models"].as_array().unwrap());
    let manifest = Manifest::current();
    let registry = operator_registry();
    let generatable: BTreeSet<_> = generatable_operators(false).map(|op| op.name).collect();
    let mut witness_names = BTreeSet::new();
    let mut kinds = BTreeMap::<&str, usize>::new();

    assert_eq!(registry.len(), 53, "frozen wire IDs 0–52");
    assert_eq!(frozen.len(), registry.len());
    for (id, (op, old)) in registry.iter().zip(frozen).enumerate() {
        assert_eq!(usize::from(op.id.0), id, "wire order");
        assert_eq!(old["id"], id, "wire {id} frozen ID");
        assert_eq!(old["name"], op.name, "wire {id} spelling");
        assert_eq!(old["semantic_key"], op.semantic_key, "wire {id} semantics");
        for (field, actual) in [
            ("parseable", op.lifecycle.parseable),
            ("executable", op.lifecycle.executable),
            ("generatable", op.lifecycle.generatable),
            ("quarantined", op.lifecycle.quarantined),
        ] {
            assert_eq!(old[field], actual, "wire {id} lifecycle {field}");
        }
        assert_eq!(generatable.contains(op.name), op.lifecycle.generatable);
        let cap = manifest
            .op(op.name)
            .unwrap_or_else(|| panic!("wire {id} missing manifest"));
        assert_eq!(cap.wire_id, Some(op.id.0));
        let kind = old["evidence"]["kind"].as_str().unwrap();
        if op.lifecycle.generatable {
            assert_eq!(kind, "fitted_existing", "wire {id}: search-visible witness");
        }
        *kinds.entry(kind).or_default() += 1;
        match kind {
            "fitted_existing" => {
                let name = old["evidence"]["name"].as_str().unwrap();
                let source = programs[name]["canonical"].as_str().unwrap();
                assert!(
                    source.contains(&format!("({} ", op.name))
                        || source.contains(&format!("({})", op.name)),
                    "wire {id}: witness must actually contain {}",
                    op.name
                );
                assert!(
                    model_rows.contains_key(name),
                    "wire {id}: missing fitted witness"
                );
                let parsed = GenomeIr::parse(source).unwrap();
                let canonical = parsed.canonical();
                let reparsed = GenomeIr::parse(&canonical).unwrap();
                assert_eq!(
                    reparsed.canonical(),
                    canonical,
                    "wire {id}: canonical fixed point"
                );
                assert_eq!(
                    Program::import_legacy(&parsed).to_bytes().unwrap(),
                    Program::import_legacy(&reparsed).to_bytes().unwrap(),
                    "wire {id}: executable codec round trip"
                );
                assert!(
                    supports(&reparsed).is_ok(),
                    "wire {id}: paired recipe refused"
                );
                let artifact = emit(&reparsed, &EmitOptions::default()).unwrap();
                let sidecar: Value = serde_json::from_str(&artifact.manifest).unwrap();
                assert_eq!(sidecar["program"], canonical);
                witness_names.insert(name);
            }
            "alias_case" => {
                assert!(op.lifecycle.executable && !op.lifecycle.generatable);
                assert_ne!(op.name, op.semantic_key, "wire {id}: alias key");
            }
            "quarantined_refusal" => {
                assert!(op.lifecycle.quarantined && !op.lifecycle.executable);
            }
            other => panic!("wire {id}: unclassified witness {other}"),
        }
    }
    assert_eq!(kinds["fitted_existing"], 42);
    assert_eq!(kinds["alias_case"], 4);
    assert_eq!(kinds["quarantined_refusal"], 7);
    assert_eq!(witness_names.len(), 19, "deduplicated executable programs");
    assert_eq!(generatable.len(), 35, "search-visible frozen wire subset");
}
