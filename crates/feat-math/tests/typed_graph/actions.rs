use super::*;

#[test]
fn every_enumerated_tiny_legal_action_applies_validates_and_actually_executes() {
    let s = schema(2);
    let g = graph("(pipe X (standardize) (ridge))", &s);
    let original = g.to_bytes(&limits()).unwrap();
    let d = dataset(&s, 12);
    let edits = legal_edits(&g, &s, profile(), &limits()).unwrap();
    assert!(edits.len() > 10);
    let mut executions = 0;
    for edit in &edits {
        let child = apply(&g, edit, &s, profile(), &limits()).unwrap();
        child.graph.validate(&limits()).unwrap();
        let (prediction, _, stats) = run(&child.graph, &d);
        assert_eq!(prediction.len(), 4);
        assert_eq!(stats.fit_calls, 1);
        executions += 1;
    }
    assert_eq!(g.to_bytes(&limits()).unwrap(), original);
    println!("LEGAL_EDIT_EXECUTIONS={executions}");
}

#[test]
fn forged_edits_versions_profiles_parameters_and_parents_cannot_bypass_admission() {
    let s = schema(2);
    let g = graph("(pipe X (ridge))", &s);
    for kind in [
        EditKind::Delete { site: usize::MAX },
        EditKind::Parameter {
            site: 1,
            key: "unknown".into(),
            value: Value::Int(1),
        },
        EditKind::Parameter {
            site: 1,
            key: "lambda_frac".into(),
            value: Value::Text("wrong".into()),
        },
        EditKind::Insert {
            before: 1,
            operator: OperatorRef::new("legacy_recipe", "count_sketch").unwrap(),
            parameters: Default::default(),
        },
        EditKind::Insert {
            before: 1,
            operator: OperatorRef::new("legacy_recipe", "poly_interact").unwrap(),
            parameters: [("topk".into(), Value::Int(4))].into(),
        },
        EditKind::ReplaceHead {
            site: 0,
            operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
            parameters: Default::default(),
        },
    ] {
        let edit = Edit::new(&g, kind, &limits()).unwrap();
        assert!(apply(&g, &edit, &s, profile(), &limits()).is_err());
    }
    let mut e = Edit::new(&g, EditKind::NoOp, &limits()).unwrap();
    e.parents[0].sha256 = canon::sha256(b"wrong parent");
    assert!(apply(&g, &e, &s, profile(), &limits()).is_err());
    assert!(legal_edits(
        &g,
        &s,
        registry_v2::RegistryProfile {
            dialect: "advanced_v3"
        },
        &limits()
    )
    .is_err());
    let bad = forged(&g, |v| {
        v["nodes"][1]["operator"]["semantic_version"] = json!("99")
    });
    assert!(bad.validate(&limits()).is_err());
}

#[test]
fn crossover_serializes_both_exact_parents_and_rechecks_donor_binding() {
    let s = schema(2);
    let left = graph("(pipe X (standardize) (ridge))", &s);
    let right = graph("(pipe X (elasticnet alpha=0.01))", &s);
    let edit = crossover_edit(&left, &right, &s, profile(), &limits()).unwrap();
    let a = apply(&left, &edit, &s, profile(), &limits()).unwrap();
    let b = apply(&left, &edit, &s, profile(), &limits()).unwrap();
    assert_eq!(a.record, b.record);
    let bytes = canon::encode(&a.record).unwrap();
    let restored: EditRecord = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        restored.edit.parents,
        vec![
            left.identity(&limits()).unwrap(),
            right.identity(&limits()).unwrap()
        ]
    );
    assert_eq!(restored.child, a.graph.identity(&limits()).unwrap());
    run(&a.graph, &dataset(&s, 12));
    let mut forged = edit;
    forged.parents[1] = left.identity(&limits()).unwrap();
    assert!(apply(&left, &forged, &s, profile(), &limits())
        .unwrap_err()
        .contains("second-parent"));
}

#[test]
fn unchanged_actions_keep_explicit_outcome_and_original_parent() {
    let s = schema(2);
    let g = graph("(pipe X (ridge))", &s);
    let e = Edit::new(&g, EditKind::NoOp, &limits()).unwrap();
    let applied = apply(&g, &e, &s, profile(), &limits()).unwrap();
    assert_eq!(applied.record.outcome, EditOutcome::Unchanged);
    assert_eq!(applied.graph, g);
    assert_eq!(applied.record.child, applied.record.edit.parents[0]);
}
