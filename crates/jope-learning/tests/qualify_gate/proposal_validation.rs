use super::*;

#[test]
fn roster_defects_are_named() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "in both",
            Box::new(|p| p["partition"]["confirmation"] = json!(["c1", "s1"])),
        ),
        (
            "listed twice",
            Box::new(|p| p["partition"]["meta_training"] = json!(["t1", "t1"])),
        ),
        (
            "unknown role",
            Box::new(|p| p["partition"]["holdout"] = json!(["h1"])),
        ),
        (
            "no development list",
            Box::new(|p| {
                p["partition"]
                    .as_object_mut()
                    .unwrap()
                    .remove("development");
            }),
        ),
        (
            "roles",
            Box::new(|p| p["roles"] = json!(["meta_training", "sealed_final"])),
        ),
        (
            "members outside",
            Box::new(|p| {
                p["sealed_family_records"]["s2"]["member_roles"] = json!(["sealed_final", "train"]);
            }),
        ),
        (
            "dataset_hashes shared",
            Box::new(|p| {
                p["sealed_family_records"]["s3"]["dataset_hashes"] = json!(["d1"]);
            }),
        ),
        (
            "no context, dataset or view id",
            Box::new(|p| {
                p["sealed_family_records"]["s1"] = json!({"context_ids": [], "dataset_hashes": [], "member_roles": ["sealed_final"], "view_ids": []});
            }),
        ),
        (
            "without a record",
            Box::new(|p| {
                p["sealed_family_records"]
                    .as_object_mut()
                    .unwrap()
                    .remove("s3");
            }),
        ),
        (
            "outside sealed_final",
            Box::new(|p| {
                p["sealed_family_records"]["c1"] = p["sealed_family_records"]["s1"].clone();
            }),
        ),
        (
            "proof did not pass",
            Box::new(|p| p["proof"]["passed"] = json!(false)),
        ),
        (
            "proof covers",
            Box::new(|p| p["proof"]["families"] = json!(2)),
        ),
        (
            "sealed_views_in_closure is 4",
            Box::new(|p| {
                p["sources"]["closure"]["sealed_views_in_closure"] = json!(4);
            }),
        ),
        (
            "sources.closure is absent",
            Box::new(|p| p["sources"] = json!({})),
        ),
        (
            "cannot claim adopted",
            Box::new(|p| p["adopted"] = json!(true)),
        ),
        (
            "cannot claim seal_spent",
            Box::new(|p| p["seal_spent"] = json!(true)),
        ),
        (
            "sealed_final is empty",
            Box::new(|p| {
                p["partition"]["sealed_final"] = json!([]);
                p["sealed_family_records"] = json!({});
                p["proof"]["families"] = json!(0);
            }),
        ),
    ];
    for (expected, mutate) in cases {
        let mut document = proposal(json!(0.05));
        mutate(&mut document);
        let report = roster::check(&bytes(&document)).unwrap();
        assert!(
            report.defects.iter().any(|item| item.contains(expected)),
            "{expected}: {:?}",
            report.defects
        );
        assert!(!report.adoptable(), "{expected}");
        let refusal = request_seal(&Gate::On, &bytes(&document)).unwrap_err();
        assert!(
            matches!(refusal, Refusal::NotReady { .. }),
            "{expected}: {refusal:?}"
        );
    }
}

#[test]
fn only_a_decided_alpha_level_settles_the_budget() {
    for budget in [
        json!("unspent_jepson_decision"),
        json!(null),
        json!(0),
        json!(1),
        json!(1.5),
    ] {
        let report = roster::check(&bytes(&proposal(budget.clone()))).unwrap();
        assert!(report.defects.is_empty(), "{budget}");
        assert_eq!(report.owner_decisions.len(), 1, "{budget}");
    }
}

#[test]
fn a_document_of_another_schema_is_invalid() {
    let mut document = proposal(json!(0.05));
    document["schema"] = json!("jope.rlfw-judge-seal/v1");
    assert!(roster::check(&bytes(&document))
        .unwrap_err()
        .contains("schema"));
    let refusal = request_seal(&Gate::On, &bytes(&document)).unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::Invalid {
            action: Action::Seal,
            ..
        }
    ));
}

// ---- promotion (G36 / J18) ---------------------------------------------------------------------
