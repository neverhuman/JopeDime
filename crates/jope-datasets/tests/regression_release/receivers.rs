#[test]
fn development_receiver_rejects_changed_bytes_and_realigned_labels() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let dev = &f.staged["reg2"].development;
    release.verify_development("reg2", dev).unwrap();
    let mut table: Value = serde_json::from_slice(dev).unwrap();
    let rows = table["rows"].as_array_mut().unwrap();
    let t = rows[0]["target"].clone();
    rows[0]["target"] = rows[1]["target"].clone();
    rows[1]["target"] = t;
    let swapped = canonical(&table);
    assert!(err_text(release.verify_development("reg2", &swapped)).contains("development bytes"));
    assert!(release.verify_development("reg9", dev).is_err());
    assert!(release.verify_development("bin0", dev).is_err());
}

#[test]
fn unknown_fields_versions_and_noncanonical_encodings_are_rejected() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let bytes = release.encode().unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let reject = |v: &Value, needle: &str| {
        let text = err_text(RegressionRelease::open(&canonical(v)));
        assert!(text.contains(needle), "{needle}: {text}");
    };
    let mut v = value.clone();
    v["future_field"] = json!(true);
    reject(&v, "fields");
    let mut v = value.clone();
    v["members"][0]["raw"]["mirror"] = json!("s3://latest");
    reject(&v, "fields");
    let mut v = value.clone();
    v["schema"] = json!("jope.regression-release/v2");
    reject(&v, "unsupported regression release schema");
    let mut v = value.clone();
    v["release_id"] = json!(hex("forged"));
    reject(&v, "identity");
    let mut v = value.clone();
    v["members"][0]["rows"]["development"] = json!(9);
    reject(&v, "");
    let mut v = value.clone();
    v["members"][0]["target"]["column"] = json!("target_alias");
    reject(&v, "target identity");
    let mut v = value.clone();
    v["members"][0]["target"]["development_label_role"] = json!("outer_test");
    reject(&v, "label roles");
    let mut v = value.clone();
    v["members"][0]["rows"]["total"] = json!(12.0);
    assert!(RegressionRelease::open(&serde_json::to_vec(&v).unwrap()).is_err());
    assert!(err_text(RegressionRelease::open(
        &serde_json::to_vec_pretty(&value).unwrap()
    ))
    .contains("canonical"));
    assert!(RegressionRelease::open(&bytes[..bytes.len() - 1]).is_err());
    let text = String::from_utf8(bytes.clone()).unwrap();
    let duplicated = text.replacen('{', "{\"schema\":\"jope.regression-release/v0\",", 1);
    assert!(RegressionRelease::open(duplicated.as_bytes()).is_err());

    let registry = f.registry.encode().unwrap();
    assert_eq!(FamilyRegistry::open(&registry).unwrap(), f.registry);
    let rv: Value = serde_json::from_slice(&registry).unwrap();
    let mut v = rv.clone();
    v["exposures"][0]["encoder"]["assumed"] = json!("unseen");
    assert!(FamilyRegistry::open(&canonical(&v)).is_err());
    let mut v = rv.clone();
    v["schema"] = json!("jope.regression-family-registry/v2");
    assert!(FamilyRegistry::open(&canonical(&v)).is_err());
    assert!(FamilyRegistry::open(&serde_json::to_vec_pretty(&rv).unwrap()).is_err());
}

#[test]
fn membership_is_every_regression_entry_of_the_frozen_thirty_entry_panel() {
    let f = Fixture::new();
    assert_eq!(
        panel_regression_members(&f.panel_bytes()).unwrap(),
        (0..10).map(|i| format!("reg{i}")).collect::<Vec<_>>()
    );
    let release = f.derive().unwrap();
    let mut g = Fixture::new();
    g.panel["datasets"][0]["task"] = json!("binary");
    assert!(err_text(g.derive()).contains("exactly ten regression"));
    let mut g = Fixture::new();
    g.panel["datasets"][1]["id"] = json!("reg0");
    assert!(g.derive().is_err());
    let mut g = Fixture::new();
    g.panel["datasets"].as_array_mut().unwrap().pop();
    assert!(err_text(g.derive()).contains("thirty"));
    let mut g = Fixture::new();
    g.panel["datasets"][0]["replacement_for"] = json!("reg1");
    assert!(g.derive().is_err());
    let mut g = Fixture::new();
    g.panel["datasets"][4]["source"]["sha256"] = json!(hex("other version"));
    assert!(check_against_panel(&release, &g.panel_bytes()).is_err());
    let mut g = Fixture::new();
    g.panel["datasets"][0]["split"]["protocol"] = json!("other-protocol");
    assert!(err_text(g.derive()).contains("preparation protocol"));
}

#[test]
fn family_registry_keeps_unknown_distinct_from_unseen_and_blocks_never_seen_claims() {
    let f = Fixture::new();
    f.registry.validate().unwrap();
    for id in f.registry.members() {
        let blockers = f.registry.never_seen_claim_blockers(id).unwrap();
        assert_eq!(blockers.len(), 3, "{id}: {blockers:?}");
        assert!(blockers[0].starts_with("encoder exposure unknown"));
    }
    assert_ne!(ExposureState::Unknown, ExposureState::Unseen);

    let unseen = || Exposure {
        state: ExposureState::Unseen,
        basis: "auditable clean root".into(),
        evidence: vec![EvidenceRef {
            description: "complete read log".into(),
            sha256: Some(hex("read log")),
        }],
    };
    let mut clean = f.registry.clone();
    for e in &mut clean.exposures {
        e.encoder = unseen();
        e.teacher = unseen();
        e.checkpoint = unseen();
    }
    clean.validate().unwrap();
    assert!(clean.never_seen_claim_blockers("reg0").unwrap().is_empty());

    // Unseen or exposed without evidence is not a claim; it must stay unknown.
    let mut r = clean.clone();
    r.exposures[0].checkpoint.evidence.clear();
    assert!(err_text(r.validate()).contains("requires evidence"));
    let mut r = clean.clone();
    r.exposures[0].encoder = Exposure {
        state: ExposureState::Exposed,
        basis: "trained".into(),
        evidence: vec![],
    };
    assert!(r.validate().is_err());

    // One unknown component blocks the claim even when the others are unseen.
    let mut r = clean.clone();
    r.exposures[0].checkpoint = unknown("ancestry untraceable");
    assert_eq!(r.never_seen_claim_blockers("reg0").unwrap().len(), 1);
    assert!(r.never_seen_claim_blockers("reg1").unwrap().is_empty());

    // Unresolved relationships block both sides.
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "reg1".into(),
        kind: RelationKind::SharedEntity,
        status: RelationStatus::Unresolved,
        evidence: "same collection name".into(),
    });
    r.validate().unwrap();
    assert_eq!(r.never_seen_claim_blockers("reg0").unwrap().len(), 1);
    assert_eq!(r.never_seen_claim_blockers("reg1").unwrap().len(), 1);
    assert!(r.never_seen_claim_blockers("reg2").unwrap().is_empty());

    // A resolved alias/alternate target cannot straddle the exclusion boundary.
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "reg1".into(),
        kind: RelationKind::Alias,
        status: RelationStatus::Resolved,
        evidence: "documented alias".into(),
    });
    assert!(err_text(r.validate()).contains("straddles"));
    let mut r = clean.clone();
    r.families[0].relations.push(Relation {
        dataset: "reg0".into(),
        other: "external:derived".into(),
        kind: RelationKind::AlternateTarget,
        status: RelationStatus::Resolved,
        evidence: "documented alternate target".into(),
    });
    assert!(r.validate().is_err());
    r.families[0].source_keys.push("external:derived".into());
    r.validate().unwrap();
    // The same external key cannot belong to two families.
    r.families[1].source_keys.push("external:derived".into());
    assert!(err_text(r.validate()).contains("cannot span families"));

    // Every member needs exactly one exposure record; releases need full coverage.
    let mut r = clean.clone();
    r.exposures.remove(3);
    assert!(r.validate().is_err());
    let mut g = Fixture::new();
    g.registry.families.pop();
    g.registry.exposures.pop();
    g.registry.validate().unwrap();
    assert!(err_text(g.derive()).contains("cover exactly"));
}
