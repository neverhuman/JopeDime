#[test]
fn derived_release_reopens_to_identical_bytes_and_identity() {
    let f = Fixture::new();
    let release = f.derive().unwrap();
    let bytes = release.encode().unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let reopened = RegressionRelease::open(&bytes).unwrap();
    assert_eq!(reopened, release);
    assert_eq!(reopened.encode().unwrap(), bytes);
    assert_eq!(
        f.derive().unwrap().encode().unwrap(),
        bytes,
        "derivation is deterministic"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("release.json");
    std::fs::write(&path, &bytes).unwrap();
    let stored = std::fs::read(&path).unwrap();
    assert_eq!(
        RegressionRelease::open(&stored).unwrap().encode().unwrap(),
        stored
    );
    assert_eq!(release.release_id, release.compute_id().unwrap());

    let names: Vec<&str> = release.members.iter().map(|m| m.dataset.as_str()).collect();
    assert_eq!(
        names,
        (0..10).map(|i| format!("reg{i}")).collect::<Vec<_>>()
    );
    assert_eq!(
        release
            .members
            .iter()
            .map(|m| m.panel_index)
            .collect::<Vec<_>>(),
        (0..10).map(|i| i * 2).collect::<Vec<u64>>()
    );
    assert_eq!(release.exclusions.len(), 20);
    assert!(release
        .exclusions
        .iter()
        .all(|e| e.reason == format!("task_not_regression:{}", e.task)));
    assert_eq!(
        release.derived_from.task_counts,
        BTreeMap::from([
            ("binary".to_owned(), 10),
            ("multiclass".to_owned(), 10),
            ("regression".to_owned(), 10),
            ("timeseries".to_owned(), 0)
        ])
    );
    assert_eq!(
        release.family_registry_sha256,
        sha256_hex(&f.registry.encode().unwrap())
    );
    for (id, staged) in &f.staged {
        release.verify_staged(id, staged).unwrap();
        release.verify_development(id, &staged.development).unwrap();
    }
    check_against_panel(&release, &f.panel_bytes()).unwrap();
    let m = release.member("reg0").unwrap();
    assert_eq!(m.target.column, TARGET_COLUMN);
    assert_eq!(m.target.development_label_role, DEVELOPMENT_LABEL_ROLE);
    assert_eq!(m.target.sealed_label_role, SEALED_LABEL_ROLE);
    assert_eq!(
        (m.rows.total, m.rows.development, m.rows.sealed),
        (12, 10, 2)
    );
}

type Mutation = (&'static str, fn(&mut MemberData));

fn consistent_mutations() -> Vec<Mutation> {
    vec![
        ("changed_raw_source_bytes", |m| m.raw.push(b'!')),
        ("changed_source_metadata", |m| m.metadata.push(b'!')),
        ("changed_one_target", |m| m.rows[3].target = "999".into()),
        ("swapped_two_targets_row_label_misalignment", |m| {
            let t = m.rows[1].target.clone();
            m.rows[1].target = m.rows[2].target.clone();
            m.rows[2].target = t;
        }),
        ("changed_feature_kind", |m| {
            m.features[0]["kind"] = json!("categorical")
        }),
        ("renamed_feature", |m| {
            m.features[1]["name"] = json!("colour")
        }),
        ("changed_feature_value", |m| {
            m.rows[4].values[0] = Some("0.25".into())
        }),
        ("changed_row_identity", |m| {
            m.rows[5].id = hex("replacement")
        }),
        ("merged_two_groups_within_a_fold", |m| {
            m.rows[5].group = m.rows[0].group.clone()
        }),
        ("reassigned_folds_with_equal_counts", |m| {
            m.rows[0].fold = Some(1);
            m.rows[1].fold = Some(0);
        }),
        ("moved_a_development_row_to_sealed", |m| {
            m.rows[10].fold = Some(0);
            m.rows[0].fold = None;
        }),
        ("reordered_rows", |m| m.rows.swap(0, 1)),
    ]
}

#[test]
fn changed_source_target_rows_schema_groups_folds_or_family_mapping_is_a_new_identity_or_rejected()
{
    let base = Fixture::new();
    let original = base.derive().unwrap();
    let original_bytes = original.encode().unwrap();
    let mut decisions = Vec::new();
    for (label, mutate) in consistent_mutations() {
        // (1) Inputs change but the frozen declarations do not: refusal.
        let mut f = Fixture::new();
        let panel = f.panel.clone();
        mutate(&mut f.members[0]);
        f.rebuild();
        f.panel = panel;
        let refused = err_text(f.derive());
        // (2) The old release never re-interprets the changed bytes.
        let old_check = original.verify_staged("reg0", &f.staged["reg0"]);
        assert!(
            old_check.is_err(),
            "{label}: old release accepted changed bytes"
        );
        // (3) A consistently re-declared input is a different release identity.
        let mut g = Fixture::new();
        mutate(&mut g.members[0]);
        g.rebuild();
        let changed = g.derive().unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_ne!(changed.release_id, original.release_id, "{label}");
        assert_ne!(changed.encode().unwrap(), original_bytes, "{label}");
        decisions.push(
            json!({"mutation":label,"stale_declaration":"rejected","stale_reason":refused,
            "old_release_verify":"rejected","redeclared":"new_release_identity",
            "new_release_id":changed.release_id}),
        );
    }

    // A declaration-only change (grouping columns) is a new identity as well.
    let mut f = Fixture::new();
    f.members[0].group_columns = vec!["c".into()];
    f.rebuild();
    assert_ne!(f.derive().unwrap().release_id, original.release_id);
    decisions.push(json!({"mutation":"declared_group_column","decision":"new_release_identity"}));

    // A reordered split (source indices no longer positional) is refused outright.
    let mut staged = base.staged["reg0"].clone();
    let mut split: Value = serde_json::from_slice(&staged.split).unwrap();
    split["rows"].as_array_mut().unwrap().swap(0, 1);
    staged.split = canonical(&split);
    let mut g = Fixture::new();
    g.panel["datasets"][0]["split"]["sha256"] = json!(sha256_hex(&staged.split));
    g.staged.insert("reg0".into(), staged);
    assert!(err_text(g.derive()).contains("not in original source order"));
    decisions.push(json!({"mutation":"split_rows_out_of_source_order","decision":"rejected"}));

    // A family mapping or exposure change is a new release identity.
    for (label, change) in [
        ("added_unresolved_relation", 0),
        ("changed_exposure_basis", 1),
        ("added_family_source_key", 2),
    ] {
        let mut f = Fixture::new();
        match change {
            0 => f.registry.families[0].relations.push(Relation {
                dataset: "reg0".into(),
                other: "external:variant".into(),
                kind: RelationKind::AlternateTarget,
                status: RelationStatus::Unresolved,
                evidence: "name similarity only".into(),
            }),
            1 => f.registry.exposures[0].encoder.basis = "different basis".into(),
            _ => f.registry.families[0]
                .source_keys
                .push("origin:alias".into()),
        }
        let changed = f.derive().unwrap();
        assert_ne!(
            changed.family_registry_sha256,
            original.family_registry_sha256
        );
        assert_ne!(changed.release_id, original.release_id, "{label}");
        decisions.push(json!({"mutation":label,"decision":"new_release_identity"}));
    }

    // Changed preparation implementation is a new identity.
    let mut f = Fixture::new();
    f.preparation.extend_from_slice(b"# changed\n");
    assert_ne!(f.derive().unwrap().release_id, original.release_id);
    decisions
        .push(json!({"mutation":"changed_preparation_source","decision":"new_release_identity"}));

    // Missing, substituted or surplus objects never fall back.
    let mut f = Fixture::new();
    f.staged.remove("reg3");
    assert!(err_text(f.derive()).contains("no substitution"));
    let mut f = Fixture::new();
    let other = f.staged["reg4"].clone();
    f.staged.insert("reg3".into(), other);
    assert!(f.derive().is_err());
    let mut f = Fixture::new();
    let extra = f.staged["reg4"].clone();
    f.staged.insert("bin0".into(), extra);
    assert!(err_text(f.derive()).contains("non-members"));
    decisions
        .push(json!({"mutation":"missing_substituted_or_surplus_object","decision":"rejected"}));

    if let Ok(path) = std::env::var("W1_MUTATION_RECEIPT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({"schema":"jope.w1.regression-release-mutations/v1",
                "original_release_id":original.release_id,"decisions":decisions}))
            .unwrap(),
        )
        .unwrap();
    }
}
