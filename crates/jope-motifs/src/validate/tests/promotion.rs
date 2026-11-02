use super::*;

#[test]
fn promotion_gates_are_recorded_required_not_run() {
    let table = fixture::table();
    for motif in &table.body.motifs {
        assert_eq!(motif.lifecycle, Lifecycle::Quarantined);
        assert_eq!(motif.promotion, Promotion::required_not_run());
        let value = serde_json::to_value(&motif.promotion).expect("json");
        let object = value.as_object().expect("an object");
        assert_eq!(object.len(), 6);
        for key in [
            "ablation",
            "semantic_backend_coverage",
            "versioned_vocabulary_compatibility",
            "expanded_runtime",
            "cross_family_usefulness",
            "replay",
        ] {
            assert_eq!(object[key], "required_not_run", "{key}");
        }
    }
    let validation = fixture::passing_validation(&table);
    let export = crate::export::PublicExport::from_validation(&validation, &table);
    assert_eq!(export.promoted, 0);
    let value = serde_json::to_value(export).expect("json");
    assert_eq!(value["promotion"], "required_not_run");
}

#[test]
fn empty_table_never_passes() {
    let parameters = MineParameters {
        min_datasets: 5,
        ..MineParameters::default()
    };
    let empty = mine(&fixture::set(), None, parameters).expect("mines");
    assert!(empty.body.motifs.is_empty());
    let validation = validate(&empty, &ValidateOptions::default()).expect("validates");
    assert!(validation.motifs.is_empty());
    assert!(!validation.passed());
    let checks = crate::export::PublicChecks::from_results(&validation.motifs);
    assert_eq!(checks.round_trip.passed, 0);
}

#[test]
fn source_verification_confirms_every_occurrence_and_never_nothing() {
    let table = fixture::small_table();
    let lines = fixture::small_lines();
    let mut verified = fixture::passing_validation(&table);
    assert!(verified.passed());
    let code = verify_source(&mut verified, &table, &fixture::set_of(lines.clone()));
    assert_eq!(code, None);
    assert!(verified.source_verified);
    assert_eq!(verified.source_digest_equal, Some(true));
    assert!(verified.passed());

    let host = lines[1].program().to_owned();
    let stale_sets = [
        // The record is gone: a record with another identity took its place.
        fixture::qualified(&host, TaskLabel::Regression, "d2", "s02-other"),
        // The record's labels became optimization-visible test labels.
        fixture::qualified_with(
            &host,
            TaskLabel::Regression,
            "d2",
            "s02",
            fixture::adaptive_label(),
        ),
        // The record now holds another program.
        fixture::qualified(
            &fixture::v3("(standardize) (rank)", "elasticnet"),
            TaskLabel::Regression,
            "d2",
            "s02",
        ),
        // The record moved to another dataset.
        fixture::qualified(&host, TaskLabel::Regression, "d9", "s02"),
    ];
    for (index, replaced) in stale_sets.into_iter().enumerate() {
        let mut stale = fixture::passing_validation(&table);
        let set = fixture::set_of(vec![lines[0].clone(), replaced]);
        let code = verify_source(&mut stale, &table, &set);
        assert_eq!(code, Some("source_not_confirmed"), "case {index}");
        assert!(!stale.source_verified, "case {index}");
        // Every fixture set carries the same digest: equal digests prove nothing alone.
        assert_eq!(stale.source_digest_equal, Some(true), "case {index}");
        assert!(!stale.passed(), "case {index}");
    }

    // Nothing examined: a table without occurrences is never confirmed.
    let parameters = MineParameters {
        min_datasets: 5,
        ..MineParameters::default()
    };
    let empty = mine(&fixture::set_of(lines.clone()), None, parameters).expect("mines");
    let mut nothing = fixture::passing_validation(&empty);
    let code = verify_source(&mut nothing, &empty, &fixture::set_of(lines));
    assert_eq!(code, Some("source_not_exercised"));
    assert!(!nothing.source_verified);
    assert_eq!(nothing.source_digest_equal, Some(true));
    assert!(!nothing.passed());
}

#[test]
fn validation_is_deterministic() {
    let table = fixture::small_table();
    let first = validate(&table, &ValidateOptions::default()).expect("validates");
    let second = validate(&table, &ValidateOptions::default()).expect("validates");
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_vec(&first).expect("json"),
        serde_json::to_vec(&second).expect("json")
    );
    assert!(first.passed());
}
