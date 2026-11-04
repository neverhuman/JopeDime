#[test]
fn mixed_status_denominator_retains_every_original_occurrence() {
    let archive = import(INDEX);
    assert_eq!(archive.observations.len(), 10);
    assert_eq!(
        archive.primary_counts,
        BTreeMap::from([
            ("scored".into(), 3),
            ("failed".into(), 1),
            ("timeout".into(), 1),
            ("partial".into(), 1),
            ("unscored".into(), 1),
            ("unknown".into(), 2),
            ("rejected_measurement".into(), 1)
        ])
    );
    assert_eq!(archive.primary_counts.values().sum::<usize>(), 10);
    assert_eq!(archive.original_index.as_bytes(), INDEX);
    assert_eq!(archive.observations[9].original_row, Value::Null);
    assert_eq!(
        archive.observations[2].event_references["attempt_id"],
        "attempt-failed"
    );
    assert_eq!(
        serde_json::from_str::<ObservationArchive>(&serde_json::to_string(&archive).unwrap())
            .unwrap(),
        archive
    );
}

#[test]
fn unknown_failure_reason_and_missing_status_never_become_labels() {
    let archive = import(INDEX);
    let failed = &archive.observations[2];
    assert_eq!(failed.outcome, ObservedOutcome::Failed);
    assert_eq!(failed.failure_reason, None);
    assert_eq!(
        archive.observations[0].outcome,
        ObservedOutcome::Unknown,
        "a score does not imply success"
    );
    assert_eq!(archive.observations[6].outcome, ObservedOutcome::Unknown);
    assert_eq!(archive.observations[6].measurement, Measurement::Missing);
    assert!(!archive.observations[6].labels.failure.observed);
}

#[test]
fn scores_preserve_exact_binary64_identity_and_legacy_order() {
    let archive = import(INDEX);
    let bits: Vec<Value> = serde_json::from_str(include_str!(
        "../fixtures/observation-import-v2/score-bits.json"
    ))
    .unwrap();
    for b in bits {
        let obs = archive
            .observations
            .iter()
            .find(|o| o.identity.locator == b["locator"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            format!("{:016x}", obs.measurement.value().unwrap().to_bits()),
            b["f64_bits"].as_str().unwrap()
        );
        if let Measurement::Finite { score_name, .. } = &obs.measurement {
            assert_eq!(score_name.as_deref(), b["score_name"].as_str());
        } else {
            panic!("missing finite score");
        }
    }
    let old = distilled::distill_gp_results_index_with_stats(
        &serde_json::from_slice(INDEX).unwrap(),
        distilled::SplitRole::Unknown,
        &known(),
    )
    .unwrap();
    let new =
        distilled::legacy_scored_projection(&archive, distilled::SplitRole::Unknown, &known())
            .unwrap();
    assert_eq!(old, new);
    assert_eq!(
        old.0
            .iter()
            .map(|r| r.score.unwrap().to_bits())
            .collect::<Vec<_>>(),
        vec![0.61f64.to_bits(), 0.59f64.to_bits()]
    );
    assert_eq!(old.2.total_index_runs, 10);
    assert_eq!(old.2.distilled_rows, 2);
    assert_eq!(old.2.skipped_unscored_runs, 7);
    assert_eq!(old.2.skipped_incomplete_provenance_runs, 1);
    assert_eq!(
        distilled::write(&old.0, &known(), &[]).unwrap(),
        distilled::write(&new.0, &known(), &[]).unwrap()
    );
}

#[test]
fn signed_zero_and_unknown_score_identity_are_retained_without_validation_upgrade() {
    let bytes = br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"score":-0.0,"score_name":"unknown_identity"}]}"#;
    let archive = import(bytes);
    assert_eq!(
        archive.observations[0]
            .measurement
            .value()
            .unwrap()
            .to_bits(),
        (-0.0f64).to_bits()
    );
    assert!(archive.observations[0]
        .provenance_holds
        .iter()
        .any(|s| s == "missing_or_unknown_score_identity"));
    assert!(!archive.observations[0].labels.quality.reasons.is_empty());
}

#[test]
fn missing_or_protected_role_and_forged_dependency_json_cannot_grant_replay() {
    let mut deps = BTreeMap::new();
    let base = import(INDEX);
    deps.insert(
        base.observations[2].identity.occurrence_id.clone(),
        DependencyEvidence::from_records(
            Some(&json!({"verified":true,"role":"train"})),
            Some(&json!({"complete":true})),
        )
        .unwrap(),
    );
    let archive =
        distilled::import_gp_results_index_complete(INDEX, None, &known(), &deps, None).unwrap();
    assert_eq!(archive.observations[2].original_role, None);
    assert!(archive.observations[2]
        .provenance_holds
        .contains(&"missing_role".into()));
    assert_eq!(
        archive.observations[3].original_role.as_deref(),
        Some("sealed")
    );
    for o in &archive.observations {
        for mask in [
            &o.labels.quality,
            &o.labels.runtime,
            &o.labels.failure,
            &o.labels.policy,
        ] {
            assert!(!mask.reasons.is_empty());
        }
    }
    assert_eq!(
        archive.observations[2]
            .dependencies
            .i04_admission
            .as_ref()
            .unwrap()["verified"],
        true
    );
}

#[test]
fn producer_binding_is_deterministic_and_requires_independently_pinned_closure() {
    let m = manifest(INDEX, spec());
    let pin = sha256_hex(&m);
    assert_eq!(m, manifest(INDEX, spec()));
    let archive = bound(INDEX, &m, &pin);
    assert!(matches!(
        archive.producer_closure,
        ProducerClosure::Bound { .. }
    ));
    assert_eq!(archive.input_sha256, sha256_hex(INDEX));
    assert_eq!(
        archive
            .observations
            .iter()
            .map(|o| &o.identity.occurrence_id)
            .collect::<BTreeSet<_>>()
            .len(),
        10
    );
    assert!(matches!(
        import(INDEX).producer_closure,
        ProducerClosure::Held { .. }
    ));
    let no_pin = distilled::import_gp_results_index_complete(
        INDEX,
        Some(ProducerInput {
            manifest_bytes: &m,
            authenticated_manifest_sha256: None,
        }),
        &known(),
        &BTreeMap::new(),
        None,
    )
    .unwrap();
    assert!(matches!(
        no_pin.producer_closure,
        ProducerClosure::Held { .. }
    ));
    let malformed = [0xffu8, 0x00];
    let archive = bound(INDEX, &malformed, &sha256_hex(&malformed));
    assert!(matches!(
        archive.producer_closure,
        ProducerClosure::Held { .. }
    ));
    assert_eq!(archive.producer_manifest, Some(malformed.to_vec()));
    assert_eq!(archive.observations.len(), 10);
}
