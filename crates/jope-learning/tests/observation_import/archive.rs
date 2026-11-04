#[test]
fn input_tampering_and_missing_source_command_or_roots_hold_all_rows() {
    let m = manifest(INDEX, spec());
    let pin = sha256_hex(&m);
    let mut index: Value = serde_json::from_slice(INDEX).unwrap();
    index["runs"][0]["score"] = json!(0.7);
    let altered = serde_json::to_vec(&index).unwrap();
    for archive in [bound(&altered, &m, &pin), bound(INDEX, &m, &"0".repeat(64))] {
        assert_eq!(archive.observations.len(), 10);
        assert!(matches!(
            archive.producer_closure,
            ProducerClosure::Held { .. }
        ));
    }
    for field in ["source_commit", "command", "input_roots"] {
        let mut s = spec();
        match field {
            "source_commit" => s.source_commit = None,
            "command" => s.command = None,
            _ => s.input_roots.clear(),
        }
        let m = manifest(INDEX, s);
        let archive = bound(INDEX, &m, &sha256_hex(&m));
        assert_eq!(archive.observations.len(), 10);
        assert!(matches!(
            archive.producer_closure,
            ProducerClosure::Held { .. }
        ));
    }
}

#[test]
fn duplicate_delivery_keeps_multiplicity_and_conflicts_hold_every_participant() {
    let mut index: Value = serde_json::from_slice(INDEX).unwrap();
    let duplicate = index["runs"][0].clone();
    index["runs"]
        .as_array_mut()
        .unwrap()
        .push(duplicate.clone());
    let archive = import(&serde_json::to_vec(&index).unwrap());
    assert_eq!(archive.observations.len(), 11);
    assert_eq!(
        archive.observations[10].disposition,
        Disposition::DuplicateDelivery
    );
    assert_ne!(
        archive.observations[0].identity.occurrence_id,
        archive.observations[10].identity.occurrence_id
    );
    let mut conflict = duplicate;
    conflict["score"] = json!(0.2);
    index["runs"].as_array_mut().unwrap().push(conflict);
    let archive = import(&serde_json::to_vec(&index).unwrap());
    assert_eq!(archive.primary_counts["conflict"], 3);
    assert_eq!(archive.primary_counts.values().sum::<usize>(), 12);
    for i in [0, 10, 11] {
        assert_eq!(archive.observations[i].disposition, Disposition::Conflict);
        assert_eq!(archive.observations[i].related_occurrences.len(), 2);
    }
}

#[test]
fn all_unscored_and_empty_inputs_are_honest_archives() {
    let bytes=serde_json::to_vec(&json!({"schema":"jopedime.gp-results-index/v1","runs":[{"outcome":"failed"},{"outcome":"timeout"},{}]})).unwrap();
    let archive = import(&bytes);
    assert_eq!(archive.observations.len(), 3);
    assert!(!archive.primary_counts.contains_key("scored"));
    assert!(
        distilled::legacy_scored_projection(&archive, distilled::SplitRole::Unknown, &known())
            .is_err()
    );
    let bytes = br#"{"schema":"jopedime.gp-results-index/v1","runs":[]}"#;
    let archive = import(bytes);
    assert!(archive.observations.is_empty());
    assert!(archive.primary_counts.is_empty());
    assert_eq!(
        FileReader::open(&archive.columnar(&opts()).unwrap())
            .unwrap()
            .num_rows,
        0
    );
}

#[test]
fn timeout_is_censored_and_malformed_scores_keep_their_original_evidence() {
    let archive = import(INDEX);
    assert_eq!(
        archive.observations[3].duration,
        DurationObservation::RightCensored { lower_bound_ms: 32 }
    );
    assert_eq!(archive.observations[3].measurement, Measurement::Missing);
    assert!(!archive.observations[3].labels.quality.observed);
    assert!(archive.observations[3].labels.runtime.observed);
    assert!(matches!(
        archive.observations[8].measurement,
        Measurement::Rejected { .. }
    ));
    assert_eq!(archive.observations[8].original_row["score"], "NaN");
    assert_eq!(
        archive.observations[8].failure_reason.as_deref(),
        Some("fit_failed")
    );
}

#[test]
fn historical_18380_is_explicitly_held_without_original_authenticated_fixture() {
    let archive = import(INDEX);
    assert!(matches!(
        archive.historical,
        HistoricalStatus::Held {
            dated_target: 18380,
            ..
        }
    ));
    let claim = HistoricalFixture {
        date: "2026-10-17".into(),
        expected_rows: 18380,
        input_sha256: archive.input_sha256.clone(),
        manifest_sha256: "f".repeat(64),
        expected_dispositions: archive.primary_counts.clone(),
    };
    let a = distilled::import_gp_results_index_complete(
        INDEX,
        None,
        &known(),
        &BTreeMap::new(),
        Some(&claim),
    )
    .unwrap();
    assert!(matches!(a.historical, HistoricalStatus::Held { .. }));
    assert_eq!(a.observations.len(), 10);
}

#[test]
fn archive_multigroup_zstd_pages_preserve_scores_nulls_and_observation_masks() {
    let archive = import(INDEX);
    let bytes = archive.columnar(&opts()).unwrap();
    if let Some(path) = std::env::var_os("JOPE_OBSERVATION_PARQUET_SAMPLE") {
        std::fs::write(path, &bytes).unwrap();
    }
    let reader = FileReader::open(&bytes).unwrap();
    assert_eq!(reader.row_groups.len(), 4);
    assert_eq!(reader.num_rows, 10);
    let mut scores = Vec::new();
    let mut observations = Vec::new();
    for g in 0..reader.row_groups.len() {
        let (v, pages) = reader.read_column(g, 1).unwrap();
        assert_eq!(pages.codecs, BTreeSet::from([6]));
        if let Values::F64(v) = v {
            scores.extend(v);
        } else {
            panic!("score type");
        }
        let (v, _) = reader.read_column(g, 2).unwrap();
        if let Values::Bytes(v) = v {
            observations.extend(v.into_iter().map(|b| {
                serde_json::from_slice::<jope_learning::replay::observation::Observation>(
                    &b.unwrap(),
                )
                .unwrap()
            }));
        } else {
            panic!("observation type");
        }
        let stats = reader.statistics(g, 1).unwrap().unwrap();
        assert_eq!(stats.value_count, reader.row_groups[g].rows);
    }
    assert_eq!(
        scores
            .iter()
            .map(|x| x.map(f64::to_bits))
            .collect::<Vec<_>>(),
        archive
            .observations
            .iter()
            .map(|o| o.measurement.value().map(f64::to_bits))
            .collect::<Vec<_>>()
    );
    assert_eq!(observations, archive.observations);
    assert_eq!(
        reader.statistics(1, 1).unwrap().unwrap().null_count,
        Some(3)
    );
    assert_eq!(reader.statistics(1, 1).unwrap().unwrap().min, None);
    assert_eq!(
        reader.statistics(0, 1).unwrap().unwrap().min,
        Some(StatisticValue::F64(0.59))
    );
}
