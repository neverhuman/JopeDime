use jope_learning::canon::sha256_hex;
use jope_learning::replay::index_producer::{
    self, HistoricalFixture, HistoricalStatus, ProducerClosure, ProducerInput, ProducerSpec,
};
use jope_learning::replay::observation::{
    DependencyEvidence, Disposition, DurationObservation, Measurement, ObservationArchive,
    ObservedOutcome,
};
use jope_learning::retention::{
    distilled,
    parquet::{
        self, Cell, Codec, ColumnSpec, ColumnType, FileReader, StatisticValue, Table, Values,
        WriteOptions,
    },
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const INDEX: &[u8] = include_bytes!("fixtures/observation-import-v2/index.json");
const SPEC: &[u8] = include_bytes!("fixtures/observation-import-v2/producer-spec.json");
fn known() -> BTreeSet<String> {
    BTreeSet::from(["validation_rmse".into()])
}
fn spec() -> ProducerSpec {
    serde_json::from_slice(SPEC).unwrap()
}
fn import(bytes: &[u8]) -> ObservationArchive {
    distilled::import_gp_results_index_complete(bytes, None, &known(), &BTreeMap::new(), None)
        .unwrap()
}
fn manifest(bytes: &[u8], producer: ProducerSpec) -> Vec<u8> {
    serde_json::to_vec(&index_producer::produce_manifest(bytes, producer).unwrap()).unwrap()
}
fn bound(bytes: &[u8], manifest: &[u8], pin: &str) -> ObservationArchive {
    distilled::import_gp_results_index_complete(
        bytes,
        Some(ProducerInput {
            manifest_bytes: manifest,
            authenticated_manifest_sha256: Some(pin),
        }),
        &known(),
        &BTreeMap::new(),
        None,
    )
    .unwrap()
}
fn opts() -> WriteOptions {
    WriteOptions {
        codec: Codec::Zstd,
        row_group_rows: 3,
        page_rows: 1,
        dictionary: true,
    }
}

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
        "fixtures/observation-import-v2/score-bits.json"
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

#[test]
fn typed_statistics_use_numeric_and_unsigned_byte_order_and_exclude_nonfinite_bounds() {
    let table = Table {
        columns: [
            ("n", ColumnType::Int64),
            ("s", ColumnType::Utf8),
            ("b", ColumnType::Boolean),
            ("d", ColumnType::Double),
        ]
        .into_iter()
        .map(|(name, ty)| ColumnSpec {
            name: name.into(),
            ty,
            required: false,
        })
        .collect(),
        rows: vec![
            vec![
                Cell::I64(-10),
                Cell::Str("é".into()),
                Cell::Bool(true),
                Cell::F64(-0.0),
            ],
            vec![
                Cell::I64(2),
                Cell::Str("z".into()),
                Cell::Bool(false),
                Cell::F64(0.0),
            ],
            vec![Cell::Null, Cell::Null, Cell::Null, Cell::Null],
        ],
        metadata: vec![],
    };
    let opts = WriteOptions {
        row_group_rows: 0,
        ..opts()
    };
    let bytes = parquet::write_with_statistics(&table, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let n = reader.statistics(0, 0).unwrap().unwrap();
    assert_eq!(n.null_count, Some(1));
    assert_eq!(n.min, Some(StatisticValue::I64(-10)));
    assert_eq!(n.max, Some(StatisticValue::I64(2)));
    assert_eq!(
        reader.statistics(0, 1).unwrap().unwrap().min,
        Some(StatisticValue::Bytes(b"z".to_vec()))
    );
    assert_eq!(
        reader.statistics(0, 2).unwrap().unwrap().min,
        Some(StatisticValue::Bool(false))
    );
    match reader.statistics(0, 3).unwrap().unwrap().min {
        Some(StatisticValue::F64(v)) => assert_eq!(v.to_bits(), (-0.0f64).to_bits()),
        _ => panic!("missing double bound"),
    }
    let mut zero = table.clone();
    zero.rows.truncate(1);
    zero.rows[0][3] = Cell::F64(0.0);
    let bytes = parquet::write_with_statistics(&zero, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let stats = reader.statistics(0, 3).unwrap().unwrap();
    assert_eq!(stats.min_is_exact, Some(false));
    assert_eq!(stats.max_is_exact, Some(true));
    assert_eq!(stats.nan_count, Some(0));
    if let Some(StatisticValue::F64(v)) = stats.min {
        assert_eq!(v.to_bits(), (-0.0f64).to_bits());
    } else {
        panic!("zero lower bound");
    }
    assert!(
        matches!(reader.read_column(0,3).unwrap().0,Values::F64(v) if v[0].unwrap().to_bits()==0.0f64.to_bits())
    );
    let mut nonfinite = table;
    nonfinite.rows[0][3] = Cell::F64(f64::NAN);
    let bytes = parquet::write_with_statistics(&nonfinite, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let stats = reader.statistics(0, 3).unwrap().unwrap();
    assert_eq!(stats.null_count, Some(1));
    assert_eq!(stats.nan_count, Some(1));
    assert_eq!(stats.min, None);
    assert_eq!(stats.max, None);
    assert!(
        matches!(reader.read_column(0,3).unwrap().0,Values::F64(v) if v[0].is_some_and(f64::is_nan))
    );
}

#[test]
fn fixed_fixture_manifest_and_source_root_are_authenticated_as_fixture_only() {
    let bytes = include_bytes!("fixtures/observation-import-v2/producer-manifest.json");
    let pin = include_str!("fixtures/observation-import-v2/producer-manifest.sha256").trim();
    assert_eq!(sha256_hex(bytes), pin);
    assert_eq!(
        spec().input_roots[0].sha256,
        sha256_hex(include_bytes!(
            "fixtures/observation-import-v2/producer-source.txt"
        ))
    );
    let archive = bound(INDEX, bytes, pin);
    assert!(matches!(
        archive.producer_closure,
        ProducerClosure::Bound { .. }
    ));
    assert!(matches!(archive.historical, HistoricalStatus::Held { .. }));
    let historical: Value = serde_json::from_str(include_str!(
        "fixtures/observation-import-v2/historical-status.json"
    ))
    .unwrap();
    assert_eq!(historical["historical_input"], "UNKNOWN");
    assert_eq!(historical["current_estate"], "UNKNOWN");
}

#[test]
fn malformed_or_ambiguous_index_is_refused_without_a_completeness_claim() {
    for bytes in [
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"outcome":"failed"}],"runs":[]}"#
            .as_slice(),
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"score":1.0,"score":2.0}]}"#,
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"score":1e999}]}"#,
        br#"{"schema":"jopedime.gp-results-index/v1"}"#,
        br#"{"schema":"other","runs":[]}"#,
    ] {
        assert!(distilled::import_gp_results_index_complete(
            bytes,
            None,
            &known(),
            &BTreeMap::new(),
            None
        )
        .is_err());
        assert!(index_producer::produce_manifest(bytes, spec()).is_err());
    }
}
