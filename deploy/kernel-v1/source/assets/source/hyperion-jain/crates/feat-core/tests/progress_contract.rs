//! Drift guard: the Rust `progress::Event`, the committed JSON Schema, and the golden fixture under
//! `contracts/` must stay in lockstep. The web (`apps/web/src/protocol.contract.test.ts`) and Python
//! (`python/ai-service/tests/test_contract.py`) suites validate the SAME fixture against the SAME
//! schema, so a drift in any single language turns this lane (or its peers) red.

use feat_core::progress::{
    EnrichedFeature, Event, GpuBadge, HyperionConcernRange, HyperionConcernRow, HyperionPreview,
    HyperionPreviewPoint, WinnerFeature, EVENT_SCHEMA_VERSION,
};
use std::collections::BTreeSet;

const SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/progress-event.schema.json"
));
const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/progress-events.jsonl"
));

fn schema_property_keys() -> BTreeSet<String> {
    let v: serde_json::Value = serde_json::from_str(SCHEMA).expect("schema is valid JSON");
    v["properties"]
        .as_object()
        .expect("schema has properties")
        .keys()
        .cloned()
        .collect()
}

fn fixture_lines() -> impl Iterator<Item = &'static str> {
    FIXTURE.lines().filter(|l| !l.trim().is_empty())
}

/// A fully-populated Event (every field set) — its serialized key set must equal the schema's
/// property set. Adding a field to `Event` without updating the schema (or vice versa) fails here.
#[test]
fn event_keys_match_committed_schema() {
    let e = Event {
        phase: 1,
        event: "gen".into(),
        t: 1.0,
        survivors: Some(1),
        admitted: Some(1),
        best_signal: Some(0.1),
        best_maxcorr: Some(0.2),
        gen: Some(0),
        gens: Some(4),
        blocks_total: Some(5),
        done: Some(1),
        done_frac: Some(0.2),
        top: Some(vec![(1, 0.5)]),
        best_val_auc: Some(0.7),
        best_holdout_auc: Some(0.6),
        mean_val_auc: Some(0.65),
        best_size: Some(10),
        mean_size: Some(8),
        eta: Some(3.0),
        model: Some("m".into()),
        accepted_features: Some(2),
        selected_features: Some(3),
        final_features: Some(4),
        winners: Some(vec![WinnerFeature {
            formula: "f".into(),
            lift: 0.1,
            stability: 0.2,
            canary: false,
        }]),
        metric: Some("auc".into()),
        rows: Some(100),
        cols: Some(12),
        output_dim: Some(128),
        hyperion_status: Some("used".into()),
        hyperion_engine: Some("v7".into()),
        hyperion_device: Some("cpu".into()),
        hyperion_preview: Some(HyperionPreview {
            rows: 100,
            threshold_z: 3.5,
            points: vec![HyperionPreviewPoint {
                row: 42,
                score: 12.5,
            }],
            top_rows: vec![HyperionConcernRow {
                row: 42,
                score: 12.5,
                z: 4.1,
            }],
            concern_ranges: vec![HyperionConcernRange {
                start: 41,
                end: 43,
                max_score: 12.5,
                max_z: 4.1,
            }],
        }),
        folds: Some(vec![0.85, 0.86, 0.84]),
        baseline_auc: Some(0.80),
        top_features: Some(vec![EnrichedFeature {
            name: "log(spend)/hyperion::v007".into(),
            importance: 0.33,
            source: "gp".into(),
            refs_hyperion: true,
            refs_canary: true,
        }]),
        best_code: Some("import math\n".into()),
        genome: Some("(pipe X (robust_scale) (ridge))".into()),
        evals: Some(412),
        evals_per_sec: Some(180.5),
        tokens_total: Some(1_250_000.0),
        tokens_per_sec: Some(540_000.0),
        pop_fitness: Some(vec![0.71, 0.69, 0.42]),
        note: Some("phase skipped".into()),
        gpu: Some(GpuBadge {
            available: true,
            free_mb: Some(4096),
            total_mb: Some(8192),
        }),
        eval_score: Some(0.73),
        v: EVENT_SCHEMA_VERSION,
    };
    let val = serde_json::to_value(&e).unwrap();
    let keys: BTreeSet<String> = val.as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        keys,
        schema_property_keys(),
        "Event fields drifted from contracts/progress-event.schema.json"
    );
}

#[test]
fn schema_version_matches_code_and_fixtures() {
    let schema: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
    assert_eq!(
        schema["version"].as_u64().unwrap() as u32,
        EVENT_SCHEMA_VERSION,
        "schema `version` != EVENT_SCHEMA_VERSION"
    );
    for line in fixture_lines() {
        let e: Event = serde_json::from_str(line).expect("fixture line deserializes into Event");
        assert_eq!(e.v, EVENT_SCHEMA_VERSION, "fixture has stale v: {line}");
    }
}

#[test]
fn fixtures_round_trip_and_use_only_schema_keys() {
    let props = schema_property_keys();
    for line in fixture_lines() {
        let raw: serde_json::Value = serde_json::from_str(line).unwrap();
        for k in raw.as_object().unwrap().keys() {
            assert!(
                props.contains(k),
                "fixture key '{k}' not declared in schema"
            );
        }
        // And it must deserialize into the Rust type without loss of the required fields.
        let _e: Event = serde_json::from_str(line).unwrap();
    }
}
