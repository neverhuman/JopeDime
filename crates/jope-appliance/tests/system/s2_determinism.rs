//! INT-02 S2: compare independent CPU reruns across a persisted-event reconnect.

use super::{durable, PersistedEventTransport};
use jope_api::{decimal::DecimalU64, EventBody, PublicEvent, RunSnapshot, RunState};
use jope_client::{EventStream, Transport};
use jope_service::{dataset::register, engine::InProcessEngine, events::EventLog, store::Store};
use serde_json::Value;
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");
const TRAIN_SHA256: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";
const SEED: &str = "73";

fn training_csv() -> Vec<u8> {
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(jope_runtime::sha256_hex(raw), TRAIN_SHA256);
    let mut names: Vec<_> = (0..21).map(|index| format!("x{index}")).collect();
    names.push("target".into());
    let mut csv = names.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    csv.into_bytes()
}

fn run_once() -> (tempfile::TempDir, RunSnapshot, Vec<PublicEvent>) {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int02-s2");
    fs::create_dir_all(&target).unwrap();
    let home = tempfile::tempdir_in(target).unwrap();
    let pinned_bin = home.path().join("jopedime");
    fs::copy(BIN, &pinned_bin).unwrap();

    let csv = training_csv();
    let engine = InProcessEngine::open_with_worker(home.path(), &pinned_bin).unwrap();
    let dataset =
        register::register(engine.store(), Cursor::new(csv), &Default::default()).unwrap();
    drop(engine);

    let output = Command::new(&pinned_bin)
        .env("JOPEDIME_HOME", home.path())
        .env_remove("JOPEDIME_ENDPOINT")
        .args([
            "run",
            "--json",
            "--wait",
            "1",
            "--dataset",
            &dataset.id.to_string(),
            "--target",
            "target",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"4","wall_seconds":"30"}"#,
            "--landmarks",
            "none",
            "--guidance",
            "auto",
            "--language",
            "rust",
            "--require-all-languages",
            "--device",
            "cpu",
            "--seed",
            SEED,
            "--idempotency-key",
            "int02-s2-deterministic-rerun",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "bounded CPU rerun failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshot: RunSnapshot = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(snapshot.state, RunState::Completed);
    assert!(
        snapshot.best.is_some(),
        "rerun completed without a champion"
    );

    let events = {
        let store = Arc::new(Store::open(home.path()).unwrap());
        EventLog::new(store)
            .page(&snapshot.run_id, 0, 256, &[])
            .unwrap()
            .unwrap()
            .events
    };
    assert!(
        events.len() > 2,
        "bounded run emitted too few durable events"
    );
    assert!(matches!(
        &events.last().unwrap().event,
        EventBody::RunFinished(_)
    ));
    (home, snapshot, events)
}

fn deterministic_projection(events: &[PublicEvent]) -> Vec<Value> {
    events
        .iter()
        .map(|event| {
            let mut value = serde_json::to_value(event).unwrap();
            let object = value.as_object_mut().unwrap();
            object.remove("run_id");
            object.remove("time");
            if let Some(budget) = object
                .get_mut("body")
                .and_then(Value::as_object_mut)
                .and_then(|body| body.get_mut("budget_used"))
                .and_then(Value::as_object_mut)
            {
                // Wall-clock accounting is outside S2's seeded output identity.
                budget.remove("wall_seconds");
            }
            value
        })
        .collect()
}

fn without_model_artifact_ids(mut projection: Vec<Value>) -> Vec<Value> {
    fn strip(value: &mut Value) {
        match value {
            Value::Object(object) => {
                object.remove("model_artifact");
                for value in object.values_mut() {
                    strip(value);
                }
            }
            Value::Array(values) => values.iter_mut().for_each(strip),
            _ => {}
        }
    }
    projection.iter_mut().for_each(strip);
    projection
}

#[test]
fn s2_seeded_cpu_rerun_reports_projection_after_persisted_stream_reconnect() {
    let (first_home, first_snapshot, first_events) = run_once();
    let (_second_home, second_snapshot, second_events) = run_once();
    assert_ne!(first_snapshot.run_id, second_snapshot.run_id);
    let first_best = first_snapshot.best.as_ref().unwrap();
    let second_best = second_snapshot.best.as_ref().unwrap();

    let cut = first_events.len() / 2;
    assert!(cut > 0 && cut < first_events.len());
    let transport = PersistedEventTransport {
        home: first_home.path().to_path_buf(),
        run: first_snapshot.run_id.clone(),
        initial_cut: cut,
        cursors: Mutex::new(vec![]),
    };
    let mut stream = EventStream::new(&transport, first_snapshot.run_id.clone(), None).unwrap();
    let mut observed = stream.by_ref().map(durable).collect::<Vec<_>>();
    assert_eq!(observed, first_events[..cut]);
    let cursor = stream.last_event_id();
    assert_eq!(cursor, Some(DecimalU64(cut as u64)));
    drop(stream);

    let stream = EventStream::new(&transport, first_snapshot.run_id.clone(), cursor).unwrap();
    observed.extend(stream.map(durable));
    assert_eq!(
        observed, first_events,
        "persisted reconnect changed the run"
    );
    assert_eq!(
        *transport.cursors.lock().unwrap(),
        vec![None, cursor],
        "reconnect cursor did not resume the committed watermark"
    );
    let page = transport
        .run_events(&first_snapshot.run_id, DecimalU64(0), 256)
        .unwrap();
    assert_eq!(page.events, first_events);

    let first_projection = deterministic_projection(&observed);
    let second_projection = deterministic_projection(&second_events);
    let projection_matches = first_projection == second_projection;
    let projection_without_model_matches = without_model_artifact_ids(first_projection.clone())
        == without_model_artifact_ids(second_projection.clone());
    let differing_events = first_projection
        .iter()
        .zip(&second_projection)
        .filter(|(first, second)| first != second)
        .map(|(event, _)| serde_json::json!({"sequence": event["sequence"], "kind": event["kind"]}))
        .collect::<Vec<_>>();
    let first_encoded = serde_json::to_vec(&first_projection).unwrap();
    let second_encoded = serde_json::to_vec(&second_projection).unwrap();
    println!(
        "INT02_S2_RERUN={{\"seed\":{SEED},\"event_count\":{},\"reconnect_cursor\":{},\"reconnect_stream_matches_journal\":true,\"first_projection_sha256\":\"{}\",\"second_projection_sha256\":\"{}\",\"projection_matches\":{},\"projection_matches_without_model_artifact_ids\":{},\"champion_algorithm_matches\":{},\"champion_metric_matches\":{},\"champion_cv_matches\":{},\"artifact_identity_matches\":{},\"differing_events\":{},\"s2_qualified\":{}}}",
        first_projection.len(),
        cursor.unwrap().0,
        jope_runtime::sha256_hex(&first_encoded),
        jope_runtime::sha256_hex(&second_encoded),
        projection_matches,
        projection_without_model_matches,
        first_best.algorithm_id == second_best.algorithm_id,
        first_best.metric == second_best.metric,
        first_best.cv_value == second_best.cv_value,
        first_best.model_artifact == second_best.model_artifact,
        serde_json::to_string(&differing_events).unwrap(),
        projection_matches
    );
}
