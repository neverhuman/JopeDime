//! Bounded real CPU transport tests and synthetic comparator controls (§1.2/S1).
//!
//! Adapter tests should turn each real surface's response into the same typed
//! `SurfaceCapture`; this module compares the champion, final scores, and the
//! ordered durable event stream without rewriting any of those values.

use jope_api::{
    decimal::DecimalU64, BestAlgorithm, EventBody, FinalScores, PublicEvent, RunSnapshot, RunSpec,
};
use serde_json::{json, Value};

const SURFACES: [Surface; 5] = [
    Surface::BatchCli,
    Surface::Mcp,
    Surface::LoopbackHttp,
    Surface::InProcess,
    Surface::Terminal,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    BatchCli,
    Mcp,
    LoopbackHttp,
    InProcess,
    Terminal,
}

impl Surface {
    fn name(self) -> &'static str {
        match self {
            Self::BatchCli => "batch CLI",
            Self::Mcp => "MCP",
            Self::LoopbackHttp => "loopback HTTP",
            Self::InProcess => "in-process client",
            Self::Terminal => "terminal",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Projection {
    champion: Option<BestAlgorithm>,
    scores: FinalScores,
    events: Vec<PublicEvent>,
}

#[derive(Clone, Debug, PartialEq)]
struct SurfaceCapture {
    surface: Surface,
    projection: Projection,
}

impl SurfaceCapture {
    fn from_observation(
        surface: Surface,
        snapshot: RunSnapshot,
        events: Vec<PublicEvent>,
    ) -> Result<Self, String> {
        if events.iter().any(|event| event.run_id != snapshot.run_id) {
            return Err(format!(
                "{} returned an event for another run",
                surface.name()
            ));
        }
        let finished = match events.last().map(|event| &event.event) {
            Some(EventBody::RunFinished(finished)) => finished,
            _ => {
                return Err(format!(
                    "{} did not end with the durable run.finished event",
                    surface.name()
                ));
            }
        };
        if snapshot.best != finished.champion {
            return Err(format!(
                "{} snapshot champion differs from run.finished",
                surface.name()
            ));
        }
        Ok(Self {
            surface,
            projection: Projection {
                champion: finished.champion.clone(),
                scores: finished.scores.clone(),
                events,
            },
        })
    }
}

fn assert_equivalent(captures: &[SurfaceCapture]) -> Result<(), String> {
    for surface in SURFACES {
        if captures
            .iter()
            .filter(|capture| capture.surface == surface)
            .count()
            != 1
        {
            return Err(format!(
                "expected exactly one capture from {}",
                surface.name()
            ));
        }
    }
    if captures.len() != SURFACES.len() {
        return Err(format!(
            "expected {} surfaces, received {}",
            SURFACES.len(),
            captures.len()
        ));
    }
    let baseline = &captures[0].projection;
    for capture in &captures[1..] {
        if capture.projection != *baseline {
            return Err(format!(
                "{} differs from {} in champion, final scores, or durable events",
                capture.surface.name(),
                captures[0].surface.name()
            ));
        }
    }
    Ok(())
}

// Exact field map accepted by owner ruling rulings_t1900 R2 (§5.6).
// Keep the raw captures; operational provenance is excluded only for this comparison.
fn canonical_events(events: &[PublicEvent]) -> Value {
    let mut values = Vec::new();
    for event in events {
        let mut value = serde_json::to_value(event).expect("public event serializes");
        let envelope = value.as_object_mut().unwrap();
        envelope.remove("run_id");
        envelope.remove("time");
        let body = value.get_mut("body").unwrap();
        if let Some(warnings) = body.get_mut("warnings").and_then(Value::as_array_mut) {
            warnings.retain(|warning| warning["host"] != true);
        }
        if let Some(budget) = body.get_mut("budget_used").and_then(Value::as_object_mut) {
            budget.remove("wall_seconds");
        }
        if body.get("host") != Some(&Value::Bool(true)) {
            values.push(value);
        }
    }
    Value::Array(values)
}

fn assert_independent_equivalent(captures: &[SurfaceCapture]) -> Result<(), String> {
    for surface in SURFACES {
        if captures.iter().filter(|c| c.surface == surface).count() != 1 {
            return Err(format!("expected one independent {} run", surface.name()));
        }
    }
    let ids: std::collections::BTreeSet<_> = captures
        .iter()
        .filter_map(|c| c.projection.events.first().map(|event| &event.run_id))
        .collect();
    if captures.len() != SURFACES.len() || ids.len() != SURFACES.len() {
        return Err("independent surface runs must have five distinct run IDs".into());
    }
    let baseline = &captures[0].projection;
    for capture in &captures[1..] {
        let actual = &capture.projection;
        if actual.champion != baseline.champion
            || actual.scores != baseline.scores
            || canonical_events(&actual.events) != canonical_events(&baseline.events)
        {
            return Err(format!(
                "{} independent R2 projection differs",
                capture.surface.name()
            ));
        }
    }
    Ok(())
}

fn bounded_synthetic_spec() -> RunSpec {
    let mut spec: Value = json!({
        "dataset": "ds_00112233445566778899aabb",
        "target": "y",
        "task": "regression",
        "ignore_columns": [],
        "test_dataset": null,
        "split": {
            "kind": "random",
            "holdout_fraction": 0.2,
            "group_column": null,
            "time_column": null
        },
        "objective": "rmse",
        "effort": "fast",
        "budget": {"evaluations": "4", "wall_seconds": "10"},
        "landmarks": "none",
        "guidance": "heuristic",
        "languages": ["rust"],
        "require_all_languages": true,
        "device": "cpu",
        "seed": "1",
        "dry_run": false,
        "plan_digest": null
    });
    // Round-trip once here so the same validated public RunSpec is what later
    // surface adapters receive, rather than separately assembled request bodies.
    serde_json::from_value(std::mem::take(&mut spec)).expect("valid bounded RunSpec fixture")
}

fn synthetic_observation() -> (RunSnapshot, Vec<PublicEvent>) {
    // Values are deliberately synthetic harness data, not product goldens.
    let snapshot: RunSnapshot = serde_json::from_value(json!({
        "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
        "revision": "2",
        "state": "completed",
        "phase": "finalizing",
        "pending": null,
        "effort": "fast",
        "budget_used": {"evaluations": "4", "wall_seconds": "1.0"},
        "budget_total": {"evaluations": "4", "wall_seconds": "10"},
        "best": {
            "algorithm_id": "alg_0001",
            "metric": "rmse",
            "cv_value": 0.25,
            "model_artifact": null
        },
        "event_watermark": "2",
        "warnings": [],
        "completion_reason": "budget_evaluations"
    }))
    .expect("valid synthetic snapshot");
    let events = vec![
        serde_json::from_value(json!({
            "run_id": snapshot.run_id,
            "sequence": "1",
            "time": "2026-01-01T00:00:00.000Z",
            "kind": "algorithm.accepted",
            "body": {
                "algorithm_id": "alg_0001",
                "metric": "rmse",
                "cv_value": 0.25,
                "previous_value": null,
                "summary": "synthetic fixture champion",
                "model_artifact": null,
                "excerpt": "synthetic"
            }
        }))
        .expect("valid synthetic accepted event"),
        serde_json::from_value(json!({
            "run_id": snapshot.run_id,
            "sequence": "2",
            "time": "2026-01-01T00:00:00.001Z",
            "kind": "run.finished",
            "body": {
                "completion_reason": "budget_evaluations",
                "champion": snapshot.best,
                "scores": {"cv": 0.25, "holdout": null, "test": null},
                "budget_used": {"evaluations": "4", "wall_seconds": "1.0"},
                "nondeterminism": {"count": "0", "experiment_ids": []}
            }
        }))
        .expect("valid synthetic finished event"),
    ];
    (snapshot, events)
}

#[derive(Default)]
struct FakeEngine {
    requests: Vec<Value>,
}

impl FakeEngine {
    fn execute(&mut self, spec: &RunSpec) -> (RunSnapshot, Vec<PublicEvent>) {
        // Retain each request so the harness can prove that every fake surface
        // submits the same bounded, typed input to the engine boundary.
        self.requests
            .push(serde_json::to_value(spec).expect("RunSpec serializes"));
        // Engine-free synthetic output: never a product score or golden result.
        synthetic_observation()
    }
}

fn capture_from_fake_engine(
    engine: &mut FakeEngine,
    surface: Surface,
    spec: &RunSpec,
) -> SurfaceCapture {
    let (snapshot, events) = engine.execute(spec);
    SurfaceCapture::from_observation(surface, snapshot, events)
        .expect("fake engine returns a coherent capture")
}

#[test]
fn bounded_run_spec_is_a_single_typed_request_for_every_surface() {
    let spec = bounded_synthetic_spec();
    assert_eq!(spec.budget.evaluations, Some(DecimalU64(4)));
    assert_eq!(spec.budget.wall_seconds, Some(DecimalU64(10)));
    let encoded = serde_json::to_value(spec).unwrap();
    assert_eq!(encoded["dataset"], "ds_00112233445566778899aabb");
}

#[test]
fn synthetic_surface_capture_harness_accepts_equal_projections_and_rejects_drift() {
    let (snapshot, events) = synthetic_observation();
    let captures = SURFACES
        .into_iter()
        .map(|surface| {
            SurfaceCapture::from_observation(surface, snapshot.clone(), events.clone()).unwrap()
        })
        .collect::<Vec<_>>();
    assert_equivalent(&captures).unwrap();
    assert!(assert_independent_equivalent(&captures).is_err());
    let independent: Vec<_> = captures
        .iter()
        .enumerate()
        .map(|(i, capture)| {
            let mut capture = capture.clone();
            let run: jope_api::ids::RunId =
                format!("run_01JBD8Q6N4V2T5Y8K3M7P9R1S{i}").parse().unwrap();
            for event in &mut capture.projection.events {
                event.run_id = run.clone();
                if let EventBody::RunFinished(finished) = &mut event.event {
                    finished.budget_used.wall_seconds = (i + 2).to_string();
                }
            }
            capture
        })
        .collect();
    assert_independent_equivalent(&independent).unwrap();
    let mut divergent_independent = independent;
    if let EventBody::RunFinished(finished) = &mut divergent_independent[2]
        .projection
        .events
        .last_mut()
        .unwrap()
        .event
    {
        finished.budget_used.evaluations = DecimalU64(99);
    }
    assert!(assert_independent_equivalent(&divergent_independent).is_err());

    let mut divergent = captures.clone();
    if let EventBody::RunFinished(finished) =
        &mut divergent[2].projection.events.last_mut().unwrap().event
    {
        finished.scores.cv = Some(0.5);
    } else {
        unreachable!("fixture ends with run.finished");
    }
    assert_eq!(
        assert_equivalent(&divergent).unwrap_err(),
        "loopback HTTP differs from batch CLI in champion, final scores, or durable events"
    );
}

#[test]
fn terminal_capture_rejects_durable_events_after_run_finished() {
    let (snapshot, mut events) = synthetic_observation();
    events.push(
        serde_json::from_value(json!({
            "run_id": snapshot.run_id,
            "sequence": "3",
            "time": "2026-01-01T00:00:00.002Z",
            "kind": "run.state",
            "body": {
                "state": "completed",
                "phase": "finalizing",
                "pending": null,
                "reason": null
            }
        }))
        .expect("valid synthetic post-finish event"),
    );

    assert_eq!(
        SurfaceCapture::from_observation(Surface::Terminal, snapshot, events).unwrap_err(),
        "terminal did not end with the durable run.finished event"
    );
}

#[test]
fn fake_engine_receives_one_bounded_spec_for_each_surface_capture() {
    let spec = bounded_synthetic_spec();
    let expected_request = serde_json::to_value(&spec).unwrap();
    let mut engine = FakeEngine::default();
    let captures = SURFACES
        .into_iter()
        .map(|surface| capture_from_fake_engine(&mut engine, surface, &spec))
        .collect::<Vec<_>>();

    assert_eq!(engine.requests, vec![expected_request; SURFACES.len()]);
    assert_equivalent(&captures).unwrap();
}

#[cfg(all(
    feature = "engine",
    feature = "mcp",
    any(target_os = "linux", target_os = "macos")
))]
mod real;

#[cfg(all(
    feature = "engine",
    feature = "mcp",
    any(target_os = "linux", target_os = "macos")
))]
mod qualification;

#[cfg(all(
    feature = "engine",
    feature = "mcp",
    any(target_os = "linux", target_os = "macos")
))]
#[test]
fn all_product_surfaces_share_the_bounded_run_projection() -> anyhow::Result<()> {
    if let Some(home) = std::env::var_os("JOPE_INT01_QUALIFY_HOME") {
        return qualification::from_env(std::path::Path::new(&home));
    }
    real::shared_run_surfaces()
}

#[cfg(all(
    feature = "engine",
    feature = "mcp",
    any(target_os = "linux", target_os = "macos")
))]
#[test]
fn batch_fallback_executes_a_real_bounded_cpu_run_without_a_daemon() -> anyhow::Result<()> {
    real::batch_fallback()
}
