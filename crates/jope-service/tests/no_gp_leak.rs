//! T2 pre-stage sentinel harness.
//!
//! These fixtures exercise the existing public projection and wire shapes with synthetic
//! internal fields. They are not a full discovery run. The PTY case remains gated on TUI-09.

use jope_api::{decimal::DecimalU64, ids::RunId, PublicEvent};
use jope_service::public::{
    phrase::{CatalogEntry, EventDetails, Template},
    project::{AdmittedProposal, EvaluationOrdinal, ProblemDigest, Projector},
};

const SENTINELS: [&str; 4] = [
    "T2_DSL_SENTINEL_8f27",
    "T2_OPERATOR_SENTINEL_19ac",
    "T2_LINEAGE_SENTINEL_43de",
    "T2_ROUND_SENTINEL_71b0",
];

#[derive(Debug)]
struct InternalFixture {
    dsl: String,
    operator: String,
    lineage: String,
    round: String,
}

impl InternalFixture {
    fn synthetic() -> Self {
        Self {
            dsl: SENTINELS[0].into(),
            operator: SENTINELS[1].into(),
            lineage: SENTINELS[2].into(),
            round: SENTINELS[3].into(),
        }
    }

    fn opaque_round_ordinal(&self) -> EvaluationOrdinal {
        // The public projection accepts only an ordinal and emits a digest-derived ID.
        EvaluationOrdinal(self.round.len() as u64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Http,
    Sse,
    Mcp,
    CliStdout,
    CliStderr,
    Terminal,
    Error,
    ExportManifest,
    Log,
}

#[derive(Debug)]
struct Leak {
    surface: Surface,
    sentinel: &'static str,
}

fn scan_surfaces(outputs: &[(Surface, String)]) -> Result<(), Leak> {
    for (surface, output) in outputs {
        for sentinel in SENTINELS {
            if output.contains(sentinel) {
                return Err(Leak {
                    surface: *surface,
                    sentinel,
                });
            }
        }
    }
    Ok(())
}

fn projected_event(internal: &InternalFixture) -> PublicEvent {
    // Internal DSL/operator/lineage values are deliberately not inputs to this public shape.
    let run_id = RunId::new();
    let projector = Projector::new(run_id.clone(), ProblemDigest([3; 32]));
    let proposal = AdmittedProposal {
        ordinal: internal.opaque_round_ordinal(),
        template: Template::StageInsert,
        event_digest: [9; 32],
        entry: CatalogEntry {
            public_name: "log scaling",
            purpose: "reduce skew in numeric features",
        },
        details: EventDetails::default(),
    };
    PublicEvent {
        run_id,
        sequence: DecimalU64(1),
        time: "2026-10-30T00:00:00Z".into(),
        event: projector.experiment_started(&proposal).unwrap(),
    }
}

fn synthetic_surface_outputs(event: &PublicEvent) -> Vec<(Surface, String)> {
    let wire = serde_json::to_string(event).unwrap();
    let mcp = serde_json::json!({
        "content": [{"type": "text", "text": wire}],
        "isError": false,
    });
    let manifest = serde_json::json!({
        "schema_version": 1,
        "model_object_digest": "0".repeat(64),
        "retains_training_data": false,
        "entries": {"predictor.py": "1".repeat(64)},
    });

    vec![
        (Surface::Http, wire.clone()),
        (Surface::Sse, format!("data: {wire}\n\n")),
        (Surface::Mcp, mcp.to_string()),
        (Surface::CliStdout, format!("{wire}\n")),
        (Surface::CliStderr, String::new()),
        (Surface::Terminal, format!("terminal transcript: {wire}")),
        (Surface::Error, "The request could not be completed.".into()),
        (Surface::ExportManifest, manifest.to_string()),
        (Surface::Log, format!("public event: {wire}")),
    ]
}

#[test]
fn synthetic_internal_fields_stay_out_of_current_public_wire_surfaces() {
    let internal = InternalFixture::synthetic();
    assert_eq!(internal.dsl, SENTINELS[0]);
    assert_eq!(internal.operator, SENTINELS[1]);
    assert_eq!(internal.lineage, SENTINELS[2]);

    let outputs = synthetic_surface_outputs(&projected_event(&internal));
    scan_surfaces(&outputs).unwrap();
}

#[test]
fn scanner_reports_a_deliberately_leaking_surface() {
    let outputs = vec![(Surface::Mcp, format!("tool output: {}", SENTINELS[1]))];
    let leak = scan_surfaces(&outputs).unwrap_err();
    assert_eq!(leak.surface, Surface::Mcp);
    assert_eq!(leak.sentinel, SENTINELS[1]);
}

#[test]
fn scanner_rejects_each_sentinel_on_every_public_surface() {
    let surfaces = [
        Surface::Http,
        Surface::Sse,
        Surface::Mcp,
        Surface::CliStdout,
        Surface::CliStderr,
        Surface::Terminal,
        Surface::Error,
        Surface::ExportManifest,
        Surface::Log,
    ];

    for surface in surfaces {
        for sentinel in SENTINELS {
            let output = format!("leaking {surface:?} capture: {sentinel}");
            let leak = scan_surfaces(&[(surface, output)]).unwrap_err();
            assert_eq!(leak.surface, surface);
            assert_eq!(leak.sentinel, sentinel);
        }
    }
}

#[test]
#[ignore = "terminal transcript scan requires the TUI-09 PTY surface"]
fn terminal_transcript_contains_no_internal_sentinels() {
    // TUI-09 will replace this pre-stage placeholder with captured PTY transcript bytes.
    let transcript = String::new();
    scan_surfaces(&[(Surface::CliStdout, transcript)]).unwrap();
}
