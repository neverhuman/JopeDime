//! Executable delivery contracts: retained text, cursors, corruption and MCP framing.
use jope_narrator::{Claim, Cursor, Envelope, FactJournal, LineageFact, Links, Narration, SCHEMA};
use jope_observatory::narration::{self, Index, Limits, MAX_PAGE_BYTES};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Cursor as Input, Write},
    path::{Path, PathBuf},
};

fn fixture_dir() -> tempfile::TempDir {
    let exe = std::env::current_exe().unwrap();
    let root = exe
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("narration-test-fixtures");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn record(run: &str, seq: u64, previous: &str) -> Envelope {
    let evidence = format!("{run}:{seq}");
    let mut event = Envelope {
        schema: SCHEMA.into(),
        renderer: jope_narrator::render::RENDERER.into(),
        run_id: run.into(),
        sequence: seq.to_string(),
        cursor: Cursor {
            run_id: run.into(),
            sequence: seq.to_string(),
        },
        fact: LineageFact::RunStopped {
            status: "fixture".into(),
        },
        links: Links::default(),
        narration: Narration {
            text: "Recorded run status: fixture.".into(),
            category: "status_fact".into(),
            claims: vec![Claim {
                category: "status_fact".into(),
                predicate: "run_status".into(),
                evidence: vec![evidence],
                details: json!({"status":"fixture"}),
            }],
            delta: None,
            variant: None,
        },
        previous_sha256: previous.into(),
        sha256: String::new(),
    };
    event.sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&event).unwrap()));
    event.validate(previous, seq, run).unwrap();
    event
}
fn records(run: &str, n: usize) -> Vec<Envelope> {
    let mut events: Vec<Envelope> = Vec::new();
    for seq in 1..=n {
        events.push(record(
            run,
            seq as u64,
            events.last().map_or("", |e| e.sha256.as_str()),
        ));
    }
    events
}
fn bytes(events: &[Envelope]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in events {
        serde_json::to_writer(&mut out, e).unwrap();
        out.push(b'\n');
    }
    out
}
fn commit_fixture(path: &Path, events: &[Envelope]) {
    let summary = jope_narrator::JournalSummary {
        run_id: "run-a".into(),
        records: events.len().to_string(),
        dropped: "0".into(),
        sha256: events.last().map_or_else(String::new, |e| e.sha256.clone()),
        bytes: bytes(events).len().to_string(),
        status: "capturing".into(),
        error: None,
    };
    fs::File::open(path).unwrap().sync_all().unwrap();
    fs::write(
        jope_narrator::commit_path(path),
        serde_json::to_vec(&summary).unwrap(),
    )
    .unwrap();
}
fn fixture(root: &Path, n: usize) -> (PathBuf, Vec<Envelope>) {
    let path = root.join("journal.jsonl");
    let events = records("run-a", n);
    fs::write(&path, bytes(&events)).unwrap();
    commit_fixture(&path, &events);
    (path, events)
}
fn zero() -> Cursor {
    Cursor {
        run_id: "run-a".into(),
        sequence: "0".into(),
    }
}
fn index(path: &Path) -> Index {
    Index::open(path, "run-a", Limits::default()).unwrap()
}

#[path = "narration_parts/mcp.rs"]
mod mcp;
#[path = "narration_parts/retention.rs"]
mod retention;
#[path = "narration_parts/runtime.rs"]
mod runtime;
