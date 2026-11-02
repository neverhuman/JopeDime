use feat_math::{
    dsl::{DslExpr, ParamValue, Stage},
    GenomeIr,
};
use jope_narrator::{
    diff_programs, fact_channel, read_journal, spawn_journal, AtomicEdit, Cursor, EmitStatus,
    Envelope, FactJournal, Limits, LineageFact, Links, Metric, ProgramRef, ProgramSnapshot,
    RecordingTap, MAX_FRAME_BYTES, MIN_FRAME_BYTES,
};
use std::{collections::BTreeMap, fs, io::Write, path::PathBuf};

fn scratch() -> tempfile::TempDir {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    fs::create_dir_all(root.join("narrator-test-artifacts")).unwrap();
    tempfile::tempdir_in(root.join("narrator-test-artifacts")).unwrap()
}
fn program(text: &str) -> ProgramRef {
    use sha2::{Digest, Sha256};
    ProgramRef {
        sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        canonical: text.into(),
    }
}
fn stop(n: usize) -> LineageFact {
    LineageFact::ExplorationStopped {
        cut: n,
        reason: "evaluations".into(),
        generations: 1,
        evaluations: n,
    }
}
fn draw(parent: &str, child: &str) -> LineageFact {
    LineageFact::Drawn {
        cut: 0,
        phase: "exploration".into(),
        coordinate: "exploration/1/0/4/0".into(),
        generation: 1,
        island: 0,
        slot: Some(4),
        attempt: Some(0),
        origin: "mutate".into(),
        seed: Some(u64::MAX),
        parents: vec![program(parent)],
        child: Some(program(child)),
        outcome: "candidate".into(),
        detail: None,
    }
}
fn journal(path: &std::path::Path, run: &str, facts: Vec<LineageFact>) -> Vec<Envelope> {
    let (tap, journal) = FactJournal::create(path, run, Limits::default()).unwrap();
    for fact in facts {
        assert_eq!(tap.emit(fact), EmitStatus::Queued);
    }
    let clone = tap.clone();
    let summary = journal.finish().unwrap();
    assert_eq!(clone.emit(stop(1)), EmitStatus::Closed);
    let page = read_journal(path, None, 1024, 16 * 1024 * 1024).unwrap();
    assert_eq!(summary.records, page.events.len().to_string());
    page.events
        .into_iter()
        .filter(|e| !matches!(e.fact, LineageFact::TapDropped { .. }))
        .collect()
}

mod journal;
mod snapshot;

#[path = "support/golden_v1.rs"]
mod golden_v1;
