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

#[test]
fn typed_snapshot_preserves_signed_zero_adjacent_floats_and_large_integers() {
    let make = |v: f64| {
        let mut params = BTreeMap::new();
        params.insert("lambda_frac".into(), ParamValue::Float(v));
        params.insert("seed".into(), ParamValue::Int(i64::MAX));
        GenomeIr::Pipeline(DslExpr {
            stages: vec![Stage {
                op: "ridge".into(),
                params,
            }],
        })
    };
    for (a, b) in [
        (0.0, -0.0),
        (1.0, f64::from_bits(1.0f64.to_bits() + 1)),
        (
            f64::from_bits(0x7ff8_0000_0000_0042),
            f64::from_bits(0x7ff8_0000_0000_0043),
        ),
    ] {
        let (p, c) = (
            ProgramSnapshot::from_typed(&make(a)).unwrap(),
            ProgramSnapshot::from_typed(&make(b)).unwrap(),
        );
        assert_ne!(p.identity, c.identity);
        let delta = diff_programs(&p, &c).unwrap();
        assert_eq!(delta.edits.len(), 1);
        assert_eq!(delta.replay().unwrap(), c);
    }
}

#[test]
fn patches_reconstruct_ten_thousand_real_operator_transitions() {
    use feat_math::evolve::{mutate_canonical_pipeline, random_canonical_pipeline};
    for parent_seed in 0..50u64 {
        let text = random_canonical_pipeline(parent_seed, 4);
        let p = ProgramSnapshot::from_legacy(&text).unwrap();
        for seed in 0..200 {
            let child = mutate_canonical_pipeline(&text, seed, 4).unwrap();
            let c = ProgramSnapshot::from_legacy(&child).unwrap();
            let delta = diff_programs(&p, &c).unwrap();
            assert_eq!(
                delta.replay().unwrap(),
                c,
                "parent {parent_seed}, mutation {seed}"
            );
        }
    }
}

#[test]
fn ordered_duplicates_and_general_edits_replay_with_preconditions() {
    for (parent, child) in [
        (
            "(pipe X (robust_scale) (robust_scale) (ridge))",
            "(pipe X (robust_scale) (ridge))",
        ),
        (
            "(pipe X (rank) (robust_scale) (ridge))",
            "(pipe X (robust_scale) (rank) (ridge))",
        ),
        (
            "(pipe X (ridge))",
            "(pipe X (rank) (cheb degree=8) (ridge lambda_frac=0.3))",
        ),
        (
            "(pipe X (rank) (cheb degree=4) (ridge))",
            "(pipe X (winsor_clip q=0.1) (ridge))",
        ),
    ] {
        let (p, c) = (
            ProgramSnapshot::from_legacy(parent).unwrap(),
            ProgramSnapshot::from_legacy(child).unwrap(),
        );
        let mut delta = diff_programs(&p, &c).unwrap();
        assert_eq!(delta.replay().unwrap(), c);
        if let Some(AtomicEdit::Delete { before, .. }) = delta.edits.first_mut() {
            before.op = "wrong".into();
            assert!(delta.replay().is_err());
        }
    }
}

#[test]
fn unsupported_representations_have_exact_whole_snapshot_patch() {
    let p = ProgramSnapshot::from_legacy("(sym raw \"c0 + 1\")").unwrap();
    let c = ProgramSnapshot::from_legacy("(sym affine \"c0 * 2\")").unwrap();
    let delta = diff_programs(&p, &c).unwrap();
    assert_eq!(delta.coverage, "whole_program_only");
    assert_eq!(delta.replay().unwrap(), c);
}

#[test]
fn queue_count_bytes_and_frame_bounds_produce_explicit_gaps() {
    let dir = scratch();
    let limits = |max_frame_bytes| Limits {
        max_frame_bytes,
        ..Limits::default()
    };
    for bound in [0, 4096, MIN_FRAME_BYTES - 1, MAX_FRAME_BYTES + 1] {
        let path = dir.path().join(format!("unsupported-frame-{bound}.jsonl"));
        assert!(FactJournal::create(&path, "unsupported-frame", limits(bound)).is_err());
        assert!(!path.exists(), "refuse before creating a journal");
        assert!(!jope_narrator::commit_path(&path).exists());
    }
    let path = dir.path().join("minimum-frame.jsonl");
    let (tap, writer) =
        FactJournal::create(&path, "minimum-frame", limits(MIN_FRAME_BYTES)).unwrap();
    assert_eq!(tap.emit(stop(1)), EmitStatus::Queued);
    let summary = writer.finish().unwrap();
    assert_eq!(summary.status, "complete");
    assert_eq!(summary.dropped, "0");
    let page = read_journal(&path, None, 10, MAX_FRAME_BYTES).unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].fact, stop(1));
    assert!(fs::read(&path).unwrap().len() <= MIN_FRAME_BYTES + 1);

    let path = dir.path().join("minimum-frame-loss.jsonl");
    let (tap, writer) =
        FactJournal::create(&path, "minimum-frame-loss", limits(MIN_FRAME_BYTES)).unwrap();
    assert_eq!(tap.emit(stop(2)), EmitStatus::Queued);
    assert_eq!(
        tap.emit(LineageFact::RunStopped {
            status: "x".repeat(MIN_FRAME_BYTES),
        }),
        EmitStatus::Invalid
    );
    let summary = writer.finish().unwrap();
    assert_eq!(summary.status, "incomplete");
    assert_eq!(summary.dropped, "1");
    let page = read_journal(&path, None, 10, MAX_FRAME_BYTES).unwrap();
    assert_eq!(page.events.len(), 2);
    assert!(page
        .events
        .iter()
        .any(|e| e.fact == LineageFact::TapDropped { dropped: 1 }));
    assert!(fs::read_to_string(&path)
        .unwrap()
        .lines()
        .all(|line| line.len() <= MIN_FRAME_BYTES));

    let (tap, mut recv) = fact_channel(2, 1024 * 1024);
    assert_eq!(tap.emit(stop(0)), EmitStatus::Queued);
    assert_eq!(tap.emit(stop(1)), EmitStatus::Queued);
    assert_eq!(tap.emit(stop(2)), EmitStatus::Dropped);
    tap.close();
    let facts = std::iter::from_fn(|| recv.recv().unwrap()).collect::<Vec<_>>();
    assert_eq!(facts.len(), 3);
    assert!(facts
        .iter()
        .any(|f| matches!(f, LineageFact::TapDropped { dropped: 1 })));
    let (tap, mut recv) = fact_channel(100, 1);
    assert_eq!(tap.emit(stop(1)), EmitStatus::Dropped);
    tap.close();
    assert!(matches!(
        recv.recv().unwrap(),
        Some(LineageFact::TapDropped { dropped: 1 })
    ));
    let (tap, _recv) = fact_channel(100, 1024 * 1024);
    let mut text = String::with_capacity(16 * 1024 * 1024);
    text.push('x');
    assert_eq!(
        tap.emit(LineageFact::RunStopped { status: text }),
        EmitStatus::Queued,
        "retained queue owns only normalized bytes"
    );
    assert_eq!(
        tap.emit(LineageFact::RunStopped {
            status: "x".repeat(2 * 1024 * 1024)
        }),
        EmitStatus::Invalid
    );
}

#[test]
fn deterministic_journal_finishes_with_live_tap_and_preserves_prefix() {
    let dir = scratch();
    let a = dir.path().join("a.jsonl");
    let b = dir.path().join("b.jsonl");
    let prefix = draw(
        "(pipe X (ridge lambda_frac=0.02))",
        "(pipe X (ridge lambda_frac=0.04))",
    );
    let future = |rmse| LineageFact::Measured {
        cut: 0,
        phase: "exploration".into(),
        generation: 1,
        island: Some(0),
        program_sha256: program("(pipe X (ridge lambda_frac=0.04))").sha256,
        origin: "mutate".into(),
        status: "measured".into(),
        rmse: Some(rmse),
        hold_r2: None,
        train_r2: None,
        objective_loss: None,
        seconds: 0.1,
        failure: None,
    };
    let x = journal(&a, "same-run", vec![prefix.clone(), future(0.1)]);
    let y = journal(&b, "same-run", vec![prefix, future(0.9)]);
    assert_eq!(x[0], y[0]);
    assert!(!x[0].narration.text.contains("0.9"));
    assert!(!y[0].narration.text.contains("0.1"));
    // The retained prefix remains identical after the future record exists.
    let replay = read_journal(&a, None, 1024, 1 << 20).unwrap();
    assert_eq!(
        replay
            .events
            .iter()
            .find(|e| matches!(e.fact, LineageFact::Drawn { .. }))
            .unwrap(),
        &x[0]
    );
    assert!(x[0]
        .narration
        .claims
        .iter()
        .any(|c| c.category == "prospective_hypothesis"));
    assert!(spawn_journal(&a).is_err());
}

#[test]
fn corrupt_complete_records_refuse_partial_tails_wait_and_cursor_scope_is_strict() {
    let dir = scratch();
    let path = dir.path().join("facts.jsonl");
    let events = journal(&path, "run-a", vec![stop(1), stop(2)]);
    let wrong = Cursor {
        run_id: "other".into(),
        sequence: "1".into(),
    };
    assert!(read_journal(&path, Some(&wrong), 10, 1024 * 1024).is_err());
    let future = Cursor {
        run_id: "run-a".into(),
        sequence: "3".into(),
    };
    assert!(read_journal(&path, Some(&future), 10, 1024 * 1024).is_err());
    let page = read_journal(&path, Some(&events[0].cursor), 10, 1024 * 1024).unwrap();
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0], events[1]);
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"{\"schema\":").unwrap();
    drop(file);
    assert!(
        read_journal(&path, None, 10, 1024 * 1024)
            .unwrap()
            .partial_tail
    );
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"\n").unwrap();
    drop(file);
    assert!(
        read_journal(&path, None, 10, 1024 * 1024)
            .unwrap()
            .partial_tail,
        "a complete uncommitted suffix is withheld after a crash"
    );
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'x';
    fs::write(&path, bytes).unwrap();
    assert!(read_journal(&path, None, 10, 1024 * 1024).is_err());
}

#[test]
fn forged_claim_or_text_fails_even_after_recomputing_content_hash() {
    use sha2::{Digest, Sha256};
    let dir = scratch();
    let mut event = journal(&dir.path().join("facts.jsonl"), "run", vec![stop(1)]).remove(0);
    event.narration.text = "The hypothesis was confirmed; error fell 99%.".into();
    event.sha256.clear();
    event.sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&event).unwrap()));
    assert!(event.validate("", 1, "run").is_err());
}

#[test]
fn terminal_controls_are_inert_and_clamped_parameter_scope_is_narrow() {
    let dir = scratch();
    let events = journal(
        &dir.path().join("safe.jsonl"),
        "run",
        vec![
            draw(
                "(pipe X (winsor_clip q=0.30) (ridge))",
                "(pipe X (winsor_clip q=0.35) (ridge))",
            ),
            LineageFact::RunStopped {
                status: "failed\u{1b}]8;;https://bad\u{7}".into(),
            },
        ],
    );
    assert!(events[0]
        .narration
        .text
        .contains("both requests resolve to q=0.25"));
    assert!(events[0]
        .narration
        .claims
        .iter()
        .any(|c| c.category == "implementation_fact"));
    assert!(!events[1].narration.text.chars().any(char::is_control));
    assert!(!events[0].narration.text.contains("identical predictions"));
}

#[test]
fn metrics_refuse_mismatched_scope_and_nonfinite_values() {
    let baseline = Metric::new("rmse", "minimize", "cut-1/evaluator-1", 0.5).unwrap();
    assert_eq!(
        Metric::new("rmse", "minimize", "cut-1/evaluator-1", 0.25)
            .unwrap()
            .improvement_over(&baseline)
            .unwrap(),
        0.25
    );
    assert!(Metric::new("rmse", "minimize", "cut-2/evaluator-1", 0.25)
        .unwrap()
        .improvement_over(&baseline)
        .is_err());
    assert!(Metric::new("rmse", "minimize", "cut-1", f64::NAN).is_err());
}

#[test]
fn native_links_roundtrip_and_recording_collector_keeps_concurrent_inputs() {
    let recording = RecordingTap::default();
    let tap = recording.tap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let tap = tap.clone();
            scope.spawn(move || {
                for n in 0..100 {
                    tap.emit_linked(
                        stop(n),
                        Links {
                            attempt_id: Some("a".repeat(64)),
                            ..Links::default()
                        },
                    );
                }
            });
        }
    });
    let inputs = recording.linked_facts();
    let expected = "a".repeat(64);
    assert_eq!(inputs.len(), 400);
    assert!(inputs
        .iter()
        .all(|(_, l)| l.attempt_id.as_deref() == Some(expected.as_str())));
}

#[test]
fn durable_watermark_distinguishes_capture_closure_loss_and_source_status() {
    let dir = scratch();
    let path = dir.path().join("closed.jsonl");
    let (tap, writer) = FactJournal::create(&path, "capture", Limits::default()).unwrap();
    assert_eq!(
        jope_narrator::read_commit(&path).unwrap().status,
        "capturing"
    );
    tap.emit(LineageFact::RunStopped {
        status: "failed".into(),
    });
    let summary = writer.finish().unwrap();
    assert_eq!(
        summary.status, "complete",
        "source failure can still be captured completely"
    );
    let state = jope_narrator::read_commit(&path).unwrap();
    assert_eq!(state.bytes, fs::metadata(&path).unwrap().len().to_string());
    assert_eq!(state.sha256, summary.sha256);
    let limited = dir.path().join("limited.jsonl");
    let (tap, writer) = FactJournal::create(
        &limited,
        "limited",
        Limits {
            max_records: 1,
            ..Limits::default()
        },
    )
    .unwrap();
    tap.emit(stop(1));
    tap.emit(stop(2));
    assert!(writer.finish().is_err());
    let state = jope_narrator::read_commit(&limited).unwrap();
    assert_eq!(state.status, "resource_limit");
    assert_eq!(state.records, "1");
    assert_eq!(tap.emit(stop(3)), EmitStatus::Closed);
    assert_eq!(
        read_journal(&limited, None, 10, 1 << 20)
            .unwrap()
            .events
            .len(),
        1
    );
}

#[test]
fn watermark_failure_never_acknowledges_unsynced_or_uncommitted_line() {
    let dir = scratch();
    let path = dir.path().join("failed.jsonl");
    let (tap, writer) = FactJournal::create(&path, "failed", Limits::default()).unwrap();
    // Hold the writer's exclusive pending path to reproduce a real publication failure.
    let mut pending = jope_narrator::commit_path(&path).into_os_string();
    pending.push(".pending");
    fs::write(std::path::PathBuf::from(pending), b"occupied").unwrap();
    tap.emit(stop(1));
    assert!(writer.finish().is_err());
    let state = jope_narrator::read_commit(&path).unwrap();
    assert_eq!(state.records, "0");
    assert_eq!(
        state.status, "capturing",
        "unpersistable failure leaves closure unknown"
    );
    let page = read_journal(&path, None, 10, 1 << 20).unwrap();
    assert!(page.events.is_empty());
    assert!(page.partial_tail);
}

#[test]
fn real_bounded_queue_preserves_parallel_producers_below_capacity() {
    let (tap, mut receiver) = fact_channel(2048, 4 << 20);
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let tap = tap.clone();
            scope.spawn(move || {
                for n in 0..128 {
                    assert_eq!(tap.emit(stop(worker * 128 + n)), EmitStatus::Queued);
                }
            });
        }
    });
    tap.close();
    let facts = std::iter::from_fn(|| receiver.recv().unwrap()).collect::<Vec<_>>();
    assert_eq!(facts.len(), 1024);
    assert_eq!(tap.dropped(), 0);
    let mut ids = std::collections::BTreeSet::new();
    for fact in facts {
        if let LineageFact::ExplorationStopped { cut, .. } = fact {
            assert!(ids.insert(cut));
        } else {
            panic!("unexpected gap")
        }
    }
    assert_eq!(ids.len(), 1024);
}

#[test]
fn finish_racing_producer_drains_every_accepted_record_and_closes_retained_taps() {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    };
    let dir = scratch();
    let path = dir.path().join("close-race.jsonl");
    let (tap, writer) = FactJournal::create(&path, "race", Limits::default()).unwrap();
    let accepted = Arc::new(AtomicU64::new(0));
    let (ready, started) = std::sync::mpsc::channel();
    let worker_tap = tap.clone();
    let count = accepted.clone();
    let worker = std::thread::spawn(move || {
        assert_eq!(worker_tap.emit(stop(0)), EmitStatus::Queued);
        count.fetch_add(1, Ordering::SeqCst);
        ready.send(()).unwrap();
        for n in 1..512 {
            match worker_tap.emit(stop(n)) {
                EmitStatus::Queued => {
                    count.fetch_add(1, Ordering::SeqCst);
                }
                EmitStatus::Closed => {}
                other => panic!("unexpected admission {other:?}"),
            }
        }
    });
    started.recv().unwrap();
    let summary = writer.finish().unwrap();
    worker.join().unwrap();
    assert_eq!(summary.records, accepted.load(Ordering::SeqCst).to_string());
    assert_eq!(summary.dropped, "0");
    assert_eq!(summary.status, "complete");
    assert_eq!(tap.emit(stop(999)), EmitStatus::Closed);
}

#[test]
fn readable_stage_details_clamps_aliases_and_ordered_equal_parents_are_grounded() {
    let dir = scratch();
    let mut crossover = draw("(pipe X (ridge))", "(pipe X (logistic lambda_frac=0.04))");
    if let LineageFact::Drawn {
        parents, origin, ..
    } = &mut crossover
    {
        parents.push(parents[0].clone());
        *origin = "crossover".into();
    }
    let events = journal(
        &dir.path().join("prose.jsonl"),
        "prose",
        vec![
            draw("(pipe X (ridge))", "(pipe X (winsor_clip q=0.1) (ridge))"),
            draw(
                "(pipe X (winsor_clip q=0.30) (ridge))",
                "(pipe X (winsor_clip q=0.35) (ridge))",
            ),
            crossover,
        ],
    );
    assert!(events[0].narration.text.contains("q=0.1"));
    assert!(!events[0].narration.text.contains("float_bits"));
    assert!(events[1]
        .narration
        .claims
        .iter()
        .all(|c| c.category != "prospective_hypothesis"));
    assert!(events[2].narration.text.contains("same program"));
    assert!(events[2].narration.text.contains("logistic head to ridge"));
    assert!(events[2]
        .narration
        .claims
        .iter()
        .any(|c| c.predicate == "additional_parent_difference"));
    assert_eq!(events[0].renderer, jope_narrator::render::RENDERER);
}

#[test]
fn automatic_capture_sessions_are_bounded_per_process() {
    const CHILD: &str = "JOPE_NARRATION_SESSION_LIMIT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        for _ in 0..jope_narrator::journal::MAX_ENV_SESSIONS {
            let (_, writer) = jope_narrator::journal_from_env().unwrap().unwrap();
            assert_eq!(writer.finish().unwrap().status, "complete");
        }
        assert!(jope_narrator::journal_from_env().is_err());
        assert!(jope_narrator::journal_from_env().unwrap().is_none());
        let root = std::path::PathBuf::from(std::env::var_os(jope_narrator::DIR_ENV).unwrap());
        assert_eq!(
            fs::read_dir(root).unwrap().count(),
            2 * jope_narrator::journal::MAX_ENV_SESSIONS as usize
        );
        return;
    }
    let dir = scratch();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "automatic_capture_sessions_are_bounded_per_process",
        ])
        .env(CHILD, "1")
        .env(jope_narrator::DIR_ENV, dir.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
}

#[path = "support/golden_v1.rs"]
mod golden_v1;
