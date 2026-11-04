use super::*;

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
