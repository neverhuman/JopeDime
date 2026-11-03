use super::*;

// Library level: the per-cut `current.json` pointers, which wedge the same way.
// --------------------------------------------------------------------------------------

const BOUND: &str = "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393";

/// One candidate for a single fixed cut, scored `value` (rmse: lower is better).
pub(super) fn candidate(value: f64, ts: u64) -> Candidate {
    let identity = HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: "d0".into(),
        cut_id: jope_learning::canon::sha256_hex(b"cut-0"),
        evaluator_sha256: BOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: "test".into(),
    };
    let dsl = format!("(pipe X (ridge alpha={value}))");
    let blob = format!("d0/cut-0/{value}").into_bytes();
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256().unwrap(),
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: jope_learning::canon::sha256_hex(dsl.as_bytes()),
            dsl,
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: BOUND.into(),
            version: "test".into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: true,
        provenance: Provenance {
            source: "test".into(),
            run_id: "run".into(),
            git_sha: UNBOUND.into(),
            bundle_id: UNBOUND.into(),
            record_sha256: jope_learning::canon::sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: ts,
    };
    Candidate {
        record,
        blob,
        chain_from_source: false,
    }
}

fn pointer_key() -> PointerKey {
    PointerKey::of(&candidate(1.0, 1).record)
}

/// [`candidate`] for another dataset, so it lands at another pointer.
fn candidate_in(dataset: &str, value: f64, ts: u64) -> Candidate {
    let mut c = candidate(value, ts);
    c.record.identity.dataset_hash = dataset.into();
    c.record.identity_sha256 = c.record.identity.sha256().unwrap();
    c.blob = format!("{dataset}/cut-0/{value}").into_bytes();
    c.record.provenance.record_sha256 = jope_learning::canon::sha256_hex(&c.blob);
    c
}

/// Make every write under the mirror's `hwm/regression/<dataset>` fail: a regular file sits
/// where that directory must be. Stands in for a transient store failure of exactly a pointer
/// write, after the batch objects and `HEAD.json` went up. Returns the planted path.
fn break_pointer_writes(mirror_dir: &Path, dataset: &str) -> PathBuf {
    let blocked = mirror_dir.join("hwm/regression").join(dataset);
    fs::create_dir_all(blocked.parent().unwrap()).unwrap();
    fs::write(&blocked, b"an object store that cannot take this pointer\n").unwrap();
    blocked
}

/// W8-T06: a `current.json` whose mirror write fails *after* the mirror's `HEAD.json` moved
/// must stay owed until it is written.
///
/// `sync::plan` derives the push window from the mirror's `HEAD.json` alone. Once that names
/// the batch, the pointers the batch moved were recorded as owed nowhere. So every later push
/// reported success while the mirror's own ledger accepted a record at a pointer the mirror
/// did not hold. It stayed that way until some later batch happened to accept at that exact
/// pointer, which for a settled cut may be never.
///
/// Fails on the pre-fix code at the first catch-up assertion. A push with nothing new, and a
/// push of an unrelated cut, both leave the pointer absent from the mirror.
#[test]
fn a_pointer_write_lost_after_the_head_moved_is_still_owed() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mirror_head = mirror_dir.join("hwm/ledger/HEAD.json");
    let local_head = || fs::read(Ledger::open(state.root(), "hwm").root().join("HEAD.json"));
    let d0 = PointerKey::of(&candidate_in("d0", 0.5, 1).record);
    let d1 = PointerKey::of(&candidate_in("d1", 0.6, 2).record);

    // 1. The pointer write fails after the batch objects and HEAD.json went up.
    let blocked = break_pointer_writes(&mirror_dir, "d0");
    hwm::ingest(
        &state,
        Some(&m),
        vec![candidate_in("d0", 0.5, 1)],
        json!({"k": 1}),
    )
    .expect_err("the pointer write must fail the push");
    assert_eq!(fs::read(&mirror_head).unwrap(), local_head().unwrap());
    assert!(!mirror_dir.join(d0.rel()).is_file());

    // 2. The store recovers, and a push with nothing new settles what is owed.
    fs::remove_file(&blocked).unwrap();
    hwm::ingest(&state, Some(&m), vec![], json!({"k": 2})).unwrap();
    assert_eq!(
        fs::read(mirror_dir.join(d0.rel())).ok(),
        fs::read(state.pointer_path(&d0).unwrap()).ok(),
        "the pointer owed since the failed push was never written"
    );

    // 3. The same failure on another cut, then a push that only touches a third cut.
    let blocked = break_pointer_writes(&mirror_dir, "d1");
    hwm::ingest(
        &state,
        Some(&m),
        vec![candidate_in("d1", 0.6, 2)],
        json!({"k": 3}),
    )
    .expect_err("the pointer write must fail the push");
    fs::remove_file(&blocked).unwrap();
    hwm::ingest(
        &state,
        Some(&m),
        vec![candidate_in("d2", 0.7, 3)],
        json!({"k": 4}),
    )
    .unwrap();
    assert_eq!(
        fs::read(mirror_dir.join(d1.rel())).ok(),
        fs::read(state.pointer_path(&d1).unwrap()).ok(),
        "a push of an unrelated cut left the owed pointer behind"
    );
    // Every pointer the local ledger accepted is now on the mirror, byte for byte.
    for (key, _) in state.pointers().unwrap() {
        assert_eq!(
            fs::read(mirror_dir.join(key.rel())).ok(),
            fs::read(state.pointer_path(&key).unwrap()).ok(),
            "{}",
            key.rel()
        );
    }
    assert_eq!(fs::read(&mirror_head).unwrap(), local_head().unwrap());

    // A clean push settles everything it adds, and the durable owed record says so: a stale
    // record would cost one GET per listed pointer on the next push.
    let rep = hwm::ingest(
        &state,
        Some(&m),
        vec![candidate_in("d3", 0.8, 4)],
        json!({"k": 5}),
    )
    .unwrap();
    assert_eq!(rep.mirror.unwrap().pointers_owed, 0);
    let owed = jope_learning::hwm::Owed::load(&state, &m.url).unwrap();
    assert!(
        owed.pointers.is_empty(),
        "the owed record still lists settled pointers: {:?}",
        owed.pointers
    );
}

#[test]
fn a_pointer_left_behind_by_a_local_only_ingest_catches_up() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let key = pointer_key();

    // 1. mirrored: the mirror's pointer is rmse 0.5.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    let at_05 = fs::read(mirror_dir.join(key.rel())).unwrap();

    // 2. local only: local improves to 0.4, the mirror still holds 0.5.
    hwm::ingest(&state, None, vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();
    assert_eq!(fs::read(mirror_dir.join(key.rel())).unwrap(), at_05);

    // 3. mirrored again: the pointer must move to the local tip, skipping the state the mirror
    //    never saw, rather than failing `pointer changed` for ever.
    let rep = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 3)], json!({"k": 3})).unwrap();
    assert_eq!(rep.accepted, 1);
    assert_eq!(
        fs::read(mirror_dir.join(key.rel())).unwrap(),
        fs::read(state.pointer_path(&key).unwrap()).unwrap(),
        "the pointer did not catch up"
    );
    let record: HwmRecord =
        serde_json::from_slice(&fs::read(mirror_dir.join(key.rel())).unwrap()).unwrap();
    assert_eq!(record.score.value, 0.3);
    assert_eq!(record.improvement_number, 2);
    // The skipped 0.4 record is on the mirror too: the backfilled batch names it.
    for sha in ["d0/cut-0/0.4", "d0/cut-0/0.5"] {
        let sha = jope_learning::canon::sha256_hex(sha.as_bytes());
        assert!(
            mirror_dir.join(State::blob_rel(&sha)).exists(),
            "blob {sha} of a backfilled batch is missing from the mirror"
        );
    }
}

#[test]
fn a_pointer_from_a_chain_this_host_does_not_have_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let key = pointer_key();

    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();

    // Another host wins the same cut with 0.2 and publishes it; this host never ingested it, so
    // its own ledger has no event for that record.
    let theirs = candidate(0.2, 9).record;
    let mut bytes = serde_json::to_vec_pretty(&theirs).unwrap();
    bytes.push(b'\n');
    fs::write(mirror_dir.join(key.rel()), &bytes).unwrap();

    // Catching up must not mean clobbering: 0.4 is better than the 0.5 we knew about, but worse
    // than the 0.2 on the mirror, and nothing in our ledger explains where 0.2 came from.
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.4, 2)], json!({"k": 2}))
        .expect_err("a foreign pointer must not be overwritten");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("this host's ledger never accepted at that pointer"),
        "expected the foreign-pointer refusal, got: {msg}"
    );
    assert_eq!(
        fs::read(mirror_dir.join(key.rel())).unwrap(),
        bytes,
        "the other host's pointer was overwritten"
    );
}
