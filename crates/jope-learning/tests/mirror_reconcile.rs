//! A mirror that is merely *behind* must be able to catch up; a mirror that has *diverged* must
//! still be refused.
//!
//! Making the `HEAD.json` write conditional on the writer's own pre-append HEAD stops a second
//! host's batch being orphaned, but on its own it is a one-way door: after a single
//! `hwm ingest` run without `--store` (or one transient push failure) the local pre-append HEAD
//! never equals the mirror's again, and every later push — and every retry of it — fails with
//! `mirror pointer changed`. There is no push/pull/backfill verb to get out of it, so that host's
//! mirror is wedged for good while each retry seals another local batch and leaves another object
//! on the mirror that the mirror's own HEAD cannot reach.
//!
//! `a_local_only_ingest_no_longer_wedges_the_mirror` is the three-step reproduction; it fails on
//! the pre-fix code at step 3 with rc=2. `a_diverged_mirror_is_refused_and_nothing_is_written` is
//! the negative control: catching up must not become "overwrite whatever is there".
//!
//! Reconciling correctly is not enough on its own, because `HEAD.json` is also the *only* input the
//! reconcile has: `plan` returns the batches after the mirror's tip and nothing else, so anything a
//! push skips under a tip it has already moved is never offered to a later push.
//! `a_failed_blob_push_leaves_the_mirrors_head_where_it_was` is the ordering that follows from
//! that, with an object-store failure injected mid-push.
//!
//! That ordering covers a blob missing from the *mirror*. It does not cover a blob missing from
//! the *local store*, which the push used to skip in silence —
//! `a_blob_the_local_store_cannot_supply_is_never_published_over` is that case,
//! `a_blob_only_the_mirror_still_has_does_not_wedge_the_push` is its negative control, and
//! `a_wrong_object_under_the_immutable_key_cannot_stand_in_for_the_blob` is the boundary between
//! the two: the mirror backs the batch only when its bytes *are* the record.

mod common;

use common::{jl, Run};
use jope_learning::hwm::{self, Candidate};
use jope_learning::ledger::Ledger;
use jope_learning::metric::Metric;
use jope_learning::mirror::Mirror;
use jope_learning::record::*;
use jope_learning::state::{PointerKey, State};
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

// --------------------------------------------------------------------------------------
// CLI level: the reported three-step sequence, run through the built binary.
// --------------------------------------------------------------------------------------

/// `champions[skip..skip+take]` copied into a directory of its own.
fn champion_subset(dest: &Path, skip: usize, take: usize) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/champions");
    fs::create_dir_all(dest).unwrap();
    let mut names: Vec<_> = fs::read_dir(&src)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    names.sort();
    let chosen: Vec<_> = names.into_iter().skip(skip).take(take).collect();
    assert_eq!(chosen.len(), take, "fixture corpus too small");
    for name in chosen {
        fs::copy(src.join(&name), dest.join(&name)).unwrap();
    }
    dest.to_path_buf()
}

fn ingest(state: &Path, store: Option<&str>, dir: &Path, bundle: &str) -> Run {
    let mut args: Vec<&str> = Vec::new();
    if let Some(url) = store {
        args.extend(["--store", url]);
    }
    args.extend([
        "hwm",
        "ingest",
        "--source",
        "champions",
        dir.to_str().unwrap(),
        "--bundle-id",
        bundle,
    ]);
    jl(state, &args)
}

/// Every file under `root`, as paths relative to it.
fn tree(root: &Path) -> BTreeSet<String> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            e.path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

#[test]
fn a_local_only_ingest_no_longer_wedges_the_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);
    let mirror_head = mirror.join("hwm/ledger/HEAD.json");

    // 1. mirrored ingest: mirror HEAD = batch 1.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    let head_1 = fs::read(&mirror_head).unwrap();

    // 2. the same host ingests again with NO --store: local moves to batch 2, mirror stays at 1.
    ingest(&state, None, &dir_b, "b").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "an unmirrored ingest must not touch the mirror"
    );

    // 3. mirrored again. On the pre-fix code this is rc=2 `mirror pointer changed`, for ever.
    let r = ingest(&state, Some(&url), &dir_a, "a");
    assert_eq!(
        r.status, 0,
        "the mirror must catch up, not wedge; stderr: {}",
        r.stderr
    );

    // The mirror is now a complete copy: same batches, same tip, same blobs, same pointers, and
    // no object left unreachable from HEAD.
    let ledger = Ledger::open(&state, "hwm");
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap()
    );
    let local_batches: BTreeSet<String> =
        ledger.batches().unwrap().into_iter().map(|b| b.0).collect();
    assert_eq!(
        local_batches.len(),
        3,
        "three local batches: {local_batches:?}"
    );
    let mirror_batches: BTreeSet<String> = tree(&mirror.join("hwm/ledger"))
        .into_iter()
        .filter(|p| p.ends_with(".jsonl"))
        .collect();
    assert_eq!(
        mirror_batches, local_batches,
        "the mirror must hold exactly this host's batches — no gap and no unreachable object"
    );
    assert_eq!(
        tree(&mirror.join("blobs")),
        tree(&state.join("blobs")),
        "every blob the mirrored ledger names must be on the mirror"
    );
    let st = State::at(&state);
    for (key, _) in st.pointers().unwrap() {
        assert_eq!(
            fs::read(mirror.join(key.rel())).unwrap(),
            fs::read(st.pointer_path(&key).unwrap()).unwrap(),
            "pointer {} was not caught up",
            key.rel()
        );
    }
    // And it stays unwedged.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap()
    );
}

#[test]
fn a_diverged_mirror_is_refused_and_nothing_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);

    // Host A owns the mirror's chain.
    let state_a = tmp.path().join("state-a");
    ingest(&state_a, Some(&url), &dir_a, "host-a").ok();
    let before = tree(&mirror);
    let head_a = fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap();

    // Host B has a state tree that has never seen host A's batch: it cannot continue that chain,
    // and catching it up would mean orphaning A's tip.
    let state_b = tmp.path().join("state-b");
    let r = ingest(&state_b, Some(&url), &dir_b, "host-b");
    assert_ne!(
        r.status,
        0,
        "a diverged mirror must still fail closed; stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains("holds a chain this host cannot continue"),
        "expected the divergence refusal, got: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("re-run the same command without --store"),
        "the refusal must say what to run, got: {}",
        r.stderr
    );

    // Nothing on the mirror moved, and — because the check runs before the local append — host B
    // sealed no batch and moved no pointer either, so a retry has no wider gap to close.
    assert_eq!(tree(&mirror), before, "the mirror was written to");
    assert_eq!(
        fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap(),
        head_a
    );
    assert!(
        Ledger::open(&state_b, "hwm").batches().unwrap().is_empty(),
        "host B appended locally despite the refusal: {:?}",
        Ledger::open(&state_b, "hwm").batches().unwrap()
    );
    assert!(State::at(&state_b).pointers().unwrap().is_empty());
}

// --------------------------------------------------------------------------------------
// Library level: the per-cut `current.json` pointers, which wedge the same way.
// --------------------------------------------------------------------------------------

const BOUND: &str = "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393";

/// One candidate for a single fixed cut, scored `value` (rmse: lower is better).
fn candidate(value: f64, ts: u64) -> Candidate {
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

/// Every `record_sha256` the mirror's own ledger batches name whose blob is *not* on the mirror.
/// The mirror's copy is self-contained exactly when this is empty: `hwm verify` reads a blob per
/// event, so a name with no blob is a ledger that cannot be walked from the mirror alone.
fn named_blobs_missing_from(mirror: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for entry in walkdir::WalkDir::new(mirror.join("hwm/ledger"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
    {
        let (events, _, _, _) = Ledger::read_batch(entry.path()).unwrap();
        for v in events {
            let sha = v["record"]["provenance"]["record_sha256"]
                .as_str()
                .expect("every event names a record sha256");
            if !mirror.join(State::blob_rel(sha)).exists() {
                missing.push(sha.to_string());
            }
        }
    }
    missing.sort();
    missing.dedup();
    missing
}

/// One failed blob PUT must not leave the mirror's `HEAD.json` naming batches whose blobs the
/// mirror does not hold.
///
/// The invariant: **while any blob a batch names is still missing from the mirror, the mirror's
/// `HEAD.json` has not moved past that batch.** It is worth exactly one thing, and that thing is
/// recovery: `sync::plan` derives `missing` from the mirror's tip alone, so a tip that has moved
/// past a batch puts that batch out of reach of every later push. Hold the invariant and the retry
/// sees the same gap and closes it; break it and the mirror's own ledger names records it will
/// never hold, while each later push reports success.
///
/// Fails on the round-two code at the first assertion: `sync::push` moved `HEAD.json` before
/// `push_mirror` uploaded the blobs, so the injected failure leaves the mirror's tip at the local
/// tip with the blobs of two batches missing, and steps 4 and 5 then backfill nothing.
#[test]
fn a_failed_blob_push_leaves_the_mirrors_head_where_it_was() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mirror_head = mirror_dir.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the blob it names.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    assert!(named_blobs_missing_from(&mirror_dir).is_empty());
    let head_1 = fs::read_to_string(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror_dir.join("hwm/ledger"));

    // 2. local only: the mirror is behind by a batch whose blob it has never seen.
    hwm::ingest(&state, None, vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // 3. every blob PUT now fails: a regular file sits where the store must create the `sha256`
    //    directory, which no uid can write through. Stands in for one transient object-store
    //    error — a 503, a throttle, a revoked key.
    let blobs = mirror_dir.join("blobs/sha256");
    let stashed = tmp.path().join("blobs-sha256");
    fs::rename(&blobs, &stashed).unwrap();
    fs::write(&blobs, b"an object store that cannot take a blob\n").unwrap();
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 3)], json!({"k": 3}))
        .expect_err("the blob PUT must fail the push");
    assert!(
        format!("{err:#}").contains("blobs/sha256/"),
        "expected the blob put to be what failed, got: {err:#}"
    );
    assert_eq!(
        fs::read_to_string(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over batches whose blobs never went up; no later push will \
         backfill them, because `plan` only ever returns batches after that tip"
    );
    assert_eq!(
        tree(&mirror_dir.join("hwm/ledger")),
        ledger_1,
        "a batch object went up before the blobs it names"
    );

    // 4. the object store recovers. The gap is still the gap, so the next push closes it — batches,
    //    the blobs they name, and the pointers they moved.
    fs::remove_file(&blobs).unwrap();
    fs::rename(&stashed, &blobs).unwrap();
    hwm::ingest(&state, Some(&m), vec![candidate(0.25, 4)], json!({"k": 4})).unwrap();

    let ledger = Ledger::open(state.root(), "hwm");
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up after the store recovered"
    );
    assert_eq!(
        named_blobs_missing_from(&mirror_dir),
        Vec::<String>::new(),
        "the mirror's ledger still names blobs the mirror does not hold"
    );
    // Including the record from the batch the failed push was in the middle of, and the one it
    // never reached.
    for value in ["0.5", "0.4", "0.3", "0.25"] {
        let sha = jope_learning::canon::sha256_hex(format!("d0/cut-0/{value}").as_bytes());
        assert!(
            mirror_dir.join(State::blob_rel(&sha)).exists(),
            "blob of the {value} record is missing from the mirror"
        );
    }
    let key = pointer_key();
    assert_eq!(
        fs::read(mirror_dir.join(key.rel())).unwrap(),
        fs::read(state.pointer_path(&key).unwrap()).unwrap(),
        "the pointer did not catch up"
    );
}

/// A blob a batch names and the **local** store cannot supply must never be skipped.
///
/// The round-three ordering guarantees that while a blob is missing from the *mirror*, the
/// mirror's `HEAD.json` has not moved past the batch that names it. It says nothing about a blob
/// missing from the *local* store, and that hole was wide enough to falsify the whole guarantee:
/// `state.read_blob` returning `None` simply skipped the upload, the batch object went up anyway,
/// `HEAD.json` moved past it, and `sync::plan` — which derives the backfill window from that
/// `HEAD.json` and nothing else — never offered that batch again. The mirror was left naming a
/// record that neither host could supply, permanently, at exit 0.
///
/// Fails on the pre-fix code at the first assertion: step 4 exits `Ok`, the mirror's tip advances
/// to `batches: 3`, and the mirror's own batch events name ten record blobs while it holds nine.
#[test]
fn a_blob_the_local_store_cannot_supply_is_never_published_over() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);
    let mirror_head = mirror.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the five blobs it names.
    ingest(&state, Some(&url), &dir_a, "a").ok();
    let head_1 = fs::read(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror.join("hwm/ledger"));

    // 2. local only: the mirror is a batch behind, and has never seen that batch's blobs.
    ingest(&state, None, &dir_b, "b").ok();

    // 3. one blob local batch 2 names is lost from the *local* store — a pruned cache, a bad
    //    disk, an operator `rm`. The ledger still names it; `hwm verify` calls it `blob-missing`.
    let ledger = Ledger::open(&state, "hwm");
    let batch_2 = ledger.batches().unwrap().into_iter().next_back().unwrap().1;
    let (events, _, _, _) = Ledger::read_batch(&batch_2).unwrap();
    let lost = events[0]["record"]["provenance"]["record_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let lost_path = State::at(&state).blob_path(&lost);
    let lost_bytes = fs::read(&lost_path).unwrap();
    fs::remove_file(&lost_path).unwrap();

    // 4. mirrored again. The push must refuse: publishing batch 2 would move the mirror's tip
    //    past a batch whose blob nobody holds, and no later push would ever come back for it.
    let r = ingest(&state, Some(&url), &dir_a, "a");
    assert_ne!(
        r.status,
        0,
        "a blob no one can supply was published over; stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains(&lost)
            && r.stderr
                .contains("Refusing to publish a batch the mirror could not back"),
        "the refusal must name the blob it could not supply, got: {}",
        r.stderr
    );
    assert_eq!(
        named_blobs_missing_from(&mirror),
        Vec::<String>::new(),
        "the mirror's own ledger names a blob the mirror does not hold"
    );
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over a batch naming a blob nobody can supply"
    );
    assert_eq!(
        tree(&mirror.join("hwm/ledger")),
        ledger_1,
        "a batch object went up before the blobs it names could be supplied"
    );

    // 5. the blob is restored from a bundle. The gap is still the gap, so the next push closes it.
    fs::write(&lost_path, &lost_bytes).unwrap();
    ingest(&state, Some(&url), &dir_a, "a").ok();
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up once the blob was restored"
    );
    assert_eq!(named_blobs_missing_from(&mirror), Vec::<String>::new());
    assert_eq!(
        tree(&mirror.join("blobs")),
        tree(&state.join("blobs")),
        "every blob the mirrored ledger names must be on the mirror"
    );
}

/// The negative control for the refusal above: a blob only the *mirror* still has is not a gap,
/// and must not wedge the push.
///
/// Failing closed on `read_blob == None` alone would turn every pruned local blob into a mirror
/// that can never be written again — the same one-way door the reconcile was built to remove. The
/// batch is publishable exactly when *someone* can back it, so the local store is asked first and
/// the mirror second.
#[test]
fn a_blob_only_the_mirror_still_has_does_not_wedge_the_push() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));

    // Both records are on the mirror, pushed while the local store still had their blobs.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    hwm::ingest(&state, Some(&m), vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // A local-only batch of one tie: it names the 0.4 record again, and nothing else.
    hwm::ingest(&state, None, vec![candidate(0.4, 3)], json!({"k": 3})).unwrap();

    // The local blob for 0.4 is pruned. The mirror still has it, so the backfilled batch is
    // already backed and the push has nothing to upload for it.
    let sha = jope_learning::canon::sha256_hex(b"d0/cut-0/0.4");
    assert!(mirror_dir.join(State::blob_rel(&sha)).exists());
    fs::remove_file(state.blob_path(&sha)).unwrap();

    let rep = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 4)], json!({"k": 4}))
        .expect("a blob the mirror already holds must not wedge the push");
    let ledger = Ledger::open(state.root(), "hwm");
    assert_eq!(
        fs::read(mirror_dir.join("hwm/ledger/HEAD.json")).unwrap(),
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        "the mirror did not catch up"
    );
    assert_eq!(named_blobs_missing_from(&mirror_dir), Vec::<String>::new());
    assert_eq!(rep.accepted, 1);
}

/// The mirror-only fallback must prove **identity**, not presence.
///
/// `read_blob` re-hashes what it reads and `put_immutable` compares bytes on an `Exists`, so every
/// other way a record enters this system is checked against the sha that names it. The fallback
/// was the one path that accepted an object this host never wrote, and it accepted it on the
/// strength of a successful GET alone. One wrong object under the immutable key — a truncated
/// upload, a key reused by an unrelated writer, a restore from the wrong bundle — would then
/// launder itself into a published, `HEAD`-reachable batch that no reader can verify, and
/// `hwm verify` against the mirror reports `blob-corrupt` for a record the push called backed.
///
/// Fails on the pre-fix code at the first assertion: the ingest returns `Ok` and the mirror's tip
/// advances over a batch whose record the mirror does not actually hold.
#[test]
fn a_wrong_object_under_the_immutable_key_cannot_stand_in_for_the_blob() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mirror_head = mirror_dir.join("hwm/ledger/HEAD.json");

    // 1. mirrored: the mirror holds batch 1 and the blob it names.
    hwm::ingest(&state, Some(&m), vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    let head_1 = fs::read(&mirror_head).unwrap();
    let ledger_1 = tree(&mirror_dir.join("hwm/ledger"));

    // 2. local only: the mirror is a batch behind and has never seen that batch's blob.
    hwm::ingest(&state, None, vec![candidate(0.4, 2)], json!({"k": 2})).unwrap();

    // 3. the local blob is lost, and something else is sitting under its immutable key on the
    //    mirror. Nothing on this path re-reads it, so nothing has ever checked it.
    let sha = jope_learning::canon::sha256_hex(b"d0/cut-0/0.4");
    fs::remove_file(state.blob_path(&sha)).unwrap();
    let rel = State::blob_rel(&sha);
    let wrong = b"wrong bytes under the required immutable key".to_vec();
    fs::create_dir_all(mirror_dir.join(&rel).parent().unwrap()).unwrap();
    fs::write(mirror_dir.join(&rel), &wrong).unwrap();

    // 4. the push must refuse: a GET that returns *something* is not proof the mirror can back
    //    the record.
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.3, 3)], json!({"k": 3}))
        .expect_err("mirror blob presence was taken as proof of content identity");
    let msg = format!("{err:#}");
    assert!(
        msg.contains(&sha) && msg.contains("not this record"),
        "the refusal must name the record and say the mirror's object is not it, got: {msg}"
    );
    assert_eq!(
        fs::read(&mirror_head).unwrap(),
        head_1,
        "the mirror's tip moved over a batch the mirror cannot back"
    );
    assert_eq!(
        tree(&mirror_dir.join("hwm/ledger")),
        ledger_1,
        "a batch object went up although its record could not be supplied"
    );
    // The refusal does not "repair" the mirror by overwriting an immutable key: whatever is there
    // is another writer's business, and this host has nothing to replace it with anyway.
    assert_eq!(
        fs::read(mirror_dir.join(&rel)).unwrap(),
        wrong,
        "the push overwrote an immutable key to make itself succeed"
    );

    // 5. restore the real blob locally. The correct bytes now conflict with the wrong object, and
    //    `put_immutable` refuses that too rather than silently replacing it — the push still fails
    //    closed, and still leaves the tip and the foreign object alone.
    state.put_blob(b"d0/cut-0/0.4").unwrap();
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.25, 4)], json!({"k": 4}))
        .expect_err("an immutable key holding different bytes must not be overwritten");
    assert!(
        format!("{err:#}").contains("different bytes"),
        "expected the immutable-collision refusal, got: {err:#}"
    );
    assert_eq!(fs::read(&mirror_head).unwrap(), head_1);
    assert_eq!(fs::read(mirror_dir.join(&rel)).unwrap(), wrong);
}
