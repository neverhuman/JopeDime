//! What the self-contained chain check can and cannot see.
//!
//! `hwm verify` exists to report a tampered ledger, so the two ways it can fail an operator are a
//! tamper it *dies* on and a tamper it *misses*. One of each was found at `5737aca`:
//!
//! * a seal declaring `u64::MAX` events reached `Vec::with_capacity` before the count had been
//!   reconciled with the file, and aborted the process with `capacity overflow` — walking straight
//!   past `verify`'s deliberate conversion of a `read_batch` failure into a `Break`, and taking
//!   `require_intact` (and therefore the write path) down with it. That is now an ordinary break.
//! * a removed tail with `HEAD.json` rewritten to match is *not* detected, and the module doc used
//!   to claim it was. The doc is now accurate and this file pins the real behaviour, including the
//!   external anchor that does catch it.
//!
//! And one way the *export* could launder a broken chain: `hwm freeze` used to take `HEAD.json` at
//! its word and stamp it into a manifest as provenance.

mod common;

use common::{fixtures, jl};
use jope_learning::hwm::{self, Candidate};
use jope_learning::ledger::{Head, Ledger};
use jope_learning::metric::Metric;
use jope_learning::mirror::Mirror;
use jope_learning::record::*;
use jope_learning::state::State;
use serde_json::json;
use std::fs;

const BOUND: &str = "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393";

fn candidate(value: f64, ts: u64) -> Candidate {
    let identity = HwmIdentity {
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

/// Rewrite the seal's `events` count in place, leaving everything else byte-identical.
fn forge_event_count(path: &std::path::Path, declared: &str) {
    let text = fs::read_to_string(path).unwrap();
    let cut = text[..text.len() - 1].rfind('\n').map_or(0, |i| i + 1);
    let (body, seal) = text.split_at(cut);
    let mut v: serde_json::Value = serde_json::from_str(seal.trim_end()).unwrap();
    v["events"] = serde_json::from_str(declared).unwrap();
    fs::write(
        path,
        format!("{body}{}\n", serde_json::to_string(&v).unwrap()),
    )
    .unwrap();
}

/// Fails on the pre-fix code by **aborting**: `rc=101`, `thread 'main' panicked … capacity
/// overflow`, on `hwm verify` and on the ingest path alike.
#[test]
fn a_seal_declaring_an_impossible_event_count_is_a_break_not_an_abort() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));
    hwm::ingest(&state, None, vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    let batch = Ledger::open(state.root(), "hwm").batches().unwrap()[0]
        .1
        .clone();
    let original = fs::read(&batch).unwrap();

    // Counts no file of this size could ever hold: refused before the allocation.
    for declared in ["18446744073709551615", "1000000000"] {
        forge_event_count(&batch, declared);

        // The library call fails rather than aborting, and says what is wrong with the file.
        let err = Ledger::read_batch(&batch)
            .expect_err("a seal count the file cannot hold must not be trusted");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(declared) && msg.contains("cannot hold"),
            "{declared}: expected the count refusal, got: {msg}"
        );

        // `hwm verify` reports it the way it reports every other malformed batch: exit 1 with a
        // break naming the batch — not exit 101 with a panic.
        let r = jl(state.root(), &["hwm", "verify"]);
        assert_eq!(
            r.status, 1,
            "{declared}: hwm verify exited {} (101 = it aborted); stderr: {}",
            r.status, r.stderr
        );
        let v = r.json();
        let breaks = v["ledger"]["breaks"].as_array().unwrap();
        assert!(
            breaks.iter().any(|b| b["kind"] == "unreadable"
                && b["detail"].as_str().is_some_and(|d| d.contains(declared))),
            "{declared}: expected an `unreadable` break naming the count, got: {breaks:?}"
        );

        // And the write path refuses through `require_intact` rather than aborting under it.
        let champions = fixtures().join("champions");
        let r = jl(
            state.root(),
            &[
                "hwm",
                "ingest",
                "--source",
                "champions",
                champions.to_str().unwrap(),
                "--bundle-id",
                "probe",
            ],
        );
        assert_eq!(
            r.status, 2,
            "{declared}: hwm ingest exited {} (101 = it aborted); stderr: {}",
            r.status, r.stderr
        );
        assert!(
            r.stderr.contains("chain broken"),
            "{declared}: expected the chain-broken refusal, got: {}",
            r.stderr
        );
    }

    // A merely *wrong* count is a different thing and keeps its existing treatment: the file can
    // hold two events, so it is read, and the cross-check in `Ledger::verify` reports the
    // mismatch. The bound added here is a capacity sanity check, not the count check, and it must
    // not quietly take the count check's place.
    forge_event_count(&batch, "2");
    assert_eq!(
        Ledger::read_batch(&batch).unwrap().0.len(),
        1,
        "a plausible-but-wrong count must still be readable"
    );
    let breaks = Ledger::open(state.root(), "hwm").verify().unwrap().breaks;
    assert!(
        breaks.iter().any(|b| b.kind == "event-count"),
        "the count cross-check was lost: {breaks:?}"
    );

    // The control: the untouched batch still verifies, so the bound did not become a new refusal
    // of honest batches. (Restoring the original *bytes* rather than rewriting the count back,
    // because re-serialising the seal changes the file hash `HEAD.json` recorded.)
    fs::write(&batch, &original).unwrap();
    assert_eq!(Ledger::read_batch(&batch).unwrap().0.len(), 1);
    assert!(hwm::verify(&state, &[]).unwrap().ok());
}

/// A removed tail with `HEAD.json` rewritten to match is **not** detectable from the tree alone —
/// and the mirror is what catches it.
///
/// This is a characterization test, not a fix: it passes before and after, and it is here because
/// the property it pins used to be documented backwards (`ledger.rs` claimed a truncated batch
/// "shows up as a break"). `HEAD.json` is the only anchor and it lives inside the directory it
/// anchors, so no hash in the tree can notice history that is no longer in the tree. Writing the
/// real behaviour down as an executable assertion is what stops the doc drifting again, and the
/// second half names the external anchor that does detect it.
#[test]
fn a_removed_tail_is_invisible_locally_and_refused_by_the_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror_dir = tmp.path().join("mirror");
    fs::create_dir_all(&mirror_dir).unwrap();
    let m = Mirror::open(&format!("file://{}", mirror_dir.display())).unwrap();
    let state = State::at(&tmp.path().join("state"));

    for (i, v) in [0.5, 0.4, 0.3].into_iter().enumerate() {
        hwm::ingest(
            &state,
            Some(&m),
            vec![candidate(v, i as u64 + 1)],
            json!({"k": i}),
        )
        .unwrap();
    }
    let ledger = Ledger::open(state.root(), "hwm");
    let batches = ledger.batches().unwrap();
    assert_eq!(batches.len(), 3);
    assert!(hwm::verify(&state, &[]).unwrap().ok());

    // Erase the two newest batches and rewrite HEAD.json to name the survivor, with its real file
    // hash and corrected counts — the tamper a careful hand would make.
    for (_, path) in &batches[1..] {
        fs::remove_file(path).unwrap();
    }
    let (events, seal, file_sha, _) = Ledger::read_batch(&batches[0].1).unwrap();
    fs::write(
        ledger.root().join("HEAD.json"),
        serde_json::to_vec_pretty(&Head {
            schema: jope_learning::ledger::HEAD_SCHEMA.into(),
            ledger: "hwm".into(),
            batch: seal.batch.clone(),
            path: batches[0].0.clone(),
            sha256: file_sha,
            batches: 1,
            events_total: events.len() as u64,
        })
        .unwrap(),
    )
    .unwrap();

    // Locally: a clean bill of health. Eight events are gone and nothing in the tree can say so.
    let report = hwm::verify(&state, &[]).unwrap();
    assert!(
        report.ledger.ok() && report.ledger.breaks.is_empty(),
        "local verify unexpectedly detected the truncation: {:?}",
        report.ledger.breaks
    );
    let r = jl(state.root(), &["hwm", "verify"]);
    assert_eq!(
        (
            r.status,
            r.json()["ledger"]["breaks"].as_array().unwrap().len()
        ),
        (1, 0),
        "the documented limit changed; update ledger.rs, docs/learning.md and the runbook"
    );
    // (exit 1 here is the *pointer* half — the two erased batches moved pointers that are still on
    // disk — not the chain. The chain itself reports zero breaks, which is the point.)

    // Externally: the mirror still holds the tip this host can no longer produce, so the push is
    // refused rather than silently orphaning it. That is the anchor the docs now name.
    let err = hwm::ingest(&state, Some(&m), vec![candidate(0.2, 9)], json!({"k": 9}))
        .expect_err("a truncated host must not be able to push over the mirror's tip");
    assert!(
        format!("{err:#}").contains("holds a chain this host cannot continue"),
        "expected the divergence refusal, got: {err:#}"
    );
    // The mirror kept its own tip, untouched.
    let mirror_head: Head =
        serde_json::from_slice(&fs::read(mirror_dir.join("hwm/ledger/HEAD.json")).unwrap())
            .unwrap();
    assert_eq!(mirror_head.batches, 3);
}

/// `hwm freeze` must verify the chain before exporting from it.
///
/// `freeze` was the only path in the crate that wrote without a chain check: `ingest` and
/// `scores::append` call `require_intact`, `find` and `rows` go through the fail-closed
/// `events()`, and `freeze` parsed `HEAD.json`, walked the pointers and wrote a manifest carrying
/// `batches` and `events_total` it had never checked. The blobs were never the problem —
/// `read_blob` re-hashes each one it copies — but a frozen corpus is the artifact a later reader
/// trusts *without* re-deriving it, and `ledger_head` in that manifest reads as provenance for the
/// chain it came from. That is precisely the claim a corpus consumer cannot check for itself.
///
/// Fails on the pre-fix code at the first assertion: the export exits `Ok` on a tree whose
/// `hwm verify` reports breaks, and writes a `manifest.json` stamped with the broken chain's
/// `batches`/`events_total`.
#[test]
fn freeze_refuses_to_export_from_a_chain_it_cannot_walk() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));
    for (i, v) in [0.5, 0.4, 0.3].into_iter().enumerate() {
        hwm::ingest(
            &state,
            None,
            vec![candidate(v, i as u64 + 1)],
            json!({"k": i}),
        )
        .unwrap();
    }
    let out = tmp.path().join("freeze");

    // The control first: a clean tree exports, so the check is not a blanket refusal.
    let clean = hwm::freeze(&state, &out, false).unwrap();
    assert_eq!((clean.pointers, clean.included), (1, 1));
    assert!(out.join("manifest.json").is_file());
    fs::remove_dir_all(&out).unwrap();

    // Break the chain the way a tamper does, and confirm the verifier sees it.
    let batches = Ledger::open(state.root(), "hwm").batches().unwrap();
    forge_event_count(&batches[1].1, "18446744073709551615");
    let report = hwm::verify(&state, &[]).unwrap();
    assert!(
        !report.ok() && report.total_breaks() > 0,
        "the fixture is only meaningful if verify reports breaks"
    );

    // The export must refuse, and say what it refused and why.
    let err = hwm::freeze(&state, &out, false)
        .expect_err("a corpus was exported from a chain this host cannot walk");
    let msg = format!("{err:#}");
    for needle in [
        "refusing to freeze into",
        "nothing was written",
        "hwm verify",
        "chain broken",
        "break(s)",
        "unreadable",
    ] {
        assert!(
            msg.contains(needle),
            "the refusal must be legible — missing {needle:?} in: {msg}"
        );
    }
    assert!(
        !out.join("manifest.json").exists(),
        "a manifest was written for a chain that does not verify"
    );

    // Through the binary, the same: exit 2, not a silent 0 with an exported corpus.
    let out2 = tmp.path().join("freeze-cli");
    let r = jl(
        state.root(),
        &["hwm", "freeze", "--out", out2.to_str().unwrap()],
    );
    assert_eq!(
        r.status,
        2,
        "hwm freeze exited {} on a broken chain; stdout: {}",
        r.status,
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(r.stderr.contains("refusing to freeze into"), "{}", r.stderr);
    assert!(
        !out2.join("manifest.json").exists(),
        "the CLI wrote a manifest for a chain that does not verify"
    );
    // `hwm verify` still works on the same tree — the export refusing must not take the
    // diagnostic down with it.
    assert_eq!(jl(state.root(), &["hwm", "verify"]).status, 1);
}

/// A destination path under `blobs/` is content-addressed, but a path existing is not the same
/// claim as its bytes hashing to the name it sits under. `freeze` skipped the write whenever the
/// destination existed, so a corrupted or planted export was published under a manifest asserting
/// `record_sha256` over it — and `verify`, which walks the *source* ledger, is structurally unable
/// to notice, because the source is intact. Reported on PR #34 by the independent reviewer.
#[test]
fn freeze_refuses_a_destination_blob_that_does_not_hash_to_the_path_it_sits_under() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));
    for (i, v) in [0.5, 0.4, 0.3].into_iter().enumerate() {
        hwm::ingest(
            &state,
            None,
            vec![candidate(v, i as u64 + 1)],
            json!({"k": i}),
        )
        .unwrap();
    }
    let out = tmp.path().join("freeze");

    // Control 1: a clean export writes the blob and a manifest.
    let first = hwm::freeze(&state, &out, false).unwrap();
    assert_eq!(first.included, 1);
    let blob = out.join(&first.entries[0].blob);
    let expected_sha = first.entries[0].record_sha256.clone();
    assert!(blob.is_file(), "the control export wrote no blob");
    let original = fs::read(&blob).unwrap();

    // Control 2: re-exporting the same corpus over the same tree stays idempotent. Without this
    // the refusal below could be satisfied by simply banning every re-export.
    let again = hwm::freeze(&state, &out, false).unwrap();
    assert_eq!(again.included, first.included);
    assert_eq!(
        fs::read(&blob).unwrap(),
        original,
        "an idempotent re-export disturbed a blob whose bytes already agreed"
    );

    // The tamper: the path still claims `expected_sha`; the bytes under it no longer hash to it.
    fs::write(&blob, b"not the champion these bytes claim to be").unwrap();

    // The source ledger is untouched, which is the whole point — destination integrity is what
    // freeze is being asked to establish, and no amount of source verification can supply it.
    assert!(
        hwm::verify(&state, &[]).unwrap().ok(),
        "the fixture is only meaningful while the source ledger still verifies clean"
    );

    let err = hwm::freeze(&state, &out, false)
        .expect_err("a manifest was published over a blob that does not hash to its own path");
    let msg = format!("{err:#}");
    for needle in [
        "refusing to freeze",
        "already holds bytes hashing to",
        expected_sha.as_str(),
    ] {
        assert!(
            msg.contains(needle),
            "the refusal must be legible — missing {needle:?} in: {msg}"
        );
    }
}
