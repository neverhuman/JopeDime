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
