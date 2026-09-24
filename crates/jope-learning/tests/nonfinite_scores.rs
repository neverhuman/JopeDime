//! A candidate whose score is not a finite number must never reach the append-only ledger, and a
//! ledger that already holds one must say so in terms an operator can act on.
//!
//! JSON has no NaN and no infinity. `serde_json` writes both as `null` and refuses to read them
//! back, so an `HwmEvent` carrying one is not merely wrong, it is unreadable — and because the
//! ledger is append-only and hash-chained, it cannot be edited out. One such event turned every
//! later `hwm::events()` call into `invalid type: null, expected f64`: a single bad candidate,
//! rejected on its merits, made the whole ledger unreadable.
//!
//! The fix is at the point of entry, not in the schema: `jope.hwm/v1` keeps its `number` score and
//! no new outcome is invented. A non-finite candidate is refused before the lock, before the
//! mirror preflight and before any blob — nothing is written that cannot be read back, and no
//! fabricated replacement score is recorded in place of the one that was measured. Candidates that
//! are merely *not better* are unaffected: they are representable, so they keep the readable
//! rejection event they have always had.

use jope_learning::hwm::{self, Candidate};
use jope_learning::ledger::Ledger;
use jope_learning::metric::Metric;
use jope_learning::record::*;
use jope_learning::state::State;
use serde_json::json;

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

/// Every file under `root`, relative to it.
fn files(root: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            e.path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    v.sort();
    v
}

/// Fails on the pre-fix code at the first assertion: the ingest returns `Ok`, having written a
/// batch whose event carries `"value": null`, and every later read of the ledger fails.
#[test]
fn a_non_finite_score_is_refused_before_anything_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));

    for (label, value) in [
        ("NaN", f64::NAN),
        ("+inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
    ] {
        let err = match hwm::ingest(&state, None, vec![candidate(value, 1)], json!({"k": label})) {
            Ok(r) => panic!("{label} was admitted: {r:?}"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not a finite number") && msg.contains("Nothing was written"),
            "{label}: expected the entry refusal, got: {msg}"
        );
        // The refusal names the candidate, so an operator can find the row in the source.
        assert!(
            msg.contains(&candidate(value, 1).record.identity_sha256),
            "{label}: the refusal must name the candidate, got: {msg}"
        );
        assert_eq!(
            files(state.root()),
            Vec::<String>::new(),
            "{label}: the refused ingest wrote something"
        );
    }

    // A good candidate lands, and is readable.
    hwm::ingest(&state, None, vec![candidate(0.5, 2)], json!({"k": "good"})).unwrap();
    let before = files(state.root());
    assert_eq!(hwm::events(&state).unwrap().len(), 1);

    // A batch that *contains* a non-finite candidate is refused whole: nothing half-written, and
    // the ledger stays exactly as it was.
    let err = hwm::ingest(
        &state,
        None,
        vec![candidate(0.4, 3), candidate(f64::NAN, 4)],
        json!({"k": "mixed"}),
    )
    .expect_err("a batch carrying a non-finite score was admitted");
    assert!(format!("{err:#}").contains("not a finite number"));
    assert_eq!(
        files(state.root()),
        before,
        "the refused batch wrote something"
    );
    assert_eq!(
        Ledger::open(state.root(), "hwm").batches().unwrap().len(),
        1,
        "the refused batch was sealed anyway"
    );

    // And the ledger is still readable — which is the whole point.
    assert_eq!(hwm::events(&state).unwrap().len(), 1);
    assert!(hwm::verify(&state, &[]).unwrap().ok());
}

/// Representable candidates that are merely *not better* keep their rejection event: the fix must
/// narrow the admitted set by exactly the unrepresentable ones and nothing else.
#[test]
fn an_ordinary_rejection_still_gets_its_readable_ledger_event() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));
    hwm::ingest(&state, None, vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();
    // A tie and an inferior candidate: both rejected, both written down, both readable.
    let rep = hwm::ingest(
        &state,
        None,
        vec![candidate(0.5, 2), candidate(0.9, 3)],
        json!({"k": 2}),
    )
    .unwrap();
    assert_eq!((rep.accepted, rep.rejected), (0, 2));
    let events = hwm::events(&state).unwrap();
    assert_eq!(events.len(), 3);
    let reasons: Vec<Option<&str>> = events.iter().skip(1).map(|e| e.reason.as_deref()).collect();
    assert_eq!(reasons, vec![Some("tie"), Some("not-better")]);
    assert!(hwm::verify(&state, &[]).unwrap().ok());
}

/// An already-poisoned ledger must give an error that names the offending event.
///
/// Fails on the pre-fix code at the `contains` assertions: the error is the bare
/// `parse hwm event: invalid type: null, expected f64`, which names no batch, no event, no
/// pointer, no identity and no field — on a twelve-thousand-event corpus that is not enough to
/// find the line, let alone decide what to do about it.
#[test]
fn a_poisoned_ledger_names_the_offending_event() {
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));

    // A healthy event first, so the report has to point at the right one of the two.
    hwm::ingest(&state, None, vec![candidate(0.5, 1)], json!({"k": 1})).unwrap();

    // Poison the ledger exactly the way the old ingest did: serialise an event whose record holds
    // a NaN. `serde_json` turns it into `"value": null` without complaint on the way out.
    let poisoned = candidate(f64::NAN, 2).record;
    let identity = poisoned.identity_sha256.clone();
    let record_sha = poisoned.provenance.record_sha256.clone();
    let event = json!({
        "schema": "jope.hwm-event/v1",
        "seq": 0,
        "outcome": "rejected",
        "reason": "invalid",
        "pointer": jope_learning::state::PointerKey::of(&poisoned),
        "record": poisoned,
        "displaced_record_sha256": null,
        "ts_unix": 1_700_000_000u64,
    });
    assert_eq!(
        event["record"]["score"]["value"],
        serde_json::Value::Null,
        "the fixture is only meaningful if serde really writes the NaN as null"
    );
    Ledger::open(state.root(), "hwm")
        .append(&[event], json!({"k": "poison"}))
        .unwrap();

    let err = hwm::events(&state).expect_err("a null score must not parse as an f64");
    let msg = format!("{err:#}");
    for needle in [
        "event #1",
        "regression/d0/",
        &identity,
        &record_sha,
        "record.score.value",
        "null",
        "append-only",
    ] {
        assert!(
            msg.contains(needle),
            "the error must name {needle:?}; got: {msg}"
        );
    }

    // `hwm verify` stays lenient — it is the tool an operator reaches for once a read has failed,
    // so it must survive the poison and report it rather than abort on it.
    let report = hwm::verify(&state, &[]).unwrap();
    assert!(!report.ok());
    let brk = report
        .breaks
        .iter()
        .find(|b| b.kind == "event-unparseable")
        .expect("verify must report the poisoned event");
    assert!(brk.detail.contains(&record_sha), "{}", brk.detail);
    assert!(brk.detail.contains("record.score.value"), "{}", brk.detail);
    // The healthy event is still counted: one bad line does not hide the rest of the ledger.
    assert_eq!(report.events, 1);
    // The reader left the ledger alone: both batches are still there, poison and all.
    assert_eq!(
        Ledger::open(state.root(), "hwm").batches().unwrap().len(),
        2
    );
}

/// The guard must be symmetric: `cost` is two more IEEE doubles on the way into the same ledgers.
///
/// A non-finite *score* makes a record unreadable, which is loud. A non-finite *cost* is worse in
/// the way that matters: `Option<f64>` serialises NaN to `null` and reads `null` back as `None`,
/// `row_sha256` still validates, and the number is simply gone with nothing to notice. It is
/// unreachable today only because `sotarena::parse_line` hardcodes `cost: None` — which is a
/// property of one source, not of the ledger.
///
/// Fails on the pre-fix code at the `ScoreRecord` assertions: `validate()` checked `value` and
/// never looked at `cost`.
#[test]
fn a_non_finite_cost_is_refused_on_both_record_types() {
    // jope.hwm/v1: the record's own structural check sees both fields...
    for (field, cost) in [
        (
            "cost.runtime_seconds",
            Cost {
                runtime_seconds: Some(f64::NAN),
                search_seconds: None,
            },
        ),
        (
            "cost.search_seconds",
            Cost {
                runtime_seconds: None,
                search_seconds: Some(f64::INFINITY),
            },
        ),
    ] {
        let mut r = candidate(0.5, 1).record;
        r.cost = cost;
        let err = r
            .validate()
            .err()
            .unwrap_or_else(|| panic!("{field} was accepted by HwmRecord::validate"));
        assert!(format!("{err:#}").contains(field), "{err:#}");
    }
    // ...and a finite cost still validates, so the guard did not become a blanket refusal.
    let mut ok = candidate(0.5, 1).record;
    ok.cost = Cost {
        runtime_seconds: Some(1.5),
        search_seconds: Some(0.0),
    };
    ok.validate().expect("a finite cost must still validate");

    // ...and it is refused at entry, before anything is written.
    let tmp = tempfile::tempdir().unwrap();
    let state = State::at(&tmp.path().join("state"));
    let mut c = candidate(0.5, 1);
    c.record.cost.search_seconds = Some(f64::NAN);
    let err = hwm::ingest(&state, None, vec![c], json!({"k": "cost"}))
        .expect_err("a non-finite cost was admitted");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cost.search_seconds") && msg.contains("Nothing was written"),
        "the refusal must name the field, got: {msg}"
    );
    assert_eq!(files(state.root()), Vec::<String>::new());

    // jope.score/v1: refused by the row's own structural check, which `scores::append` runs on
    // every row before it can become an event.
    let mut row = ScoreRecord {
        schema: SCORE_SCHEMA.into(),
        dataset_hash: "019de5d76a718163".into(),
        cut_id: "0".repeat(64),
        method_id: "jope-dime".into(),
        evaluator_sha256: UNBOUND.into(),
        protocol: "public-development".into(),
        metric: Metric::Rmse,
        value: 0.5,
        direction: Metric::Rmse.direction(),
        cost: None,
        benchmark_valid: true,
        receipt_sha256: "a".repeat(64),
        run_id: "run".into(),
        git_sha: UNBOUND.into(),
        ts_unix: 1,
        source: ScoreSource {
            kind: "test".into(),
            track: None,
            version: None,
            status: None,
            source_kind: None,
            source_revision: None,
            source_uri: None,
            notes: None,
        },
        row_sha256: String::new(),
    };
    row.row_sha256 = row.compute_row_sha256().unwrap();
    row.validate().expect("the control row must be valid");

    row.cost = Some(Cost {
        runtime_seconds: Some(f64::NAN),
        search_seconds: None,
    });
    row.row_sha256 = row.compute_row_sha256().unwrap();
    let err = row
        .validate()
        .expect_err("a score row with a non-finite cost was accepted");
    assert!(
        format!("{err:#}").contains("cost.runtime_seconds"),
        "{err:#}"
    );
    // And note *why* it has to be caught here: the row hash does not notice, because the NaN is
    // already `null` by the time it is hashed.
    assert_eq!(row.row_sha256, row.compute_row_sha256().unwrap());
}
