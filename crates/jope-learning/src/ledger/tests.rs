use super::*;

/// The ordering `Ledger::batches()` sorts by, pinned without depending on machine speed.
///
/// A `prev` stamped in the future is one a fresh `Ulid::new()` cannot beat, so the clamp in
/// `next_batch_id` is the only thing that can satisfy this -- no timing, no retries, no
/// "usually passes". Reverting the clamp fails this test every run on every machine.
#[test]
fn a_batch_id_is_always_greater_than_the_one_it_chains_onto() {
    let now = ulid::Ulid::new();

    // Unreachable by the wall clock for the next minute.
    let future = ulid::Ulid::from_parts(now.timestamp_ms() + 60_000, 0);
    let next = next_batch_id(Some(&future.to_string())).unwrap();
    assert!(
        next > future,
        "{next} does not follow {future}; batches() would sort them backwards"
    );
    assert!(
        next.to_string() > future.to_string(),
        "ids order differently as strings than as ULIDs, and batches() sorts strings"
    );

    // Repeated against its own output: a run of appends inside one tick must keep climbing.
    let mut prev = future;
    for _ in 0..1000 {
        let cur = next_batch_id(Some(&prev.to_string())).unwrap();
        assert!(cur.to_string() > prev.to_string(), "{cur} <= {prev}");
        prev = cur;
    }

    // An ordinary prev, and the two degenerate inputs, must not panic.
    assert!(next_batch_id(Some(&now.to_string())).unwrap() > now);
    assert!(next_batch_id(None).is_ok());
    assert!(
        next_batch_id(Some("not-a-ulid")).is_ok(),
        "an unparseable id must leave verify() to report the malformed chain, not panic here"
    );
}

/// Batches sealed inside the same millisecond must still verify.
///
/// `Ulid::new()` is not monotonic within a millisecond -- it randomises the low 80 bits -- so
/// batch names sorted lexicographically came out in a different order than they were appended
/// whenever the host was fast enough to seal two in one tick. `verify()` walks prev-links in
/// that order, so it reported `prev-link` BREAK on a ledger nobody had touched. It surfaced as
/// a CI failure on the NVMe runner while passing on slower disks, which is the worst shape for
/// a bug in a tamper-evident structure: it accuses when it is fastest.
#[test]
fn batches_sealed_within_one_millisecond_still_verify() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path(), "hwm");

    // NOTE: whether this actually seals two batches inside one millisecond depends on how
    // fast the disk is -- on a slow one every append takes longer than a tick and the bug
    // cannot appear. So this is an end-to-end guard, not the regression test; the property
    // itself is pinned deterministically by
    // `a_batch_id_is_always_greater_than_the_one_it_chains_onto` below.
    let mut ids = Vec::new();
    for n in 0..64u64 {
        ids.push(
            ledger
                .append(&[json!({"n": n})], json!({"t": n}))
                .unwrap()
                .batch,
        );
    }

    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(
        ids, sorted,
        "batch ids are not in append order, so batches() cannot sort by them"
    );
    assert_eq!(
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        ids.len(),
        "two batches were given the same id"
    );

    let report = ledger.verify().unwrap();
    assert!(
        report.ok(),
        "an untouched ledger reported breaks: {:?}",
        report.breaks
    );
    assert_eq!((report.batches, report.events), (64, 64));

    // And the events come back in the order they were appended, which is the property the
    // name ordering exists to provide.
    assert_eq!(
        ledger
            .events()
            .unwrap()
            .iter()
            .map(|e| e["n"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        (0..64).collect::<Vec<_>>()
    );
}

use super::*;
use serde_json::json;

fn numbers(ledger: &Ledger) -> Vec<u64> {
    ledger
        .events()
        .unwrap()
        .iter()
        .map(|e| e["n"].as_u64().unwrap())
        .collect()
}

/// Append batches 1 and 2, then put back batch 1's `HEAD.json`. This is the state a process
/// leaves when it dies after batch 2's atomic rename but before `HEAD.json`'s: a complete,
/// sealed batch that the head does not name. Returns (ledger dir, head 1, head 2).
fn crashed_between_batch_and_head() -> (tempfile::TempDir, Vec<u8>, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path(), "hwm");
    ledger.append(&[json!({"n": 1})], json!({"t": 1})).unwrap();
    let head_1 = fs::read(ledger.root().join("HEAD.json")).unwrap();
    ledger.append(&[json!({"n": 2})], json!({"t": 2})).unwrap();
    let head_2 = fs::read(ledger.root().join("HEAD.json")).unwrap();
    fs::write(ledger.root().join("HEAD.json"), &head_1).unwrap();
    (dir, head_1, head_2)
}

/// W8-T06: a crash between a batch's rename and its `HEAD.json`'s must not wedge the ledger.
///
/// Every reader already treats the batch as part of the ledger (`events()` walks batch
/// files). `require_intact`, the gate in front of every write, reported `head-mismatch` and
/// refused for ever, so each later `hwm ingest`, `scores append` and eligibility decision
/// failed until someone repaired the tree by hand.
#[test]
fn a_crash_between_a_batch_and_its_head_is_rolled_forward_not_a_wedge() {
    let (dir, _, head_2) = crashed_between_batch_and_head();
    let ledger = Ledger::open(dir.path(), "hwm");
    assert_eq!(numbers(&ledger), [1, 2], "readers already see batch 2");
    let report = ledger
        .require_intact()
        .expect("a crash-lagged HEAD.json wedged the ledger");
    assert_eq!((report.batches, report.events), (2, 2));
    assert_eq!(
        fs::read(ledger.root().join("HEAD.json")).unwrap(),
        head_2,
        "the head rolled forward to exactly what the interrupted append would have written"
    );
    ledger.append(&[json!({"n": 3})], json!({"t": 3})).unwrap();
    let report = ledger.verify().unwrap();
    assert!(report.ok(), "{:?}", report.breaks);
    assert_eq!(numbers(&ledger), [1, 2, 3]);
}

/// The same crash, followed directly by another append, must not fork the chain: the next
/// batch has to chain onto batch 2, not onto the stale head's batch 1.
#[test]
fn an_append_after_a_lagging_head_never_forks_the_chain() {
    let (dir, _, _) = crashed_between_batch_and_head();
    let ledger = Ledger::open(dir.path(), "hwm");
    ledger.append(&[json!({"n": 3})], json!({"t": 3})).unwrap();
    let report = ledger.verify().unwrap();
    assert!(
        report.ok(),
        "an append chained onto a stale head and forked the ledger: {:?}",
        report.breaks
    );
    assert_eq!((report.batches, report.events), (3, 3));
    assert_eq!(numbers(&ledger), [1, 2, 3]);
}

/// Only a complete, sealed continuation of the head is rolled forward. Anything else is still
/// a break that refuses, and nothing is written.
#[test]
fn only_a_sealed_continuation_of_the_head_is_rolled_forward() {
    // The unnamed batch was altered after its rename.
    let (dir, head_1, _) = crashed_between_batch_and_head();
    let ledger = Ledger::open(dir.path(), "hwm");
    let (_, tip) = ledger.batches().unwrap().pop().unwrap();
    let mut bytes = fs::read(&tip).unwrap();
    let i = bytes.iter().position(|&b| b == b'2').unwrap();
    bytes[i] = b'7';
    fs::write(&tip, &bytes).unwrap();
    assert!(
        ledger.require_intact().is_err(),
        "an altered batch was adopted"
    );
    assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head_1);

    // A batch that does not chain onto the head: its predecessor is missing.
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path(), "hwm");
    ledger.append(&[json!({"n": 1})], json!({})).unwrap();
    let head_1 = fs::read(ledger.root().join("HEAD.json")).unwrap();
    let second = ledger.append(&[json!({"n": 2})], json!({})).unwrap();
    ledger.append(&[json!({"n": 3})], json!({})).unwrap();
    fs::remove_file(ledger.root().join(&second.path)).unwrap();
    fs::write(ledger.root().join("HEAD.json"), &head_1).unwrap();
    assert!(
        ledger.require_intact().is_err(),
        "a gapped suffix was adopted"
    );
    assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head_1);

    // A head that does not match the batch it names.
    let (dir, head_1, _) = crashed_between_batch_and_head();
    let ledger = Ledger::open(dir.path(), "hwm");
    let mut forged: Head = serde_json::from_slice(&head_1).unwrap();
    forged.sha256 = "0".repeat(64);
    let forged = serde_json::to_vec_pretty(&forged).unwrap();
    fs::write(ledger.root().join("HEAD.json"), &forged).unwrap();
    assert!(
        ledger.require_intact().is_err(),
        "a forged head was rolled forward"
    );
    assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), forged);

    // Positive control: the very first append died before any HEAD.json was written.
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path(), "hwm");
    ledger.append(&[json!({"n": 1})], json!({})).unwrap();
    let head = fs::read(ledger.root().join("HEAD.json")).unwrap();
    fs::remove_file(ledger.root().join("HEAD.json")).unwrap();
    assert!(ledger.require_intact().is_ok());
    assert_eq!(fs::read(ledger.root().join("HEAD.json")).unwrap(), head);
}

#[test]
fn chain_appends_and_verifies_and_detects_tampering() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path(), "hwm");
    assert!(ledger.head().unwrap().is_none());
    assert!(ledger.verify().unwrap().ok());
    let r1 = ledger
        .append(&[json!({"n": 1}), json!({"n": 2})], json!({"t": "a"}))
        .unwrap();
    let r2 = ledger
        .append(&[json!({"n": 3})], json!({"t": "b"}))
        .unwrap();
    let r3 = ledger
        .append(&[json!({"n": 4})], json!({"t": "c"}))
        .unwrap();
    assert_eq!(r2.prev_batch_sha256.as_deref(), Some(r1.sha256.as_str()));
    assert_eq!(r3.prev_batch_sha256.as_deref(), Some(r2.sha256.as_str()));
    let report = ledger.verify().unwrap();
    assert!(report.ok(), "{:?}", report.breaks);
    assert_eq!((report.batches, report.events), (3, 4));
    assert_eq!(
        ledger
            .events()
            .unwrap()
            .iter()
            .map(|e| e["n"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    assert!(ledger.append(&[], json!({})).is_err());

    // Tamper with one byte of an event in the middle batch.
    let p2 = ledger.root().join(&r2.path);
    let mut bytes = fs::read(&p2).unwrap();
    let i = bytes.iter().position(|&b| b == b'3').unwrap();
    bytes[i] = b'9';
    fs::write(&p2, &bytes).unwrap();
    let report = ledger.verify().unwrap();
    let kinds: Vec<&str> = report.breaks.iter().map(|b| b.kind.as_str()).collect();
    assert!(kinds.contains(&"events-sha256"), "{kinds:?}");
    assert!(kinds.contains(&"prev-link"), "{kinds:?}");
    assert!(ledger.require_intact().is_err());

    // Restore, then drop the tip: HEAD dangles.
    fs::write(&p2, {
        bytes[i] = b'3';
        &bytes
    })
    .unwrap();
    assert!(ledger.verify().unwrap().ok());
    fs::remove_file(ledger.root().join(&r3.path)).unwrap();
    let kinds: Vec<String> = ledger
        .verify()
        .unwrap()
        .breaks
        .into_iter()
        .map(|b| b.kind)
        .collect();
    assert!(kinds.contains(&"head-mismatch".to_string()), "{kinds:?}");
}
