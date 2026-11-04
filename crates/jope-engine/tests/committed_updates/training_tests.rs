use super::*;
use serde_json::json;
use std::collections::BTreeSet;

#[test]
fn interrupted_and_uninterrupted_runs_consume_the_same_records_and_commit_the_same_state() {
    let base = tempfile::tempdir().unwrap();
    // Uninterrupted reference run.
    let straight = base.path().join("straight");
    let reference = attempt(&straight, 0, None);
    let reference_bytes = fs::read(&reference.journal).unwrap();
    let whole = consumption::reconcile(&reference_bytes, durable(&straight)).unwrap();
    assert_eq!(whole.committed_updates, TARGET);
    assert_eq!(whole.applied_updates, TARGET);
    assert_eq!(whole.uncommitted_applied_updates, 0);
    assert_eq!(whole.commit_events, TARGET / CHECKPOINT_EVERY);
    assert_eq!(whole.computed.microbatches, 2 * TARGET);
    assert_eq!(whole.committed.presentations, TARGET * GLOBAL as u64);
    assert_eq!(
        consumption::verify_records(&reference_bytes, sampler).unwrap(),
        2 * TARGET
    );

    // Interrupted run: step 4's update is applied but never committed.
    let broken = base.path().join("broken");
    let first = attempt(&broken, 1, Some(Stop::AfterApplied(4)));
    assert_eq!(first.applied_through, 5);
    let first_bytes = fs::read(&first.journal).unwrap();
    let one = consumption::reconcile(&first_bytes, durable(&broken)).unwrap();
    assert_eq!(
        (
            one.applied_updates,
            one.committed_updates,
            one.uncommitted_applied_updates
        ),
        (5, 4, 1)
    );
    assert_eq!(one.committed_through, 4);
    // The receipt alone reconstructs the next sample sequence.
    assert_eq!(one.next.optimizer_step, 4);
    assert_eq!(one.next.serial, 4 * GLOBAL as u64);
    let next = sampler(one.next.serial as usize, MICRO);

    // Recovery from the durable checkpoint.
    let second = attempt(&broken, 2, None);
    let second_bytes = fs::read(&second.journal).unwrap();
    let two = consumption::reconcile(&second_bytes, durable(&broken)).unwrap();
    let resumed = microbatches(&second.journal);
    assert_eq!(resumed[0].0, 4);
    assert_eq!(
        resumed[0].2,
        next.iter().map(|&r| r as u64).collect::<Vec<_>>(),
        "recovery did not consume the reconstructed next records"
    );
    let reference_mbs = microbatches(&reference.journal);
    let interrupted_mbs = microbatches(&first.journal)
        .into_iter()
        .filter(|(step, _, _)| *step < 4)
        .chain(resumed)
        .collect::<Vec<_>>();
    assert_eq!(
        interrupted_mbs, reference_mbs,
        "committed consumption differs from the uninterrupted run"
    );
    assert_eq!(
        parameters(&broken),
        parameters(&straight),
        "interrupted run reached a different committed state"
    );

    // Run totals keep computation, applied and committed updates apart.
    let run = consumption::reconcile_run(&[one.clone(), two.clone()]).unwrap();
    assert_eq!(run.committed_updates, TARGET);
    assert_eq!(run.committed_through, TARGET);
    assert_eq!(run.applied_updates, TARGET + 1);
    assert_eq!(run.lost_applied_updates, 1);
    assert_eq!(run.computed.microbatches, 2 * (TARGET + 1));
    assert_eq!(run.committed.microbatches, 2 * TARGET);
    assert_eq!(run.committed, whole.committed);
    assert_eq!(run.unique_committed_records, whole.unique_committed_records);
    assert!(run.unique_committed_records < run.committed.presentations);
    // Attempts must chain at the durable commit.
    assert!(consumption::reconcile_run(&[two.clone(), one.clone()]).is_err());
    let mut fork = two.clone();
    fork.header.seed ^= 1;
    assert!(consumption::reconcile_run(&[one, fork]).is_err());

    // Every checkpoint binds a byte-exact journal prefix that still verifies.
    for step in [2u64, 4] {
        let state: Value =
            read_json(&broken.join(format!("checkpoints/step-{step}/state.json"))).unwrap();
        let binding: consumption::Binding =
            serde_json::from_value(state["consumption_journal"].clone()).unwrap();
        consumption::verify_binding(&first_bytes, &binding).unwrap();
        let mut altered = first_bytes.clone();
        altered[40] ^= 1;
        assert!(consumption::verify_binding(&altered, &binding).is_err());
    }
}

#[test]
fn a_rejected_update_commits_nothing_and_ends_the_attempt() {
    let base = tempfile::tempdir().unwrap();
    let run = base.path().join("run");
    let rejected = attempt(&run, 0, Some(Stop::RejectAt(3)));
    assert_eq!(rejected.applied_through, 3);
    let bytes = fs::read(&rejected.journal).unwrap();
    let r = consumption::reconcile(&bytes, durable(&run)).unwrap();
    assert!(r.ended_by_rejection);
    assert_eq!(
        (r.attempted_updates, r.applied_updates, r.rejected_updates),
        (4, 3, 1)
    );
    assert_eq!((r.committed_through, r.committed_updates), (2, 2));
    assert_eq!(r.uncommitted_applied_updates, 1);
    // The rejected step's microbatches were computed but never applied.
    assert_eq!(r.computed.microbatches, 8);
    assert_eq!(r.committed.microbatches, 4);
    assert_eq!(r.next.optimizer_step, 2);
    // The durable checkpoint still holds step 2; nothing claims step 4.
    assert_eq!(durable(&run), Some(2));
    // A reader cannot be persuaded that the rejected step was applied.
    let mut forged = bytes.clone();
    forged.extend_from_slice(
        b"{\"event\":\"update\",\"step\":3,\"outcome\":\"applied\",\"microbatches\":2,\
          \"gradient_norm\":0.5,\"clipped\":false,\"reason\":null}\n",
    );
    assert!(consumption::reconcile(&forged, None).is_err());
}

#[test]
fn the_writer_refuses_out_of_order_or_unsupported_accounting() {
    let base = tempfile::tempdir().unwrap();
    let header = |initial: u64| Header {
        schema: consumption::SCHEMA.into(),
        rank: 0,
        world: 1,
        microbatch: MICRO,
        global_batch: GLOBAL,
        accumulation: 2,
        seed: SEED,
        sampler_id: "fixture".into(),
        rng: consumption::RNG_ID.into(),
        sampling_offset: 0,
        initial_optimizer_step: initial,
        target_optimizer_step: 4,
        corpus_sha256: DIGEST.into(),
        replay_policy_sha256: None,
        resume_state_sha256: None,
    };
    let ids = sampler(0, MICRO);
    let supervision = {
        let (rows, lengths): (Vec<_>, Vec<_>) = ids.iter().map(|&r| labels(r)).unzip();
        objective::supervision(&rows, &lengths).unwrap()
    };
    let path = base.path().join("journal.jsonl");
    let mut journal = Journal::create(&path, header(0)).unwrap();
    // Wrong cursor, wrong RNG identity, wrong microbatch index.
    let seed0 = consumption::dropout_seed(SEED, 0, 0, 0);
    assert!(journal
        .microbatch(0, 0, 8, seed0, &ids, supervision.clone())
        .is_err());
    assert!(journal
        .microbatch(0, 0, 0, seed0 ^ 1, &ids, supervision.clone())
        .is_err());
    assert!(journal
        .microbatch(0, 1, 16, seed0, &ids, supervision.clone())
        .is_err());
    journal
        .microbatch(0, 0, 0, seed0, &ids, supervision.clone())
        .unwrap();
    // An update before every accumulation microbatch, a commit of an
    // unapplied update, and a commit inside an unfinished update.
    assert!(journal.applied(0, 0.5).is_err());
    assert!(journal.commit(1, DIGEST).is_err());
    let ids1 = sampler(16, MICRO);
    journal
        .microbatch(
            0,
            1,
            16,
            consumption::dropout_seed(SEED, 0, 0, 1),
            &ids1,
            supervision.clone(),
        )
        .unwrap();
    assert!(journal.applied(0, f64::NAN).is_err());
    journal.applied(0, 1.5).unwrap();
    assert!(journal.commit(1, "not-a-digest").is_err());
    journal.commit(1, DIGEST).unwrap();
    // Commits must advance.
    assert!(journal.commit(1, DIGEST).is_err());
    // A journal cannot be created twice for one attempt.
    assert!(Journal::create(&path, header(0)).is_err());
    // Header validation: initial beyond target, partition mismatch.
    let mut bad = header(5);
    assert!(Journal::create(&base.path().join("bad-1.jsonl"), bad.clone()).is_err());
    bad = header(0);
    bad.accumulation = 3;
    assert!(Journal::create(&base.path().join("bad-2.jsonl"), bad).is_err());
}

#[test]
fn journal_ingestion_is_idempotent_and_refuses_inconsistent_history() {
    let base = tempfile::tempdir().unwrap();
    let run = base.path().join("run");
    let first = attempt(&run, 1, Some(Stop::AfterApplied(4)));
    let bytes = fs::read(&first.journal).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let lines = text.lines().collect::<Vec<_>>();
    let baseline = consumption::reconcile(&bytes, Some(4)).unwrap();

    // Exact duplicate ingestion leaves every total unchanged.
    let mut duplicated = bytes.clone();
    for line in [lines[1], lines[3], lines[lines.len() - 1]] {
        duplicated.extend_from_slice(line.as_bytes());
        duplicated.push(b'\n');
    }
    let again = consumption::reconcile(&duplicated, Some(4)).unwrap();
    assert_eq!(again.duplicate_events_ignored, 3);
    assert_eq!(
        (
            again.computed,
            again.committed,
            again.applied_updates,
            again.committed_records.clone()
        ),
        (
            baseline.computed,
            baseline.committed,
            baseline.applied_updates,
            baseline.committed_records.clone()
        )
    );
    // A conflicting duplicate is refused.
    let mut conflicting = bytes.clone();
    let mut event: Value = serde_json::from_str(lines[1]).unwrap();
    event["records"][0] = json!(event["records"][1].clone());
    conflicting.extend_from_slice(serde_json::to_string(&event).unwrap().as_bytes());
    conflicting.push(b'\n');
    assert!(consumption::reconcile(&conflicting, Some(4)).is_err());

    // A torn final line (hard kill mid-write) is reported and not counted.
    let mut torn = bytes.clone();
    torn.extend_from_slice(&lines[1].as_bytes()[..30]);
    let t = consumption::reconcile(&torn, Some(4)).unwrap();
    assert!(t.torn_tail);
    assert_eq!(t.computed, baseline.computed);

    // An interrupted step's microbatches are unfinished work, not an update.
    let cut = lines[..lines.len() - 1].join("\n") + "\n";
    let partial = consumption::reconcile(cut.as_bytes(), Some(4)).unwrap();
    assert_eq!(partial.unfinished_step, Some(4));
    assert_eq!(partial.unfinished.microbatches, 2);
    assert_eq!(partial.applied_updates, 4);

    // Tampered cursor, RNG identity or records are refused.
    for (field, value) in [
        ("serial", json!(1)),
        ("dropout_seed", json!(7)),
        ("records", json!(vec![0; MICRO])),
    ] {
        let mut altered = lines
            .clone()
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let mut event: Value = serde_json::from_str(&altered[1]).unwrap();
        event[field] = value;
        altered[1] = serde_json::to_string(&event).unwrap();
        let joined = altered.join("\n") + "\n";
        let refused = consumption::reconcile(joined.as_bytes(), Some(4)).is_err()
            || consumption::verify_records(joined.as_bytes(), sampler).is_err();
        assert!(refused, "{field} tampering was accepted");
    }
    // Unknown fields, a missing header and an unknown schema are refused.
    let mut unknown: Value = serde_json::from_str(lines[1]).unwrap();
    unknown["weight"] = json!(2.0);
    let mut altered = lines.iter().map(|l| l.to_string()).collect::<Vec<_>>();
    altered[1] = unknown.to_string();
    assert!(consumption::reconcile((altered.join("\n") + "\n").as_bytes(), None).is_err());
    let headless = lines[1..].join("\n") + "\n";
    assert!(consumption::reconcile(headless.as_bytes(), None).is_err());

    // Durable state versus commit events.
    assert!(
        consumption::reconcile(&bytes, Some(2)).is_err(),
        "journal commits beyond the durable checkpoint were accepted"
    );
    assert!(
        consumption::reconcile(&bytes, Some(6)).is_err(),
        "a durable step beyond applied updates was accepted"
    );
    // Crash between checkpoint publication and the journal commit event.
    let without_last_commit = lines
        .iter()
        .filter(|line| !line.contains("\"event\":\"commit\",\"step\":4"))
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let missing = consumption::reconcile(without_last_commit.as_bytes(), Some(4)).unwrap();
    assert!(missing.commit_event_missing);
    assert_eq!(missing.committed_through, 4);
    let unknown_durable = consumption::reconcile(without_last_commit.as_bytes(), None).unwrap();
    assert_eq!(unknown_durable.committed_through, 2);
    assert_eq!(unknown_durable.uncommitted_applied_updates, 3);
    let _: BTreeSet<u64> = missing.committed_records;
}
