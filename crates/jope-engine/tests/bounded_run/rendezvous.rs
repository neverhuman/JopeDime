// ---------------------------------------------------------------- W7-T04 rendezvous

struct Rv {
    run: Run,
    c: Coordinator<FakeSupervisor>,
    fake: FakeSupervisor,
    rv: Rendezvous,
    closure: String,
    now: u64,
}
/// A planned, preflighted run whose three ranks are launched (fake supervisor) and whose
/// rendezvous is open: ranks exist but none may start work before admission.
fn rendezvous_run() -> Rv {
    let now = now_ms();
    let run = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&run.dir, &run.sha, fake.clone(), now).unwrap();
    c.preflight(
        &serde_json::to_vec(&census(&run.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c.dispatch_all(now).unwrap();
    let rv = c
        .open_rendezvous(
            &run.keys.coordinator,
            "fixture-coordinator-boot",
            &rendezvous::fresh_nonce().unwrap(),
            now,
        )
        .unwrap();
    let closure =
        rendezvous::observed_closure_digest(&run.root, &run.contract.artifact_closure).unwrap();
    Rv {
        run,
        c,
        fake,
        rv,
        closure,
        now,
    }
}
impl Rv {
    fn honest(&self, rank: u32) -> HelloBody {
        hello_body(
            &self.run.contract,
            &self.run.sha,
            self.rv.challenge(),
            rank,
            self.now,
            &self.closure,
        )
    }
    fn frame(&self, body: HelloBody) -> Vec<u8> {
        hello_frame(&self.run.keys, body)
    }
    /// Persist the refusal and prove nothing started and nothing was committed.
    fn refused_with_zero_commits(mut self, refusal: GroupRefusal, cause: &RefusalCause) {
        assert_eq!(&refusal.cause, cause);
        assert_eq!((refusal.started_ranks, refusal.committed_updates), (0, 0));
        self.c.record_refusal(&refusal, self.now).unwrap();
        assert_eq!(
            self.fake.running(),
            0,
            "launched rank work survived refusal"
        );
        let late_admission = dummy_admission(&self);
        assert!(self.c.record_admission(late_admission, self.now).is_err());
        assert!(self.c.record_dispatch_intent(0, self.now).is_err());
        let dir = self.run.dir.clone();
        drop(self.c);
        let s = controller::status(&dir).unwrap();
        assert_eq!(s.outcome, Some(Outcome::Refused));
        assert_eq!(s.committed_updates_total, 60_000);
        assert!(s
            .ranks
            .iter()
            .all(|r| r.disposition == Some(Disposition::Refused)));
        assert!(!dir.join("results").exists());
        controller::verify(&dir, Some(&self.run.sha)).unwrap();
    }
}
fn dummy_admission(r: &Rv) -> rendezvous::GroupAdmission {
    rendezvous::GroupAdmission {
        body: rendezvous::AdmissionBody {
            schema: "jope.native_group_admission.v1".into(),
            run: r.run.contract.run.clone(),
            attempt: 1,
            plan_sha256: r.run.sha.clone(),
            rendezvous_id: r.rv.rendezvous_id().into(),
            admitted_unix_ms: r.now,
            original_deadline_unix_ms: r.run.contract.original_deadline_unix_ms,
            grants_sha256: String::new(),
            members: Vec::new(),
            communicator_bootstrap: String::new(),
        },
        signature: String::new(),
    }
}

#[test]
fn full_group_is_admitted_and_acknowledged_before_any_rank_may_start() {
    let mut r = rendezvous_run();
    for rank in 0..3 {
        assert!(r
            .rv
            .admit(&r.run.keys.coordinator, r.now)
            .unwrap()
            .is_none());
        assert_eq!(r.rv.offer(&r.frame(r.honest(rank)), r.now).unwrap(), rank);
    }
    let admission = r.rv.admit(&r.run.keys.coordinator, r.now).unwrap().unwrap();
    // An admission missing a member, or signed by anyone but the planned coordinator,
    // cannot be persisted.
    let mut partial = admission.clone();
    partial.body.members.pop();
    partial.signature = jope_parallel::native_loop::bounded::sign(
        &r.run.keys.coordinator,
        b"jope.native_group_admission.signature.v1\0",
        &partial.body,
    )
    .unwrap();
    assert!(r.c.record_admission(partial, r.now).is_err());
    let mut forged = admission.clone();
    forged.signature = jope_parallel::native_loop::bounded::sign(
        &r.run.keys.ranks[0],
        b"jope.native_group_admission.signature.v1\0",
        &forged.body,
    )
    .unwrap();
    assert!(r.c.record_admission(forged, r.now).is_err());
    // Persisted but not yet acknowledged: no result can be accepted.
    let persisted = r.c.record_admission(admission, r.now).unwrap();
    let id0 = rendezvous::dispatch_id(&r.run.sha, 1, 0).unwrap();
    let early = result(
        &r.run.contract,
        &r.run.sha,
        persisted.admission_sha256(),
        0,
        &r.run.keys.ranks[0],
    );
    r.fake.set(
        &id0,
        Liveness::Exited {
            code: Some(0),
            signal: None,
        },
        Some(early),
    );
    let s = r.c.poll(r.now).unwrap();
    assert_eq!(
        s.outcome,
        Some(Outcome::Drained),
        "exit before acknowledgement"
    );
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(s.ranks[0].disposition, Some(Disposition::Failed));
    assert_eq!(r.fake.running(), 0);
}

#[test]
fn missing_participant_refuses_the_group_with_zero_commits() {
    let mut r = rendezvous_run();
    r.rv.offer(&r.frame(r.honest(0)), r.now).unwrap();
    r.rv.offer(&r.frame(r.honest(1)), r.now).unwrap();
    assert!(r
        .rv
        .admit(&r.run.keys.coordinator, r.now + 1_000)
        .unwrap()
        .is_none());
    let late = r.rv.challenge().body.join_deadline_unix_ms + 1;
    let refusal = r.rv.admit(&r.run.keys.coordinator, late).unwrap_err();
    // Terminal: the missing rank arriving now is refused too.
    assert!(r.rv.offer(&r.frame(r.honest(2)), late).is_err());
    r.refused_with_zero_commits(refusal, &RefusalCause::JoinTimeout { missing: vec![2] });
}

#[test]
fn substituted_source_or_input_refuses_the_group() {
    // A different executable on an authenticated participant.
    let mut r = rendezvous_run();
    r.rv.offer(&r.frame(r.honest(0)), r.now).unwrap();
    let mut body = r.honest(1);
    body.executable_sha256 = sha(b"substituted executable");
    let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
    assert!(
        r.rv.offer(&r.frame(r.honest(1)), r.now).is_err(),
        "fail-closed"
    );
    r.refused_with_zero_commits(refusal, &RefusalCause::SubstitutedSource { rank: 1 });
    // A different runtime closure.
    let mut r = rendezvous_run();
    let mut body = r.honest(2);
    body.runtime.nccl_sha256 = Some(sha(b"other nccl"));
    let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::SubstitutedSource { rank: 2 });
    // A substituted input byte in the participant's own local copy.
    let mut r = rendezvous_run();
    fs::write(
        r.run.root.join("inputs/corpus.bin"),
        b"fixture corpus bytez",
    )
    .unwrap();
    let observed =
        rendezvous::observed_closure_digest(&r.run.root, &r.run.contract.artifact_closure).unwrap();
    assert_ne!(observed, r.closure);
    let mut body = r.honest(2);
    body.artifact_closure_sha256 = observed;
    let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::SubstitutedInput { rank: 2 });
}

#[test]
fn forged_peer_refuses_the_group() {
    let mut r = rendezvous_run();
    let intruder = ed25519_dalek::SigningKey::from_bytes(&[99; 32]);
    let forged = rendezvous::sign_hello(&intruder, r.honest(1)).unwrap();
    let refusal =
        r.rv.offer(&serde_json::to_vec(&forged).unwrap(), r.now)
            .unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::ForgedPeer { claimed_rank: 1 });
    // Valid signature, then a field altered in transit.
    let mut r = rendezvous_run();
    let mut hello = rendezvous::sign_hello(&r.run.keys.ranks[0], r.honest(0)).unwrap();
    hello.body.boot_id = "fixture-boot-0".into();
    hello.body.sent_unix_ms += 1;
    let refusal =
        r.rv.offer(&serde_json::to_vec(&hello).unwrap(), r.now)
            .unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::ForgedPeer { claimed_rank: 0 });
    // A rank that is not in the plan.
    let mut r = rendezvous_run();
    let mut body = r.honest(0);
    body.global_rank = 7;
    let hello = rendezvous::sign_hello(&r.run.keys.ranks[0], body).unwrap();
    let refusal =
        r.rv.offer(&serde_json::to_vec(&hello).unwrap(), r.now)
            .unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::UnknownRank { rank: 7 });
}

#[test]
fn expired_grant_refuses_the_group() {
    let mut r = rendezvous_run();
    let expiry = r.run.contract.participants[1].grant.body.expires_unix_ms;
    r.rv.offer(&r.frame(r.honest(0)), r.now).unwrap();
    let refusal = r.rv.offer(&r.frame(r.honest(1)), expiry).unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::GrantExpired { rank: 1 });
    // Every hello accepted in time, but a grant expires before admission.
    let mut r = rendezvous_run();
    let expiry = r.run.contract.participants[0].grant.body.expires_unix_ms;
    for rank in 0..3 {
        r.rv.offer(&r.frame(r.honest(rank)), r.now).unwrap();
    }
    let refusal = r.rv.admit(&r.run.keys.coordinator, expiry).unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::GrantExpired { rank: 0 });
}

#[test]
fn stale_or_replayed_rendezvous_is_refused() {
    // A hello captured from an earlier rendezvous (for example before a coordinator
    // restart) cannot join a new one.
    let r0 = rendezvous_run();
    let old = r0.frame(r0.honest(0));
    let mut r = rendezvous_run();
    let refusal = r.rv.offer(&old, r.now).unwrap_err();
    assert!(
        matches!(
            refusal.cause,
            RefusalCause::StaleRendezvous | RefusalCause::WrongPlan
        ),
        "{refusal:?}"
    );
    r.refused_with_zero_commits(refusal.clone(), &refusal.cause.clone());
    for (name, edit, cause) in [
        (
            "other attempt",
            Box::new(|b: &mut HelloBody| b.attempt = 2) as Box<dyn Fn(&mut HelloBody)>,
            RefusalCause::ReplayedAttempt { attempt: 2 },
        ),
        (
            "other boot",
            Box::new(|b: &mut HelloBody| b.boot_id = "fixture-boot-after-reboot".into()),
            RefusalCause::ReplayedBoot { rank: 1 },
        ),
        (
            "other plan",
            Box::new(|b: &mut HelloBody| b.plan_sha256 = "0".repeat(64)),
            RefusalCause::WrongPlan,
        ),
        (
            "old message",
            Box::new(|b: &mut HelloBody| b.sent_unix_ms -= 60_000),
            RefusalCause::StaleMessage { rank: 1 },
        ),
        (
            "other rendezvous",
            Box::new(|b: &mut HelloBody| b.rendezvous_id = "e".repeat(64)),
            RefusalCause::StaleRendezvous,
        ),
    ] {
        let mut r = rendezvous_run();
        let mut body = r.honest(1);
        edit(&mut body);
        let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
        assert_eq!(refusal.cause, cause, "{name}");
        r.refused_with_zero_commits(refusal, &cause);
    }
    // Participant side: a forged or expired challenge, and an admission for another
    // rendezvous, are refused before the rank does anything.
    let r = rendezvous_run();
    let challenge = serde_json::to_vec(r.rv.challenge()).unwrap();
    rendezvous::verify_challenge(&r.run.contract, &r.run.sha, &challenge, r.now).unwrap();
    let late = r.rv.challenge().body.join_deadline_unix_ms + 1;
    assert_eq!(
        rendezvous::verify_challenge(&r.run.contract, &r.run.sha, &challenge, late).unwrap_err(),
        RefusalCause::StaleRendezvous
    );
    let mut forged: rendezvous::Challenge = r.rv.challenge().clone();
    forged.body.join_deadline_unix_ms -= 1;
    assert_eq!(
        rendezvous::verify_challenge(
            &r.run.contract,
            &r.run.sha,
            &serde_json::to_vec(&forged).unwrap(),
            r.now
        )
        .unwrap_err(),
        RefusalCause::ForgedCoordinator
    );
    let mut r1 = rendezvous_run();
    let mut r2 = rendezvous_run();
    let mut hellos = Vec::new();
    for rank in 0..3 {
        let h = rendezvous::sign_hello(&r1.run.keys.ranks[rank as usize], r1.honest(rank)).unwrap();
        r1.rv
            .offer(&serde_json::to_vec(&h).unwrap(), r1.now)
            .unwrap();
        hellos.push(h);
    }
    let admission1 = r1
        .rv
        .admit(&r1.run.keys.coordinator, r1.now)
        .unwrap()
        .unwrap();
    for rank in 0..3 {
        r2.rv.offer(&r2.frame(r2.honest(rank)), r2.now).unwrap();
    }
    let own2 = rendezvous::sign_hello(&r2.run.keys.ranks[0], r2.honest(0)).unwrap();
    let frame1 = serde_json::to_vec(&admission1).unwrap();
    rendezvous::verify_admission(&r1.run.contract, &r1.run.sha, &hellos[0], &frame1, r1.now)
        .unwrap();
    assert!(
        rendezvous::verify_admission(&r2.run.contract, &r2.run.sha, &own2, &frame1, r2.now)
            .is_err(),
        "admission from another rendezvous/plan"
    );
    let expired = r1.run.contract.participants[0].grant.body.expires_unix_ms;
    assert_eq!(
        rendezvous::verify_admission(&r1.run.contract, &r1.run.sha, &hellos[0], &frame1, expired)
            .unwrap_err(),
        RefusalCause::GrantExpired { rank: 0 }
    );
    assert_eq!(
        rendezvous::verify_admission(&r1.run.contract, &r1.run.sha, &hellos[1], &frame1, r1.now)
            .unwrap()
            .global_rank(),
        1
    );
    let mut other_hello = hellos[1].clone();
    other_hello.body.delivery_id = 2;
    assert_eq!(
        rendezvous::verify_admission(&r1.run.contract, &r1.run.sha, &other_hello, &frame1, r1.now)
            .unwrap_err(),
        RefusalCause::Equivocation { rank: 1 }
    );
}

#[test]
fn delivery_ids_message_limits_and_timeouts_are_enforced() {
    let mut r = rendezvous_run();
    let first = r.frame(r.honest(0));
    assert_eq!(r.rv.offer(&first, r.now).unwrap(), 0);
    assert_eq!(
        r.rv.offer(&first, r.now).unwrap(),
        0,
        "identical redelivery"
    );
    let mut second = r.honest(0);
    second.delivery_id = 2;
    let refusal = r.rv.offer(&r.frame(second), r.now).unwrap_err();
    r.refused_with_zero_commits(refusal, &RefusalCause::Equivocation { rank: 0 });

    let mut r = rendezvous_run();
    let limit = r.run.contract.rendezvous.max_message_bytes as usize;
    let refusal = r.rv.offer(&vec![b'{'; limit + 1], r.now).unwrap_err();
    r.refused_with_zero_commits(
        refusal,
        &RefusalCause::Oversized {
            bytes: limit as u64 + 1,
            limit: limit as u64,
        },
    );
    let mut r = rendezvous_run();
    let mut body = r.honest(0);
    body.delivery_id = 0;
    let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
    assert!(matches!(refusal.cause, RefusalCause::Malformed { .. }));
    let mut r = rendezvous_run();
    let mut body = r.honest(1);
    body.communicator_bootstrap = Some(hex::encode([1; 16]));
    let refusal = r.rv.offer(&r.frame(body), r.now).unwrap_err();
    assert!(matches!(refusal.cause, RefusalCause::Malformed { .. }));

    // Framing: a declared length above the limit is refused before the body is read,
    // and a silent peer times out instead of blocking.
    let (mut a, mut b) = UnixStream::pair().unwrap();
    use std::io::Write;
    a.write_all(&(1024u32 * 1024 + 1).to_be_bytes()).unwrap();
    let e = rendezvous::read_frame(&mut b, 1024 * 1024).unwrap_err();
    assert!(e.to_string().contains("exceeds message limit"), "{e}");
    assert!(rendezvous::write_frame(&mut a, &vec![0; 2048], 1024).is_err());
    b.set_read_timeout(Some(std::time::Duration::from_millis(100)))
        .unwrap();
    let started = std::time::Instant::now();
    assert!(rendezvous::read_frame(&mut b, 1024).is_err());
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn results_are_authenticated_before_acknowledgement() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let plan = plan_sha(&c);
    let adm = "a".repeat(64);
    let (env, body) = result(&c, &plan, &adm, 1, &k.ranks[1]);
    let ok = rendezvous::verify_result(&c, &plan, &adm, &env, &body).unwrap();
    assert_eq!(ok.body().global_rank, 1);
    let (wrong_key, b) = result(&c, &plan, &adm, 1, &k.ranks[0]);
    assert!(rendezvous::verify_result(&c, &plan, &adm, &wrong_key, &b).is_err());
    assert!(rendezvous::verify_result(&c, &plan, &"b".repeat(64), &env, &body).is_err());
    assert!(rendezvous::verify_result(&c, &"c".repeat(64), &adm, &env, &body).is_err());
    assert!(rendezvous::verify_result(&c, &plan, &adm, &env, b"{\"fixture_rank\":2}").is_err());
    let mut big = env.clone();
    big.resize(c.rendezvous.max_message_bytes as usize + 1, b' ');
    assert!(rendezvous::verify_result(&c, &plan, &adm, &big, &body).is_err());
    let mut more_updates: rendezvous::SignedResult = serde_json::from_slice(&env).unwrap();
    more_updates.body.updates = 4;
    let resigned = rendezvous::sign_result(&k.ranks[1], more_updates.body).unwrap();
    assert!(rendezvous::verify_result(
        &c,
        &plan,
        &adm,
        &serde_json::to_vec(&resigned).unwrap(),
        &body
    )
    .is_err());
    let mut tight = c.clone();
    tight.participants[1].result_destination.max_bytes = 4;
    assert!(rendezvous::verify_result(&tight, &plan, &adm, &env, &body).is_err());

    // In the controller, a forged retained result fails the rank and commits nothing.
    let run = planned(now);
    let fake = FakeSupervisor::default();
    let mut co = Coordinator::open(&run.dir, &run.sha, fake.clone(), now).unwrap();
    co.preflight(
        &serde_json::to_vec(&census(&run.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    co.dispatch_all(now).unwrap();
    let (persisted, hellos) = admit(&mut co, &run, now);
    deliver(&mut co, &run, &persisted, &hellos, now);
    for rank in 0..3u32 {
        let signer = &run.keys.ranks[if rank == 2 { 0 } else { rank as usize }];
        let retained = result(
            &run.contract,
            &run.sha,
            persisted.admission_sha256(),
            rank,
            signer,
        );
        let id = rendezvous::dispatch_id(&run.sha, 1, rank).unwrap();
        fake.set(
            &id,
            Liveness::Exited {
                code: Some(0),
                signal: None,
            },
            Some(retained),
        );
    }
    let s = co.poll(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(s.ranks[2].disposition, Some(Disposition::Failed));
    assert!(
        s.ranks[2].detail.contains("result refused"),
        "{}",
        s.ranks[2].detail
    );
    assert_eq!(
        s.ranks[..2]
            .iter()
            .map(|r| r.disposition)
            .collect::<Vec<_>>(),
        vec![Some(Disposition::CompletedUncommitted); 2]
    );
    drop(co);
    controller::verify(&run.dir, Some(&run.sha)).unwrap();
}

#[test]
fn immutable_artifact_exchange_verifies_bytes_without_replacement() {
    let (_t, root) = tempdir();
    let k = keys(3);
    let c = contract(now_ms(), 3, &k);
    let entry = &c.artifact_closure[0];
    let good = closure_files()[0].1.clone();
    rendezvous::receive_artifact(&root, entry, &good).unwrap();
    rendezvous::receive_artifact(&root, entry, &good).unwrap();
    assert!(rendezvous::receive_artifact(&root, entry, b"fixture corpus bytez").is_err());
    assert_eq!(fs::read(root.join(&entry.path)).unwrap(), good);
    let other = &c.artifact_closure[1];
    fs::write(root.join(&other.path), b"conflicting existing bytes").unwrap();
    assert!(rendezvous::receive_artifact(&root, other, &closure_files()[1].1).is_err());
    assert_eq!(
        fs::read(root.join(&other.path)).unwrap(),
        b"conflicting existing bytes"
    );
    let mut escape = entry.clone();
    escape.path = "../escape.bin".into();
    assert!(rendezvous::receive_artifact(&root, &escape, &good).is_err());
}
