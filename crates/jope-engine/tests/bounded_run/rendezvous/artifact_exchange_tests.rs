use super::*;

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
