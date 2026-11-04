#[test]
fn same_host_socket_rendezvous_admits_only_the_authenticated_full_group() {
    for forged_rank in [None, Some(2u32)] {
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
        let mut rv = c
            .open_rendezvous(
                &run.keys.coordinator,
                "fixture-coordinator-boot",
                &rendezvous::fresh_nonce().unwrap(),
                now,
            )
            .unwrap();
        let socket = run.root.join("control.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let closure =
            rendezvous::observed_closure_digest(&run.root, &run.contract.artifact_closure).unwrap();
        let threads: Vec<_> = (0..3u32)
            .map(|rank| {
                let (contract, sha, socket, closure) = (
                    run.contract.clone(),
                    run.sha.clone(),
                    socket.clone(),
                    closure.clone(),
                );
                let key = if forged_rank == Some(rank) {
                    ed25519_dalek::SigningKey::from_bytes(&[98; 32])
                } else {
                    ed25519_dalek::SigningKey::from_bytes(&rank_seed(rank as usize))
                };
                std::thread::spawn(move || -> anyhow::Result<u32> {
                    // After a refusal the listener is gone; a late connect is a refusal too.
                    let mut stream = UnixStream::connect(&socket)?;
                    rendezvous::join_unix(
                        &mut stream,
                        &contract,
                        &sha,
                        &key,
                        |ch| hello_body(&contract, &sha, ch, rank, now_ms(), &closure),
                        &now_ms,
                    )
                    .map(|m| m.global_rank())
                })
            })
            .collect();
        let gathered = rendezvous::gather_unix(&listener, &mut rv, &run.keys.coordinator, &now_ms);
        match forged_rank {
            None => {
                let (admission, mut connected) = gathered.unwrap();
                let persisted = c.record_admission(admission, now_ms()).unwrap();
                c.deliver_admission(&persisted, &mut connected, now_ms())
                    .unwrap();
                let admitted: Vec<u32> = threads
                    .into_iter()
                    .map(|t| t.join().unwrap().unwrap())
                    .collect();
                assert_eq!(admitted, vec![0, 1, 2]);
                assert_eq!(c.status().unwrap().ranks[0].state, RankState::Running);
            }
            Some(rank) => {
                let refusal = gathered.err().expect("forged participant admitted");
                assert_eq!(
                    refusal.cause,
                    RefusalCause::ForgedPeer { claimed_rank: rank }
                );
                c.record_refusal(&refusal, now_ms()).unwrap();
                drop(listener);
                drop(c);
                for t in threads {
                    assert!(
                        t.join().unwrap().is_err(),
                        "a rank was admitted after refusal"
                    );
                }
                let s = controller::status(&run.dir).unwrap();
                assert_eq!(s.outcome, Some(Outcome::Refused));
                assert_eq!(s.committed_updates_total, 60_000);
            }
        }
    }
}
