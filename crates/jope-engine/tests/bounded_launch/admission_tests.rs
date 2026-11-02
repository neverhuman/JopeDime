use super::*;

#[test]
fn the_nccl_bootstrap_comes_only_from_this_plans_verified_admission() {
    let k = keys(3);
    let c = contract(now_ms(), 3, &k);
    let plan = digest(&c);
    let (rv, frame, hellos, members) = admit(&c, &k, None);
    // Positive control: every rank takes rank 0's id from the signed admission.
    for rank in 0..3u32 {
        let record = launch::launch_record(&c, &plan, rank).unwrap();
        let b = launch::nccl_bootstrap(&c, &plan, &record, &members[rank as usize]).unwrap();
        assert_eq!(b.id(), &[0xab; NCCL_UNIQUE_ID_BYTES]);
        assert_eq!((b.global_rank(), b.world_size()), (rank, 3));
        assert_eq!((b.plan_sha256(), b.attempt()), (plan.as_str(), c.attempt));
        assert_eq!(b.rendezvous_id(), rv.rendezvous_id());
        assert_eq!(
            b.admission_sha256(),
            members[rank as usize].admission_sha256()
        );
    }
    let now = now_ms();
    let r1 = launch::launch_record(&c, &plan, 1).unwrap();
    // Forged: a bootstrap altered in transit, or an admission not signed by the coordinator.
    let mut altered: GroupAdmission = serde_json::from_slice(&frame).unwrap();
    altered.body.communicator_bootstrap = hex::encode([0xcd; NCCL_UNIQUE_ID_BYTES]);
    let forged = serde_json::to_vec(&altered).unwrap();
    assert_eq!(
        rendezvous::verify_admission(&c, &plan, &hellos[1], &forged, now).unwrap_err(),
        RefusalCause::ForgedCoordinator
    );
    altered.signature = jope_parallel::native_loop::bounded::sign(
        &k.ranks[0],
        b"jope.native_group_admission.signature.v1\0",
        &altered.body,
    )
    .unwrap();
    let forged = serde_json::to_vec(&altered).unwrap();
    assert_eq!(
        rendezvous::verify_admission(&c, &plan, &hellos[1], &forged, now).unwrap_err(),
        RefusalCause::ForgedCoordinator
    );
    // Replayed: an admission from an earlier rendezvous never becomes a membership of the
    // current one.
    let (_, stale_frame, _, _) = admit(&c, &k, None);
    let (_, _, current_hellos, _) = admit(&c, &k, None);
    assert_eq!(
        rendezvous::verify_admission(&c, &plan, &current_hellos[1], &stale_frame, now).unwrap_err(),
        RefusalCause::StaleRendezvous
    );
    // Wrong attempt: attempt 2 of the same run is refused at verification, and a membership
    // verified under attempt 2 never yields attempt 1's bootstrap.
    let mut next = c.clone();
    next.attempt = 2;
    next.predecessor = Some(Predecessor {
        plan_sha256: plan.clone(),
        attempt: 1,
        journal_tip_sha256: "0".repeat(64),
        committed_updates: c.counters.committed_updates_at_start,
    });
    resign(&mut next, &k);
    next.validate_structure().unwrap();
    let (_, next_frame, _, next_members) = admit(&next, &k, None);
    assert_eq!(
        rendezvous::verify_admission(&c, &plan, &hellos[1], &next_frame, now).unwrap_err(),
        RefusalCause::ReplayedAttempt { attempt: 2 }
    );
    let e = launch::nccl_bootstrap(&c, &plan, &r1, &next_members[1]).unwrap_err();
    assert!(e.to_string().contains("attempt 2"), "{e}");
    // Another plan's membership, another rank's membership, and an edited launch record.
    let mut other = c.clone();
    other.run = "w7-fixture-other".into();
    resign(&mut other, &k);
    let (_, _, _, other_members) = admit(&other, &k, None);
    assert!(launch::nccl_bootstrap(&c, &plan, &r1, &other_members[1]).is_err());
    assert!(launch::nccl_bootstrap(&c, &plan, &r1, &members[2]).is_err());
    let mut edited = r1.clone();
    edited.world_size = 2;
    assert!(launch::nccl_bootstrap(&c, &plan, &edited, &members[1]).is_err());
    // A signed bootstrap that is not exactly an NCCL unique id is refused.
    let (_, _, _, short) = admit(&c, &k, Some(hex::encode([0xab; 64])));
    let e = launch::nccl_bootstrap(&c, &plan, &r1, &short[1]).unwrap_err();
    assert!(e.to_string().contains("an NCCL unique id is 128"), "{e}");
}
