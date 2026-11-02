use super::*;

/// W7-T03 delivery: a launched worker's topology comes only from the frozen plan and its
/// launch record, and every mismatch is refused before any communicator could exist.
#[test]
fn a_worker_binds_only_from_its_frozen_launch_and_visibility() {
    for layout in [vec![0, 1, 2], vec![0, 0, 1]] {
        let (c, _) = plan(&layout);
        let plan_sha = sha(&bytes(&c));
        for rank in 0..3u32 {
            let record = launch::launch_record(&c, &plan_sha, rank).unwrap();
            let own = record.cuda_visible_devices.clone();
            let (a, b) = bound_worker(&c, &plan_sha, &record, Some(OsStr::new(&own)), || {
                Ok(restricted_view(&c, rank))
            })
            .unwrap();
            assert_eq!(a, trainer_assignment(&c).unwrap());
            assert_eq!(b.global_rank(), rank as usize);
            assert_eq!(b.world(), 3);
            assert_eq!(b.device_index(), RESTRICTED_DEVICE_INDEX);
            assert_eq!(b.gpu_uuid(), Some(own.as_str()));
            assert_eq!(b.receipt()["plan_sha256"], plan_sha);
        }
    }
    let (c, _) = plan(&[0, 0, 1]);
    let plan_sha = sha(&bytes(&c));
    let record = launch::launch_record(&c, &plan_sha, 0).unwrap();
    let (own, other, foreign) = (uuid(0x10), uuid(0x11), uuid(0x99));
    let host = &c.participants[0].host;
    let view = |visible: Vec<String>| LocalDevices {
        host_id: host.machine_id.clone(),
        boot_id: host.boot_id.clone(),
        visible,
    };
    // Environment refusals happen before the device view is even read.
    for env in [
        None,
        Some("0".to_string()),
        Some(format!("{own},{own}")),
        Some(other.clone()),
        Some(format!("{own},{other}")),
        Some(String::new()),
    ] {
        let refused = bound_worker(
            &c,
            &plan_sha,
            &record,
            env.as_deref().map(OsStr::new),
            || panic!("the device view is not read before the environment is verified"),
        );
        assert!(refused.is_err(), "{env:?}");
    }
    // Mismatched visibility, a duplicated UUID and a wrong local index in the device view.
    let env = Some(OsStr::new(&own));
    for visible in [
        vec![other.clone()],
        vec![own.clone(), own.clone()],
        vec![foreign.clone(), own.clone()],
        vec![own.clone(), foreign.clone()],
        vec![own.clone(), other.clone()],
        Vec::new(),
    ] {
        let refused = bound_worker(&c, &plan_sha, &record, env, || Ok(view(visible.clone())));
        assert!(refused.is_err(), "{visible:?}");
    }
    let mut rebooted = view(vec![own.clone()]);
    rebooted.boot_id = "rebooted".into();
    assert!(bound_worker(&c, &plan_sha, &record, env, || Ok(rebooted)).is_err());
    assert!(bound_worker(&c, &plan_sha, &record, env, || anyhow::bail!(
        "no device view"
    ))
    .is_err());
    // Launch records the plan does not derive.
    let mut edited = record.clone();
    edited.cuda_visible_devices = other.clone();
    assert!(
        bound_worker(&c, &plan_sha, &edited, Some(OsStr::new(&other)), || Ok(
            view(vec![other.clone()])
        ))
        .is_err()
    );
    let mut other_plan = c.clone();
    other_plan.run = "w7-fixture-other".into();
    resign(&mut other_plan, &keys(3));
    let other_sha = sha(&bytes(&other_plan));
    let foreign_record = launch::launch_record(&other_plan, &other_sha, 0).unwrap();
    assert!(
        bound_worker(&c, &plan_sha, &foreign_record, env, || Ok(view(vec![
            own.clone()
        ])))
        .is_err()
    );
    assert!(bound_worker(&c, &other_sha, &record, env, || Ok(view(vec![own.clone()]))).is_err());
    // Rank 2's genuine launch on rank 0's host view: another host.
    let (c3, _) = plan(&[0, 1, 2]);
    let sha3 = sha(&bytes(&c3));
    let r2 = launch::launch_record(&c3, &sha3, 2).unwrap();
    let wrong_host = LocalDevices {
        visible: vec![r2.cuda_visible_devices.clone()],
        ..restricted_view(&c3, 0)
    };
    assert!(bound_worker(
        &c3,
        &sha3,
        &r2,
        Some(OsStr::new(&r2.cuda_visible_devices)),
        || Ok(wrong_host)
    )
    .is_err());
}

/// W7-T04: a bounded rank takes its NCCL unique id only from its plan's verified admission;
/// the shared `nccl.id` rendezvous keeps its meaning only for ranks without a frozen plan.
/// Every refusal here happens before any native call (no CUDA or NCCL is touched).
#[test]
fn bounded_plans_take_the_communicator_only_from_the_verified_admission() {
    let (c, k) = plan(&[0, 1, 2]);
    let plan_sha = sha(&bytes(&c));
    let assignment = trainer_assignment(&c).unwrap();
    let (_, _, _, members) = admit_group(&c, &k, None);
    let bootstrap = |rank: u32| {
        let record = launch::launch_record(&c, &plan_sha, rank).unwrap();
        launch::nccl_bootstrap(&c, &plan_sha, &record, &members[rank as usize]).unwrap()
    };
    let (b0, b1) = (bootstrap(0), bootstrap(1));
    let (_t, root) = tempdir();
    let launch_dir = root.join("launch");
    std::fs::create_dir(&launch_dir).unwrap();
    // No frozen plan: the historical per-launch file, unchanged.
    assert_eq!(
        communicator_source(None, None, 0, &launch_dir).unwrap(),
        CommunicatorSource::LaunchFile(launch_dir.join("nccl.id"))
    );
    // A frozen plan uses only its own admitted bootstrap for its own rank.
    assert_eq!(
        communicator_source(Some(&assignment), Some(&b0), 0, &launch_dir).unwrap(),
        CommunicatorSource::Admitted
    );
    let refused = |a: Option<&Assignment>, b: Option<&launch::AdmittedBootstrap>, rank| {
        communicator_source(a, b, rank, &launch_dir)
            .unwrap_err()
            .to_string()
    };
    assert!(refused(Some(&assignment), None, 0).contains("nccl.id rendezvous is refused"));
    assert!(refused(None, Some(&b0), 0).contains("requires its plan's frozen assignment"));
    assert!(refused(Some(&assignment), Some(&b1), 0).contains("another plan, rank or world"));
    let mut other = c.clone();
    other.run = "w7-fixture-other".into();
    resign(&mut other, &k);
    let other_assignment = trainer_assignment(&other).unwrap();
    assert!(refused(Some(&other_assignment), Some(&b0), 0).contains("another plan"));
    // A shared nccl.id in a bounded launch directory is refused, not read.
    std::fs::write(launch_dir.join("nccl.id"), [0u8; 128]).unwrap();
    assert!(refused(Some(&assignment), Some(&b0), 0).contains("holds a shared nccl.id"));
    // The communicator constructors refuse mismatches before any native call.
    let bound = |rank: u32| {
        topology::bind(&assignment, rank as usize, 3, &restricted_view(&c, rank)).unwrap()
    };
    verify_admitted_bootstrap(&bound(0), &b0).unwrap();
    assert!(verify_admitted_bootstrap(&bound(0), &b1).is_err());
    let legacy = topology::Binding::legacy_single_host(0, 3).unwrap();
    assert!(verify_admitted_bootstrap(&legacy, &b0).is_err());
    let e = jope_parallel::neural::Nccl::init(&bound(0), &launch_dir.join("nccl.id"))
        .err()
        .expect("a plan-bound binding never uses the shared file");
    assert!(
        e.to_string()
            .contains("shared nccl.id rendezvous is refused"),
        "{e}"
    );
    assert!(jope_parallel::neural::Nccl::init_admitted(&bound(0), &b1).is_err());
}
