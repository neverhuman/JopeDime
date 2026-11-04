use super::*;

#[test]
fn plan_maps_to_exactly_the_assignment_topology_accepts() {
    // One GPU on each of three hosts, and two workers sharing one host.
    for layout in [vec![0, 1, 2], vec![0, 0, 1]] {
        let (c, _) = plan(&layout);
        let derived = trainer_assignment(&c).unwrap();
        assert_eq!(derived, expected(&c));
        topology::validate(&derived).unwrap();
        // Pure: the same plan always yields the same assignment.
        assert_eq!(trainer_assignment(&c).unwrap(), derived);
        verify_trainer_assignment(&c, &derived).unwrap();
        // The digest is the controller's frozen plan digest.
        let (_t, root) = tempdir();
        let planned = controller::plan(&root.join("run"), &bytes(&c), now_ms(), None).unwrap();
        assert_eq!(derived.plan_sha256, planned.plan_sha256);
        for rank in 0..c.participants.len() as u32 {
            let b = topology::bind(
                &derived,
                rank as usize,
                derived.world,
                &restricted_view(&c, rank),
            )
            .unwrap();
            assert_eq!(b.global_rank(), rank as usize);
            assert_eq!(b.device_index(), RESTRICTED_DEVICE_INDEX);
            assert_eq!(
                b.local_rank(),
                c.participants[rank as usize].local_rank as usize
            );
            assert_eq!(
                b.gpu_uuid(),
                c.participants[rank as usize]
                    .device
                    .as_ref()
                    .map(|d| d.uuid.as_str())
            );
            assert_eq!(b.receipt()["plan_sha256"], planned.plan_sha256);
        }
    }
    // The shared host's second worker is rank 1 at CUDA ordinal 0, not its host index 1.
    let (c, _) = plan(&[0, 0, 1]);
    assert_eq!(c.participants[1].device.as_ref().unwrap().local_index, 1);
    assert_eq!(
        trainer_assignment(&c).unwrap().participants[1].device_index,
        0
    );
}

#[test]
fn plan_assignment_mismatch_is_refused() {
    let (c, _) = plan(&[0, 1, 2]);
    let good = trainer_assignment(&c).unwrap();
    type Edit = Box<dyn Fn(&mut Assignment)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "other plan digest",
            Box::new(|a| a.plan_sha256 = "0".repeat(64)),
        ),
        (
            "swapped GPUs",
            Box::new(|a| {
                let u = a.participants[0].gpu_uuid.clone();
                a.participants[0].gpu_uuid = a.participants[1].gpu_uuid.clone();
                a.participants[1].gpu_uuid = u;
            }),
        ),
        (
            "host index as CUDA ordinal",
            Box::new(|a| a.participants[2].device_index = 1),
        ),
        (
            "other boot",
            Box::new(|a| a.participants[1].boot_id = "rebooted".into()),
        ),
        (
            "other machine",
            Box::new(|a| a.participants[1].host_id = "f".repeat(32)),
        ),
        (
            "dropped rank",
            Box::new(|a| {
                a.participants.pop();
                a.world = 2;
            }),
        ),
        (
            "renumbered local rank",
            Box::new(|a| a.participants[0].local_rank = 1),
        ),
        (
            "schema",
            Box::new(|a| a.schema = "jope.trainer_assignment.v0".into()),
        ),
    ];
    for (name, edit) in cases {
        let mut a = good.clone();
        edit(&mut a);
        assert!(verify_trainer_assignment(&c, &a).is_err(), "{name}");
    }
    // An assignment derived from another plan (another attempt) is refused for this one,
    // even though topology alone accepts it.
    let mut other = c.clone();
    other.run = "w7-fixture-other".into();
    let k = keys(3);
    resign(&mut other, &k);
    let foreign = trainer_assignment(&other).unwrap();
    topology::validate(&foreign).unwrap();
    assert!(verify_trainer_assignment(&c, &foreign).is_err());
}

#[test]
fn plans_the_trainer_cannot_bind_are_refused() {
    // A CPU participant has no device for the trainer.
    let (mut c, k) = plan(&[0, 1, 2]);
    c.participants[2].device = None;
    c.participants[2].ceilings.gpu_vram_bytes = None;
    for w in c.work_census.iter_mut().filter(|w| w.global_rank == 2) {
        w.gpu_ms = 0;
        w.peak_vram_bytes = 0;
    }
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    let e = trainer_assignment(&c).unwrap_err().to_string();
    assert!(e.contains("has no GPU"), "{e}");
    assert!(worker_cuda_visible_devices(&c, 2).is_err());
    // A device name the contract accepts but that is not an nvidia-smi UUID.
    let (mut c, k) = plan(&[0, 1, 2]);
    c.participants[0].device.as_mut().unwrap().uuid = "GPU-fixture-0".into();
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    assert!(trainer_assignment(&c).is_err());
    // Without the visibility contract, a worker that sees every GPU is refused at bind.
    let (c, _) = plan(&[0, 0, 1]);
    let a = trainer_assignment(&c).unwrap();
    let p = &c.participants[0];
    let unrestricted = LocalDevices {
        host_id: p.host.machine_id.clone(),
        boot_id: p.host.boot_id.clone(),
        visible: vec![uuid(0x10), uuid(0x11)],
    };
    assert!(topology::bind(&a, 0, 3, &unrestricted).is_err());
    // A worker on another boot of the planned host is refused at bind.
    let mut rebooted = restricted_view(&c, 0);
    rebooted.boot_id = "rebooted".into();
    assert!(topology::bind(&a, 0, 3, &rebooted).is_err());
}
