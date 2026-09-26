#![cfg(feature = "neural")]
//! W7-T02 × W7-T03 join (CPU only; no device is enumerated or selected): a frozen
//! bounded-run plan maps, by a pure function, to exactly the trainer Assignment that
//! `training::topology::validate` accepts, and a plan/assignment mismatch is refused.
//! Synthetic identities; not the three-host proof (W7-T07..T09).
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::{
    native_loop::bounded::{
        assignment::{
            bound_worker, trainer_assignment, verify_trainer_assignment,
            worker_cuda_visible_devices, RESTRICTED_DEVICE_INDEX,
        },
        contract::{BoundedRunContract, DeviceAssignment},
        controller, launch,
    },
    training::topology::{self, Assignment, LocalDevices, Participant, SCHEMA},
};
use std::ffi::OsStr;

fn uuid(n: usize) -> String {
    format!("GPU-{n:08x}-0000-4000-8000-{n:012x}")
}
/// The fixture plan with nvidia-smi-form GPU UUIDs. `layout[i]` is the host of rank `i`;
/// ranks sharing a host get consecutive local ranks and distinct host GPU indices.
fn plan(layout: &[usize]) -> (BoundedRunContract, Keys) {
    let k = keys(layout.len());
    let mut c = contract(now_ms(), layout.len(), &k);
    let mut next_local = std::collections::BTreeMap::new();
    for (rank, host) in layout.iter().enumerate() {
        let local = next_local.entry(*host).or_insert(0u32);
        let p = &mut c.participants[rank];
        p.host = identity(*host);
        p.local_rank = *local;
        p.device = Some(DeviceAssignment {
            uuid: uuid(0x10 + rank),
            local_index: *local,
        });
        *local += 1;
    }
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    (c, k)
}
fn expected(c: &BoundedRunContract) -> Assignment {
    Assignment {
        schema: SCHEMA.into(),
        plan_sha256: sha(&bytes(c)),
        world: c.participants.len(),
        participants: c
            .participants
            .iter()
            .map(|p| Participant {
                global_rank: p.global_rank as usize,
                host_id: p.host.machine_id.clone(),
                boot_id: p.host.boot_id.clone(),
                local_rank: p.local_rank as usize,
                device_index: 0,
                gpu_uuid: p.device.as_ref().unwrap().uuid.clone(),
            })
            .collect(),
    }
}
/// What a worker launched under the visibility contract sees.
fn restricted_view(c: &BoundedRunContract, rank: u32) -> LocalDevices {
    let p = &c.participants[rank as usize];
    LocalDevices {
        host_id: p.host.machine_id.clone(),
        boot_id: p.host.boot_id.clone(),
        visible: vec![worker_cuda_visible_devices(c, rank).unwrap()],
    }
}

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
