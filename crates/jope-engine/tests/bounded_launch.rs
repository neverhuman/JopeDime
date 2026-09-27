//! W7-T03 delivery half, default features (CPU; no GPU, NCCL or device enumeration):
//! the frozen per-rank launch record, the launcher's exact worker environment and the
//! worker's own visibility check. Synthetic identities; not the three-host proof.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    contract::{BoundedRunContract, Predecessor},
    controller,
    launch::{self, LaunchRecord, LAUNCH_SCHEMA, NCCL_UNIQUE_ID_BYTES, VISIBILITY_VARIABLE},
    rendezvous::{self, AdmittedMember, GroupAdmission, RefusalCause, Rendezvous},
};
use std::{
    ffi::{OsStr, OsString},
    fs,
    os::unix::ffi::OsStrExt,
    path::Path,
};

/// The fixture plan with rank 2 turned into a CPU participant.
fn with_cpu_rank(keys: &Keys) -> BoundedRunContract {
    let mut c = contract(now_ms(), 3, keys);
    c.participants[2].device = None;
    c.participants[2].ceilings.gpu_vram_bytes = None;
    for w in c.work_census.iter_mut().filter(|w| w.global_rank == 2) {
        w.gpu_ms = 0;
        w.peak_vram_bytes = 0;
    }
    let p = c.participants[2].clone();
    c.participants[2].grant = grant(&c, &p, keys);
    c.validate_structure().unwrap();
    c
}
fn digest(c: &BoundedRunContract) -> String {
    sha(&bytes(c))
}

#[test]
fn launch_records_are_exactly_what_the_plan_derives() {
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let plan = digest(&c);
    for rank in 0..3u32 {
        let r = launch::launch_record(&c, &plan, rank).unwrap();
        let p = &c.participants[rank as usize];
        assert_eq!(r.schema, LAUNCH_SCHEMA);
        assert_eq!((r.run.as_str(), r.attempt), (c.run.as_str(), c.attempt));
        assert_eq!((r.global_rank, r.world_size), (rank, 3));
        assert_eq!(
            r.dispatch_id,
            rendezvous::dispatch_id(&plan, c.attempt, rank).unwrap()
        );
        assert_eq!(
            (&r.host, &r.machine_id, &r.boot_id),
            (&p.host.host, &p.host.machine_id, &p.host.boot_id)
        );
        let expected = p
            .device
            .as_ref()
            .map(|d| d.uuid.clone())
            .unwrap_or_default();
        assert_eq!(r.cuda_visible_devices, expected);
        launch::verify_launch_record(&c, &plan, &r).unwrap();
    }
    assert_eq!(
        launch::launch_record(&c, &plan, 2)
            .unwrap()
            .cuda_visible_devices,
        "",
        "a CPU rank hides every GPU"
    );
    // Every field is bound: any edit is refused.
    let good = launch::launch_record(&c, &plan, 1).unwrap();
    type Edit = Box<dyn Fn(&mut LaunchRecord)>;
    let edits: Vec<(&str, Edit)> = vec![
        (
            "schema",
            Box::new(|r| r.schema = "jope.native_bounded_launch.v0".into()),
        ),
        ("run", Box::new(|r| r.run = "other-run".into())),
        ("attempt", Box::new(|r| r.attempt = 2)),
        ("plan", Box::new(|r| r.plan_sha256 = "0".repeat(64))),
        ("world", Box::new(|r| r.world_size = 4)),
        ("dispatch id", Box::new(|r| r.dispatch_id = "0".repeat(64))),
        ("host", Box::new(|r| r.host = "fixture-host-0".into())),
        ("machine", Box::new(|r| r.machine_id = "f".repeat(32))),
        ("boot", Box::new(|r| r.boot_id = "rebooted".into())),
        (
            "visibility",
            Box::new(|r| r.cuda_visible_devices = "GPU-fixture-0".into()),
        ),
        ("rank", Box::new(|r| r.global_rank = 0)),
    ];
    for (name, edit) in edits {
        let mut r = good.clone();
        edit(&mut r);
        assert!(
            launch::verify_launch_record(&c, &plan, &r).is_err(),
            "{name}"
        );
    }
    // A record from another plan, and a digest that does not name the plan, are refused.
    let mut other = c.clone();
    other.run = "w7-fixture-other".into();
    resign(&mut other, &k);
    let foreign = launch::launch_record(&other, &digest(&other), 1).unwrap();
    assert!(launch::verify_launch_record(&c, &plan, &foreign).is_err());
    assert!(launch::launch_record(&c, &"0".repeat(64), 0).is_err());
    assert!(launch::launch_record(&c, &plan, 3).is_err());
}

#[test]
fn load_launch_reads_only_the_frozen_plan_and_record() {
    let (_t, root) = tempdir();
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let planned = controller::plan(&root.join("run"), &bytes(&c), now_ms(), None).unwrap();
    let plan_path = root.join("run/plan.json");
    let record_path = root.join("launch-1.json");
    let record = launch::launch_record(&c, &planned.plan_sha256, 1).unwrap();
    fs::write(&record_path, serde_json::to_vec(&record).unwrap()).unwrap();
    let (loaded, r) = launch::load_launch(&plan_path, &planned.plan_sha256, &record_path).unwrap();
    assert_eq!((loaded, r), (c.clone(), record.clone()));
    // Plan bytes that differ from the admitted digest, or that are not canonical.
    let pretty = root.join("pretty.json");
    fs::write(&pretty, serde_json::to_vec_pretty(&c).unwrap()).unwrap();
    let e = launch::load_launch(&pretty, &planned.plan_sha256, &record_path).unwrap_err();
    assert!(
        e.to_string()
            .contains("differ from the admitted plan digest"),
        "{e}"
    );
    let pretty_sha = sha(&fs::read(&pretty).unwrap());
    let e = launch::load_launch(&pretty, &pretty_sha, &record_path).unwrap_err();
    assert!(e.to_string().contains("not the canonical contract"), "{e}");
    // A record that is not exactly the derived one, has an unknown field, is a symlink or
    // is oversized.
    let mut forged = serde_json::to_value(&record).unwrap();
    forged["global_rank"] = 0.into();
    fs::write(&record_path, serde_json::to_vec(&forged).unwrap()).unwrap();
    assert!(launch::load_launch(&plan_path, &planned.plan_sha256, &record_path).is_err());
    let mut extra = serde_json::to_value(&record).unwrap();
    extra["world_rank_flag"] = 1.into();
    fs::write(&record_path, serde_json::to_vec(&extra).unwrap()).unwrap();
    assert!(launch::load_launch(&plan_path, &planned.plan_sha256, &record_path).is_err());
    let real = root.join("real.json");
    fs::write(&real, serde_json::to_vec(&record).unwrap()).unwrap();
    let link = root.join("link.json");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert!(launch::load_launch(&plan_path, &planned.plan_sha256, &link).is_err());
    fs::write(&record_path, vec![b' '; 64 * 1024 + 1]).unwrap();
    assert!(launch::load_launch(&plan_path, &planned.plan_sha256, &record_path).is_err());
}

#[test]
fn the_launcher_environment_is_exactly_the_frozen_visibility() {
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let plan = digest(&c);
    // `env` prints the whole environment it was exec'd with.
    let env = Path::new("/usr/bin/env");
    for rank in 0..3u32 {
        let out = launch::worker_command(env, &c, &plan, rank)
            .unwrap()
            .output()
            .unwrap();
        assert!(out.status.success());
        let expected = c.participants[rank as usize]
            .device
            .as_ref()
            .map(|d| d.uuid.clone())
            .unwrap_or_default();
        assert_eq!(
            String::from_utf8(out.stdout).unwrap(),
            format!("{VISIBILITY_VARIABLE}={expected}\n"),
            "rank {rank}: nothing but the frozen visibility reaches the worker"
        );
        assert_eq!(
            launch::worker_environment(&c, rank).unwrap(),
            [(VISIBILITY_VARIABLE.to_string(), expected)].into()
        );
    }
    // Nothing is launched for a plan digest that is not this plan, or an unplanned rank.
    assert!(launch::worker_command(env, &c, &"0".repeat(64), 0).is_err());
    assert!(launch::worker_command(env, &c, &plan, 3).is_err());
}

#[test]
fn a_worker_refuses_any_other_visibility_before_starting() {
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let plan = digest(&c);
    let gpu = launch::launch_record(&c, &plan, 0).unwrap();
    let cpu = launch::launch_record(&c, &plan, 2).unwrap();
    let own = gpu.cuda_visible_devices.clone();
    let other = c.participants[1].device.as_ref().unwrap().uuid.clone();
    let check =
        |r: &LaunchRecord, v: Option<&str>| launch::verify_worker_visibility(r, v.map(OsStr::new));
    check(&gpu, Some(&own)).unwrap();
    check(&cpu, Some("")).unwrap();
    // SmartCluster's executor hides every device from a job without a GPU with `-1`.
    check(&cpu, Some(launch::HIDE_ALL_DEVICES)).unwrap();
    for (record, value, needle) in [
        (&gpu, None, "unset"),
        (&gpu, Some("0".to_string()), "local index"),
        (&gpu, Some(format!("{own},{own}")), "names a device twice"),
        (
            &gpu,
            Some(format!("{own},{other}")),
            "differs from the planned",
        ),
        (&gpu, Some(other.clone()), "differs from the planned"),
        (&gpu, Some(String::new()), "differs from the planned"),
        (
            &gpu,
            Some(launch::HIDE_ALL_DEVICES.to_string()),
            "local index",
        ),
        (&cpu, Some(own.clone()), "exposes a device"),
        (&cpu, Some("0".to_string()), "exposes a device"),
        (
            &cpu,
            Some(format!("{},{own}", launch::HIDE_ALL_DEVICES)),
            "exposes a device",
        ),
        (&cpu, None, "unset"),
    ] {
        let e = check(record, value.as_deref()).unwrap_err().to_string();
        assert!(e.contains(needle), "{value:?}: {e}");
    }
    let non_utf8 = OsString::from(OsStr::from_bytes(b"GPU-\xff"));
    assert!(launch::verify_worker_visibility(&gpu, Some(&non_utf8)).is_err());
}

// ------------------------------------------------ W7-T04: the admitted communicator bootstrap

fn admit(
    c: &BoundedRunContract,
    k: &Keys,
    rank0_bootstrap: Option<String>,
) -> (
    Rendezvous,
    Vec<u8>,
    Vec<rendezvous::Hello>,
    Vec<AdmittedMember>,
) {
    admit_group(c, k, rank0_bootstrap)
}

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
