//! W7-T03 delivery half, default features (CPU; no GPU, NCCL or device enumeration):
//! the frozen per-rank launch record, the launcher's exact worker environment and the
//! worker's own visibility check, including SmartCluster's GPU index translated to the planned
//! UUID through an injected `nvidia-smi` (W7-D3). Synthetic identities; not the three-host
//! proof.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    collect::{CommandOutput, CommandRunner},
    contract::{BoundedRunContract, Predecessor},
    controller,
    launch::{
        self, LaunchRecord, ResolvedVisibility, LAUNCH_SCHEMA, NCCL_UNIQUE_ID_BYTES,
        VISIBILITY_VARIABLE,
    },
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

// ------------------------------------ W7-D3 SmartCluster's GPU allocation stays authoritative

/// A SmartCluster supervisor for the fixture host of `rank`, bound to a planned run. It never
/// contacts its daemon: `launch_spec` only freezes the launch and builds the job.
fn fixture_supervisor(
    root: &Path,
    run_dir: &Path,
    host: &str,
    rank: u32,
) -> jope_parallel::native_loop::bounded::supervisor::SmartClusterSupervisor {
    use jope_parallel::native_loop::bounded::{
        participant::{Installation, SmartClusterInstallation, INSTALLATION_SCHEMA},
        supervisor::{DaemonPin, HostSupervision, SmartClusterSupervisor, SERVICE},
    };
    let daemon = DaemonPin {
        socket: root.join("scqd.sock"),
        uid: 1000,
        executable_sha256: sha(b"fixture scqd"),
    };
    let installation = Installation {
        schema: INSTALLATION_SCHEMA.into(),
        address: "127.0.0.1".into(),
        certificate: root.join("host.der"),
        certificate_key: root.join("host.key"),
        source_commit: sha(b"fixture-source")[..40].to_string(),
        executable: root.join("jope-native-loop"),
        runtime_libraries: vec![root.join("libjope-runtime.so")],
        compiler: root.join("compiler"),
        cuda_libraries: Vec::new(),
        nccl_libraries: Vec::new(),
        installed_policies: vec![root.join("scqd.toml")],
        gpu_inventory: true,
        input_root: root.to_path_buf(),
        smartcluster: SmartClusterInstallation {
            daemon: daemon.clone(),
            service: SERVICE.into(),
        },
    };
    let path = root.join(format!("installation-{rank}.json"));
    fs::write(&path, serde_json::to_vec(&installation).unwrap()).unwrap();
    SmartClusterSupervisor::new(HostSupervision {
        host: host.into(),
        daemon,
        tenant: "unix:1000".into(),
        project: "w7-cpu-proof".into(),
        service: SERVICE.into(),
        plan: run_dir.join("plan.json"),
        state: root.join(format!("state-{rank}")),
        coordinator: "127.0.0.1:7460".into(),
        installation: path,
        participant_keys: std::collections::BTreeMap::from([(rank, root.join("rank.key"))]),
    })
    .unwrap()
}

#[test]
fn smartcluster_allocates_a_gpu_rank_exactly_one_exclusive_gpu() {
    use jain_smartcluster::{GpuMode, GpuRequest};
    use jope_parallel::native_loop::bounded::controller::RankLaunch;
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let (_t, root) = tempdir();
    let run_dir = root.join("run");
    let plan = controller::plan(&run_dir, &bytes(&c), now_ms(), None)
        .unwrap()
        .plan_sha256;
    let intent = now_ms() - 1000;
    let spec = |rank: u32| {
        fixture_supervisor(&root, &run_dir, &format!("fixture-host-{rank}"), rank).launch_spec(
            &RankLaunch {
                dispatch_intent_unix_ms: intent,
                contract: &c,
                plan_sha256: &plan,
                global_rank: rank,
                dispatch_id: &rendezvous::dispatch_id(&plan, c.attempt, rank).unwrap(),
            },
        )
    };
    // A GPU rank asks SmartCluster for one exclusive GPU with its planned VRAM ceiling;
    // SmartCluster chooses the device and the rank checks it is the planned one.
    let (_, gpu) = spec(1).unwrap();
    assert_eq!(
        gpu.deadline,
        Some(jain_smartcluster::Timestamp::from(
            intent + c.participants[1].ceilings.wall_ms
        ))
    );
    assert_eq!(
        gpu.resources.gpu,
        Some(GpuRequest {
            count: 1,
            vram_bytes: 20 * GIB,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        })
    );
    // A GPU rank with no VRAM ceiling in its plan cannot be allocated a GPU: refused.
    let mut unbounded = c.clone();
    unbounded.participants[1].ceilings.gpu_vram_bytes = None;
    let e = fixture_supervisor(&root, &run_dir, "fixture-host-1", 11)
        .launch_spec(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &unbounded,
            plan_sha256: &plan,
            global_rank: 1,
            dispatch_id: &rendezvous::dispatch_id(&plan, c.attempt, 1).unwrap(),
        })
        .unwrap_err();
    assert!(e.to_string().contains("has no VRAM ceiling"), "{e}");
    // A CPU rank asks for none, and keeps its CPU and RAM ceilings.
    let (_, cpu) = spec(2).unwrap();
    assert_eq!(cpu.resources.gpu, None);
    assert_eq!(
        (cpu.resources.cpu, cpu.resources.ram),
        (
            jain_smartcluster::ResourceRequest::cpu_and_ram(4_000, 8 * GIB).cpu,
            jain_smartcluster::ResourceRequest::cpu_and_ram(4_000, 8 * GIB).ram
        )
    );
}

/// An injected `nvidia-smi --query-gpu=index,uuid`; `output: None` when it is not installed.
struct Smi {
    output: Option<(bool, String)>,
    calls: std::cell::Cell<u32>,
}
impl Smi {
    fn listing(text: &str) -> Self {
        Self::with(Some((true, text.into())))
    }
    fn with(output: Option<(bool, String)>) -> Self {
        Self {
            output,
            calls: 0.into(),
        }
    }
}
impl CommandRunner for Smi {
    fn run(&self, program: &str, args: &[&str]) -> anyhow::Result<Option<CommandOutput>> {
        assert_eq!(program, "nvidia-smi");
        assert_eq!(
            args,
            ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
        );
        self.calls.set(self.calls.get() + 1);
        Ok(self.output.clone().map(|(success, stdout)| CommandOutput {
            success,
            stdout: stdout.into_bytes(),
            stderr: b"NVML failure".to_vec(),
        }))
    }
}

#[test]
fn a_smartcluster_gpu_index_must_translate_to_the_planned_uuid() {
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let plan = digest(&c);
    // Rank 1 is planned on GPU-fixture-1; rank 2 is a CPU rank.
    let gpu = launch::launch_record(&c, &plan, 1).unwrap();
    let cpu = launch::launch_record(&c, &plan, 2).unwrap();
    let host = "0, GPU-fixture-0\n1, GPU-another-device\n2, GPU-fixture-1\n";
    // Positive control: SmartCluster allocated index 2, which is the planned GPU; the worker
    // pins that UUID for itself.
    let smi = Smi::listing(host);
    let resolved = launch::resolve_visibility(&gpu, Some(OsStr::new("2")), &smi).unwrap();
    assert_eq!(
        resolved,
        ResolvedVisibility::SchedulerIndex {
            index: 2,
            uuid: "GPU-fixture-1".into()
        }
    );
    assert_eq!(resolved.pinned(), Some("GPU-fixture-1"));
    assert_eq!(smi.calls.get(), 1);
    // The launcher's exact UUID needs no translation, and a CPU rank never runs nvidia-smi.
    let never = Smi::with(None);
    assert_eq!(
        launch::resolve_visibility(&gpu, Some(OsStr::new("GPU-fixture-1")), &never).unwrap(),
        ResolvedVisibility::PlannedUuid("GPU-fixture-1".into())
    );
    for hidden in ["-1", ""] {
        let resolved = launch::resolve_visibility(&cpu, Some(OsStr::new(hidden)), &never).unwrap();
        assert_eq!(
            (resolved.clone(), resolved.pinned()),
            (ResolvedVisibility::Cpu, None)
        );
    }
    assert!(launch::resolve_visibility(&cpu, Some(OsStr::new("2")), &never).is_err());
    assert_eq!(never.calls.get(), 0);
    // Everything else is refused before any device work.
    let refuse = |observed: Option<&str>, smi: &Smi, needle: &str| {
        let e = launch::resolve_visibility(&gpu, observed.map(OsStr::new), smi).unwrap_err();
        assert!(
            format!("{e:#}").contains(needle),
            "{observed:?}: {e:#} (expected `{needle}`)"
        );
    };
    // SmartCluster allocated another GPU than the planned one.
    refuse(Some("0"), &Smi::listing(host), "not rank 1's planned GPU");
    refuse(Some("1"), &Smi::listing(host), "not rank 1's planned GPU");
    refuse(
        Some("7"),
        &Smi::listing(host),
        "which nvidia-smi does not report",
    );
    for malformed in ["1,2", "2,2", "-1", " 2", "2 ", "+2", "0x2", "", "10000"] {
        refuse(
            Some(malformed),
            &Smi::listing(host),
            "is not exactly one SmartCluster GPU index",
        );
    }
    refuse(None, &Smi::listing(host), "is unset");
    refuse(Some("GPU-fixture-0"), &never, "differs from the planned");
    refuse(Some("2"), &Smi::with(None), "nvidia-smi is not installed");
    refuse(
        Some("2"),
        &Smi::with(Some((false, String::new()))),
        "nvidia-smi --query-gpu=index,uuid failed: NVML failure",
    );
    refuse(
        Some("2"),
        &Smi::listing("2 GPU-fixture-1\n"),
        "malformed nvidia-smi line",
    );
    refuse(
        Some("2"),
        &Smi::listing("2, [N/A]\n"),
        "malformed nvidia-smi line",
    );
    refuse(
        Some("2"),
        &Smi::listing("2, GPU-fixture-1\n2, GPU-another-device\n"),
        "reports GPU index 2 twice",
    );
    refuse(
        Some("2"),
        &Smi::listing("1, GPU-fixture-1\n2, GPU-fixture-1\n"),
        "reports GPU GPU-fixture-1 twice",
    );
}
