//! W7-T01/T02/T04 fixtures: typed host census preflight refusals, the versioned
//! bounded-run contract, Plan, read-only Status/Verify and idempotent Cancel, the
//! controller CLI, and authenticated rendezvous, full-group admission and result
//! authentication. In-process and same-host only: no host is contacted, no GPU state is
//! read and no numerical work is started. These are fixtures, not the three-host proof
//! (W7-T07).
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    census::{preflight, HostCensus, OwnerKind, Ownership, RefusalReason},
    contract::{BoundedRunContract, PlanRefusal, WorkKind},
    controller::{self, Coordinator, Disposition, Liveness, Outcome, RankState},
    rendezvous::{self, Connected, GroupRefusal, Hello, HelloBody, RefusalCause, Rendezvous},
};
use std::{
    fs,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::Command,
};

fn plan_sha(c: &BoundedRunContract) -> String {
    sha(&bytes(c))
}
fn reasons(c: &BoundedRunContract, census: &HostCensus, now: u64) -> Vec<RefusalReason> {
    preflight(c, &plan_sha(c), census, "census", now)
        .unwrap_err()
        .reasons
}

// ---------------------------------------------------------------- W7-T01 preflight

#[test]
fn preflight_admits_a_fresh_matching_funded_census() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let receipt = preflight(&c, &plan_sha(&c), &census(&c, now), "census-sha", now).unwrap();
    assert_eq!(receipt.hosts.len(), 3);
    assert!(!receipt.allocation_held);
    for h in receipt.hosts.values() {
        assert_eq!(h.free_cpu_millicores, 24_000);
        assert_eq!(h.free_ram_bytes, 48 * GIB);
        assert_eq!(h.devices.len(), 1);
    }
}

#[test]
fn preflight_refuses_stale_and_future_telemetry() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let mut census = census(&c, now);
    census.captured_unix_ms = now + 10_000;
    census.hosts[1].observed_unix_ms = now - 120_000;
    census.hosts[2].observed_unix_ms = now + 10_000;
    let r = reasons(&c, &census, now);
    assert!(r.contains(&RefusalReason::StaleTelemetry {
        host: "fixture-host-1".into(),
        age_ms: 120_000
    }));
    assert!(r.contains(&RefusalReason::FutureTelemetry {
        host: "fixture-host-2".into(),
        ahead_ms: 10_000
    }));
    assert_eq!(r.len(), 2);
}

#[test]
fn preflight_refuses_mismatched_boot_gpu_source_executable_runtime_policy() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let mut census = census(&c, now);
    census.hosts[0].identity.boot_id = "rebooted".into();
    census.hosts[0].identity.machine_id = sha(b"another machine")[..32].into();
    census.hosts[0].gpus[0].uuid = "GPU-substituted".into();
    census.hosts[1].identity.executable_sha256 = sha(b"other executable");
    census.hosts[1].identity.runtime.nccl_sha256 = None;
    census.hosts[1].gpus[0].local_index = 1;
    census.hosts[2].identity.source_commit = sha(b"other source")[..40].into();
    census.hosts[2].identity.installed_policy_sha256 = sha(b"other policy");
    census.hosts[2].identity.certificate_sha256 = sha(b"other certificate");
    census.hosts[2].identity.address = "moved.invalid".into();
    let r = reasons(&c, &census, now);
    let h = |i: usize| format!("fixture-host-{i}");
    for expected in [
        RefusalReason::MachineMismatch { host: h(0) },
        RefusalReason::BootMismatch { host: h(0) },
        RefusalReason::GpuMissing {
            host: h(0),
            uuid: "GPU-fixture-0".into(),
        },
        RefusalReason::ExecutableMismatch { host: h(1) },
        RefusalReason::RuntimeMismatch { host: h(1) },
        RefusalReason::GpuIndexMismatch {
            host: h(1),
            uuid: "GPU-fixture-1".into(),
        },
        RefusalReason::SourceMismatch { host: h(2) },
        RefusalReason::PolicyMismatch { host: h(2) },
        RefusalReason::CertificateMismatch { host: h(2) },
        RefusalReason::AddressMismatch { host: h(2) },
    ] {
        assert!(r.contains(&expected), "missing {expected:?} in {r:?}");
    }
    assert_eq!(r.len(), 10);
}

#[test]
fn preflight_refuses_conflicting_device_ownership_but_not_its_own_attempt() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let owner = |kind, run: Option<&str>, attempt| Ownership {
        owner: "holder".into(),
        kind,
        device_uuid: Some("GPU-fixture-1".into()),
        cpu_millicores: 0,
        ram_bytes: 0,
        run: run.map(Into::into),
        attempt,
    };
    let mut ours = census(&c, now);
    ours.hosts[1]
        .ownership
        .push(owner(OwnerKind::Managed, Some("w7-fixture-run"), Some(1)));
    ours.validate().unwrap();
    preflight(&c, &plan_sha(&c), &ours, "census", now).unwrap();
    for conflicting in [
        owner(OwnerKind::Ci, None, None),
        owner(OwnerKind::User, None, None),
        owner(OwnerKind::Managed, Some("w7-fixture-run"), Some(2)),
        owner(OwnerKind::Managed, Some("another-run"), Some(1)),
    ] {
        let mut census = census(&c, now);
        census.hosts[1].ownership.push(conflicting);
        census.validate().unwrap();
        assert_eq!(
            reasons(&c, &census, now),
            vec![RefusalReason::ConflictingOwnership {
                host: "fixture-host-1".into(),
                uuid: "GPU-fixture-1".into(),
                owner: "holder".into()
            }]
        );
    }
}

#[test]
fn preflight_refuses_unavailable_capacity() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let mut census = census(&c, now);
    census.hosts[0].cpu_millicores_allocatable = 10_000; // 8000 reserved by CI
    census.hosts[1].ram_bytes_allocatable = 20 * GIB; // 16 GiB reserved by CI
    census.hosts[2].gpus[0].available_vram_bytes = 10 * GIB;
    let r = reasons(&c, &census, now);
    assert_eq!(
        r,
        vec![
            RefusalReason::InsufficientCpu {
                host: "fixture-host-0".into(),
                free: 2_000,
                required: 4_000
            },
            RefusalReason::InsufficientRam {
                host: "fixture-host-1".into(),
                free: 4 * GIB,
                required: 8 * GIB
            },
            RefusalReason::InsufficientVram {
                host: "fixture-host-2".into(),
                uuid: "GPU-fixture-2".into(),
                free: 10 * GIB,
                required: 20 * GIB
            },
        ]
    );
    let mut missing = census_for(&c, now);
    missing.hosts.remove(2);
    assert_eq!(
        reasons(&c, &missing, now),
        vec![RefusalReason::MissingHost {
            host: "fixture-host-2".into()
        }]
    );
}
fn census_for(c: &BoundedRunContract, now: u64) -> HostCensus {
    census(c, now)
}

#[test]
fn census_record_is_typed_versioned_and_bounded() {
    let now = now_ms();
    let k = keys(3);
    let c = contract(now, 3, &k);
    let good = serde_json::to_vec(&census(&c, now)).unwrap();
    HostCensus::from_bytes(&good).unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(&good).unwrap();
    v["schema"] = "jope.native_host_census.v0".into();
    assert!(HostCensus::from_bytes(&serde_json::to_vec(&v).unwrap()).is_err());
    let mut v: serde_json::Value = serde_json::from_slice(&good).unwrap();
    v["hosts"][0]["unexpected"] = true.into();
    assert!(HostCensus::from_bytes(&serde_json::to_vec(&v).unwrap()).is_err());
    let mut dup = census(&c, now);
    dup.hosts.push(dup.hosts[0].clone());
    assert!(dup.validate().is_err());
    let mut dup_gpu = census(&c, now);
    dup_gpu.hosts[1].gpus[0].uuid = "GPU-fixture-0".into();
    assert!(dup_gpu.validate().is_err());
    let mut ghost = census(&c, now);
    ghost.hosts[0].ownership[0].device_uuid = Some("GPU-absent".into());
    assert!(ghost.validate().is_err());
    let mut late = census(&c, now);
    late.hosts[0].observed_unix_ms = now + 1;
    assert!(late.validate().is_err(), "observation after capture");
    assert!(HostCensus::from_bytes(&vec![b' '; 1024 * 1024 + 1]).is_err());
}

// ---------------------------------------------------------------- W7-T02 plan

struct Run {
    _t: tempfile::TempDir,
    root: PathBuf,
    dir: PathBuf,
    keys: Keys,
    contract: BoundedRunContract,
    sha: String,
}
fn planned(now: u64) -> Run {
    let (t, root) = tempdir();
    write_closure(&root);
    let k = keys(3);
    let c = contract(now, 3, &k);
    let dir = root.join("run");
    let report = controller::plan(&dir, &bytes(&c), now, None).unwrap();
    assert!(report.created);
    Run {
        _t: t,
        root,
        dir,
        sha: report.plan_sha256,
        keys: k,
        contract: c,
    }
}

#[test]
fn plan_freezes_canonical_contract_and_is_idempotent() {
    let now = now_ms();
    let r = planned(now);
    let frozen = fs::read(r.dir.join("plan.json")).unwrap();
    assert_eq!(frozen, r.contract.canonical_bytes().unwrap());
    assert_eq!(r.sha, sha(&frozen));
    let status = controller::status(&r.dir).unwrap();
    assert_eq!(status.journal_records, 1);
    assert_eq!(
        status.original_deadline_unix_ms,
        r.contract.original_deadline_unix_ms
    );
    assert!(status.ranks.iter().all(|x| x.state == RankState::Scheduled));
    let before = tree(&r.dir);
    let again = controller::plan(&r.dir, &bytes(&r.contract), now, None).unwrap();
    assert!(!again.created);
    assert_eq!(again.plan_sha256, r.sha);
    assert_eq!(again.review["dispatch_started"], false);
    assert_eq!(tree(&r.dir), before);
    let mut other = r.contract.clone();
    other.counters.planned_updates = 4;
    for w in other
        .work_census
        .iter_mut()
        .filter(|w| w.kind == WorkKind::Numerical)
    {
        w.updates = 4;
    }
    assert!(controller::plan(&r.dir, &bytes(&other), now, None)
        .unwrap_err()
        .to_string()
        .contains("different plan"));
    assert_eq!(tree(&r.dir), before);
}

fn refused(c: &BoundedRunContract, now: u64) -> (PlanRefusal, String) {
    let typed = match BoundedRunContract::parse(&bytes(c)) {
        Err(e) => e,
        Ok(parsed) => parsed.validate_at(now).unwrap_err(),
    };
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let e = controller::plan(&dir, &bytes(c), now, None)
        .unwrap_err()
        .to_string();
    assert!(!dir.exists(), "a refused plan created its run directory");
    (typed, e)
}

#[test]
fn invalid_plans_fail_before_anything_is_created() {
    let now = now_ms();
    let k = keys(3);
    let base = contract(now, 3, &k);
    type Edit = Box<dyn Fn(&mut BoundedRunContract)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("rank order", Box::new(|c| c.participants.swap(0, 1))),
        (
            "duplicate device",
            Box::new(|c| c.participants[1].device = c.participants[0].device.clone()),
        ),
        (
            "cpu participant with vram",
            Box::new(|c| c.participants[2].device = None),
        ),
        (
            "local rank gap",
            Box::new(|c| c.participants[1].local_rank = 1),
        ),
        (
            "grant for another rank",
            Box::new(|c| c.participants[1].grant = c.participants[0].grant.clone()),
        ),
        (
            "grant signature",
            Box::new(|c| c.participants[0].grant.body.expires_unix_ms -= 1),
        ),
        (
            "no verification work",
            Box::new(|c| {
                c.work_census
                    .retain(|w| !(w.global_rank == 2 && w.kind == WorkKind::Verification))
            }),
        ),
        (
            "no cleanup work",
            Box::new(|c| {
                c.work_census
                    .retain(|w| !(w.global_rank == 0 && w.kind == WorkKind::Cleanup))
            }),
        ),
        (
            "unequal synchronous updates",
            Box::new(|c| c.work_census[0].updates = 2),
        ),
        (
            "updates on non-numerical work",
            Box::new(|c| c.work_census[1].updates = 1),
        ),
        ("closure order", Box::new(|c| c.artifact_closure.reverse())),
        (
            "result destination collides with closure",
            Box::new(|c| c.participants[0].result_destination.path = "inputs/corpus.bin".into()),
        ),
        (
            "message limit",
            Box::new(|c| c.rendezvous.max_message_bytes = 16 * 1024 * 1024),
        ),
        ("attempt without predecessor", Box::new(|c| c.attempt = 2)),
        (
            "two hosts share a machine id",
            Box::new(|c| {
                c.participants[1].host.machine_id = c.participants[0].host.machine_id.clone()
            }),
        ),
        (
            "malformed machine id",
            Box::new(|c| c.participants[2].host.machine_id = "fixture-host-2".into()),
        ),
        (
            "shared participant key",
            Box::new(|c| c.participants[1].public_key = c.participants[0].public_key.clone()),
        ),
        (
            "grant outlives deadline",
            Box::new(|c| {
                c.participants[0].grant.body.expires_unix_ms = c.original_deadline_unix_ms + 1
            }),
        ),
    ];
    for (name, edit) in cases {
        let mut c = base.clone();
        edit(&mut c);
        if !matches!(name, "grant signature" | "grant for another rank") {
            resign(&mut c, &k);
        }
        let (typed, message) = refused(&c, now);
        assert!(
            matches!(typed, PlanRefusal::Invalid(_)),
            "{name}: expected invalid, got {typed:?}"
        );
        assert!(
            message.contains("invalid bounded-run plan"),
            "{name}: {message}"
        );
    }
}

#[test]
fn underfunded_plans_fail_before_dispatch() {
    let now = now_ms();
    let k = keys(3);
    let base = contract(now, 3, &k);
    type Edit = Box<dyn Fn(&mut BoundedRunContract)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "grant below cpu ceiling",
            Box::new(|c| c.participants[0].grant.body.cpu_millicores = 3_999),
        ),
        (
            "grant below vram ceiling",
            Box::new(|c| c.participants[1].grant.body.gpu_vram_bytes = Some(GIB)),
        ),
        (
            "cpu work exceeds funded core-ms",
            Box::new(|c| c.work_census[0].cpu_core_ms = 3_000_000),
        ),
        (
            "wall census exceeds ceiling",
            Box::new(|c| c.work_census[0].wall_ms = 590_000),
        ),
        (
            "peak memory exceeds ceiling",
            Box::new(|c| c.work_census[0].peak_ram_bytes = 9 * GIB),
        ),
        (
            "grant window shorter than wall ceiling",
            Box::new(|c| {
                c.participants[2].grant.body.expires_unix_ms = c.not_before_unix_ms + 500_000
            }),
        ),
    ];
    for (name, edit) in cases {
        let mut c = base.clone();
        edit(&mut c);
        resign(&mut c, &k);
        let (typed, message) = refused(&c, now);
        assert!(
            matches!(typed, PlanRefusal::Underfunded(_)),
            "{name}: expected underfunded, got {typed:?}"
        );
        assert!(
            message.contains("underfunded bounded-run plan"),
            "{name}: {message}"
        );
    }
    // The remaining original window, not the nominal one, must fund the ceilings.
    let late = base.original_deadline_unix_ms - 500_000;
    let (typed, _) = refused(&base, late);
    assert!(matches!(typed, PlanRefusal::Underfunded(_)), "{typed:?}");
    let (typed, _) = refused(&base, base.original_deadline_unix_ms);
    assert!(matches!(typed, PlanRefusal::Invalid(_)), "{typed:?}");
}

#[test]
fn legacy_manifests_keep_legacy_semantics() {
    use jain_smartcluster::{JobId, ResourceRequest, Timestamp};
    use jope_parallel::native_graph::Artifact;
    let a = Artifact {
        path: "a.json".into(),
        sha256: "a".repeat(64),
    };
    let legacy = jope_parallel::native_loop::Config {
        schema: "jope.native_loop.v1".into(),
        run: "legacy".into(),
        accepted_checkpoint: a.clone(),
        initial_corpus: a.clone(),
        accepted_decision: a.clone(),
        scientific_policy: a.clone(),
        label_admission_policy: a,
        fit_policy: "policy.json".into(),
        run_approval: "/etc/jope-dime/approval.json".into(),
        resident_job: JobId::new(),
        workers: 48,
        cycles: 2,
        initial_optimizer_step: 60_000,
        predecessor_run: None,
        updates_per_block: 3_000,
        global_batch: 2_048,
        microbatch: 256,
        direct_seconds: 30,
        continuation_evaluations: 16,
        continuation_seconds: 60,
        cpu_resources: ResourceRequest::cpu_and_ram(1_000, GIB),
        task_deadline: Timestamp::from_unix_millis(u64::MAX),
        qualification: false,
        preparation_overlay: None,
        cuts: 1,
        cut_start: 0,
        materialization: None,
    };
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    // The old loader still reads its own manifest unchanged ...
    let reread: jope_parallel::native_loop::Config = serde_json::from_slice(&legacy_bytes).unwrap();
    assert_eq!(serde_json::to_vec(&reread).unwrap(), legacy_bytes);
    // ... is never reinterpreted as a bounded plan ...
    assert!(matches!(
        BoundedRunContract::parse(&legacy_bytes),
        Err(PlanRefusal::Legacy(_))
    ));
    let (_t, root) = tempdir();
    let e = controller::plan(&root.join("run"), &legacy_bytes, now_ms(), None).unwrap_err();
    assert!(e.to_string().contains("legacy fixed run contract"));
    assert!(!root.join("run").exists());
    // ... and a bounded contract is not an old manifest.
    let k = keys(3);
    let bounded = bytes(&contract(now_ms(), 3, &k));
    assert!(serde_json::from_slice::<jope_parallel::native_loop::Config>(&bounded).is_err());
}

// ------------------------------------------------- W7-T02 status / verify / cancel

/// Drive an in-process rendezvous to a persisted admission.
fn admit(
    c: &mut Coordinator<FakeSupervisor>,
    r: &Run,
    now: u64,
) -> (controller::PersistedAdmission, Vec<Hello>) {
    let mut rv = c
        .open_rendezvous(
            &r.keys.coordinator,
            "fixture-coordinator-boot",
            &[1; 16],
            now,
        )
        .unwrap();
    let closure =
        rendezvous::observed_closure_digest(&r.root, &r.contract.artifact_closure).unwrap();
    let mut hellos = Vec::new();
    for rank in 0..3 {
        let body = hello_body(&r.contract, &r.sha, rv.challenge(), rank, now, &closure);
        let hello = rendezvous::sign_hello(&r.keys.ranks[rank as usize], body).unwrap();
        assert_eq!(
            rv.offer(&serde_json::to_vec(&hello).unwrap(), now).unwrap(),
            rank
        );
        hellos.push(hello);
    }
    let admission = rv.admit(&r.keys.coordinator, now).unwrap().unwrap();
    (c.record_admission(admission, now).unwrap(), hellos)
}
fn deliver(
    c: &mut Coordinator<FakeSupervisor>,
    r: &Run,
    persisted: &controller::PersistedAdmission,
    hellos: &[Hello],
    now: u64,
) {
    let mut connected = Vec::new();
    let mut peers = Vec::new();
    for rank in 0..3 {
        let (a, b) = UnixStream::pair().unwrap();
        connected.push(Connected {
            global_rank: rank,
            stream: a,
        });
        peers.push(b);
    }
    c.deliver_admission(persisted, &mut connected, now).unwrap();
    for (rank, mut peer) in peers.into_iter().enumerate() {
        let frame =
            rendezvous::read_frame(&mut peer, r.contract.rendezvous.max_message_bytes).unwrap();
        let member =
            rendezvous::verify_admission(&r.contract, &r.sha, &hellos[rank], &frame, now).unwrap();
        assert_eq!(member.global_rank(), rank as u32);
        assert_eq!(member.communicator_bootstrap(), [0xab; 128]);
    }
}
fn states(dir: &Path) -> Vec<RankState> {
    controller::status(dir)
        .unwrap()
        .ranks
        .iter()
        .map(|r| r.state)
        .collect()
}
fn live_states(c: &Coordinator<FakeSupervisor>) -> Vec<RankState> {
    c.status().unwrap().ranks.iter().map(|r| r.state).collect()
}

#[test]
fn status_distinguishes_all_seven_states() {
    use RankState::*;
    let now = now_ms();
    // scheduled, unknown, admitted, running, committed
    let r = planned(now);
    assert_eq!(states(&r.dir), vec![Scheduled; 3]);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.record_dispatch_intent(0, now).unwrap();
    assert_eq!(live_states(&c), vec![Unknown, Scheduled, Scheduled]);
    assert!(controller::status(&r.dir).unwrap().coordinator_live);
    c.dispatch_all(now).unwrap();
    assert_eq!(live_states(&c), vec![Scheduled; 3]);
    let (persisted, hellos) = admit(&mut c, &r, now);
    assert_eq!(live_states(&c), vec![Admitted; 3]);
    deliver(&mut c, &r, &persisted, &hellos, now);
    assert_eq!(live_states(&c), vec![Running; 3]);
    assert_eq!(states(&r.dir), vec![Running; 3]);
    for rank in 0..3u32 {
        let id = rendezvous::dispatch_id(&r.sha, 1, rank).unwrap();
        let retained = result(
            &r.contract,
            &r.sha,
            persisted.admission_sha256(),
            rank,
            &r.keys.ranks[rank as usize],
        );
        fake.set(
            &id,
            Liveness::Exited {
                code: Some(0),
                signal: None,
            },
            Some(retained),
        );
    }
    let done = c.poll(now).unwrap();
    assert_eq!(done.outcome, Some(Outcome::Committed));
    assert_eq!(done.committed_updates_total, 60_003);
    drop(c);
    assert_eq!(states(&r.dir), vec![Committed; 3]);
    let v = controller::verify(&r.dir, Some(&r.sha)).unwrap();
    assert_eq!(v.results_verified, 3);
    assert!(!v.numerical_work_started);

    // unknown: launched work with no live coordinator holding the run
    let r2 = planned(now);
    let fake2 = FakeSupervisor::default();
    let mut c2 = Coordinator::open(&r2.dir, &r2.sha, fake2.clone(), now).unwrap();
    c2.preflight(
        &serde_json::to_vec(&census(&r2.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c2.dispatch_all(now).unwrap();
    let (persisted2, hellos2) = admit(&mut c2, &r2, now);
    deliver(&mut c2, &r2, &persisted2, &hellos2, now);
    drop(c2);
    let s = controller::status(&r2.dir).unwrap();
    assert!(!s.coordinator_live);
    assert_eq!(
        s.ranks.iter().map(|x| x.state).collect::<Vec<_>>(),
        vec![Unknown; 3]
    );

    // failed + drained: one rank exits nonzero; the whole group drains
    let mut c2 = Coordinator::open(&r2.dir, &r2.sha, fake2.clone(), now).unwrap();
    assert!(c2.needs_reconcile());
    assert!(
        c2.poll(now).is_err(),
        "no supervision step before reconcile"
    );
    drop(c2);
    let r3 = planned(now);
    let fake3 = FakeSupervisor::default();
    let mut c3 = Coordinator::open(&r3.dir, &r3.sha, fake3.clone(), now).unwrap();
    c3.preflight(
        &serde_json::to_vec(&census(&r3.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c3.dispatch_all(now).unwrap();
    let (persisted3, hellos3) = admit(&mut c3, &r3, now);
    deliver(&mut c3, &r3, &persisted3, &hellos3, now);
    let id1 = rendezvous::dispatch_id(&r3.sha, 1, 1).unwrap();
    fake3.set(
        &id1,
        Liveness::Exited {
            code: Some(1),
            signal: None,
        },
        None,
    );
    let s = c3.poll(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        s.ranks.iter().map(|x| x.state).collect::<Vec<_>>(),
        vec![Drained, Failed, Drained]
    );
    assert_eq!(fake3.running(), 0);
    controller::verify(&r3.dir, Some(&r3.sha)).unwrap();
}

#[test]
fn status_and_verify_are_read_only_and_start_nothing() {
    let now = now_ms();
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let (persisted, hellos) = admit(&mut c, &r, now);
    deliver(&mut c, &r, &persisted, &hellos, now);
    let launches = fake.0.lock().unwrap().launches;
    drop(c);
    let before = tree(&r.root);
    for _ in 0..2 {
        controller::status(&r.dir).unwrap();
        let v = controller::verify(&r.dir, Some(&r.sha)).unwrap();
        assert!(!v.numerical_work_started);
    }
    let exe = env!("CARGO_BIN_EXE_jope-native-loop");
    for args in [
        vec!["status", "--run-dir", r.dir.to_str().unwrap()],
        vec![
            "verify",
            "--run-dir",
            r.dir.to_str().unwrap(),
            "--plan-sha256",
            &r.sha,
        ],
    ] {
        let out = Command::new(exe).args(&args).output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        tree(&r.root),
        before,
        "status/verify changed the run directory"
    );
    let s = fake.0.lock().unwrap();
    assert_eq!((s.launches, s.terminations), (launches, 0));
    drop(s);
    let absent = r.root.join("never-planned");
    assert!(controller::status(&absent).is_err());
    assert!(controller::verify(&absent, None).is_err());
    assert!(!absent.exists());
    assert!(controller::verify(&r.dir, Some(&"0".repeat(64))).is_err());
}

#[test]
fn verify_detects_tampering_with_plan_journal_or_results() {
    let now = now_ms();
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let (persisted, hellos) = admit(&mut c, &r, now);
    deliver(&mut c, &r, &persisted, &hellos, now);
    for rank in 0..3u32 {
        let id = rendezvous::dispatch_id(&r.sha, 1, rank).unwrap();
        let retained = result(
            &r.contract,
            &r.sha,
            persisted.admission_sha256(),
            rank,
            &r.keys.ranks[rank as usize],
        );
        fake.set(
            &id,
            Liveness::Exited {
                code: Some(0),
                signal: None,
            },
            Some(retained),
        );
    }
    assert_eq!(c.poll(now).unwrap().outcome, Some(Outcome::Committed));
    drop(c);
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
    let result_path = r.dir.join("results/rank-1/result.json");
    let original = fs::read(&result_path).unwrap();
    fs::write(&result_path, b"{\"fixture_rank\":9}").unwrap();
    assert!(controller::verify(&r.dir, None).is_err());
    fs::write(&result_path, &original).unwrap();
    controller::verify(&r.dir, None).unwrap();
    let record = r.dir.join("journal/00000004.json");
    let original = fs::read(&record).unwrap();
    let mut edited = original.clone();
    let at = edited.iter().position(|b| *b == b'1').unwrap();
    edited[at] = b'2';
    fs::write(&record, &edited).unwrap();
    assert!(controller::verify(&r.dir, None).is_err());
    assert!(controller::status(&r.dir).is_err());
    fs::write(&record, &original).unwrap();
    let plan = r.dir.join("plan.json");
    let frozen = fs::read(&plan).unwrap();
    let mut other = r.contract.clone();
    other.run = "w7-fixture-other".into();
    resign(&mut other, &r.keys);
    fs::write(&plan, bytes(&other)).unwrap();
    assert!(
        controller::verify(&r.dir, None).is_err(),
        "journal bound to another plan"
    );
    fs::write(&plan, &frozen).unwrap();
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

#[test]
fn cancel_is_idempotent_and_bound_to_the_plan() {
    let now = now_ms();
    // Before dispatch and with no live coordinator: terminal immediately.
    let r = planned(now);
    let before = tree(&r.dir);
    assert!(controller::cancel(&r.dir, &"f".repeat(64), "wrong plan", now).is_err());
    assert_eq!(tree(&r.dir), before, "a refused cancel wrote state");
    let first = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(!first.already_requested && first.journaled);
    assert_eq!(first.outcome, Some(Outcome::Cancelled));
    let after = tree(&r.dir);
    let second = controller::cancel(&r.dir, &r.sha, "a different reason", now + 5).unwrap();
    assert!(second.already_requested && !second.journaled);
    assert_eq!(second.request, first.request);
    assert_eq!(tree(&r.dir), after, "repeat cancel changed state");
    let s = controller::status(&r.dir).unwrap();
    assert!(s
        .ranks
        .iter()
        .all(|x| x.disposition == Some(Disposition::NeverStarted)));
    controller::verify(&r.dir, Some(&r.sha)).unwrap();

    // With a live coordinator: delivered, then the whole group is drained by poll.
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let journal = controller::status(&r.dir).unwrap().journal_records;
    let live = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(live.delivered_to_live_coordinator && !live.journaled);
    assert_eq!(controller::status(&r.dir).unwrap().journal_records, journal);
    assert!(controller::status(&r.dir).unwrap().cancel_requested);
    assert!(
        c.record_dispatch_intent(0, now).is_err(),
        "no dispatch once cancelled"
    );
    let s = c.poll(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert_eq!(fake.running(), 0);
    assert!(s.ranks.iter().all(|x| x.state == RankState::Drained));
    drop(c);
    let again = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(again.already_requested && !again.journaled);

    // Dispatched work and no live coordinator: the drain waits for reconciliation.
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    drop(c);
    let pending = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(pending.drain_pending_reconcile && pending.outcome.is_none());
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    assert!(c.needs_reconcile());
    assert!(c.record_dispatch_intent(0, now).is_err());
    let s = c.reconcile(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert_eq!(fake.running(), 0);
}

#[test]
fn execution_binds_to_the_planned_artifact() {
    let now = now_ms();
    let r = planned(now);
    let other = sha(b"some other plan");
    assert!(Coordinator::open(&r.dir, &other, FakeSupervisor::default(), now).is_err());
    let c = Coordinator::open(&r.dir, &r.sha, FakeSupervisor::default(), now).unwrap();
    assert!(
        Coordinator::open(&r.dir, &r.sha, FakeSupervisor::default(), now).is_err(),
        "one writer per run"
    );
    drop(c);
    // Dispatch before an admitted preflight is refused and starts nothing.
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    assert!(c.record_dispatch_intent(0, now).is_err());
    let mut stale = census(&r.contract, now);
    stale.hosts[0].observed_unix_ms = now - 120_000;
    assert!(c
        .preflight(&serde_json::to_vec(&stale).unwrap(), now)
        .is_err());
    assert!(c.record_dispatch_intent(0, now).is_err());
    assert!(c
        .open_rendezvous(&r.keys.coordinator, "boot", &[0; 16], now)
        .is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    let s = c.status().unwrap();
    assert!(s.ranks.iter().all(|x| x.state == RankState::Scheduled));
    // A fresh preflight admits; a stale receipt no longer permits dispatch later.
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    assert!(c.record_dispatch_intent(0, now + 61_000).is_err());
    c.record_dispatch_intent(0, now).unwrap();
    assert_eq!(fake.0.lock().unwrap().launches, 0, "intent is not launch");
}

// ---------------------------------------------------------------- W7-T02 CLI

fn cli(args: &[&str]) -> (bool, serde_json::Value, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let json = serde_json::from_str(stdout.trim()).unwrap_or(serde_json::Value::Null);
    (
        out.status.success(),
        json,
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn cli_plan_status_verify_cancel_round_trip() {
    let (_t, root) = tempdir();
    let k = keys(3);
    let c = contract(now_ms(), 3, &k);
    let file = root.join("contract.json");
    fs::write(&file, bytes(&c)).unwrap();
    let dir = root.join("run");
    let d = dir.to_str().unwrap();
    let (ok, plan, err) = cli(&["plan", "--run-dir", d, "--contract", file.to_str().unwrap()]);
    assert!(ok, "{err}");
    let sha = plan["plan_sha256"].as_str().unwrap().to_string();
    assert_eq!(plan["review"]["world_size"], 3);
    assert_eq!(plan["review"]["dispatch_started"], false);
    let (ok, status, _) = cli(&["status", "--run-dir", d]);
    assert!(ok);
    assert!(status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["state"] == "scheduled"));
    let (ok, verify, _) = cli(&["verify", "--run-dir", d, "--plan-sha256", &sha]);
    assert!(ok);
    assert_eq!(verify["numerical_work_started"], false);
    let (ok, _, _) = cli(&["verify", "--run-dir", d, "--plan-sha256", &"0".repeat(64)]);
    assert!(!ok);
    let (ok, cancel, _) = cli(&[
        "cancel",
        "--run-dir",
        d,
        "--plan-sha256",
        &sha,
        "--reason",
        "stop",
    ]);
    assert!(ok);
    assert_eq!(cancel["outcome"], "cancelled");
    let (ok, again, _) = cli(&[
        "cancel",
        "--run-dir",
        d,
        "--plan-sha256",
        &sha,
        "--reason",
        "stop",
    ]);
    assert!(ok);
    assert_eq!(again["already_requested"], true);
    assert_eq!(again["journaled"], false);
}

#[test]
fn cli_refuses_mixed_legacy_arguments_and_bad_contracts_before_creating_state() {
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let d = dir.to_str().unwrap();
    let legacy_args = [
        "--root",
        "/x",
        "--state",
        "/x/s",
        "--graph-state",
        "/x/g",
        "--worker-state",
        "/x/w",
        "--config",
        "/x/c",
        "--config-sha256",
        "00",
        "--socket",
        "/x/sock",
        "--tenant",
        "t",
        "--project",
        "p",
    ];
    let mut mixed: Vec<&str> = legacy_args.to_vec();
    mixed.extend(["status", "--run-dir", d]);
    let (ok, _, err) = cli(&mixed);
    assert!(!ok && err.contains("take no fixed-loop arguments"), "{err}");
    let (ok, _, err) = cli(&["tick"]);
    assert!(!ok && err.contains("require --root"), "{err}");
    let k = keys(3);
    let mut bad = contract(now_ms(), 3, &k);
    bad.work_census[0].cpu_core_ms = 3_000_000;
    resign(&mut bad, &k);
    let file = root.join("bad.json");
    fs::write(&file, bytes(&bad)).unwrap();
    let (ok, _, err) = cli(&["plan", "--run-dir", d, "--contract", file.to_str().unwrap()]);
    assert!(!ok && err.contains("underfunded"), "{err}");
    assert!(!dir.exists());
    let help = Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .arg("--help")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    for verb in [
        "bind",
        "bind-parent-decision",
        "tick",
        "serve",
        "plan",
        "status",
        "verify",
        "cancel",
    ] {
        assert!(
            help.lines().any(|l| l.trim_start().starts_with(verb)),
            "help lacks {verb}: {help}"
        );
    }
}

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
