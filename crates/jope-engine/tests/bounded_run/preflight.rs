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
