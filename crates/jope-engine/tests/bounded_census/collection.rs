use super::*;

#[test]
fn collected_census_round_trips_through_preflight() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let nvidia = FakeNvidia::with(&one_gpu(), "");
    let (census, report) = collect::collect(&spec, &with_commands(&nvidia), now).unwrap();
    assert_eq!(nvidia.calls(), 2, "one GPU query and one process query");
    // The record is exactly the typed census the preflight validator reads.
    let bytes = serde_json::to_vec(&census).unwrap();
    assert_eq!(HostCensus::from_bytes(&bytes).unwrap(), census);
    assert_eq!(report.census_sha256, sha(&bytes));
    assert_eq!(report.gpu_probe, GpuProbe::NvidiaSmi);
    assert!(!report.allocation_held);
    let h = &census.hosts[0];
    // The observed identity is the fixture host's files, exactly as written.
    assert_eq!(h.identity.machine_id, fixture_file("machine-id"));
    assert_eq!(h.identity.boot_id, fixture_file("boot_id"));
    assert_eq!(h.identity.host, fixture_file("hostname"));
    assert_eq!(
        (
            h.identity.host.as_str(),
            h.identity.machine_id.as_str(),
            h.identity.boot_id.as_str()
        ),
        (
            fixture_host::HOST,
            fixture_host::MACHINE_ID,
            fixture_host::BOOT_ID
        )
    );
    assert_eq!(h.identity.address, "127.0.0.1");
    assert_eq!(
        h.identity.certificate_sha256,
        sha(&[0x30, 0x05, 0x30, 0x03, 0x02, 0x01, 0x07])
    );
    assert_eq!(
        h.identity.executable_sha256,
        sha(b"fixture worker executable")
    );
    assert_eq!(h.identity.runtime.compiler_sha256, sha(b"fixture compiler"));
    assert_eq!(h.identity.runtime.cuda_sha256, None);
    assert_eq!(h.identity.runtime.nccl_sha256, None);
    assert_eq!(
        h.identity.runtime.driver_version.as_deref(),
        Some("550.54.14")
    );
    assert_eq!(h.gpus.len(), 1);
    assert_eq!(h.gpus[0].uuid, UUID0);
    assert_eq!(h.gpus[0].total_vram_bytes, 24_576 * MIB);
    assert_eq!(h.gpus[0].available_vram_bytes, 23_552 * MIB);
    assert!(h.ownership.is_empty());
    assert!(h.cpu_millicores_allocatable >= 1_000 && h.ram_bytes_allocatable > 0);
    // The identical collection is deterministic apart from the clock.
    let (again, _) = collect::collect(
        &spec,
        &with_commands(&FakeNvidia::with(&one_gpu(), "")),
        now,
    )
    .unwrap();
    assert_eq!(again.hosts[0].identity, h.identity);

    let plan = plan_for(now, &[(h.identity.clone(), Some((UUID0, 0)))]);
    let receipt = preflight(&root, "gpu", &plan, &census, now).unwrap();
    assert!(!receipt.allocation_held);
    assert_eq!(receipt.census_sha256, report.census_sha256);
    assert_eq!(receipt.hosts[&h.identity.host].devices.len(), 1);
    // A changed installed file is a different executable identity: preflight refuses it.
    fs::write(&spec.executable, b"substituted worker").unwrap();
    let (substituted, _) = collect::collect(
        &spec,
        &with_commands(&FakeNvidia::with(&one_gpu(), "")),
        now,
    )
    .unwrap();
    let refused = err(preflight(&root, "substituted", &plan, &substituted, now));
    assert!(refused.contains("executable_mismatch"), "{refused}");
}

#[test]
fn cpu_only_or_absent_nvidia_smi_records_no_gpu() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    // CUDA_VISIBLE_DEVICES="": nothing is queried and nothing is recorded.
    let never = FakeNvidia::with(&one_gpu(), "");
    let (hidden, report) =
        collect::collect(&cpu_only(spec.clone()), &with_commands(&never), now).unwrap();
    assert_eq!(never.calls(), 0, "no GPU probe runs for a CPU-only census");
    assert_eq!(
        report.gpu_probe,
        GpuProbe::NotQueriedCudaVisibleDevicesEmpty
    );
    // nvidia-smi absent: one attempted query, no GPU recorded.
    let absent = FakeNvidia {
        absent: true,
        ..FakeNvidia::default()
    };
    let (missing, report) = collect::collect(&spec, &with_commands(&absent), now).unwrap();
    assert_eq!(absent.calls(), 1);
    assert_eq!(report.gpu_probe, GpuProbe::NvidiaSmiAbsent);
    for census in [&hidden, &missing] {
        let h = &census.hosts[0];
        assert!(h.gpus.is_empty() && h.ownership.is_empty());
        assert_eq!(h.identity.runtime.driver_version, None);
    }
    // Both are admitted for a CPU participant, and neither can fund a GPU participant.
    let identity = hidden.hosts[0].identity.clone();
    preflight(
        &root,
        "cpu",
        &plan_for(now, &[(identity.clone(), None)]),
        &hidden,
        now,
    )
    .unwrap();
    preflight(
        &root,
        "absent",
        &plan_for(now, &[(identity.clone(), None)]),
        &missing,
        now,
    )
    .unwrap();
    let mut gpu_identity = identity;
    gpu_identity.runtime.driver_version = None;
    let refused = err(preflight(
        &root,
        "gpu-on-cpu-census",
        &plan_for(now, &[(gpu_identity, Some((UUID0, 0)))]),
        &hidden,
        now,
    ));
    assert!(refused.contains("gpu_missing"), "{refused}");
}

#[test]
fn failing_malformed_or_unavailable_gpu_probes_refuse_the_collection() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let failing = FakeNvidia {
        fail: true,
        ..FakeNvidia::with(&one_gpu(), "")
    };
    assert!(
        err(collect::collect(&spec, &with_commands(&failing), now)).contains("GPU query failed")
    );
    for (gpus, apps, needle) in [
        (
            format!("0, {UUID0}, Fixture GPU, 550.54.14, [N/A], 23552\n"),
            String::new(),
            "unavailable value",
        ),
        (
            format!("0, {UUID0}, Fixture GPU, 550.54.14, 24576\n"),
            String::new(),
            "unexpected nvidia-smi record",
        ),
        (
            "0, 1111-2222, Fixture GPU, 550.54.14, 24576, 23552\n".to_string(),
            String::new(),
            "GPU-<uuid> form",
        ),
        (
            format!(
                "{}1, {UUID1}, Fixture GPU, 535.1, 24576, 23552\n",
                one_gpu()
            ),
            String::new(),
            "different driver versions",
        ),
        (
            one_gpu(),
            format!("{UUID1}, 4242, /usr/bin/fixture\n"),
            "did not report",
        ),
        (
            one_gpu(),
            format!("{UUID0}, pid, /usr/bin/fixture\n"),
            "nvidia-smi pid",
        ),
    ] {
        let refused = err(collect::collect(
            &spec,
            &with_commands(&FakeNvidia::with(&gpus, &apps)),
            now,
        ));
        assert!(refused.contains(needle), "{needle}: {refused}");
    }
    // A partial CUDA_VISIBLE_DEVICES would hide part of the host: refused before any probe.
    let mut partial = spec.clone();
    partial.cuda_visible_devices = Some(OsString::from("0"));
    let never = FakeNvidia::with(&one_gpu(), "");
    assert!(err(collect::collect(&partial, &with_commands(&never), now)).contains("host-wide"));
    assert_eq!(never.calls(), 0);
}

#[test]
fn device_processes_are_owners_that_preflight_refuses() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let apps = format!("No running processes found\n{UUID0}, 4242, /usr/bin/fixture, with comma\n");
    let (census, report) = collect::collect(
        &spec,
        &with_commands(&FakeNvidia::with(&one_gpu(), &apps)),
        now,
    )
    .unwrap();
    assert_eq!(report.device_owners, 1);
    let o = &census.hosts[0].ownership[0];
    assert_eq!(o.owner, "gpu-process-4242");
    assert_eq!(o.device_uuid.as_deref(), Some(UUID0));
    assert!(o.run.is_none() && o.attempt.is_none());
    let identity = census.hosts[0].identity.clone();
    let refused = collect_refusal(preflight(
        &root,
        "owned",
        &plan_for(now, &[(identity.clone(), Some((UUID0, 0)))]),
        &census,
        now,
    ));
    assert!(refused.iter().any(|r| matches!(r,
        RefusalReason::ConflictingOwnership { owner, uuid, .. }
            if owner == "gpu-process-4242" && uuid == UUID0)));
    // A CPU participant on the same host is not in conflict with a device process.
    preflight(
        &root,
        "cpu",
        &plan_for(now, &[(identity, None)]),
        &census,
        now,
    )
    .unwrap();
}
fn collect_refusal<T: std::fmt::Debug>(r: Result<T>) -> Vec<RefusalReason> {
    let e = r.expect_err("expected a preflight refusal");
    e.downcast_ref::<jope_parallel::native_loop::bounded::census::PreflightRefusal>()
        .expect("typed preflight refusal")
        .reasons
        .clone()
}

#[test]
fn unverifiable_identity_inputs_refuse_the_collection() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let never = FakeNvidia::default();
    let host = with_commands(&never);
    type SpecMutation<'a> = (&'a str, Box<dyn Fn(&mut CollectSpec)>);
    let cases: Vec<SpecMutation<'_>> = vec![
        (
            "not assigned to a local interface",
            Box::new(|s| s.address = "192.0.2.1".into()),
        ),
        (
            "not an IP address",
            Box::new(|s| s.address = "xbabe1".into()),
        ),
        (
            "source commit",
            Box::new(|s| s.source_commit = "ABC".into()),
        ),
        (
            "source commit",
            Box::new(|s| s.source_commit = s.source_commit.to_uppercase()),
        ),
        (
            "certificate",
            Box::new(|s| s.certificate = s.certificate.with_file_name("absent.der")),
        ),
        (
            "must be absolute",
            Box::new(|s| s.executable = PathBuf::from("bin/jope-worker")),
        ),
        (
            "resolve",
            Box::new(|s| s.compiler = s.compiler.with_file_name("absent")),
        ),
        (
            "at least one runtime library",
            Box::new(|s| s.runtime_libraries.clear()),
        ),
        (
            "at least one runtime library",
            Box::new(|s| s.installed_policies.clear()),
        ),
        (
            "named twice",
            Box::new(|s| s.installed_policies.push(s.installed_policies[0].clone())),
        ),
    ];
    for (needle, edit) in cases {
        let mut s = cpu_only(spec.clone());
        edit(&mut s);
        let refused = err(collect::collect(&s, &host, now));
        assert!(refused.contains(needle), "{needle}: {refused}");
    }
    for garbage in [
        b"not a certificate".to_vec(),
        b"-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----\n".to_vec(),
        vec![0x02, 0x01, 0x00],
    ] {
        fs::write(&spec.certificate, garbage).unwrap();
        assert!(err(collect::collect(&cpu_only(spec.clone()), &host, now)).contains("certificate"));
    }
    assert_eq!(never.calls(), 0);
}
