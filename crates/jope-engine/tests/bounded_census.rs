//! W7-T01 census collector and merge fixtures.
//!
//! The collector reads this host's real machine id, boot id, host name and interface
//! addresses. GPU inventory always comes from an injected command runner (or is hidden with
//! `CUDA_VISIBLE_DEVICES=""`): no fixture runs `nvidia-smi` or touches a GPU. Multi-host
//! documents are derived from one local collection with synthetic identities, so they are
//! fixtures, not a live three-host census.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use anyhow::Result;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    census::{HostCensus, HostIdentity, RefusalReason},
    collect::{
        self, CollectSpec, CommandOutput, CommandRunner, GpuProbe, MergePolicy, PinnedDocument,
    },
    contract::{BoundedRunContract, Ceilings, DeviceAssignment, WorkItem, WorkKind},
    controller,
};
use std::{
    cell::RefCell,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const MIB: u64 = 1024 * 1024;
const UUID0: &str = "GPU-11111111-2222-3333-4444-555555555555";
const UUID1: &str = "GPU-66666666-7777-8888-9999-aaaaaaaaaaaa";

/// Injected `nvidia-smi`: records every call, never runs anything.
#[derive(Default)]
struct FakeNvidia {
    absent: bool,
    fail: bool,
    gpus: String,
    apps: String,
    calls: RefCell<Vec<Vec<String>>>,
}
impl FakeNvidia {
    fn with(gpus: &str, apps: &str) -> Self {
        Self {
            gpus: gpus.into(),
            apps: apps.into(),
            ..Self::default()
        }
    }
    fn calls(&self) -> usize {
        self.calls.borrow().len()
    }
}
impl CommandRunner for FakeNvidia {
    fn run(&self, program: &str, args: &[&str]) -> Result<Option<CommandOutput>> {
        assert_eq!(program, "nvidia-smi", "the collector runs nothing else");
        self.calls
            .borrow_mut()
            .push(args.iter().map(|a| a.to_string()).collect());
        if self.absent {
            return Ok(None);
        }
        let stdout = if args[0].starts_with("--query-gpu") {
            &self.gpus
        } else {
            assert!(args[0].starts_with("--query-compute-apps"));
            &self.apps
        };
        Ok(Some(CommandOutput {
            success: !self.fail,
            stdout: stdout.clone().into_bytes(),
            stderr: b"fixture failure".to_vec(),
        }))
    }
}
fn one_gpu() -> String {
    format!("0, {UUID0}, Fixture GPU, 550.54.14, 24576, 23552\n")
}

/// Installed files as a host would have them.
fn install(root: &Path) -> CollectSpec {
    let files: [(&str, &[u8]); 5] = [
        ("bin/jope-worker", b"fixture worker executable"),
        ("lib/libjope-runtime.so", b"fixture runtime library"),
        ("bin/compiler", b"fixture compiler"),
        ("etc/worker.json", b"{\"fixture\":\"policy\"}"),
        // A syntactically framed DER certificate (outer and TBS SEQUENCE); the transport,
        // not the collector, verifies certificates.
        ("etc/host.der", &[0x30, 0x05, 0x30, 0x03, 0x02, 0x01, 0x07]),
    ];
    for (p, b) in files {
        let path = root.join(p);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b).unwrap();
    }
    CollectSpec {
        address: "127.0.0.1".into(),
        certificate: root.join("etc/host.der"),
        source_commit: sha(b"fixture-source")[..40].to_string(),
        executable: root.join("bin/jope-worker"),
        runtime_libraries: vec![root.join("lib/libjope-runtime.so")],
        compiler: root.join("bin/compiler"),
        cuda_libraries: Vec::new(),
        nccl_libraries: Vec::new(),
        installed_policies: vec![root.join("etc/worker.json")],
        cuda_visible_devices: None,
    }
}
fn cpu_only(mut spec: CollectSpec) -> CollectSpec {
    spec.cuda_visible_devices = Some(OsString::new());
    spec
}
fn local(path: &str) -> String {
    fs::read_to_string(path).unwrap().trim().to_string()
}

/// A small funded plan whose participants are exactly the given observed hosts, so it can be
/// admitted on any machine that runs the fixture (1 CPU and 256 MiB per participant).
fn plan_for(now: u64, hosts: &[(HostIdentity, Option<(&str, u32)>)]) -> BoundedRunContract {
    let keys = keys(hosts.len());
    let mut c = contract(now, hosts.len(), &keys);
    c.work_census.clear();
    for (i, (identity, device)) in hosts.iter().enumerate() {
        let p = &mut c.participants[i];
        p.host = identity.clone();
        p.device = device.map(|(uuid, local_index)| DeviceAssignment {
            uuid: uuid.into(),
            local_index,
        });
        p.ceilings = Ceilings {
            cpu_millicores: 1_000,
            ram_bytes: 256 * MIB,
            gpu_vram_bytes: device.map(|_| GIB),
            wall_ms: 600_000,
        };
        let gpu = device.is_some();
        for (kind, cpu, wall, ram, updates) in [
            (WorkKind::Numerical, 300_000, 300_000, 128 * MIB, 3),
            (WorkKind::Verification, 10_000, 30_000, 64 * MIB, 0),
            (WorkKind::ArtifactMovement, 5_000, 20_000, 64 * MIB, 0),
            (WorkKind::Cleanup, 1_000, 10_000, 64 * MIB, 0),
            (WorkKind::FailureAllowance, 20_000, 60_000, 64 * MIB, 0),
        ] {
            let numerical = kind == WorkKind::Numerical;
            c.work_census.push(WorkItem {
                id: format!(
                    "rank-{i}-{}",
                    serde_json::to_value(kind).unwrap().as_str().unwrap()
                ),
                global_rank: i as u32,
                kind,
                cpu_core_ms: cpu,
                gpu_ms: if gpu && numerical { 100_000 } else { 0 },
                wall_ms: wall,
                peak_ram_bytes: ram,
                peak_vram_bytes: if gpu && numerical { 512 * MIB } else { 0 },
                updates,
            });
        }
    }
    for i in 0..c.participants.len() {
        let p = c.participants[i].clone();
        c.participants[i].grant = grant(&c, &p, &keys);
    }
    c
}
/// Plan `c`, write `census`, and run the operator's read-only preflight on them.
fn preflight(
    root: &Path,
    name: &str,
    c: &BoundedRunContract,
    census: &HostCensus,
    now: u64,
) -> Result<jope_parallel::native_loop::bounded::census::PreflightReceipt> {
    let dir = root.join(format!("run-{name}"));
    let plan = controller::plan(&dir, &bytes(c), now, None).unwrap();
    let path = root.join(format!("census-{name}.json"));
    collect::write_census(&path, census).unwrap();
    controller::preflight_dry_run(&dir, &plan.plan_sha256, &path, now)
}
/// A per-host document derived from one local collection with a synthetic identity.
fn synthetic(base: &HostCensus, i: usize, observed: u64) -> HostCensus {
    let mut c = base.clone();
    let h = &mut c.hosts[0];
    h.identity.host = format!("fixture-host-{i}");
    h.identity.machine_id = sha(format!("fixture-machine-{i}").as_bytes())[..32].to_string();
    h.identity.boot_id = format!("fixture-boot-{i}");
    h.identity.address = format!("192.0.2.{}", i + 1);
    h.identity.certificate_sha256 = sha(format!("fixture-certificate-{i}").as_bytes());
    h.observed_unix_ms = observed;
    c.captured_unix_ms = observed;
    c
}
fn pinned(label: &str, census: &HostCensus) -> PinnedDocument {
    let bytes = serde_json::to_vec(census).unwrap();
    PinnedDocument {
        label: label.into(),
        sha256: sha(&bytes),
        bytes,
    }
}
fn policy() -> MergePolicy {
    MergePolicy {
        max_age_ms: 60_000,
        clock_skew_ms: 2_000,
    }
}
fn err<T: std::fmt::Debug>(r: Result<T>) -> String {
    format!("{:#}", r.expect_err("expected a refusal"))
}

#[test]
fn collected_census_round_trips_through_preflight() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let nvidia = FakeNvidia::with(&one_gpu(), "");
    let (census, report) = collect::collect(&spec, &nvidia, now).unwrap();
    assert_eq!(nvidia.calls(), 2, "one GPU query and one process query");
    // The record is exactly the typed census the preflight validator reads.
    let bytes = serde_json::to_vec(&census).unwrap();
    assert_eq!(HostCensus::from_bytes(&bytes).unwrap(), census);
    assert_eq!(report.census_sha256, sha(&bytes));
    assert_eq!(report.gpu_probe, GpuProbe::NvidiaSmi);
    assert!(!report.allocation_held);
    let h = &census.hosts[0];
    assert_eq!(h.identity.machine_id, local("/etc/machine-id"));
    assert_eq!(h.identity.boot_id, local("/proc/sys/kernel/random/boot_id"));
    assert_eq!(h.identity.host, local("/proc/sys/kernel/hostname"));
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
    let (again, _) = collect::collect(&spec, &FakeNvidia::with(&one_gpu(), ""), now).unwrap();
    assert_eq!(again.hosts[0].identity, h.identity);

    let plan = plan_for(now, &[(h.identity.clone(), Some((UUID0, 0)))]);
    let receipt = preflight(&root, "gpu", &plan, &census, now).unwrap();
    assert!(!receipt.allocation_held);
    assert_eq!(receipt.census_sha256, report.census_sha256);
    assert_eq!(receipt.hosts[&h.identity.host].devices.len(), 1);
    // A changed installed file is a different executable identity: preflight refuses it.
    fs::write(&spec.executable, b"substituted worker").unwrap();
    let (substituted, _) = collect::collect(&spec, &FakeNvidia::with(&one_gpu(), ""), now).unwrap();
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
    let (hidden, report) = collect::collect(&cpu_only(spec.clone()), &never, now).unwrap();
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
    let (missing, report) = collect::collect(&spec, &absent, now).unwrap();
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
    assert!(err(collect::collect(&spec, &failing, now)).contains("GPU query failed"));
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
            &FakeNvidia::with(&gpus, &apps),
            now,
        ));
        assert!(refused.contains(needle), "{needle}: {refused}");
    }
    // A partial CUDA_VISIBLE_DEVICES would hide part of the host: refused before any probe.
    let mut partial = spec.clone();
    partial.cuda_visible_devices = Some(OsString::from("0"));
    let never = FakeNvidia::with(&one_gpu(), "");
    assert!(err(collect::collect(&partial, &never, now)).contains("host-wide"));
    assert_eq!(never.calls(), 0);
}

#[test]
fn device_processes_are_owners_that_preflight_refuses() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let spec = install(&root);
    let apps = format!("No running processes found\n{UUID0}, 4242, /usr/bin/fixture, with comma\n");
    let (census, report) =
        collect::collect(&spec, &FakeNvidia::with(&one_gpu(), &apps), now).unwrap();
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
    let cases: Vec<(&str, Box<dyn Fn(&mut CollectSpec)>)> = vec![
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
        let refused = err(collect::collect(&s, &never, now));
        assert!(refused.contains(needle), "{needle}: {refused}");
    }
    for garbage in [
        b"not a certificate".to_vec(),
        b"-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----\n".to_vec(),
        vec![0x02, 0x01, 0x00],
    ] {
        fs::write(&spec.certificate, garbage).unwrap();
        assert!(err(collect::collect(&cpu_only(spec.clone()), &never, now)).contains("certificate"));
    }
    assert_eq!(never.calls(), 0);
}

#[test]
fn merge_combines_pinned_documents_into_a_preflight_census() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let (base, _) =
        collect::collect(&cpu_only(install(&root)), &FakeNvidia::default(), now).unwrap();
    let docs: Vec<HostCensus> = [2, 0, 1]
        .into_iter()
        .map(|i| synthetic(&base, i, now - 1_000 * i as u64))
        .collect();
    let pins: Vec<PinnedDocument> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| pinned(&format!("doc-{i}"), d))
        .collect();
    let (merged, report) = collect::merge(&pins, policy(), now).unwrap();
    merged.validate().unwrap();
    assert_eq!(
        merged
            .hosts
            .iter()
            .map(|h| h.identity.host.as_str())
            .collect::<Vec<_>>(),
        ["fixture-host-0", "fixture-host-1", "fixture-host-2"]
    );
    assert_eq!(merged.captured_unix_ms, now);
    assert_eq!(
        report.census_sha256,
        sha(&serde_json::to_vec(&merged).unwrap())
    );
    assert!(!report.allocation_held);
    let expected: std::collections::BTreeMap<String, String> = docs
        .iter()
        .zip(&pins)
        .map(|(d, p)| (d.hosts[0].identity.host.clone(), p.sha256.clone()))
        .collect();
    assert_eq!(report.documents.len(), 3);
    for d in &report.documents {
        assert_eq!(expected[&d.host], d.sha256);
    }
    let hosts: Vec<_> = merged
        .hosts
        .iter()
        .map(|h| (h.identity.clone(), None))
        .collect();
    let receipt = preflight(&root, "merged", &plan_for(now, &hosts), &merged, now).unwrap();
    assert_eq!(receipt.hosts.len(), 3);
    assert_eq!(receipt.census_sha256, report.census_sha256);
}

#[test]
fn merge_refuses_tampered_stale_future_duplicate_and_mixed_documents() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let (base, _) =
        collect::collect(&cpu_only(install(&root)), &FakeNvidia::default(), now).unwrap();
    let good = |i| pinned(&format!("doc-{i}"), &synthetic(&base, i, now));
    let refuse = |docs: Vec<PinnedDocument>, needle: &str| {
        let refused = err(collect::merge(&docs, policy(), now));
        assert!(refused.contains(needle), "{needle}: {refused}");
    };
    // Tampered in transit: the bytes no longer match the collector's printed digest.
    let mut tampered = good(1);
    tampered.bytes = String::from_utf8(tampered.bytes)
        .unwrap()
        .replace("fixture-host-1", "fixture-host-9")
        .into_bytes();
    refuse(vec![good(0), tampered], "was altered");
    let mut bad_pin = good(1);
    bad_pin.sha256 = "0".repeat(63);
    refuse(vec![good(0), bad_pin], "64 lowercase hex");
    // Stale and future telemetry.
    refuse(
        vec![good(0), pinned("stale", &synthetic(&base, 1, now - 60_001))],
        "stale",
    );
    refuse(
        vec![good(0), pinned("future", &synthetic(&base, 1, now + 2_001))],
        "future",
    );
    // Duplicates: the same document twice, or two documents sharing any identity.
    refuse(vec![good(0), good(0)], "given twice");
    for (what, edit) in [
        (
            "host",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.host = "fixture-host-0".into())
                as Box<dyn Fn(&mut HostCensus)>,
        ),
        (
            "machine id",
            Box::new(|c: &mut HostCensus| {
                c.hosts[0].identity.machine_id = synthetic(&base, 0, now).hosts[0]
                    .identity
                    .machine_id
                    .clone()
            }),
        ),
        (
            "boot id",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.boot_id = "fixture-boot-0".into()),
        ),
        (
            "address",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.address = "192.0.2.1".into()),
        ),
        (
            "certificate",
            Box::new(|c: &mut HostCensus| {
                c.hosts[0].identity.certificate_sha256 = sha(b"fixture-certificate-0")
            }),
        ),
    ] {
        let mut twin = synthetic(&base, 1, now);
        edit(&mut twin);
        refuse(
            vec![good(0), pinned("twin", &twin)],
            &format!("duplicates the {what}"),
        );
    }
    // Mixed or unsupported schema versions.
    let mut old = synthetic(&base, 1, now);
    old.schema = "jope.native_host_census.v0".into();
    refuse(vec![good(0), pinned("old", &old)], "mixed or unsupported");
    // A document is one host's observation.
    let mut two = synthetic(&base, 1, now);
    two.hosts.push(synthetic(&base, 2, now).hosts[0].clone());
    refuse(vec![good(0), pinned("two", &two)], "exactly one host");
    // Freshness policy bounds.
    for bad in [
        MergePolicy {
            max_age_ms: 0,
            clock_skew_ms: 0,
        },
        MergePolicy {
            max_age_ms: 60_000,
            clock_skew_ms: 10_001,
        },
    ] {
        assert!(collect::merge(&[good(0)], bad, now).is_err());
    }
    refuse(Vec::new(), "1..=64");
}

fn run(prefix: &[&str], args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .args(prefix)
        .args(args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}
/// `tree` without the root directory's own entry (its mtime changes when a file is added).
fn contents(root: &Path) -> std::collections::BTreeMap<PathBuf, String> {
    let mut t = tree(root);
    t.remove(Path::new(""));
    t
}
fn collect_argv(spec: &CollectSpec, out: &Path) -> Vec<String> {
    let s = |p: &Path| p.display().to_string();
    vec![
        "census".into(),
        "collect".into(),
        "--out".into(),
        s(out),
        "--address".into(),
        spec.address.clone(),
        "--certificate".into(),
        s(&spec.certificate),
        "--source-commit".into(),
        spec.source_commit.clone(),
        "--executable".into(),
        s(&spec.executable),
        "--runtime-library".into(),
        s(&spec.runtime_libraries[0]),
        "--compiler".into(),
        s(&spec.compiler),
        "--installed-policy".into(),
        s(&spec.installed_policies[0]),
    ]
}

#[test]
fn cli_collect_merge_and_preflight_round_trip_without_side_effects() {
    let (_t, root) = tempdir();
    let spec = install(&root);
    let out = root.join("host.census.json");
    let argv = collect_argv(&spec, &out);
    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    let before = contents(&root);
    let o = run(&[], &argv);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let written = fs::read(&out).unwrap();
    assert_eq!(report["census_sha256"], sha(&written));
    assert_eq!(
        report["gpu_probe"],
        "not_queried_cuda_visible_devices_empty"
    );
    assert_eq!(report["allocation_held"], false);
    // The collector wrote its one output and nothing else.
    let mut after = contents(&root);
    assert!(after.remove(Path::new("host.census.json")).is_some());
    assert_eq!(after, before);
    // A census is never replaced.
    let before = contents(&root);
    let o = run(&[], &argv);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("already exists"));
    assert_eq!(contents(&root), before);
    // Legacy loop flags are refused for census operations.
    let o = run(&["--tenant", "t"], &argv);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("no fixed-loop arguments"));

    let base = HostCensus::from_bytes(&written).unwrap();
    let now = base.captured_unix_ms;
    let mut documents = Vec::new();
    for i in 0..3 {
        let doc = if i == 0 {
            written.clone()
        } else {
            serde_json::to_vec(&synthetic(&base, i, now)).unwrap()
        };
        let path = root.join(format!("doc-{i}.json"));
        fs::write(&path, &doc).unwrap();
        documents.push(format!("{}={}", path.display(), sha(&doc)));
    }
    let merged_path = root.join("merged.json");
    let mut merge = vec![
        "census".to_string(),
        "merge".into(),
        "--out".into(),
        merged_path.display().to_string(),
        "--max-age-ms".into(),
        "600000".into(),
        "--clock-skew-ms".into(),
        "2000".into(),
    ];
    // A wrong pin refuses the merge and writes nothing.
    let mut wrong = merge.clone();
    for (i, d) in documents.iter().enumerate() {
        wrong.push("--document".into());
        wrong.push(if i == 1 {
            format!("{}={}", d.rsplit_once('=').unwrap().0, "0".repeat(64))
        } else {
            d.clone()
        });
    }
    let o = run(&wrong.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("was altered"));
    assert!(!merged_path.exists());
    for d in &documents {
        merge.push("--document".into());
        merge.push(d.clone());
    }
    let o = run(&merge.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let merged_bytes = fs::read(&merged_path).unwrap();
    assert_eq!(report["census_sha256"], sha(&merged_bytes));
    assert_eq!(report["documents"].as_array().unwrap().len(), 3);
    let merged = HostCensus::from_bytes(&merged_bytes).unwrap();

    // Plan from the merged identities and preflight through the operator CLI.
    let hosts: Vec<_> = merged
        .hosts
        .iter()
        .map(|h| (h.identity.clone(), None))
        .collect();
    let plan_now = now_ms();
    let c = plan_for(plan_now, &hosts);
    let dir = root.join("run");
    let plan = controller::plan(&dir, &bytes(&c), plan_now, None).unwrap();
    let o = run(
        &[
            "preflight",
            "--run-dir",
            dir.to_str().unwrap(),
            "--plan-sha256",
            &plan.plan_sha256,
            "--census",
            merged_path.to_str().unwrap(),
        ],
        &[],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let receipt: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(receipt["allocation_held"], false);
    assert_eq!(receipt["census_sha256"], sha(&merged_bytes));
    assert_eq!(receipt["hosts"].as_object().unwrap().len(), 3);
}
