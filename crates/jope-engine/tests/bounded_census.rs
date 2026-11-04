//! W7-T01 census collector and merge fixtures.
//!
//! The collector observes the checked-in fixture host (`support/fixture_host.rs`), not the
//! machine the tests run on: its machine id, boot id and host name are files read by the same
//! code that reads a real host's, and its one interface address is loopback. The command-line
//! round trip runs the production command line through the test-only fixture executable, which
//! observes that same fixture host. GPU inventory always comes from an injected command runner
//! (or is hidden with `CUDA_VISIBLE_DEVICES=""`): no fixture runs `nvidia-smi` or touches a GPU.
//! Multi-host documents are derived from one local collection with synthetic identities, so they
//! are fixtures, not a live three-host census.
#[path = "support/bounded_fixture.rs"]
mod fixture;
#[path = "support/fixture_host.rs"]
mod fixture_host;
use anyhow::Result;
use fixture::*;
use fixture_host::with_commands;
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
/// One of the fixture host's checked-in identity files, read independently of the collector.
fn fixture_file(name: &str) -> String {
    fs::read_to_string(fixture_host::identity_dir().join(name))
        .unwrap()
        .trim()
        .to_string()
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

#[path = "bounded_census/cli.rs"]
mod cli;
#[path = "bounded_census/collection.rs"]
mod collection;
#[path = "bounded_census/merge.rs"]
mod merge;
