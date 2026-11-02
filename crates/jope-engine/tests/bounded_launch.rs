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

mod admission_tests;
mod gpu_tests;
mod launch_tests;
