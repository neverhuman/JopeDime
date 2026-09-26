//! Shared W7 bounded-run fixtures. Every identity here is synthetic: fixture host names,
//! boot ids, GPU UUIDs and digests are generated, not observed. Nothing contacts a host,
//! reads GPU state or starts numerical work.
#![allow(dead_code)]
use ed25519_dalek::SigningKey;
use jope_parallel::native_loop::bounded::{
    census::{
        GpuObservation, HostCensus, HostIdentity, HostObservation, OwnerKind, Ownership,
        RuntimeIdentity,
    },
    contract::{
        sign_grant, BoundedRunContract, Ceilings, ClosureEntry, Counters, DeviceAssignment,
        GrantBody, Participant, RendezvousPolicy, ResultDestination, TelemetryPolicy, WorkItem,
        WorkKind,
    },
    controller::{Liveness, RankLaunch, Supervisor, SupervisorHandle},
    rendezvous::{self, Challenge, HelloBody, ResultBody},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

pub const GIB: u64 = 1024 * 1024 * 1024;

pub fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
/// Canonical physical temporary directory (TMPDIR is outside /tmp for this lane).
pub fn tempdir() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let p = fs::canonicalize(t.path()).unwrap();
    (t, p)
}

pub struct Keys {
    pub coordinator: SigningKey,
    pub authority: SigningKey,
    pub ranks: Vec<SigningKey>,
}
pub fn keys(n: usize) -> Keys {
    Keys {
        coordinator: SigningKey::from_bytes(&[7; 32]),
        authority: SigningKey::from_bytes(&[9; 32]),
        ranks: (0..n)
            .map(|i| SigningKey::from_bytes(&[40 + i as u8; 32]))
            .collect(),
    }
}
pub fn rank_seed(i: usize) -> [u8; 32] {
    [40 + i as u8; 32]
}
pub fn public(k: &SigningKey) -> String {
    hex::encode(k.verifying_key().to_bytes())
}
pub fn identity(i: usize) -> HostIdentity {
    let tag = |s: &str| sha(format!("fixture-{s}-{i}").as_bytes());
    HostIdentity {
        host: format!("fixture-host-{i}"),
        boot_id: format!("fixture-boot-{i}"),
        address: format!("fixture-host-{i}.invalid"),
        certificate_sha256: tag("certificate"),
        source_commit: sha(b"fixture-source")[..40].to_string(),
        executable_sha256: sha(b"fixture-executable"),
        installed_policy_sha256: sha(b"fixture-policy"),
        runtime: RuntimeIdentity {
            runtime_sha256: sha(b"fixture-runtime"),
            compiler_sha256: sha(b"fixture-compiler"),
            cuda_sha256: Some(sha(b"fixture-cuda")),
            nccl_sha256: Some(sha(b"fixture-nccl")),
            driver_version: Some("fixture-driver".into()),
        },
    }
}
pub fn closure_files() -> Vec<(PathBuf, Vec<u8>)> {
    vec![
        (
            PathBuf::from("inputs/corpus.bin"),
            b"fixture corpus bytes".to_vec(),
        ),
        (
            PathBuf::from("inputs/program.json"),
            br#"{"fixture":true}"#.to_vec(),
        ),
    ]
}
pub fn write_closure(root: &Path) {
    for (p, b) in closure_files() {
        let path = root.join(p);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b).unwrap();
    }
}
pub fn grant(
    c: &BoundedRunContract,
    p: &Participant,
    keys: &Keys,
) -> jope_parallel::native_loop::bounded::contract::SignedGrant {
    sign_grant(
        &keys.authority,
        GrantBody {
            schema: "jope.native_participant_grant.v1".into(),
            run: c.run.clone(),
            attempt: c.attempt,
            global_rank: p.global_rank,
            host: p.host.host.clone(),
            boot_id: p.host.boot_id.clone(),
            device_uuid: p.device.as_ref().map(|d| d.uuid.clone()),
            participant_public_key: p.public_key.clone(),
            cpu_millicores: p.ceilings.cpu_millicores,
            ram_bytes: p.ceilings.ram_bytes,
            gpu_vram_bytes: p.ceilings.gpu_vram_bytes,
            wall_ms: p.ceilings.wall_ms,
            not_before_unix_ms: c.not_before_unix_ms,
            expires_unix_ms: c.not_before_unix_ms + 900_000,
        },
    )
    .unwrap()
}
/// Re-sign every grant from its (possibly edited) participant fields.
pub fn resign(c: &mut BoundedRunContract, keys: &Keys) {
    for i in 0..c.participants.len() {
        let p = c.participants[i].clone();
        let mut g = grant(c, &p, keys);
        g.body.expires_unix_ms = p.grant.body.expires_unix_ms;
        g.body.not_before_unix_ms = p.grant.body.not_before_unix_ms;
        g.body.cpu_millicores = p.grant.body.cpu_millicores;
        g.body.ram_bytes = p.grant.body.ram_bytes;
        g.body.gpu_vram_bytes = p.grant.body.gpu_vram_bytes;
        g.body.wall_ms = p.grant.body.wall_ms;
        c.participants[i].grant = sign_grant(&keys.authority, g.body).unwrap();
    }
}
/// A funded three-participant, one-GPU-per-host contract shaped like the W7 topology.
pub fn contract(now: u64, n: usize, keys: &Keys) -> BoundedRunContract {
    let not_before = now - 1_000;
    let mut c = BoundedRunContract {
        schema: "jope.native_bounded_run.v1".into(),
        run: "w7-fixture-run".into(),
        attempt: 1,
        coordinator_host: "fixture-host-1".into(),
        coordinator_public_key: public(&keys.coordinator),
        grant_authority_public_key: public(&keys.authority),
        not_before_unix_ms: not_before,
        original_deadline_unix_ms: not_before + 1_200_000,
        telemetry: TelemetryPolicy {
            max_age_ms: 60_000,
            clock_skew_ms: 2_000,
        },
        rendezvous: RendezvousPolicy {
            join_timeout_ms: 30_000,
            max_message_bytes: 64 * 1024,
            heartbeat_timeout_ms: 3_000,
        },
        artifact_closure: closure_files()
            .into_iter()
            .map(|(path, b)| ClosureEntry {
                path,
                sha256: sha(&b),
                bytes: b.len() as u64,
            })
            .collect(),
        participants: Vec::new(),
        work_census: Vec::new(),
        counters: Counters {
            committed_updates_at_start: 60_000,
            planned_updates: 3,
        },
        predecessor: None,
    };
    for i in 0..n {
        let mut p = Participant {
            global_rank: i as u32,
            local_rank: 0,
            host: identity(i),
            device: Some(DeviceAssignment {
                uuid: format!("GPU-fixture-{i}"),
                local_index: 0,
            }),
            ceilings: Ceilings {
                cpu_millicores: 4_000,
                ram_bytes: 8 * GIB,
                gpu_vram_bytes: Some(20 * GIB),
                wall_ms: 600_000,
            },
            public_key: public(&keys.ranks[i]),
            grant: jope_parallel::native_loop::bounded::contract::SignedGrant {
                body: GrantBody {
                    schema: String::new(),
                    run: String::new(),
                    attempt: 0,
                    global_rank: 0,
                    host: String::new(),
                    boot_id: String::new(),
                    device_uuid: None,
                    participant_public_key: String::new(),
                    cpu_millicores: 0,
                    ram_bytes: 0,
                    gpu_vram_bytes: None,
                    wall_ms: 0,
                    not_before_unix_ms: 0,
                    expires_unix_ms: 0,
                },
                signature: String::new(),
            },
            result_destination: ResultDestination {
                path: PathBuf::from(format!("rank-{i}/result.json")),
                max_bytes: 1024 * 1024,
            },
        };
        p.grant = grant(&c, &p, keys);
        c.participants.push(p);
        for (kind, cpu, gpu, wall, ram, vram, updates) in [
            (
                WorkKind::Numerical,
                600_000,
                300_000,
                400_000,
                4 * GIB,
                16 * GIB,
                3,
            ),
            (WorkKind::Verification, 10_000, 0, 30_000, GIB, 0, 0),
            (WorkKind::ArtifactMovement, 5_000, 0, 20_000, GIB, 0, 0),
            (WorkKind::Cleanup, 1_000, 0, 10_000, GIB, 0, 0),
            (WorkKind::FailureAllowance, 20_000, 0, 60_000, GIB, 0, 0),
        ] {
            c.work_census.push(WorkItem {
                id: format!(
                    "rank-{i}-{}",
                    serde_json::to_value(kind).unwrap().as_str().unwrap()
                ),
                global_rank: i as u32,
                kind,
                cpu_core_ms: cpu,
                gpu_ms: gpu,
                wall_ms: wall,
                peak_ram_bytes: ram,
                peak_vram_bytes: vram,
                updates,
            });
        }
    }
    c
}
pub fn bytes(c: &BoundedRunContract) -> Vec<u8> {
    serde_json::to_vec(c).unwrap()
}
/// A census that matches the plan, observed at `now`, with an unrelated CI reservation.
pub fn census(c: &BoundedRunContract, now: u64) -> HostCensus {
    HostCensus {
        schema: "jope.native_host_census.v1".into(),
        captured_unix_ms: now,
        hosts: c
            .participants
            .iter()
            .map(|p| HostObservation {
                identity: p.host.clone(),
                observed_unix_ms: now,
                cpu_millicores_allocatable: 32_000,
                ram_bytes_allocatable: 64 * GIB,
                gpus: vec![GpuObservation {
                    uuid: p.device.as_ref().unwrap().uuid.clone(),
                    model: "fixture-gpu".into(),
                    local_index: p.device.as_ref().unwrap().local_index,
                    total_vram_bytes: 24 * GIB,
                    available_vram_bytes: 23 * GIB,
                }],
                ownership: vec![Ownership {
                    owner: "ci-reservation".into(),
                    kind: OwnerKind::Ci,
                    device_uuid: None,
                    cpu_millicores: 8_000,
                    ram_bytes: 16 * GIB,
                    run: None,
                    attempt: None,
                }],
            })
            .collect(),
    }
}
/// The hello an honest participant whose observed host matches the plan would send.
pub fn hello_body(
    c: &BoundedRunContract,
    plan_sha256: &str,
    challenge: &Challenge,
    rank: u32,
    sent: u64,
    closure_sha256: &str,
) -> HelloBody {
    let p = &c.participants[rank as usize];
    HelloBody {
        schema: "jope.native_rendezvous_hello.v1".into(),
        run: c.run.clone(),
        attempt: c.attempt,
        plan_sha256: plan_sha256.into(),
        rendezvous_id: challenge.body.rendezvous_id.clone(),
        global_rank: rank,
        delivery_id: 1,
        sent_unix_ms: sent,
        host: p.host.host.clone(),
        boot_id: p.host.boot_id.clone(),
        device_uuid: p.device.as_ref().map(|d| d.uuid.clone()),
        source_commit: p.host.source_commit.clone(),
        executable_sha256: p.host.executable_sha256.clone(),
        runtime: p.host.runtime.clone(),
        installed_policy_sha256: p.host.installed_policy_sha256.clone(),
        artifact_closure_sha256: closure_sha256.into(),
        grant_sha256: sha(&serde_json::to_vec(&p.grant).unwrap()),
        communicator_bootstrap: (rank == 0).then(|| hex::encode([0xab; 128])),
    }
}
pub fn hello_frame(keys: &Keys, body: HelloBody) -> Vec<u8> {
    let rank = body.global_rank as usize;
    serde_json::to_vec(&rendezvous::sign_hello(&keys.ranks[rank], body).unwrap()).unwrap()
}
/// A signed result for `rank` bound to `admission_sha256`.
pub fn result(
    c: &BoundedRunContract,
    plan_sha256: &str,
    admission_sha256: &str,
    rank: u32,
    key: &SigningKey,
) -> (Vec<u8>, Vec<u8>) {
    let payload =
        serde_json::to_vec(&serde_json::json!({"fixture_rank": rank, "numerical": false})).unwrap();
    let envelope = rendezvous::sign_result(
        key,
        ResultBody {
            schema: "jope.native_rank_result.v1".into(),
            run: c.run.clone(),
            attempt: c.attempt,
            plan_sha256: plan_sha256.into(),
            admission_sha256: admission_sha256.into(),
            global_rank: rank,
            delivery_id: 1,
            result_sha256: sha(&payload),
            result_bytes: payload.len() as u64,
            updates: c.counters.planned_updates,
            completed_unix_ms: now_ms(),
        },
    )
    .unwrap();
    (serde_json::to_vec(&envelope).unwrap(), payload)
}

/// Deterministic in-memory supervisor for state-machine tests. The shared state lets the
/// test change liveness and retained results after the coordinator owns the supervisor.
#[derive(Default)]
pub struct FakeState {
    pub work: BTreeMap<String, (SupervisorHandle, Liveness, Option<(Vec<u8>, Vec<u8>)>)>,
    pub launches: u32,
    pub terminations: u32,
}
#[derive(Clone, Default)]
pub struct FakeSupervisor(pub Arc<Mutex<FakeState>>);
impl FakeSupervisor {
    pub fn set(&self, dispatch_id: &str, l: Liveness, retained: Option<(Vec<u8>, Vec<u8>)>) {
        let mut s = self.0.lock().unwrap();
        let w = s.work.get_mut(dispatch_id).expect("launched work");
        w.1 = l;
        w.2 = retained;
    }
    /// Work started out of band (a crash between spawn and the launch record).
    pub fn inject(&self, dispatch_id: &str) {
        let mut s = self.0.lock().unwrap();
        s.work.insert(
            dispatch_id.into(),
            (
                SupervisorHandle {
                    supervisor: "fake".into(),
                    identity: dispatch_id.into(),
                },
                Liveness::Running,
                None,
            ),
        );
    }
    pub fn running(&self) -> usize {
        self.0
            .lock()
            .unwrap()
            .work
            .values()
            .filter(|w| w.1 == Liveness::Running)
            .count()
    }
}
impl Supervisor for FakeSupervisor {
    fn launch(&mut self, l: &RankLaunch<'_>) -> anyhow::Result<SupervisorHandle> {
        let mut s = self.0.lock().unwrap();
        anyhow::ensure!(!s.work.contains_key(l.dispatch_id), "duplicate launch");
        s.launches += 1;
        let h = SupervisorHandle {
            supervisor: "fake".into(),
            identity: l.dispatch_id.into(),
        };
        s.work
            .insert(l.dispatch_id.into(), (h.clone(), Liveness::Running, None));
        Ok(h)
    }
    fn find(&mut self, dispatch_id: &str) -> anyhow::Result<Option<SupervisorHandle>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(dispatch_id)
            .map(|w| w.0.clone()))
    }
    fn probe(&mut self, h: &SupervisorHandle) -> anyhow::Result<Liveness> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(&h.identity)
            .map(|w| w.1.clone())
            .unwrap_or(Liveness::Gone))
    }
    fn terminate(&mut self, h: &SupervisorHandle) -> anyhow::Result<Liveness> {
        let mut s = self.0.lock().unwrap();
        s.terminations += 1;
        let w = s.work.get_mut(&h.identity).expect("work");
        if w.1 == Liveness::Running {
            w.1 = Liveness::Exited {
                code: None,
                signal: Some(15),
            };
        }
        Ok(w.1.clone())
    }
    fn retained_result(
        &mut self,
        h: &SupervisorHandle,
    ) -> anyhow::Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(&h.identity)
            .and_then(|w| w.2.clone()))
    }
}

/// Snapshot of every path, type, length, mode, mtime and content digest under `root`.
pub fn tree(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    for e in walkdir::WalkDir::new(root).sort_by_file_name() {
        let e = e.unwrap();
        let m = fs::symlink_metadata(e.path()).unwrap();
        use std::os::unix::fs::MetadataExt;
        let content = if m.is_file() {
            sha(&fs::read(e.path()).unwrap())
        } else {
            String::new()
        };
        out.insert(
            e.path().strip_prefix(root).unwrap().to_path_buf(),
            format!(
                "{:?}:{}:{:o}:{}.{}:{content}",
                m.file_type(),
                m.len(),
                m.mode(),
                m.mtime(),
                m.mtime_nsec()
            ),
        );
    }
    out
}
