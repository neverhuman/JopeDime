//! Frozen worker launch (W7-T03 delivery half).
//!
//! * [`LaunchRecord`] (`jope.native_bounded_launch.v1`) is what a supervisor freezes for one
//!   rank before it starts that rank's worker: the plan digest, run, attempt, rank, world,
//!   dispatch id, the planned host identity and the exact device visibility the plan implies.
//!   A worker takes its identity from this record and the frozen plan artifact, never from
//!   mutable flags, and refuses any record that is not exactly the one the plan derives.
//! * [`worker_command`] is the launcher: it clears the environment and sets only
//!   `CUDA_VISIBLE_DEVICES`, to the planned GPU UUID (empty for a CPU rank), before exec.
//! * [`verify_worker_visibility`] is the worker's own check of that environment, before any
//!   input, device or communicator work.
//!
//! Nothing here starts work: a launch record authorizes nothing by itself. A rank still
//! joins the authenticated rendezvous and holds a verified full-group admission before it
//! starts (W7-T04).
//!
//! * [`nccl_bootstrap`] takes the NCCL unique id only from a verified full-group admission of
//!   this plan, attempt and rank (W7-T04). Bounded plans never read a shared `nccl.id` file;
//!   that path keeps its meaning only for the historical same-allocation launches.
use super::{
    contract::BoundedRunContract,
    rendezvous::{self, AdmittedMember},
};
use anyhow::{anyhow, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    process::Command,
};

pub const LAUNCH_SCHEMA: &str = "jope.native_bounded_launch.v1";
/// The only variable a launcher sets from the plan; every other variable is cleared.
pub const VISIBILITY_VARIABLE: &str = "CUDA_VISIBLE_DEVICES";
/// `NCCL_UNIQUE_ID_BYTES`: the only communicator bootstrap a bounded trainer accepts.
pub const NCCL_UNIQUE_ID_BYTES: usize = 128;
const MAX_LAUNCH_RECORD_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaunchRecord {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub global_rank: u32,
    pub world_size: u32,
    pub dispatch_id: String,
    pub host: String,
    pub machine_id: String,
    pub boot_id: String,
    /// Exactly the planned GPU UUID, or empty for a CPU rank (every GPU hidden).
    pub cuda_visible_devices: String,
}

fn bound_plan(contract: &BoundedRunContract, plan_sha256: &str) -> Result<()> {
    contract.validate_structure().map_err(|e| anyhow!("{e}"))?;
    ensure!(
        crate::manifest::hash(&contract.canonical_bytes()?) == plan_sha256,
        "plan digest does not name this frozen plan"
    );
    Ok(())
}

/// The visibility the plan implies for one rank: its GPU UUID, or empty for a CPU rank.
pub fn planned_visibility(contract: &BoundedRunContract, global_rank: u32) -> Result<String> {
    let p = contract
        .participant(global_rank)
        .with_context(|| format!("rank {global_rank} is not in the plan"))?;
    Ok(p.device
        .as_ref()
        .map(|d| d.uuid.clone())
        .unwrap_or_default())
}

/// The launch record the plan derives for `global_rank`.
pub fn launch_record(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    global_rank: u32,
) -> Result<LaunchRecord> {
    bound_plan(contract, plan_sha256)?;
    let p = contract
        .participant(global_rank)
        .with_context(|| format!("rank {global_rank} is not in the plan"))?;
    Ok(LaunchRecord {
        schema: LAUNCH_SCHEMA.into(),
        run: contract.run.clone(),
        attempt: contract.attempt,
        plan_sha256: plan_sha256.into(),
        global_rank,
        world_size: contract.world_size() as u32,
        dispatch_id: rendezvous::dispatch_id(plan_sha256, contract.attempt, global_rank)?,
        host: p.host.host.clone(),
        machine_id: p.host.machine_id.clone(),
        boot_id: p.host.boot_id.clone(),
        cuda_visible_devices: planned_visibility(contract, global_rank)?,
    })
}

/// Refuse any launch record that is not exactly the one the plan derives.
pub fn verify_launch_record(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    record: &LaunchRecord,
) -> Result<()> {
    let expected = launch_record(contract, plan_sha256, record.global_rank)?;
    ensure!(
        *record == expected,
        "launch record for rank {} differs from the plan-derived launch",
        record.global_rank
    );
    Ok(())
}

pub(crate) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    ensure!(
        f.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(b)
}

/// The frozen plan artifact at `plan`, bound to its reviewed digest: its canonical bytes and
/// the contract they parse to.
pub fn load_plan(plan: &Path, plan_sha256: &str) -> Result<(BoundedRunContract, Vec<u8>)> {
    let bytes = read_bounded(plan, super::contract::MAX_CONTRACT_BYTES as u64)?;
    ensure!(
        crate::manifest::hash(&bytes) == plan_sha256,
        "frozen plan bytes differ from the admitted plan digest"
    );
    let contract = BoundedRunContract::parse(&bytes).map_err(|e| anyhow!("{e}"))?;
    ensure!(
        contract.canonical_bytes()? == bytes,
        "frozen plan bytes are not the canonical contract"
    );
    Ok((contract, bytes))
}

/// A worker's view of its launch: the exact frozen plan bytes (by digest) and its launch
/// record, both verified. The worker's rank, world and devices come only from these.
pub fn load_launch(
    plan: &Path,
    plan_sha256: &str,
    record: &Path,
) -> Result<(BoundedRunContract, LaunchRecord)> {
    let (contract, _) = load_plan(plan, plan_sha256)?;
    let record: LaunchRecord =
        serde_json::from_slice(&read_bounded(record, MAX_LAUNCH_RECORD_BYTES)?)?;
    verify_launch_record(&contract, plan_sha256, &record)?;
    Ok((contract, record))
}

/// The exact environment of a worker process: `CUDA_VISIBLE_DEVICES` and nothing else.
pub fn worker_environment(
    contract: &BoundedRunContract,
    global_rank: u32,
) -> Result<BTreeMap<String, String>> {
    Ok(BTreeMap::from([(
        VISIBILITY_VARIABLE.to_string(),
        planned_visibility(contract, global_rank)?,
    )]))
}

/// The launcher: `program` with a cleared environment holding only the frozen visibility.
/// Arguments are the caller's; the plan never becomes argv.
pub fn worker_command(
    program: &Path,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    global_rank: u32,
) -> Result<Command> {
    bound_plan(contract, plan_sha256)?;
    let mut command = Command::new(program);
    command
        .env_clear()
        .envs(worker_environment(contract, global_rank)?);
    Ok(command)
}

/// The value SmartCluster's executor sets for a job without a GPU allocation; like the empty
/// string, it hides every device.
pub const HIDE_ALL_DEVICES: &str = "-1";

/// The worker's own check, before any input, device or communicator work: its
/// `CUDA_VISIBLE_DEVICES` must be exactly the frozen visibility. Refuses an unset variable,
/// a device index instead of the planned UUID, a duplicated or additional device, another
/// rank's device, and any GPU for a CPU rank. A CPU rank accepts exactly the two values that
/// hide every device: the empty string (this launcher) and `-1` (SmartCluster's executor).
pub fn verify_worker_visibility(record: &LaunchRecord, observed: Option<&OsStr>) -> Result<()> {
    let observed = observed
        .context("CUDA_VISIBLE_DEVICES is unset; a bounded worker sees only its planned device")?
        .to_str()
        .context("CUDA_VISIBLE_DEVICES is not UTF-8")?;
    if record.cuda_visible_devices.is_empty() {
        ensure!(
            observed.is_empty() || observed == HIDE_ALL_DEVICES,
            "CUDA_VISIBLE_DEVICES `{observed}` exposes a device to CPU rank {}",
            record.global_rank
        );
        return Ok(());
    }
    let devices: Vec<&str> = if observed.is_empty() {
        Vec::new()
    } else {
        observed.split(',').map(str::trim).collect()
    };
    ensure!(
        devices.iter().collect::<BTreeSet<_>>().len() == devices.len(),
        "CUDA_VISIBLE_DEVICES `{observed}` names a device twice"
    );
    ensure!(
        devices.iter().all(|d| d.starts_with("GPU-")),
        "CUDA_VISIBLE_DEVICES `{observed}` selects a device by local index, not by its planned UUID"
    );
    ensure!(
        observed == record.cuda_visible_devices,
        "CUDA_VISIBLE_DEVICES `{observed}` differs from the planned `{}` for rank {}",
        record.cuda_visible_devices,
        record.global_rank
    );
    Ok(())
}

/// The communicator bootstrap of a verified full-group admission, bound to one plan,
/// attempt, rendezvous and rank. Only [`nccl_bootstrap`] constructs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedBootstrap {
    plan_sha256: String,
    attempt: u32,
    rendezvous_id: String,
    admission_sha256: String,
    global_rank: u32,
    world_size: u32,
    id: [u8; NCCL_UNIQUE_ID_BYTES],
}
impl AdmittedBootstrap {
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
    pub fn rendezvous_id(&self) -> &str {
        &self.rendezvous_id
    }
    pub fn admission_sha256(&self) -> &str {
        &self.admission_sha256
    }
    pub fn global_rank(&self) -> u32 {
        self.global_rank
    }
    pub fn world_size(&self) -> u32 {
        self.world_size
    }
    pub fn id(&self) -> &[u8; NCCL_UNIQUE_ID_BYTES] {
        &self.id
    }
}

/// The NCCL unique id this launch's rank may use: only from a full-group admission that the
/// planned coordinator signed for exactly this plan and attempt, held as this rank's
/// membership. Refuses a forged or foreign admission, another attempt's or plan's admission,
/// another rank's membership and any bootstrap that is not exactly an NCCL unique id.
/// (A stale or replayed admission never becomes an [`AdmittedMember`]: the participant's
/// [`rendezvous::verify_admission`] binds it to the rank's own hello and rendezvous.)
pub fn nccl_bootstrap(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    record: &LaunchRecord,
    member: &AdmittedMember,
) -> Result<AdmittedBootstrap> {
    verify_launch_record(contract, plan_sha256, record)?;
    let admission = member.admission();
    let body = &admission.body;
    ensure!(
        body.run == contract.run && body.attempt == contract.attempt,
        "admission is for run `{}` attempt {}, not this launch's attempt {}",
        body.run,
        body.attempt,
        contract.attempt
    );
    ensure!(
        body.plan_sha256 == plan_sha256,
        "admission is for another plan"
    );
    // The member may have been verified against another plan: re-check the coordinator
    // signature and the full-group binding against this one.
    rendezvous::verify_group_admission(contract, plan_sha256, admission)?;
    ensure!(
        member.admission_sha256() == rendezvous::admission_sha256(admission)?,
        "admission digest differs from the admitted member's"
    );
    ensure!(
        member.global_rank() == record.global_rank && member.world_size() == record.world_size,
        "admitted membership is rank {} of {}, not this launch's rank {} of {}",
        member.global_rank(),
        member.world_size(),
        record.global_rank,
        record.world_size
    );
    let bytes = member.communicator_bootstrap();
    let id: [u8; NCCL_UNIQUE_ID_BYTES] = bytes.try_into().map_err(|_| {
        anyhow!(
            "admitted communicator bootstrap is {} bytes; an NCCL unique id is {NCCL_UNIQUE_ID_BYTES}",
            bytes.len()
        )
    })?;
    Ok(AdmittedBootstrap {
        plan_sha256: plan_sha256.into(),
        attempt: contract.attempt,
        rendezvous_id: body.rendezvous_id.clone(),
        admission_sha256: member.admission_sha256().into(),
        global_rank: record.global_rank,
        world_size: record.world_size,
        id,
    })
}
