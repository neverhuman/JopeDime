//! Join of the W7-T02 bounded-run plan with the W7-T03 trainer assignment.
//!
//! [`trainer_assignment`] is a pure function of the frozen contract: it reads no clock,
//! file, environment or device. The plan digest is the SHA-256 of the contract's canonical
//! bytes, which is exactly the digest of the controller's `plan.json`.
//!
//! **Visibility contract.** Each worker is launched with `CUDA_VISIBLE_DEVICES` set to
//! exactly its planned GPU UUID ([`worker_cuda_visible_devices`]), so it sees its own
//! device at CUDA ordinal 0 and no other participant's device. The plan's
//! `DeviceAssignment::local_index` is the host-wide index the census observed; it is
//! *not* a CUDA ordinal (CUDA's default enumeration order need not match `nvidia-smi`),
//! so it never becomes `device_index`.
//!
//! **Delivery.** A launched worker takes its topology from the frozen plan artifact and its
//! launch record only ([`bound_worker`]); the launcher that sets the per-worker environment is
//! [`super::launch::worker_command`].
use super::{
    contract::BoundedRunContract,
    launch::{self, LaunchRecord},
};
use crate::training::topology::{self, Assignment, Binding, LocalDevices, Participant, SCHEMA};
use anyhow::{anyhow, ensure, Context, Result};

/// The CUDA ordinal of a worker restricted to its own GPU.
pub const RESTRICTED_DEVICE_INDEX: usize = 0;

/// The trainer assignment a frozen bounded-run plan implies. Refuses a plan that is
/// invalid, names a rank without a GPU, or that `topology::validate` does not accept.
pub fn trainer_assignment(contract: &BoundedRunContract) -> Result<Assignment> {
    contract.validate_structure().map_err(|e| anyhow!("{e}"))?;
    let plan_sha256 = crate::manifest::hash(&contract.canonical_bytes()?);
    let participants = contract
        .participants
        .iter()
        .map(|p| {
            let device = p.device.as_ref().with_context(|| {
                format!(
                    "rank {} has no GPU; a trainer assignment needs one device per rank",
                    p.global_rank
                )
            })?;
            Ok(Participant {
                global_rank: p.global_rank as usize,
                host_id: p.host.machine_id.clone(),
                boot_id: p.host.boot_id.clone(),
                local_rank: p.local_rank as usize,
                device_index: RESTRICTED_DEVICE_INDEX,
                gpu_uuid: device.uuid.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let assignment = Assignment {
        schema: SCHEMA.into(),
        plan_sha256,
        world: contract.world_size(),
        participants,
    };
    topology::validate(&assignment).context("plan does not yield a valid trainer assignment")?;
    Ok(assignment)
}

/// Refuse an assignment that is not exactly the one the frozen plan implies.
pub fn verify_trainer_assignment(
    contract: &BoundedRunContract,
    assignment: &Assignment,
) -> Result<()> {
    let expected = trainer_assignment(contract)?;
    ensure!(
        assignment.plan_sha256 == expected.plan_sha256,
        "trainer assignment names another plan"
    );
    ensure!(
        assignment.schema == expected.schema && assignment.world == expected.world,
        "trainer assignment schema or world differs from the plan"
    );
    ensure!(
        assignment.participants.len() == expected.participants.len(),
        "trainer assignment participant count differs from the plan"
    );
    for (got, want) in assignment.participants.iter().zip(&expected.participants) {
        ensure!(
            got == want,
            "trainer assignment for rank {} differs from the plan",
            want.global_rank
        );
    }
    Ok(())
}

/// The trainer topology of one launched worker, read only from the frozen plan and that
/// worker's launch record, never from flags. Everything is refused before any communicator
/// exists:
/// - a launch record the plan does not derive (another plan, attempt, rank or host);
/// - a `CUDA_VISIBLE_DEVICES` that is not exactly the planned UUID (unset, a local index, a
///   duplicated, additional or foreign device);
/// - a local device view that is not exactly that one device at ordinal 0;
/// - everything [`topology::bind`] refuses (another host or boot, a UUID mismatch).
///
/// `local` is read only after the environment check has passed.
pub fn bound_worker(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    record: &LaunchRecord,
    observed_visibility: Option<&std::ffi::OsStr>,
    local: impl FnOnce() -> Result<LocalDevices>,
) -> Result<(Assignment, Binding)> {
    launch::verify_launch_record(contract, plan_sha256, record)?;
    launch::verify_worker_visibility(record, observed_visibility)?;
    let assignment = trainer_assignment(contract)?;
    ensure!(
        assignment.plan_sha256 == plan_sha256,
        "trainer assignment names another plan"
    );
    let local = local()?;
    ensure!(
        local.visible.len() == 1 && local.visible[0] == record.cuda_visible_devices,
        "a bounded worker must see exactly its planned device {} at CUDA ordinal {RESTRICTED_DEVICE_INDEX}; it sees {:?}",
        record.cuda_visible_devices,
        local.visible
    );
    let binding = topology::bind(
        &assignment,
        record.global_rank as usize,
        assignment.world,
        &local,
    )?;
    Ok((assignment, binding))
}

/// The exact `CUDA_VISIBLE_DEVICES` value for one worker: its planned GPU UUID only.
pub fn worker_cuda_visible_devices(
    contract: &BoundedRunContract,
    global_rank: u32,
) -> Result<String> {
    let p = contract
        .participant(global_rank)
        .with_context(|| format!("rank {global_rank} is not in the plan"))?;
    Ok(p.device
        .as_ref()
        .with_context(|| format!("rank {global_rank} has no GPU"))?
        .uuid
        .clone())
}
