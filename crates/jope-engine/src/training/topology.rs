//! Communicator identity separated from the local CUDA device (W7-T03).
//!
//! A training participant has two independent identities:
//! - its **communicator identity**: global rank in a world of known size;
//! - its **local device**: the host it runs on (host and boot identity), its
//!   local rank among that host's participants, the CUDA ordinal it must use
//!   as this worker sees devices, and that device's GPU UUID.
//!
//! Global rank 2 does not imply CUDA index 2. On one-GPU hosts every
//! participant uses index 0.
//!
//! A frozen [`Assignment`] names every participant. [`bind`] checks that
//! assignment and the worker's actual local view ([`LocalDevices`]) before
//! any communicator exists. The only way to create a communicator is from
//! the [`Binding`] it returns.
//!
//! Join point with the W7-T02 bounded-run contract: the bounded-run plan
//! supplies `plan_sha256`, and for every participant the global rank, host
//! and boot identity, local rank, device index and GPU UUID. The trainer
//! needs exactly these fields; the plan owns grants, deadlines and ceilings.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "jope.trainer_assignment.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Participant {
    /// Communicator rank, `0..world`.
    pub global_rank: usize,
    /// Stable host identity; the local view reads `/etc/machine-id`.
    pub host_id: String,
    /// Boot identity; the local view reads `/proc/sys/kernel/random/boot_id`.
    pub boot_id: String,
    /// Rank among this host's participants, `0..participants_on_host`.
    pub local_rank: usize,
    /// CUDA ordinal as this worker sees devices (after `CUDA_VISIBLE_DEVICES`).
    pub device_index: usize,
    /// The device's UUID, in `nvidia-smi` form (`GPU-` + 8-4-4-4-12 hex).
    pub gpu_uuid: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub schema: String,
    /// Digest of the W7-T02 bounded-run plan this assignment belongs to.
    pub plan_sha256: String,
    pub world: usize,
    pub participants: Vec<Participant>,
}

/// What one worker process can actually see, captured before any
/// communicator exists.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalDevices {
    pub host_id: String,
    pub boot_id: String,
    /// GPU UUIDs by visible CUDA ordinal.
    pub visible: Vec<String>,
}

/// A validated communicator identity and its owned local device.
/// Constructible only through [`bind`] or the explicit legacy constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    global_rank: usize,
    world: usize,
    local_rank: usize,
    device_index: usize,
    gpu_uuid: Option<String>,
    plan_sha256: Option<String>,
}

impl Binding {
    pub fn global_rank(&self) -> usize {
        self.global_rank
    }
    pub fn world(&self) -> usize {
        self.world
    }
    pub fn local_rank(&self) -> usize {
        self.local_rank
    }
    pub fn device_index(&self) -> usize {
        self.device_index
    }
    pub fn gpu_uuid(&self) -> Option<&str> {
        self.gpu_uuid.as_deref()
    }

    /// The pre-W7 single-host mapping, stated explicitly: every rank runs on
    /// one host whose CUDA ordinal equals its rank. No UUID or visibility is
    /// verified; receipts record it as `legacy_single_host`.
    pub fn legacy_single_host(rank: usize, world: usize) -> Result<Self> {
        ensure!(world > 0 && rank < world, "invalid legacy rank topology");
        Ok(Self {
            global_rank: rank,
            world,
            local_rank: rank,
            device_index: rank,
            gpu_uuid: None,
            plan_sha256: None,
        })
    }

    pub fn receipt(&self) -> Value {
        json!({"schema":SCHEMA,
            "binding":if self.gpu_uuid.is_some() {"assigned"} else {"legacy_single_host"},
            "global_rank":self.global_rank,"world":self.world,"local_rank":self.local_rank,
            "device_index":self.device_index,"gpu_uuid":self.gpu_uuid,
            "plan_sha256":self.plan_sha256})
    }
}

fn gpu_uuid_well_formed(uuid: &str) -> bool {
    let Some(hex) = uuid.strip_prefix("GPU-") else {
        return false;
    };
    let groups = hex.split('-').map(str::len).collect::<Vec<_>>();
    groups == [8, 4, 4, 4, 12]
        && hex
            .bytes()
            .all(|b| b == b'-' || b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Static validation of a frozen assignment: the exact participant count,
/// each global rank exactly once, each GPU owned by exactly one worker, one
/// boot per host, and per host local ranks exactly `0..n`.
pub fn validate(assignment: &Assignment) -> Result<()> {
    ensure!(
        assignment.schema == SCHEMA,
        "unknown trainer assignment schema"
    );
    ensure!(
        assignment.plan_sha256.len() == 64
            && assignment
                .plan_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "assignment plan digest malformed"
    );
    ensure!(assignment.world > 0, "assignment world must be positive");
    ensure!(
        assignment.participants.len() == assignment.world,
        "assignment names {} participants for a world of {}",
        assignment.participants.len(),
        assignment.world
    );
    let ranks = assignment
        .participants
        .iter()
        .map(|p| p.global_rank)
        .collect::<BTreeSet<_>>();
    ensure!(
        ranks.len() == assignment.world && ranks.iter().copied().eq(0..assignment.world),
        "global ranks must be exactly 0..{} once each",
        assignment.world
    );
    // Ownership is by GPU UUID. `device_index` is the ordinal as each worker
    // sees devices, so two workers on one host that are each restricted to
    // their own GPU both legitimately use index 0.
    let mut uuids = BTreeSet::new();
    let mut hosts = BTreeMap::<&str, (&str, BTreeSet<usize>)>::new();
    for p in &assignment.participants {
        ensure!(
            !p.host_id.is_empty() && !p.boot_id.is_empty(),
            "participant {} lacks host or boot identity",
            p.global_rank
        );
        ensure!(
            gpu_uuid_well_formed(&p.gpu_uuid),
            "participant {} GPU UUID malformed",
            p.global_rank
        );
        ensure!(
            uuids.insert(p.gpu_uuid.as_str()),
            "GPU {} is assigned to more than one worker",
            p.gpu_uuid
        );
        let host = hosts
            .entry(&p.host_id)
            .or_insert((&p.boot_id, BTreeSet::new()));
        ensure!(
            host.0 == p.boot_id,
            "host {} is assigned under two boot identities",
            p.host_id
        );
        ensure!(
            host.1.insert(p.local_rank),
            "local rank {} on host {} is assigned twice",
            p.local_rank,
            p.host_id
        );
    }
    for (host, (_, locals)) in &hosts {
        ensure!(
            locals.iter().copied().eq(0..locals.len()),
            "local ranks on host {host} must be exactly 0..{}",
            locals.len()
        );
    }
    Ok(())
}

/// Bind this worker's communicator identity to its owned device. Refuses:
/// - an invalid assignment, or an unassigned rank;
/// - a different world size;
/// - a different host or boot;
/// - an assigned index this worker cannot see;
/// - a UUID mismatch at that index;
/// - visibility of another participant's GPU (each worker owns exactly one
///   device).
pub fn bind(
    assignment: &Assignment,
    global_rank: usize,
    world: usize,
    local: &LocalDevices,
) -> Result<Binding> {
    validate(assignment)?;
    ensure!(
        world == assignment.world,
        "requested world {world} differs from the assignment's {}",
        assignment.world
    );
    let me = assignment
        .participants
        .iter()
        .find(|p| p.global_rank == global_rank)
        .with_context(|| format!("global rank {global_rank} is not assigned"))?;
    ensure!(
        me.host_id == local.host_id && me.boot_id == local.boot_id,
        "global rank {global_rank} is assigned to another host or boot"
    );
    let visible = local.visible.get(me.device_index).with_context(|| {
        format!(
            "assigned CUDA index {} is not visible to this worker ({} visible)",
            me.device_index,
            local.visible.len()
        )
    })?;
    ensure!(
        *visible == me.gpu_uuid,
        "CUDA index {} is {visible}, not the assigned {}",
        me.device_index,
        me.gpu_uuid
    );
    for other in &assignment.participants {
        ensure!(
            other.global_rank == global_rank || !local.visible.contains(&other.gpu_uuid),
            "worker for rank {global_rank} can see rank {}'s GPU {}",
            other.global_rank,
            other.gpu_uuid
        );
    }
    Ok(Binding {
        global_rank,
        world,
        local_rank: me.local_rank,
        device_index: me.device_index,
        gpu_uuid: Some(me.gpu_uuid.clone()),
        plan_sha256: Some(assignment.plan_sha256.clone()),
    })
}

/// Host and boot identity as the assignment names them.
pub fn host_identity() -> Result<(String, String)> {
    let read = |path: &str| -> Result<String> {
        Ok(std::fs::read_to_string(path)
            .with_context(|| format!("read {path}"))?
            .trim()
            .to_string())
    };
    Ok((
        read("/etc/machine-id")?,
        read("/proc/sys/kernel/random/boot_id")?,
    ))
}
