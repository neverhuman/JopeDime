//! Typed, versioned host census and admission preflight (W7-T01, source half).
//!
//! A census is an *observation record* supplied to the controller; this module never
//! gathers one. It does not SSH, probe a network or read live GPU state. The live
//! three-host census (actual boot ids, GPU UUIDs, free VRAM, CI allocations) remains a
//! separately admitted phase. Preflight compares a census against the frozen plan and
//! refuses, listing every reason, rather than admitting a partially matching group.
use super::{contract::BoundedRunContract, is_commit, is_digest, is_name};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CENSUS_SCHEMA: &str = "jope.native_host_census.v1";
pub const PREFLIGHT_SCHEMA: &str = "jope.native_preflight.v1";
/// A census record larger than this is refused before parsing.
pub const MAX_CENSUS_BYTES: usize = 1024 * 1024;
const MAX_HOSTS: usize = 64;
const MAX_GPUS_PER_HOST: usize = 16;
const MAX_OWNERSHIP_PER_HOST: usize = 4096;

/// Loaded runtime closure. Optional members are `None` on hosts without that component and
/// must then be `None` in the plan as well; absence is compared, never assumed equal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub runtime_sha256: String,
    pub compiler_sha256: String,
    pub cuda_sha256: Option<String>,
    pub nccl_sha256: Option<String>,
    pub driver_version: Option<String>,
}
/// Everything that must be byte-identical between the frozen plan and admission time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostIdentity {
    pub host: String,
    pub boot_id: String,
    pub address: String,
    pub certificate_sha256: String,
    pub source_commit: String,
    pub executable_sha256: String,
    pub installed_policy_sha256: String,
    pub runtime: RuntimeIdentity,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GpuObservation {
    pub uuid: String,
    pub model: String,
    pub local_index: u32,
    pub total_vram_bytes: u64,
    pub available_vram_bytes: u64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OwnerKind {
    Ci,
    User,
    Managed,
}
/// A reservation or device owner observed on a host. `run`/`attempt` identify a managed
/// bounded-run owner so that re-running preflight for the *same* attempt is not a conflict.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ownership {
    pub owner: String,
    pub kind: OwnerKind,
    pub device_uuid: Option<String>,
    pub cpu_millicores: u32,
    pub ram_bytes: u64,
    pub run: Option<String>,
    pub attempt: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostObservation {
    pub identity: HostIdentity,
    pub observed_unix_ms: u64,
    /// Schedulable capacity; reservations in `ownership` are subtracted from it.
    pub cpu_millicores_allocatable: u32,
    pub ram_bytes_allocatable: u64,
    pub gpus: Vec<GpuObservation>,
    pub ownership: Vec<Ownership>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostCensus {
    pub schema: String,
    pub captured_unix_ms: u64,
    pub hosts: Vec<HostObservation>,
}

/// Every reason a preflight refuses. The receipt is issued only when this list is empty.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefusalReason {
    StaleTelemetry {
        host: String,
        age_ms: u64,
    },
    FutureTelemetry {
        host: String,
        ahead_ms: u64,
    },
    MissingHost {
        host: String,
    },
    BootMismatch {
        host: String,
    },
    AddressMismatch {
        host: String,
    },
    CertificateMismatch {
        host: String,
    },
    SourceMismatch {
        host: String,
    },
    ExecutableMismatch {
        host: String,
    },
    RuntimeMismatch {
        host: String,
    },
    PolicyMismatch {
        host: String,
    },
    GpuMissing {
        host: String,
        uuid: String,
    },
    GpuIndexMismatch {
        host: String,
        uuid: String,
    },
    ConflictingOwnership {
        host: String,
        uuid: String,
        owner: String,
    },
    InsufficientCpu {
        host: String,
        free: u64,
        required: u64,
    },
    InsufficientRam {
        host: String,
        free: u64,
        required: u64,
    },
    InsufficientVram {
        host: String,
        uuid: String,
        free: u64,
        required: u64,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreflightRefusal {
    pub reasons: Vec<RefusalReason>,
}
impl std::fmt::Display for PreflightRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bounded-run preflight refused: {}",
            serde_json::to_string(&self.reasons).unwrap_or_default()
        )
    }
}
impl std::error::Error for PreflightRefusal {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostReceipt {
    pub boot_id: String,
    pub observed_unix_ms: u64,
    pub free_cpu_millicores: u64,
    pub free_ram_bytes: u64,
    pub devices: BTreeSet<String>,
}
/// Issued only for a complete, fresh, matching and funded census. It records what was
/// checked; it is not an allocation and holds no device.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreflightReceipt {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub census_sha256: String,
    pub checked_unix_ms: u64,
    pub hosts: BTreeMap<String, HostReceipt>,
    pub allocation_held: bool,
}

impl HostCensus {
    /// Bounded parse of an untrusted census record plus structural validation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_CENSUS_BYTES,
            "host census exceeds {MAX_CENSUS_BYTES} bytes"
        );
        let census: Self = serde_json::from_slice(bytes)?;
        census.validate()?;
        Ok(census)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CENSUS_SCHEMA,
            "unsupported host census schema `{}`",
            self.schema
        );
        ensure!(
            !self.hosts.is_empty() && self.hosts.len() <= MAX_HOSTS,
            "host census host count"
        );
        let mut hosts = BTreeSet::new();
        let mut uuids = BTreeSet::new();
        for h in &self.hosts {
            validate_identity(&h.identity)?;
            ensure!(
                hosts.insert(h.identity.host.clone()),
                "duplicate census host `{}`",
                h.identity.host
            );
            ensure!(
                h.observed_unix_ms <= self.captured_unix_ms,
                "host `{}` observed after census capture",
                h.identity.host
            );
            ensure!(
                h.gpus.len() <= MAX_GPUS_PER_HOST && h.ownership.len() <= MAX_OWNERSHIP_PER_HOST,
                "census host `{}` inventory bound",
                h.identity.host
            );
            let mut indices = BTreeSet::new();
            let mut local = BTreeSet::new();
            for g in &h.gpus {
                ensure!(
                    is_name(&g.uuid) && !g.model.is_empty() && g.model.len() <= 128,
                    "census GPU identity"
                );
                ensure!(
                    uuids.insert(g.uuid.clone()),
                    "duplicate GPU UUID `{}`",
                    g.uuid
                );
                ensure!(indices.insert(g.local_index), "duplicate local GPU index");
                ensure!(
                    g.available_vram_bytes <= g.total_vram_bytes && g.total_vram_bytes > 0,
                    "census VRAM accounting"
                );
                local.insert(g.uuid.clone());
            }
            for o in &h.ownership {
                ensure!(is_name(&o.owner), "census owner identity");
                if let Some(uuid) = &o.device_uuid {
                    ensure!(
                        local.contains(uuid),
                        "ownership names a device absent from host `{}`",
                        h.identity.host
                    );
                }
                ensure!(
                    o.run.is_some() == o.attempt.is_some()
                        && (o.run.is_none() || o.kind == OwnerKind::Managed),
                    "only managed owners carry a run/attempt"
                );
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_identity(i: &HostIdentity) -> Result<()> {
    ensure!(is_name(&i.host), "host name");
    ensure!(
        !i.boot_id.is_empty()
            && i.boot_id.len() <= 64
            && !i.address.is_empty()
            && i.address.len() <= 256,
        "host `{}` boot/address identity",
        i.host
    );
    ensure!(
        is_commit(&i.source_commit),
        "host `{}` source commit",
        i.host
    );
    for d in [
        &i.certificate_sha256,
        &i.executable_sha256,
        &i.installed_policy_sha256,
        &i.runtime.runtime_sha256,
        &i.runtime.compiler_sha256,
    ] {
        ensure!(is_digest(d), "host `{}` digest", i.host);
    }
    for d in [&i.runtime.cuda_sha256, &i.runtime.nccl_sha256]
        .into_iter()
        .flatten()
    {
        ensure!(is_digest(d), "host `{}` runtime component digest", i.host);
    }
    Ok(())
}

/// Compare a census against the frozen plan at `now_unix_ms`.
///
/// Refuses stale or future telemetry, any boot/address/certificate/source/executable/
/// runtime/policy difference, a missing or re-indexed planned GPU, a device owned by
/// anything other than this exact run attempt, and insufficient CPU, RAM or VRAM after
/// existing reservations. It holds nothing: capacity can change after the receipt, which
/// is why admission re-checks and SmartCluster remains the allocator.
pub fn preflight(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    census: &HostCensus,
    census_sha256: &str,
    now_unix_ms: u64,
) -> std::result::Result<PreflightReceipt, PreflightRefusal> {
    let policy = &contract.telemetry;
    let mut reasons = BTreeSet::new();
    let by_host: BTreeMap<&str, &HostObservation> = census
        .hosts
        .iter()
        .map(|h| (h.identity.host.as_str(), h))
        .collect();
    let mut hosts = BTreeMap::new();
    let mut planned: BTreeMap<&str, Vec<&super::contract::Participant>> = BTreeMap::new();
    for p in &contract.participants {
        planned.entry(p.host.host.as_str()).or_default().push(p);
    }
    for (name, members) in planned {
        let host = name.to_string();
        let Some(obs) = by_host.get(name) else {
            reasons.insert(RefusalReason::MissingHost { host });
            continue;
        };
        let expected = &members[0].host;
        if obs.observed_unix_ms > now_unix_ms.saturating_add(policy.clock_skew_ms) {
            reasons.insert(RefusalReason::FutureTelemetry {
                host: host.clone(),
                ahead_ms: obs.observed_unix_ms - now_unix_ms,
            });
        } else if now_unix_ms.saturating_sub(obs.observed_unix_ms) > policy.max_age_ms {
            reasons.insert(RefusalReason::StaleTelemetry {
                host: host.clone(),
                age_ms: now_unix_ms - obs.observed_unix_ms,
            });
        }
        let seen = &obs.identity;
        let mismatch = [
            (
                seen.boot_id != expected.boot_id,
                RefusalReason::BootMismatch { host: host.clone() },
            ),
            (
                seen.address != expected.address,
                RefusalReason::AddressMismatch { host: host.clone() },
            ),
            (
                seen.certificate_sha256 != expected.certificate_sha256,
                RefusalReason::CertificateMismatch { host: host.clone() },
            ),
            (
                seen.source_commit != expected.source_commit,
                RefusalReason::SourceMismatch { host: host.clone() },
            ),
            (
                seen.executable_sha256 != expected.executable_sha256,
                RefusalReason::ExecutableMismatch { host: host.clone() },
            ),
            (
                seen.runtime != expected.runtime,
                RefusalReason::RuntimeMismatch { host: host.clone() },
            ),
            (
                seen.installed_policy_sha256 != expected.installed_policy_sha256,
                RefusalReason::PolicyMismatch { host: host.clone() },
            ),
        ];
        for (differs, reason) in mismatch {
            if differs {
                reasons.insert(reason);
            }
        }
        let ours = |o: &Ownership| {
            o.kind == OwnerKind::Managed
                && o.run.as_deref() == Some(contract.run.as_str())
                && o.attempt == Some(contract.attempt)
        };
        let reserved_cpu: u64 = obs
            .ownership
            .iter()
            .filter(|o| !ours(o))
            .map(|o| u64::from(o.cpu_millicores))
            .sum();
        let reserved_ram: u64 = obs
            .ownership
            .iter()
            .filter(|o| !ours(o))
            .map(|o| o.ram_bytes)
            .sum();
        let free_cpu = u64::from(obs.cpu_millicores_allocatable).saturating_sub(reserved_cpu);
        let free_ram = obs.ram_bytes_allocatable.saturating_sub(reserved_ram);
        let need_cpu: u64 = members
            .iter()
            .map(|p| u64::from(p.ceilings.cpu_millicores))
            .sum();
        let need_ram: u64 = members.iter().map(|p| p.ceilings.ram_bytes).sum();
        if free_cpu < need_cpu {
            reasons.insert(RefusalReason::InsufficientCpu {
                host: host.clone(),
                free: free_cpu,
                required: need_cpu,
            });
        }
        if free_ram < need_ram {
            reasons.insert(RefusalReason::InsufficientRam {
                host: host.clone(),
                free: free_ram,
                required: need_ram,
            });
        }
        let mut devices = BTreeSet::new();
        for p in &members {
            let Some(device) = &p.device else { continue };
            let uuid = device.uuid.clone();
            let Some(gpu) = obs.gpus.iter().find(|g| g.uuid == device.uuid) else {
                reasons.insert(RefusalReason::GpuMissing {
                    host: host.clone(),
                    uuid,
                });
                continue;
            };
            if gpu.local_index != device.local_index {
                reasons.insert(RefusalReason::GpuIndexMismatch {
                    host: host.clone(),
                    uuid: uuid.clone(),
                });
            }
            for o in obs
                .ownership
                .iter()
                .filter(|o| o.device_uuid.as_deref() == Some(device.uuid.as_str()) && !ours(o))
            {
                reasons.insert(RefusalReason::ConflictingOwnership {
                    host: host.clone(),
                    uuid: uuid.clone(),
                    owner: o.owner.clone(),
                });
            }
            let need = p.ceilings.gpu_vram_bytes.unwrap_or(0);
            if gpu.available_vram_bytes < need {
                reasons.insert(RefusalReason::InsufficientVram {
                    host: host.clone(),
                    uuid: uuid.clone(),
                    free: gpu.available_vram_bytes,
                    required: need,
                });
            }
            devices.insert(uuid);
        }
        hosts.insert(
            host,
            HostReceipt {
                boot_id: seen.boot_id.clone(),
                observed_unix_ms: obs.observed_unix_ms,
                free_cpu_millicores: free_cpu,
                free_ram_bytes: free_ram,
                devices,
            },
        );
    }
    if !reasons.is_empty() {
        return Err(PreflightRefusal {
            reasons: reasons.into_iter().collect(),
        });
    }
    Ok(PreflightReceipt {
        schema: PREFLIGHT_SCHEMA.into(),
        run: contract.run.clone(),
        attempt: contract.attempt,
        plan_sha256: plan_sha256.into(),
        census_sha256: census_sha256.into(),
        checked_unix_ms: now_unix_ms,
        hosts,
        allocation_held: false,
    })
}
