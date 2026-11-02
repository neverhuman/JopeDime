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

mod model;
pub use model::{
    GpuObservation, HostCensus, HostIdentity, HostObservation, HostReceipt, OwnerKind, Ownership,
    PreflightReceipt, PreflightRefusal, RefusalReason, RuntimeIdentity,
};

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

/// A machine id as `/etc/machine-id` holds it: exactly 32 lowercase hex digits.
pub(crate) fn is_machine_id(s: &str) -> bool {
    s.len() == 32
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// A kernel boot id as a census records it: 1..=64 bytes.
pub(crate) fn is_boot_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64
}
pub(crate) fn validate_identity(i: &HostIdentity) -> Result<()> {
    ensure!(is_name(&i.host), "host name");
    ensure!(
        is_machine_id(&i.machine_id),
        "host `{}` machine id must be 32 lowercase hex (/etc/machine-id)",
        i.host
    );
    ensure!(
        is_boot_id(&i.boot_id) && !i.address.is_empty() && i.address.len() <= 256,
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

mod admission;

pub use admission::preflight;
