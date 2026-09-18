//! Optional native admission limits. Filesystem reserves are admission guards, not quotas.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::PathBuf;

use crate::capacity::CpuSet;
use jain_smartcluster_core::{JobSpec, Workload};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourcePolicy {
    pub service_cpu_sets: BTreeMap<String, String>,
    pub minimum_available_memory_bytes: u64,
    pub filesystem_reserves: Vec<FilesystemReserve>,
    pub trusted_scratch: Option<ScratchReservationPool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FilesystemReserve {
    pub path: PathBuf,
    pub minimum_available_bytes: u64,
    /// Requires this exact mountpoint to be backed by this block device (UUID symlinks supported).
    pub expected_device: Option<PathBuf>,
}

/// Cooperative trusted-worker reservation, deliberately not a kernel disk quota.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScratchReservationPool {
    pub path: PathBuf,
    pub capacity_bytes: u64,
    pub minimum_available_bytes: u64,
    pub trusted_services: BTreeSet<String>,
}

impl ResourcePolicy {
    pub fn enabled(&self) -> bool {
        self.trusted_scratch.is_some()
            || !self.service_cpu_sets.is_empty()
            || self.minimum_available_memory_bytes != 0
            || !self.filesystem_reserves.is_empty()
    }

    pub fn validate(&self, delegated: &CpuSet) -> Result<(), String> {
        for (service, text) in &self.service_cpu_sets {
            let cpus = CpuSet::parse(text).map_err(|error| error.to_string())?;
            if cpus.is_empty() || cpus.intersect(delegated) != cpus {
                return Err(format!("service {service} CPU set must be nonempty and contained in delegated capacity"));
            }
        }
        if let Some(pool) = &self.trusted_scratch {
            if !pool.path.is_absolute()
                || pool.capacity_bytes == 0
                || pool.capacity_bytes > 32 * 1024 * 1024 * 1024
                || pool.trusted_services.is_empty()
            {
                return Err("trusted scratch requires absolute path, nonempty service allowlist and capacity in 1..=32GiB".to_owned());
            }
        }
        for reserve in &self.filesystem_reserves {
            if !reserve.path.is_absolute()
                || reserve
                    .expected_device
                    .as_ref()
                    .is_some_and(|path| !path.is_absolute())
            {
                return Err("resource reserve paths must be absolute".to_owned());
            }
        }
        Ok(())
    }

    pub fn cpus_for(&self, spec: &JobSpec, delegated: &CpuSet) -> Result<CpuSet, String> {
        if let Workload::TrustedService { service, .. } = &spec.workload {
            if let Some(text) = self.service_cpu_sets.get(service.as_str()) {
                return CpuSet::parse(text).map_err(|error| error.to_string());
            }
        }
        Ok(delegated.clone())
    }

    pub fn validate_scratch_request(&self, spec: &JobSpec) -> Result<(), String> {
        if spec.resources.scratch_bytes == 0 {
            return Ok(());
        }
        let pool = self
            .trusted_scratch
            .as_ref()
            .ok_or("scratch reservations are not configured")?;
        if !matches!(&spec.workload, Workload::TrustedService { service, .. } if pool.trusted_services.contains(service.as_str()))
        {
            return Err(
                "scratch reservations require an explicitly allowlisted trusted service".to_owned(),
            );
        }
        if spec.resources.scratch_bytes > pool.capacity_bytes {
            return Err("scratch request exceeds configured reservation pool".to_owned());
        }
        Ok(())
    }

    pub fn scratch_capacity(&self) -> Result<u64, String> {
        let Some(pool) = &self.trusted_scratch else {
            return Ok(0);
        };
        if !pool.path.is_dir() {
            return Err("scratch root is not an available directory".to_owned());
        }
        let stats = nix::sys::statvfs::statvfs(&pool.path).map_err(|error| error.to_string())?;
        Ok(scratch_capacity_from_free(
            pool.capacity_bytes,
            stats
                .blocks_available()
                .saturating_mul(stats.fragment_size()),
            pool.minimum_available_bytes,
        ))
    }

    pub fn check_live(&self) -> Result<(), String> {
        if self.minimum_available_memory_bytes != 0 {
            let source = fs::read_to_string("/proc/meminfo").map_err(|error| error.to_string())?;
            let available = memory_available(&source).ok_or("MemAvailable is absent or invalid")?;
            if available < self.minimum_available_memory_bytes {
                return Err(format!(
                    "RAM reserve: {available} bytes available, {} required",
                    self.minimum_available_memory_bytes
                ));
            }
        }
        for reserve in &self.filesystem_reserves {
            if let Some(device) = &reserve.expected_device {
                let mountpoint =
                    fs::canonicalize(&reserve.path).map_err(|error| error.to_string())?;
                let mounts = fs::read_to_string("/proc/self/mountinfo")
                    .map_err(|error| error.to_string())?;
                let source = mount_source(&mounts, &mountpoint).ok_or_else(|| {
                    format!("required archive mount {} is absent", mountpoint.display())
                })?;
                let expected = fs::metadata(device).map_err(|error| error.to_string())?;
                let actual = fs::metadata(&source).map_err(|error| error.to_string())?;
                if !expected.file_type().is_block_device()
                    || !actual.file_type().is_block_device()
                    || expected.rdev() != actual.rdev()
                {
                    return Err(format!(
                        "mount {} does not match device {}",
                        mountpoint.display(),
                        device.display()
                    ));
                }
            }
            let stats =
                nix::sys::statvfs::statvfs(&reserve.path).map_err(|error| error.to_string())?;
            let available = stats
                .blocks_available()
                .saturating_mul(stats.fragment_size());
            if available < reserve.minimum_available_bytes {
                return Err(format!(
                    "storage reserve {}: {available} bytes available, {} required",
                    reserve.path.display(),
                    reserve.minimum_available_bytes
                ));
            }
        }
        Ok(())
    }
}

fn scratch_capacity_from_free(capacity: u64, free: u64, reserve: u64) -> u64 {
    capacity.min(free.saturating_sub(reserve))
}

/// Conservative overlap accounting: every intersecting allocation's full maximum
/// CPU request counts against the candidate pool, including recovered allocations.
pub(crate) fn cpu_pool_fits(candidate: &CpuSet, requested: u64, running: &[(CpuSet, u64)]) -> bool {
    let used = running
        .iter()
        .filter(|(pool, _)| !candidate.intersect(pool).is_empty())
        .fold(0u64, |sum, (_, cpu)| sum.saturating_add(*cpu));
    used.saturating_add(requested) <= (candidate.len() as u64).saturating_mul(1000)
}

fn memory_available(source: &str) -> Option<u64> {
    let line = source
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let mut fields = line.split_whitespace();
    fields.next()?;
    let value = fields.next()?.parse::<u64>().ok()?;
    (fields.next()? == "kB").then(|| value.saturating_mul(1024))
}

fn mount_source(source: &str, mountpoint: &std::path::Path) -> Option<PathBuf> {
    source
        .lines()
        .filter_map(|line| {
            let (left, right) = line.split_once(" - ")?;
            let path = unescape_mount(left.split_whitespace().nth(4)?);
            if std::path::Path::new(&path) != mountpoint {
                return None;
            }
            Some(PathBuf::from(unescape_mount(
                right.split_whitespace().nth(1)?,
            )))
        })
        .next_back()
}

fn unescape_mount(text: &str) -> String {
    text.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_capacity_shrinks_with_external_use_and_preserves_reserve() {
        assert_eq!(scratch_capacity_from_free(32, 150, 100), 32);
        assert_eq!(scratch_capacity_from_free(32, 115, 100), 15);
        assert_eq!(scratch_capacity_from_free(32, 99, 100), 0);
    }

    #[test]
    fn overlapping_pools_count_full_concurrent_requests() {
        let a = CpuSet::parse("8-11").unwrap();
        let b = CpuSet::parse("10-13").unwrap();
        let c = CpuSet::parse("60-63").unwrap();
        assert!(cpu_pool_fits(&a, 2000, &[(a.clone(), 2000)]));
        assert!(!cpu_pool_fits(&a, 2001, &[(a.clone(), 2000)]));
        assert!(!cpu_pool_fits(&b, 3000, &[(a.clone(), 2000)]));
        assert!(cpu_pool_fits(&c, 4000, &[(a, 4000)]));
    }
    #[test]
    fn cpu_pool_outside_delegation_fails_closed() {
        let policy = ResourcePolicy {
            service_cpu_sets: BTreeMap::from([("gp".into(), "0-7".into())]),
            ..Default::default()
        };
        assert!(policy.validate(&CpuSet::parse("8-15").unwrap()).is_err());
    }
    #[test]
    fn mount_identity_requires_exact_mount_and_preserves_escaped_paths() {
        let source = "11 1 8:1 / /archive\\040disk rw - fuseblk /dev/sdc1 rw\n";
        assert_eq!(
            mount_source(source, std::path::Path::new("/archive disk")),
            Some(PathBuf::from("/dev/sdc1"))
        );
        assert!(mount_source(source, std::path::Path::new("/archive disk/missing")).is_none());
        assert!(mount_source("", std::path::Path::new("/archive disk")).is_none());
    }
    #[test]
    fn missing_storage_and_exhausted_reserves_block_admission() {
        let dir = tempfile::tempdir().unwrap();
        let mut policy = ResourcePolicy {
            filesystem_reserves: vec![FilesystemReserve {
                path: dir.path().join("missing"),
                minimum_available_bytes: 1,
                expected_device: None,
            }],
            ..Default::default()
        };
        assert!(policy.check_live().is_err());
        policy.filesystem_reserves[0].path = dir.path().to_owned();
        policy.filesystem_reserves[0].minimum_available_bytes = u64::MAX;
        assert!(policy.check_live().is_err());
        policy.filesystem_reserves[0].minimum_available_bytes = 0;
        assert!(policy.check_live().is_ok());
        policy.filesystem_reserves[0].expected_device =
            Some(PathBuf::from("/dev/disk/by-uuid/missing"));
        assert!(policy.check_live().is_err());
    }
    #[test]
    fn memory_measurement_requires_available_field_and_units() {
        assert_eq!(
            memory_available("MemFree: 999 kB\nMemAvailable: 24 kB\n"),
            Some(24576)
        );
        assert_eq!(memory_available("MemFree: 999 kB"), None);
        assert_eq!(memory_available("MemAvailable: 24 bytes"), None);
    }
}
