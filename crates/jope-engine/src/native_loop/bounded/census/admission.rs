use super::*;

/// Compare a census against the frozen plan at `now_unix_ms`.
///
/// Refuses stale or future telemetry, any machine/boot/address/certificate/source/
/// executable/runtime/policy difference, a missing or re-indexed planned GPU, a device owned by
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
    let mut planned: BTreeMap<&str, Vec<&super::super::contract::Participant>> = BTreeMap::new();
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
                seen.machine_id != expected.machine_id,
                RefusalReason::MachineMismatch { host: host.clone() },
            ),
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
