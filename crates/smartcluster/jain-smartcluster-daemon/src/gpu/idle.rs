use super::*;

/// Administrator-qualified driver bookkeeping, never a general free-memory allowance.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedIdleBaseline {
    pub driver_version: String,
    pub idle_used_vram_bytes: u64,
    pub qualification_receipt_sha256: String,
}

impl QualifiedIdleBaseline {
    pub fn validate(&self, uuid: &str) -> Result<(), String> {
        if !uuid.starts_with("GPU-")
            || uuid.len() != 40
            || !uuid[4..].bytes().enumerate().all(|(index, byte)| {
                if [8, 13, 18, 23].contains(&index) {
                    byte == b'-'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
            || self.driver_version.is_empty()
            || self.driver_version.len() > 64
            || !self
                .driver_version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
            || self.idle_used_vram_bytes == 0
            || self.idle_used_vram_bytes > 512 * MEBIBYTE
            || self.qualification_receipt_sha256.len() != 64
            || !self
                .qualification_receipt_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("idle baseline requires a physical GPU UUID, exact driver, 1..=512MiB measured NVML v1 driver usage and a lowercase qualification receipt SHA256".to_owned());
        }
        Ok(())
    }
}

/// Both process inventories must succeed before any idle-baseline decision.
fn complete_process_inventory<T>(
    compute: Result<Vec<T>, NvmlError>,
    graphics: Result<Vec<T>, NvmlError>,
) -> Result<Vec<T>, NvmlError> {
    let mut processes = compute?;
    processes.extend(graphics?);
    Ok(processes)
}

pub(crate) fn qualified_idle_usage(
    device: &GpuDevice,
    baselines: &BTreeMap<String, QualifiedIdleBaseline>,
) -> bool {
    baselines.get(&device.uuid).is_some_and(|baseline| {
        baseline.validate(&device.uuid).is_ok()
            && baseline.driver_version == device.driver_version
            && device.processes.is_empty()
            && device.utilization_per_mille == Some(0)
            && baseline.idle_used_vram_bytes <= device.total_vram_bytes / 32
            && device.used_vram_bytes <= baseline.idle_used_vram_bytes
    })
}

#[cfg(test)]
mod idle_baseline_tests {
    use super::*;
    const UUID: &str = "GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b";
    fn device() -> GpuDevice {
        GpuDevice {
            id: "0".into(),
            physical_index: 0,
            uuid: UUID.into(),
            driver_version: "570.211.01".into(),
            parent_uuid: None,
            name: "fixture".into(),
            total_vram_bytes: 24 * 1024 * MEBIBYTE,
            used_vram_bytes: MEBIBYTE,
            allocatable_vram_bytes: 20 * 1024 * MEBIBYTE,
            utilization_per_mille: Some(0),
            temperature_millidegrees_celsius: None,
            power_milliwatts: None,
            power_limit_milliwatts: None,
            mig_instance: None,
            processes: vec![],
            healthy: true,
        }
    }
    fn baselines() -> BTreeMap<String, QualifiedIdleBaseline> {
        BTreeMap::from([(
            UUID.into(),
            QualifiedIdleBaseline {
                driver_version: "570.211.01".into(),
                idle_used_vram_bytes: MEBIBYTE,
                qualification_receipt_sha256: "a".repeat(64),
            },
        )])
    }
    #[test]
    fn idle_driver_usage_requires_exact_qualified_identity_and_bound() {
        let base = baselines();
        let mut d = device();
        assert!(!qualified_idle_usage(&d, &BTreeMap::new()));
        assert!(qualified_idle_usage(&d, &base));
        d.used_vram_bytes += 1;
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.driver_version = "580.1".into();
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.uuid = "GPU-00000000-0000-0000-0000-000000000000".into();
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.utilization_per_mille = None;
        assert!(!qualified_idle_usage(&d, &base));
        d.utilization_per_mille = Some(1);
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.processes.push(GpuProcess {
            pid: 123,
            used_vram_bytes: None,
            gpu_instance_id: None,
            compute_instance_id: None,
        });
        assert!(!qualified_idle_usage(&d, &base));
    }
    #[test]
    fn missing_compute_or_graphics_inventory_cannot_become_idle() {
        assert!(complete_process_inventory::<u32>(Err(NvmlError::Unknown), Ok(vec![])).is_err());
        assert!(complete_process_inventory::<u32>(Ok(vec![]), Err(NvmlError::Unknown)).is_err());
        assert_eq!(
            complete_process_inventory(Ok(vec![1]), Ok(vec![2])).unwrap(),
            vec![1, 2]
        );
    }
    #[test]
    fn idle_baseline_rejects_unbounded_or_unqualified_configuration() {
        let mut b = baselines().remove(UUID).unwrap();
        b.idle_used_vram_bytes = 513 * MEBIBYTE;
        assert!(b.validate(UUID).is_err());
        b.idle_used_vram_bytes = MEBIBYTE;
        b.qualification_receipt_sha256.clear();
        assert!(b.validate(UUID).is_err());
    }

    #[test]
    fn v1_driver_reservation_requires_a_device_bounded_measured_baseline() {
        let mut d = device();
        let mut base = baselines();
        d.used_vram_bytes = 466_354_176;
        base.get_mut(UUID).unwrap().idle_used_vram_bytes = d.used_vram_bytes;
        assert!(qualified_idle_usage(&d, &base));
        d.used_vram_bytes += 1;
        assert!(!qualified_idle_usage(&d, &base));
        d.used_vram_bytes -= 1;
        d.total_vram_bytes = 8 * 1024 * MEBIBYTE;
        assert!(!qualified_idle_usage(&d, &base));
    }
}
