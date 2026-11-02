use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub used_vram_bytes: Option<u64>,
    pub gpu_instance_id: Option<u32>,
    pub compute_instance_id: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: String,
    pub physical_index: u32,
    pub uuid: String,
    #[serde(default)]
    pub driver_version: String,
    pub parent_uuid: Option<String>,
    pub name: String,
    pub total_vram_bytes: u64,
    pub used_vram_bytes: u64,
    pub allocatable_vram_bytes: u64,
    pub utilization_per_mille: Option<u16>,
    pub temperature_millidegrees_celsius: Option<u32>,
    pub power_milliwatts: Option<u64>,
    pub power_limit_milliwatts: Option<u64>,
    pub mig_instance: Option<String>,
    pub processes: Vec<GpuProcess>,
    pub healthy: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuInventory {
    pub nvml_available: bool,
    pub mig_inventory: bool,
    pub devices: Vec<GpuDevice>,
    pub unavailable_reason: Option<String>,
    pub mps: MpsCapability,
}

impl GpuInventory {
    pub fn discover(cuda_visible_devices: Option<&str>) -> Self {
        match discover_nvml(cuda_visible_devices) {
            Ok((devices, mig_inventory)) => Self {
                nvml_available: true,
                mig_inventory,
                devices,
                unavailable_reason: None,
                mps: MpsCapability::detect(),
            },
            Err(error) => Self {
                nvml_available: false,
                mig_inventory: false,
                devices: Vec::new(),
                unavailable_reason: Some(error.to_string()),
                mps: MpsCapability::detect(),
            },
        }
    }
}

fn discover_nvml(visible: Option<&str>) -> Result<(Vec<GpuDevice>, bool), GpuError> {
    let nvml = Nvml::init()?;
    let driver_version = nvml.sys_driver_version()?;
    let selectors = visible.map(parse_visible_devices);
    if selectors.as_ref().is_some_and(BTreeSet::is_empty) {
        return Ok((Vec::new(), false));
    }
    let mut devices = Vec::new();
    let mut mig_inventory = false;
    for index in 0..nvml.device_count()? {
        let device = nvml.device_by_index(index)?;
        let uuid = device.uuid()?;
        let parent_visible = is_visible(selectors.as_ref(), index, &uuid);
        if device.mig_mode().is_ok_and(|mode| mode.current != 0) {
            match device.mig_device_count() {
                Ok(count) => {
                    mig_inventory = true;
                    for mig_index in 0..count {
                        if let Ok(mig) = device.mig_device_by_index(mig_index) {
                            if let Ok(sample) = sample_device(
                                &mig,
                                index,
                                Some(mig_index),
                                Some(uuid.clone()),
                                &driver_version,
                            ) {
                                if parent_visible
                                    || selectors
                                        .as_ref()
                                        .is_none_or(|selectors| selectors.contains(&sample.uuid))
                                {
                                    devices.push(sample);
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // MIG mode without readable instance handles is not advertised as isolatable.
                    mig_inventory = false;
                }
            }
        } else if parent_visible {
            devices.push(sample_device(&device, index, None, None, &driver_version)?);
        }
    }
    Ok((devices, mig_inventory))
}

fn sample_device(
    device: &Device<'_>,
    physical_index: u32,
    mig_index: Option<u32>,
    parent_uuid: Option<String>,
    driver_version: &str,
) -> Result<GpuDevice, GpuError> {
    let memory = device.memory_info()?;
    let uuid = device.uuid()?;
    let processes = complete_process_inventory(
        device.running_compute_processes(),
        device.running_graphics_processes(),
    )?
    .into_iter()
    .map(|process| GpuProcess {
        pid: process.pid,
        used_vram_bytes: match process.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => Some(bytes),
            UsedGpuMemory::Unavailable => None,
        },
        gpu_instance_id: process.gpu_instance_id,
        compute_instance_id: process.compute_instance_id,
    })
    .collect();
    let utilization_per_mille = device
        .utilization_rates()
        .ok()
        .and_then(|utilization| u16::try_from(utilization.gpu.saturating_mul(10)).ok());
    let temperature = device
        .temperature(TemperatureSensor::Gpu)
        .ok()
        .map(|celsius| celsius.saturating_mul(1_000));
    let allocatable_vram_bytes = memory.total.saturating_mul(100 - VRAM_HEADROOM_PERCENT) / 100;
    Ok(GpuDevice {
        id: mig_index.map_or_else(|| physical_index.to_string(), |_| uuid.clone()),
        physical_index,
        uuid,
        driver_version: driver_version.to_owned(),
        parent_uuid,
        name: device.name()?,
        total_vram_bytes: memory.total,
        used_vram_bytes: memory.used,
        allocatable_vram_bytes,
        utilization_per_mille,
        temperature_millidegrees_celsius: temperature,
        power_milliwatts: device.power_usage().ok().map(u64::from),
        power_limit_milliwatts: device.power_management_limit().ok().map(u64::from),
        mig_instance: mig_index.map(|index| index.to_string()),
        processes,
        healthy: true,
    })
}

pub(super) fn parse_visible_devices(value: &str) -> BTreeSet<String> {
    if value.split(',').any(|token| token.trim() == "-1") {
        return BTreeSet::new();
    }
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub(super) fn is_visible(selectors: Option<&BTreeSet<String>>, index: u32, uuid: &str) -> bool {
    selectors
        .is_none_or(|selectors| selectors.contains(&index.to_string()) || selectors.contains(uuid))
}
