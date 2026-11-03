use super::*;

pub trait GpuQuery: Send {
    fn initialize(&mut self) -> Result<(), NvmlError>;
    fn reset(&mut self);
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError>;
    fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError>;
    fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError>;
}

/// `nvml-wrapper` loads the NVML library at runtime, so a host without the NVIDIA driver
/// returns an error here rather than failing to start.
#[derive(Default)]
pub struct SystemGpu {
    nvml: Option<Nvml>,
}

impl SystemGpu {
    fn nvml(&self) -> Result<&Nvml, NvmlError> {
        self.nvml.as_ref().ok_or(NvmlError::Uninitialized)
    }
}

impl GpuQuery for SystemGpu {
    fn initialize(&mut self) -> Result<(), NvmlError> {
        self.nvml = None;
        self.nvml = Some(Nvml::init()?);
        Ok(())
    }
    fn reset(&mut self) {
        self.nvml = None;
    }
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError> {
        let nvml = self.nvml()?;
        (0..nvml.device_count()?)
            .map(|index| nvml.device_by_index(index)?.uuid())
            .collect()
    }
    fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError> {
        Ok(self.nvml()?.device_by_uuid(uuid)?.utilization_rates()?.gpu)
    }
    fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError> {
        Ok(self.nvml()?.device_by_uuid(uuid)?.memory_info()?.used)
    }
}

/// Stable TEL-05 reason codes (`nvml_device::UnavailableReason`) for an NVML error.
fn nvml_reason(error: &NvmlError) -> UnavailableReason {
    match error {
        NvmlError::LibraryNotFound | NvmlError::LibloadingError(_) => {
            UnavailableReason::MissingLibrary
        }
        NvmlError::DriverNotLoaded => UnavailableReason::MissingDriver,
        NvmlError::NoPermission => UnavailableReason::PermissionDenied,
        NvmlError::GpuLost => UnavailableReason::DeviceLost,
        _ => UnavailableReason::NvmlError,
    }
}

const NVML_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const NVML_MAX_BACKOFF: Duration = Duration::from_secs(5);

/// Device-wide GPU readings with bounded exponential retry after NVML failures. It never
/// panics and never synchronizes the device (§9.1 Rates).
pub struct NvmlSensor<Q: GpuQuery> {
    pub(crate) query: Q,
    initialized: bool,
    backoff: Duration,
    retry_at: Option<Instant>,
    last_reason: Option<UnavailableReason>,
}

impl<Q: GpuQuery> NvmlSensor<Q> {
    pub fn new(query: Q) -> Self {
        Self {
            query,
            initialized: false,
            backoff: NVML_INITIAL_BACKOFF,
            retry_at: None,
            last_reason: None,
        }
    }

    fn fail(&mut self, error: &NvmlError, now: Instant) -> String {
        let reason = nvml_reason(error);
        self.query.reset();
        self.initialized = false;
        self.last_reason = Some(reason);
        self.retry_at = Some(now + self.backoff);
        self.backoff = (self.backoff * 2).min(NVML_MAX_BACKOFF);
        reason.as_str().to_owned()
    }

    pub fn readings(&mut self, now: Instant) -> Result<Vec<GpuReading>, String> {
        if self.retry_at.is_some_and(|at| now < at) {
            let reason = self.last_reason.unwrap_or(UnavailableReason::NvmlError);
            return Err(reason.as_str().to_owned());
        }
        self.retry_at = None;
        if !self.initialized {
            if let Err(error) = self.query.initialize() {
                return Err(self.fail(&error, now));
            }
            self.initialized = true;
        }
        let uuids = match self.query.device_uuids() {
            Ok(uuids) => uuids,
            Err(error) => return Err(self.fail(&error, now)),
        };
        self.backoff = NVML_INITIAL_BACKOFF;
        self.last_reason = None;
        Ok(uuids
            .into_iter()
            .map(|uuid| GpuReading {
                util_pct: self
                    .query
                    .utilization_pct(&uuid)
                    .map(f64::from)
                    .map_err(|e| nvml_reason(&e).as_str().to_owned()),
                mem_used_bytes: self
                    .query
                    .memory_used_bytes(&uuid)
                    .map_err(|e| nvml_reason(&e).as_str().to_owned()),
                uuid,
            })
            .collect())
    }
}

/// Host sensors: WRK-03 per-PID sampling plus OS memory, CPU and disk queries.
pub struct SystemSensors {
    store: PathBuf,
    gpu: Option<NvmlSensor<SystemGpu>>,
}

impl SystemSensors {
    /// `store` is the `$JOPEDIME_HOME` volume used for `disk.free`. GPU metrics come from
    /// NVML, loaded at runtime: no library, driver or device yields `unavailable:<reason>`.
    pub fn new(store: impl Into<PathBuf>) -> Self {
        Self {
            store: store.into(),
            gpu: Some(NvmlSensor::new(SystemGpu::default())),
        }
    }

    /// Host sensors without GPU sampling (`gpu.*` = `unavailable:gpu_sampling_disabled`).
    pub fn without_gpu(store: impl Into<PathBuf>) -> Self {
        Self {
            store: store.into(),
            gpu: None,
        }
    }
}

impl Sensors for SystemSensors {
    fn process(&mut self, pid: u32) -> Result<ProcessReading, String> {
        crate::sampler::sample(pid)
            .map(|(cpu_seconds, rss_bytes)| ProcessReading {
                cpu_seconds,
                rss_bytes,
            })
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => "process_exited".to_owned(),
                _ => "sampler_error".to_owned(),
            })
    }
    fn logical_cpus(&mut self) -> Result<f64, String> {
        platform::logical_cpus()
    }
    fn allowed_cpus(&mut self) -> Result<f64, String> {
        platform::allowed_cpus()
    }
    fn mem_available(&mut self) -> Result<u64, String> {
        platform::mem_available()
    }
    fn disk_free(&mut self) -> Result<u64, String> {
        platform::disk_free(&self.store)
    }
    fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
        match &mut self.gpu {
            Some(gpu) => gpu.readings(Instant::now()),
            None => Err("gpu_sampling_disabled".to_owned()),
        }
    }
}
