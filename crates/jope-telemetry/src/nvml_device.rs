//! NVML device discovery keyed by the device UUID.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use nvml_wrapper::{error::NvmlError, Nvml};

const INITIAL_RETRY_BACKOFF: Duration = Duration::from_millis(100);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(5);

/// The NVML operations needed to discover devices.
///
/// This boundary keeps discovery tests independent of a local NVIDIA driver.
pub trait NvmlInterface {
    fn initialize(&mut self) -> Result<(), NvmlError>;
    fn reset(&mut self);
    fn device_count(&self) -> Result<u32, NvmlError>;
    fn device_uuid(&self, index: u32) -> Result<String, NvmlError>;
}

/// Production NVML implementation backed by `nvml-wrapper`.
#[derive(Default)]
pub struct SystemNvml {
    nvml: Option<Nvml>,
}

impl NvmlInterface for SystemNvml {
    fn initialize(&mut self) -> Result<(), NvmlError> {
        self.nvml = None;
        self.nvml = Some(Nvml::init()?);
        Ok(())
    }

    fn reset(&mut self) {
        self.nvml = None;
    }

    fn device_count(&self) -> Result<u32, NvmlError> {
        self.nvml
            .as_ref()
            .ok_or(NvmlError::Uninitialized)?
            .device_count()
    }

    fn device_uuid(&self, index: u32) -> Result<String, NvmlError> {
        self.nvml
            .as_ref()
            .ok_or(NvmlError::Uninitialized)?
            .device_by_index(index)?
            .uuid()
    }
}

/// Stable reason codes used in `unavailable:<reason>` telemetry values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnavailableReason {
    MissingLibrary,
    MissingDriver,
    PermissionDenied,
    DeviceLost,
    NvmlError,
}

impl UnavailableReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingLibrary => "missing_library",
            Self::MissingDriver => "missing_driver",
            Self::PermissionDenied => "permission_denied",
            Self::DeviceLost => "device_lost",
            Self::NvmlError => "nvml_error",
        }
    }

    fn from_error(error: &NvmlError) -> Self {
        match error {
            NvmlError::LibraryNotFound | NvmlError::LibloadingError(_) => Self::MissingLibrary,
            NvmlError::DriverNotLoaded => Self::MissingDriver,
            NvmlError::NoPermission => Self::PermissionDenied,
            NvmlError::GpuLost => Self::DeviceLost,
            _ => Self::NvmlError,
        }
    }
}

/// UUID-to-index mapping, or an explicit unavailable reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceMapStatus {
    Available(BTreeMap<String, u32>),
    Unavailable(UnavailableReason),
}

impl DeviceMapStatus {
    pub fn availability(&self) -> String {
        match self {
            Self::Available(_) => "measured".to_owned(),
            Self::Unavailable(reason) => format!("unavailable:{}", reason.as_str()),
        }
    }

    pub fn devices(&self) -> Option<&BTreeMap<String, u32>> {
        match self {
            Self::Available(devices) => Some(devices),
            Self::Unavailable(_) => None,
        }
    }
}

/// Discovers NVML devices and retries after errors using bounded exponential backoff.
pub struct NvmlDeviceMapper<I: NvmlInterface = SystemNvml> {
    interface: I,
    initialized: bool,
    initial_backoff: Duration,
    max_backoff: Duration,
    retry_backoff: Duration,
    retry_at: Option<Instant>,
    last_reason: Option<UnavailableReason>,
}

impl NvmlDeviceMapper<SystemNvml> {
    pub fn new() -> Self {
        Self::with_backoff(
            SystemNvml::default(),
            INITIAL_RETRY_BACKOFF,
            MAX_RETRY_BACKOFF,
        )
    }
}

impl Default for NvmlDeviceMapper<SystemNvml> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: NvmlInterface> NvmlDeviceMapper<I> {
    fn with_backoff(interface: I, initial_backoff: Duration, max_backoff: Duration) -> Self {
        assert!(!initial_backoff.is_zero());
        assert!(max_backoff >= initial_backoff);
        Self {
            interface,
            initialized: false,
            initial_backoff,
            max_backoff,
            retry_backoff: initial_backoff,
            retry_at: None,
            last_reason: None,
        }
    }

    pub fn scan(&mut self) -> DeviceMapStatus {
        self.scan_at(Instant::now())
    }

    fn scan_at(&mut self, now: Instant) -> DeviceMapStatus {
        if self.retry_at.is_some_and(|retry_at| now < retry_at) {
            return DeviceMapStatus::Unavailable(
                self.last_reason.unwrap_or(UnavailableReason::NvmlError),
            );
        }
        self.retry_at = None;

        if !self.initialized {
            if let Err(error) = self.interface.initialize() {
                return self.unavailable(error, now);
            }
            self.initialized = true;
        }

        let result = (|| {
            let count = self.interface.device_count()?;
            let mut devices = BTreeMap::new();
            for index in 0..count {
                devices.insert(self.interface.device_uuid(index)?, index);
            }
            Ok(devices)
        })();

        match result {
            Ok(devices) => {
                self.retry_backoff = self.initial_backoff;
                self.last_reason = None;
                DeviceMapStatus::Available(devices)
            }
            Err(error) => self.unavailable(error, now),
        }
    }

    fn unavailable(&mut self, error: NvmlError, now: Instant) -> DeviceMapStatus {
        self.interface.reset();
        self.initialized = false;
        let reason = UnavailableReason::from_error(&error);
        self.last_reason = Some(reason);
        self.retry_at = Some(now + self.retry_backoff);
        self.retry_backoff = self.retry_backoff.saturating_mul(2).min(self.max_backoff);
        DeviceMapStatus::Unavailable(reason)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;

    struct FakeSession {
        count: RefCell<Option<Result<u32, NvmlError>>>,
        uuids: RefCell<VecDeque<Result<String, NvmlError>>>,
    }

    impl FakeSession {
        fn new(count: Result<u32, NvmlError>, uuids: Vec<Result<String, NvmlError>>) -> Self {
            Self {
                count: RefCell::new(Some(count)),
                uuids: RefCell::new(uuids.into()),
            }
        }
    }

    struct FakeNvml {
        sessions: VecDeque<Result<FakeSession, NvmlError>>,
        active: Option<FakeSession>,
        initializations: Arc<AtomicUsize>,
    }

    impl FakeNvml {
        fn new(
            sessions: Vec<Result<FakeSession, NvmlError>>,
            initializations: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                sessions: sessions.into(),
                active: None,
                initializations,
            }
        }
    }

    impl NvmlInterface for FakeNvml {
        fn initialize(&mut self) -> Result<(), NvmlError> {
            self.initializations.fetch_add(1, Ordering::SeqCst);
            self.active = Some(
                self.sessions
                    .pop_front()
                    .unwrap_or(Err(NvmlError::Uninitialized))?,
            );
            Ok(())
        }

        fn reset(&mut self) {
            self.active = None;
        }

        fn device_count(&self) -> Result<u32, NvmlError> {
            self.active
                .as_ref()
                .ok_or(NvmlError::Uninitialized)?
                .count
                .borrow_mut()
                .take()
                .unwrap_or(Err(NvmlError::Uninitialized))
        }

        fn device_uuid(&self, _index: u32) -> Result<String, NvmlError> {
            self.active
                .as_ref()
                .ok_or(NvmlError::Uninitialized)?
                .uuids
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(NvmlError::NotFound))
        }
    }

    fn mapper(fake: FakeNvml) -> NvmlDeviceMapper<FakeNvml> {
        NvmlDeviceMapper::with_backoff(fake, Duration::from_millis(10), Duration::from_millis(40))
    }

    fn session(
        count: Result<u32, NvmlError>,
        uuids: Vec<Result<String, NvmlError>>,
    ) -> FakeSession {
        FakeSession::new(count, uuids)
    }

    #[test]
    fn maps_devices_by_uuid() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut mapper = mapper(FakeNvml::new(
            vec![Ok(session(
                Ok(2),
                vec![Ok("GPU-a".into()), Ok("GPU-b".into())],
            ))],
            calls,
        ));

        let status = mapper.scan();
        assert_eq!(status.availability(), "measured");
        assert_eq!(status.devices().unwrap().get("GPU-a"), Some(&0));
        assert_eq!(status.devices().unwrap().get("GPU-b"), Some(&1));
    }

    #[test]
    fn reports_missing_library_and_driver() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut missing_library = mapper(FakeNvml::new(
            vec![Err(NvmlError::LibraryNotFound)],
            Arc::clone(&calls),
        ));
        assert_eq!(
            missing_library.scan().availability(),
            "unavailable:missing_library"
        );

        let mut missing_driver =
            mapper(FakeNvml::new(vec![Err(NvmlError::DriverNotLoaded)], calls));
        assert_eq!(
            missing_driver.scan().availability(),
            "unavailable:missing_driver"
        );
    }

    #[test]
    fn reports_permission_denied_and_lost_device() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut permission_denied = mapper(FakeNvml::new(
            vec![Ok(session(Err(NvmlError::NoPermission), vec![]))],
            Arc::clone(&calls),
        ));
        assert_eq!(
            permission_denied.scan().availability(),
            "unavailable:permission_denied"
        );

        let mut lost_device = mapper(FakeNvml::new(
            vec![Ok(session(Ok(1), vec![Err(NvmlError::GpuLost)]))],
            calls,
        ));
        assert_eq!(lost_device.scan().availability(), "unavailable:device_lost");
    }

    #[test]
    fn reinitializes_after_recoverable_error_with_backoff() {
        let initializations = Arc::new(AtomicUsize::new(0));
        let mut mapper = mapper(FakeNvml::new(
            vec![
                Ok(session(Err(NvmlError::GpuLost), vec![])),
                Ok(session(Ok(1), vec![Ok("GPU-recovered".into())])),
            ],
            Arc::clone(&initializations),
        ));
        let start = Instant::now();

        assert_eq!(
            mapper.scan_at(start).availability(),
            "unavailable:device_lost"
        );
        assert_eq!(
            mapper
                .scan_at(start + Duration::from_millis(5))
                .availability(),
            "unavailable:device_lost"
        );
        assert_eq!(initializations.load(Ordering::SeqCst), 1);

        let recovered = mapper.scan_at(start + Duration::from_millis(10));
        assert_eq!(recovered.devices().unwrap().get("GPU-recovered"), Some(&0));
        assert_eq!(initializations.load(Ordering::SeqCst), 2);
    }
}
