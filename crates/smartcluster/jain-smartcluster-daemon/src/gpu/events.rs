use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEventKind {
    CriticalXid(Option<String>),
    DoubleBitEcc,
    DeviceLost,
    MigConfigurationChanged,
    MonitorFailed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuEvent {
    pub device_uuid: Option<String>,
    pub kind: GpuEventKind,
}

pub struct GpuEventMonitor {
    stopped: Arc<AtomicBool>,
}

impl GpuEventMonitor {
    pub fn spawn(sender: tokio::sync::mpsc::Sender<GpuEvent>) -> Result<Self, GpuError> {
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopped);
        let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
        tokio::task::spawn_blocking(move || monitor_events(sender, &stop, ready_sender));
        match ready_receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(message)) => return Err(GpuError::EventMonitorSetup(message)),
            Err(error) => return Err(GpuError::EventMonitorSetup(error.to_string())),
        }
        Ok(Self { stopped })
    }
}

impl Drop for GpuEventMonitor {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
    }
}

fn monitor_events(
    sender: tokio::sync::mpsc::Sender<GpuEvent>,
    stopped: &AtomicBool,
    ready: std_mpsc::SyncSender<Result<(), String>>,
) {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let mut set = match nvml.create_event_set() {
        Ok(set) => set,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let count = match nvml.device_count() {
        Ok(count) => count,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let mut registered_any = false;
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            let _ = ready.send(Err(format!("cannot open NVML device {index}")));
            return;
        };
        let Ok(supported) = device.supported_event_types() else {
            let _ = ready.send(Err(format!("cannot query NVML events for device {index}")));
            return;
        };
        let requested = supported
            & (EventTypes::CRITICAL_XID_ERROR
                | EventTypes::DOUBLE_BIT_ECC_ERROR
                | EventTypes::MIG_CONFIG_CHANGE);
        if requested.is_empty() {
            let _ = ready.send(Err(format!(
                "visible NVML device {index} exposes none of the required safety events"
            )));
            return;
        }
        let Ok(registered) = device.register_events(requested, set) else {
            let _ = ready.send(Err(format!(
                "cannot register NVML events for device {index}"
            )));
            return;
        };
        set = registered;
        registered_any = true;
    }
    if !registered_any {
        let _ = ready.send(Err(
            "no visible GPU supports required NVML safety events".to_owned()
        ));
        return;
    }
    if ready.send(Ok(())).is_err() {
        return;
    }
    while !stopped.load(Ordering::Acquire) {
        match set.wait(500) {
            Ok(event) => {
                let uuid = event.device.uuid().ok();
                let kind = if event.event_type.contains(EventTypes::CRITICAL_XID_ERROR) {
                    GpuEventKind::CriticalXid(event.event_data.map(|value| format!("{value:?}")))
                } else if event.event_type.contains(EventTypes::DOUBLE_BIT_ECC_ERROR) {
                    GpuEventKind::DoubleBitEcc
                } else if event.event_type.contains(EventTypes::MIG_CONFIG_CHANGE) {
                    GpuEventKind::MigConfigurationChanged
                } else {
                    continue;
                };
                if sender
                    .blocking_send(GpuEvent {
                        device_uuid: uuid,
                        kind,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(NvmlError::Timeout) => {}
            Err(NvmlError::GpuLost) => {
                let _ = sender.blocking_send(GpuEvent {
                    device_uuid: None,
                    kind: GpuEventKind::DeviceLost,
                });
                return;
            }
            Err(error) => {
                let _ = sender.blocking_send(GpuEvent {
                    device_uuid: None,
                    kind: GpuEventKind::MonitorFailed(error.to_string()),
                });
                return;
            }
        }
    }
}
