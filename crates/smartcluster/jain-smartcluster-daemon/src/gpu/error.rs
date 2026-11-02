use super::*;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("NVML operation failed: {0}")]
    Nvml(#[from] NvmlError),
    #[error("MPS is unavailable")]
    MpsUnavailable,
    #[error("MPS control executable and pipe directory must be absolute")]
    MpsPathNotAbsolute,
    #[error("MPS control executable is unavailable: {0}")]
    MpsControlExecutable(PathBuf),
    #[error("MPS pipe directory is unavailable: {0}")]
    MpsPipeDirectory(PathBuf),
    #[error("MPS control timeout or output bound is invalid")]
    MpsControlBounds,
    #[error("MPS pipe directory is not valid UTF-8")]
    MpsPathNotUtf8,
    #[error("failed to start MPS control executable {executable}: {source}")]
    MpsControlSpawn {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to communicate with MPS control daemon: {0}")]
    MpsControlIo(#[source] io::Error),
    #[error("MPS control command timed out after {0:?}")]
    MpsControlTimeout(Duration),
    #[error("MPS control output exceeded {0} bytes")]
    MpsControlOutputTooLarge(usize),
    #[error("MPS control command exited unsuccessfully: {0:?}")]
    MpsControlExit(Option<i32>),
    #[error("MPS control command produced unexpected stderr")]
    MpsControlStderr,
    #[error("MPS control command did not expose all configured pipes")]
    MpsControlPipe,
    #[error("MPS control output was not UTF-8")]
    MpsControlUtf8,
    #[error("MPS ps output did not contain the required Volta protocol header")]
    MpsPsHeader,
    #[error("MPS ps output contained an invalid client record")]
    MpsPsRecord,
    #[error("invalid MPS compute share: {0}; at least one whole percent is required")]
    ComputeShare(u16),
    #[error("at least one MPS device VRAM limit is required")]
    MpsVramLimitsEmpty,
    #[error("invalid MPS device selector: {0}")]
    MpsDeviceSelector(String),
    #[error("MPS VRAM limit is below the one-MiB representable minimum: {0}")]
    MpsVramLimit(u64),
    #[error("MPS server and client PIDs must be positive")]
    MpsClientIdentity,
    #[error("MPS client {client_pid} is not connected to server {server_pid}")]
    MpsClientNotFound { server_pid: u32, client_pid: u32 },
    #[error("MPS client {0} remained connected after terminate_client")]
    MpsClientStillConnected(u32),
    #[error("MPS cgroup membership changed during the no-client safety proof")]
    MpsMembershipChanged,
    #[error("cannot read MPS cgroup membership: {0}")]
    MpsCgroupMembership(String),
    #[error("safe MPS terminate_client is unavailable; refusing to signal a shared client")]
    UnsafeMpsTermination,
    #[error("cannot identify MPS client membership for host pid {pid}: {source}")]
    MpsPidIdentity {
        pid: u32,
        #[source]
        source: io::Error,
    },
    #[error("NVML event monitor setup failed: {0}")]
    EventMonitorSetup(String),
    #[error("configured MPS worker identity cannot access {path}: {reason}")]
    MpsWorkerAccess { path: PathBuf, reason: String },
}
