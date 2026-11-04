use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MpsCapability {
    pub control_executable: Option<PathBuf>,
    pub pipe_directory: Option<PathBuf>,
    pub compute_limit: bool,
    pub vram_limit: bool,
    pub safe_client_termination: bool,
    pub shareable: bool,
    pub unavailable_reason: Option<String>,
}

impl MpsCapability {
    pub fn detect() -> Self {
        Self {
            control_executable: None,
            pipe_directory: None,
            compute_limit: false,
            vram_limit: false,
            safe_client_termination: false,
            shareable: false,
            unavailable_reason: Some(
                "MPS sharing requires an explicitly configured and successfully probed controller"
                    .to_owned(),
            ),
        }
    }

    pub fn unavailable(config: &MpsConfig, reason: impl Into<String>) -> Self {
        Self {
            control_executable: config.control_executable.clone(),
            pipe_directory: config.pipe_directory.clone(),
            compute_limit: false,
            vram_limit: false,
            safe_client_termination: false,
            shareable: false,
            unavailable_reason: Some(reason.into()),
        }
    }

    pub fn disable(&mut self, reason: impl Into<String>) {
        self.compute_limit = false;
        self.vram_limit = false;
        self.safe_client_termination = false;
        self.shareable = false;
        self.unavailable_reason = Some(reason.into());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpsClientLimits {
    pub compute_share_per_mille: u16,
    pub vram_bytes_by_device: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpsClient {
    pub pid: u32,
    pub client_id: String,
    pub server_pid: u32,
    pub device: String,
    pub namespace_pid: u64,
    pub command: String,
}

/// Matches an MPS-reported client to host cgroup PIDs, including PID namespaces.
pub fn mps_client_matches_pids(
    client: &MpsClient,
    host_pids: &BTreeSet<u32>,
) -> Result<bool, GpuError> {
    for host_pid in host_pids {
        if *host_pid == client.pid {
            return Ok(true);
        }
        let namespace = match pid_namespace_inode(*host_pid) {
            Ok(namespace) => namespace,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(GpuError::MpsPidIdentity {
                    pid: *host_pid,
                    source,
                });
            }
        };
        if namespace == client.namespace_pid {
            let namespace_pids =
                namespace_pids(*host_pid).map_err(|source| GpuError::MpsPidIdentity {
                    pid: *host_pid,
                    source,
                })?;
            if namespace_pids.contains(&client.pid) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn pid_namespace_inode(pid: u32) -> io::Result<u64> {
    let target = std::fs::read_link(format!("/proc/{pid}/ns/pid"))?;
    let target = target.to_string_lossy();
    target
        .strip_prefix("pid:[")
        .and_then(|value| value.strip_suffix(']'))
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid PID namespace link"))
}

fn namespace_pids(pid: u32) -> io::Result<BTreeSet<u32>> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
    let values = status
        .lines()
        .find_map(|line| line.strip_prefix("NSpid:"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing NSpid status"))?;
    values
        .split_whitespace()
        .map(|value| {
            value.parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid NSpid status value")
            })
        })
        .collect()
}

/// A controller is only constructed after a bounded `ps` exchange proves that
/// the configured daemon exposes the Volta MPS client/server identity protocol.
#[derive(Clone, Debug)]
pub struct MpsController {
    pub(crate) control_executable: PathBuf,
    pub(crate) pipe_directory: PathBuf,
    pub(crate) command_timeout: Duration,
    pub(crate) max_control_output_bytes: usize,
}

impl MpsController {
    pub async fn probe(config: &MpsConfig) -> Result<Self, GpuError> {
        let (control_executable, pipe_directory) =
            match (&config.control_executable, &config.pipe_directory) {
                (Some(executable), Some(directory)) => (executable.clone(), directory.clone()),
                _ => return Err(GpuError::MpsUnavailable),
            };
        if !control_executable.is_absolute() || !pipe_directory.is_absolute() {
            return Err(GpuError::MpsPathNotAbsolute);
        }
        if !control_executable.is_file() {
            return Err(GpuError::MpsControlExecutable(control_executable));
        }
        if !pipe_directory.is_dir() {
            return Err(GpuError::MpsPipeDirectory(pipe_directory));
        }
        if config.command_timeout_ms == 0
            || config.max_control_output_bytes == 0
            || config.max_control_output_bytes > 16 * 1024 * 1024
        {
            return Err(GpuError::MpsControlBounds);
        }
        let controller = Self {
            control_executable,
            pipe_directory,
            command_timeout: config.command_timeout(),
            max_control_output_bytes: config.max_control_output_bytes,
        };

        // `ps` is a Volta MPS command and its SERVER column is the identity
        // required by safe `terminate_client`. The two per-client limits are
        // only advertised together with this successfully probed protocol.
        controller.clients().await?;
        Ok(controller)
    }

    pub fn capability(&self) -> MpsCapability {
        MpsCapability {
            control_executable: Some(self.control_executable.clone()),
            pipe_directory: Some(self.pipe_directory.clone()),
            compute_limit: true,
            vram_limit: true,
            safe_client_termination: true,
            shareable: true,
            unavailable_reason: None,
        }
    }

    pub fn validate_worker_access(&self, uid: u32, gid: u32) -> Result<(), GpuError> {
        let executable = std::fs::metadata(&self.control_executable).map_err(|source| {
            GpuError::MpsWorkerAccess {
                path: self.control_executable.clone(),
                reason: source.to_string(),
            }
        })?;
        if !mode_allows(&executable, uid, gid, 0o5) {
            return Err(GpuError::MpsWorkerAccess {
                path: self.control_executable.clone(),
                reason: format!("uid {uid} gid {gid} lacks read/execute access"),
            });
        }
        let directory = std::fs::metadata(&self.pipe_directory).map_err(|source| {
            GpuError::MpsWorkerAccess {
                path: self.pipe_directory.clone(),
                reason: source.to_string(),
            }
        })?;
        if !mode_allows(&directory, uid, gid, 0o7) {
            return Err(GpuError::MpsWorkerAccess {
                path: self.pipe_directory.clone(),
                reason: format!("uid {uid} gid {gid} lacks read/write/search access"),
            });
        }
        Ok(())
    }

    pub const fn command_timeout(&self) -> Duration {
        self.command_timeout
    }

    pub fn client_environment(
        &self,
        limits: &MpsClientLimits,
    ) -> Result<BTreeMap<String, String>, GpuError> {
        let compute_percent = compute_percent(limits.compute_share_per_mille)?;
        let vram_limits = format_vram_limits(&limits.vram_bytes_by_device)?;
        let pipe_directory = self
            .pipe_directory
            .to_str()
            .ok_or(GpuError::MpsPathNotUtf8)?;
        Ok([
            (
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE".to_owned(),
                compute_percent.to_string(),
            ),
            ("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT".to_owned(), vram_limits),
            (
                "CUDA_MPS_PIPE_DIRECTORY".to_owned(),
                pipe_directory.to_owned(),
            ),
        ]
        .into_iter()
        .collect())
    }

    pub async fn clients(&self) -> Result<Vec<MpsClient>, GpuError> {
        let output = self.run_control("ps\n").await?;
        parse_ps(&output)
    }

    pub async fn terminate_client(&self, server_pid: u32, client_pid: u32) -> Result<(), GpuError> {
        if server_pid == 0 || client_pid == 0 {
            return Err(GpuError::MpsClientIdentity);
        }
        let clients = self.clients().await?;
        if !clients
            .iter()
            .any(|client| client.server_pid == server_pid && client.pid == client_pid)
        {
            return Err(GpuError::MpsClientNotFound {
                server_pid,
                client_pid,
            });
        }
        let output = self
            .run_control(&format!("terminate_client {server_pid} {client_pid}\n"))
            .await?;
        parse_termination_response(&output)
    }

    async fn run_control(&self, input: &str) -> Result<String, GpuError> {
        let mut child = Command::new(&self.control_executable)
            .env_clear()
            .env("CUDA_MPS_PIPE_DIRECTORY", &self.pipe_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| GpuError::MpsControlSpawn {
                executable: self.control_executable.clone(),
                source,
            })?;
        let mut stdin = child.stdin.take().ok_or(GpuError::MpsControlPipe)?;
        let stdout = child.stdout.take().ok_or(GpuError::MpsControlPipe)?;
        let stderr = child.stderr.take().ok_or(GpuError::MpsControlPipe)?;
        let bound = self.max_control_output_bytes;
        let operation = async {
            stdin.write_all(input.as_bytes()).await?;
            stdin.shutdown().await?;
            let (status, stdout, stderr) = tokio::join!(
                child.wait(),
                read_bounded(stdout, bound),
                read_bounded(stderr, bound)
            );
            Ok::<_, io::Error>((status?, stdout?, stderr?))
        };
        let (status, stdout, stderr) =
            match tokio::time::timeout(self.command_timeout, operation).await {
                Ok(result) => result.map_err(GpuError::MpsControlIo)?,
                Err(_) => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err(GpuError::MpsControlTimeout(self.command_timeout));
                }
            };
        if stdout.len() > bound || stderr.len() > bound {
            return Err(GpuError::MpsControlOutputTooLarge(bound));
        }
        if !status.success() {
            return Err(GpuError::MpsControlExit(status.code()));
        }
        if !stderr.iter().all(u8::is_ascii_whitespace) {
            return Err(GpuError::MpsControlStderr);
        }
        String::from_utf8(stdout).map_err(|_| GpuError::MpsControlUtf8)
    }
}

fn mode_allows(metadata: &std::fs::Metadata, uid: u32, gid: u32, required: u32) -> bool {
    if uid == 0 {
        return true;
    }
    let granted = if metadata.uid() == uid {
        (metadata.mode() >> 6) & 0o7
    } else if metadata.gid() == gid {
        (metadata.mode() >> 3) & 0o7
    } else {
        metadata.mode() & 0o7
    };
    granted & required == required
}

async fn read_bounded(reader: impl AsyncRead + Unpin, max_bytes: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    reader
        .take(
            u64::try_from(max_bytes)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut output)
        .await?;
    Ok(output)
}

pub(crate) fn compute_percent(compute_share_per_mille: u16) -> Result<u16, GpuError> {
    if !(10..=1_000).contains(&compute_share_per_mille) {
        return Err(GpuError::ComputeShare(compute_share_per_mille));
    }
    // MPS accepts whole percentages. Rounding down keeps the actual limit at
    // or below the scheduler's per-mille budget.
    Ok(compute_share_per_mille / 10)
}

pub(crate) fn format_vram_limits(limits: &BTreeMap<String, u64>) -> Result<String, GpuError> {
    if limits.is_empty() {
        return Err(GpuError::MpsVramLimitsEmpty);
    }
    limits
        .iter()
        .map(|(device, bytes)| {
            if !valid_device_selector(device) {
                return Err(GpuError::MpsDeviceSelector(device.clone()));
            }
            let mebibytes = bytes / MEBIBYTE;
            if mebibytes == 0 {
                return Err(GpuError::MpsVramLimit(*bytes));
            }
            Ok(format!("{device}={mebibytes}M"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join(","))
}

fn valid_device_selector(device: &str) -> bool {
    let ordinal = !device.is_empty() && device.bytes().all(|byte| byte.is_ascii_digit());
    let uuid = device.strip_prefix("GPU-").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.len() <= 124
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() || byte == b'-')
    });
    ordinal || uuid
}

pub(crate) fn parse_ps(output: &str) -> Result<Vec<MpsClient>, GpuError> {
    let mut lines = output.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<_> = lines
        .next()
        .ok_or(GpuError::MpsPsHeader)?
        .split_whitespace()
        .collect();
    if header.len() < 6 || header[..6] != ["PID", "ID", "SERVER", "DEVICE", "NAMESPACE", "COMMAND"]
    {
        return Err(GpuError::MpsPsHeader);
    }
    lines
        .map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 6 {
                return Err(GpuError::MpsPsRecord);
            }
            Ok(MpsClient {
                pid: fields[0].parse().map_err(|_| GpuError::MpsPsRecord)?,
                client_id: fields[1].to_owned(),
                server_pid: fields[2].parse().map_err(|_| GpuError::MpsPsRecord)?,
                device: fields[3].to_owned(),
                namespace_pid: fields[4].parse().map_err(|_| GpuError::MpsPsRecord)?,
                command: fields[5..].join(" "),
            })
        })
        .collect()
}

pub(crate) fn parse_termination_response(output: &str) -> Result<(), GpuError> {
    if output.trim() == "0" {
        Ok(())
    } else {
        Err(GpuError::UnsafeMpsTermination)
    }
}
