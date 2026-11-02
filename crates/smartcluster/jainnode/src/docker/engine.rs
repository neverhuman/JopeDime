use super::protocol::read_stream_header;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerRunSpec {
    pub name: String,
    pub image: String,
    pub argv: Vec<String>,
    pub working_directory: String,
    pub environment: BTreeMap<String, String>,
    pub readonly_binds: Vec<(PathBuf, String)>,
    pub output_staging: PathBuf,
    pub scratch_bytes: u64,
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub pids: u32,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub io_operations: u64,
    pub gpu_uuids: Vec<String>,
    pub network_name: String,
    pub seccomp_profile: String,
    pub apparmor_profile: String,
}

impl DockerRunSpec {
    pub fn create_body(&self) -> Value {
        let mut binds = self
            .readonly_binds
            .iter()
            .map(|(source, target)| format!("{}:{target}:ro", source.display()))
            .collect::<Vec<_>>();
        binds.push(format!("{}:/outputs:rw", self.output_staging.display()));
        let devices = if self.gpu_uuids.is_empty() {
            Vec::new()
        } else {
            vec![json!({
                "Driver": "nvidia",
                "Count": 0,
                "DeviceIDs": self.gpu_uuids,
                "Capabilities": [["gpu", "compute", "utility"]],
                "Options": {}
            })]
        };
        json!({
            "Image": self.image,
            "Cmd": self.argv,
            "Env": self.environment.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>(),
            "WorkingDir": format!("/workspace/{}", self.working_directory),
            "User": "65532:65532",
            "OpenStdin": false,
            "StdinOnce": false,
            "AttachStdout": true,
            "AttachStderr": true,
            "Tty": false,
            "NetworkDisabled": false,
            "ExposedPorts": {},
            "HostConfig": {
                "Privileged": false,
                "ReadonlyRootfs": true,
                "CapAdd": [],
                "CapDrop": ["ALL"],
                "SecurityOpt": [
                    "no-new-privileges=true",
                    format!("seccomp={}", self.seccomp_profile),
                    format!("apparmor={}", self.apparmor_profile)
                ],
                "NetworkMode": self.network_name,
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "CgroupnsMode": "private",
                "ReadonlyPaths": ["/proc/asound", "/proc/acpi", "/proc/kcore", "/proc/keys", "/proc/timer_list", "/proc/scsi"],
                "MaskedPaths": ["/proc/acpi", "/proc/kcore", "/proc/keys", "/proc/latency_stats", "/proc/timer_list", "/proc/timer_stats", "/proc/sched_debug", "/proc/scsi", "/sys/firmware"],
                "Binds": binds,
                "Tmpfs": {
                    "/tmp": format!("rw,noexec,nosuid,nodev,size={}", self.scratch_bytes),
                    "/workspace": format!("rw,noexec,nosuid,nodev,size={}", self.scratch_bytes)
                },
                "Memory": self.memory_bytes,
                "MemorySwap": self.memory_bytes,
                "NanoCpus": u64::from(self.cpu_millicores) * 1_000_000,
                "PidsLimit": self.pids,
                "OomKillDisable": false,
                "PublishAllPorts": false,
                "PortBindings": {},
                "DeviceRequests": devices,
                "Ulimits": [{"Name": "nofile", "Soft": 1024, "Hard": 1024}],
                "LogConfig": {"Type": "none", "Config": {}},
                "AutoRemove": false
            },
            "Labels": {
                "org.jain.fabric": "governed-rd-v1",
                "org.jain.no-docker-socket": "true",
                "org.jain.io-read-limit": self.read_bytes.to_string(),
                "org.jain.io-write-limit": self.write_bytes.to_string(),
                "org.jain.io-operation-limit": self.io_operations.to_string()
            }
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerLogChunk {
    pub stderr: bool,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DockerLogEvent {
    Chunk(DockerLogChunk),
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitStatus {
    pub code: i32,
    pub oom_killed: bool,
}

#[derive(Debug, Error)]
pub enum DockerError {
    #[error("docker transport failed: {0}")]
    Transport(String),
    #[error("docker response was malformed")]
    MalformedResponse,
    #[error("docker rejected the request with status {0}")]
    Rejected(u16),
    #[error("docker response exceeded its bound")]
    OversizedResponse,
}

#[async_trait]
pub trait DockerEnginePort: Send + Sync {
    async fn create(&self, spec: &DockerRunSpec) -> Result<String, DockerError>;
    async fn start(&self, container_id: &str) -> Result<(), DockerError>;
    async fn logs(
        &self,
        container_id: &str,
    ) -> Result<mpsc::Receiver<Result<DockerLogEvent, DockerError>>, DockerError>;
    async fn wait(&self, container_id: &str) -> Result<ExitStatus, DockerError>;
    async fn kill(&self, container_id: &str) -> Result<(), DockerError>;
    async fn remove(&self, container_id: &str) -> Result<(), DockerError>;
}

#[derive(Clone, Debug)]
pub struct DockerUnixEngine {
    socket: PathBuf,
    timeout: Duration,
}

impl DockerUnixEngine {
    pub fn new(socket: PathBuf, timeout: Duration) -> Result<Self, DockerError> {
        if !socket.is_absolute() || timeout.is_zero() {
            return Err(DockerError::Transport("invalid Docker endpoint".to_owned()));
        }
        Ok(Self { socket, timeout })
    }

    pub(crate) async fn json_request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        accepted: &[u16],
    ) -> Result<Vec<u8>, DockerError> {
        let body = body.map_or_else(Vec::new, |body| body.to_string().into_bytes());
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let response = tokio::time::timeout(self.timeout, async {
            let mut stream = UnixStream::connect(&self.socket)
                .await
                .map_err(|error| DockerError::Transport(error.to_string()))?;
            stream
                .write_all(request.as_bytes())
                .await
                .map_err(|error| DockerError::Transport(error.to_string()))?;
            stream
                .write_all(&body)
                .await
                .map_err(|error| DockerError::Transport(error.to_string()))?;
            let mut response = Vec::new();
            stream
                .take((MAX_DOCKER_RESPONSE + 1) as u64)
                .read_to_end(&mut response)
                .await
                .map_err(|error| DockerError::Transport(error.to_string()))?;
            Ok::<_, DockerError>(response)
        })
        .await
        .map_err(|_| DockerError::Transport("Docker request timed out".to_owned()))??;
        if response.len() > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        let header_end = find_subsequence(&response, b"\r\n\r\n")
            .filter(|index| *index <= MAX_HEADER_BYTES)
            .ok_or(DockerError::MalformedResponse)?;
        let header = std::str::from_utf8(&response[..header_end])
            .map_err(|_| DockerError::MalformedResponse)?;
        let status = header
            .lines()
            .next()
            .and_then(|line| line.split_ascii_whitespace().nth(1))
            .and_then(|status| status.parse::<u16>().ok())
            .ok_or(DockerError::MalformedResponse)?;
        if !accepted.contains(&status) {
            return Err(DockerError::Rejected(status));
        }
        let payload = &response[header_end + 4..];
        decode_http_body(header, payload, status)
    }
}

#[async_trait]
impl DockerEnginePort for DockerUnixEngine {
    async fn create(&self, spec: &DockerRunSpec) -> Result<String, DockerError> {
        let path = format!(
            "/v1.47/containers/create?name={}",
            percent_encode(&spec.name)
        );
        let payload = self
            .json_request("POST", &path, Some(&spec.create_body()), &[201])
            .await?;
        serde_json::from_slice::<CreateResponse>(&payload)
            .map_err(|_| DockerError::MalformedResponse)
            .and_then(|response| {
                (!response.id.is_empty())
                    .then_some(response.id)
                    .ok_or(DockerError::MalformedResponse)
            })
    }

    async fn start(&self, container_id: &str) -> Result<(), DockerError> {
        self.json_request(
            "POST",
            &format!("/v1.47/containers/{}/start", percent_encode(container_id)),
            None,
            &[204, 304],
        )
        .await
        .map(|_| ())
    }

    async fn logs(
        &self,
        container_id: &str,
    ) -> Result<mpsc::Receiver<Result<DockerLogEvent, DockerError>>, DockerError> {
        let socket = self.socket.clone();
        let id = percent_encode(container_id);
        let timeout = self.timeout;
        let (sender, receiver) = mpsc::channel(32);
        tokio::spawn(async move {
            let chunk_sender = sender.clone();
            let result: Result<(), DockerError> = async move {
                let mut stream = tokio::time::timeout(timeout, UnixStream::connect(socket))
                    .await
                    .map_err(|_| DockerError::Transport("Docker log connect timed out".to_owned()))?
                    .map_err(|error| DockerError::Transport(error.to_string()))?;
                let request = format!(
                    "GET /v1.47/containers/{id}/logs?follow=1&stdout=1&stderr=1 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                );
                tokio::time::timeout(timeout, stream.write_all(request.as_bytes()))
                    .await
                    .map_err(|_| DockerError::Transport("Docker log request timed out".to_owned()))?
                    .map_err(|error| DockerError::Transport(error.to_string()))?;
                let (header, mut wire) = read_stream_header(&mut stream, timeout).await?;
                let framing = parse_http_body_framing(&header, 200)?;
                let mut content_length_remaining = match framing {
                    HttpBodyFraming::ContentLength(length) => length,
                    HttpBodyFraming::Chunked | HttpBodyFraming::Empty => 0,
                };
                let mut multiplexed = Vec::new();
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let complete = match framing {
                        HttpBodyFraming::Chunked => {
                            drain_chunked(&mut wire, &mut multiplexed)?
                        }
                        HttpBodyFraming::ContentLength(_) => drain_content_length(
                            &mut wire,
                            &mut multiplexed,
                            &mut content_length_remaining,
                        )?,
                        HttpBodyFraming::Empty => true,
                    };
                    for chunk in take_multiplexed_logs(&mut multiplexed)? {
                        if chunk_sender
                            .send(Ok(DockerLogEvent::Chunk(chunk)))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }
                    }
                    if complete {
                        return multiplexed
                            .is_empty()
                            .then_some(())
                            .ok_or(DockerError::MalformedResponse);
                    }
                    let count = tokio::time::timeout(timeout, stream.read(&mut buffer))
                        .await
                        .map_err(|_| {
                            DockerError::Transport("Docker log stream timed out".to_owned())
                        })?
                        .map_err(|error| DockerError::Transport(error.to_string()))?;
                    if count == 0 {
                        return Err(DockerError::MalformedResponse);
                    }
                    wire.extend_from_slice(&buffer[..count]);
                    if wire.len() > MAX_DOCKER_RESPONSE {
                        return Err(DockerError::OversizedResponse);
                    }
                }
            }
            .await;
            let terminal = match result {
                Ok(()) => Ok(DockerLogEvent::Complete),
                Err(error) => Err(error),
            };
            let _ = sender.send(terminal).await;
        });
        Ok(receiver)
    }

    async fn wait(&self, container_id: &str) -> Result<ExitStatus, DockerError> {
        let payload = self
            .json_request(
                "POST",
                &format!(
                    "/v1.47/containers/{}/wait?condition=not-running",
                    percent_encode(container_id)
                ),
                None,
                &[200],
            )
            .await?;
        let response: WaitResponse =
            serde_json::from_slice(&payload).map_err(|_| DockerError::MalformedResponse)?;
        let inspection = self
            .json_request(
                "GET",
                &format!("/v1.47/containers/{}/json", percent_encode(container_id)),
                None,
                &[200],
            )
            .await?;
        let inspection: InspectResponse =
            serde_json::from_slice(&inspection).map_err(|_| DockerError::MalformedResponse)?;
        Ok(ExitStatus {
            code: response.status_code,
            oom_killed: inspection.state.oom_killed,
        })
    }

    async fn kill(&self, container_id: &str) -> Result<(), DockerError> {
        self.json_request(
            "POST",
            &format!(
                "/v1.47/containers/{}/kill?signal=SIGKILL",
                percent_encode(container_id)
            ),
            None,
            &[204, 409],
        )
        .await
        .map(|_| ())
    }

    async fn remove(&self, container_id: &str) -> Result<(), DockerError> {
        self.json_request(
            "DELETE",
            &format!(
                "/v1.47/containers/{}?force=1&v=1",
                percent_encode(container_id)
            ),
            None,
            &[204, 404],
        )
        .await
        .map(|_| ())
    }
}
