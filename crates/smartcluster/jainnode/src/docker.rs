use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    sync::mpsc,
};

const MAX_DOCKER_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 32 * 1024;

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

#[derive(Deserialize)]
struct CreateResponse {
    #[serde(rename = "Id")]
    id: String,
}

#[derive(Deserialize)]
struct WaitResponse {
    #[serde(rename = "StatusCode")]
    status_code: i32,
}

#[derive(Deserialize)]
struct InspectResponse {
    #[serde(rename = "State")]
    state: InspectState,
}

#[derive(Deserialize)]
struct InspectState {
    #[serde(rename = "OOMKilled")]
    oom_killed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HttpBodyFraming {
    Chunked,
    ContentLength(usize),
    Empty,
}

fn decode_http_body(header: &str, body: &[u8], status: u16) -> Result<Vec<u8>, DockerError> {
    match parse_http_body_framing(header, status)? {
        HttpBodyFraming::Chunked => decode_chunked(body),
        HttpBodyFraming::ContentLength(length) if length == body.len() => Ok(body.to_vec()),
        HttpBodyFraming::ContentLength(_) => Err(DockerError::MalformedResponse),
        HttpBodyFraming::Empty if body.is_empty() => Ok(Vec::new()),
        HttpBodyFraming::Empty => Err(DockerError::MalformedResponse),
    }
}

fn parse_http_body_framing(header: &str, status: u16) -> Result<HttpBodyFraming, DockerError> {
    let mut transfer_encoding = None;
    let mut content_length = None;
    let mut lines = header.split("\r\n");
    let status_line = lines.next().ok_or(DockerError::MalformedResponse)?;
    if status_line.is_empty() || status_line.contains(['\r', '\n']) {
        return Err(DockerError::MalformedResponse);
    }
    for line in lines {
        if line.is_empty() || line.contains(['\r', '\n']) {
            return Err(DockerError::MalformedResponse);
        }
        let (name, value) = line.split_once(':').ok_or(DockerError::MalformedResponse)?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(DockerError::MalformedResponse);
        }
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            if transfer_encoding.replace(value).is_some() {
                return Err(DockerError::MalformedResponse);
            }
        } else if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some()
                || value.is_empty()
                || !value.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(DockerError::MalformedResponse);
            }
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| DockerError::MalformedResponse)?,
            );
            if content_length.is_some_and(|length| length > MAX_DOCKER_RESPONSE) {
                return Err(DockerError::OversizedResponse);
            }
        }
    }
    let framing = match (transfer_encoding, content_length) {
        (Some(_), Some(_)) => Err(DockerError::MalformedResponse),
        (Some(value), None) if value.eq_ignore_ascii_case("chunked") => {
            Ok(HttpBodyFraming::Chunked)
        }
        (Some(_), None) => Err(DockerError::MalformedResponse),
        (None, Some(0)) if matches!(status, 204 | 304) => Ok(HttpBodyFraming::Empty),
        (None, Some(_)) if matches!(status, 204 | 304) => Err(DockerError::MalformedResponse),
        (None, Some(length)) => Ok(HttpBodyFraming::ContentLength(length)),
        (None, None) if matches!(status, 204 | 304) => Ok(HttpBodyFraming::Empty),
        (None, None) => Err(DockerError::MalformedResponse),
    }?;
    Ok(framing)
}

fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, DockerError> {
    let mut output = Vec::new();
    loop {
        let line_end = find_subsequence(body, b"\r\n").ok_or(DockerError::MalformedResponse)?;
        let size_text = std::str::from_utf8(&body[..line_end])
            .map_err(|_| DockerError::MalformedResponse)?
            .split(';')
            .next()
            .ok_or(DockerError::MalformedResponse)?;
        let size =
            usize::from_str_radix(size_text, 16).map_err(|_| DockerError::MalformedResponse)?;
        body = &body[line_end + 2..];
        if size == 0 {
            return (body == b"\r\n")
                .then_some(output)
                .ok_or(DockerError::MalformedResponse);
        }
        let framed = size.checked_add(2).ok_or(DockerError::OversizedResponse)?;
        if body.len() < framed || &body[size..framed] != b"\r\n" {
            return Err(DockerError::MalformedResponse);
        }
        output.extend_from_slice(&body[..size]);
        if output.len() > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        body = &body[size + 2..];
    }
}

async fn read_stream_header(
    stream: &mut UnixStream,
    timeout: Duration,
) -> Result<(String, Vec<u8>), DockerError> {
    let mut response = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = tokio::time::timeout(timeout, stream.read(&mut buffer))
            .await
            .map_err(|_| DockerError::Transport("Docker log header timed out".to_owned()))?
            .map_err(|error| DockerError::Transport(error.to_string()))?;
        if count == 0 {
            return Err(DockerError::MalformedResponse);
        }
        response.extend_from_slice(&buffer[..count]);
        if response.len() > MAX_HEADER_BYTES + 64 * 1024 {
            return Err(DockerError::OversizedResponse);
        }
        if let Some(header_end) = find_subsequence(&response, b"\r\n\r\n") {
            if header_end > MAX_HEADER_BYTES {
                return Err(DockerError::OversizedResponse);
            }
            let header = std::str::from_utf8(&response[..header_end])
                .map_err(|_| DockerError::MalformedResponse)?
                .to_owned();
            let status = header
                .lines()
                .next()
                .and_then(|line| line.split_ascii_whitespace().nth(1))
                .and_then(|status| status.parse::<u16>().ok())
                .ok_or(DockerError::MalformedResponse)?;
            if status != 200 {
                return Err(DockerError::Rejected(status));
            }
            return Ok((header, response[header_end + 4..].to_vec()));
        }
    }
}

fn drain_chunked(wire: &mut Vec<u8>, output: &mut Vec<u8>) -> Result<bool, DockerError> {
    loop {
        let Some(line_end) = find_subsequence(wire, b"\r\n") else {
            return Ok(false);
        };
        let size_text = std::str::from_utf8(&wire[..line_end])
            .map_err(|_| DockerError::MalformedResponse)?
            .split(';')
            .next()
            .ok_or(DockerError::MalformedResponse)?;
        let size =
            usize::from_str_radix(size_text, 16).map_err(|_| DockerError::MalformedResponse)?;
        if size == 0 {
            let terminator_end = line_end
                .checked_add(4)
                .ok_or(DockerError::OversizedResponse)?;
            if wire.len() < terminator_end {
                return Ok(false);
            }
            if wire.len() != terminator_end || &wire[line_end + 2..terminator_end] != b"\r\n" {
                return Err(DockerError::MalformedResponse);
            }
            wire.clear();
            return Ok(true);
        }
        let payload_start = line_end + 2;
        let frame_end = payload_start
            .checked_add(size)
            .ok_or(DockerError::OversizedResponse)?;
        if size > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        if wire.len() < frame_end + 2 {
            return Ok(false);
        }
        if &wire[frame_end..frame_end + 2] != b"\r\n" {
            return Err(DockerError::MalformedResponse);
        }
        output.extend_from_slice(&wire[payload_start..frame_end]);
        if output.len() > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        wire.drain(..frame_end + 2);
    }
}

fn drain_content_length(
    wire: &mut Vec<u8>,
    output: &mut Vec<u8>,
    remaining: &mut usize,
) -> Result<bool, DockerError> {
    if wire.len() > *remaining {
        return Err(DockerError::MalformedResponse);
    }
    *remaining -= wire.len();
    output.append(wire);
    Ok(*remaining == 0)
}

fn take_multiplexed_logs(body: &mut Vec<u8>) -> Result<Vec<DockerLogChunk>, DockerError> {
    let mut chunks = Vec::new();
    loop {
        if body.len() < 8 {
            return Ok(chunks);
        }
        let stream = body[0];
        if !matches!(stream, 1 | 2) || body[1..4] != [0, 0, 0] {
            return Err(DockerError::MalformedResponse);
        }
        let length = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
        if length > 64 * 1024 {
            return Err(DockerError::MalformedResponse);
        }
        if body.len() < 8 + length {
            return Ok(chunks);
        }
        chunks.push(DockerLogChunk {
            stderr: stream == 2,
            bytes: body[8..8 + length].to_vec(),
        });
        body.drain(..8 + length);
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn percent_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixListener;

    #[test]
    fn create_body_is_non_privileged_readonly_and_device_exact() {
        let spec = DockerRunSpec {
            name: "jain-run-1".to_owned(),
            image: "registry.invalid/x@sha256:00".to_owned(),
            argv: vec!["/app/run".to_owned()],
            working_directory: "work".to_owned(),
            environment: BTreeMap::new(),
            readonly_binds: vec![(PathBuf::from("/staging/input"), "/inputs/data".to_owned())],
            output_staging: PathBuf::from("/staging/output"),
            scratch_bytes: 1_000,
            cpu_millicores: 500,
            memory_bytes: 2_000,
            pids: 32,
            read_bytes: 1_000,
            write_bytes: 1_000,
            io_operations: 100,
            gpu_uuids: vec!["GPU-exact".to_owned()],
            network_name: "jain-net-1".to_owned(),
            seccomp_profile: "/etc/jain/seccomp.json".to_owned(),
            apparmor_profile: "jain-governed-rd".to_owned(),
        };
        let body = spec.create_body();
        assert_eq!(body.pointer("/HostConfig/Privileged"), Some(&json!(false)));
        assert_eq!(
            body.pointer("/HostConfig/ReadonlyRootfs"),
            Some(&json!(true))
        );
        assert_eq!(
            body.pointer("/HostConfig/NetworkMode"),
            Some(&json!("jain-net-1"))
        );
        assert_eq!(
            body.pointer("/HostConfig/DeviceRequests/0/DeviceIDs/0"),
            Some(&json!("GPU-exact"))
        );
        let encoded = body.to_string();
        assert!(!encoded.contains("docker.sock"));
        assert!(!encoded.contains("\"PidMode\":\"host\""));
        assert!(!encoded.contains("\"NetworkMode\":\"host\""));
    }

    #[test]
    fn multiplexed_frames_are_bounded_and_ordered() {
        let mut bytes = vec![1, 0, 0, 0, 0, 0, 0, 3];
        bytes.extend_from_slice(b"out");
        bytes.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 3]);
        bytes.extend_from_slice(b"err");
        let chunks = take_multiplexed_logs(&mut bytes).unwrap();
        assert_eq!(
            chunks[0],
            DockerLogChunk {
                stderr: false,
                bytes: b"out".to_vec()
            }
        );
        assert_eq!(
            chunks[1],
            DockerLogChunk {
                stderr: true,
                bytes: b"err".to_vec()
            }
        );
        let mut oversized = vec![1, 0, 0, 0, 0, 0, 0, 0];
        oversized[4..8].copy_from_slice(&(65_u32 * 1024).to_be_bytes());
        assert!(take_multiplexed_logs(&mut oversized).is_err());
    }

    #[test]
    fn streaming_chunk_and_multiplex_decoders_preserve_partial_frames() {
        let frame = [1_u8, 0, 0, 0, 0, 0, 0, 5, b'h', b'e', b'l', b'l', b'o'];
        let mut wire = format!("{:X}\r\n", frame.len()).into_bytes();
        wire.extend_from_slice(&frame[..6]);
        let mut payload = Vec::new();
        assert!(!drain_chunked(&mut wire, &mut payload).unwrap());
        assert!(payload.is_empty());
        wire.extend_from_slice(&frame[6..]);
        wire.extend_from_slice(b"\r\n0\r\n\r\n");
        assert!(drain_chunked(&mut wire, &mut payload).unwrap());
        let first = payload.drain(..6).collect::<Vec<_>>();
        let mut partial = first.clone();
        assert!(take_multiplexed_logs(&mut partial).unwrap().is_empty());
        let mut joined = first;
        joined.extend_from_slice(&payload);
        let decoded = take_multiplexed_logs(&mut joined).unwrap();
        assert_eq!(decoded[0].bytes, b"hello");
        assert!(joined.is_empty());

        let mut trailing = b"0\r\n\r\nextra".to_vec();
        assert!(drain_chunked(&mut trailing, &mut Vec::new()).is_err());
    }

    #[test]
    fn ordinary_responses_require_exact_unambiguous_framing() {
        let exact = "HTTP/1.1 200 OK\r\nContent-Length: 2";
        assert_eq!(decode_http_body(exact, b"{}", 200).unwrap(), b"{}");
        assert!(decode_http_body(exact, b"{", 200).is_err());
        assert!(decode_http_body(exact, b"{}x", 200).is_err());
        assert!(decode_http_body("HTTP/1.1 200 OK", b"{}", 200).is_err());
        assert!(decode_http_body(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2",
            b"{}",
            200,
        )
        .is_err());
        assert!(decode_http_body(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2",
            b"0\r\n\r\n",
            200,
        )
        .is_err());
        assert!(decode_http_body("HTTP/1.1 204 No Content", b"", 204).is_ok());
    }

    #[test]
    fn chunked_responses_require_complete_zero_terminator_and_no_trailing_bytes() {
        let header = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked";
        assert_eq!(
            decode_http_body(header, b"2\r\n{}\r\n0\r\n\r\n", 200).unwrap(),
            b"{}"
        );
        assert!(decode_http_body(header, b"2\r\n{}\r\n0\r\n", 200).is_err());
        assert!(decode_http_body(header, b"2\r\n{}\r\n0\r\n\r\nextra", 200).is_err());
    }

    async fn log_events_for_response(
        response: Vec<u8>,
    ) -> Vec<Result<DockerLogEvent, DockerError>> {
        let root = tempfile::tempdir().unwrap();
        let socket = root.path().join("docker.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.write_all(&response).await.unwrap();
            stream.shutdown().await.unwrap();
        });
        let engine = DockerUnixEngine::new(socket, Duration::from_secs(1)).unwrap();
        let mut receiver = engine.logs("container-1").await.unwrap();
        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        server.await.unwrap();
        events
    }

    #[tokio::test]
    async fn live_log_transport_requires_exact_unambiguous_framing() {
        for response in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nextra".to_vec(),
        ] {
            let events = log_events_for_response(response).await;
            assert!(events.iter().any(Result::is_err), "events={events:?}");
            assert!(!events
                .iter()
                .any(|event| matches!(event, Ok(DockerLogEvent::Complete))));
        }

        let multiplexed = [1, 0, 0, 0, 0, 0, 0, 2, b'o', b'k'];
        let mut exact = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n".to_vec();
        exact.extend_from_slice(&multiplexed);
        let events = log_events_for_response(exact).await;
        assert!(matches!(
            events.as_slice(),
            [
                Ok(DockerLogEvent::Chunk(DockerLogChunk { stderr: false, bytes })),
                Ok(DockerLogEvent::Complete)
            ] if bytes == b"ok"
        ));
    }
}
