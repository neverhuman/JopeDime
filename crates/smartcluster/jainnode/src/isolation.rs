use std::{
    fs,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
    time::Duration,
};

use async_trait::async_trait;
use jain_smartcluster_fabric::{sha256, Digest32};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
};

use crate::{
    ContainerIsolationPort, ContainerIsolationReceipt, DockerRunSpec, ExecutionError,
    IsolationEvidence,
};

const MAX_BROKER_FRAME_BYTES: usize = 64 * 1024;
const QUALIFICATION_DOMAIN: &[u8] = b"jain.node.linux-isolation-qualification.v1\0";
const BROKER_REQUEST_DOMAIN: &[u8] = b"jain.node.isolation-broker-request.v1\0";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LinuxIsolationReceipt {
    pub schema_version: u16,
    pub node_id: String,
    pub kernel_release: String,
    pub namespaces: bool,
    pub device_filter: bool,
    pub seccomp: bool,
    pub apparmor_or_landlock: bool,
    pub descendant_fencing: bool,
    pub network_isolation: bool,
    pub metadata_and_control_plane_denied: bool,
    pub io_limits: bool,
    pub seccomp_profile: String,
    pub apparmor_profile: String,
    pub cgroup_root: String,
    pub evidence_digest: Digest32,
}

impl LinuxIsolationReceipt {
    fn signed_bytes(&self) -> Result<Vec<u8>, ExecutionError> {
        let mut unsigned = self.clone();
        unsigned.evidence_digest = [0; 32];
        let payload =
            postcard::to_stdvec(&unsigned).map_err(|_| ExecutionError::UnsupportedHost)?;
        let mut bytes = Vec::with_capacity(QUALIFICATION_DOMAIN.len() + payload.len());
        bytes.extend_from_slice(QUALIFICATION_DOMAIN);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    fn validate(&self) -> Result<(), ExecutionError> {
        let cgroup_root = Path::new(&self.cgroup_root);
        if self.schema_version != 1
            || self.node_id.is_empty()
            || self.kernel_release.is_empty()
            || !self.namespaces
            || !self.device_filter
            || !self.seccomp
            || !self.apparmor_or_landlock
            || !self.descendant_fencing
            || !self.network_isolation
            || !self.metadata_and_control_plane_denied
            || !self.io_limits
            || !Path::new(&self.seccomp_profile).is_absolute()
            || self.seccomp_profile.contains("..")
            || self.apparmor_profile.is_empty()
            || !cgroup_root.is_absolute()
            || !cgroup_root.join("cgroup.controllers").is_file()
            || !cgroup_root.join("cgroup.kill").is_file()
            || sha256(&self.signed_bytes()?) != self.evidence_digest
        {
            return Err(ExecutionError::UnsupportedHost);
        }
        let controllers = fs::read_to_string(cgroup_root.join("cgroup.controllers"))
            .map_err(|_| ExecutionError::UnsupportedHost)?;
        if ["cpu", "io", "memory", "pids"].iter().any(|controller| {
            !controllers
                .split_ascii_whitespace()
                .any(|item| item == *controller)
        }) {
            return Err(ExecutionError::UnsupportedHost);
        }
        Ok(())
    }
}

impl IsolationEvidence {
    /// Load root-issued host qualification. User-writable or symlinked receipts are rejected.
    pub fn from_root_receipt(path: &Path) -> Result<Self, ExecutionError> {
        validate_root_file(path)?;
        let bytes = fs::read(path).map_err(|_| ExecutionError::UnsupportedHost)?;
        if bytes.len() > MAX_BROKER_FRAME_BYTES {
            return Err(ExecutionError::UnsupportedHost);
        }
        let receipt: LinuxIsolationReceipt =
            serde_json::from_slice(&bytes).map_err(|_| ExecutionError::UnsupportedHost)?;
        receipt.validate()?;
        validate_root_file(Path::new(&receipt.seccomp_profile))?;
        Ok(Self {
            digest: receipt.evidence_digest,
            namespaces: receipt.namespaces,
            device_filter: receipt.device_filter,
            seccomp: receipt.seccomp,
            apparmor_or_landlock: receipt.apparmor_or_landlock,
            descendant_fencing: receipt.descendant_fencing,
            network_isolation: receipt.network_isolation,
            metadata_and_control_plane_denied: receipt.metadata_and_control_plane_denied,
            io_limits: receipt.io_limits,
            seccomp_profile: receipt.seccomp_profile,
            apparmor_profile: receipt.apparmor_profile,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RootBrokerIsolation {
    socket: PathBuf,
    timeout: Duration,
}

impl RootBrokerIsolation {
    pub fn new(socket: PathBuf, timeout: Duration) -> Result<Self, ExecutionError> {
        if !socket.is_absolute() || timeout.is_zero() {
            return Err(ExecutionError::UnsupportedHost);
        }
        let metadata =
            fs::symlink_metadata(&socket).map_err(|_| ExecutionError::UnsupportedHost)?;
        if !metadata.file_type().is_socket() || metadata.uid() != 0 || metadata.mode() & 0o022 != 0
        {
            return Err(ExecutionError::UnsupportedHost);
        }
        Ok(Self { socket, timeout })
    }
}

#[derive(Serialize)]
struct BrokerRequest<'a> {
    container_id: &'a str,
    qualification_digest: Digest32,
    work_unit_digest: Digest32,
    placement_digest: Digest32,
    image: &'a str,
    argv_digest: Digest32,
    environment_digest: Digest32,
    cpu_millicores: u32,
    memory_bytes: u64,
    pids: u32,
    read_bytes: u64,
    write_bytes: u64,
    io_operations: u64,
    gpu_uuids: &'a [String],
    network_name: &'a str,
}

#[derive(Deserialize)]
struct BrokerResponse {
    request_digest: Digest32,
    receipt: ContainerIsolationReceipt,
}

#[async_trait]
impl ContainerIsolationPort for RootBrokerIsolation {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError> {
        let argv = postcard::to_stdvec(&spec.argv).map_err(|_| ExecutionError::UnsupportedHost)?;
        let environment =
            postcard::to_stdvec(&spec.environment).map_err(|_| ExecutionError::UnsupportedHost)?;
        let request = BrokerRequest {
            container_id,
            qualification_digest,
            work_unit_digest,
            placement_digest,
            image: &spec.image,
            argv_digest: sha256(&argv),
            environment_digest: sha256(&environment),
            cpu_millicores: spec.cpu_millicores,
            memory_bytes: spec.memory_bytes,
            pids: spec.pids,
            read_bytes: spec.read_bytes,
            write_bytes: spec.write_bytes,
            io_operations: spec.io_operations,
            gpu_uuids: &spec.gpu_uuids,
            network_name: &spec.network_name,
        };
        let payload = postcard::to_stdvec(&request).map_err(|_| ExecutionError::UnsupportedHost)?;
        let mut request_bytes = Vec::with_capacity(BROKER_REQUEST_DOMAIN.len() + payload.len());
        request_bytes.extend_from_slice(BROKER_REQUEST_DOMAIN);
        request_bytes.extend_from_slice(&payload);
        if request_bytes.len() > MAX_BROKER_FRAME_BYTES {
            return Err(ExecutionError::UnsupportedHost);
        }
        let request_digest = sha256(&request_bytes);
        let response = tokio::time::timeout(self.timeout, async {
            let mut stream = UnixStream::connect(&self.socket)
                .await
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?;
            let credentials = getsockopt(&stream, PeerCredentials)
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?;
            if credentials.uid() != 0 {
                return Err(ExecutionError::UnsupportedHost);
            }
            stream
                .write_u32(request_bytes.len() as u32)
                .await
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?;
            stream
                .write_all(&request_bytes)
                .await
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?;
            let response_length = stream
                .read_u32()
                .await
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?
                as usize;
            if response_length == 0 || response_length > MAX_BROKER_FRAME_BYTES {
                return Err(ExecutionError::UnsupportedHost);
            }
            let mut response = vec![0; response_length];
            stream
                .read_exact(&mut response)
                .await
                .map_err(|error| ExecutionError::IsolationBroker(error.to_string()))?;
            postcard::from_bytes::<BrokerResponse>(&response)
                .map_err(|_| ExecutionError::UnsupportedHost)
        })
        .await
        .map_err(|_| ExecutionError::IsolationBroker("broker request timed out".to_owned()))??;
        if response.request_digest != request_digest {
            return Err(ExecutionError::UnsupportedHost);
        }
        Ok(response.receipt)
    }
}

fn validate_root_file(path: &Path) -> Result<(), ExecutionError> {
    if !path.is_absolute() {
        return Err(ExecutionError::UnsupportedHost);
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| ExecutionError::UnsupportedHost)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.uid() != 0
        || metadata.mode() & 0o022 != 0
        || metadata.nlink() != 1
        || path
            .canonicalize()
            .map_err(|_| ExecutionError::UnsupportedHost)?
            != path
    {
        return Err(ExecutionError::UnsupportedHost);
    }
    Ok(())
}
