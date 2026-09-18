use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    net::SocketAddr,
    os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use jain_smartcluster_fabric::{valid_identifier, Digest32};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ControlError, ExecutionError, IsolationEvidence, MtlsClientConfig, NodeProfile,
    NodeRuntimeIdentity, StorageConfig, StorageConfigError, WorkerProvision,
};

const ROLE_SCHEMA_V1: &str = "jain.appliance-role/v1";
const ROLE_SCHEMA_V2: &str = "jain.appliance-role/v2";
const RELEASE_ID: &str = "9.0.0-appliance.1";
const ROLE_SIGNATURE_DOMAIN_V1: &[u8] = b"jain.node.appliance-role.v1\0";
const ROLE_SIGNATURE_DOMAIN_V2: &[u8] = b"jain.node.appliance-role.v2\0";
const MAX_ROLE_ENVELOPE_BYTES: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApplianceRole {
    AllInOne,
    EdgeHub,
    ComputeNode,
    FollowerComputeNode,
    WitnessStorageNode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReleaseStatus {
    Candidate,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RoleServices {
    jainnode: bool,
    scqd: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RoleControl {
    hub_address: SocketAddr,
    server_name: String,
    ca_certificate: PathBuf,
    client_certificate: PathBuf,
    client_private_key: PathBuf,
    connect_timeout_ms: u64,
    frame_timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodeRoleEnvelope {
    schema: String,
    release: String,
    status: ReleaseStatus,
    formal_ga: bool,
    image_digest: String,
    role: ApplianceRole,
    node_id: String,
    session_id: String,
    profile: NodeProfile,
    provision: Option<WorkerProvision>,
    compute_slots: u16,
    storage_enabled: bool,
    services: RoleServices,
    state_root: PathBuf,
    qualification_receipt: PathBuf,
    control: RoleControl,
    node_key_id: String,
    node_public_key: String,
    node_private_key: PathBuf,
    hub_issuer: String,
    hub_public_key: String,
    image_issuer: String,
    image_public_key: String,
    max_inflight: u32,
    #[serde(skip)]
    v2_bindings: Option<RoleV2Bindings>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoleV2Bindings {
    storage_config_path: PathBuf,
    storage_config_sha256: String,
    docker_socket: PathBuf,
    isolation_broker_socket: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignedNodeRoleEnvelope {
    payload: NodeRoleEnvelope,
    #[serde(default)]
    bindings: Option<RoleV2Bindings>,
    signature: String,
}

#[derive(Serialize)]
struct RoleV2SignedPayload<'a> {
    payload: &'a NodeRoleEnvelope,
    bindings: &'a RoleV2Bindings,
}

impl NodeRoleEnvelope {
    pub fn open(path: &Path, trusted_hub_public_key: Digest32) -> Result<Self, RoleError> {
        if trusted_hub_public_key == [0; 32] {
            return Err(RoleError::InvalidEnvelope);
        }
        validate_absolute_path(path)?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
            .open(path)
            .map_err(RoleError::Io)?;
        validate_envelope_file(path, &file)?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(MAX_ROLE_ENVELOPE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(RoleError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_ROLE_ENVELOPE_BYTES {
            return Err(RoleError::InvalidEnvelope);
        }
        let signed: SignedNodeRoleEnvelope =
            serde_json::from_slice(&bytes).map_err(RoleError::Json)?;
        let SignedNodeRoleEnvelope {
            mut payload,
            bindings,
            signature,
        } = signed;
        payload.validate_shape(bindings.as_ref())?;
        payload.verify_signature(&signature, trusted_hub_public_key, bindings.as_ref())?;
        payload.v2_bindings = bindings;
        Ok(payload)
    }

    pub fn role(&self) -> ApplianceRole {
        self.role
    }

    pub fn profile(&self) -> NodeProfile {
        self.profile
    }

    pub fn provision(&self) -> Option<WorkerProvision> {
        self.provision
    }

    pub fn signing_bytes(&self) -> Result<Vec<u8>, RoleError> {
        if self.schema != ROLE_SCHEMA_V1 {
            return Err(RoleError::InvalidEnvelope);
        }
        let payload = postcard::to_stdvec(self).map_err(|_| RoleError::InvalidEnvelope)?;
        let mut bytes = Vec::with_capacity(ROLE_SIGNATURE_DOMAIN_V1.len() + payload.len());
        bytes.extend_from_slice(ROLE_SIGNATURE_DOMAIN_V1);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn signing_bytes_v2(&self, bindings: &RoleV2Bindings) -> Result<Vec<u8>, RoleError> {
        if self.schema != ROLE_SCHEMA_V2 {
            return Err(RoleError::InvalidEnvelope);
        }
        bindings.validate_shape()?;
        let payload = postcard::to_stdvec(&RoleV2SignedPayload {
            payload: self,
            bindings,
        })
        .map_err(|_| RoleError::InvalidEnvelope)?;
        let mut bytes = Vec::with_capacity(ROLE_SIGNATURE_DOMAIN_V2.len() + payload.len());
        bytes.extend_from_slice(ROLE_SIGNATURE_DOMAIN_V2);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn execution_bindings(&self) -> Result<&RoleV2Bindings, RoleError> {
        if self.schema != ROLE_SCHEMA_V2 {
            return Err(RoleError::RoleV1CannotExecute);
        }
        self.v2_bindings.as_ref().ok_or(RoleError::InvalidEnvelope)
    }

    pub fn validate_bootstrap(&self) -> Result<Digest32, RoleError> {
        validate_state_root(&self.state_root)?;
        let control = self.control_config()?;
        control.validate()?;
        let _identity = self.runtime_identity()?;
        let isolation = IsolationEvidence::from_root_receipt(&self.qualification_receipt)?;
        if self.schema == ROLE_SCHEMA_V2 {
            let bindings = self.execution_bindings()?;
            let storage = bindings.open_storage_config()?;
            if storage.node_id() != self.node_id {
                return Err(RoleError::InvalidEnvelope);
            }
            validate_bound_socket(bindings.docker_socket())?;
            validate_bound_socket(bindings.isolation_broker_socket())?;
            storage.validate_host_topology()?;
        }
        Ok(isolation.digest())
    }

    pub fn runtime_identity(&self) -> Result<NodeRuntimeIdentity, RoleError> {
        let signing_key = read_signing_key(&self.node_private_key)?;
        if signing_key.verifying_key().to_bytes() != parse_digest(&self.node_public_key)? {
            return Err(RoleError::InvalidEnvelope);
        }
        NodeRuntimeIdentity::new(
            self.node_id.clone(),
            self.session_id.clone(),
            self.node_key_id.clone(),
            self.hub_issuer.clone(),
            parse_digest(&self.hub_public_key)?,
            self.image_issuer.clone(),
            parse_digest(&self.image_public_key)?,
            signing_key,
        )
        .map_err(|error| RoleError::Execution(ExecutionError::Assignment(error)))
    }

    fn verify_signature(
        &self,
        signature: &str,
        trusted_hub_public_key: Digest32,
        bindings: Option<&RoleV2Bindings>,
    ) -> Result<(), RoleError> {
        if parse_hex::<32>(&self.hub_public_key)? != trusted_hub_public_key {
            return Err(RoleError::InvalidEnvelope);
        }
        let signature = Signature::from_bytes(&parse_hex::<64>(signature)?);
        VerifyingKey::from_bytes(&trusted_hub_public_key)
            .map_err(|_| RoleError::InvalidEnvelope)?
            .verify_strict(&self.signature_bytes(bindings)?, &signature)
            .map_err(|_| RoleError::InvalidEnvelope)
    }

    fn signature_bytes(&self, bindings: Option<&RoleV2Bindings>) -> Result<Vec<u8>, RoleError> {
        match (self.schema.as_str(), bindings) {
            (ROLE_SCHEMA_V1, None) => self.signing_bytes(),
            (ROLE_SCHEMA_V2, Some(bindings)) => self.signing_bytes_v2(bindings),
            _ => Err(RoleError::InvalidEnvelope),
        }
    }

    fn validate_shape(&self, bindings: Option<&RoleV2Bindings>) -> Result<(), RoleError> {
        let schema_is_valid = match (self.schema.as_str(), bindings) {
            (ROLE_SCHEMA_V1, None) => true,
            (ROLE_SCHEMA_V2, Some(bindings)) => bindings.validate_shape().is_ok(),
            _ => false,
        };
        if !schema_is_valid
            || self.release != RELEASE_ID
            || self.status != ReleaseStatus::Candidate
            || self.formal_ga
            || !valid_image_digest(&self.image_digest)
            || !valid_identifier(&self.node_id)
            || !valid_identifier(&self.session_id)
            || !valid_identifier(&self.hub_issuer)
            || !valid_identifier(&self.image_issuer)
            || parse_digest(&self.hub_public_key)? == [0; 32]
            || parse_digest(&self.image_public_key)? == [0; 32]
            || !self.services.jainnode
            || self.services.scqd
            || !valid_config_path(&self.state_root)
            || !valid_config_path(&self.qualification_receipt)
            || !valid_config_path(&self.control.ca_certificate)
            || !valid_config_path(&self.control.client_certificate)
            || !valid_config_path(&self.control.client_private_key)
            || !valid_identifier(&self.node_key_id)
            || parse_digest(&self.node_public_key)? == [0; 32]
            || !valid_config_path(&self.node_private_key)
            || self.control.server_name.is_empty()
            || self.control.connect_timeout_ms == 0
            || self.control.connect_timeout_ms > 300_000
            || self.control.frame_timeout_ms == 0
            || self.control.frame_timeout_ms > 300_000
            || !self.role_shape_is_valid()
        {
            return Err(RoleError::InvalidEnvelope);
        }
        Ok(())
    }

    fn role_shape_is_valid(&self) -> bool {
        match self.role {
            ApplianceRole::EdgeHub => {
                self.profile == NodeProfile::StorageOnly
                    && self.provision.is_none()
                    && self.compute_slots == 0
                    && self.max_inflight == 0
                    && self.storage_enabled
            }
            ApplianceRole::ComputeNode | ApplianceRole::FollowerComputeNode => {
                self.profile != NodeProfile::StorageOnly
                    && self.provision.is_some()
                    && self.compute_slots > 0
                    && self.max_inflight > 0
                    && self.max_inflight <= u32::from(self.compute_slots)
            }
            ApplianceRole::WitnessStorageNode => {
                self.storage_enabled
                    && ((self.profile == NodeProfile::StorageOnly
                        && self.provision.is_none()
                        && self.compute_slots == 0
                        && self.max_inflight == 0)
                        || (self.profile != NodeProfile::StorageOnly
                            && self.provision.is_some()
                            && self.compute_slots > 0
                            && self.max_inflight > 0
                            && self.max_inflight <= u32::from(self.compute_slots)))
            }
            ApplianceRole::AllInOne => {
                self.profile != NodeProfile::StorageOnly
                    && self.provision.is_some()
                    && self.compute_slots > 0
                    && self.max_inflight > 0
                    && self.max_inflight <= u32::from(self.compute_slots)
                    && self.storage_enabled
            }
        }
    }

    fn control_config(&self) -> Result<MtlsClientConfig, RoleError> {
        Ok(MtlsClientConfig {
            hub_address: self.control.hub_address,
            server_name: self.control.server_name.clone(),
            ca_certificate: self.control.ca_certificate.clone(),
            client_certificate: self.control.client_certificate.clone(),
            client_private_key: self.control.client_private_key.clone(),
            connect_timeout: Duration::from_millis(self.control.connect_timeout_ms),
            frame_timeout: Duration::from_millis(self.control.frame_timeout_ms),
        })
    }
}

impl RoleV2Bindings {
    pub fn storage_config_path(&self) -> &Path {
        &self.storage_config_path
    }

    pub fn storage_config_sha256(&self) -> Result<Digest32, RoleError> {
        parse_hex(&self.storage_config_sha256)
    }

    pub fn docker_socket(&self) -> &Path {
        &self.docker_socket
    }

    pub fn isolation_broker_socket(&self) -> &Path {
        &self.isolation_broker_socket
    }

    pub fn open_storage_config(&self) -> Result<StorageConfig, RoleError> {
        let config = StorageConfig::open(&self.storage_config_path)?;
        if config.source_path() != self.storage_config_path
            || config.source_sha256() != self.storage_config_sha256()?
        {
            return Err(RoleError::InvalidEnvelope);
        }
        Ok(config)
    }

    fn validate_shape(&self) -> Result<(), RoleError> {
        if !valid_config_path(&self.storage_config_path)
            || !valid_config_path(&self.docker_socket)
            || !valid_config_path(&self.isolation_broker_socket)
            || self.docker_socket == self.isolation_broker_socket
            || self.storage_config_sha256()? == [0; 32]
        {
            return Err(RoleError::InvalidEnvelope);
        }
        Ok(())
    }
}

fn validate_envelope_file(path: &Path, file: &File) -> Result<(), RoleError> {
    let opened = file.metadata().map_err(RoleError::Io)?;
    let named = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() > MAX_ROLE_ENVELOPE_BYTES
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

fn read_signing_key(path: &Path) -> Result<SigningKey, RoleError> {
    validate_absolute_path(path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(RoleError::Io)?;
    let opened = file.metadata().map_err(RoleError::Io)?;
    let named = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() != 32
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)?;
    let mut secret = [0_u8; 32];
    file.read_exact(&mut secret).map_err(RoleError::Io)?;
    let signing_key = SigningKey::from_bytes(&secret);
    secret.fill(0);
    Ok(signing_key)
}

fn validate_state_root(path: &Path) -> Result<(), RoleError> {
    validate_absolute_path(path)?;
    let metadata = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::effective().as_raw()
        || metadata.mode() & 0o777 != 0o700
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

fn validate_bound_socket(path: &Path) -> Result<(), RoleError> {
    validate_absolute_path(path)?;
    let metadata = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    let owner = metadata.uid();
    if !metadata.file_type().is_socket()
        || (owner != 0 && owner != Uid::effective().as_raw())
        || metadata.mode() & 0o002 != 0
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

fn validate_physical_components(path: &Path) -> Result<(), RoleError> {
    let mut current = PathBuf::from("/");
    for component in path.components().skip(1) {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)
            .map_err(RoleError::Io)?
            .file_type()
            .is_symlink()
        {
            return Err(RoleError::InvalidEnvelope);
        }
    }
    Ok(())
}

fn validate_absolute_path(path: &Path) -> Result<(), RoleError> {
    if !valid_config_path(path) {
        return Err(RoleError::InvalidEnvelope);
    }
    Ok(())
}

fn valid_config_path(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|part| !matches!(part, Component::CurDir | Component::ParentDir))
}

fn valid_image_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().any(|byte| byte != b'0')
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn parse_digest(value: &str) -> Result<Digest32, RoleError> {
    parse_hex(value)
}

fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], RoleError> {
    if value.len() != N * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(RoleError::InvalidEnvelope);
    }
    let mut digest = [0; N];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).map_err(|_| RoleError::InvalidEnvelope)?;
        digest[index] = u8::from_str_radix(text, 16).map_err(|_| RoleError::InvalidEnvelope)?;
    }
    Ok(digest)
}

#[derive(Debug, Error)]
pub enum RoleError {
    #[error("invalid physical JainNode role envelope")]
    InvalidEnvelope,
    #[error("production JainNode runtime composition is unavailable")]
    CompositionUnavailable,
    #[error("appliance-role v1 cannot authorize production execution")]
    RoleV1CannotExecute,
    #[error("role envelope I/O failed: {0}")]
    Io(std::io::Error),
    #[error("role envelope JSON failed: {0}")]
    Json(serde_json::Error),
    #[error(transparent)]
    Storage(#[from] StorageConfigError),
    #[error(transparent)]
    Control(#[from] ControlError),
    #[error(transparent)]
    Execution(#[from] ExecutionError),
}
