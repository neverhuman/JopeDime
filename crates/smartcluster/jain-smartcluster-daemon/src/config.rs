use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use jain_smartcluster_core::TenantId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
mod deployed_tests;
mod fabric;
mod product_worker;

pub use fabric::FabricConfig;
pub use product_worker::ProductWorkerTemplate;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub state_path: PathBuf,
    pub event_dir: PathBuf,
    pub cgroup_root: PathBuf,
    pub socket_mode: u32,
    pub embedded: bool,
    pub scheduler: SchedulerConfig,
    pub auth: AuthConfig,
    pub workers: WorkerConfig,
    pub execution: ExecutionConfig,
    pub mps: MpsConfig,
    pub psi: PsiConfig,
    /// Parsed for deployed-file compatibility; `enabled = true` refuses startup.
    pub fabric: FabricConfig,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/run/jain-smartcluster/scqd.sock"),
            state_path: PathBuf::from("/var/lib/jain-smartcluster/scqd.redb"),
            event_dir: PathBuf::from("/var/lib/jain-smartcluster/events"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup/jain-smartcluster.slice"),
            socket_mode: 0o660,
            embedded: false,
            scheduler: SchedulerConfig::default(),
            auth: AuthConfig::default(),
            workers: WorkerConfig::default(),
            execution: ExecutionConfig::default(),
            mps: MpsConfig::default(),
            psi: PsiConfig::default(),
            fabric: FabricConfig::default(),
        }
    }
}

impl DaemonConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Self = toml::from_str(&source).map_err(ConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, path) in [
            ("socket_path", &self.socket_path),
            ("state_path", &self.state_path),
            ("event_dir", &self.event_dir),
            ("cgroup_root", &self.cgroup_root),
            ("workers.jain_worker", &self.workers.jain_worker),
        ] {
            if !path.is_absolute() {
                return Err(ConfigError::RelativePath {
                    name,
                    path: path.clone(),
                });
            }
        }
        self.fabric.validate()?;
        self.workers.validate_product_worker()?;
        for path in self.workers.trusted_services.values() {
            if !path.is_absolute() {
                return Err(ConfigError::RelativePath {
                    name: "workers.trusted_services",
                    path: path.clone(),
                });
            }
        }
        for service in self.execution.resource_policy.service_cpu_sets.keys() {
            if !self.workers.trusted_services.contains_key(service) {
                return Err(ConfigError::ResourcePolicy(format!(
                    "CPU pool names unregistered trusted service {service}"
                )));
            }
        }
        if let Some(policy) = &self.execution.trusted_fit_child_policy {
            policy
                .validate()
                .map_err(ConfigError::TrustedFitChildPolicy)?;
            if !self
                .workers
                .trusted_services
                .contains_key("jope-fit-authority")
            {
                return Err(ConfigError::TrustedFitChildPolicy(
                    "fit authority must be explicitly registered".to_owned(),
                ));
            }
        }
        for (uuid, baseline) in &self.execution.gpu_idle_baselines {
            baseline
                .validate(uuid)
                .map_err(ConfigError::GpuIdleBaseline)?;
        }
        if let Some(pool) = &self.execution.resource_policy.trusted_scratch {
            for service in &pool.trusted_services {
                if !self.workers.trusted_services.contains_key(service) {
                    return Err(ConfigError::ResourcePolicy(format!(
                        "scratch allowlist names unregistered trusted service {service}"
                    )));
                }
            }
        }
        if self.socket_mode & !0o777 != 0 {
            return Err(ConfigError::SocketMode(self.socket_mode));
        }
        if self.auth.web_principal.trim().is_empty() {
            return Err(ConfigError::EmptyWebPrincipal);
        }
        self.auth.validate_reserved_roles()?;
        self.execution.validate()?;
        self.mps.validate()?;
        self.psi.validate()?;
        if self.scheduler.max_running_jobs == 0
            || self.scheduler.max_running_jobs_per_tenant == 0
            || self.scheduler.max_running_jobs_per_tenant > self.scheduler.max_running_jobs
        {
            return Err(ConfigError::SchedulerRunningLimits);
        }
        if self.scheduler.max_queued_jobs == 0
            || self.scheduler.max_queued_jobs_per_tenant == 0
            || self.scheduler.max_queued_jobs_per_tenant > self.scheduler.max_queued_jobs
        {
            return Err(ConfigError::SchedulerQueueLimits);
        }
        for (tenant, weight) in &self.scheduler.tenant_weights {
            TenantId::new(tenant.clone()).map_err(|error| ConfigError::InvalidTenantWeight {
                tenant: tenant.clone(),
                reason: error.to_string(),
            })?;
            if *weight == 0 {
                return Err(ConfigError::InvalidTenantWeight {
                    tenant: tenant.clone(),
                    reason: "weight must be greater than zero".to_owned(),
                });
            }
        }
        if self.scheduler.interactive_gpu_headroom_per_mille > 1_000 {
            return Err(ConfigError::SchedulerGpuHeadroom(
                self.scheduler.interactive_gpu_headroom_per_mille,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub interactive_cpu_headroom_millicores: u64,
    pub interactive_ram_headroom_bytes: u64,
    pub interactive_gpu_headroom_per_mille: u16,
    pub recovery_quiet_ms: u64,
    pub max_running_jobs: usize,
    pub max_running_jobs_per_tenant: usize,
    pub max_queued_jobs: usize,
    pub max_queued_jobs_per_tenant: usize,
    pub tenant_weights: BTreeMap<String, u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            interactive_cpu_headroom_millicores: 2_000,
            interactive_ram_headroom_bytes: 4 * 1024 * 1024 * 1024,
            interactive_gpu_headroom_per_mille: 0,
            recovery_quiet_ms: 2_000,
            max_running_jobs: 1_024,
            max_running_jobs_per_tenant: 128,
            max_queued_jobs: 10_000,
            max_queued_jobs_per_tenant: 2_500,
            tenant_weights: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    pub admin_uids: Vec<u32>,
    pub system_uids: Vec<u32>,
    pub system_principal: String,
    /// Reserved fabric worker-edge UID. Parsed for deployed-file compatibility;
    /// this build grants it no role. Unset by default, so a file that omits it
    /// can never fail the overlap check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_uid: Option<u32>,
    /// Reserved fabric broker UID. Parsed for deployed-file compatibility;
    /// this build grants it no role. Unset by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub broker_uid: Option<u32>,
    pub web_uid: u32,
    pub web_principal: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            admin_uids: vec![0],
            system_uids: vec![0],
            system_principal: "sagemaker".to_owned(),
            edge_uid: None,
            broker_uid: None,
            web_uid: 994,
            web_principal: "jain-web".to_owned(),
        }
    }
}

impl AuthConfig {
    /// The reserved fabric UIDs must never alias a broader role, so a later
    /// build that grants them authority cannot widen an existing identity.
    /// Only UIDs that are set are checked.
    fn validate_reserved_roles(&self) -> Result<(), ConfigError> {
        if let Some(edge) = self.edge_uid {
            if edge == self.web_uid
                || self.system_uids.contains(&edge)
                || self.admin_uids.contains(&edge)
            {
                return Err(ConfigError::OverlappingEdgeUid(edge));
            }
        }
        if let Some(broker) = self.broker_uid {
            if self.edge_uid == Some(broker)
                || broker == self.web_uid
                || self.admin_uids.contains(&broker)
                || self.system_uids.contains(&broker)
            {
                return Err(ConfigError::BrokerIdentityOverlap(broker));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    pub jain_worker: PathBuf,
    /// Reserved: digest-pinned product worker for explicitly declared
    /// templates. Parsed and validated only; this build has no executor hook
    /// that launches it, so every typed workload still runs `jain_worker`.
    pub product_worker: Option<PathBuf>,
    pub product_worker_sha256: Option<String>,
    pub product_worker_templates: BTreeSet<ProductWorkerTemplate>,
    pub trusted_services: BTreeMap<String, PathBuf>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            jain_worker: PathBuf::from("/usr/libexec/jain/jain-worker"),
            product_worker: None,
            product_worker_sha256: None,
            product_worker_templates: BTreeSet::new(),
            trusted_services: BTreeMap::new(),
        }
    }
}

impl WorkerConfig {
    fn validate_product_worker(&self) -> Result<(), ConfigError> {
        if let Some(path) = &self.product_worker {
            if !path.is_absolute() {
                return Err(ConfigError::RelativePath {
                    name: "workers.product_worker",
                    path: path.clone(),
                });
            }
        }
        match (
            self.product_worker.as_ref(),
            self.product_worker_sha256.as_deref(),
            self.product_worker_templates.is_empty(),
        ) {
            (None, None, true) => Ok(()),
            (Some(_), Some(digest), false) => {
                if digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
                    Ok(())
                } else {
                    Err(ConfigError::InvalidProductWorkerDigest)
                }
            }
            _ => Err(ConfigError::IncompleteProductWorkerConfiguration),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionConfig {
    pub trusted_fit_child_policy: Option<crate::fit_child_policy::TrustedFitChildPolicy>,
    pub resource_policy: crate::resource_policy::ResourcePolicy,
    /// Empty by default: unaccounted GPU use stays inadmissible without qualified evidence.
    pub gpu_idle_baselines: BTreeMap<String, crate::gpu::QualifiedIdleBaseline>,
    /// Dedicated unprivileged account used for every workload process.
    pub worker_uid: Option<u32>,
    pub worker_gid: Option<u32>,
    pub checkpoint_timeout_ms: u64,
    pub terminate_grace_ms: u64,
    pub kill_grace_ms: u64,
    pub launch_stop_timeout_ms: u64,
    pub actor_io_timeout_ms: u64,
    pub max_control_frame_bytes: usize,
    pub subscriber_capacity: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            trusted_fit_child_policy: None,
            resource_policy: Default::default(),
            gpu_idle_baselines: BTreeMap::new(),
            worker_uid: None,
            worker_gid: None,
            checkpoint_timeout_ms: 5_000,
            terminate_grace_ms: 10_000,
            kill_grace_ms: 2_000,
            launch_stop_timeout_ms: 2_000,
            actor_io_timeout_ms: 1_000,
            max_control_frame_bytes: 1024 * 1024,
            subscriber_capacity: 512,
        }
    }
}

impl ExecutionConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.worker_uid.is_some() != self.worker_gid.is_some() {
            return Err(ConfigError::IncompleteWorkerIdentity);
        }
        if self.max_control_frame_bytes == 0 || self.max_control_frame_bytes > 16 * 1024 * 1024 {
            return Err(ConfigError::ControlFrameBound(self.max_control_frame_bytes));
        }
        if self.subscriber_capacity == 0 {
            return Err(ConfigError::SubscriberCapacity);
        }
        if self.checkpoint_timeout_ms == 0
            || self.terminate_grace_ms == 0
            || self.kill_grace_ms == 0
            || self.launch_stop_timeout_ms == 0
            || self.actor_io_timeout_ms == 0
        {
            return Err(ConfigError::ExecutionTimeout);
        }
        Ok(())
    }

    pub const fn checkpoint_timeout(&self) -> Duration {
        Duration::from_millis(self.checkpoint_timeout_ms)
    }

    pub const fn terminate_grace(&self) -> Duration {
        Duration::from_millis(self.terminate_grace_ms)
    }

    pub const fn kill_grace(&self) -> Duration {
        Duration::from_millis(self.kill_grace_ms)
    }

    pub const fn launch_stop_timeout(&self) -> Duration {
        Duration::from_millis(self.launch_stop_timeout_ms)
    }

    pub const fn actor_io_timeout(&self) -> Duration {
        Duration::from_millis(self.actor_io_timeout_ms)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MpsConfig {
    /// Absolute path to `nvidia-cuda-mps-control`. `None` disables MPS sharing.
    pub control_executable: Option<PathBuf>,
    /// Absolute pipe directory belonging to the configured MPS control daemon.
    pub pipe_directory: Option<PathBuf>,
    pub command_timeout_ms: u64,
    pub max_control_output_bytes: usize,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self {
            control_executable: None,
            pipe_directory: None,
            command_timeout_ms: 1_000,
            max_control_output_bytes: 1024 * 1024,
        }
    }
}

impl MpsConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        match (&self.control_executable, &self.pipe_directory) {
            (None, None) => {}
            (Some(executable), Some(directory)) => {
                for (name, path) in [
                    ("mps.control_executable", executable),
                    ("mps.pipe_directory", directory),
                ] {
                    if !path.is_absolute() {
                        return Err(ConfigError::RelativePath {
                            name,
                            path: path.clone(),
                        });
                    }
                }
            }
            _ => return Err(ConfigError::IncompleteMpsConfiguration),
        }
        if self.command_timeout_ms == 0 {
            return Err(ConfigError::MpsCommandTimeout);
        }
        if self.max_control_output_bytes == 0 || self.max_control_output_bytes > 16 * 1024 * 1024 {
            return Err(ConfigError::MpsOutputBound(self.max_control_output_bytes));
        }
        Ok(())
    }

    pub const fn command_timeout(&self) -> Duration {
        Duration::from_millis(self.command_timeout_ms)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PsiConfig {
    pub window_us: u64,
    pub warning_stall_us: u64,
    pub critical_stall_us: u64,
}

impl Default for PsiConfig {
    fn default() -> Self {
        Self {
            window_us: 500_000,
            warning_stall_us: 100_000,
            critical_stall_us: 250_000,
        }
    }
}

impl PsiConfig {
    fn validate(self) -> Result<(), ConfigError> {
        if self.window_us == 0
            || self.warning_stall_us == 0
            || self.warning_stall_us >= self.critical_stall_us
            || self.critical_stall_us > self.window_us
        {
            return Err(ConfigError::PsiThresholds);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid trusted fit child policy: {0}")]
    TrustedFitChildPolicy(String),
    #[error("invalid resource policy: {0}")]
    ResourcePolicy(String),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid daemon configuration: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("{name} must be an absolute path, got {path}")]
    RelativePath { name: &'static str, path: PathBuf },
    #[error("socket_mode must be a Unix permission mask, got {0:#o}")]
    SocketMode(u32),
    #[error("auth.web_principal must not be empty")]
    EmptyWebPrincipal,
    #[error("execution.max_control_frame_bytes is out of range: {0}")]
    ControlFrameBound(usize),
    #[error("execution.subscriber_capacity must be positive")]
    SubscriberCapacity,
    #[error("execution timeouts must be positive")]
    ExecutionTimeout,
    #[error("execution.worker_uid and execution.worker_gid must be configured together")]
    IncompleteWorkerIdentity,
    #[error("mps.control_executable and mps.pipe_directory must be configured together")]
    IncompleteMpsConfiguration,
    #[error("mps.command_timeout_ms must be positive")]
    MpsCommandTimeout,
    #[error("mps.max_control_output_bytes is out of range: {0}")]
    MpsOutputBound(usize),
    #[error("PSI thresholds must satisfy 0 < warning < critical <= window")]
    PsiThresholds,
    #[error("scheduler running limits must satisfy 0 < per_tenant <= global")]
    SchedulerRunningLimits,
    #[error("scheduler queue limits must satisfy 0 < per_tenant <= global")]
    SchedulerQueueLimits,
    #[error("invalid scheduler tenant weight for {tenant}: {reason}")]
    InvalidTenantWeight { tenant: String, reason: String },
    #[error("scheduler GPU headroom must be at most 1000 per mille, got {0}")]
    SchedulerGpuHeadroom(u16),
    #[error("invalid qualified GPU idle baseline: {0}")]
    GpuIdleBaseline(String),
    #[error("fabric.enabled = true refused: fabric networking is not implemented in this build")]
    FabricNotImplemented,
    #[error("invalid fabric configuration: {0}")]
    InvalidFabric(String),
    #[error("auth.edge_uid {0} overlaps the web, system, or administrator identity")]
    OverlappingEdgeUid(u32),
    #[error("auth.broker_uid {0} overlaps the edge, web, system, or administrator identity")]
    BrokerIdentityOverlap(u32),
    #[error(
        "workers.product_worker, product_worker_sha256, and a nonempty product_worker_templates set must be configured together"
    )]
    IncompleteProductWorkerConfiguration,
    #[error("workers.product_worker_sha256 must be exactly 64 lowercase hexadecimal characters")]
    InvalidProductWorkerDigest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_surface_parses() {
        let config: DaemonConfig = toml::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../smartcluster.example.toml"
        )))
        .expect("example config parses");
        config.validate().expect("example config validates");
        assert_eq!(config.socket_mode, 0o660);
    }

    #[test]
    fn rejects_socket_privilege_bits() {
        let config = DaemonConfig {
            socket_mode: 0o1660,
            ..DaemonConfig::default()
        };
        assert!(matches!(config.validate(), Err(ConfigError::SocketMode(_))));
    }

    #[test]
    fn worker_identity_is_an_atomic_pair() {
        let config = DaemonConfig {
            execution: ExecutionConfig {
                worker_uid: Some(993),
                worker_gid: None,
                ..ExecutionConfig::default()
            },
            ..DaemonConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::IncompleteWorkerIdentity)
        ));
    }

    #[test]
    fn mps_paths_are_an_absolute_pair() {
        let incomplete = DaemonConfig {
            mps: MpsConfig {
                control_executable: Some(PathBuf::from("/usr/bin/nvidia-cuda-mps-control")),
                ..MpsConfig::default()
            },
            ..DaemonConfig::default()
        };
        assert!(matches!(
            incomplete.validate(),
            Err(ConfigError::IncompleteMpsConfiguration)
        ));

        let relative = DaemonConfig {
            mps: MpsConfig {
                control_executable: Some(PathBuf::from("usr/bin/nvidia-cuda-mps-control")),
                pipe_directory: Some(PathBuf::from("/run/nvidia-mps")),
                ..MpsConfig::default()
            },
            ..DaemonConfig::default()
        };
        assert!(matches!(
            relative.validate(),
            Err(ConfigError::RelativePath {
                name: "mps.control_executable",
                ..
            })
        ));
    }
}
