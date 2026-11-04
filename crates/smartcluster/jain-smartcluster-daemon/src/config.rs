use std::fs;
use std::path::{Path, PathBuf};

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

mod settings;

#[cfg(test)]
mod tests;

pub use settings::{
    AuthConfig, ExecutionConfig, MpsConfig, PsiConfig, SchedulerConfig, WorkerConfig,
};

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
    // jankurai:allow HLT-001-DEAD-MARKER reason=typed-unsupported-operation-error expires=2026-12-31
    #[error(
        "fabric.enabled = true refused: fabric networking is not \
         implemented in this build"
    )]
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
