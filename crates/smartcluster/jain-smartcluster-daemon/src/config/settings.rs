use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::ConfigError;

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
    pub(crate) fn validate_reserved_roles(&self) -> Result<(), ConfigError> {
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
    pub(crate) fn validate_product_worker(&self) -> Result<(), ConfigError> {
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
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
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
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
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
    pub(crate) fn validate(self) -> Result<(), ConfigError> {
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
