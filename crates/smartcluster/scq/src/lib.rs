//! SCQ TOML v0 adapter shared by the `scq` CLI and its contract tests.
//!
//! This crate converts the single-node subset of an SCQ TOML v0 file into a
//! [`jain_smartcluster_core::JobSpec`]. Use [`parse_scq_toml_v0`] to parse and
//! validate text, or [`ScqTomlSpec::into_job_spec`] when the spec has already
//! been deserialized. The returned job is an argv workload; this adapter does
//! not submit or execute it.
//!
//! The v1 boundary is intentionally narrow: `resource.gang_size` must be one,
//! placement must be `single_node`, `single`, or empty, and node affinity is
//! unsupported. Preemption resumes by rerunning the workload. Dependency
//! probes are rejected; place those checks in an allowlisted Jain worker image
//! instead. Shell and SSH launchers are rejected rather than interpreted.
//! Unsupported features return [`AdapterError::Unsupported`], invalid values
//! return [`AdapterError::Invalid`], and malformed TOML returns
//! [`AdapterError::Toml`].

use std::{collections::BTreeMap, path::Path};

use jain_smartcluster_core::{
    CheckpointPolicy, CpuRequest, Environment, GpuMode, GpuRequest, IdempotencyKey, IoMode, JobId,
    JobSpec, MemoryRequest, PreemptionPolicy, PriorityClass, ProjectId, ResourceRequest,
    RetryPolicy, TenantId, Timestamp, Workload,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("SCQ TOML v0 is invalid: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("SCQ TOML v0 specification is unsupported: {0}")]
    Unsupported(String),
    #[error("SCQ TOML v0 specification is invalid: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScqTomlSpec {
    pub job: ScqTomlJob,
    #[serde(default)]
    pub resource: ScqTomlResource,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub preempt: ScqTomlPreemption,
    #[serde(default)]
    pub limits: ScqTomlLimits,
    #[serde(default)]
    pub dependencies: ScqTomlDependencies,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScqTomlJob {
    pub id: Option<String>,
    pub project: String,
    #[serde(default = "default_tier")]
    pub tier: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub workdir: Option<String>,
    pub venv: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ScqTomlResource {
    #[serde(default)]
    pub gpus_per_task: u16,
    pub gpu_type: Option<String>,
    #[serde(default)]
    pub cpus_reserved: u32,
    #[serde(default)]
    pub omp_threads: u32,
    #[serde(default)]
    pub dataloader_workers: u32,
    #[serde(default)]
    pub ram_gb: u64,
    #[serde(default = "default_gang_size")]
    pub gang_size: u32,
    #[serde(default = "default_placement")]
    pub placement: String,
    pub node: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScqTomlPreemption {
    #[serde(default = "default_true")]
    pub preemptible: bool,
    #[serde(default = "default_signal")]
    pub signal: String,
    #[serde(default = "default_grace_seconds")]
    pub grace_seconds: u64,
    #[serde(default = "default_checkpoint")]
    pub checkpoint: String,
    #[serde(default = "default_resume")]
    pub resume: String,
}

impl Default for ScqTomlPreemption {
    fn default() -> Self {
        Self {
            preemptible: true,
            signal: default_signal(),
            grace_seconds: default_grace_seconds(),
            checkpoint: default_checkpoint(),
            resume: default_resume(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScqTomlLimits {
    #[serde(default)]
    pub max_runtime_hours: u64,
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
}

impl Default for ScqTomlLimits {
    fn default() -> Self {
        Self {
            max_runtime_hours: 0,
            max_restarts: default_max_restarts(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ScqTomlDependencies {
    #[serde(default)]
    pub binaries: Vec<String>,
    pub python: Option<String>,
    #[serde(default)]
    pub python_imports: Vec<String>,
    #[serde(default)]
    pub python_packages: Vec<String>,
    #[serde(default)]
    pub shared_libraries: Vec<String>,
    #[serde(default)]
    pub system_packages: Vec<String>,
    #[serde(default)]
    pub smoke: Vec<ScqTomlSmokeCheck>,
}

impl ScqTomlDependencies {
    fn is_empty(&self) -> bool {
        self.binaries.is_empty()
            && self.python.as_deref().is_none_or(|value| value == "auto")
            && self.python_imports.is_empty()
            && self.python_packages.is_empty()
            && self.shared_libraries.is_empty()
            && self.system_packages.is_empty()
            && self.smoke.is_empty()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScqTomlSmokeCheck {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub timeout_secs: Option<u64>,
}

pub fn parse_scq_toml_v0(
    input: &str,
    unix_uid: u32,
    now: Timestamp,
) -> Result<JobSpec, AdapterError> {
    let parsed: ScqTomlSpec = toml::from_str(input)?;
    parsed.into_job_spec(unix_uid, now)
}

impl ScqTomlSpec {
    pub fn into_job_spec(self, unix_uid: u32, now: Timestamp) -> Result<JobSpec, AdapterError> {
        validate_scq_toml_v0_scope(&self)?;

        let mut environment = self.env;
        let threads = if self.resource.omp_threads == 0 {
            if self.resource.gpus_per_task == 0 {
                1
            } else {
                4
            }
        } else {
            self.resource.omp_threads
        };
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ] {
            environment.insert(key.into(), threads.to_string());
        }

        let program = resolve_venv_program(&self.job.command, self.job.venv.as_deref());
        reject_shell_program(&program)?;
        let idempotency = self
            .job
            .id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| JobId::new().to_string());

        let cpu_millicores = self.resource.cpus_reserved.max(1).saturating_mul(1000);
        let hard_bytes = if self.resource.ram_gb == 0 {
            u64::MAX
        } else {
            self.resource.ram_gb.saturating_mul(GIB)
        };
        let soft_bytes = if hard_bytes == u64::MAX {
            0
        } else {
            hard_bytes.saturating_mul(3) / 4
        };
        let gpu = (self.resource.gpus_per_task > 0).then_some(GpuRequest {
            count: self.resource.gpus_per_task,
            vram_bytes: u64::MAX,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        });

        let mut labels = BTreeMap::from([("adapter.source".into(), "scq-toml-v0".into())]);
        if let Some(gpu_type) = self
            .resource
            .gpu_type
            .filter(|value| !value.trim().is_empty())
        {
            labels.insert("adapter.gpu_type".into(), gpu_type);
        }
        if self.resource.dataloader_workers > 0 {
            labels.insert(
                "adapter.dataloader_workers".into(),
                self.resource.dataloader_workers.to_string(),
            );
        }

        let retry = RetryPolicy {
            max_failures: self.limits.max_restarts,
            ..RetryPolicy::default()
        };

        Ok(JobSpec {
            tenant: TenantId::new(format!("unix:{unix_uid}"))
                .map_err(|error| AdapterError::Invalid(error.to_string()))?,
            project: ProjectId::new(self.job.project.trim())
                .map_err(|error| AdapterError::Invalid(error.to_string()))?,
            idempotency_key: IdempotencyKey::new(idempotency)
                .map_err(|error| AdapterError::Invalid(error.to_string()))?,
            priority: map_priority(&self.job.tier)?,
            workload: Workload::AdministratorArgv {
                program,
                arguments: self.job.args,
                environment: Environment(environment),
                working_directory: self.job.workdir,
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(cpu_millicores),
                ram: MemoryRequest {
                    soft_bytes,
                    hard_bytes,
                },
                scratch_bytes: 0,
                gpu,
            },
            retry,
            preemption: map_preemption(&self.preempt)?,
            io: IoMode::Pipes,
            deadline: (self.limits.max_runtime_hours > 0).then(|| {
                Timestamp::from_unix_millis(
                    now.as_unix_millis()
                        .saturating_add(self.limits.max_runtime_hours.saturating_mul(3_600_000)),
                )
            }),
            labels,
        })
    }
}

fn validate_scq_toml_v0_scope(spec: &ScqTomlSpec) -> Result<(), AdapterError> {
    if spec.resource.gang_size != 1 {
        return Err(AdapterError::Unsupported(
            "gang_size must be 1 in single-node v1".into(),
        ));
    }
    if !matches!(
        spec.resource.placement.as_str(),
        "single_node" | "single" | ""
    ) {
        return Err(AdapterError::Unsupported(
            "only single_node placement is available in v1".into(),
        ));
    }
    if spec
        .resource
        .node
        .as_deref()
        .is_some_and(|node| !node.is_empty())
    {
        return Err(AdapterError::Unsupported(
            "node affinity and multinode agents are not available in v1".into(),
        ));
    }
    if spec.preempt.resume.trim() != "rerun" {
        return Err(AdapterError::Unsupported(
            "only preempt.resume = \"rerun\" is supported".into(),
        ));
    }
    if !spec.dependencies.is_empty() {
        return Err(AdapterError::Unsupported(
            "dependency probes must move into an allowlisted Jain worker image".into(),
        ));
    }
    Ok(())
}

fn resolve_venv_program(command: &str, venv: Option<&str>) -> String {
    match (venv, Path::new(command).components().count()) {
        (Some(root), 1) if !root.trim().is_empty() => Path::new(root)
            .join("bin")
            .join(command)
            .to_string_lossy()
            .into_owned(),
        _ => command.to_owned(),
    }
}

fn reject_shell_program(program: &str) -> Result<(), AdapterError> {
    let name = Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program)
        .to_ascii_lowercase();
    if matches!(
        name.as_str(),
        "bash" | "sh" | "dash" | "zsh" | "fish" | "ssh"
    ) {
        return Err(AdapterError::Unsupported(format!(
            "shell or SSH launcher {name:?} is not allowed"
        )));
    }
    Ok(())
}

fn map_priority(tier: &str) -> Result<PriorityClass, AdapterError> {
    match tier.trim().to_ascii_lowercase().as_str() {
        "system" => Ok(PriorityClass::System),
        "high" => Ok(PriorityClass::Interactive),
        "normal" | "" => Ok(PriorityClass::Batch),
        "low" | "idle" => Ok(PriorityClass::Opportunistic),
        other => Err(AdapterError::Invalid(format!("unknown job.tier {other:?}"))),
    }
}

fn map_preemption(preempt: &ScqTomlPreemption) -> Result<PreemptionPolicy, AdapterError> {
    if !preempt.preemptible {
        return Ok(PreemptionPolicy::non_preemptible());
    }
    match preempt.checkpoint.trim().to_ascii_lowercase().as_str() {
        "periodic" | "on_signal" | "onsignal" | "signal" | "none" | "" => {}
        other => {
            return Err(AdapterError::Invalid(format!(
                "unknown preempt.checkpoint {other:?}"
            )))
        }
    }
    Ok(PreemptionPolicy {
        preemptible: true,
        checkpoint: CheckpointPolicy::Signal {
            signal: parse_signal(&preempt.signal)?,
            grace_ms: preempt.grace_seconds.max(1).saturating_mul(1000),
        },
    })
}

fn parse_signal(signal: &str) -> Result<i32, AdapterError> {
    match signal.trim().to_ascii_uppercase().as_str() {
        "SIGTERM" | "TERM" | "15" => Ok(15),
        "SIGINT" | "INT" | "2" => Ok(2),
        "SIGUSR1" | "USR1" | "10" => Ok(10),
        "SIGUSR2" | "USR2" | "12" => Ok(12),
        other => Err(AdapterError::Invalid(format!(
            "unsupported preempt.signal {other:?}"
        ))),
    }
}

fn default_tier() -> String {
    "normal".into()
}
const fn default_gang_size() -> u32 {
    1
}
fn default_placement() -> String {
    "single_node".into()
}
const fn default_true() -> bool {
    true
}
fn default_signal() -> String {
    "SIGTERM".into()
}
const fn default_grace_seconds() -> u64 {
    90
}
fn default_checkpoint() -> String {
    "periodic".into()
}
fn default_resume() -> String {
    "rerun".into()
}
const fn default_max_restarts() -> u32 {
    3
}

#[cfg(test)]
mod tests {
    use super::*;
    use jain_smartcluster_core::{GpuMode, PriorityClass};

    const MINIMAL: &str = r#"
[job]
id = "demo"
project = "examples"
command = "python"
args = ["train.py"]
venv = "/opt/venv"

[resource]
gpus_per_task = 1
cpus_reserved = 8
ram_gb = 32
"#;

    #[test]
    fn maps_scq_toml_resources_without_enabling_sharing() {
        let spec = parse_scq_toml_v0(MINIMAL, 1000, Timestamp::from_unix_millis(100)).unwrap();
        assert_eq!(spec.tenant.as_str(), "unix:1000");
        assert_eq!(spec.priority, PriorityClass::Batch);
        assert_eq!(spec.resources.cpu.minimum_millicores, 8000);
        assert_eq!(spec.resources.gpu.unwrap().mode, GpuMode::Exclusive);
        let Workload::AdministratorArgv { program, .. } = spec.workload else {
            panic!("expected argv workload");
        };
        assert_eq!(program, "/opt/venv/bin/python");
    }

    #[test]
    fn rejects_multinode_shell_and_dependency_probes() {
        let gang = MINIMAL.replace("cpus_reserved = 8", "cpus_reserved = 8\ngang_size = 2");
        assert!(matches!(
            parse_scq_toml_v0(&gang, 1, 0.into()),
            Err(AdapterError::Unsupported(_))
        ));
        let shell = MINIMAL.replace("command = \"python\"", "command = \"bash\"");
        assert!(matches!(
            parse_scq_toml_v0(&shell, 1, 0.into()),
            Err(AdapterError::Unsupported(_))
        ));
    }

    #[test]
    fn maps_tiers_and_deadline() {
        let input = MINIMAL.replace(
            "project = \"examples\"",
            "project = \"examples\"\ntier = \"high\"",
        ) + "\n[limits]\nmax_runtime_hours = 2\n";
        let spec = parse_scq_toml_v0(&input, 1, Timestamp::from_unix_millis(500)).unwrap();
        assert_eq!(spec.priority, PriorityClass::Interactive);
        assert_eq!(spec.deadline.unwrap().as_unix_millis(), 7_200_500);
    }
}
