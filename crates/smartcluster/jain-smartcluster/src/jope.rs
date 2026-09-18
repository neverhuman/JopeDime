//! JOPE declares eligible work; the native scheduler owns its allocation.
//!
//! A submission binds exact manifest bytes. The registered service must recheck
//! that digest and every dependency before doing work, and write a durable
//! receipt. A submission/control acknowledgement is never a result receipt.
use std::{collections::BTreeMap, io::Read, os::unix::fs::OpenOptionsExt, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    CheckpointPolicy, Client, DispatchError, Environment, GpuMode, IdempotencyKey, IoMode,
    JobRecord, JobSpec, PreemptionPolicy, PriorityClass, ProjectId, ResourceRequest, RetryPolicy,
    ServiceName, TenantId, Timestamp, Validate, ValidationError, Workload,
};

pub const MANIFEST_SCHEMA: &str = "jope.smartcluster_task.v1";
pub const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

/// No executable, shell, environment override, or free-form argv is accepted.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    GpSearch,
    Continuation,
    Packing,
    Verification,
    Publication,
    ResidentGpu,
}

impl TaskKind {
    pub const fn service(self) -> &'static str {
        match self {
            Self::GpSearch => "jope-gp-search",
            Self::Continuation => "jope-continuation",
            Self::Packing => "jope-packing",
            Self::Verification => "jope-verification",
            Self::Publication => "jope-publication",
            Self::ResidentGpu => "jope-resident-gpu",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskManifest {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub task_id: String,
    pub kind: TaskKind,
    pub input_manifest_sha256: String,
    pub dependencies: BTreeMap<String, String>,
    pub parent_checkpoint_sha256: Option<String>,
    pub corpus_sha256: Option<String>,
    pub idempotency_key: IdempotencyKey,
    pub deadline: Timestamp,
    pub retry_allowance: u32,
    #[serde(deserialize_with = "closed_resources")]
    pub resources: ResourceRequest,
    pub expected_receipt_id: String,
    /// Measured checkpoint allowance, required for the resident worker. It is
    /// explicit so the daemon's generic five-second fallback is never selected.
    pub checkpoint_allowance_ms: Option<u64>,
}

// Keep the general scheduler wire format compatible while rejecting misspelled
// resource fields in the stricter JOPE task contract, including nested fields.
fn closed_resources<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<ResourceRequest, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Cpu {
        minimum_millicores: u32,
        desired_millicores: u32,
        maximum_millicores: u32,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Ram {
        soft_bytes: u64,
        hard_bytes: u64,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Gpu {
        count: u16,
        vram_bytes: u64,
        compute_share_per_mille: u16,
        mode: GpuMode,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Resources {
        cpu: Cpu,
        ram: Ram,
        scratch_bytes: u64,
        gpu: Option<Gpu>,
    }
    let value = Resources::deserialize(deserializer)?;
    Ok(ResourceRequest {
        cpu: crate::CpuRequest {
            minimum_millicores: value.cpu.minimum_millicores,
            desired_millicores: value.cpu.desired_millicores,
            maximum_millicores: value.cpu.maximum_millicores,
        },
        ram: crate::MemoryRequest {
            soft_bytes: value.ram.soft_bytes,
            hard_bytes: value.ram.hard_bytes,
        },
        scratch_bytes: value.scratch_bytes,
        gpu: value.gpu.map(|gpu| crate::GpuRequest {
            count: gpu.count,
            vram_bytes: gpu.vram_bytes,
            compute_share_per_mille: gpu.compute_share_per_mille,
            mode: gpu.mode,
        }),
    })
}

fn invalid(field: &str, message: &str) -> ValidationError {
    ValidationError::new(format!("jope.{field}"), message)
}

fn digest(field: &str, value: &str) -> Result<(), ValidationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid(field, "expected lowercase SHA-256"));
    }
    Ok(())
}

fn identity(field: &str, value: &str) -> Result<(), ValidationError> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-:/".contains(&b))
    {
        return Err(invalid(field, "invalid bounded task identity"));
    }
    Ok(())
}

impl TaskManifest {
    pub fn validate_at(&self, now: Timestamp) -> Result<(), ValidationError> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(invalid("schema", "unsupported task manifest"));
        }
        identity("run", &self.run)?;
        identity("task_id", &self.task_id)?;
        identity("expected_receipt_id", &self.expected_receipt_id)?;
        self.idempotency_key.validate()?;
        digest("input_manifest_sha256", &self.input_manifest_sha256)?;
        if self.dependencies.is_empty() || self.dependencies.len() > 4096 {
            return Err(invalid(
                "dependencies",
                "expected 1..=4096 dependency bindings",
            ));
        }
        for (name, hash) in &self.dependencies {
            identity("dependency", name)?;
            digest("dependency", hash)?;
        }
        for hash in [&self.parent_checkpoint_sha256, &self.corpus_sha256]
            .into_iter()
            .flatten()
        {
            digest("parent", hash)?;
        }
        if self.deadline <= now {
            return Err(invalid("deadline", "original task deadline elapsed"));
        }
        self.resources.validate()?;
        if self.resources.cpu.minimum_millicores == 0 || self.resources.ram.soft_bytes == 0 {
            return Err(invalid(
                "resources",
                "every JOPE service requires host CPU and RAM reservations",
            ));
        }
        let retry = RetryPolicy {
            max_failures: self.retry_allowance,
            ..RetryPolicy::default()
        };
        retry.validate()?;
        match (self.kind, self.resources.gpu, self.checkpoint_allowance_ms) {
            (TaskKind::ResidentGpu, Some(gpu), Some(allowance))
                if gpu.count == 1 && gpu.mode == GpuMode::Exclusive
                    && allowance > 0 && allowance <= self.deadline.as_unix_millis() - now.as_unix_millis()
                    && self.parent_checkpoint_sha256.is_some() && self.corpus_sha256.is_some() => (),
            (TaskKind::ResidentGpu, _, _) => return Err(invalid("resident", "requires one exclusive GPU, parent/corpus bindings and an explicit checkpoint allowance within the deadline")),
            (_, None, None) => (),
            _ => return Err(invalid("resources", "only the resident service may allocate a GPU or checkpoint channel")),
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
}

/// Read bounded bytes once and bind their digest into the native JobSpec.
/// Re-reading this path at execution must produce the same digest, or fail.
pub fn job_spec(
    path: &Path,
    expected_sha256: &str,
    tenant: TenantId,
    project: ProjectId,
    now: Timestamp,
) -> Result<JobSpec, TaskError> {
    digest("manifest_sha256", expected_sha256)?;
    if !path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err(invalid("manifest_path", "expected absolute normalized path").into());
    }
    let path_text = path
        .to_str()
        .ok_or_else(|| invalid("manifest_path", "expected UTF-8"))?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(invalid("manifest_path", "expected a regular file").into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(invalid("manifest", "manifest exceeds 1 MiB").into());
    }
    if format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
        return Err(invalid("manifest_sha256", "manifest bytes changed").into());
    }
    let task: TaskManifest = serde_json::from_slice(&bytes)?;
    task.validate_at(now)?;
    let preemption =
        task.checkpoint_allowance_ms
            .map_or_else(PreemptionPolicy::default, |deadline_ms| PreemptionPolicy {
                preemptible: true,
                checkpoint: CheckpointPolicy::ControlFd { deadline_ms },
            });
    let spec = JobSpec {
        tenant,
        project,
        idempotency_key: task.idempotency_key,
        priority: PriorityClass::Batch,
        workload: Workload::TrustedService {
            service: ServiceName::new(task.kind.service())?,
            arguments: vec![
                "--task-manifest".into(),
                path_text.into(),
                "--manifest-sha256".into(),
                expected_sha256.into(),
            ],
            environment: Environment::default(),
        },
        resources: task.resources,
        retry: RetryPolicy {
            max_failures: task.retry_allowance,
            ..RetryPolicy::default()
        },
        preemption,
        io: IoMode::Pipes,
        deadline: Some(task.deadline),
        labels: BTreeMap::from([
            ("jope.run".into(), task.run),
            ("jope.generation".into(), task.generation.to_string()),
            ("jope.task".into(), task.task_id),
            ("jope.manifest_sha256".into(), expected_sha256.into()),
            ("jope.expected_receipt".into(), task.expected_receipt_id),
        ]),
    };
    spec.validate()?;
    Ok(spec)
}

pub async fn submit(
    client: &Client,
    path: &Path,
    expected_sha256: &str,
    tenant: TenantId,
    project: ProjectId,
    now: Timestamp,
) -> Result<JobRecord, TaskError> {
    Ok(client
        .submit(job_spec(path, expected_sha256, tenant, project, now)?)
        .await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuRequest;

    fn fixture() -> TaskManifest {
        TaskManifest {
            schema: MANIFEST_SCHEMA.into(),
            run: "xbabe2-60000".into(),
            generation: 1,
            task_id: "generation-1/gp-1".into(),
            kind: TaskKind::GpSearch,
            input_manifest_sha256: "a".repeat(64),
            dependencies: BTreeMap::from([("input".into(), "b".repeat(64))]),
            parent_checkpoint_sha256: None,
            corpus_sha256: None,
            idempotency_key: IdempotencyKey::new("generation-1/gp-1").unwrap(),
            deadline: Timestamp::from(100_000),
            retry_allowance: 2,
            resources: ResourceRequest::cpu_and_ram(1000, 1024 * 1024 * 1024),
            expected_receipt_id: "generation-1/gp-1/result".into(),
            checkpoint_allowance_ms: None,
        }
    }

    fn write(dir: &tempfile::TempDir, task: &TaskManifest) -> (std::path::PathBuf, String) {
        let path = dir.path().join("task.json");
        let bytes = serde_json::to_vec(task).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        (path, format!("{:x}", Sha256::digest(&bytes)))
    }

    fn spec(path: &Path, hash: &str) -> Result<JobSpec, TaskError> {
        job_spec(
            path,
            hash,
            TenantId::new("jope").unwrap(),
            ProjectId::new("learning").unwrap(),
            Timestamp::from(1),
        )
    }

    #[test]
    fn retries_are_identical_but_conflicting_manifests_change_the_native_spec() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = fixture();
        let (path, hash) = write(&dir, &task);
        let first = spec(&path, &hash).unwrap();
        assert_eq!(first, spec(&path, &hash).unwrap());
        task.dependencies.insert("input".into(), "c".repeat(64));
        let (_, changed_hash) = write(&dir, &task);
        assert!(spec(&path, &hash).is_err());
        let changed = spec(&path, &changed_hash).unwrap();
        assert_eq!(first.idempotency_key, changed.idempotency_key);
        assert_ne!(first, changed); // native idempotency compares the complete spec
        assert_eq!(changed.resources, task.resources);
        assert_eq!(changed.deadline, Some(task.deadline));
    }

    #[test]
    fn resident_requires_exclusive_gpu_and_explicit_checkpoint_budget() {
        let mut task = fixture();
        task.kind = TaskKind::ResidentGpu;
        assert!(task.validate_at(Timestamp::from(1)).is_err());
        task.resources.gpu = Some(GpuRequest {
            count: 1,
            vram_bytes: 1024,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        });
        task.parent_checkpoint_sha256 = Some("d".repeat(64));
        task.corpus_sha256 = Some("e".repeat(64));
        task.checkpoint_allowance_ms = Some(30_000);
        task.validate_at(Timestamp::from(1)).unwrap();
        let reserved = task.resources;
        task.resources.cpu = crate::CpuRequest::fixed(0);
        assert!(task.validate_at(Timestamp::from(1)).is_err());
        task.resources = reserved;
        task.resources.ram.soft_bytes = 0;
        assert!(task.validate_at(Timestamp::from(1)).is_err());
        task.resources = reserved;
        let dir = tempfile::tempdir().unwrap();
        let (path, hash) = write(&dir, &task);
        assert_eq!(
            spec(&path, &hash).unwrap().preemption.checkpoint,
            CheckpointPolicy::ControlFd {
                deadline_ms: 30_000
            }
        );
        task.resources.gpu.as_mut().unwrap().mode = GpuMode::Shareable;
        assert!(task.validate_at(Timestamp::from(1)).is_err());
    }

    #[test]
    fn deadline_corrupt_dependency_and_unknown_executable_fail_closed() {
        let mut task = fixture();
        assert!(task.validate_at(task.deadline).is_err());
        task.dependencies
            .insert("input".into(), "not-a-digest".into());
        assert!(task.validate_at(Timestamp::from(1)).is_err());
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["executable"] = "/bin/sh".into();
        assert!(serde_json::from_value::<TaskManifest>(value).is_err());
    }

    #[test]
    fn manifest_symlink_and_oversize_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (path, hash) = write(&dir, &fixture());
        let link = dir.path().join("link.json");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(spec(&link, &hash).is_err());
        std::fs::write(&path, vec![b' '; MAX_MANIFEST_BYTES as usize + 1]).unwrap();
        assert!(spec(&path, &hash).is_err());
    }

    #[test]
    fn nested_resource_fields_are_closed() {
        let mut task = fixture();
        task.resources.gpu = Some(GpuRequest {
            count: 1,
            vram_bytes: 1024,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        });
        let original = serde_json::to_value(task).unwrap();
        for pointer in [
            "/resources",
            "/resources/cpu",
            "/resources/ram",
            "/resources/gpu",
        ] {
            let mut value = original.clone();
            value.pointer_mut(pointer).unwrap()["unrecognized_reservation"] = 1.into();
            assert!(
                serde_json::from_value::<TaskManifest>(value).is_err(),
                "{pointer}"
            );
        }
        assert!(serde_json::from_value::<TaskManifest>(original).is_ok());
    }
}
