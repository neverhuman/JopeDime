//! Only the root-qualified fit authority may place its own stopped numerical child.
use super::*;
use jain_smartcluster_core::{AttachTrustedFitChildRequest, TrustedFitChildPlacement, Workload};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
struct FitPlacementIntent {
    request: AttachTrustedFitChildRequest,
    authority_pid: u32,
    authority_birth: Option<u64>,
    authority_cgroup: PathBuf,
    target_cgroup: PathBuf,
}

fn refused(message: impl Into<String>) -> ServerError {
    ServerError::new(ErrorCode::Forbidden, message)
}
fn unavailable_fit(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(ErrorCode::Unavailable, error.to_string())
}
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn file_digest(path: &Path) -> Result<String, ServerError> {
    let mut file = fs::File::open(path).map_err(unavailable_fit)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(unavailable_fit)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}
fn bounded_json(path: &Path, sha: &str) -> Result<serde_json::Value, ServerError> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
        .open(path)
        .map_err(unavailable_fit)?;
    if !file.metadata().map_err(unavailable_fit)?.is_file() {
        return Err(refused("manifest is not regular"));
    }
    let mut bytes = Vec::new();
    file.take(1048577)
        .read_to_end(&mut bytes)
        .map_err(unavailable_fit)?;
    if bytes.len() > 1048576 || digest(&bytes) != sha {
        return Err(refused("manifest bytes changed or exceeded bound"));
    }
    serde_json::from_slice(&bytes).map_err(unavailable_fit)
}
#[derive(Debug, Eq, PartialEq)]
struct ProcessState {
    parent: u32,
    birth: u64,
    stopped: bool,
    dead: bool,
}
fn parse_stat(text: &str) -> Result<ProcessState, ServerError> {
    let (_, fields) = text
        .rsplit_once(") ")
        .ok_or_else(|| refused("invalid process stat"))?;
    let fields: Vec<_> = fields.split_whitespace().collect();
    if fields.len() < 20 {
        return Err(refused("short process stat"));
    }
    Ok(ProcessState {
        parent: fields[1].parse().map_err(unavailable_fit)?,
        birth: fields[19].parse().map_err(unavailable_fit)?,
        dead: matches!(fields[0], "Z" | "X" | "x"),
        stopped: fields[0] == "T", // traced children ('t') are not accepted
    })
}
fn proc_state(pid: u32) -> Result<ProcessState, ServerError> {
    parse_stat(&fs::read_to_string(format!("/proc/{pid}/stat")).map_err(unavailable_fit)?)
}
fn proc_cgroup(pid: u32) -> Result<PathBuf, ServerError> {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup")).map_err(unavailable_fit)?;
    let relative = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| refused("unified cgroup absent"))?;
    Ok(Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}
fn service(job: &JobRecord) -> Result<(&str, &[String]), ServerError> {
    match &job.spec.workload {
        Workload::TrustedService {
            service, arguments, ..
        } => Ok((service.as_str(), arguments)),
        _ => Err(refused("native trusted service required")),
    }
}
fn live(job: &JobRecord, attempt: FencedAttempt) -> Result<(), ServerError> {
    if job.active_attempt != Some(attempt)
        || !matches!(job.state, JobState::Starting | JobState::Running)
        || job
            .spec
            .deadline
            .is_some_and(|deadline| deadline <= now_timestamp())
    {
        return Err(ServerError::new(
            ErrorCode::SupersededAttempt,
            "placement attempt is no longer live",
        ));
    }
    Ok(())
}
fn validate_candidate(
    job: &JobRecord,
    request: &AttachTrustedFitChildRequest,
) -> Result<(), ServerError> {
    let (name, args) = service(job)?;
    if !matches!(name, "jope-gp-search" | "jope-continuation")
        || args.len() != 4
        || args[0] != "--task-manifest"
        || args[2] != "--manifest-sha256"
        || args[3] != request.task_manifest_sha256
    {
        return Err(refused("target is not the exact native GP manifest"));
    }
    let path = Path::new(&args[1]);
    if !path.is_absolute() {
        return Err(refused("target manifest path is not absolute"));
    }
    let manifest = bounded_json(path, &request.task_manifest_sha256)?;
    let task: jain_smartcluster::jope::TaskManifest =
        serde_json::from_value(manifest.clone()).map_err(unavailable_fit)?;
    task.validate_at(now_timestamp()).map_err(unavailable_fit)?;
    // Reconstruct all resource/deadline/idempotency fields, not merely the path hint.
    let spec = jain_smartcluster::jope::job_spec(
        path,
        &request.task_manifest_sha256,
        job.spec.tenant.clone(),
        job.spec.project.clone(),
        now_timestamp(),
    )
    .map_err(unavailable_fit)?;
    if spec != job.spec {
        return Err(refused("target JobSpec differs from native task"));
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| refused("manifest stem"))?;
    let input = bounded_json(
        &path.with_file_name(format!("{stem}.input.json")),
        &task.input_manifest_sha256,
    )?;
    let operation = &input["operation"];
    if input["schema"] != "jope.smartcluster_input.v1" {
        return Err(refused("native input schema"));
    }
    let evaluations = match operation["kind"].as_str() {
        Some("Search") => {
            if (name == "jope-continuation") == operation["seeds"].is_null() {
                return Err(refused("legacy Search fixed service mismatch"));
            }
            operation["evaluations"].as_u64()
        }
        Some(kind @ ("DirectEvaluation" | "SeededContinuation")) => {
            let direct = kind == "DirectEvaluation";
            if direct != (name == "jope-gp-search") {
                return Err(refused("native genome fixed service mismatch"));
            }
            let root = Path::new(
                input["root"]
                    .as_str()
                    .ok_or_else(|| refused("native root"))?,
            );
            let nested = Path::new(
                operation["manifest"]
                    .as_str()
                    .ok_or_else(|| refused("genome manifest"))?,
            );
            if !root.is_absolute()
                || nested.as_os_str().is_empty()
                || nested
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(refused("genome manifest path escape"));
            }
            let genome = bounded_json(
                &root.join(nested),
                operation["sha256"]
                    .as_str()
                    .ok_or_else(|| refused("genome manifest digest"))?,
            )?;
            let n = genome["evaluations"].as_u64();
            if genome["schema"] != "jope.native_genome_task.v1"
                || genome["stage"] != if direct { "direct" } else { "continuation" }
                || (direct && (n != Some(1) || !genome["direct_parent"].is_null()))
                || (!direct && genome["direct_parent"].is_null())
                || genome["seconds"].as_u64().is_none_or(|n| n == 0 || n > 600)
            {
                return Err(refused("native genome stage or budget differs"));
            }
            let candidate_ref = &genome["candidate"];
            let path = Path::new(
                candidate_ref["path"]
                    .as_str()
                    .ok_or_else(|| refused("candidate path"))?,
            );
            if path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(refused("candidate path escape"));
            }
            let candidate = bounded_json(
                &root.join(path),
                candidate_ref["sha256"]
                    .as_str()
                    .ok_or_else(|| refused("candidate digest"))?,
            )?;
            if candidate["schema"] != "jope.native_candidate.v1"
                || candidate["run"] != task.run
                || candidate["generation"] != task.generation
                || candidate["context"]["partition"] != "meta_train"
            {
                return Err(refused(
                    "candidate run/generation/scientific partition differs",
                ));
            }
            n
        }
        _ => return Err(refused("unsupported native numerical operation")),
    };
    if evaluations.is_none_or(|n| n == 0 || n > 1000 || u64::from(request.candidate_index) >= n) {
        return Err(refused("candidate exceeds original numerical budget"));
    }
    Ok(())
}

mod actor;
#[cfg(test)]
mod tests;
