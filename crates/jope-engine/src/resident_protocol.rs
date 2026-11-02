//! CPU-testable custody and immutable completion for resident GPU tasks.
use crate::manifest::{file_hash, hash, read_json, write_json};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::unix::{fs::MetadataExt, net::UnixStream},
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "parallel.resident_request.v3";
pub const DEADLINE_ENV: &str = "JOPE_GPU_TASK_DEADLINE_MS";

/// Rank-independent task bytes for an independently signed funding demand.
/// Reservation IDs/signatures and per-rank transport identities are deliberately
/// outside these bytes. Constructing this metadata grants no execution.
pub use crate::task_contract::CommonTaskDescriptor;

pub fn common_task_descriptor(request: &Value) -> Result<CommonTaskDescriptor> {
    verify_request_inputs(request)?;
    let root = PathBuf::from(
        request["root"]
            .as_str()
            .context("common task root absent")?,
    );
    ensure!(
        root.is_absolute() && root.canonicalize()? == root,
        "common task root must be canonical"
    );
    let environment: BTreeMap<String, String> = serde_json::from_value(request["env"].clone())?;
    ensure!(
        environment.keys().all(|key| [
            "JOPE_CHAMPION_SNAPSHOT",
            "JOPE_GP_EXECUTABLE",
            "JOPE_EXPERIMENT_DIR"
        ]
        .contains(&key.as_str())),
        "unbound common task environment"
    );
    let executable_sha256 = request["executable_sha256"]
        .as_str()
        .context("common executable absent")?;
    ensure!(
        executable_sha256.len() == 64
            && executable_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "invalid common executable digest"
    );
    let deadline_unix_ms = request["deadline_unix_ms"]
        .as_u64()
        .context("common original deadline absent")?;
    remaining(deadline_unix_ms)?;
    Ok(CommonTaskDescriptor {
        schema: "parallel.resident_common_task.v1".into(),
        root,
        task: request["task"].clone(),
        environment,
        input_artifacts: serde_json::from_value(request["input_artifacts"].clone())?,
        executable_sha256: executable_sha256.into(),
        deadline_unix_ms,
    })
}

pub fn now_ms() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

pub fn remaining(deadline_ms: u64) -> Result<std::time::Duration> {
    let left = deadline_ms
        .checked_sub(now_ms()?)
        .filter(|left| *left > 0)
        .context("resident task absolute deadline elapsed")?;
    Ok(std::time::Duration::from_millis(left))
}

pub fn inherited_deadline() -> Result<u64> {
    let deadline = std::env::var(DEADLINE_ENV)
        .context("resident task requires a bound absolute deadline")?
        .parse()?;
    remaining(deadline)?;
    Ok(deadline)
}

pub fn read_line_until(stream: &mut UnixStream, limit: usize, deadline: u64) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        ensure!(
            bytes.len() < limit,
            "resident IPC message exceeds its bound"
        );
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        let mut block = [0u8; 4096];
        let capacity = block.len().min(limit - bytes.len());
        let count = stream.read(&mut block[..capacity])?;
        ensure!(count > 0, "resident IPC closed before a complete message");
        bytes.extend_from_slice(&block[..count]);
        remaining(deadline)?;
        if let Some(end) = bytes.iter().position(|byte| *byte == b'\n') {
            ensure!(
                end + 1 == bytes.len(),
                "resident IPC contains trailing messages"
            );
            return Ok(String::from_utf8(bytes)?);
        }
    }
}

pub fn write_until(stream: &mut UnixStream, mut bytes: &[u8], deadline: u64) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        let written = stream.write(bytes)?;
        ensure!(written > 0, "resident IPC closed during write");
        bytes = &bytes[written..];
    }
    remaining(deadline)?;
    Ok(())
}

mod artifacts;
pub use artifacts::*;
mod dispatch;
pub use dispatch::*;
