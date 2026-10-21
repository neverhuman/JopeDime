//! Atomic outputs and input/output aliasing (SPECIFICATION.md §13.3).
//!
//! Results are written to a fresh temporary file in the destination directory, synced, and then
//! renamed over the destination, so a failed run leaves the previous file intact and a reader
//! never sees a partial one. An output that names an input is refused before any work.
use jope_runtime::receipt::FileRecord;
use jope_runtime::Failure;
use std::io::Write;
use std::path::{Path, PathBuf};

fn identity(path: &Path) -> Option<PathBuf> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    Some(std::fs::canonicalize(parent).ok()?.join(path.file_name()?))
}

/// Refuse an output path that is also an input.
pub fn refuse_aliasing(output: &Path, inputs: &[&Path]) -> anyhow::Result<()> {
    let out = identity(output).ok_or_else(|| {
        Failure::output(format!(
            "{}: the output directory does not exist",
            output.display()
        ))
    })?;
    for input in inputs {
        let resolved = |p: &Path| std::fs::canonicalize(p).ok();
        let same = identity(input).is_some_and(|i| i == out)
            || resolved(input).is_some_and(|i| resolved(output) == Some(i));
        if same {
            return Err(Failure::input(format!(
                "output {} is also an input",
                output.display()
            )));
        }
    }
    Ok(())
}

/// Write `bytes` to `path` atomically and return its record.
pub fn write_atomic(role: &'static str, path: &Path, bytes: &[u8]) -> anyhow::Result<FileRecord> {
    let output = |e: std::io::Error| Failure::output(format!("{}: {e}", path.display()));
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_owned(),
        _ => PathBuf::from("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| Failure::input(format!("{}: invalid output path", path.display())))?;
    let temp = dir.join(format!(
        ".{}.jopedime-{}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(output(error));
    }
    Ok(FileRecord {
        role,
        path: path.display().to_string(),
        bytes: bytes.len() as u64,
        sha256: jope_runtime::sha256_hex(bytes),
    })
}
