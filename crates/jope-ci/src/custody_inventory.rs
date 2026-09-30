//! Lossless, read-only inventory of a source tree. The partial JSONL remains on any error; only
//! an error-free scan receives the requested final name. It does not decide what may be deleted.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(out, "{byte:02x}").expect("write to String");
    }
    out
}

fn file_hash(path: &Path, before: &fs::Metadata) -> Result<String, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        hash.update(&buffer[..n]);
    }
    let after = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if bytes != before.len()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ino() != after.ino()
        || before.dev() != after.dev()
    {
        return Err("changed_during_scan".into());
    }
    Ok(hex(&hash.finalize()))
}

fn dataset_name(name: &str) -> bool {
    let hex = name.strip_prefix("synth_").unwrap_or(name);
    hex.len() == 16 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

fn explicit_origin(dir: &Path) -> Option<&'static str> {
    let meta: Value = serde_json::from_slice(&fs::read(dir.join("meta.json")).ok()?).ok()?;
    match meta.get("origin")?.as_str()? {
        "real" => Some("real"),
        "synthetic" => Some("confirmed_synthetic"),
        _ => None,
    }
}

fn shard(root: &Path, rel: &Path) -> String {
    let mut parts = rel
        .components()
        .map(|part| part.as_os_str().to_string_lossy());
    let first = parts.next().unwrap_or_default();
    if root.file_name().is_some_and(|n| n == "remote_super") && first == "quant" {
        let second = parts.next().unwrap_or_default();
        if dataset_name(&second) {
            format!("quant/root-{}", &second[..2])
        } else if second.is_empty() {
            "quant/root".into()
        } else {
            format!("quant/{second}")
        }
    } else if first.is_empty() {
        root.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    } else {
        format!(
            "{}/{}",
            root.file_name().unwrap_or_default().to_string_lossy(),
            first
        )
    }
}

/// Write one JSONL record for every reachable entry. Any unreadable/unstable entry fails the
/// census but is retained as an explicit partial record. The complete output is never appended.
pub fn run(root: &Path, out: &Path) -> Result<String, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let parent = out
        .parent()
        .ok_or("output has no parent")?
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if parent.starts_with(&root) {
        return Err("output must be outside source tree".into());
    }
    if out.exists() {
        return Err(format!("{} already exists", out.display()));
    }
    let partial = out.with_extension("jsonl.partial");
    let file = File::options()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    let mut writer = BufWriter::new(file);
    let mut classes: HashMap<PathBuf, &'static str> = HashMap::new();
    let mut entries = 0u64;
    let mut errors = 0u64;
    let mut files = 0u64;
    let mut stack = vec![root.clone()];
    while let Some(next_path) = stack.pop() {
        let path = next_path.as_path();
        let rel = path.strip_prefix(&root).map_err(|e| e.to_string())?;
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                errors += 1;
                writeln!(
                    writer,
                    "{}",
                    json!({"path_hex":hex(rel.as_os_str().as_bytes()),
                    "path":rel.to_string_lossy(),"type":"stat_error","error":error.to_string()})
                )
                .map_err(|e| e.to_string())?;
                continue;
            }
        };
        let inherited = path
            .parent()
            .and_then(|parent| classes.get(parent))
            .copied()
            .unwrap_or("unresolved");
        let in_quant = root.file_name().is_some_and(|name| name == "remote_super")
            && rel
                .components()
                .next()
                .is_some_and(|part| part.as_os_str() == "quant");
        let is_dataset = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(dataset_name);
        let class = if in_quant && metadata.file_type().is_dir() && is_dataset {
            explicit_origin(path).unwrap_or("unresolved")
        } else if inherited == "confirmed_synthetic"
            && (!metadata.file_type().is_file()
                || !matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some("train.csv" | "test.csv" | "meta.json")
                ))
        {
            "unresolved"
        } else {
            inherited
        };
        if metadata.file_type().is_dir() {
            classes.insert(path.to_path_buf(), class);
        }
        let kind = if metadata.file_type().is_dir() {
            "directory"
        } else if metadata.file_type().is_file() {
            "file"
        } else if metadata.file_type().is_symlink() {
            "symlink"
        } else {
            "special"
        };
        let mut row = json!({
            "source_root":root.display().to_string(),
            "path":rel.to_string_lossy(),
            "path_hex":hex(rel.as_os_str().as_bytes()),
            "type":kind,"size":metadata.len(),
            "mtime_sec":metadata.mtime(),"mtime_nsec":metadata.mtime_nsec(),
            "mode":metadata.mode(),"uid":metadata.uid(),"gid":metadata.gid(),
            "classification":class,"usb_shard":shard(&root,rel),
            "sha256":null,"s3_object":null,"s3_version":null
        });
        if kind == "file" {
            files += 1;
            match file_hash(path, &metadata) {
                Ok(hash) => row["sha256"] = hash.into(),
                Err(error) => {
                    row["error"] = error.into();
                    errors += 1;
                }
            }
        } else if kind == "symlink" {
            match fs::read_link(path) {
                Ok(target) => row["link_target_hex"] = hex(target.as_os_str().as_bytes()).into(),
                Err(error) => {
                    row["error"] = error.to_string().into();
                    errors += 1;
                }
            }
        } else if kind == "directory" && in_quant && is_dataset {
            let missing: Vec<&str> = ["train.csv", "test.csv", "meta.json"]
                .into_iter()
                .filter(|name| !path.join(name).is_file())
                .collect();
            if !missing.is_empty() {
                row["incomplete_dataset_missing"] = json!(missing);
            }
        }
        writeln!(writer, "{row}").map_err(|e| e.to_string())?;
        entries += 1;
        if kind == "directory" {
            match fs::read_dir(path) {
                Ok(children) => {
                    let mut paths = Vec::new();
                    for child in children {
                        match child {
                            Ok(child) => paths.push(child.path()),
                            Err(error) => {
                                errors += 1;
                                writeln!(
                                    writer,
                                    "{}",
                                    json!({"path":rel.to_string_lossy(),
                                    "type":"traversal_error","error":error.to_string()})
                                )
                                .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
                    stack.extend(paths.into_iter().rev());
                }
                Err(error) => {
                    errors += 1;
                    writeln!(
                        writer,
                        "{}",
                        json!({"path":rel.to_string_lossy(),
                        "type":"traversal_error","error":error.to_string()})
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
        }
    }
    writer.flush().map_err(|e| e.to_string())?;
    writer
        .into_inner()
        .map_err(|e| e.to_string())?
        .sync_all()
        .map_err(|e| e.to_string())?;
    if errors > 0 {
        return Err(format!(
            "{errors} unreadable or changing entries; partial: {}",
            partial.display()
        ));
    }
    fs::rename(&partial, out).map_err(|e| e.to_string())?;
    Ok(json!({"entries":entries,"files":files,"errors":errors,"out":out}).to_string())
}
