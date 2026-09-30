//! Allowlisted Cargo test-executable janitor. Fingerprint JSON proves the artifact is a test;
//! identity includes target, profile, features and rustflags. Every run writes a dry-run receipt.
//! Worktrees and all other `target/` contents are outside its deletion surface.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct Candidate {
    path: PathBuf,
    identity: String,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
}

fn test_target(name: &str) -> Option<(&'static str, &str)> {
    let stem = name.strip_suffix(".json")?;
    for prefix in [
        "test-integration-test-",
        "test-bin-",
        "test-example-",
        "test-lib-",
    ] {
        if let Some(target) = stem.strip_prefix(prefix) {
            if !target.is_empty() {
                return Some((prefix, target));
            }
        }
    }
    None
}

fn is_elf_executable(path: &Path) -> bool {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return false;
    };
    if !meta.file_type().is_file() || meta.permissions().mode() & 0o111 == 0 {
        return false;
    }
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && magic == *b"\x7fELF"
}

fn fingerprint_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for directory in fs::read_dir(root).map_err(|e| e.to_string())? {
        let directory = directory.map_err(|e| e.to_string())?;
        if !directory.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        for file in fs::read_dir(directory.path()).map_err(|e| e.to_string())? {
            let file = file.map_err(|e| e.to_string())?;
            if file.file_type().map_err(|e| e.to_string())?.is_file() {
                files.push(file.path());
            }
        }
    }
    Ok(files)
}

fn scan_target(target_dir: &Path) -> Result<Vec<Candidate>, String> {
    let mut found = Vec::new();
    for profile in ["debug", "release"] {
        let fingerprints = target_dir.join(profile).join(".fingerprint");
        if !fingerprints.is_dir() {
            continue;
        }
        for fingerprint_path in fingerprint_files(&fingerprints)? {
            let Some((kind, test)) = fingerprint_path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(test_target)
            else {
                continue;
            };
            let parent = fingerprint_path
                .parent()
                .ok_or("fingerprint has no parent")?;
            let package_hash = parent
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("invalid fingerprint directory")?;
            let Some((package, hash)) = package_hash.rsplit_once('-') else {
                continue;
            };
            if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                continue;
            }
            let path = target_dir
                .join(profile)
                .join("deps")
                .join(format!("{test}-{hash}"));
            if !is_elf_executable(&path) {
                continue;
            }
            let fingerprint: Value =
                serde_json::from_slice(&fs::read(&fingerprint_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let identity = json!({"package":package,"kind":kind,"test":test,
                "profile_dir":profile,"target":fingerprint.get("target"),
                "profile":fingerprint.get("profile"),"features":fingerprint.get("features"),
                "rustflags":fingerprint.get("rustflags"),
                "compile_kind":fingerprint.get("compile_kind")})
            .to_string();
            let meta = fs::metadata(&path).map_err(|e| e.to_string())?;
            found.push(Candidate {
                path,
                identity,
                size: meta.len(),
                mtime_sec: meta.mtime(),
                mtime_nsec: meta.mtime_nsec(),
            });
        }
    }
    Ok(found)
}

fn digest(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn df_free() -> Result<u64, String> {
    let output = Command::new("/usr/bin/df")
        .args(["-B1", "--output=avail", "/"])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("df failed".into());
    }
    String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .lines()
        .nth(1)
        .ok_or("df returned no available bytes")?
        .trim()
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())
}

fn active_build() -> bool {
    let Ok(entries) = fs::read_dir("/proc") else {
        return true;
    };
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .as_encoded_bytes()
            .iter()
            .all(u8::is_ascii_digit)
        {
            continue;
        }
        let Ok(comm) = fs::read_to_string(entry.path().join("comm")) else {
            continue;
        };
        if matches!(comm.trim(), "cargo" | "rustc" | "nextest" | "cargo-nextest") {
            return true;
        }
    }
    false
}

fn unused(path: &Path) -> Result<bool, String> {
    let output = Command::new("/usr/bin/fuser")
        .arg("-s")
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.stderr.is_empty() {
        return Err(format!("fuser could not inspect {}", path.display()));
    }
    match output.status.code() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(format!("fuser could not inspect {}", path.display())),
    }
}

fn git_probe(target: &Path) -> Value {
    let checkout = target.parent().unwrap_or(target);
    let probe = |args: &[&str]| -> Value {
        match Command::new("git")
            .arg("-C")
            .arg(checkout)
            .args(args)
            .output()
        {
            Ok(output) => json!({"exit":output.status.code(),
                "lines":output.stdout.iter().filter(|byte| **byte == b'\n').count()}),
            Err(error) => json!({"error":error.to_string()}),
        }
    };
    json!({"checkout":checkout,
        "status":probe(&["status","--porcelain=v1"]),
        "stashes":probe(&["stash","list"]),
        "local_only_commits":probe(&["log","--all","--not","--remotes","--oneline"]),
        "remote_tags":probe(&["ls-remote","--tags","origin"])})
}

/// `config` allows exact target directories and an explicit delete switch. `custody` receives
/// immutable time-stamped receipts. A dry-run receipt is written before any file is removed.
pub fn run(config: &Path, custody: &Path, apply: bool) -> Result<String, String> {
    let cfg: Value = serde_json::from_slice(&fs::read(config).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let roots = cfg
        .get("target_dirs")
        .and_then(Value::as_array)
        .ok_or("config needs target_dirs array")?;
    let hours = cfg
        .get("min_age_hours")
        .and_then(Value::as_u64)
        .unwrap_or(48);
    if apply && cfg.get("delete_enabled") != Some(&Value::Bool(true)) {
        return Err("delete_enabled is not true".into());
    }
    fs::create_dir_all(custody).map_err(|e| e.to_string())?;
    let custody = custody.canonicalize().map_err(|e| e.to_string())?;
    if custody.starts_with("/tmp") {
        return Err("custody cannot be under /tmp".into());
    }
    let mut allowed = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        let text = root.as_str().ok_or("target_dir must be a string")?;
        let path = PathBuf::from(text);
        let canonical = path.canonicalize().map_err(|e| e.to_string())?;
        if canonical != path
            || canonical.file_name().is_none_or(|name| name != "target")
            || custody.starts_with(&canonical)
            || !seen.insert(path.clone())
        {
            return Err(format!("invalid or duplicate allowlisted target: {text}"));
        }
        allowed.push(path);
    }
    let mut all = Vec::new();
    for target in &allowed {
        all.extend(scan_target(target)?);
    }
    let mut newest: HashMap<String, usize> = HashMap::new();
    for (i, candidate) in all.iter().enumerate() {
        let key = candidate.identity.as_str();
        match newest.get(key).copied() {
            None => {
                newest.insert(key.to_owned(), i);
            }
            Some(old)
                if (candidate.mtime_sec, candidate.mtime_nsec, &candidate.path)
                    > (all[old].mtime_sec, all[old].mtime_nsec, &all[old].path) =>
            {
                newest.insert(key.to_owned(), i);
            }
            _ => {}
        }
    }
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?;
    let now = clock.as_secs();
    let cutoff = now.saturating_sub(hours.saturating_mul(3600));
    let mut candidates: Vec<(Candidate, String)> = Vec::new();
    for (i, candidate) in all.into_iter().enumerate() {
        if candidate.mtime_sec >= 0
            && candidate.mtime_sec as u64 <= cutoff
            && newest.get(candidate.identity.as_str()) != Some(&i)
        {
            let sha = digest(&candidate.path)?;
            candidates.push((candidate, sha));
        }
    }
    candidates.sort_by(|left, right| left.0.path.cmp(&right.0.path));
    let stamp = clock.as_nanos().to_string();
    let receipt = custody.join(format!("janitor-{stamp}-dry-run.json"));
    let before = df_free()?;
    let listed: Vec<Value> = candidates
        .iter()
        .map(|(item, sha)| {
            json!({
                "path":item.path,"identity":item.identity,"bytes":item.size,
                "mtime_sec":item.mtime_sec,"mtime_nsec":item.mtime_nsec,"sha256":sha
            })
        })
        .collect();
    let git_checks: Vec<Value> = allowed.iter().map(|target| git_probe(target)).collect();
    let document = json!({"schema":"jope.janitor/v1","at":now,
        "allowlisted_targets":allowed,"min_age_hours":hours,"apply_requested":apply,
        "active_build":active_build(),"root_free_before":before,"candidates":listed,
        "git_checks":git_checks});
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(&receipt)
        .map_err(|e| e.to_string())?;
    output
        .write_all(
            serde_json::to_string_pretty(&document)
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    let mut deleted = Vec::new();
    let mut skipped = Vec::new();
    if apply {
        for (candidate, sha) in &candidates {
            if active_build() || !unused(&candidate.path)? {
                skipped.push(json!({"path":candidate.path,"reason":"active build or process"}));
                continue;
            }
            let meta = fs::metadata(&candidate.path).map_err(|e| e.to_string())?;
            if meta.len() != candidate.size
                || meta.mtime() != candidate.mtime_sec
                || meta.mtime_nsec() != candidate.mtime_nsec
                || digest(&candidate.path)? != *sha
            {
                skipped.push(json!({"path":candidate.path,"reason":"candidate changed"}));
                continue;
            }
            fs::remove_file(&candidate.path).map_err(|e| e.to_string())?;
            deleted.push(json!({"path":candidate.path,"bytes":candidate.size,"sha256":sha}));
        }
    }
    let after = df_free()?;
    let result = json!({"schema":"jope.janitor-result/v1","dry_run_receipt":receipt,
        "deleted":deleted,"skipped":skipped,"root_free_before":before,
        "root_free_after":after,"actual_recovered_bytes":after.saturating_sub(before)});
    let result_path = custody.join(format!("janitor-{stamp}-result.json"));
    fs::write(
        &result_path,
        serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(result.to_string())
}
