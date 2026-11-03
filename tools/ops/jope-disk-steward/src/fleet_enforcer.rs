//! Approved idle build-cache retention. No S3/per-file provenance or process signals.
//! Native Rust replacement for `fleet_enforcer.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::bounded_probe;
use crate::cache_locks::{fd_identity, CargoCacheLocks};
use crate::cache_namespaces::{
    independent_shared_targets, is_shared_child, is_symlink, protected_target_subtree, slot_source,
};
use crate::inherited_reference;
use crate::json::Json;
use crate::json_obj;
use crate::sha256::sha256_str;
use crate::source_preservation;
use crate::sys::{
    atomic_write_str, base_dir, close, current_uid, disk_usage, emit_jsonl_locked,
    epoch_to_utc_iso, home_dir, hostname, now_epoch_f64, open, utc_iso, utc_stamp_compact,
    FlockGuard, O_NOFOLLOW, O_RDONLY,
};

pub const APPROVAL: &str = "jepson_approval_build_output_t0135.md; Jepson 2026-11-02";

const PROTECTED_TERMS: &[&str] = &[
    "evidence",
    "custody",
    "preserv",
    "receipt",
    "signed",
    "proof",
    "health",
    "genome",
    "score",
    "winner",
    "leader",
    "golden",
    "fixture",
    "research",
    "jope_plan",
    "rc15",
    "rc16",
    "smoke",
    "render",
];

const EXTRA_PROTECTED_WAVES: &[&str] = &[
    "h199",
    "h200",
    "h211",
    "r882",
    "r785",
    "jankurai_audit_64be8b477",
    "jankurai_rerun",
];

const BUILD_DIRS: &[&str] = &[
    "debug",
    "release",
    "deps",
    "incremental",
    "build",
    ".fingerprint",
];

pub fn is_protected_str(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    PROTECTED_TERMS.iter().any(|t| lower.contains(t))
}

fn is_excluded_root_str(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    is_protected_str(&lower) || EXTRA_PROTECTED_WAVES.iter().any(|w| lower.contains(w))
}

pub fn expand_glob(base: &Path, pattern: &str) -> Vec<PathBuf> {
    let parts: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let mut current = vec![base.to_path_buf()];
    for part in parts {
        let mut next = Vec::new();
        for dir in current {
            if part == "*" {
                if let Ok(rd) = fs::read_dir(&dir) {
                    for entry in rd.flatten() {
                        next.push(entry.path());
                    }
                }
            } else {
                let candidate = dir.join(part);
                if candidate.exists() {
                    next.push(candidate);
                }
            }
        }
        current = next;
    }
    current
}

pub fn discover() -> Vec<PathBuf> {
    let home = home_dir();
    let patterns = [
        "*/target",
        "repos/*/target",
        "wt/*/target",
        ".agent/worktrees/*/target",
        "scratch/*/target",
        "scratch/*/*/target",
        "lane_work/*/target",
        "actions-runners/*/_work/*/*/target",
        "actions-runners/*/*/_work/*/*/target",
        "gate-runner/build-cache/target",
        "jope-ci-shards/*/target",
        "jk_lanes/*/target",
        "jk_lanes/*/*/target",
        "jk_offhost/*/target",
        "cifix/pool/*/target",
        "cifix/pool/*/*/target",
        "remote_tests/target",
        ".cache/*/target",
        "*/.agent/worktrees/*/target",
        "*/.agent/worktrees/*/*/target",
    ];
    let mut found = Vec::new();
    for pat in patterns {
        found.extend(expand_glob(&home, pat));
    }
    let children = independent_shared_targets(&home);
    if !children.is_empty() {
        let parent = home.join(".cache/jope/target");
        found.retain(|p| p != &parent);
        found.extend(children);
    }
    let native = Path::new("/mnt/jope-build-native");
    if native.is_dir() {
        for pat in ["*/target", "*/*/target", "*/*/*/target"] {
            found.extend(expand_glob(native, pat));
        }
    }

    let mut unique = BTreeSet::new();
    for p in found {
        if !p.is_dir() || is_symlink(&p) {
            continue;
        }
        let Ok(resolved) = fs::canonicalize(&p) else {
            continue;
        };
        if resolved != p {
            continue;
        }
        let s = p.to_string_lossy();
        if is_excluded_root_str(&s) || protected_target_subtree(&p) {
            continue;
        }
        unique.insert(p);
    }
    unique.into_iter().collect()
}

pub fn inventory(root: &Path) -> (Vec<(PathBuf, Metadata)>, i64, Vec<String>) {
    let mut entries = Vec::new();
    let mut newest: i64 = 0;
    let mut errors = Vec::new();
    let root_meta = match fs::metadata(root) {
        Ok(m) => m,
        Err(e) => return (entries, 0, vec![e.to_string()]),
    };
    let dev = root_meta.dev();
    newest = newest.max(root_meta.mtime());

    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("{}: {e}", dir.display()));
                continue;
            }
        };
        let mut dir_entries: Vec<PathBuf> = Vec::new();
        for item in rd {
            match item {
                Ok(e) => dir_entries.push(e.path()),
                Err(e) => errors.push(format!("{}: {e}", dir.display())),
            }
        }
        dir_entries.sort();
        for p in dir_entries {
            match fs::symlink_metadata(&p) {
                Ok(s) => {
                    newest = newest.max(s.mtime());
                    if s.dev() != dev {
                        errors.push(format!("foreign filesystem: {}", p.display()));
                    }
                    let is_dir = s.file_type().is_dir() && !s.file_type().is_symlink();
                    entries.push((p.clone(), s));
                    if is_dir {
                        stack.push(p);
                    }
                }
                Err(e) => errors.push(format!("{}: {e}", p.display())),
            }
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    (entries, newest, errors)
}

pub fn classify(
    root: &Path,
    path: &Path,
    s: &Metadata,
    allow_internal_links: bool,
) -> std::io::Result<bool> {
    let Ok(rel) = path.strip_prefix(root) else {
        return Ok(false);
    };
    let is_reg = s.file_type().is_file() && !s.file_type().is_symlink();
    if !is_reg || s.uid() != current_uid() || (s.nlink() != 1 && !allow_internal_links) {
        return Ok(false);
    }
    let rel_str = rel.to_string_lossy();
    if is_protected_str(&rel_str) {
        return Ok(false);
    }
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if parts.iter().any(|p| {
        matches!(
            p.as_str(),
            "tmp" | "runs" | "data" | "src" | ".git" | "testdata" | "tests" | "examples"
        )
    }) {
        return Ok(false);
    }
    if parts.len() < 2
        || !parts[..parts.len() - 1]
            .iter()
            .any(|p| BUILD_DIRS.contains(&p.as_str()))
    {
        return Ok(false);
    }

    let mut f = File::open(path)?;
    let mut magic = [0u8; 16];
    let n = f.read(&mut magic)?;
    let slice = &magic[..n];

    let is_compiler_magic = slice.starts_with(b"\x7fELF")
        || slice.starts_with(b"!<arch>\n")
        || slice.starts_with(b"rust\x00")
        || slice.starts_with(b"rust\x01")
        || slice.starts_with(b"RUST");

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_default();

    if is_compiler_magic {
        let denied_ext = matches!(
            ext.as_str(),
            ".csv" | ".json" | ".npz" | ".step" | ".stl" | ".pt" | ".pth" | ".pkl"
        );
        return Ok(!denied_ext);
    }
    if matches!(ext.as_str(), ".o" | ".obj" | ".bc")
        && (slice.starts_with(b"BC\xc0\xde") || slice.starts_with(b"\xde\xc0\x17\x0b"))
    {
        return Ok(true);
    }
    Ok(false)
}

pub fn select_groups(
    root: &Path,
    entries: &[(PathBuf, Metadata)],
) -> std::io::Result<Vec<Vec<(PathBuf, Metadata)>>> {
    let mut groups: BTreeMap<(u64, u64), Vec<(PathBuf, Metadata)>> = BTreeMap::new();
    for (p, s) in entries {
        if classify(root, p, s, true)? {
            groups
                .entry((s.dev(), s.ino()))
                .or_default()
                .push((p.clone(), s.clone()));
        }
    }
    Ok(groups
        .into_values()
        .filter(|g| !g.is_empty() && (g.len() as u64) == g[0].1.nlink())
        .collect())
}

fn unlink_group(group: &[(PathBuf, Metadata)], manifest: &Path) -> Result<(u64, usize), String> {
    let (first, _old) = &group[0];
    let c_first = CString::new(first.as_os_str().as_bytes())
        .map_err(|_| "NUL in path".to_string())?;
    let fd = unsafe { open(c_first.as_ptr(), O_RDONLY | O_NOFOLLOW, 0) };
    if fd < 0 {
        return Err(format!(
            "open {}: {}",
            first.display(),
            std::io::Error::last_os_error()
        ));
    }
    let mut freed: u64 = 0;
    let mut count: usize = 0;
    let res = (|| -> Result<(), String> {
        for (idx, (p, s)) in group.iter().enumerate() {
            let now = fs::symlink_metadata(p).map_err(|e| format!("lstat {}: {e}", p.display()))?;
            let opened = fd_identity(fd)?;
            let baseline = (
                s.dev(),
                s.ino(),
                s.size(),
                s.mtime_nsec(),
                s.uid(),
                s.mode(),
            );
            let observed = (
                now.dev(),
                now.ino(),
                now.size(),
                now.mtime_nsec(),
                now.uid(),
                now.mode(),
            );
            if observed != baseline || now.nlink() != (group.len() - idx) as u64 {
                return Err("inode baseline or link count changed".to_string());
            }
            if (now.dev(), now.ino(), now.ctime_nsec())
                != (opened.dev, opened.ino, opened.ctime_ns)
            {
                return Err("opened descriptor identity mismatch".to_string());
            }
            if idx == 0 && now.ctime_nsec() != s.ctime_nsec() {
                return Err("initial ctime changed".to_string());
            }
            let allocated = if now.nlink() == 1 {
                now.blocks().saturating_mul(512)
            } else {
                0
            };
            let intent = json_obj! {
                "path" => p.to_string_lossy().to_string(),
                "inode" => s.ino(),
                "allocated_bytes" => allocated,
                "event" => "unlink_intent",
            };
            emit_jsonl_locked(manifest, &intent.to_compact())
                .map_err(|e| format!("emit unlink_intent: {e}"))?;
            fs::remove_file(p).map_err(|e| format!("unlink {}: {e}", p.display()))?;
            let deleted = json_obj! {
                "path" => p.to_string_lossy().to_string(),
                "inode" => s.ino(),
                "allocated_bytes" => allocated,
                "event" => "deleted",
            };
            emit_jsonl_locked(manifest, &deleted.to_compact())
                .map_err(|e| format!("emit deleted: {e}"))?;
            freed = freed.saturating_add(allocated);
            count += 1;
        }
        Ok(())
    })();
    unsafe {
        let _ = close(fd);
    }
    res?;
    Ok((freed, count))
}

pub fn consumers(roots: &[PathBuf], maintenance_locks: Option<&CargoCacheLocks>) -> Json {
    let base = base_dir();
    let mut raw = bounded_probe::request(roots, &base, 120, bounded_probe::UNIT, None);
    if let Some(locks) = maintenance_locks {
        if let Ok(filtered) = locks.filter_probe(raw.clone()) {
            raw = filtered;
        }
    }
    inherited_reference::decorate(raw, &base)
}

pub fn source_commit(root: &Path) -> Option<Json> {
    let mut candidates = Vec::new();
    if let Some(parent) = root.parent() {
        let pointer = parent.join("source");
        if pointer.exists() {
            if let Ok(resolved) = fs::canonicalize(&pointer) {
                candidates.push(resolved);
            }
        }
        candidates.push(parent.to_path_buf());
        for p in expand_glob(parent, "*/Cargo.toml") {
            if let Some(cp) = p.parent() {
                candidates.push(cp.to_path_buf());
            }
        }
    }
    let home = home_dir();
    if root == home.join("remote_tests/target")
        || root == home.join(".cache/jope/target")
        || is_shared_child(root)
    {
        let project = home.join("JopeDime");
        if let Some(lane) = slot_source(root, &home) {
            if lane.is_dir() && project.join(".git").exists() {
                candidates.insert(0, lane);
            }
        }
        if project.join(".git").exists() {
            candidates.push(project);
            candidates.push(home.join("jk_lanes/rustmem_b/JopeDime"));
            candidates.push(home.join("audit/JopeDime_main"));
        }
    } else if root.starts_with("/mnt/jope-build-native") {
        let audit_repo = home.join("audit/JopeDime_main");
        if audit_repo.join(".git").exists() {
            candidates.push(audit_repo);
        }
        let main_repo = home.join("JopeDime");
        if main_repo.join(".git").exists() {
            candidates.push(main_repo);
        }
    }

    for cwd in candidates {
        let out = Command::new("git")
            .arg("-C")
            .arg(&cwd)
            .args(["rev-parse", "--verify", "HEAD^{commit}"])
            .output();
        if let Ok(o) = out {
            if o.status.success() {
                let commit = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !commit.is_empty() {
                    return Some(json_obj! {
                        "source" => cwd.to_string_lossy().to_string(),
                        "commit" => commit,
                    });
                }
            }
        }
    }
    None
}

fn docker_images(out: &Path) -> Json {
    let base = base_dir();
    let marker = base.join("state/docker-image-prune-last.json");
    if marker.exists() {
        if let Ok(j) = Json::load_file(&marker) {
            if let Some(ep) = j.get("epoch").and_then(|v| v.as_f64()) {
                if now_epoch_f64() - ep < 600.0 {
                    return json_obj! { "status" => "held_duplicate_attempt_cooldown" };
                }
            }
        }
    }
    if let Ok(d) = disk_usage(Path::new("/")) {
        if d.used_percent < 85.0 {
            return json_obj! { "status" => "floor_already_reached" };
        }
    }
    let Ok(p) = Command::new("sudo")
        .args(["-n", "docker", "image", "prune", "-a", "-f"])
        .output()
    else {
        return json_obj! { "status" => "failed" };
    };
    let stdout = String::from_utf8_lossy(&p.stdout).to_string();
    let stderr = String::from_utf8_lossy(&p.stderr).to_string();
    let _ = fs::write(out.join("docker-image-prune.stdout"), &stdout);
    let _ = fs::write(out.join("docker-image-prune.stderr"), &stderr);
    let reclaimed_line = stdout
        .lines()
        .find(|l| l.starts_with("Total reclaimed space:"))
        .map(|s| Json::Str(s.to_string()))
        .unwrap_or(Json::Null);
    let rc = p.status.code().unwrap_or(1);
    let row = json_obj! {
        "utc" => utc_iso(),
        "host" => hostname(),
        "path" => "docker unused images",
        "event" => "docker_image_prune_batch",
        "status" => if rc == 0 { "complete" } else { "failed" },
        "returncode" => rc,
        "docker_reported_reclamation" => reclaimed_line,
        "receipt_dir" => out.to_string_lossy().to_string(),
        "containers_or_volumes_touched" => false,
    };
    if rc == 0 {
        let _ = atomic_write_str(
            &marker,
            &format!(
                "{}\n",
                json_obj! {
                    "epoch" => now_epoch_f64(),
                    "utc" => utc_iso(),
                    "receipt" => out.to_string_lossy().to_string(),
                }
                .to_compact()
            ),
        );
        let _ = emit_jsonl_locked(&base.join("DELETED.log"), &row.to_compact());
    }
    row
}

fn sweep_advice(root: &Path, source: &Json, out: &Path) -> Json {
    if root.starts_with("/mnt/jope-build-native") || is_shared_child(root) {
        return json_obj! {
            "status" => "skipped_external_source_checkout",
            "reason" => "separate cache source pointer may refer to another active checkout; compiler-only executor and source commit check still apply",
        };
    }
    let base = base_dir();
    let binary = [base.join("tools/bin/cargo-sweep"), base.join("bin/cargo-sweep")]
        .into_iter()
        .find(|p| p.is_file());
    let Some(bin_path) = binary else {
        return json_obj! { "status" => "not_installed" };
    };
    let name = format!("{}.cargo-sweep", sha256_str(&root.to_string_lossy()));
    let src_dir = source.get_str("source").unwrap_or(".");
    let path_env = format!(
        "{}/.cargo/bin:{}",
        home_dir().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out_res = Command::new(&bin_path)
        .args(["sweep", "--dry-run", "--time", "2", src_dir])
        .env("CARGO_TARGET_DIR", root)
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_INCREMENTAL", "0")
        .env("PATH", path_env)
        .output();
    let rc = match out_res {
        Ok(o) => {
            let _ = fs::write(out.join(format!("{name}.stdout")), &o.stdout);
            let _ = fs::write(out.join(format!("{name}.stderr")), &o.stderr);
            o.status.code().unwrap_or(1)
        }
        Err(_) => 1,
    };
    json_obj! {
        "status" => "read_only_advisory",
        "returncode" => rc,
        "days" => 2,
        "log_prefix" => out.join(&name).to_string_lossy().to_string(),
        "deletion_executor" => "approved2h/content/no-consumer classifier; cargo-sweep never deletes mixed targets",
    }
}

fn reconcile_interrupted_targets(base: &Path) {
    let dir = base.join("state/enforcer-target-status");
    let _ = fs::create_dir_all(&dir);
    let Ok(rd) = fs::read_dir(&dir) else {
        return;
    };
    for entry in rd.flatten() {
        let marker = entry.path();
        if marker.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(mut saved) = Json::load_file(&marker) else {
            continue;
        };
        if saved.get_str("status") != Some("running") {
            continue;
        }
        let manifest_str = saved.get_str("manifest").unwrap_or("").to_string();
        let manifest = PathBuf::from(&manifest_str);
        let mut deleted_bytes: u64 = 0;
        let mut deleted_count: usize = 0;
        let mut intents = BTreeSet::new();
        let mut deleted_set = BTreeSet::new();
        if let Ok(text) = fs::read_to_string(&manifest) {
            for line in text.lines() {
                if let Ok(r) = Json::parse(line) {
                    let ev = r.get_str("event").unwrap_or("");
                    let p = r.get_str("path").unwrap_or("").to_string();
                    if ev == "deleted" {
                        deleted_bytes += r.get_u64("allocated_bytes").unwrap_or(0);
                        deleted_count += 1;
                        deleted_set.insert(p);
                    } else if ev == "unlink_intent" {
                        intents.insert(p);
                    }
                }
            }
        }
        let unconfirmed = intents.difference(&deleted_set).count();
        let log_path = base.join("DELETED.log");
        let already_logged = fs::read_to_string(&log_path)
            .map(|s| s.contains(&manifest_str))
            .unwrap_or(false);
        if !already_logged {
            let row = json_obj! {
                "utc" => utc_iso(),
                "host" => hostname(),
                "path" => saved.get_str("path").unwrap_or(""),
                "event" => "interrupted_build_cleanup_batch",
                "bytes" => deleted_bytes,
                "files" => deleted_count,
                "unconfirmed_intents" => unconfirmed,
                "manifest" => manifest_str,
                "approval_ref" => APPROVAL,
                "consumer_check" => "see durable preflight; no new deletion during reconciliation",
            };
            let _ = emit_jsonl_locked(&log_path, &row.to_compact());
        }
        saved.insert("status", "interrupted_reconciled");
        saved.insert("utc", utc_iso());
        let _ = atomic_write_str(&marker, &format!("{}\n", saved.to_compact()));
    }
}

fn disk_json() -> Json {
    match disk_usage(Path::new("/")) {
        Ok(d) => json_obj! {
            "used_bytes" => d.used_bytes,
            "available_bytes" => d.available_bytes,
            "used_percent" => d.used_percent,
        },
        Err(_) => json_obj! {
            "used_bytes" => 0u64,
            "available_bytes" => 0u64,
            "used_percent" => 100.0,
        },
    }
}

pub fn run(apply: bool) -> Json {
    let base = base_dir();
    let _ = fs::create_dir_all(&base);
    let Ok(_reclaim_lock) = FlockGuard::acquire_nonblocking(&base.join("reclaim.lock")) else {
        return json_obj! {
            "status" => "busy_existing_reclaimer",
            "utc" => utc_iso(),
        };
    };

    reconcile_interrupted_targets(&base);
    let stamp = utc_stamp_compact();
    let out = base.join("state").join(format!("enforcer-{stamp}"));
    let _ = fs::create_dir_all(&out);
    let roots = discover();
    let before = disk_json();
    let probe = consumers(&roots, None);
    let _ = atomic_write_str(&out.join("initial-consumers.json"), &probe.to_compact());

    let mut rows: Vec<Json> = Vec::new();
    let mut total_bytes_freed: u64 = 0;

    for root in &roots {
        let root_str = root.to_string_lossy().to_string();
        let mut row = json_obj! {
            "host" => hostname(),
            "path" => root_str.clone(),
            "utc" => utc_iso(),
            "approval_ref" => APPROVAL,
        };
        let (entries, newest, errors) = inventory(root);
        let mut unique_inodes: BTreeMap<(u64, u64), u64> = BTreeMap::new();
        for (_, s) in &entries {
            if s.file_type().is_file() && !s.file_type().is_symlink() {
                unique_inodes.insert((s.dev(), s.ino()), s.blocks().saturating_mul(512));
            }
        }
        let total_allocated: u64 = unique_inodes.values().sum();
        row.insert("newest_mtime", epoch_to_utc_iso(newest, 0));
        row.insert("total_allocated_bytes", total_allocated);

        let active = probe
            .get("consumers")
            .and_then(|c| c.get(&root_str))
            .and_then(|v| v.as_arr())
            .cloned()
            .unwrap_or_default();
        let probe_complete = probe.get_bool("complete").unwrap_or(false);
        let now_secs = crate::sys::now_epoch_secs();

        if !errors.is_empty() || !probe_complete || !active.is_empty() || now_secs - newest < 7200 {
            let err_slice: Vec<String> = errors.into_iter().take(5).collect();
            row.insert("status", "held");
            row.insert("reason", "inventory/consumer/age");
            row.insert("consumers", Json::Arr(active));
            row.insert("errors", err_slice);
        } else {
            let source = source_commit(root);
            let mut groups = match select_groups(root, &entries) {
                Ok(g) => g,
                Err(e) => {
                    row.insert("status", "held");
                    row.insert("reason", "incomplete content probe");
                    row.insert("error", e.to_string());
                    println!("{}", row.to_compact());
                    rows.push(row);
                    continue;
                }
            };
            let exception = probe
                .get("approved_reference_exceptions")
                .and_then(|m| m.get(&root_str))
                .cloned();
            if let Some(ref exc_ctx) = exception {
                match inherited_reference::restrict_groups(groups, exc_ctx) {
                    Ok(restricted) => {
                        groups = restricted;
                        row.insert("inherited_reference_exception", exc_ctx.clone());
                    }
                    Err(e) => {
                        row.insert("status", "held");
                        row.insert("reason", "incomplete content probe");
                        row.insert("error", e);
                        println!("{}", row.to_compact());
                        rows.push(row);
                        continue;
                    }
                }
            }
            let selected_paths: Vec<PathBuf> = groups
                .iter()
                .flat_map(|g| g.iter().map(|(p, _)| p.clone()))
                .collect();
            let eligible_allocated: u64 = groups
                .iter()
                .map(|g| g[0].1.blocks().saturating_mul(512))
                .sum();
            row.insert(
                "source",
                source.clone().unwrap_or(Json::Null),
            );
            row.insert("eligible_files", selected_paths.len());
            row.insert("eligible_allocated_bytes", eligible_allocated);

            let mut source_guard = None;
            if apply && source.is_some() && !selected_paths.is_empty() {
                let src_dir = PathBuf::from(
                    source
                        .as_ref()
                        .and_then(|s| s.get_str("source"))
                        .unwrap_or(""),
                );
                match source_preservation::ensure(&src_dir, &out, None) {
                    Ok(sg) => {
                        row.insert(
                            "source_preservation",
                            json_obj! {
                                "audit" => sg.audit.clone(),
                                "preservation" => sg.preservation.clone(),
                            },
                        );
                        source_guard = Some(sg);
                    }
                    Err(e) => {
                        row.insert("status", "held");
                        row.insert("reason", "source preservation preflight unavailable");
                        row.insert("error", e);
                        println!("{}", row.to_compact());
                        rows.push(row);
                        continue;
                    }
                }
            }
            if apply && source.is_some() && !selected_paths.is_empty() && exception.is_none() {
                row.insert(
                    "cargo_sweep",
                    sweep_advice(root, source.as_ref().unwrap(), &out),
                );
            }

            if source.is_none() {
                row.insert("status", "held");
                row.insert("reason", "source commit not located");
            } else if !apply {
                row.insert("status", "qualified_dry_run");
                row.insert("consumer_check", "complete/no live or stopped consumers");
                row.insert(
                    "source_preservation",
                    json_obj! { "status" => "not_evaluated_dry_run" },
                );
            } else if !selected_paths.is_empty() {
                match CargoCacheLocks::acquire(root, &entries, &selected_paths) {
                    Ok(held) => {
                        row.insert("cargo_lock_check", "exclusive/nonblocking/revalidated");
                        if let Some(ref sg) = source_guard {
                            if let Err(e) = source_preservation::revalidate(sg) {
                                row.insert("status", "held");
                                row.insert("reason", "source preservation changed before cleanup");
                                row.insert("error", e);
                                println!("{}", row.to_compact());
                                rows.push(row);
                                continue;
                            }
                        }
                        let fresh = consumers(std::slice::from_ref(root), Some(&held));
                        let mut exc_ok = true;
                        if let Some(ref exc_ctx) = exception {
                            let approved = fresh
                                .get("approved_reference_exceptions")
                                .and_then(|m| m.get(&root_str));
                            if let Some(app) = approved {
                                if app.get_str("plan_sha256") != exc_ctx.get_str("plan_sha256") {
                                    exc_ok = false;
                                }
                            }
                            if let Some(pin) = exc_ctx.get("retained_source_commit") {
                                let pin_src = pin.get_str("source").unwrap_or("");
                                let pin_commit = pin.get_str("commit").unwrap_or("");
                                let ok = Command::new("git")
                                    .arg("-C")
                                    .arg(pin_src)
                                    .args(["cat-file", "-e", &format!("{pin_commit}^{{commit}}")])
                                    .output()
                                    .map(|o| o.status.success())
                                    .unwrap_or(false);
                                if !ok {
                                    exc_ok = false;
                                }
                            }
                        }
                        let (cur_entries, cur_newest, cur_errors) = inventory(root);
                        let same = entries.len() == cur_entries.len()
                            && entries.iter().zip(cur_entries.iter()).all(
                                |((p1, s1), (p2, s2))| {
                                    p1 == p2
                                        && s1.ino() == s2.ino()
                                        && s1.size() == s2.size()
                                        && s1.mtime_nsec() == s2.mtime_nsec()
                                        && s1.ctime_nsec() == s2.ctime_nsec()
                                },
                            );
                        let fresh_complete = fresh.get_bool("complete").unwrap_or(false);
                        let fresh_active = fresh
                            .get("consumers")
                            .and_then(|c| c.get(&root_str))
                            .and_then(|v| v.as_arr())
                            .map(|a| !a.is_empty())
                            .unwrap_or(true);

                        if !exc_ok
                            || !fresh_complete
                            || fresh_active
                            || !cur_errors.is_empty()
                            || !same
                            || crate::sys::now_epoch_secs() - cur_newest < 7200
                        {
                            row.insert("status", "held");
                            row.insert("reason", "fresh consumer/tree check changed");
                        } else {
                            let target_hash = sha256_str(&root_str);
                            let manifest = out.join(format!("{target_hash}.jsonl"));
                            let preflight = out.join(format!("{target_hash}.preflight.json"));
                            let preflight_obj = json_obj! {
                                "utc" => utc_iso(),
                                "target" => row.clone(),
                                "consumer_probe" => fresh,
                                "cargo_directory_locks" => held.snapshot(),
                                "selected_files" => selected_paths.len(),
                                "approval_ref" => APPROVAL,
                            };
                            let _ = atomic_write_str(
                                &preflight,
                                &format!("{}\n", preflight_obj.to_pretty()),
                            );
                            let marker = base
                                .join("state/enforcer-target-status")
                                .join(format!("{target_hash}.json"));
                            let _ = atomic_write_str(
                                &marker,
                                &format!(
                                    "{}\n",
                                    json_obj! {
                                        "status" => "running",
                                        "path" => root_str.clone(),
                                        "manifest" => manifest.to_string_lossy().to_string(),
                                        "preflight" => preflight.to_string_lossy().to_string(),
                                    }
                                    .to_compact()
                                ),
                            );

                            let mut freed: u64 = 0;
                            let mut count: usize = 0;
                            for group in &groups {
                                if let Err(e) = held.validate() {
                                    row.insert(
                                        "mutation_warning",
                                        format!("identity changed; stopped target: {e}"),
                                    );
                                    break;
                                }
                                match unlink_group(group, &manifest) {
                                    Ok((gf, gc)) => {
                                        freed = freed.saturating_add(gf);
                                        count += gc;
                                    }
                                    Err(e) => {
                                        row.insert(
                                            "mutation_warning",
                                            format!("identity changed; stopped target: {e}"),
                                        );
                                        break;
                                    }
                                }
                            }
                            row.insert("status", "reclaimed");
                            row.insert("files", count);
                            row.insert("bytes", freed);
                            row.insert("consumer_check", "complete/no live or stopped consumers");
                            row.insert("manifest", manifest.to_string_lossy().to_string());
                            let _ = emit_jsonl_locked(&base.join("DELETED.log"), &row.to_compact());
                            let _ = atomic_write_str(
                                &marker,
                                &format!(
                                    "{}\n",
                                    json_obj! {
                                        "status" => "complete",
                                        "path" => root_str,
                                        "manifest" => manifest.to_string_lossy().to_string(),
                                        "preflight" => preflight.to_string_lossy().to_string(),
                                        "bytes" => freed,
                                    }
                                    .to_compact()
                                ),
                            );
                            total_bytes_freed = total_bytes_freed.saturating_add(freed);
                        }
                    }
                    Err(e) => {
                        row.insert("status", "held");
                        row.insert("reason", "Cargo directory lock unavailable");
                        row.insert("error", e);
                    }
                }
            } else {
                row.insert("status", "retained_nonbuild_or_unrecognized");
                row.insert("bytes", 0u64);
            }
        }
        println!("{}", row.to_compact());
        rows.push(row);
    }

    let docker = if apply {
        docker_images(&out)
    } else {
        json_obj! { "status" => "dry_run" }
    };
    let after = disk_json();
    let floor_reached = after
        .get("used_percent")
        .and_then(|v| v.as_f64())
        .map(|p| p < 85.0)
        .unwrap_or(false);

    let mut result = json_obj! {
        "utc" => utc_iso(),
        "host" => hostname(),
        "status" => "complete",
        "apply" => apply,
        "docker_images" => docker,
        "before" => before,
        "after" => after.clone(),
        "bytes_freed" => total_bytes_freed,
        "targets" => Json::Arr(rows),
        "free_space_floor_used_percent" => 85,
        "aggregate_target_cap_bytes" => 40u64 * 1024 * 1024 * 1024,
        "floor_reached" => floor_reached,
        "other_policy_stages" => json_obj! {
            "docker_unused_images" => "enabled under floor pressure;10min duplicate suppression; containers/volumes retained",
            "island_compaction" => "same owned worker triggered after pass; see compaction-latest.json and optional priority pause",
            "pip_uv_hf_tmp" => "pending separate content/consumer qualification; retained",
        },
    };
    let receipt_text = format!("{}\n", result.to_pretty());
    let _ = atomic_write_str(&out.join("receipt.json"), &receipt_text);
    let _ = atomic_write_str(&base.join("state/enforcer-latest.json"), &receipt_text);

    let compaction_unit = home_dir().join(".config/systemd/user/disk-steward-compaction.service");
    if apply && compaction_unit.is_file() {
        let _ = Command::new("systemctl")
            .args([
                "--user",
                "start",
                "--no-block",
                "disk-steward-compaction.service",
            ])
            .status();
    }
    if apply && !floor_reached {
        let alert = json_obj! {
            "utc" => utc_iso(),
            "host" => hostname(),
            "after" => after.clone(),
            "receipt" => out.join("receipt.json").to_string_lossy().to_string(),
            "reason" => "eligible idle compiler output insufficient; retained active/recent/nonbuild/protected data",
        };
        let _ = atomic_write_str(
            &base.join("state/enforcer-floor-alert.json"),
            &format!("{}\n", alert.to_pretty()),
        );
    }

    result.remove("targets");
    result
}
