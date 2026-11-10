//! Read-only deep inventory; root privileges are used only by scan workers.
//! Native Rust replacement for `full_scan.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::cache_namespaces::is_symlink;
use crate::json::Json;
use crate::json_obj;
use crate::sys::{
    atomic_write_str, base_dir, epoch_to_utc_iso, hostname, now_epoch_f64, utc_iso,
    utc_stamp_compact, FlockGuard,
};

const GIB: u64 = 1024 * 1024 * 1024;

fn load_id_map(path: &Path) -> BTreeMap<u32, String> {
    let mut map = BTreeMap::new();
    if let Ok(text) = fs::read_to_string(path) {
        for line in text.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 3 {
                if let Ok(id) = parts[2].parse::<u32>() {
                    map.insert(id, parts[0].to_string());
                }
            }
        }
    }
    map
}

fn mounts() -> Vec<Json> {
    let Ok(out) = Command::new("findmnt")
        .args(["--json", "--list", "--output", "TARGET,FSTYPE,SOURCE"])
        .output()
    else {
        return Vec::new();
    };
    let Ok(parsed) = Json::parse(&String::from_utf8_lossy(&out.stdout)) else {
        return Vec::new();
    };
    parsed
        .get("filesystems")
        .and_then(|v| v.as_arr())
        .cloned()
        .unwrap_or_default()
}

pub fn run_worker(root: &Path, shallow: bool) -> i32 {
    let Ok(rs) = fs::symlink_metadata(root) else {
        return 1;
    };
    let device = rs.dev();
    let root_str = root.to_string_lossy().to_string();
    let excluded: BTreeSet<String> = mounts()
        .into_iter()
        .filter_map(|m| m.get_str("target").map(|s| s.to_string()))
        .filter(|t| t != &root_str)
        .collect();

    let uid_map = load_id_map(Path::new("/etc/passwd"));
    let gid_map = load_id_map(Path::new("/etc/group"));
    let host = hostname();
    let stdout = std::io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut files_count: u64 = 0;
    let mut dirs_count: u64 = 0;
    let mut errors_count: u64 = 0;
    let mut volatile_count: u64 = 0;
    let mut last_progress = Instant::now();

    struct Frame {
        path: PathBuf,
        meta: fs::Metadata,
        children: Vec<PathBuf>,
        idx: usize,
        allocated: u64,
        latest: i64,
        partial: bool,
    }

    let make_frame = |p: PathBuf, m: fs::Metadata| -> Result<Frame, std::io::Error> {
        let mut children = Vec::new();
        for entry in fs::read_dir(&p)? {
            children.push(entry?.path());
        }
        let latest = m.mtime().max(m.ctime());
        Ok(Frame {
            path: p,
            allocated: m.blocks().saturating_mul(512),
            meta: m,
            children,
            idx: 0,
            latest,
            partial: false,
        })
    };

    let mut stack = match make_frame(root.to_path_buf(), rs) {
        Ok(f) => vec![f],
        Err(e) => {
            eprintln!(
                "{}",
                json_obj! {
                    "event" => "errors",
                    "root" => root_str,
                    "utc" => utc_iso(),
                    "path" => root.to_string_lossy().to_string(),
                    "error" => e.to_string(),
                }
                .to_compact()
            );
            return 1;
        }
    };

    let mut total_alloc: u64 = 0;
    let mut root_latest: i64 = 0;

    while !stack.is_empty() {
        let top_idx = stack.len() - 1;
        if stack[top_idx].idx >= stack[top_idx].children.len() {
            let f = stack.pop().unwrap();
            dirs_count += 1;
            if !shallow && f.allocated >= 5 * GIB {
                let uid = f.meta.uid();
                let gid = f.meta.gid();
                let owner = uid_map.get(&uid).cloned().unwrap_or_else(|| uid.to_string());
                let group = gid_map.get(&gid).cloned().unwrap_or_else(|| gid.to_string());
                let status = if f.partial { "partial" } else { "complete" };
                let _ = writeln!(
                    writer,
                    "{}\tdirectory\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    host,
                    f.path.display(),
                    f.allocated,
                    f.allocated,
                    epoch_to_utc_iso(f.meta.mtime(), 0),
                    epoch_to_utc_iso(f.latest, 0),
                    uid,
                    owner,
                    gid,
                    group,
                    utc_iso(),
                    status
                );
                let _ = writer.flush();
            }
            if let Some(parent) = stack.last_mut() {
                parent.allocated = parent.allocated.saturating_add(f.allocated);
                parent.latest = parent.latest.max(f.latest);
                parent.partial |= f.partial;
            } else {
                total_alloc = f.allocated;
                root_latest = f.latest;
            }
            continue;
        }

        let child_path = stack[top_idx].children[stack[top_idx].idx].clone();
        stack[top_idx].idx += 1;
        let child_str = child_path.to_string_lossy().to_string();
        if excluded.contains(&child_str) {
            continue;
        }
        let s = match fs::symlink_metadata(&child_path) {
            Ok(m) => m,
            Err(e) => {
                if matches!(e.kind(), std::io::ErrorKind::NotFound) {
                    volatile_count += 1;
                } else {
                    errors_count += 1;
                }
                stack[top_idx].partial = true;
                continue;
            }
        };
        if s.dev() != device {
            continue;
        }
        let item_latest = s.mtime().max(s.ctime());
        stack[top_idx].latest = stack[top_idx].latest.max(item_latest);

        if s.file_type().is_dir() && !s.file_type().is_symlink() {
            if !shallow {
                match make_frame(child_path.clone(), s) {
                    Ok(frame) => stack.push(frame),
                    Err(e) => {
                        if matches!(e.kind(), std::io::ErrorKind::NotFound) {
                            volatile_count += 1;
                        } else {
                            errors_count += 1;
                        }
                        stack[top_idx].partial = true;
                    }
                }
            }
            continue;
        }

        files_count += 1;
        if s.file_type().is_file() && !s.file_type().is_symlink() && s.size() >= GIB {
            let uid = s.uid();
            let gid = s.gid();
            let owner = uid_map.get(&uid).cloned().unwrap_or_else(|| uid.to_string());
            let group = gid_map.get(&gid).cloned().unwrap_or_else(|| gid.to_string());
            let _ = writeln!(
                writer,
                "{}\tfile\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tcomplete",
                host,
                child_path.display(),
                s.size(),
                s.blocks().saturating_mul(512),
                epoch_to_utc_iso(s.mtime(), 0),
                epoch_to_utc_iso(item_latest, 0),
                uid,
                owner,
                gid,
                group,
                utc_iso()
            );
            let _ = writer.flush();
        }
        let inode = (s.dev(), s.ino());
        if s.nlink() <= 1 || !seen.contains(&inode) {
            stack[top_idx].allocated = stack[top_idx]
                .allocated
                .saturating_add(s.blocks().saturating_mul(512));
            if s.nlink() > 1 {
                seen.insert(inode);
            }
        }
        if last_progress.elapsed().as_secs() >= 60 {
            eprintln!(
                "{}",
                json_obj! {
                    "event" => "progress",
                    "root" => root_str.clone(),
                    "utc" => utc_iso(),
                    "path" => child_str,
                    "files" => files_count,
                    "directories" => dirs_count,
                    "errors" => errors_count,
                    "volatile" => volatile_count,
                }
                .to_compact()
            );
            last_progress = Instant::now();
        }
    }

    let _ = writer.flush();
    eprintln!(
        "{}",
        json_obj! {
            "event" => "worker_complete",
            "root" => root_str,
            "utc" => utc_iso(),
            "allocated_bytes" => total_alloc,
            "latest_write_utc" => epoch_to_utc_iso(root_latest, 0),
            "files" => files_count,
            "directories" => dirs_count,
            "errors" => errors_count,
            "volatile" => volatile_count,
        }
        .to_compact()
    );
    if errors_count > 0 {
        1
    } else {
        0
    }
}

pub fn run_controller() -> i32 {
    let base = base_dir();
    let out = base.join("scan");
    let _ = fs::create_dir_all(&out);
    let _ = fs::create_dir_all(base.join("state"));
    let Ok(_lock) = FlockGuard::acquire_nonblocking(&base.join("state/full-scan.lock")) else {
        println!("full scan already active");
        return 0;
    };

    let host = hostname();
    let stamp = utc_stamp_compact();
    let prefix = out.join(format!("{host}_{stamp}"));
    let inventory = PathBuf::from(format!("{}.tsv", prefix.display()));
    let errors_path = PathBuf::from(format!("{}.errors.log", prefix.display()));
    let summary_path = PathBuf::from(format!("{}.du3.log", prefix.display()));
    let started_epoch = now_epoch_f64();
    let started_utc = utc_iso();

    let all_mounts = mounts();
    let root_dev = fs::metadata("/").map(|m| m.dev()).unwrap_or(0);
    let allowed = [
        "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "vfat", "exfat", "fuseblk", "ntfs", "ntfs3",
    ];
    let mut volumes = Vec::new();
    let mut devices = BTreeSet::from([root_dev]);
    let mut excluded_mounts = Vec::new();
    let mut failures = Vec::new();

    for m in &all_mounts {
        let fstype = m.get_str("fstype").unwrap_or("");
        let target = m.get_str("target").unwrap_or("");
        let source = m.get_str("source").unwrap_or("");
        if !allowed.contains(&fstype) {
            let mut ex = m.clone();
            ex.insert("reason", "virtual or remote filesystem");
            excluded_mounts.push(ex);
            continue;
        }
        if source.contains('[') {
            let mut ex = m.clone();
            ex.insert(
                "reason",
                "bind-mount alias; underlying source scanned at its original path",
            );
            excluded_mounts.push(ex);
            continue;
        }
        match fs::metadata(target) {
            Ok(st) => {
                if !devices.contains(&st.dev()) {
                    volumes.push(target.to_string());
                    devices.insert(st.dev());
                } else if target != "/" {
                    let mut ex = m.clone();
                    ex.insert(
                        "reason",
                        "virtual/remote filesystem or alias of an already-covered device",
                    );
                    excluded_mounts.push(ex);
                }
            }
            Err(e) => {
                failures.push(json_obj! {
                    "mount" => target,
                    "error" => e.to_string(),
                });
            }
        }
    }

    let mut root_dirs = Vec::new();
    if let Ok(rd) = fs::read_dir("/") {
        for entry in rd.flatten() {
            let p = entry.path();
            if !is_symlink(&p) && p.is_dir() {
                if let Ok(m) = fs::metadata(&p) {
                    if m.dev() == root_dev {
                        root_dirs.push(p);
                    }
                }
            }
        }
    }
    let priority = |name: &str| -> i32 {
        match name {
            "var" => 0,
            "tmp" => 1,
            "srv" => 2,
            "opt" => 3,
            "root" => 4,
            "usr" => 5,
            "home" => 100,
            _ => 50,
        }
    };
    root_dirs.sort_by(|a, b| {
        let na = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let nb = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        (priority(na), na).cmp(&(priority(nb), nb))
    });
    volumes.sort();
    for v in &volumes {
        root_dirs.push(PathBuf::from(v));
    }

    let scan_roots: Vec<String> = root_dirs
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let mut manifest = json_obj! {
        "host" => host.clone(),
        "started_utc" => started_utc.clone(),
        "status" => "running",
        "inventory" => inventory.to_string_lossy().to_string(),
        "read_only" => true,
        "directory_threshold_bytes" => 5 * GIB,
        "file_logical_threshold_bytes" => GIB,
        "mounts" => Json::Arr(all_mounts),
        "roots" => Json::empty_arr(),
        "failures" => Json::Arr(failures.clone()),
        "excluded_mounts" => Json::Arr(excluded_mounts),
        "scan_roots" => scan_roots,
        "accounting" => "Directories: allocated bytes, hardlinks deduplicated within each top-level scan. Files: logical size plus allocated size. Do not sum nested/overlapping rows or infer reclaimable totals.",
    };
    let latest_path = out.join(format!("{host}_latest.json"));
    let _ = atomic_write_str(&latest_path, &format!("{}\n", manifest.to_pretty()));

    let Ok(mut dest) = File::create(&inventory) else {
        return 1;
    };
    let Ok(err_file) = File::create(&errors_path) else {
        return 1;
    };
    let Ok(sum_file) = File::create(&summary_path) else {
        return 1;
    };
    let _ = fs::set_permissions(&inventory, fs::Permissions::from_mode(0o600));
    let _ = fs::set_permissions(&errors_path, fs::Permissions::from_mode(0o600));
    let _ = fs::set_permissions(&summary_path, fs::Permissions::from_mode(0o600));
    drop(err_file);
    drop(sum_file);

    let _ = writeln!(
        dest,
        "host\tkind\tpath\tsize_bytes\tallocated_bytes\tmtime_utc\tlatest_write_utc\tuid\towner\tgid\tgroup\tobserved_utc\tstatus"
    );
    let _ = dest.flush();

    if let Ok(docker) = Command::new("sudo")
        .args(["-n", "nice", "-n", "19", "ionice", "-c3", "docker", "system", "df", "-v"])
        .output()
    {
        let mut combined = docker.stdout;
        combined.extend_from_slice(&docker.stderr);
        let _ = fs::write(format!("{}.docker.txt", prefix.display()), combined);
    }

    let self_exe = std::env::current_exe()
        .unwrap_or_else(|_| base.join("bin/jope-disk-steward"));

    let run_scan_step = |root_arg: &str, shallow_flag: bool| -> i32 {
        let Ok(out_f) = OpenOptions::new().append(true).open(&inventory) else {
            return 1;
        };
        let Ok(err_f) = OpenOptions::new().append(true).open(&errors_path) else {
            return 1;
        };
        let mut cmd = Command::new("sudo");
        cmd.args([
            "-n",
            "nice",
            "-n",
            "19",
            "ionice",
            "-c3",
            &self_exe.to_string_lossy(),
            "full-scan",
            "--worker",
            root_arg,
        ]);
        if shallow_flag {
            cmd.arg("--shallow");
        }
        cmd.stdout(out_f).stderr(err_f);
        cmd.status().ok().and_then(|s| s.code()).unwrap_or(1)
    };

    let _ = run_scan_step("/", true);
    let mut roots_progress = Vec::new();
    for root in &root_dirs {
        let r_str = root.to_string_lossy().to_string();
        let st_utc = utc_iso();
        manifest.insert("current", r_str.clone());
        let _ = atomic_write_str(&latest_path, &format!("{}\n", manifest.to_pretty()));
        let rc = run_scan_step(&r_str, false);
        let row = json_obj! {
            "path" => r_str,
            "started_utc" => st_utc,
            "inventory_returncode" => rc,
            "inventory_finished_utc" => utc_iso(),
            "finished_utc" => utc_iso(),
        };
        if rc != 0 {
            failures.push(row.clone());
        }
        roots_progress.push(row);
        manifest.insert("roots", Json::Arr(roots_progress.clone()));
        manifest.insert("failures", Json::Arr(failures.clone()));
        let _ = atomic_write_str(&latest_path, &format!("{}\n", manifest.to_pretty()));
    }

    manifest.remove("current");
    manifest.insert("finished_utc", utc_iso());
    manifest.insert(
        "status",
        if failures.is_empty() {
            "complete"
        } else {
            "complete_with_errors"
        },
    );
    let manifest_text = format!("{}\n", manifest.to_pretty());
    let _ = atomic_write_str(
        &PathBuf::from(format!("{}.manifest.json", prefix.display())),
        &manifest_text,
    );
    let _ = atomic_write_str(&latest_path, &manifest_text);

    let comparison = base.join("state/full-scan-comparison-scheduled.json");
    if !comparison.exists() {
        let elapsed = (now_epoch_f64() - started_epoch).max(0.0) as i64;
        let delay = (1200 - elapsed).max(1);
        let out_res = Command::new("systemd-run")
            .args([
                "--user",
                "--unit=disk-steward-scan-comparison",
                &format!("--on-active={delay}s"),
                "/usr/bin/systemctl",
                "--user",
                "start",
                "disk-steward-full-scan.service",
            ])
            .env("XDG_RUNTIME_DIR", "/run/user/1000")
            .output();
        if let Ok(r) = out_res {
            let readback = format!(
                "{}{}",
                String::from_utf8_lossy(&r.stdout),
                String::from_utf8_lossy(&r.stderr)
            );
            let _ = atomic_write_str(
                &comparison,
                &format!(
                    "{}\n",
                    json_obj! {
                        "baseline" => inventory.to_string_lossy().to_string(),
                        "scheduled_utc" => utc_iso(),
                        "returncode" => r.status.code().unwrap_or(1),
                        "readback" => readback,
                    }
                    .to_pretty()
                ),
            );
        }
    }
    println!("{}", manifest.to_compact());
    0
}
