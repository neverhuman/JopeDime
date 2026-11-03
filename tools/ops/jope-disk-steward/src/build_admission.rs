//! Admission before future builds; never touches or signals existing jobs.
//! Native Rust replacement for `build_admission.py`.

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cache_namespaces::is_symlink;
use crate::fleet_enforcer::{discover, expand_glob};
use crate::sys::disk_usage;

pub fn check(target: &Path, roots: &[PathBuf]) -> (bool, String) {
    let abs = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(target)
    };
    let mut parent = abs.as_path();
    while !parent.exists() {
        let Some(p) = parent.parent() else {
            return (false, "invalid target path".to_string());
        };
        parent = p;
    }
    let stats = match disk_usage(parent) {
        Ok(s) => s,
        Err(_) => return (false, "incomplete target-pool filesystem measurement".to_string()),
    };
    if stats.used_percent >= 85.0 {
        return (
            false,
            "target filesystem at or above85%; route future work to qualified native capacity"
                .to_string(),
        );
    }
    let dev = match fs::metadata(parent) {
        Ok(m) => m.dev(),
        Err(_) => return (false, "incomplete target-pool filesystem measurement".to_string()),
    };

    let mut same_device = BTreeSet::new();
    for root in roots {
        match fs::metadata(root) {
            Ok(m) if m.dev() == dev => {
                same_device.insert(root.clone());
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => {
                return (
                    false,
                    "incomplete target-pool filesystem measurement".to_string(),
                )
            }
        }
    }
    let filtered: Vec<PathBuf> = same_device
        .iter()
        .filter(|r| !same_device.iter().any(|q| *r != q && r.starts_with(q)))
        .cloned()
        .collect();

    if !filtered.is_empty() {
        let mut cmd = Command::new("du");
        cmd.args(["-s", "-B1", "--"]);
        for r in &filtered {
            cmd.arg(r);
        }
        let Ok(out) = cmd.output() else {
            return (false, "incomplete target-pool measurement".to_string());
        };
        if !out.status.success() || !out.stderr.is_empty() {
            return (false, "incomplete target-pool measurement".to_string());
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut sizes: Vec<u64> = Vec::new();
        for line in stdout.lines() {
            if let Some((num_s, _)) = line.split_once('\t') {
                if let Ok(n) = num_s.trim().parse::<u64>() {
                    sizes.push(n);
                }
            }
        }
        if sizes.len() != filtered.len() {
            return (false, "incomplete target-pool rows".to_string());
        }
        let total: u64 = sizes.iter().sum();
        if total >= 40 * 1024 * 1024 * 1024 {
            return (
                false,
                "40GiB aggregate target-pool admission cap reached; enforcer retains active/nonbuild data".to_string(),
            );
        }
    }
    (true, "capacity admission passed".to_string())
}

pub fn run_build_admission() -> i32 {
    let Some(target_str) = std::env::var_os("CARGO_TARGET_DIR") else {
        eprintln!("Missing scoped CARGO_TARGET_DIR");
        return 1;
    };
    let target = PathBuf::from(target_str);
    let mut roots = discover();
    let external = Path::new("/mnt/xbabe3-external");
    if external.is_dir() {
        for pat in [
            "cifix/gate_targets/*",
            "cifix/ledger_target_*",
            "jk_wide/*/target",
            "jk_wide/records_target",
        ] {
            for p in expand_glob(external, pat) {
                if p.is_dir() && !is_symlink(&p) {
                    roots.push(p);
                }
            }
        }
    }
    if let Ok(existing) = fs::canonicalize(&target) {
        if existing.is_dir() {
            roots.push(existing);
        }
    }
    let (ok, reason) = check(&target, &roots);
    eprintln!("disk-steward: {reason}");
    if ok {
        0
    } else {
        6
    }
}
