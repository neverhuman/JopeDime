//! Observe EXIT publication without restarting or signalling an existing island.
//! Native Rust replacement for `watch_island_exit.py`.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

use crate::cache_namespaces::is_symlink;
use crate::fleet_enforcer::expand_glob;
use crate::sys::{
    base_dir, home_dir, hostname, inotify_add_watch, inotify_init1, now_epoch_f64, O_CLOEXEC,
    O_NONBLOCK,
};

const IN_CLOSE_WRITE: u32 = 0x0000_0008;
const IN_MOVED_TO: u32 = 0x0000_0080;
const IN_CREATE: u32 = 0x0000_0100;
const IN_Q_OVERFLOW: u32 = 0x0000_4000;
const IN_ISDIR: u32 = 0x4000_0000;

fn islands_root() -> PathBuf {
    home_dir()
        .join("jarambit_runs/evo_continuous")
        .join(hostname())
        .join("islands")
}

fn trigger_compaction() {
    let base = base_dir();
    let _ = fs::write(
        base.join("state/compaction-exit-requested"),
        format!("{}\n", now_epoch_f64()),
    );
    let _ = Command::new("systemctl")
        .args([
            "--user",
            "start",
            "--no-block",
            "disk-steward-compaction.service",
        ])
        .status();
}

fn snapshot_exits(root: &PathBuf) -> BTreeMap<String, i64> {
    let mut seen = BTreeMap::new();
    for p in expand_glob(root, "i*/EXIT") {
        if let Ok(m) = fs::metadata(&p) {
            seen.insert(p.to_string_lossy().to_string(), m.mtime_nsec());
        }
    }
    seen
}

pub fn run_watch_island_exit() -> i32 {
    let base = base_dir();
    if base.join("state/compaction-paused.json").is_file() {
        return 0;
    }
    let root = islands_root();
    if !root.is_dir() {
        return 0;
    }

    let fd = unsafe { inotify_init1(O_CLOEXEC | O_NONBLOCK) };
    if fd < 0 {
        println!("inotify unavailable; using30s EXIT-directory polling");
        let mut seen = snapshot_exits(&root);
        loop {
            thread::sleep(Duration::from_secs(30));
            if base.join("state/compaction-paused.json").is_file() {
                return 0;
            }
            let current = snapshot_exits(&root);
            if current != seen {
                trigger_compaction();
            }
            seen = current;
        }
    }

    let mut paths: BTreeMap<i32, PathBuf> = BTreeMap::new();
    let add_watch = |p: &PathBuf, map: &mut BTreeMap<i32, PathBuf>| {
        if let Ok(c_str) = CString::new(p.as_os_str().as_bytes()) {
            let wd = unsafe {
                inotify_add_watch(fd, c_str.as_ptr(), IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE)
            };
            if wd >= 0 {
                map.insert(wd, p.clone());
            }
        }
    };

    add_watch(&root, &mut paths);
    if let Ok(rd) = fs::read_dir(&root) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with('i'))
                && p.is_dir()
                && !is_symlink(&p)
            {
                add_watch(&p, &mut paths);
            }
        }
    }

    let mut buf = vec![0u8; 64 * 1024];
    loop {
        if base.join("state/compaction-paused.json").is_file() {
            return 0;
        }
        let n = unsafe {
            extern "C" {
                fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
            }
            read(fd, buf.as_mut_ptr(), buf.len())
        };
        if n <= 0 {
            thread::sleep(Duration::from_secs(2));
            continue;
        }
        let n = n as usize;
        let mut offset = 0usize;
        let mut trigger = false;
        while offset + 16 <= n {
            let wd = i32::from_ne_bytes(buf[offset..offset + 4].try_into().unwrap());
            let mask = u32::from_ne_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
            let len = u32::from_ne_bytes(buf[offset + 12..offset + 16].try_into().unwrap()) as usize;
            offset += 16;
            if offset + len > n {
                break;
            }
            let name_bytes = &buf[offset..offset + len];
            offset += len;
            let name_end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name = String::from_utf8_lossy(&name_bytes[..name_end]);

            if (mask & IN_Q_OVERFLOW) != 0 {
                trigger = true;
                continue;
            }
            if paths.get(&wd) == Some(&root) && (mask & IN_ISDIR) != 0 && name.starts_with('i') {
                let child = root.join(name.as_ref());
                if child.is_dir() && !is_symlink(&child) {
                    add_watch(&child, &mut paths);
                }
            } else if name == "EXIT" && (mask & (IN_MOVED_TO | IN_CLOSE_WRITE)) != 0 {
                trigger = true;
            }
        }
        if trigger {
            trigger_compaction();
        }
    }
}
