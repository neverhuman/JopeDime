//! Nonblocking Cargo directory locks for compiler-cache maintenance.
//! Native Rust replacement for `build_cache_locks.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::{self, Metadata};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::json_obj;
use crate::sys::{
    close, current_pid, current_uid, flock, open, LOCK_EX, LOCK_NB, O_NOFOLLOW, O_RDONLY,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InodeIdentity {
    pub dev: u64,
    pub ino: u64,
    pub mode: u32,
    pub uid: u32,
    pub nlink: u64,
    pub size: u64,
    pub mtime_ns: i64,
    pub ctime_ns: i64,
}

impl InodeIdentity {
    pub fn from_meta(m: &Metadata) -> Self {
        Self {
            dev: m.dev(),
            ino: m.ino(),
            mode: m.mode(),
            uid: m.uid(),
            nlink: m.nlink(),
            size: m.size(),
            mtime_ns: m.mtime_nsec(),
            ctime_ns: m.ctime_nsec(),
        }
    }
}

pub fn fd_identity(fd: i32) -> Result<InodeIdentity, String> {
    let proc_fd = format!("/proc/self/fd/{fd}");
    let m = fs::metadata(&proc_fd).map_err(|e| format!("fstat fd {fd}: {e}"))?;
    Ok(InodeIdentity::from_meta(&m))
}

pub struct CargoCacheLocks {
    root: PathBuf,
    opened: Vec<(i32, PathBuf, InodeIdentity)>,
}

impl CargoCacheLocks {
    pub fn acquire(
        root: &Path,
        entries: &[(PathBuf, Metadata)],
        selected_paths: &[PathBuf],
    ) -> Result<Self, String> {
        let known: BTreeMap<PathBuf, InodeIdentity> = entries
            .iter()
            .map(|(p, m)| (p.clone(), InodeIdentity::from_meta(m)))
            .collect();
        let mut required = BTreeSet::new();
        for path in selected_paths {
            if !path.starts_with(root) {
                return Err("Selected path outside the target".to_string());
            }
            let mut parent = path.parent();
            let mut found_lock = false;
            while let Some(p) = parent {
                if !p.starts_with(root) {
                    break;
                }
                let lock = p.join(".cargo-lock");
                if known.contains_key(&lock) {
                    required.insert(lock);
                    let newer = p.join(".cargo-build-lock");
                    if known.contains_key(&newer) {
                        required.insert(newer);
                    }
                    found_lock = true;
                    break;
                }
                if p == root {
                    break;
                }
                parent = p.parent();
            }
            if !found_lock {
                let rel = path.strip_prefix(root).unwrap_or(path);
                return Err(format!(
                    "No existing Cargo directory lock for {}",
                    rel.display()
                ));
            }
        }
        if required.is_empty() {
            return Err("No Cargo directory locks located".to_string());
        }

        let mut slf = Self {
            root: root.to_path_buf(),
            opened: Vec::new(),
        };
        let uid = current_uid();
        for path in required {
            let old = known.get(&path).unwrap().clone();
            let is_reg = (old.mode & 0o170000) == 0o100000;
            if !is_reg || old.uid != uid || old.nlink != 1 {
                return Err(format!("Unqualified Cargo lock inode: {}", path.display()));
            }
            let c_path = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| "NUL in lock path".to_string())?;
            let fd = unsafe { open(c_path.as_ptr(), O_RDONLY | O_NOFOLLOW, 0) };
            if fd < 0 {
                return Err(format!(
                    "open Cargo lock {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
            slf.opened.push((fd, path.clone(), old.clone()));
            if fd_identity(fd)? != old {
                return Err(format!(
                    "Cargo lock changed before acquisition: {}",
                    path.display()
                ));
            }
            let rc = unsafe { flock(fd, LOCK_EX | LOCK_NB) };
            if rc != 0 {
                return Err(format!(
                    "Cargo directory lock unavailable on {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
        }
        slf.validate()?;
        Ok(slf)
    }

    pub fn validate(&self) -> Result<(), String> {
        for (fd, path, old) in &self.opened {
            let cur_fd = fd_identity(*fd)?;
            let lstat_meta = fs::symlink_metadata(path)
                .map_err(|e| format!("lstat lock {}: {e}", path.display()))?;
            let cur_lstat = InodeIdentity::from_meta(&lstat_meta);
            if &cur_fd != old || &cur_lstat != old {
                return Err(format!(
                    "Cargo lock inode changed during maintenance: {}",
                    path.display()
                ));
            }
        }
        Ok(())
    }

    pub fn filter_probe(&self, probe: Json) -> Result<Json, String> {
        self.validate()?;
        let allowed: BTreeMap<i32, String> = self
            .opened
            .iter()
            .map(|(fd, p, _)| (*fd, p.to_string_lossy().to_string()))
            .collect();
        let root_str = self.root.to_string_lossy().to_string();
        let root_prefix = format!("{root_str}/");
        let mut observed = BTreeMap::new();
        let Ok(fd_dir) = fs::read_dir("/proc/self/fd") else {
            return Ok(probe);
        };
        for entry in fd_dir.flatten() {
            let Ok(fd_num) = entry.file_name().to_string_lossy().parse::<i32>() else {
                continue;
            };
            let Ok(target) = fs::read_link(entry.path()) else {
                continue;
            };
            let val = target.to_string_lossy().to_string();
            if val == root_str || val.starts_with(&root_prefix) {
                observed.insert(fd_num, val);
            }
        }
        let Ok(stat_text) = fs::read_to_string("/proc/self/stat") else {
            return Ok(probe);
        };
        let Some(close_idx) = stat_text.rfind(')') else {
            return Ok(probe);
        };
        let fields: Vec<&str> = stat_text[close_idx + 1..].split_whitespace().collect();
        if fields.len() < 20 {
            return Ok(probe);
        }
        let start_ticks = fields[19];

        if observed.is_empty()
            || observed
                .iter()
                .any(|(fd, path)| allowed.get(fd) != Some(path))
        {
            return Ok(probe);
        }

        let mut result = probe;
        let my_pid = current_pid() as i64;
        let claims = result
            .get("consumers")
            .and_then(|c| c.get(&root_str))
            .and_then(|v| v.as_arr())
            .cloned()
            .unwrap_or_default();

        let mut kept = Vec::new();
        let mut maint_claims = result
            .get("maintenance_lock_claims")
            .and_then(|v| v.as_arr())
            .cloned()
            .unwrap_or_default();

        for claim in claims {
            let pid = claim.get("pid").and_then(|v| v.as_i64()).unwrap_or(-1);
            let ticks = claim.get_str("start_ticks").unwrap_or(start_ticks);
            let state = claim.get_str("state").unwrap_or("");
            let reasons: Vec<&str> = claim
                .get("reasons")
                .and_then(|v| v.as_arr())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                .unwrap_or_default();
            if pid == my_pid && ticks == start_ticks && matches!(state, "R" | "S") && reasons == ["fd"]
            {
                let lock_files: Vec<String> = observed.values().cloned().collect();
                maint_claims.push(json_obj! {
                    "pid" => my_pid,
                    "start_ticks" => start_ticks,
                    "root" => root_str.clone(),
                    "lock_files" => lock_files,
                    "reason" => "own verified exclusive Cargo lock descriptors only",
                });
            } else {
                kept.push(claim);
            }
        }
        if let Some(obj) = result.as_obj_mut() {
            if let Some(Json::Obj(ref mut cons)) = obj.get_mut("consumers") {
                cons.insert(root_str, Json::Arr(kept));
            }
            if !maint_claims.is_empty() {
                obj.insert("maintenance_lock_claims".to_string(), Json::Arr(maint_claims));
            }
        }
        Ok(result)
    }

    pub fn snapshot(&self) -> Json {
        let items: Vec<Json> = self
            .opened
            .iter()
            .map(|(_, p, s)| {
                json_obj! {
                    "path" => p.to_string_lossy().to_string(),
                    "device" => s.dev,
                    "inode" => s.ino,
                    "mode" => "exclusive/nonblocking",
                }
            })
            .collect();
        Json::Arr(items)
    }
}

impl Drop for CargoCacheLocks {
    fn drop(&mut self) {
        for (fd, _, _) in self.opened.drain(..).rev() {
            unsafe {
                let _ = close(fd);
            }
        }
    }
}
