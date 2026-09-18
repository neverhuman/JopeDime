use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capacity::CpuSet;

const CPU_PERIOD_US: u64 = 100_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IsolationMode {
    Delegated,
    SingleAllocation,
}

#[derive(Clone, Debug)]
pub struct CgroupManager {
    root: PathBuf,
    mode: IsolationMode,
    unavailable_reason: Option<String>,
}

impl CgroupManager {
    pub fn initialize(root: impl Into<PathBuf>, embedded: bool) -> Result<Self, CgroupError> {
        let root = root.into();
        match initialize_delegated_root(&root) {
            Ok(()) => {
                let root = fs::canonicalize(root)?;
                Ok(Self {
                    root,
                    mode: IsolationMode::Delegated,
                    unavailable_reason: None,
                })
            }
            Err(error) if embedded => Ok(Self {
                root,
                mode: IsolationMode::SingleAllocation,
                unavailable_reason: Some(error.to_string()),
            }),
            Err(error) => Err(error),
        }
    }

    pub const fn mode(&self) -> IsolationMode {
        self.mode
    }

    pub fn unavailable_reason(&self) -> Option<&str> {
        self.unavailable_reason.as_deref()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn prepare(
        &self,
        key: &AttemptKey,
        limits: &CgroupLimits,
    ) -> Result<JobCgroup, CgroupError> {
        self.require_delegated()?;
        key.validate()?;
        limits.validate()?;
        let name = key.cgroup_name();
        let path = self.root.join(&name);
        fs::create_dir(&path).map_err(|source| CgroupError::Create {
            path: path.clone(),
            source,
        })?;

        let configure = || -> Result<(), CgroupError> {
            let cpus = limits.cpus.to_list();
            if !cpus.is_empty() {
                if let Ok(mems) = fs::read_to_string(self.root.join("cpuset.mems.effective")) {
                    let mems = mems.trim();
                    if !mems.is_empty() {
                        write_control(&path, "cpuset.mems", mems)?;
                    }
                }
                write_control(&path, "cpuset.cpus", &cpus)?;
            }
            write_control(&path, "cpu.weight", &limits.cpu_weight.to_string())?;
            write_control(&path, "cpu.max", &cpu_max(limits.cpu_max_millicores))?;
            write_control(&path, "memory.low", &memory_limit(limits.memory_low_bytes))?;
            write_control(
                &path,
                "memory.high",
                &memory_limit(limits.memory_high_bytes),
            )?;
            write_control(&path, "memory.max", &memory_limit(limits.memory_max_bytes))?;
            write_control(&path, "memory.oom.group", "1")?;
            Ok(())
        };
        if let Err(error) = configure() {
            let _ = fs::remove_dir(&path);
            return Err(error);
        }

        Ok(JobCgroup {
            key: key.clone(),
            name,
            path,
        })
    }

    pub fn attach_pid(&self, cgroup: &JobCgroup, pid: u32) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "cgroup.procs", &pid.to_string())
    }

    pub fn freeze(&self, cgroup: &JobCgroup) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "cgroup.freeze", "1")
    }

    pub fn resume(&self, cgroup: &JobCgroup) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "cgroup.freeze", "0")
    }

    pub fn set_cpu_max(&self, cgroup: &JobCgroup, millicores: u64) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "cpu.max", &cpu_max(millicores))
    }

    pub fn set_cpu_weight(&self, cgroup: &JobCgroup, weight: u16) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        if !(1..=10_000).contains(&weight) {
            return Err(CgroupError::CpuWeight(weight));
        }
        write_control(&cgroup.path, "cpu.weight", &weight.to_string())
    }

    pub fn set_memory_high(&self, cgroup: &JobCgroup, bytes: u64) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "memory.high", &memory_limit(bytes))
    }

    pub fn pids(&self, cgroup: &JobCgroup) -> Result<Vec<u32>, CgroupError> {
        self.verify_owned(cgroup)?;
        read_pids(&cgroup.path)?
            .into_iter()
            .map(|pid| u32::try_from(pid).map_err(|_| CgroupError::InvalidPid(pid.to_string())))
            .collect()
    }

    pub fn signal_all(&self, cgroup: &JobCgroup, signal: Signal) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        for pid in read_pids(&cgroup.path)? {
            match kill(Pid::from_raw(pid), signal) {
                Ok(()) | Err(nix::errno::Errno::ESRCH) => {}
                Err(source) => return Err(CgroupError::Signal { pid, source }),
            }
        }
        Ok(())
    }

    pub fn kill(&self, cgroup: &JobCgroup) -> Result<(), CgroupError> {
        self.verify_owned(cgroup)?;
        write_control(&cgroup.path, "cgroup.kill", "1")
    }

    pub fn remove_if_empty(&self, cgroup: &JobCgroup) -> Result<bool, CgroupError> {
        self.verify_owned(cgroup)?;
        if read_cgroup_events(&cgroup.path)?.populated || !read_pids(&cgroup.path)?.is_empty() {
            return Ok(false);
        }
        fs::remove_dir(&cgroup.path).map_err(|source| CgroupError::Remove {
            path: cgroup.path.clone(),
            source,
        })?;
        Ok(true)
    }

    pub fn list_attempts(&self) -> Result<Vec<AdoptedCgroup>, CgroupError> {
        self.require_delegated()?;
        let mut attempts = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            let Some(key) = AttemptKey::from_cgroup_name(&name) else {
                continue;
            };
            let path = entry.path();
            let events = read_cgroup_events(&path)?;
            attempts.push(AdoptedCgroup {
                key,
                name,
                pids: read_pids(&path)?,
                populated: events.populated,
                frozen: events.frozen,
                path,
            });
        }
        attempts.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(attempts)
    }

    pub fn metrics(&self, cgroup: &JobCgroup) -> Result<CgroupMetrics, CgroupError> {
        self.verify_owned(cgroup)?;
        Ok(CgroupMetrics {
            memory_current_bytes: read_u64(&cgroup.path.join("memory.current"))?.unwrap_or(0),
            memory_peak_bytes: read_u64(&cgroup.path.join("memory.peak"))?.unwrap_or(0),
            oom_kills: read_named_u64(&cgroup.path.join("memory.events"), "oom_kill")?.unwrap_or(0),
            cpu_throttled_periods: read_named_u64(&cgroup.path.join("cpu.stat"), "nr_throttled")?
                .unwrap_or(0),
        })
    }

    fn require_delegated(&self) -> Result<(), CgroupError> {
        if self.mode == IsolationMode::Delegated {
            Ok(())
        } else {
            Err(CgroupError::SingleAllocation(
                self.unavailable_reason
                    .clone()
                    .unwrap_or_else(|| "cgroup delegation unavailable".to_owned()),
            ))
        }
    }

    fn verify_owned(&self, cgroup: &JobCgroup) -> Result<(), CgroupError> {
        self.require_delegated()?;
        if cgroup.path.parent() != Some(self.root.as_path())
            || cgroup.path.file_name().and_then(|name| name.to_str()) != Some(&cgroup.name)
        {
            return Err(CgroupError::ForeignPath(cgroup.path.clone()));
        }
        Ok(())
    }
}

fn initialize_delegated_root(root: &Path) -> Result<(), CgroupError> {
    if !root.is_absolute() {
        return Err(CgroupError::RelativeRoot(root.to_path_buf()));
    }
    fs::create_dir_all(root).map_err(|source| CgroupError::Create {
        path: root.to_path_buf(),
        source,
    })?;
    let controllers = fs::read_to_string(root.join("cgroup.controllers")).map_err(|source| {
        CgroupError::NotDelegated {
            root: root.to_path_buf(),
            detail: source.to_string(),
        }
    })?;
    for required in ["cpu", "memory", "cpuset"] {
        if !controllers
            .split_whitespace()
            .any(|value| value == required)
        {
            return Err(CgroupError::MissingController(required));
        }
    }
    write_control(root, "cgroup.subtree_control", "+cpu +memory +cpuset")?;

    let probe = root.join(format!(".scqd-probe-{}", std::process::id()));
    fs::create_dir(&probe).map_err(|source| CgroupError::NotDelegated {
        root: root.to_path_buf(),
        detail: source.to_string(),
    })?;
    fs::remove_dir(&probe).map_err(|source| CgroupError::Remove {
        path: probe,
        source,
    })?;
    Ok(())
}

fn write_control(path: &Path, name: &str, value: &str) -> Result<(), CgroupError> {
    let path = path.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|source| CgroupError::Control {
            path: path.clone(),
            value: value.to_owned(),
            source,
        })?;
    file.write_all(value.as_bytes())
        .map_err(|source| CgroupError::Control {
            path,
            value: value.to_owned(),
            source,
        })
}

fn cpu_max(millicores: u64) -> String {
    if millicores == u64::MAX {
        format!("max {CPU_PERIOD_US}")
    } else {
        let quota = millicores.saturating_mul(CPU_PERIOD_US) / 1_000;
        format!("{} {CPU_PERIOD_US}", quota.max(1))
    }
}

fn memory_limit(bytes: u64) -> String {
    if bytes == u64::MAX {
        "max".to_owned()
    } else {
        bytes.to_string()
    }
}

fn read_pids(path: &Path) -> Result<Vec<i32>, CgroupError> {
    let mut pids = std::collections::BTreeSet::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let source = match fs::read_to_string(directory.join("cgroup.procs")) {
            Ok(source) => source,
            Err(error) if directory != path && error.kind() == std::io::ErrorKind::NotFound => {
                continue
            }
            Err(error) => return Err(error.into()),
        };
        for line in source.lines().filter(|line| !line.trim().is_empty()) {
            pids.insert(
                line.trim()
                    .parse()
                    .map_err(|_| CgroupError::InvalidPid(line.to_owned()))?,
            );
        }
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            // cgroupfs cannot contain symlinks; never follow one in fixtures either.
            if entry.file_type()?.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    Ok(pids.into_iter().collect())
}

#[derive(Clone, Copy, Debug, Default)]
struct CgroupEvents {
    populated: bool,
    frozen: bool,
}

fn read_cgroup_events(path: &Path) -> Result<CgroupEvents, CgroupError> {
    let source = fs::read_to_string(path.join("cgroup.events"))?;
    let populated = parse_named(&source, "populated")? != 0;
    let frozen = parse_named(&source, "frozen")? != 0;
    Ok(CgroupEvents { populated, frozen })
}

fn parse_named(source: &str, name: &str) -> Result<u64, CgroupError> {
    source
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(' ')?;
            (key == name).then_some(value)
        })
        .ok_or_else(|| CgroupError::MissingMetric(name.to_owned()))?
        .trim()
        .parse()
        .map_err(|_| CgroupError::InvalidMetric(name.to_owned()))
}

fn read_u64(path: &Path) -> Result<Option<u64>, CgroupError> {
    match fs::read_to_string(path) {
        Ok(source) => source
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| CgroupError::InvalidMetric(path.display().to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_named_u64(path: &Path, name: &str) -> Result<Option<u64>, CgroupError> {
    match fs::read_to_string(path) {
        Ok(source) => Ok(Some(parse_named(&source, name)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptKey {
    pub job_id: String,
    pub attempt: u32,
    pub fence: String,
}

impl AttemptKey {
    pub fn validate(&self) -> Result<(), CgroupError> {
        validate_token("job id", &self.job_id, 64)?;
        validate_token("fence", &self.fence, 128)
    }

    pub fn cgroup_name(&self) -> String {
        format!(
            "scq-{}-attempt-{}-fence-{}",
            self.job_id, self.attempt, self.fence
        )
    }

    pub fn from_cgroup_name(name: &str) -> Option<Self> {
        let rest = name.strip_prefix("scq-")?;
        let (job_id, rest) = rest.split_once("-attempt-")?;
        let (attempt, fence) = rest.split_once("-fence-")?;
        let key = Self {
            job_id: job_id.to_owned(),
            attempt: attempt.parse().ok()?,
            fence: fence.to_owned(),
        };
        key.validate().ok()?;
        Some(key)
    }
}

fn validate_token(name: &'static str, value: &str, max: usize) -> Result<(), CgroupError> {
    if value.is_empty()
        || value.len() > max
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
    {
        return Err(CgroupError::InvalidToken {
            name,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CgroupLimits {
    pub cpus: CpuSet,
    pub cpu_weight: u16,
    pub cpu_max_millicores: u64,
    pub memory_low_bytes: u64,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
}

impl CgroupLimits {
    fn validate(&self) -> Result<(), CgroupError> {
        if !(1..=10_000).contains(&self.cpu_weight) {
            return Err(CgroupError::CpuWeight(self.cpu_weight));
        }
        if self.memory_low_bytes > self.memory_high_bytes
            || self.memory_high_bytes > self.memory_max_bytes
        {
            return Err(CgroupError::MemoryOrder);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobCgroup {
    pub key: AttemptKey,
    pub name: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdoptedCgroup {
    pub key: AttemptKey,
    pub name: String,
    pub path: PathBuf,
    pub pids: Vec<i32>,
    pub populated: bool,
    pub frozen: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CgroupMetrics {
    pub memory_current_bytes: u64,
    pub memory_peak_bytes: u64,
    pub oom_kills: u64,
    pub cpu_throttled_periods: u64,
}

#[derive(Debug, Error)]
pub enum CgroupError {
    #[error("cgroup root must be absolute: {0}")]
    RelativeRoot(PathBuf),
    #[error("cannot create cgroup {path}: {source}")]
    Create {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cgroup v2 delegation is unavailable at {root}: {detail}")]
    NotDelegated { root: PathBuf, detail: String },
    #[error("required cgroup controller is unavailable: {0}")]
    MissingController(&'static str),
    #[error("cannot write {path}={value:?}: {source}")]
    Control {
        path: PathBuf,
        value: String,
        source: std::io::Error,
    },
    #[error("cannot remove cgroup {path}: {source}")]
    Remove {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cgroup operations are disabled in single-allocation mode: {0}")]
    SingleAllocation(String),
    #[error("refusing operation on foreign cgroup path: {0}")]
    ForeignPath(PathBuf),
    #[error("invalid {name}: {value:?}")]
    InvalidToken { name: &'static str, value: String },
    #[error("cpu.weight must be in 1..=10000, got {0}")]
    CpuWeight(u16),
    #[error("memory controls must satisfy low <= high <= max")]
    MemoryOrder,
    #[error("invalid pid in cgroup.procs: {0}")]
    InvalidPid(String),
    #[error("missing cgroup metric: {0}")]
    MissingMetric(String),
    #[error("invalid cgroup metric: {0}")]
    InvalidMetric(String),
    #[error("failed to signal pid {pid}: {source}")]
    Signal { pid: i32, source: nix::Error },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_pid_inventory_retains_nested_descendant_after_leader_exit() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("cgroup.procs"), "").unwrap();
        let child = directory.path().join("nested");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(child.join("cgroup.procs"), "321\n").unwrap();
        assert_eq!(read_pids(directory.path()).unwrap(), vec![321]);
    }

    #[test]
    fn cgroup_names_round_trip_without_path_syntax() {
        let key = AttemptKey {
            job_id: "01JAINWORKLOAD".to_owned(),
            attempt: 7,
            fence: "5ca1ab1e".to_owned(),
        };
        let name = key.cgroup_name();
        assert_eq!(AttemptKey::from_cgroup_name(&name), Some(key));
        assert!(!name.contains('/'));
    }

    #[test]
    fn rejects_path_traversal_tokens() {
        let key = AttemptKey {
            job_id: "../other".to_owned(),
            attempt: 0,
            fence: "fence".to_owned(),
        };
        assert!(matches!(
            key.validate(),
            Err(CgroupError::InvalidToken { .. })
        ));
    }

    #[test]
    fn cpu_quota_is_expressed_in_millicores() {
        assert_eq!(cpu_max(1_500), "150000 100000");
        assert_eq!(cpu_max(u64::MAX), "max 100000");
    }

    #[test]
    fn unbounded_memory_uses_kernel_max_sentinel() {
        assert_eq!(memory_limit(u64::MAX), "max");
        assert_eq!(memory_limit(1024), "1024");
    }
}
