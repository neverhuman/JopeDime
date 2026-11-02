use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use nix::sys::statvfs::statvfs;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuSet {
    cpus: BTreeSet<u32>,
}

impl CpuSet {
    pub fn parse(input: &str) -> Result<Self, CapacityError> {
        let mut cpus = BTreeSet::new();
        let input = input.trim();
        if input.is_empty() {
            return Ok(Self { cpus });
        }
        for item in input.split(',') {
            let item = item.trim();
            if let Some((start, end)) = item.split_once('-') {
                let start = parse_cpu(start)?;
                let end = parse_cpu(end)?;
                if start > end || end.saturating_sub(start) > 1_000_000 {
                    return Err(CapacityError::CpuSet(input.to_owned()));
                }
                cpus.extend(start..=end);
            } else {
                cpus.insert(parse_cpu(item)?);
            }
        }
        Ok(Self { cpus })
    }

    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            cpus: self.cpus.intersection(&other.cpus).copied().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.cpus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.cpus.iter().copied()
    }

    pub fn to_list(&self) -> String {
        let mut ranges = Vec::new();
        let mut iter = self.cpus.iter().copied();
        let Some(mut start) = iter.next() else {
            return String::new();
        };
        let mut end = start;
        for cpu in iter {
            if cpu == end.saturating_add(1) {
                end = cpu;
            } else {
                ranges.push(format_range(start, end));
                start = cpu;
                end = cpu;
            }
        }
        ranges.push(format_range(start, end));
        ranges.join(",")
    }
}

fn parse_cpu(value: &str) -> Result<u32, CapacityError> {
    value
        .parse()
        .map_err(|_| CapacityError::CpuSet(value.to_owned()))
}

fn format_range(start: u32, end: u32) -> String {
    if start == end {
        start.to_string()
    } else {
        format!("{start}-{end}")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NumaNode {
    pub id: u32,
    pub cpus: CpuSet,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectiveCapacity {
    pub cgroup_path: PathBuf,
    pub cpus: CpuSet,
    pub numa_nodes: Vec<NumaNode>,
    pub cpu_millicores: u64,
    pub memory_max_bytes: u64,
    pub swap_max_bytes: Option<u64>,
    pub scratch_available_bytes: u64,
}

pub struct CapacityDiscovery<'a> {
    pub cgroup_mount: &'a Path,
    pub proc_self_cgroup: &'a Path,
    pub numa_root: &'a Path,
    pub scratch_path: &'a Path,
}

impl CapacityDiscovery<'_> {
    pub fn discover(&self) -> Result<EffectiveCapacity, CapacityError> {
        let relative = current_cgroup(fs::read_to_string(self.proc_self_cgroup)?)?;
        let cgroup_path = join_beneath(self.cgroup_mount, &relative)?;
        self.discover_at(cgroup_path)
    }

    /// Discover the capacity available to workload cgroups rooted below the
    /// daemon's own effective cgroup, or its empty delegated service parent when
    /// systemd places it in a `supervisor` subgroup. Other ancestors and siblings
    /// remain forbidden; limits are discovered at the accepted workload root.
    pub fn discover_for_allocation_root(
        &self,
        allocation_root: &Path,
    ) -> Result<EffectiveCapacity, CapacityError> {
        let relative = current_cgroup(fs::read_to_string(self.proc_self_cgroup)?)?;
        let daemon_cgroup = fs::canonicalize(join_beneath(self.cgroup_mount, &relative)?)?;
        let allocation_root = fs::canonicalize(allocation_root)?;
        if !allocation_root.starts_with(&daemon_cgroup)
            && !is_systemd_delegated_root(&allocation_root, &daemon_cgroup)?
        {
            return Err(CapacityError::AllocationRootOutsideDaemon {
                allocation_root,
                daemon_cgroup,
            });
        }
        self.discover_at(allocation_root)
    }

    fn discover_at(&self, cgroup_path: PathBuf) -> Result<EffectiveCapacity, CapacityError> {
        let hierarchy = hierarchy_paths(self.cgroup_mount, &cgroup_path)?;
        let affinity = process_affinity()?;
        let cpus = hierarchy_cpuset(&hierarchy)?
            .map_or_else(|| affinity.clone(), |cpuset| cpuset.intersect(&affinity));
        if cpus.is_empty() {
            return Err(CapacityError::NoCpu);
        }

        let quota_millicores = hierarchy
            .iter()
            .filter_map(|path| read_cpu_limit(&path.join("cpu.max")).transpose())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .min();
        let cpu_millicores = quota_millicores
            .unwrap_or(u64::MAX)
            .min((cpus.len() as u64).saturating_mul(1_000));

        let memory_max_bytes = minimum_limit(&hierarchy, "memory.max")?
            .or_else(host_memory_bytes)
            .ok_or(CapacityError::NoMemoryLimit)?;
        let swap_max_bytes = minimum_limit(&hierarchy, "memory.swap.max")?;
        let scratch_available_bytes = scratch_available(self.scratch_path)?;
        let numa_nodes = discover_numa(self.numa_root, &cpus)?;

        Ok(EffectiveCapacity {
            cgroup_path,
            cpus,
            numa_nodes,
            cpu_millicores,
            memory_max_bytes,
            swap_max_bytes,
            scratch_available_bytes,
        })
    }
}

fn is_systemd_delegated_root(
    allocation_root: &Path,
    daemon_cgroup: &Path,
) -> Result<bool, CapacityError> {
    if daemon_cgroup.file_name() != Some(std::ffi::OsStr::new("supervisor"))
        || daemon_cgroup.parent() != Some(allocation_root)
    {
        return Ok(false);
    }
    if !fs::read_to_string(allocation_root.join("cgroup.procs"))?
        .trim()
        .is_empty()
    {
        return Ok(false);
    }
    let controllers = fs::read_to_string(allocation_root.join("cgroup.controllers"))?;
    Ok(["cpu", "memory", "cpuset"].into_iter().all(|required| {
        controllers
            .split_whitespace()
            .any(|value| value == required)
    }))
}

fn hierarchy_cpuset(paths: &[PathBuf]) -> Result<Option<CpuSet>, CapacityError> {
    for path in paths {
        let path = path.join("cpuset.cpus.effective");
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let cpus = CpuSet::parse(&source)?;
        if !cpus.is_empty() {
            return Ok(Some(cpus));
        }
    }
    Ok(None)
}

fn process_affinity() -> Result<CpuSet, CapacityError> {
    let source = fs::read_to_string("/proc/self/status")?;
    process_affinity_from(&source)
}

fn process_affinity_from(source: &str) -> Result<CpuSet, CapacityError> {
    let list = source
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or(CapacityError::NoProcessAffinity)?;
    CpuSet::parse(list)
}

fn current_cgroup(contents: String) -> Result<PathBuf, CapacityError> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(PathBuf::from)
        .ok_or(CapacityError::NoUnifiedCgroup)
}

fn join_beneath(root: &Path, relative: &Path) -> Result<PathBuf, CapacityError> {
    let relative = relative.strip_prefix("/").unwrap_or(relative);
    if relative
        .components()
        .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(CapacityError::CgroupEscape(relative.to_path_buf()));
    }
    Ok(root.join(relative))
}

fn hierarchy_paths(root: &Path, leaf: &Path) -> Result<Vec<PathBuf>, CapacityError> {
    if !leaf.starts_with(root) {
        return Err(CapacityError::CgroupEscape(leaf.to_path_buf()));
    }
    let mut paths = Vec::new();
    let mut cursor = leaf.to_path_buf();
    loop {
        paths.push(cursor.clone());
        if cursor == root {
            break;
        }
        if !cursor.pop() {
            return Err(CapacityError::CgroupEscape(leaf.to_path_buf()));
        }
    }
    Ok(paths)
}

fn read_cpu_limit(path: &Path) -> Result<Option<u64>, CapacityError> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut fields = source.split_whitespace();
    let quota = fields
        .next()
        .ok_or_else(|| CapacityError::CpuMax(source.clone()))?;
    let period = fields
        .next()
        .ok_or_else(|| CapacityError::CpuMax(source.clone()))?
        .parse::<u64>()
        .map_err(|_| CapacityError::CpuMax(source.clone()))?;
    if quota == "max" {
        return Ok(None);
    }
    let quota = quota
        .parse::<u64>()
        .map_err(|_| CapacityError::CpuMax(source))?;
    if period == 0 {
        return Err(CapacityError::CpuMax("zero period".to_owned()));
    }
    Ok(Some(quota.saturating_mul(1_000).div_ceil(period)))
}

fn minimum_limit(paths: &[PathBuf], name: &str) -> Result<Option<u64>, CapacityError> {
    let mut result = None;
    for path in paths {
        let path = path.join(name);
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let value = source.trim();
        if value == "max" {
            continue;
        }
        let value = value
            .parse::<u64>()
            .map_err(|_| CapacityError::Limit(path, source))?;
        result = Some(result.map_or(value, |current: u64| current.min(value)));
    }
    Ok(result)
}

fn host_memory_bytes() -> Option<u64> {
    let source = fs::read_to_string("/proc/meminfo").ok()?;
    let kib = source.lines().find_map(|line| {
        line.strip_prefix("MemTotal:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    })?;
    Some(kib.saturating_mul(1024))
}

fn scratch_available(path: &Path) -> Result<u64, CapacityError> {
    let stats = statvfs(path).map_err(|source| CapacityError::Scratch {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(stats
        .blocks_available()
        .saturating_mul(stats.fragment_size()))
}

fn discover_numa(root: &Path, effective: &CpuSet) -> Result<Vec<NumaNode>, CapacityError> {
    let mut nodes = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(nodes),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name.strip_prefix("node").and_then(|id| id.parse().ok()) else {
            continue;
        };
        let cpus =
            CpuSet::parse(&read_required(&entry.path().join("cpulist"))?)?.intersect(effective);
        if !cpus.is_empty() {
            nodes.push(NumaNode { id, cpus });
        }
    }
    nodes.sort_by_key(|node| node.id);
    Ok(nodes)
}

fn read_required(path: &Path) -> Result<String, CapacityError> {
    fs::read_to_string(path).map_err(CapacityError::Io)
}

#[derive(Debug, Error)]
pub enum CapacityError {
    #[error("invalid CPU set: {0}")]
    CpuSet(String),
    #[error("unified cgroup entry missing from /proc/self/cgroup")]
    NoUnifiedCgroup,
    #[error("cgroup path escapes configured mount: {0}")]
    CgroupEscape(PathBuf),
    #[error("allocation cgroup root {allocation_root} is outside daemon cgroup {daemon_cgroup}")]
    AllocationRootOutsideDaemon {
        allocation_root: PathBuf,
        daemon_cgroup: PathBuf,
    },
    #[error("effective CPU set is empty")]
    NoCpu,
    #[error("process CPU affinity is unavailable")]
    NoProcessAffinity,
    #[error("invalid cpu.max value: {0}")]
    CpuMax(String),
    #[error("invalid {0} value: {1:?}")]
    Limit(PathBuf, String),
    #[error("no effective memory limit and host memory is unavailable")]
    NoMemoryLimit,
    #[error("cannot inspect scratch capacity for {path}: {source}")]
    Scratch { path: PathBuf, source: nix::Error },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests;
