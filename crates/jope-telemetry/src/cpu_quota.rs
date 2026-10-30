//! CPU capacity available to this process, accounting for Linux cgroup quotas and affinity.

use std::path::{Path, PathBuf};

/// Returns the number of CPU core-equivalents available to this process.
///
/// Linux cgroup quotas can be fractional. Affinity limits are an integer count, so the tighter
/// of the quota and affinity is returned. Other platforms use the standard logical CPU count.
pub fn allowed_cpus() -> f64 {
    #[cfg(target_os = "linux")]
    {
        let logical = logical_cpu_count();
        // Rust's safe platform query uses the calling thread's CPU affinity on Linux.
        let affinity = std::thread::available_parallelism().map_or(logical, usize::from);
        let quota = cgroup_cpu_quota();
        intersect_quota_with_affinity(quota, affinity)
    }
    #[cfg(not(target_os = "linux"))]
    {
        logical as f64
    }
}

fn logical_cpu_count() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn intersect_quota_with_affinity(quota: Option<f64>, affinity: usize) -> f64 {
    quota.map_or(affinity as f64, |quota| quota.min(affinity as f64))
}

#[cfg(target_os = "linux")]
fn cgroup_cpu_quota() -> Option<f64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let v2_path = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(|path| cgroup_file_path("/sys/fs/cgroup", path, "cpu.max"));

    if let Some(v2_path) = v2_path {
        if v2_path.is_file() {
            return quota_v2(&v2_path);
        }
    }

    let v1_path = v1_cpu_membership(&membership).map(str::to_owned);
    let path = v1_path?;
    let quota = cgroup_file_path("/sys/fs/cgroup/cpu", &path, "cpu.cfs_quota_us");
    let period = cgroup_file_path("/sys/fs/cgroup/cpu", &path, "cpu.cfs_period_us");
    quota_v1(&quota, &period)
}

#[cfg(target_os = "linux")]
fn v1_cpu_membership(contents: &str) -> Option<&str> {
    contents.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let _hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        controllers
            .split(',')
            .any(|controller| controller == "cpu")
            .then_some(path)
    })
}

#[cfg(target_os = "linux")]
fn cgroup_file_path(root: &str, membership: &str, file: &str) -> PathBuf {
    Path::new(root)
        .join(membership.trim_start_matches('/'))
        .join(file)
}

fn quota_v2(path: &Path) -> Option<f64> {
    let contents = std::fs::read_to_string(path).ok()?;
    parse_quota_v2(&contents)
}

fn parse_quota_v2(contents: &str) -> Option<f64> {
    let mut fields = contents.split_whitespace();
    let quota = fields.next()?;
    let period: u64 = fields.next()?.parse().ok()?;
    if quota == "max" {
        return None;
    }
    let quota: u64 = quota.parse().ok()?;
    (quota > 0 && period > 0).then_some(quota as f64 / period as f64)
}

fn quota_v1(quota_path: &Path, period_path: &Path) -> Option<f64> {
    let quota = std::fs::read_to_string(quota_path).ok()?;
    let period = std::fs::read_to_string(period_path).ok()?;
    parse_quota_v1(&quota, &period)
}

fn parse_quota_v1(quota: &str, period: &str) -> Option<f64> {
    let quota: i64 = quota.trim().parse().ok()?;
    let period: u64 = period.trim().parse().ok()?;
    if quota <= 0 || period == 0 {
        return None;
    }
    Some(quota as f64 / period as f64)
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use super::v1_cpu_membership;
    use super::{
        intersect_quota_with_affinity, parse_quota_v1, parse_quota_v2, quota_v1, quota_v2,
    };
    use std::fs;

    #[test]
    fn v2_max_uses_affinity_limit() {
        let directory = tempfile::tempdir().unwrap();
        let quota = directory.path().join("cpu.max");
        fs::write(&quota, "max 100000\n").unwrap();
        assert_eq!(quota_v2(&quota), None);
        let affinity = 6_usize;
        let allowed = intersect_quota_with_affinity(quota_v2(&quota), affinity);
        assert_eq!(allowed, 6.0);
    }

    #[test]
    fn v2_fractional_quota_is_intersected_with_affinity() {
        let directory = tempfile::tempdir().unwrap();
        let quota = directory.path().join("cpu.max");
        fs::write(&quota, "150000 100000\n").unwrap();
        assert_eq!(quota_v2(&quota), Some(1.5));
        assert_eq!(intersect_quota_with_affinity(quota_v2(&quota), 8), 1.5);
        assert_eq!(intersect_quota_with_affinity(quota_v2(&quota), 1), 1.0);
    }

    #[test]
    fn v1_quota_and_period_are_read_from_fixture_files() {
        let directory = tempfile::tempdir().unwrap();
        let quota = directory.path().join("cpu.cfs_quota_us");
        let period = directory.path().join("cpu.cfs_period_us");
        fs::write(&quota, "250000\n").unwrap();
        fs::write(&period, "100000\n").unwrap();
        assert_eq!(quota_v1(&quota, &period), Some(2.5));
        assert_eq!(
            intersect_quota_with_affinity(quota_v1(&quota, &period), 8),
            2.5
        );
        assert_eq!(parse_quota_v1("-1", "100000"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn v1_membership_selects_cpu_controller_path() {
        let fixture = "2:cpu,cpuacct:/docker/container\n3:memory:/docker/container\n";
        assert_eq!(v1_cpu_membership(fixture), Some("/docker/container"));
    }

    #[test]
    fn affinity_is_used_when_quota_is_unlimited_or_unavailable() {
        let affinity = 3_usize;
        let v2 = intersect_quota_with_affinity(parse_quota_v2("max 100000"), affinity);
        let v1 = intersect_quota_with_affinity(parse_quota_v1("-1", "100000"), affinity);
        assert_eq!(v2, 3.0);
        assert_eq!(v1, 3.0);
    }
}
