use super::*;

pub(crate) fn cpu_max(millicores: u64) -> String {
    if millicores == u64::MAX {
        format!("max {CPU_PERIOD_US}")
    } else {
        let quota = millicores.saturating_mul(CPU_PERIOD_US) / 1_000;
        format!("{} {CPU_PERIOD_US}", quota.max(1))
    }
}

pub(crate) fn memory_limit(bytes: u64) -> String {
    if bytes == u64::MAX {
        "max".to_owned()
    } else {
        bytes.to_string()
    }
}

pub(crate) fn read_pids(path: &Path) -> Result<Vec<i32>, CgroupError> {
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

pub(crate) fn read_cgroup_events(path: &Path) -> Result<CgroupEvents, CgroupError> {
    let source = fs::read_to_string(path.join("cgroup.events"))?;
    let populated = parse_named(&source, "populated")? != 0;
    let frozen = parse_named(&source, "frozen")? != 0;
    Ok(CgroupEvents { populated, frozen })
}

pub(crate) fn parse_named(source: &str, name: &str) -> Result<u64, CgroupError> {
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

pub(crate) fn read_u64(path: &Path) -> Result<Option<u64>, CgroupError> {
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

pub(crate) fn read_named_u64(path: &Path, name: &str) -> Result<Option<u64>, CgroupError> {
    match fs::read_to_string(path) {
        Ok(source) => Ok(Some(parse_named(&source, name)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}
