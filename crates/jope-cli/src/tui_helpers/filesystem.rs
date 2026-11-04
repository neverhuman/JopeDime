use super::formatting::{to_float, to_int};
use serde_json::{Map, Value};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime},
};

pub fn read_json(path: &Path) -> Value {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| Value::Object(Map::new()))
}

pub fn first_existing(paths: &[PathBuf]) -> PathBuf {
    paths
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| paths.first().cloned().unwrap_or_default())
}

pub fn read_jsonl(path: &Path, limit: isize) -> Vec<Value> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let rows: Vec<Value> = text
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                None
            } else {
                serde_json::from_str(line).ok()
            }
        })
        .collect();
    tail_slice(&rows, limit)
}

pub fn json_objects_from_log(path: &Path) -> Vec<Map<String, Value>> {
    read_jsonl(path, 200)
        .into_iter()
        .filter_map(|row| row.as_object().cloned())
        .collect()
}

pub fn tail_lines(path: &Path, count: isize) -> Vec<String> {
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    tail_slice(&lines, count)
}

pub fn file_age(path: &Path) -> Option<f64> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    match SystemTime::now().duration_since(modified) {
        Ok(age) => Some(age.as_secs_f64()),
        Err(error) => Some(-error.duration().as_secs_f64()),
    }
}

pub fn mount_label(path: &Path) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let Ok(device) = fs::metadata(path) else {
            return "/".into();
        };
        let Ok(mounts) = fs::read_to_string("/proc/self/mounts") else {
            return "/".into();
        };
        for line in mounts.lines() {
            let Some(mount) = line.split_whitespace().nth(1) else {
                continue;
            };
            let mount = unescape_mount(mount);
            if fs::metadata(&mount).is_ok_and(|metadata| metadata.dev() == device.dev()) {
                return mount.to_string_lossy().into_owned();
            }
        }
    }
    "/".into()
}

pub fn gpu_states() -> Vec<Map<String, Value>> {
    let query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit";
    let mut child = match Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={query}"),
            "--format=csv,noheader,nounits".into(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return Vec::new(),
    };
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Vec::new();
                }
                let Some(mut stdout) = child.stdout.take() else {
                    return Vec::new();
                };
                let mut output = String::new();
                if stdout.read_to_string(&mut output).is_err() || output.trim().is_empty() {
                    return Vec::new();
                }
                return output.lines().filter_map(parse_gpu_line).collect();
            }
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Vec::new();
            }
        }
    }
}

pub fn dir_size(path: &Path) -> String {
    fn visit(path: &Path, total: &mut u128) -> std::io::Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                visit(&entry.path(), total)?;
            } else if entry.metadata()?.is_file() {
                *total += u128::from(entry.metadata()?.len());
            }
        }
        Ok(())
    }
    let mut total = 0u128;
    if path.is_dir() && visit(path, &mut total).is_err() {
        return "?".into();
    }
    let mut size = total as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 {
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

fn tail_slice<T: Clone>(values: &[T], count: isize) -> Vec<T> {
    let start = if count > 0 {
        values.len().saturating_sub(count as usize)
    } else if count == 0 {
        0
    } else {
        count.unsigned_abs().min(values.len())
    };
    values[start..].to_vec()
}

fn parse_gpu_line(line: &str) -> Option<Map<String, Value>> {
    let parts: Vec<_> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return None;
    }
    let mut row = Map::new();
    row.insert(
        "index".into(),
        Value::from(to_int(&Value::String(parts[0].into()))),
    );
    row.insert("name".into(), Value::String(parts[1].into()));
    for (key, index) in [
        ("util", 2),
        ("mem_used", 3),
        ("mem_total", 4),
        ("temp", 5),
        ("power", 6),
        ("power_limit", 7),
    ] {
        row.insert(
            key.into(),
            parts
                .get(index)
                .and_then(|text| to_float(&Value::String((*text).into())))
                .map_or(Value::Null, Value::from),
        );
    }
    Some(row)
}

fn unescape_mount(value: &str) -> PathBuf {
    PathBuf::from(
        value
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\"),
    )
}
