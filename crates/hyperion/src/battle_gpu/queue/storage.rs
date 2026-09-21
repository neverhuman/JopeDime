//! Ticket-file storage for the cross-process GPU queue.

use super::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn cleanup_expired_tickets(dir: &Path, expired_after: Duration) -> Result<usize> {
    let mut removed = 0usize;
    for path in list_tickets(dir)? {
        let expired = path
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .map(|age| age > expired_after)
            .unwrap_or(true);
        if expired {
            match fs::remove_file(&path) {
                Ok(()) => removed += 1,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("remove expired gpu ticket {}", path.display()))
                }
            }
        }
    }
    Ok(removed)
}

pub(super) fn create_ticket(dir: &Path) -> Result<PathBuf> {
    for _ in 0..16 {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = TICKET_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let name = format!("{now_ns:020}-{pid:010}-{seq:016x}.ticket");
        let path = dir.join(name);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                writeln!(file, "pid={pid}")?;
                writeln!(file, "created_ns={now_ns}")?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("create gpu ticket {}", path.display()))
            }
        }
    }
    Err(anyhow!("could not allocate unique gpu queue ticket"))
}

pub(super) fn touch_ticket(path: &Path) -> Result<()> {
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("heartbeat gpu ticket {}", path.display()))?;
    writeln!(file, "pid={}", std::process::id())?;
    writeln!(file, "heartbeat_ns={now_ns}")?;
    Ok(())
}

pub(super) fn list_tickets(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut tickets = Vec::new();
    for entry in
        fs::read_dir(dir).with_context(|| format!("read gpu queue dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|suffix| suffix.to_str()) == Some("ticket") {
            tickets.push(path);
        }
    }
    tickets.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    Ok(tickets)
}

pub(super) fn env_duration_secs(key: &str, default_secs: u64) -> Duration {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(default_secs))
}

pub(super) fn env_duration_millis(key: &str, default_ms: u64) -> Duration {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(default_ms))
}
