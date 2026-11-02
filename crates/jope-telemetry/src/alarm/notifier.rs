use super::*;

impl CommsNotifier {
    pub fn new(comms: impl Into<PathBuf>, lock: impl Into<PathBuf>) -> Self {
        Self {
            comms: comms.into(),
            lock: lock.into(),
            min_interval_secs: 3600.0,
            last: HashMap::new(),
            write_failures: 0,
            suppressed: 0,
        }
    }

    pub fn notify(&mut self, a: &Alarm, now_secs: f64) {
        if self
            .last
            .get(&a.id)
            .is_some_and(|&t| now_secs - t < self.min_interval_secs)
        {
            self.suppressed += 1;
            return;
        }
        self.last.insert(a.id, now_secs);
        if self.append(a).is_err() {
            self.write_failures += 1;
        }
    }

    fn append(&self, a: &Alarm) -> std::io::Result<()> {
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.lock)?;
        lock.lock()?;
        let ts = utc_now_rfc3339();
        let mut f = std::fs::OpenOptions::new().append(true).open(&self.comms)?;
        let sev = match a.severity {
            Severity::FailClosed => "FAIL-CLOSED",
            Severity::Alarm => "alarm",
        };
        let body = format!(
            "\n## [TEL-alarm] {ts} — {} {sev} run `{}` at {}\n{}\n",
            a.code,
            a.run_id,
            a.at,
            a.reasons
                .iter()
                .map(|r| format!("- {r}\n"))
                .collect::<String>()
                .trim_end()
        );
        f.write_all(body.as_bytes())?;
        f.flush()
        // The lock is released when `lock` drops.
    }
}
