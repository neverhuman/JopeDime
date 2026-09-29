use serde::Deserialize;
use std::{
    fs::File,
    io::Read,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

pub const MAX_BYTES: u64 = 8 * 1024 * 1024;
#[derive(Deserialize)]
pub struct Heartbeat {
    pub feed_id: String,
    pub sequence: u64,
    pub generated_utc_us: i64,
}
pub fn pulse(path: &Path, s: &mut Snapshot) {
    let Ok(f) = File::open(format!("{}.heartbeat", path.display())) else {
        return;
    };
    let mut bytes = Vec::new();
    if f.take(2049).read_to_end(&mut bytes).is_err() || bytes.len() > 2048 {
        return;
    }
    if let Ok(h) = serde_json::from_slice::<Heartbeat>(&bytes) {
        s.heartbeat(h);
    }
}
pub fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Counts {
    pub dispatched: u64,
    pub completed: u64,
    pub failed: u64,
    pub unknown: u64,
    pub in_flight: u64,
    pub jobs: u64,
    pub job_failures: u64,
    pub candidates: u64,
    pub unknown_bills: u64,
    pub archived: u64,
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Budget {
    pub cap_usd: f64,
    pub reserved_usd: Option<f64>,
    pub lambda_usd: f64,
    pub s3_usd: f64,
    pub scope: String,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Rates {
    pub calls: Vec<u64>,
    pub jobs: Vec<u64>,
    pub candidates: Vec<u64>,
    pub active: Vec<u64>,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Tier {
    pub mb: u64,
    pub calls: u64,
    pub cold: u64,
    pub peak_mb: u64,
    pub billed_ms: f64,
    pub lambda_usd: f64,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct View {
    pub id: String,
    pub name: String,
    pub task: String,
    pub fold: String,
    pub status: String,
    pub generation: Option<u64>,
    pub target_generation: Option<u64>,
    pub candidates: u64,
    pub metric: String,
    pub direction: String,
    pub score: Option<f64>,
    pub baseline: Option<f64>,
    pub history: Vec<f64>,
    pub program: String,
    pub program_sha: String,
    pub context: String,
    pub rows: u64,
    pub features: u64,
    pub memory_mb: u64,
    pub duration_ms: Option<f64>,
    pub updated_utc_us: i64,
}
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Snapshot {
    pub schema: String,
    pub feed_id: String,
    pub sequence: u64,
    pub campaign: String,
    pub mode: String,
    pub status: String,
    pub generated_utc_us: i64,
    pub cursor_utc_us: i64,
    pub last_event_utc_us: i64,
    pub scope: String,
    pub activity_label: String,
    pub notice: String,
    pub counts: Counts,
    pub budget: Budget,
    pub rates: Rates,
    pub tiers: Vec<Tier>,
    pub views: Vec<View>,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub errors: Vec<String>,
}
impl Snapshot {
    pub fn heartbeat(&mut self, h: Heartbeat) {
        if h.feed_id != self.feed_id
            || h.sequence != self.sequence
            || h.generated_utc_us < self.generated_utc_us
        {
            return;
        }
        self.generated_utc_us = h.generated_utc_us;
        if self.mode != "replay" && h.generated_utc_us > self.cursor_utc_us {
            let n =
                (h.generated_utc_us / 1_000_000 - self.cursor_utc_us / 1_000_000).max(0) as usize;
            for v in [
                &mut self.rates.calls,
                &mut self.rates.jobs,
                &mut self.rates.candidates,
                &mut self.rates.active,
            ] {
                let k = n.min(v.len());
                v.drain(..k);
                v.extend(std::iter::repeat(0).take(k));
            }
            self.cursor_utc_us = h.generated_utc_us;
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "jope.lambda-watch.snapshot.v1" {
            return Err("unsupported snapshot schema".into());
        }
        if !["live", "closed", "replay"].contains(&self.mode.as_str()) {
            return Err("invalid feed mode".into());
        }
        if self.campaign.is_empty() || self.feed_id.is_empty() || self.generated_utc_us <= 0 {
            return Err("missing campaign/feed/time identity".into());
        }
        if self.views.len() > 20_000 || self.tiers.len() > 64 || self.errors.len() > 100 {
            return Err("snapshot exceeds collection limits".into());
        }
        if [
            &self.rates.calls,
            &self.rates.jobs,
            &self.rates.candidates,
            &self.rates.active,
        ]
        .iter()
        .any(|v| v.len() > 120)
        {
            return Err("rate window exceeds 120 seconds".into());
        }
        if self.budget.cap_usd <= 0.0
            || !self.budget.cap_usd.is_finite()
            || self.budget.lambda_usd < 0.0
            || self.budget.s3_usd < 0.0
            || self
                .budget
                .reserved_usd
                .is_some_and(|x| !x.is_finite() || x < 0.0)
        {
            return Err("invalid budget accounting".into());
        }
        let mut ids = std::collections::HashSet::new();
        for v in &self.views {
            if v.id.is_empty() || !ids.insert(&v.id) {
                return Err("duplicate or absent view identity".into());
            }
            if v.history.len() > 512 || v.program.len() > 65536 {
                return Err("view exceeds display limits".into());
            }
            if v.score.is_some() && !["min", "max"].contains(&v.direction.as_str()) {
                return Err("score has no metric direction".into());
            }
        }
        Ok(())
    }
    pub fn health(&self, now: i64) -> (&str, bool) {
        if self.generated_utc_us > now + 5_000_000 {
            return ("CLOCK SKEW", false);
        }
        if now.saturating_sub(self.generated_utc_us) > 8_000_000 {
            return ("STALE FEED", false);
        }
        match self.mode.as_str() {
            "replay" => ("REPLAY", false),
            "closed" => ("CLOSED", false),
            _ => ("LIVE", true),
        }
    }
}
pub fn load(path: &Path) -> Result<Snapshot, String> {
    let mut f = File::open(path).map_err(|_| "waiting for telemetry snapshot".to_string())?;
    if f.metadata().map_err(|_| "cannot stat snapshot")?.len() > MAX_BYTES {
        return Err("snapshot larger than 8 MiB".into());
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut f)
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "snapshot read failed")?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("snapshot larger than 8 MiB".into());
    }
    let s: Snapshot = serde_json::from_slice(&bytes)
        .map_err(|_| "incomplete or invalid snapshot; retaining last good frame".to_string())?;
    s.validate()?;
    Ok(s)
}
pub fn clean(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(65536).collect()
}
pub fn clock(us: i64) -> String {
    if us <= 0 {
        return "--:--:--.------ UTC".into();
    }
    let sec = us.div_euclid(1_000_000);
    let day = sec.rem_euclid(86400);
    format!(
        "{:02}:{:02}:{:02}.{:06} UTC",
        day / 3600,
        (day / 60) % 60,
        day % 60,
        us.rem_euclid(1_000_000)
    )
}
pub fn count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}m", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Snapshot {
        Snapshot {
            schema: "jope.lambda-watch.snapshot.v1".into(),
            feed_id: "producer-instance".into(),
            campaign: "test".into(),
            mode: "live".into(),
            generated_utc_us: 10_000_000,
            budget: Budget {
                cap_usd: 100.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    #[test]
    fn stale_is_not_live() {
        let s = valid();
        assert_eq!(s.health(19_000_000).0, "STALE FEED");
        assert_eq!(s.health(11_000_000), ("LIVE", true));
    }
    #[test]
    fn heartbeat_cannot_revive_another_feed() {
        let mut s = valid();
        s.heartbeat(Heartbeat {
            feed_id: "wrong".into(),
            sequence: 0,
            generated_utc_us: 30_000_000,
        });
        assert_eq!(s.health(30_000_000).0, "STALE FEED");
    }
    #[test]
    fn idle_rates_decay_to_zero() {
        let mut s = valid();
        s.cursor_utc_us = 10_000_000;
        s.rates.calls = vec![2, 3, 4];
        s.heartbeat(Heartbeat {
            feed_id: s.feed_id.clone(),
            sequence: 0,
            generated_utc_us: 12_000_000,
        });
        assert_eq!(s.rates.calls, vec![4, 0, 0]);
        assert_eq!(s.health(12_000_000).0, "LIVE");
    }
    #[test]
    fn replay_heartbeat_does_not_move_historical_clock() {
        let mut s = valid();
        s.mode = "replay".into();
        s.cursor_utc_us = 1;
        s.heartbeat(Heartbeat {
            feed_id: s.feed_id.clone(),
            sequence: 0,
            generated_utc_us: 12_000_000,
        });
        assert_eq!(s.cursor_utc_us, 1);
    }
    #[test]
    fn replay_never_live() {
        let mut s = valid();
        s.mode = "replay".into();
        assert!(!s.health(11_000_000).1);
    }
    #[test]
    fn clock_skew_is_visible() {
        assert_eq!(valid().health(1).0, "CLOCK SKEW");
    }
    #[test]
    fn duplicate_view_fails() {
        let mut s = valid();
        s.views = vec![
            View {
                id: "a".into(),
                ..Default::default()
            };
            2
        ];
        assert!(s.validate().is_err());
    }
    #[test]
    fn unknown_budget_is_not_zero() {
        let s = valid();
        assert!(s.validate().is_ok());
        assert!(s.budget.reserved_usd.is_none());
    }
    #[test]
    fn direction_required() {
        let mut s = valid();
        s.views.push(View {
            id: "a".into(),
            score: Some(0.9),
            ..Default::default()
        });
        assert!(s.validate().is_err());
    }
    #[test]
    fn absent_accounting_is_not_zero() {
        assert!(serde_json::from_str::<Counts>("{}").is_err());
        assert!(serde_json::from_str::<Budget>("{}").is_err());
        assert!(serde_json::from_str::<Snapshot>("{}").is_err());
    }
    #[test]
    fn exact_microseconds() {
        assert_eq!(clock(1790618717413207), "18:05:17.413207 UTC");
    }
    #[test]
    fn escape_is_not_terminal_code() {
        assert!(!clean("hello\x1b[2J\rworld").contains('\x1b'));
    }
}
