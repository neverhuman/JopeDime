//! `/usage`: the account and plan, and run statistics for this machine (lifetime and recent).
//!
//! Local figures come from the engine's run list: each run ID is a ULID, so its creation time
//! is known without another record. The plan comes from the connected workspace's existing
//! `GET /api/cli/account` response, read in the background so the screen never waits on it.

use jope_api::{RunSnapshot, RunState};

/// "Recent" covers the last seven days.
pub const RECENT_MS: u64 = 7 * 24 * 60 * 60 * 1000;

#[derive(Debug, Default, PartialEq)]
pub struct Tally {
    pub runs: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub active: usize,
    pub evaluations: u64,
    pub seconds: f64,
}

impl Tally {
    fn add(&mut self, run: &RunSnapshot) {
        self.runs += 1;
        match run.state {
            RunState::Completed => self.completed += 1,
            RunState::Failed => self.failed += 1,
            RunState::Cancelled => self.cancelled += 1,
            _ => self.active += 1,
        }
        self.evaluations += run.budget_used.evaluations.0;
        self.seconds += run
            .budget_used
            .wall_seconds
            .parse::<f64>()
            .ok()
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .unwrap_or(0.0);
    }

    fn line(&self, label: &str) -> String {
        if self.runs == 0 {
            return format!("  {label:<12} no runs");
        }
        let mut parts = vec![
            format!("{} run{}", self.runs, if self.runs == 1 { "" } else { "s" }),
            format!("{} completed", self.completed),
        ];
        if self.failed > 0 {
            parts.push(format!("{} failed", self.failed));
        }
        if self.cancelled > 0 {
            parts.push(format!("{} cancelled", self.cancelled));
        }
        if self.active > 0 {
            parts.push(format!("{} active", self.active));
        }
        parts.push(format!("{} evaluations", group(self.evaluations)));
        parts.push(format!("{} compute", duration(self.seconds)));
        format!("  {label:<12} {}", parts.join(" · "))
    }
}

/// Milliseconds since the Unix epoch when the run was created, from its ULID.
pub fn created_ms(run: &RunSnapshot) -> Option<u64> {
    let text = run.run_id.to_string();
    let ulid = text.strip_prefix("run_").unwrap_or(&text);
    ulid::Ulid::from_string(ulid)
        .ok()
        .map(|id| id.timestamp_ms())
}

/// (lifetime, recent) tallies for runs recorded by this machine's engine.
pub fn tally(runs: &[RunSnapshot], now_ms: u64) -> (Tally, Tally) {
    let mut lifetime = Tally::default();
    let mut recent = Tally::default();
    for run in runs {
        lifetime.add(run);
        if created_ms(run).is_some_and(|ms| now_ms.saturating_sub(ms) <= RECENT_MS) {
            recent.add(run);
        }
    }
    (lifetime, recent)
}

/// The `/usage` lines that need no network.
pub fn local_lines(
    runs: Option<&[RunSnapshot]>,
    now_ms: u64,
    account: Option<(&str, &str)>,
    current: Option<&RunSnapshot>,
    telemetry: &str,
    terms: &str,
) -> Vec<String> {
    let mut lines = vec!["Usage".to_owned()];
    match account {
        Some((email, origin)) => {
            lines.push(format!("  {:<12} {email} · {origin}", "Account"));
            lines.push(format!("  {:<12} checking…", "Plan"));
        }
        None => {
            lines.push(format!(
                "  {:<12} not signed in · /login (accounts open from the waitlist: https://app.veox.ai/register)",
                "Account"
            ));
            lines.push(format!(
                "  {:<12} Free (signed out: bundled demos only)",
                "Plan"
            ));
        }
    }
    lines.push(match current {
        Some(run) => format!(
            "  {:<12} {} · {:?} · {} evaluations",
            "Status",
            run.run_id,
            run.state,
            group(run.budget_used.evaluations.0)
        ),
        None => format!(
            "  {:<12} no run in this session · /demo or /open to start",
            "Status"
        ),
    });
    match runs {
        Some(runs) => {
            let (lifetime, recent) = tally(runs, now_ms);
            lines.push(recent.line("Last 7 days"));
            lines.push(lifetime.line("Lifetime"));
            lines.push("  (runs recorded by the engine on this machine)".into());
        }
        None => lines.push(format!(
            "  {:<12} engine unavailable; run statistics unknown",
            "Runs"
        )),
    }
    lines.push(format!("  {:<12} {telemetry} · Terms {terms}", "Settings"));
    lines
}

/// The plan named in a `GET /api/cli/account` response: `tier`, `plan`, or `plan.name`.
pub fn plan_name(account: &serde_json::Value) -> Option<String> {
    let value = account
        .get("tier")
        .and_then(|v| v.as_str())
        .or_else(|| account.get("plan").and_then(|v| v.as_str()))
        .or_else(|| account.pointer("/plan/name").and_then(|v| v.as_str()))?;
    let clean: String = value.chars().filter(|c| !c.is_control()).take(40).collect();
    let clean = clean.trim();
    if clean.is_empty() {
        return None;
    }
    let mut chars = clean.chars();
    Some(match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    })
}

/// The background plan line for a connected account.
#[cfg(feature = "client")]
pub fn fetch_plan_line(connection: &crate::cloud::Connection) -> String {
    let result = crate::cloud::Client::connected(connection)
        .and_then(|client| client.json("GET", "/api/cli/account", serde_json::Value::Null));
    match result {
        Ok(account) => match plan_name(&account) {
            Some(plan) => format!(
                "  {:<12} {plan} · manage at {}/account",
                "Plan", connection.origin
            ),
            None => format!(
                "  {:<12} not reported by the workspace · see {}/account",
                "Plan", connection.origin
            ),
        },
        Err(error) => format!("  {:<12} unavailable ({error:#})", "Plan"),
    }
}

fn group(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn duration(seconds: f64) -> String {
    let total = seconds.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(ms: u64, state: &str, evaluations: u64, seconds: &str) -> RunSnapshot {
        let id = ulid::Ulid::from_parts(ms, 7);
        serde_json::from_value(json!({
            "run_id": format!("run_{id}"),
            "revision": "1",
            "state": state,
            "phase": "preparing",
            "pending": null,
            "effort": "medium",
            "budget_used": {"evaluations": evaluations.to_string(), "wall_seconds": seconds},
            "budget_total": {"evaluations": "100", "wall_seconds": "60"},
            "best": null,
            "event_watermark": "0",
            "warnings": [],
            "completion_reason": null,
            "evaluation_receipt": null,
        }))
        .unwrap()
    }

    #[test]
    fn lifetime_and_recent_split_by_the_run_id_time() {
        let now = 1_800_000_000_000;
        let runs = vec![
            run(now - 1_000, "completed", 1200, "65.5"),
            run(now - RECENT_MS - 1, "failed", 10, "3"),
            run(now - 60_000, "running", 5, "1"),
        ];
        let (lifetime, recent) = tally(&runs, now);
        assert_eq!(
            (
                lifetime.runs,
                lifetime.completed,
                lifetime.failed,
                lifetime.active
            ),
            (3, 1, 1, 1)
        );
        assert_eq!(lifetime.evaluations, 1215);
        assert_eq!((recent.runs, recent.failed), (2, 0));
        assert_eq!(
            recent.line("Last 7 days"),
            "  Last 7 days  2 runs · 1 completed · 1 active · 1,205 evaluations · 1m 07s compute"
        );
        let lines = local_lines(
            Some(&runs),
            now,
            None,
            Some(&runs[2]),
            "Usage statistics off",
            "2026-11-03 accepted",
        );
        assert!(lines
            .iter()
            .any(|l| l.starts_with("  Status       run_") && l.contains("Running")));
        assert!(lines.iter().any(|l| l.contains("not signed in")));
        assert!(lines.iter().any(|l| l.starts_with("  Lifetime     3 runs")));
        assert!(local_lines(
            None,
            now,
            Some(("a@b.c", "https://app.veox.ai")),
            None,
            "",
            ""
        )
        .iter()
        .any(|l| l.contains("checking")));
    }

    #[test]
    fn plan_names_are_read_from_any_reported_shape_and_cleaned() {
        assert_eq!(plan_name(&json!({"tier": "free"})).as_deref(), Some("Free"));
        assert_eq!(plan_name(&json!({"plan": "pro"})).as_deref(), Some("Pro"));
        assert_eq!(
            plan_name(&json!({"plan": {"name": "enterprise"}})).as_deref(),
            Some("Enterprise")
        );
        assert_eq!(
            plan_name(&json!({"tier": "\u{1b}[31m"})).as_deref(),
            Some("[31m")
        );
        assert_eq!(plan_name(&json!({"tier": "  "})), None);
        assert_eq!(plan_name(&json!({})), None);
    }
}
