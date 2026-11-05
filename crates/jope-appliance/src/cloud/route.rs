//! Where a run goes: this machine or the user's veox Cloud Workspace, and how a tier limit reads.
//!
//! The rule is the owner's (2026-11-05):
//!
//! * `--local` / `--cloud` decide outright. Bundled demos always run here and need no login.
//! * A Pro account's deep search (effort medium, high or ultra; the workspace's `deep_*`
//!   profiles) runs in the cloud.
//! * Otherwise the run is local, unless its table is larger than [`LOCAL_ROW_THRESHOLD`] rows,
//!   in which case the cloud is offered.
//!
//! The public build is CPU-only, so "no GPU" is the ordinary case and nothing here probes for
//! one. Everything in this file is pure: the terminal, `jopedime run` and the MCP adapter feed it
//! what they know and print [`Decision::line`].
use serde_json::Value;

/// Free's row ceiling in the workspace (`veox-app-contracts` `tier::FREE_MAX_DATA_ROWS`), which
/// is also the size above which this machine offers the cloud.
pub const LOCAL_ROW_THRESHOLD: u64 = 100_000;
/// Free's column ceiling (`tier::FREE_MAX_DATA_COLUMNS`).
pub const FREE_MAX_DATA_COLUMNS: u64 = 512;
pub const FREE_MAX_DATA_ROWS: u64 = 100_000;
pub const UPGRADE_URL: &str = "https://veox.ai/pricing";

/// The account's plan as `GET /api/cli/account` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Waitlist,
    Free,
    Pro,
}

impl Tier {
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "waitlist" => Some(Self::Waitlist),
            "free" => Some(Self::Free),
            "pro" => Some(Self::Pro),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Waitlist => "waitlist",
            Self::Free => "free",
            Self::Pro => "pro",
        }
    }
}

/// An explicit `--local` or `--cloud`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Force {
    Local,
    Cloud,
}

/// The workspace search profile for an engine effort (`veox-app-contracts` `SearchProfile`).
pub fn profile(effort: jope_api::Effort) -> &'static str {
    match effort {
        jope_api::Effort::Fast => "fast",
        jope_api::Effort::Medium => "deep_medium",
        jope_api::Effort::High => "deep_high",
        jope_api::Effort::Ultra => "deep_ultra",
    }
}

fn effort_name(effort: jope_api::Effort) -> &'static str {
    match effort {
        jope_api::Effort::Fast => "fast",
        jope_api::Effort::Medium => "medium",
        jope_api::Effort::High => "high",
        jope_api::Effort::Ultra => "ultra",
    }
}

/// What the caller knows when it asks.
#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub effort: jope_api::Effort,
    /// Data rows of the table, when known.
    pub rows: Option<u64>,
    pub demo: bool,
    pub force: Option<Force>,
    /// `None` when signed out or the workspace could not be asked.
    pub tier: Option<Tier>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Local,
    Cloud,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    AskedLocal,
    AskedCloud,
    Demo,
    ProDeepSearch,
    LargeTable,
    /// The table is too large for this machine and also for Free's cloud ceiling.
    LargeTableFree,
    Default,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub target: Target,
    pub reason: Reason,
    /// Ask "Run it in your veox Cloud Workspace instead? [Y/n]" before running locally.
    pub offer_cloud: bool,
    /// The one line the CLI prints.
    pub line: String,
}

impl Decision {
    /// The `cloud_fallback` field the workspace records for a cloud run.
    pub fn cloud_fallback(&self) -> &'static str {
        match self.reason {
            Reason::AskedCloud => "user_choice",
            _ => "no_gpu",
        }
    }
}

pub fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn decide(input: Input) -> Decision {
    let rows = input
        .rows
        .map(|rows| format!(" · {} rows", grouped(rows)))
        .unwrap_or_default();
    let effort = effort_name(input.effort);
    let local = |reason, why: &str| Decision {
        target: Target::Local,
        reason,
        offer_cloud: false,
        line: format!("Running locally on this machine · {why}{rows}."),
    };
    let cloud = |reason, why: &str| Decision {
        target: Target::Cloud,
        reason,
        offer_cloud: false,
        line: format!("Running in your veox Cloud Workspace · {why}{rows}."),
    };
    if input.demo {
        return local(Reason::Demo, "demos always run here, no login needed");
    }
    match input.force {
        Some(Force::Local) => return local(Reason::AskedLocal, "you asked for --local"),
        Some(Force::Cloud) => return cloud(Reason::AskedCloud, "you asked for --cloud"),
        None => {}
    }
    if input.effort != jope_api::Effort::Fast && input.tier == Some(Tier::Pro) {
        return cloud(
            Reason::ProDeepSearch,
            &format!("Pro deep search ({effort}) runs in the cloud"),
        );
    }
    if input.rows.is_some_and(|rows| rows > LOCAL_ROW_THRESHOLD) {
        if input.tier == Some(Tier::Free) {
            return Decision {
                target: Target::Local,
                reason: Reason::LargeTableFree,
                offer_cloud: false,
                line: format!(
                    "Running locally on this machine{rows}. That is more than runs comfortably here \
                     (about {}), and Free cloud runs take up to {} rows too; Pro lifts that limit: {UPGRADE_URL}",
                    grouped(LOCAL_ROW_THRESHOLD),
                    grouped(FREE_MAX_DATA_ROWS)
                ),
            };
        }
        return Decision {
            target: Target::Local,
            reason: Reason::LargeTable,
            offer_cloud: true,
            line: format!(
                "This table is large for this machine{rows} (about {} runs comfortably here).",
                grouped(LOCAL_ROW_THRESHOLD)
            ),
        };
    }
    local(Reason::Default, &format!("{effort} search"))
}

/// The question that follows an [`Decision::offer_cloud`] decision.
pub const OFFER_CLOUD: &str = "Run it in your veox Cloud Workspace instead? [Y/n]";

/// A machine-readable tier refusal (`{error:"limit", …}`, ai-veox-app `limit.rs`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limit {
    pub status: u16,
    pub limit: String,
    pub used: Option<u64>,
    pub allowed: Option<u64>,
    /// RFC 3339 UTC.
    pub resets_at: Option<String>,
    pub upgrade_url: String,
    pub message: String,
}

impl Limit {
    /// The question that follows a refusal or a declined upload.
    pub const fn local_question() -> &'static str {
        "Run locally instead? [Y/n]"
    }

    /// `None` unless the body is the limit shape.
    pub fn parse(status: u16, body: &Value) -> Option<Self> {
        if body["error"] != "limit" {
            return None;
        }
        let upgrade = body["upgrade_url"]
            .as_str()
            .filter(|url| url.starts_with("https://") && !url.chars().any(char::is_control))
            .unwrap_or(UPGRADE_URL);
        Some(Self {
            status,
            limit: body["limit"].as_str().unwrap_or("unknown").to_owned(),
            used: body["used"].as_u64(),
            allowed: body["allowed"].as_u64(),
            resets_at: body["resets_at"].as_str().map(str::to_owned),
            upgrade_url: upgrade.to_owned(),
            message: crate::terminal::sanitize_user_string(body["message"].as_str().unwrap_or("")),
        })
    }

    /// The plain sentence, without the question.
    pub fn sentence(&self) -> String {
        let n = |v: Option<u64>| v.map(grouped).unwrap_or_else(|| "?".into());
        match self.limit.as_str() {
            "daily_runs" => {
                let resets = self
                    .resets_at
                    .as_deref()
                    .and_then(clock)
                    .map(|at| format!(" Resets at {at}."))
                    .unwrap_or_default();
                format!(
                    "Free includes {} cloud runs a day; you've used {}.{resets}",
                    n(self.allowed),
                    n(self.used)
                )
            }
            "concurrency" => {
                "Free runs one cloud run at a time, and one of yours is already going.".into()
            }
            "profile" => {
                "Deep search in the cloud is part of Pro; Free runs the Fast search there.".into()
            }
            "waitlist" => {
                "Your account is still on the waitlist, so cloud runs are not open to it yet."
                    .into()
            }
            "size" => {
                let what = if self.status == 413 {
                    "bytes"
                } else if self.allowed == Some(FREE_MAX_DATA_COLUMNS) {
                    "columns"
                } else {
                    "rows"
                };
                if what == "bytes" {
                    format!(
                        "Free cloud runs take files up to {}; this one is {}.",
                        self.allowed.map(size).unwrap_or_else(|| "?".into()),
                        self.used.map(size).unwrap_or_else(|| "?".into())
                    )
                } else {
                    format!(
                        "Free cloud runs take tables up to {} {what}; this one has {}.",
                        n(self.allowed),
                        n(self.used)
                    )
                }
            }
            _ if !self.message.is_empty() => self.message.clone(),
            _ => "Your plan's cloud limit was reached.".into(),
        }
    }

    /// The whole line: sentence, the local alternative, and where to upgrade.
    pub fn render(&self) -> String {
        format!(
            "{} Run locally instead? [Y/n] · Upgrade: {}",
            self.sentence(),
            self.upgrade_url
        )
    }
}

/// `HH:MM UTC` from an RFC 3339 UTC timestamp.
pub fn clock(rfc3339: &str) -> Option<String> {
    let time = rfc3339.split_once('T')?.1;
    let (h, rest) = time.split_once(':')?;
    let m = rest.get(..2)?;
    (h.len() == 2 && h.bytes().all(|b| b.is_ascii_digit()) && m.bytes().all(|b| b.is_ascii_digit()))
        .then(|| format!("{h}:{m} UTC"))
}

/// A human file size: `812 KB`, `3.4 MB`.
pub fn size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} B")
    } else if b < KB * KB {
        format!("{:.0} KB", b / KB)
    } else if b < KB * KB * KB {
        format!("{:.1} MB", b / KB / KB)
    } else {
        format!("{:.1} GB", b / KB / KB / KB)
    }
}

/// A yes/no answer: empty and `y`/`yes` are yes, `n`/`no` no, anything else is not an answer.
pub fn answer(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "" | "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::Effort;
    use serde_json::json;

    fn input(effort: Effort, rows: Option<u64>, tier: Option<Tier>) -> Input {
        Input {
            effort,
            rows,
            demo: false,
            force: None,
            tier,
        }
    }

    #[test]
    fn pro_deep_search_always_goes_to_the_cloud() {
        for effort in [Effort::Medium, Effort::High, Effort::Ultra] {
            let d = decide(input(effort, Some(10), Some(Tier::Pro)));
            assert_eq!(d.target, Target::Cloud, "{effort:?}");
            assert_eq!(d.reason, Reason::ProDeepSearch);
            assert_eq!(d.cloud_fallback(), "no_gpu");
            assert!(d.line.starts_with("Running in your veox Cloud Workspace"));
        }
        // Pro Fast stays local; Free and signed-out deep search stay local.
        assert_eq!(
            decide(input(Effort::Fast, Some(10), Some(Tier::Pro))).target,
            Target::Local
        );
        for tier in [Some(Tier::Free), Some(Tier::Waitlist), None] {
            let d = decide(input(Effort::High, Some(10), tier));
            assert_eq!(d.target, Target::Local, "{tier:?}");
            assert!(!d.offer_cloud);
        }
    }

    #[test]
    fn a_large_table_offers_the_cloud_unless_free_could_not_take_it_either() {
        let small = decide(input(Effort::Fast, Some(LOCAL_ROW_THRESHOLD), None));
        assert_eq!((small.target, small.offer_cloud), (Target::Local, false));
        for tier in [None, Some(Tier::Pro)] {
            let large = decide(input(Effort::Fast, Some(LOCAL_ROW_THRESHOLD + 1), tier));
            assert_eq!((large.target, large.offer_cloud), (Target::Local, true));
            assert_eq!(large.reason, Reason::LargeTable);
            assert!(large.line.contains("100,001 rows"), "{}", large.line);
        }
        let free = decide(input(Effort::Fast, Some(250_000), Some(Tier::Free)));
        assert_eq!((free.target, free.offer_cloud), (Target::Local, false));
        assert!(free
            .line
            .contains("Free cloud runs take up to 100,000 rows"));
        assert!(free.line.contains(UPGRADE_URL));
        assert_eq!(
            decide(input(Effort::Fast, None, None)).reason,
            Reason::Default
        );
    }

    #[test]
    fn overrides_win_and_demos_stay_local() {
        let mut i = input(Effort::High, Some(500_000), Some(Tier::Pro));
        i.force = Some(Force::Local);
        assert_eq!(decide(i).target, Target::Local);
        i.force = Some(Force::Cloud);
        let d = decide(i);
        assert_eq!(
            (d.target, d.cloud_fallback()),
            (Target::Cloud, "user_choice")
        );
        let mut f = input(Effort::Fast, Some(5), None);
        f.force = Some(Force::Cloud);
        assert_eq!(decide(f).target, Target::Cloud);
        i.demo = true;
        let demo = decide(i);
        assert_eq!((demo.target, demo.reason), (Target::Local, Reason::Demo));
        assert!(demo.line.contains("no login"));
    }

    #[test]
    fn profiles_match_the_workspace_names() {
        assert_eq!(profile(Effort::Fast), "fast");
        assert_eq!(profile(Effort::Medium), "deep_medium");
        assert_eq!(profile(Effort::High), "deep_high");
        assert_eq!(profile(Effort::Ultra), "deep_ultra");
    }

    #[test]
    fn limit_refusals_render_in_plain_language() {
        let daily = Limit::parse(
            429,
            &json!({"error":"limit","limit":"daily_runs","used":10,"allowed":10,
                    "resets_at":"2026-11-05T14:20:00Z","upgrade_url":"https://veox.ai/pricing",
                    "message":"You've started 10 runs today"}),
        )
        .unwrap();
        assert_eq!(
            daily.render(),
            "Free includes 10 cloud runs a day; you've used 10. Resets at 14:20 UTC. \
             Run locally instead? [Y/n] · Upgrade: https://veox.ai/pricing"
        );
        let cases = [
            (
                409,
                json!({"limit":"concurrency","used":1,"allowed":1}),
                "one cloud run at a time",
            ),
            (
                403,
                json!({"limit":"profile"}),
                "Deep search in the cloud is part of Pro",
            ),
            (403, json!({"limit":"waitlist"}), "still on the waitlist"),
            (
                400,
                json!({"limit":"size","used":250000,"allowed":100000}),
                "up to 100,000 rows; this one has 250,000",
            ),
            (
                400,
                json!({"limit":"size","used":900,"allowed":512}),
                "up to 512 columns; this one has 900",
            ),
            (
                413,
                json!({"limit":"size","used":31457280,"allowed":20971520}),
                "files up to 20.0 MB; this one is 30.0 MB",
            ),
            (
                403,
                json!({"limit":"brand_new","message":"A new limit."}),
                "A new limit.",
            ),
        ];
        for (status, mut body, expect) in cases {
            body["error"] = json!("limit");
            let limit = Limit::parse(status, &body).unwrap();
            assert!(limit.render().contains(expect), "{}", limit.render());
            assert!(limit.render().ends_with("Upgrade: https://veox.ai/pricing"));
        }
        // Not the limit shape; a hostile upgrade URL is replaced.
        assert!(Limit::parse(400, &json!({"error":"consent missing"})).is_none());
        let odd = Limit::parse(
            429,
            &json!({"error":"limit","limit":"daily_runs","upgrade_url":"javascript:alert(1)"}),
        )
        .unwrap();
        assert_eq!(odd.upgrade_url, UPGRADE_URL);
        assert!(!odd.render().contains("Resets"));
    }

    #[test]
    fn answers_default_to_yes() {
        assert_eq!(answer(""), Some(true));
        assert_eq!(answer(" Y "), Some(true));
        assert_eq!(answer("no"), Some(false));
        assert_eq!(answer("/run"), None);
    }
}
