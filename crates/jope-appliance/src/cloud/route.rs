//! Where a run goes, who may start it, and how a tier limit reads.
//!
//! The rule is the owner's (2026-11-06):
//!
//! * Nobody downloads model weights: the public jAIn CLI is a front end. Its local engine is
//!   weightless and runs only the bundled demos (`/demo`, `example/NAME.csv`), signed out or not.
//! * The user's own data trains in the veox Cloud. That needs a login (it is how runs are
//!   metered per plan) on an account admitted from the waitlist, and the user's consent to upload
//!   each table. `--cloud` is implied; `--local` runs only the bundled examples.
//! * Engine and development builds (not the public edition) cannot reach the HTTPS cloud, so
//!   they keep training own data on this machine unless `--cloud` is given or the tests ask for
//!   the public rule with `JAIN_PUBLIC_EDITION=1`.
//!
//! Everything in this file is pure: the terminal, `jain run` and the MCP adapter feed it what
//! they know and print [`Decision::line`].
use serde_json::Value;

/// Free's row ceiling in the workspace (`veox-app-contracts` `tier::FREE_MAX_DATA_ROWS`).
pub const FREE_MAX_DATA_ROWS: u64 = 100_000;
/// Free's column ceiling (`tier::FREE_MAX_DATA_COLUMNS`).
pub const FREE_MAX_DATA_COLUMNS: u64 = 512;
pub const UPGRADE_URL: &str = "https://veox.ai/pricing";
/// Where a new account joins the waitlist (accounts open from it in waves).
pub const REGISTER_URL: &str = "https://app.veox.ai/register";

/// Whether own data trains only in the veox Cloud: always in the public edition (the released
/// CLI); in other builds only when `JAIN_PUBLIC_EDITION=1` asks for the public rule (tests).
pub fn cloud_first() -> bool {
    cloud_first_with(std::env::var("JAIN_PUBLIC_EDITION").ok())
}

pub fn cloud_first_with(var: Option<String>) -> bool {
    cfg!(feature = "public-edition")
        || var.is_some_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

/// The account's plan as `GET /api/cli/account` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountTier {
    Waitlist,
    Free,
    Pro,
}

impl AccountTier {
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

/// The profile a plan can run: Free runs the Fast search in the cloud (deep search is Pro's),
/// so a deep effort on Free is sent as `fast` with [`FREE_RUNS_FAST`] said once, rather than
/// refused after the upload.
pub fn profile_for(
    effort: jope_api::Effort,
    tier: Option<AccountTier>,
) -> (&'static str, Option<String>) {
    let wanted = profile(effort);
    if tier == Some(AccountTier::Free) && wanted != "fast" {
        return ("fast", Some(FREE_RUNS_FAST.to_owned()));
    }
    (wanted, None)
}

pub const FREE_RUNS_FAST: &str = "Free runs the Fast search in the veox Cloud; deep search (medium, high, ultra) is part of Pro: https://veox.ai/pricing";

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
    /// A bundled demo (`/demo`, `example/NAME.csv`).
    pub demo: bool,
    pub force: Option<Force>,
    /// [`cloud_first`]: own data trains only in the cloud.
    pub cloud_first: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteTarget {
    Local,
    Cloud,
    /// Not started anywhere: the line says why and what to do.
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Demo,
    /// Own data in the cloud, by the rule.
    OwnData,
    AskedCloud,
    /// `--local` on own data where only the bundled examples run here.
    LocalIsForDemos,
    /// An engine or development build training own data here.
    LocalBuild,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub target: RouteTarget,
    pub reason: Reason,
    /// The one line the CLI prints.
    pub line: String,
}

impl Decision {
    /// The `cloud_fallback` field the workspace records for a cloud run: the user's explicit
    /// `--cloud`, or this machine lacking what the run needs (the public CLI has no models).
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

/// Said when `--local` (or `/run --local`) names the user's own data.
pub const LOCAL_IS_FOR_DEMOS: &str = "Your own data trains in the veox Cloud, not on this machine: --local runs only the bundled examples (example/NAME.csv). Leave out --local to run it in the cloud; jAIn asks before uploading.";

pub fn decide(input: Input) -> Decision {
    let rows = input
        .rows
        .map(|rows| format!(" · {} rows", grouped(rows)))
        .unwrap_or_default();
    let effort = effort_name(input.effort);
    let local = |reason, why: &str| Decision {
        target: RouteTarget::Local,
        reason,
        line: format!("Running locally on this machine · {why}{rows}."),
    };
    let cloud = |reason, why: &str| Decision {
        target: RouteTarget::Cloud,
        reason,
        line: format!("Training in the veox Cloud · {why}{rows}."),
    };
    match (input.demo, input.force) {
        (_, Some(Force::Cloud)) => cloud(Reason::AskedCloud, "you asked for --cloud"),
        (true, _) => local(Reason::Demo, "a bundled demo, no account needed"),
        (false, Some(Force::Local)) if input.cloud_first => Decision {
            target: RouteTarget::Refused,
            reason: Reason::LocalIsForDemos,
            line: LOCAL_IS_FOR_DEMOS.into(),
        },
        (false, _) if input.cloud_first => cloud(
            Reason::OwnData,
            &format!("your own data trains there ({effort} search)"),
        ),
        (false, _) => local(
            Reason::LocalBuild,
            &format!("{effort} search (this build trains own data here)"),
        ),
    }
}

/// Why an own-data action cannot go on yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    SignedOut,
    /// Signed in, but the account has not been let in from the waitlist (`mayRun:false`).
    Waitlisted {
        email: String,
    },
}

/// The gate for an own-data action. `account` is the signed-in email with the plan and
/// `mayRun` the workspace reported, or `None` when it could not be asked (then the workspace
/// itself refuses a waitlisted account's upload).
pub fn gate(
    signed_in: Option<&str>,
    account: Option<(Option<AccountTier>, bool)>,
) -> Result<(), Gate> {
    let Some(email) = signed_in else {
        return Err(Gate::SignedOut);
    };
    match account {
        Some((tier, may_run)) if tier == Some(AccountTier::Waitlist) || !may_run => {
            Err(Gate::Waitlisted {
                email: email.to_owned(),
            })
        }
        _ => Ok(()),
    }
}

impl Gate {
    /// The message and its next step; `tui` picks `/login` over `jain auth login`.
    pub fn message(&self, tui: bool) -> String {
        match self {
            Self::SignedOut => {
                let (login, demo) = if tui {
                    ("type /login", "/demo")
                } else {
                    (
                        "run `jain auth login`",
                        "`jain demo run regression_parkinsons`",
                    )
                };
                format!(
                    "Your own data trains in the veox Cloud, so it needs you signed in: {login}. \
                     New to veox? Accounts open from the waitlist: {REGISTER_URL}. \
                     The bundled demos need no account: {demo}"
                )
            }
            Self::Waitlisted { email } => {
                let demo = if tui {
                    "/demo"
                } else {
                    "`jain demo run regression_parkinsons`"
                };
                format!(
                    "Your account ({email}) is on the waitlist; we'll email you when you're \
                     admitted, and then your own data can train in the veox Cloud. What that \
                     means: {REGISTER_URL}. Until then the bundled demos work: {demo}"
                )
            }
        }
    }
}

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
            "waitlist" => format!(
                "Your account is still on the waitlist, so cloud runs are not open to it yet; \
                 we'll email you when you're admitted ({REGISTER_URL})."
            ),
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

    /// The whole line: the sentence, that no run started, and where to upgrade. A waitlist
    /// refusal has no upgrade to offer.
    pub fn render(&self) -> String {
        if self.limit == "waitlist" {
            return format!("{} No run started.", self.sentence());
        }
        format!(
            "{} No run started. · Upgrade: {}",
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

/// The answer to an upload question (`[y/N]`): only `y`/`yes` uploads; empty and `n`/`no` do
/// not; anything else is not an answer.
pub fn consent_answer(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Some(true),
        "" | "n" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::Effort;
    use serde_json::json;

    fn input(demo: bool, force: Option<Force>, cloud_first: bool) -> Input {
        Input {
            effort: Effort::Fast,
            rows: Some(149),
            demo,
            force,
            cloud_first,
        }
    }

    #[test]
    fn own_data_trains_in_the_cloud_and_demos_stay_here() {
        // The public rule: own data goes to the cloud whatever the effort or size.
        for effort in [Effort::Fast, Effort::Medium, Effort::High, Effort::Ultra] {
            let mut i = input(false, None, true);
            i.effort = effort;
            let d = decide(i);
            assert_eq!((d.target, d.reason), (RouteTarget::Cloud, Reason::OwnData));
            assert_eq!(d.cloud_fallback(), "no_gpu");
            assert!(
                d.line.starts_with("Training in the veox Cloud"),
                "{}",
                d.line
            );
            assert!(d.line.ends_with("149 rows."), "{}", d.line);
        }
        let demo = decide(input(true, None, true));
        assert_eq!(
            (demo.target, demo.reason),
            (RouteTarget::Local, Reason::Demo)
        );
        assert!(demo.line.contains("no account needed"), "{}", demo.line);
        // --local is for the bundled examples only.
        let local = decide(input(false, Some(Force::Local), true));
        assert_eq!(local.target, RouteTarget::Refused);
        assert_eq!(local.line, LOCAL_IS_FOR_DEMOS);
        assert_eq!(
            decide(input(true, Some(Force::Local), true)).target,
            RouteTarget::Local
        );
        // --cloud is explicit everywhere, demos included.
        let cloud = decide(input(false, Some(Force::Cloud), true));
        assert_eq!(
            (cloud.target, cloud.cloud_fallback()),
            (RouteTarget::Cloud, "user_choice")
        );
        assert_eq!(
            decide(input(true, Some(Force::Cloud), true)).target,
            RouteTarget::Cloud
        );
    }

    #[test]
    fn engine_and_development_builds_keep_training_here_unless_asked() {
        let d = decide(input(false, None, false));
        assert_eq!(
            (d.target, d.reason),
            (RouteTarget::Local, Reason::LocalBuild)
        );
        assert!(d.line.starts_with("Running locally on this machine"));
        assert_eq!(
            decide(input(false, Some(Force::Local), false)).target,
            RouteTarget::Local
        );
        assert_eq!(
            decide(input(false, Some(Force::Cloud), false)).target,
            RouteTarget::Cloud
        );
        // The public edition always applies the rule; elsewhere JAIN_PUBLIC_EDITION=1 does.
        assert!(cloud_first_with(Some("1".into())));
        assert_eq!(
            cloud_first_with(None),
            cfg!(feature = "public-edition"),
            "the public build needs no variable"
        );
        assert_eq!(
            cloud_first_with(Some("0".into())),
            cfg!(feature = "public-edition")
        );
    }

    #[test]
    fn the_gate_needs_a_login_and_an_admitted_account() {
        assert_eq!(gate(None, None), Err(Gate::SignedOut));
        assert_eq!(
            gate(Some("a@b.c"), Some((Some(AccountTier::Free), true))),
            Ok(())
        );
        assert_eq!(
            gate(Some("a@b.c"), Some((Some(AccountTier::Pro), true))),
            Ok(())
        );
        // The workspace could not be asked: let the workspace refuse it, if it must.
        assert_eq!(gate(Some("a@b.c"), None), Ok(()));
        for account in [
            (Some(AccountTier::Waitlist), false),
            (Some(AccountTier::Free), false),
        ] {
            assert_eq!(
                gate(Some("a@b.c"), Some(account)),
                Err(Gate::Waitlisted {
                    email: "a@b.c".into()
                })
            );
        }
        let out = Gate::SignedOut.message(true);
        assert!(out.contains("/login") && out.contains("/demo"), "{out}");
        assert!(out.contains("Accounts open from the waitlist: https://app.veox.ai/register."));
        let cli = Gate::SignedOut.message(false);
        assert!(
            cli.contains("`jain auth login`") && !cli.contains("/login"),
            "{cli}"
        );
        let wait = Gate::Waitlisted {
            email: "a@b.c".into(),
        }
        .message(true);
        assert!(
            wait.starts_with(
                "Your account (a@b.c) is on the waitlist; we'll email you when you're admitted"
            ),
            "{wait}"
        );
        assert!(wait.contains(REGISTER_URL));
    }

    #[test]
    fn profiles_match_the_workspace_names_and_free_runs_fast() {
        assert_eq!(profile(Effort::Fast), "fast");
        assert_eq!(profile(Effort::Medium), "deep_medium");
        assert_eq!(profile(Effort::High), "deep_high");
        assert_eq!(profile(Effort::Ultra), "deep_ultra");
        assert_eq!(
            profile_for(Effort::High, Some(AccountTier::Pro)),
            ("deep_high", None)
        );
        assert_eq!(
            profile_for(Effort::Fast, Some(AccountTier::Free)),
            ("fast", None)
        );
        let (free, note) = profile_for(Effort::Medium, Some(AccountTier::Free));
        assert_eq!(free, "fast");
        assert!(note
            .unwrap()
            .contains("part of Pro: https://veox.ai/pricing"));
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
             No run started. · Upgrade: https://veox.ai/pricing"
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
            assert!(
                !limit.render().contains("Run locally"),
                "{}",
                limit.render()
            );
            assert!(limit.render().ends_with("Upgrade: https://veox.ai/pricing"));
        }
        let waitlist = Limit::parse(403, &json!({"error":"limit","limit":"waitlist"})).unwrap();
        assert!(
            waitlist.render().contains("still on the waitlist")
                && waitlist
                    .render()
                    .contains("we'll email you when you're admitted")
                && !waitlist.render().contains("Upgrade"),
            "{}",
            waitlist.render()
        );
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
    fn answers_default_to_yes_but_uploads_default_to_no() {
        assert_eq!(answer(""), Some(true));
        assert_eq!(answer(" Y "), Some(true));
        assert_eq!(answer("no"), Some(false));
        assert_eq!(answer("/run"), None);
        assert_eq!(consent_answer(""), Some(false), "Enter uploads nothing");
        assert_eq!(consent_answer(" y "), Some(true));
        assert_eq!(consent_answer("YES"), Some(true));
        assert_eq!(consent_answer("n"), Some(false));
        assert_eq!(consent_answer("/demo"), None);
    }
}
