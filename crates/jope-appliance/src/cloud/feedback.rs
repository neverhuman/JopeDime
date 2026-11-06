//! Feedback to the VEOX team: `jain cloud feedback`, the terminal's `/feedback`, and the local
//! MCP's `send_feedback` all end here, in one `POST /api/feedback` to the workspace.
//!
//! Signed in, the stored CLI token goes with it and the team can follow up on the account; not
//! signed in, it is sent without one, because a person (or an agent) who could not get as far as
//! logging in is exactly who the team most needs to hear from. The workspace rate-limits either
//! way and scrubs anything shaped like a credential before it stores the text.
//!
//! What is sent is only what the caller wrote, plus this binary's name and version. No dataset,
//! no file contents, no run data: a report that needs them says so in words.

use super::*;

/// The kinds the workspace accepts, as the API spells them.
pub const KINDS: [&str; 4] = ["working_well", "issue", "feature_request", "other"];

/// Who is speaking: the person, or an agent reporting on its own experience of the tools.
pub const SOURCES: [&str; 2] = ["user", "agent"];

/// The workspace's own limits, checked here first so a too-long note is a clear local error
/// rather than a refused request.
const MAX_SUMMARY: usize = 200;
const MAX_DETAILS: usize = 8000;

/// The friendlier words the terminal accepts, mapped to the API's kinds.
#[must_use]
pub fn kind_from_word(word: &str) -> Option<&'static str> {
    match word.to_ascii_lowercase().as_str() {
        "issue" | "bug" | "problem" => Some("issue"),
        "idea" | "feature" | "feature_request" | "request" => Some("feature_request"),
        "good" | "works" | "working_well" | "praise" | "thanks" => Some("working_well"),
        "other" | "note" => Some("other"),
        _ => None,
    }
}

/// The terminal's one line of text as a summary and, when it is long, the whole of it as details.
///
/// The workspace keeps summaries to [`MAX_SUMMARY`] characters so a list of them stays scannable;
/// nobody typing into `/feedback` should have to know that, so a long note is cut for the summary
/// and kept whole in the details rather than refused.
#[must_use]
pub fn split_text(text: &str) -> (String, Option<String>) {
    let text = text.trim();
    if text.chars().count() <= MAX_SUMMARY {
        return (text.to_owned(), None);
    }
    let cut: String = text.chars().take(MAX_SUMMARY - 3).collect();
    (format!("{}...", cut.trim_end()), Some(text.to_owned()))
}

/// One feedback message, validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feedback {
    pub kind: &'static str,
    pub source: &'static str,
    pub summary: String,
    pub details: Option<String>,
    /// Names of tools or commands used, for an agent's report. Names only, never arguments.
    pub tools_used: Vec<String>,
}

impl Feedback {
    /// Validate before anything leaves the machine.
    ///
    /// # Errors
    /// An unknown kind or source, an empty or over-long summary, or over-long details.
    pub fn new(kind: &str, source: &str, summary: &str, details: Option<&str>) -> Result<Self> {
        let kind = KINDS
            .iter()
            .copied()
            .find(|k| *k == kind)
            .or_else(|| kind_from_word(kind))
            .with_context(|| {
                format!("unknown feedback kind `{kind}`: use issue, idea, good or other")
            })?;
        let source = SOURCES
            .iter()
            .copied()
            .find(|s| *s == source)
            .with_context(|| format!("unknown feedback source `{source}`: use user or agent"))?;
        let summary = summary.trim();
        anyhow::ensure!(
            !summary.is_empty(),
            "say something: the feedback summary is empty"
        );
        anyhow::ensure!(
            summary.chars().count() <= MAX_SUMMARY,
            "keep the summary to {MAX_SUMMARY} characters; put the rest in details"
        );
        let details = details.map(str::trim).filter(|d| !d.is_empty());
        if let Some(details) = details {
            anyhow::ensure!(
                details.chars().count() <= MAX_DETAILS,
                "feedback details are limited to {MAX_DETAILS} characters"
            );
        }
        Ok(Self {
            kind,
            source,
            summary: summary.to_owned(),
            details: details.map(str::to_owned),
            tools_used: Vec::new(),
        })
    }

    /// The request body, exactly as `POST /api/feedback` reads it.
    #[must_use]
    pub fn body(&self) -> Value {
        let mut body = json!({
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "client": format!("jopedime {}", env!("CARGO_PKG_VERSION")),
            "surface_hint": "cli",
        });
        if let Some(details) = &self.details {
            body["details"] = json!(details);
        }
        if !self.tools_used.is_empty() {
            body["tools_used"] = json!(self.tools_used);
        }
        body
    }
}

/// Send it: signed in if this machine is, anonymous otherwise. Returns the line to show.
///
/// # Errors
/// Transport failures and a refusal from the workspace (for example its rate limit).
pub fn send(feedback: &Feedback) -> Result<String> {
    let (client, signed_in) = match connection() {
        Ok(c) => (Client::connected(&c)?, true),
        Err(_) => {
            let workspace = default_workspace();
            (
                Client::new(&workspace, None, gateway(None, &workspace)?)?,
                false,
            )
        }
    };
    let (status, bytes) = client
        .call(
            "POST",
            "/api/feedback",
            serde_json::to_vec(&feedback.body())?,
            "application/json",
            MAX_JSON,
        )
        .context("could not send feedback")?;
    let reply: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    anyhow::ensure!(status != 404 && status != 405, "{}", NOT_ACCEPTED);
    anyhow::ensure!(
        (200..300).contains(&status),
        "the workspace did not take the feedback (HTTP {status}{}); nothing was stored",
        reply
            .get("error")
            .and_then(Value::as_str)
            .map(|e| format!(": {e}"))
            .unwrap_or_default()
    );
    let id = reply
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("received");
    Ok(format!(
        "Thanks: feedback sent to the VEOX team ({}, ref {id}){}.",
        feedback.kind.replace('_', " "),
        if signed_in {
            " with your account so they can follow up"
        } else {
            " without an account (sign in with /login or `jain auth login` if you want a reply)"
        }
    ))
}

/// A workspace that predates `POST /api/feedback` answers 404: say so plainly instead of a
/// generic refusal, and nothing else breaks.
pub const NOT_ACCEPTED: &str = "this veox Cloud Workspace does not accept feedback from the CLI yet (HTTP 404); nothing was sent. Please try again after the next workspace update";

#[cfg(test)]
mod feedback_tests {
    use super::*;

    #[test]
    fn friendly_words_map_to_api_kinds() {
        assert_eq!(kind_from_word("bug"), Some("issue"));
        assert_eq!(kind_from_word("Idea"), Some("feature_request"));
        assert_eq!(kind_from_word("good"), Some("working_well"));
        assert_eq!(kind_from_word("note"), Some("other"));
        assert_eq!(kind_from_word("whatever"), None);
    }

    #[test]
    fn a_long_terminal_note_is_cut_for_the_summary_and_kept_whole() {
        assert_eq!(split_text("  short  "), ("short".into(), None));
        let long = "a".repeat(250);
        let (summary, details) = split_text(&long);
        assert_eq!(summary.chars().count(), 200);
        assert!(summary.ends_with("..."));
        assert_eq!(details.as_deref(), Some(long.as_str()));
        assert!(Feedback::new("other", "user", &summary, details.as_deref()).is_ok());
    }

    #[test]
    fn validation_refuses_bad_input_before_sending() {
        assert!(Feedback::new("issue", "user", "  ", None).is_err());
        assert!(Feedback::new("nonsense", "user", "x", None).is_err());
        assert!(Feedback::new("issue", "robot", "x", None).is_err());
        assert!(Feedback::new("issue", "user", &"x".repeat(201), None).is_err());
        assert!(Feedback::new("issue", "user", "x", Some(&"y".repeat(8001))).is_err());
        let ok = Feedback::new("idea", "agent", " export to ONNX ", Some("  ")).unwrap();
        assert_eq!(ok.kind, "feature_request");
        assert_eq!(ok.summary, "export to ONNX");
        assert_eq!(ok.details, None);
    }

    #[test]
    fn body_carries_only_what_the_caller_wrote_plus_the_client() {
        let mut fb = Feedback::new("issue", "user", "upload stalls", Some("at 90%")).unwrap();
        fb.tools_used = vec!["run".into()];
        let body = fb.body();
        assert_eq!(body["kind"], "issue");
        assert_eq!(body["source"], "user");
        assert_eq!(body["summary"], "upload stalls");
        assert_eq!(body["details"], "at 90%");
        assert_eq!(body["surface_hint"], "cli");
        assert_eq!(body["tools_used"], json!(["run"]));
        assert!(body["client"].as_str().unwrap().starts_with("jopedime "));
        let keys: Vec<_> = body.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys.len(), 7, "nothing else rides along: {keys:?}");
    }
}
