//! Cloud runs from the CLI (ai-veox-app `docs/connected-cli.md`, *Cloud runs from the jAIn
//! CLI*): one `POST /api/sessions` with the user's recorded upload consent, then `GET
//! /api/sessions/:id` every ~2 s, rendering each new `progressEvents[].sequence` once.
use super::route::{Limit, Tier};
use super::*;

/// What the user agreed to upload: the bytes are exactly the file read from disk.
pub struct Upload<'a> {
    pub file_name: &'a str,
    pub bytes: &'a [u8],
    pub rows: u64,
    pub columns: u64,
}

/// Why a cloud run was not created.
#[derive(Debug)]
pub enum Refusal {
    /// A tier limit, machine-readable.
    Limit(Limit),
    /// Any other refusal, as the workspace worded it.
    Other(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Limit(limit) => f.write_str(&limit.sentence()),
            Self::Other(message) => f.write_str(message),
        }
    }
}

/// The `consent` multipart field, exactly as the contract spells it.
pub fn consent_field(rows: u64, columns: u64, bytes: u64) -> String {
    format!(r#"{{"uploads_data":true,"rows":{rows},"columns":{columns},"bytes":{bytes}}}"#)
}

fn quoted(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c == '"' || c == '\\' || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// The multipart body. Text fields first, the file last.
pub fn multipart(boundary: &str, upload: &Upload<'_>, profile: &str, fallback: &str) -> Vec<u8> {
    let mut body = Vec::with_capacity(upload.bytes.len() + 1024);
    let consent = consent_field(upload.rows, upload.columns, upload.bytes.len() as u64);
    for (name, value) in [
        ("consent", consent.as_str()),
        ("cloud_fallback", fallback),
        ("profile", profile),
        ("mode", "detached"),
    ] {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    let media = if upload.file_name.to_ascii_lowercase().ends_with(".tsv") {
        "text/tab-separated-values"
    } else {
        "text/csv"
    };
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: {media}\r\n\r\n",
            quoted(upload.file_name)
        )
        .as_bytes(),
    );
    body.extend_from_slice(upload.bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// A refusal body: the limit shape, `{"error": "<sentence>"}`, or just the status.
pub fn refusal(status: u16, bytes: &[u8]) -> Refusal {
    let value: Value = serde_json::from_slice(bytes).unwrap_or(Value::Null);
    if let Some(limit) = Limit::parse(status, &value) {
        return Refusal::Limit(limit);
    }
    Refusal::Other(match value["error"].as_str() {
        Some(message) => format!(
            "The workspace refused the run (HTTP {status}): {}",
            crate::terminal::sanitize_user_string(message)
        ),
        None => format!("The workspace refused the run (HTTP {status})."),
    })
}

impl Client {
    /// The account's plan, and whether it may start runs.
    pub fn account_tier(&self) -> Result<(Option<Tier>, bool)> {
        let account = self.json("GET", "/api/cli/account", Value::Null)?;
        Ok((
            account["tier"].as_str().and_then(Tier::parse),
            account["mayRun"].as_bool().unwrap_or(false),
        ))
    }

    /// `POST /api/sessions`. The outer error is transport; the inner one is the workspace's no.
    pub fn start_cloud_run(
        &self,
        upload: &Upload<'_>,
        profile: &str,
        fallback: &str,
    ) -> Result<std::result::Result<String, Refusal>> {
        let boundary = format!("jain{}", ulid::Ulid::new());
        let body = multipart(&boundary, upload, profile, fallback);
        // An upload may take longer than an ordinary request: allow a minute per 4 MiB.
        let seconds = 60 + 60 * (upload.bytes.len() as u64 / (4 << 20));
        let (status, bytes) = self.with_timeout(Duration::from_secs(seconds))?.call(
            "POST",
            "/api/sessions",
            body,
            &format!("multipart/form-data; boundary={boundary}"),
            MAX_JSON,
        )?;
        if status != 200 {
            return Ok(Err(refusal(status, &bytes)));
        }
        let value: Value =
            serde_json::from_slice(&bytes).context("workspace returned invalid run JSON")?;
        let id = value["id"]
            .as_str()
            .filter(|id| super::sync::is_uuid(id))
            .context("workspace returned no run identity")?;
        Ok(Ok(id.to_owned()))
    }

    pub fn cloud_session(&self, id: &str) -> Result<Value> {
        anyhow::ensure!(super::sync::is_uuid(id), "invalid cloud run");
        self.json("GET", &format!("/api/sessions/{id}"), Value::Null)
    }

    pub fn set_cloud_target(&self, id: &str, column: &str) -> Result<()> {
        anyhow::ensure!(super::sync::is_uuid(id), "invalid cloud run");
        self.json(
            "POST",
            &format!("/api/sessions/{id}/target"),
            json!({"target": column}),
        )
        .map(|_| ())
    }
}

/// States after which nothing more streams.
pub fn finished(state: &str) -> bool {
    matches!(state, "completed" | "exploring" | "failed" | "cancelled")
}

fn duration(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    if s >= 60 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// The "thinking" card body for one progress event.
pub fn event_text(event: &Value) -> Option<String> {
    let payload = &event["payload"];
    let clean = crate::terminal::sanitize_user_string;
    if payload["type"] == "telemetry" {
        let mut parts = Vec::new();
        let stage = payload["stage"].as_str().unwrap_or("search");
        parts.push(match stage {
            "screening" => "Screening".to_owned(),
            "search" => "Searching".to_owned(),
            other => clean(other),
        });
        if let (Some(step), Some(steps)) = (payload["step"].as_u64(), payload["steps"].as_u64()) {
            parts.push(format!("step {step}/{steps}"));
        } else if let Some(fraction) = payload["fraction"].as_f64() {
            parts.push(format!("{:.0}%", fraction * 100.0));
        }
        if let Some(n) = payload["evaluations"].as_u64() {
            parts.push(format!("{} evaluations", super::route::grouped(n)));
        }
        if let (Some(score), Some(metric)) = (
            payload["bestScore"].as_f64(),
            payload["bestScoreMetric"].as_str(),
        ) {
            parts.push(format!("best {} {score:.4}", clean(metric)));
        }
        if let Some(eta) = payload["etaSeconds"].as_f64() {
            parts.push(format!("about {} left", duration(eta)));
        }
        return Some(parts.join(" · "));
    }
    let message = clean(event["message"].as_str()?.trim());
    (!message.is_empty()).then_some(message)
}

/// Renders each `progressEvents[].sequence` once, however often the run page repeats it.
#[derive(Default)]
pub struct Progress {
    seen: std::collections::BTreeSet<i64>,
}

impl Progress {
    pub fn fresh(&mut self, view: &Value) -> Vec<String> {
        let mut events: Vec<&Value> = view["progressEvents"]
            .as_array()
            .map(|events| events.iter().collect())
            .unwrap_or_default();
        events.sort_by_key(|e| e["sequence"].as_i64().unwrap_or(i64::MAX));
        events
            .into_iter()
            .filter(|e| e["sequence"].as_i64().is_some_and(|n| self.seen.insert(n)))
            .filter_map(event_text)
            .collect()
    }
}

/// A score as the run page says it: four significant figures, no float noise.
fn figure(value: f64) -> String {
    if value == 0.0 || !value.is_finite() {
        return format!("{value}");
    }
    let digits = (4 - 1 - value.abs().log10().floor() as i32).clamp(0, 8) as usize;
    let text = format!("{value:.digits$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

/// The held-out headline of a completed run, labelled as the run page leads with it (journey
/// J3-F03): the metric, which way is better, the unit, and the baseline on the same rows.
pub fn headline(summary: &Value) -> Vec<String> {
    let clean = crate::terminal::sanitize_user_string;
    let metric = &summary["headline_metric"];
    let (Some(name), Some(value)) = (metric["name"].as_str(), metric["value"].as_f64()) else {
        return Vec::new();
    };
    let mut parts = vec![format!("held-out {} {}", clean(name), figure(value))];
    match metric["direction"].as_str() {
        Some("minimize") => parts.push("lower is better".into()),
        Some("maximize") => parts.push("higher is better".into()),
        _ => {}
    }
    if let Some(unit) = metric["unit"].as_str().filter(|u| !u.is_empty()) {
        parts.push(format!("in units of `{}`", clean(unit)));
    }
    let mut lines = vec![format!("Result: {}.", parts.join(" · "))];
    let comparison = &summary["comparison"];
    if let (Some(baseline), Some(lift)) = (
        comparison["baseline_score"].as_f64(),
        comparison["lift"].as_f64(),
    ) {
        let verdict = match comparison["outcome"].as_str() {
            Some("veox") => format!("VEOX ahead by {}", figure(lift.abs())),
            Some("baseline") => format!("the baseline ahead by {}", figure(lift.abs())),
            _ => "the same score".to_owned(),
        };
        lines.push(format!(
            "Baseline (a plain model scored on the same held-out rows): {} {} · {verdict}.",
            clean(comparison["metric"].as_str().unwrap_or(name)),
            figure(baseline)
        ));
    }
    lines
}

/// The closing lines of a finished cloud run: the outcome, the held-out score and baseline,
/// the run's own sentences and its workspace link (clickable where the terminal can).
pub fn result_lines(view: &Value, url: &str) -> Vec<String> {
    let clean = crate::terminal::sanitize_user_string;
    let state = view["state"].as_str().unwrap_or("unknown");
    let mut lines = Vec::new();
    match state {
        "failed" => lines.push(format!(
            "Cloud run failed: {}",
            view["failure"]["message"]
                .as_str()
                .map(clean)
                .unwrap_or_else(|| "the workspace did not say why.".into())
        )),
        "cancelled" => lines.push("Cloud run cancelled.".into()),
        _ => {
            lines.push("✓ Cloud run finished.".into());
            lines.extend(headline(&view["summary"]));
            let narrative = &view["summary"]["narrative"];
            match narrative {
                Value::Array(items) => {
                    lines.extend(items.iter().filter_map(Value::as_str).map(clean))
                }
                Value::String(text) => lines.extend(text.lines().map(clean)),
                _ => {}
            }
        }
    }
    lines.push(format!("Open it in your workspace: {url}"));
    lines
}

/// The `Next:` line after a finished cloud run. `tui` picks slash commands.
pub fn next_line(url: &str, tui: bool) -> String {
    if tui {
        format!(
            "Next: predict new rows in your workspace: {url}/predict (/copy link copies it) · /open FILE.csv trains another table"
        )
    } else {
        format!(
            "Next: predict new rows in your workspace: {url}/predict · `jain run FILE.csv --target COLUMN` trains another table"
        )
    }
}

/// What a followed cloud run reports to whoever is drawing it.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Started { id: String, url: String },
    Thinking(String),
    Finished(Vec<String>),
    Error(String),
}

/// Poll a cloud run until it finishes, the user cancels, or polling keeps failing. A target the
/// user chose locally is applied while the workspace still lets it be changed.
pub fn follow(
    client: &Client,
    id: &str,
    target: Option<&str>,
    interval: Duration,
    cancel: &AtomicBool,
    mut emit: impl FnMut(Update),
) {
    let url = format!("{}/runs/{id}", client.origin);
    let mut progress = Progress::default();
    let mut target_sent = false;
    let mut failures = 0u32;
    loop {
        if cancel.load(Ordering::SeqCst) {
            emit(Update::Error(format!(
                "Stopped following the cloud run; it continues in your workspace: {url}"
            )));
            return;
        }
        match client.cloud_session(id) {
            Ok(view) => {
                failures = 0;
                if !target_sent && view["target"]["open"] == true {
                    if let Some(column) = target {
                        if view["target"]["column"].as_str() != Some(column) {
                            let _ = client.set_cloud_target(id, column);
                        }
                    }
                    target_sent = true;
                }
                for line in progress.fresh(&view) {
                    emit(Update::Thinking(line));
                }
                if view["state"].as_str().is_some_and(finished) {
                    emit(Update::Finished(result_lines(&view, &url)));
                    return;
                }
            }
            Err(error) => {
                failures += 1;
                if failures >= 5 {
                    emit(Update::Error(format!(
                        "Lost touch with the cloud run ({error:#}); it continues in your workspace: {url}"
                    )));
                    return;
                }
            }
        }
        let mut waited = Duration::ZERO;
        while waited < interval && !cancel.load(Ordering::SeqCst) {
            let step = Duration::from_millis(50).min(interval - waited);
            std::thread::sleep(step);
            waited += step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_renders_each_sequence_once_in_order() {
        let mut progress = Progress::default();
        let first = json!({"progressEvents": [
            {"sequence": 2, "phase": "p", "message": "Profiling the table", "payload": {"type":"phase"}},
            {"sequence": 1, "phase": "p", "message": "Container started", "payload": {"type":"phase"}},
        ]});
        assert_eq!(
            progress.fresh(&first),
            vec!["Container started", "Profiling the table"]
        );
        assert!(
            progress.fresh(&first).is_empty(),
            "a repeat renders nothing"
        );
        let second = json!({"progressEvents": [
            {"sequence": 1, "message": "Container started", "payload": {"type":"phase"}},
            {"sequence": 2, "message": "Profiling the table", "payload": {"type":"phase"}},
            {"sequence": 3, "message": "", "payload": {"type":"telemetry","stage":"search","step":3,"steps":10,
              "evaluations":1200,"bestScore":0.12345,"bestScoreMetric":"rmse","etaSeconds":95}},
            {"message": "no sequence is never rendered"},
        ]});
        assert_eq!(
            progress.fresh(&second),
            vec![
                "Searching · step 3/10 · 1,200 evaluations · best rmse 0.1235 · about 1m 35s left"
            ]
        );
        assert!(progress.fresh(&second).is_empty());
    }

    #[test]
    fn the_upload_is_the_documented_multipart_form() {
        let upload = Upload {
            file_name: "my \"data\".csv",
            bytes: b"a,b\n1,2\n",
            rows: 1,
            columns: 2,
        };
        let body = String::from_utf8(multipart("B", &upload, "deep_high", "no_gpu")).unwrap();
        assert!(body.contains(
            "name=\"consent\"\r\n\r\n{\"uploads_data\":true,\"rows\":1,\"columns\":2,\"bytes\":8}\r\n"
        ), "{body}");
        assert!(body.contains("name=\"cloud_fallback\"\r\n\r\nno_gpu\r\n"));
        assert!(body.contains("name=\"profile\"\r\n\r\ndeep_high\r\n"));
        assert!(body.contains("name=\"mode\"\r\n\r\ndetached\r\n"));
        assert!(body.contains(
            "filename=\"my _data_.csv\"\r\nContent-Type: text/csv\r\n\r\na,b\n1,2\n\r\n--B--\r\n"
        ));
    }

    #[test]
    fn refusals_keep_the_limit_shape_and_the_sentence() {
        match refusal(429, br#"{"error":"limit","limit":"daily_runs","used":10,"allowed":10,"resets_at":null,"upgrade_url":"https://veox.ai/pricing","message":"x"}"#) {
            Refusal::Limit(limit) => assert_eq!(limit.limit, "daily_runs"),
            other => panic!("{other:?}"),
        }
        match refusal(
            400,
            br#"{"error":"the upload consent says uploads_data is false"}"#,
        ) {
            Refusal::Other(message) => assert!(message.contains("uploads_data is false")),
            other => panic!("{other:?}"),
        }
        assert!(matches!(refusal(502, b"<html>"), Refusal::Other(m) if m.contains("502")));
    }

    #[test]
    fn a_finished_run_shows_its_narrative_and_link() {
        let view = json!({"state":"completed","summary":{"narrative":["Selected 'y' as the target.","CV rmse 0.41"]}});
        assert_eq!(
            result_lines(&view, "https://app.veox.ai/runs/x"),
            vec![
                "✓ Cloud run finished.",
                "Selected 'y' as the target.",
                "CV rmse 0.41",
                "Open it in your workspace: https://app.veox.ai/runs/x"
            ]
        );
        assert!(next_line("https://app.veox.ai/runs/x", true)
            .starts_with("Next: predict new rows in your workspace: https://app.veox.ai/runs/x/predict (/copy link copies it)"));
        assert!(next_line("u", false).contains("`jain run FILE.csv --target COLUMN`"));
        let failed = json!({"state":"failed","failure":{"message":"The table had no rows."}});
        assert_eq!(
            result_lines(&failed, "u")[0],
            "Cloud run failed: The table had no rows."
        );
        assert!(finished("exploring") && !finished("running"));
    }

    #[test]
    fn the_headline_leads_with_the_held_out_score_its_unit_and_the_baseline() {
        let view = json!({"state":"completed","summary":{
            "headline_metric":{"name":"rmse","value":0.027712345,"direction":"minimize","unit":"price"},
            "comparison":{"metric":"rmse","veox_score":0.0277,"baseline_score":0.2104,"lift":0.1827,"outcome":"veox"},
            "narrative":["Selected 'price' as the target."]}});
        let lines = result_lines(&view, "https://app.veox.ai/runs/x");
        assert_eq!(lines[0], "✓ Cloud run finished.");
        assert_eq!(
            lines[1],
            "Result: held-out rmse 0.02771 · lower is better · in units of `price`."
        );
        assert_eq!(
            lines[2],
            "Baseline (a plain model scored on the same held-out rows): rmse 0.2104 · VEOX ahead by 0.1827."
        );
        assert_eq!(lines[3], "Selected 'price' as the target.");
        // No headline metric: only the run's own sentences.
        assert!(headline(&json!({"narrative":[]})).is_empty());
        assert_eq!(figure(1234.5678), "1235");
        assert_eq!(figure(0.5), "0.5");
    }
}
