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

/// A model name a person can read, as the run page says it (ai-veox-app `readableModelName`,
/// `veox_app_contracts::method::readable_name`): an egress-redacted identifier is a veox
/// proprietary blend or model, and the common ensembles have their names.
pub fn readable_model(id: &str) -> String {
    let clean = crate::terminal::sanitize_user_string(id);
    if clean.contains("[redacted]") {
        return if clean.to_ascii_lowercase().contains("blend") {
            "veox proprietary blend".into()
        } else {
            "veox proprietary model".into()
        };
    }
    match clean.as_str() {
        "bma" => "Bayesian model averaging".into(),
        "bma_blend" => "Blend of CatBoost and Bayesian model averaging".into(),
        "catboost" => "CatBoost".into(),
        name if name.to_ascii_lowercase().contains("blend") => "Blend".into(),
        _ => clean,
    }
}

/// The run's lead result exactly as the workspace leads with it: the app's own `headline`
/// (`veox_app_contracts::headline`). The score is on the rows it names (for a synthetic-first
/// run, the held-back real rows); a baseline is named only when the app scored one on those same
/// rows, and otherwise the app's own note says there is none (journey J6). The search's
/// synthetic score is secondary and said as such. Nothing from different row sets is compared.
pub fn headline(view: &Value) -> Vec<String> {
    let clean = crate::terminal::sanitize_user_string;
    let headline = &view["headline"];
    let score = &headline["score"];
    let (Some(value), Some(label)) = (
        score["value"].as_f64(),
        score["label"].as_str().or(score["metric"].as_str()),
    ) else {
        return Vec::new();
    };
    let unit = score["unit"]
        .as_str()
        .filter(|u| !u.is_empty())
        .map(|u| format!(" {}", clean(u)))
        .unwrap_or_default();
    let better = match score["direction"].as_str() {
        Some("minimize") => " (lower is better)",
        Some("maximize") => " (higher is better)",
        _ => "",
    };
    let rows = headline["rows"]
        .as_u64()
        .map(|n| format!("{} ", super::route::grouped(n)))
        .unwrap_or_default();
    let on = match headline["basis"].as_str() {
        Some("real_holdout") => format!("on {rows}held-back real rows"),
        Some("synthetic_rows") => {
            "on synthetic training-like rows only; no score on your real rows was recorded".into()
        }
        _ => format!("on {rows}held-out evaluation rows"),
    };
    let mut lines = vec![format!(
        "Result: {} {}{unit}{better} {on}.",
        clean(label),
        figure(value)
    )];
    let baselines = headline["baselines"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for baseline in &baselines {
        let Some(b) = baseline["score"]["value"].as_f64() else {
            continue;
        };
        let lift = baseline["lift"].as_f64().unwrap_or(0.0).abs();
        let verdict = match baseline["outcome"].as_str() {
            Some("veox") => format!("VEOX ahead by {}", figure(lift)),
            Some("baseline") => format!("the baseline ahead by {}", figure(lift)),
            _ => "the same score".to_owned(),
        };
        lines.push(format!(
            "Baseline on the same rows ({}): {} {}{unit} · {verdict}.",
            clean(baseline["description"].as_str().unwrap_or("a plain model")),
            clean(label),
            figure(b)
        ));
    }
    if baselines.is_empty() && headline["basis"] != "synthetic_rows" {
        lines.push("No baseline on these rows.".into());
    }
    if let Some(synthetic) = headline["synthetic"]["value"].as_f64() {
        lines.push(format!(
            "Secondary: on synthetic training-like rows the search scored {} {}{unit} (not your data).",
            clean(label),
            figure(synthetic)
        ));
    }
    lines.extend(
        headline["notes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(clean),
    );
    for warning in headline["data_warnings"].as_array().into_iter().flatten() {
        if let Some(message) = warning["message"].as_str() {
            lines.push(format!("Check the data: {}", clean(message)));
        }
    }
    lines
}

/// The served model by its readable name: the app's `method.name`, else [`readable_model`].
fn model_line(view: &Value) -> Option<String> {
    let name = view["method"]["name"]
        .as_str()
        .filter(|n| !n.is_empty() && !n.contains("[redacted]"))
        .map(crate::terminal::sanitize_user_string)
        .or_else(|| {
            view["servedModel"]["name"]
                .as_str()
                .or(view["summary"]["final_model"].as_str())
                .map(readable_model)
        })?;
    Some(format!("Model: {name}."))
}

/// A run sentence worth repeating under the app's headline: how the table was read and trained.
/// Score and model sentences are left out (the headline and [`model_line`] say them, without
/// mixing row sets or internal names).
fn context_sentence(line: &str) -> bool {
    line.starts_with("Profiled") || line.starts_with("Selected '") || line.starts_with("Trained on")
}

/// Rewrites a model identifier in a run sentence ("Selected bma_blend as the served model.").
fn readable_sentence(line: &str) -> String {
    match line
        .strip_prefix("Selected ")
        .and_then(|rest| rest.strip_suffix(" as the served model."))
    {
        Some(id) if !id.starts_with('\'') => {
            format!("Selected {} as the served model.", readable_model(id))
        }
        _ => line.to_owned(),
    }
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
            let lead = headline(view);
            let narrative: Vec<String> = match &view["summary"]["narrative"] {
                Value::Array(items) => items.iter().filter_map(Value::as_str).map(clean).collect(),
                Value::String(text) => text.lines().map(clean).collect(),
                _ => Vec::new(),
            };
            if lead.is_empty() {
                // An app without the headline: its own sentences, with readable model names.
                lines.extend(narrative.iter().map(|l| readable_sentence(l)));
            } else {
                lines.extend(lead);
                lines.extend(model_line(view));
                lines.extend(narrative.into_iter().filter(|l| context_sentence(l)));
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
    fn the_result_leads_with_the_real_held_back_rows_like_the_workspace() {
        // The shape of a synthetic-first run on prod (2026-11-06, run 853ca6d9…, journey J6):
        // the search's 3.117 and the baseline arm's 5.378 are on synthetic rows; the real
        // held-back rows scored 5.316, with no baseline on them.
        let view = json!({"state":"completed",
            "headline":{"basis":"real_holdout","rows":60,"baselines":[],
                "score":{"direction":"minimize","label":"RMSE","metric":"rmse","unit":"sales_k","value":5.31614294894832},
                "synthetic":{"direction":"minimize","label":"RMSE","metric":"rmse","unit":"sales_k","value":3.1166468523193056},
                "notes":["The engine's baseline arm scored RMSE 5.378 on synthetic rows. That is a different set of rows from the 60 real rows above, so it is not a comparison with this score."],
                "data_warnings":[]},
            "method":{"id":"bma_blend","name":"Blend of CatBoost and Bayesian model averaging"},
            "summary":{"final_model":"bma_blend","narrative":[
                "Profiled the synthetic copy: 240 rows and 9 columns, from an upload of 300 rows and 9 columns.",
                "Selected 'sales_k' as the target.",
                "On 60 held-back real rows, rmse was 5.3161 sales_k (lower is better).",
                "The search's own holdout, on synthetic rows, scored rmse 3.1166 sales_k.",
                "Selected bma_blend as the served model.",
                "Trained on 240 synthetic rows generated from 240 real rows.",
                "The baseline arm scored 5.3781 sales_k on the same 48 synthetic evaluation rows."]}});
        let lines = result_lines(&view, "https://app.veox.ai/runs/x");
        assert_eq!(
            lines[1],
            "Result: RMSE 5.316 sales_k (lower is better) on 60 held-back real rows."
        );
        assert_eq!(lines[2], "No baseline on these rows.");
        assert!(lines[3].starts_with(
            "Secondary: on synthetic training-like rows the search scored RMSE 3.117 sales_k"
        ));
        assert!(lines[4].contains("not a comparison with this score"));
        assert_eq!(
            lines[5],
            "Model: Blend of CatBoost and Bayesian model averaging."
        );
        let text = lines.join("\n");
        assert!(
            !text.contains("ahead by") && !text.contains("same held-out rows"),
            "{text}"
        );
        assert!(
            !text.contains("bma_blend") && !text.contains("baseline arm scored 5.3781"),
            "{text}"
        );
        assert!(
            text.contains("Selected 'sales_k' as the target.")
                && text.contains("Trained on 240 synthetic rows")
        );
        // A baseline scored on the same real rows is named, with who is ahead.
        let mut same = view.clone();
        same["headline"]["baselines"] = json!([{"kind":"linear","description":"a linear model","rows":60,
            "score":{"label":"RMSE","value":9.5},"lift":4.18,"outcome":"veox"}]);
        let lines = result_lines(&same, "u");
        assert_eq!(
            lines[2],
            "Baseline on the same rows (a linear model): RMSE 9.5 sales_k · VEOX ahead by 4.18."
        );
        // Without the app's headline: its sentences, with readable model names.
        let old = json!({"state":"completed","summary":{"narrative":["Selected bma_blend as the served model.","Selected [redacted]_regressor_blend as the served model."]}});
        let lines = result_lines(&old, "u");
        assert_eq!(
            lines[1],
            "Selected Blend of CatBoost and Bayesian model averaging as the served model."
        );
        assert_eq!(
            lines[2],
            "Selected veox proprietary blend as the served model."
        );
        assert_eq!(
            readable_model("[redacted]_regressor"),
            "veox proprietary model"
        );
        assert_eq!(readable_model("bma"), "Bayesian model averaging");
        assert_eq!(figure(1234.5678), "1235");
        assert_eq!(figure(0.5), "0.5");
    }
}
