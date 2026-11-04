use super::*;

pub(crate) fn label<'a>(key: &str, value: impl Into<String>) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{key:<23}"), Style::default().fg(MUTED)),
        Span::raw(value.into()),
    ])
}
pub(crate) struct TaskRow {
    pub(crate) name: String,
    pub(crate) status: String,
    pub(crate) progress: String,
    pub(crate) detail: String,
}
impl TaskRow {
    pub(crate) fn table_row(self) -> Row<'static> {
        let color = match self.status.as_str() {
            "DONE" => GREEN,
            "RUNNING" => BLUE,
            "FAILED" => RED,
            "STALE" | "HELD" => GOLD,
            _ => MUTED,
        };
        Row::new(vec![
            Cell::from(self.name).style(Style::default().bold()),
            Cell::from(self.status).style(Style::default().fg(color).bold()),
            Cell::from(self.progress),
            Cell::from(self.detail),
        ])
    }
}
pub(crate) fn task_row(name: &str, status: &str, progress: String, detail: String) -> TaskRow {
    TaskRow {
        name: name.into(),
        status: status.into(),
        progress,
        detail,
    }
}
pub(crate) fn training_hold(run: &Value, policy: &Value) -> Option<String> {
    for value in [&run["continuous"], policy] {
        let state = value["status"].as_str().unwrap_or("").to_ascii_lowercase();
        let phase = phase(value).to_ascii_lowercase();
        if matches!(state.as_str(), "held" | "paused" | "blocked")
            || phase == "held"
            || phase.starts_with("held ")
            || phase == "paused"
            || phase.contains("training held")
            || value["automatic_training"] == false
            || value["automatic_training_hold"] == true
            || value["automatic_training_hold"]["block_new_training"] == true
        {
            return Some(
                value["training_hold_reason"]
                    .as_str()
                    .or_else(|| value["reason"].as_str())
                    .unwrap_or("automatic training is held")
                    .into(),
            );
        }
    }
    None
}
pub(crate) fn receipt_task_state(v: &Value) -> &'static str {
    if v["invalidated"] == true || v["status"] == "failed" {
        "FAILED"
    } else if training_hold(&Value::Null, v).is_some() {
        "HELD"
    } else if completed(v) {
        "DONE"
    } else if number(v, "timestamp") == 0 {
        "QUEUED"
    } else if number(v, "timestamp") > now() || now().saturating_sub(number(v, "timestamp")) > 15 {
        "STALE"
    } else if phase(v).to_ascii_lowercase().contains("waiting") {
        "WAITING"
    } else {
        "RUNNING"
    }
}
pub(crate) fn policy_task_state(run: &Value, policy: &Value) -> (&'static str, String) {
    if let Some(reason) = training_hold(run, policy) {
        let active = run["continuous"]["training_task_reservation"].is_object();
        return (
            "HELD",
            format!(
                "{reason} · policy receipt {}{}",
                status_age(policy),
                if active {
                    " · existing funded task requires separate reconciliation"
                } else {
                    ""
                }
            ),
        );
    }
    (
        receipt_task_state(policy),
        format!(
            "{} trained families · NLL {}",
            number(policy, "training_families"),
            decimal(policy, "sequence_nll")
        ),
    )
}
// Campaign progress is written at phase transitions. It is not a heartbeat.
pub(crate) fn campaign_task_state(run: &Value, neural: &Value) -> (&'static str, String) {
    let campaign = &run["campaign_service"];
    let actual = &run["operations"]["services"]["jope-parallel-campaign"];
    let continuous = &run["continuous"];
    if continuous["phase"].is_string() {
        if actual["ActiveState"] != "active" {
            return (
                "STALE",
                format!(
                    "controller {} · saved {}",
                    actual["ActiveState"],
                    phase(continuous)
                ),
            );
        }
        if now().saturating_sub(number(continuous, "timestamp")) > 15 {
            return ("STALE", "continuous controller heartbeat stale".into());
        }
        return (
            if continuous["phase"] == "paused" {
                "PAUSED"
            } else {
                "RUNNING"
            },
            format!(
                "{} training blocks committed · next {}",
                number(continuous, "completed_training_blocks"),
                continuous["next_scheduled_work"]
            ),
        );
    }
    let deadline = phase(campaign).to_ascii_lowercase().contains("deadline")
        || campaign["status"] == "deadline";
    if deadline {
        if actual["ActiveState"] == "active" {
            return (
                "RECOVERY",
                format!("recorded continuation · {}", phase(&run["cycle"])),
            );
        }
        return (
            "DEADLINE",
            "qualification deadline reached; retraining completion requires a cycle receipt".into(),
        );
    }
    if actual["ActiveState"] == "failed" {
        return ("FAILED", format!("systemd result {}", actual["Result"]));
    }
    if completed(campaign) {
        return ("DONE", phase(campaign).into());
    }
    if campaign["invalidated"] == true || campaign["status"] == "failed" {
        return ("FAILED", phase(campaign).into());
    }
    if !campaign["phase"].is_string() {
        return (
            "QUEUED",
            "Awaiting corpus admission and launch receipt".into(),
        );
    }
    let relative = |path: &str| {
        path.split_once("/runs/")
            .map(|(_, tail)| tail)
            .unwrap_or(path.strip_prefix("runs/").unwrap_or(path))
            .to_owned()
    };
    let current = campaign["current_cycle"].as_str().map(relative);
    let fresh = |v: &Value| {
        number(v, "timestamp") > 0
            && now().saturating_sub(number(v, "timestamp")) <= 15
            && !completed(v)
    };
    let same_cycle = current.as_ref().is_some_and(|current| {
        run["active_cycle"].as_str().map(relative).as_ref() == Some(current)
    });
    let neural_in_cycle = current.as_ref().is_some_and(|current| {
        neural["run"]
            .as_str()
            .is_some_and(|path| Path::new(&relative(path)).starts_with(current))
    });
    let active = (same_cycle && fresh(&run["cycle"])) || (neural_in_cycle && fresh(neural));
    let timestamp = number(campaign, "timestamp");
    let phase_age = if timestamp > 0 {
        format!("phase age {}", duration(now().saturating_sub(timestamp)))
    } else {
        "phase age unavailable".into()
    };
    (
        if active { "RUNNING" } else { "PHASE" },
        format!(
            "{} · {}{}",
            phase(campaign),
            phase_age,
            if active {
                " · current-cycle activity live"
            } else {
                " · transition receipt; inspect worker activity"
            }
        ),
    )
}
pub(crate) fn proposal_library_status(v: &Value) -> &'static str {
    if v["complete"] == true || phase(v).ends_with("comparison complete") {
        return "SAVED";
    }
    match phase(v) {
        "failed" => "FAILED",
        "completed" => "SAVED",
        "stopped" => "STOPPED",
        "paused" => "PAUSED",
        _ if number(v, "timestamp") == 0 => "QUEUED",
        _ if now().saturating_sub(number(v, "timestamp")) > 15 => "STALE",
        p if p.to_ascii_lowercase().contains("waiting") => "WAITING",
        _ => "RUNNING",
    }
}
pub(crate) fn report_status(v: &Value) -> &'static str {
    if completed(v) {
        "DONE"
    } else if v["invalidated"] == true || phase(v) == "Report needs inspection" {
        "FAILED"
    } else if number(v, "timestamp") == 0 {
        "QUEUED"
    } else if now().saturating_sub(number(v, "timestamp")) > 45 {
        "STALE"
    } else if phase(v).starts_with("Waiting") {
        "WAITING"
    } else {
        "RUNNING"
    }
}
