use super::*;

// ---------------------------------------------------------------------------
// VERDICT
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SupervisorVerdict {
    /// Busy and every producer is producing.
    Ok,
    /// Not busy, and there is genuinely nothing left to run. Not an alarm.
    IdleNoWork,
    /// Busy right now, but a producer is broken: the cluster is coasting on
    /// in-flight work and will fall off a cliff when it drains. This is the
    /// early-warning state that would have caught 2026-07-28 at 10:19Z, when
    /// the queue read 3 queued / 258 running twenty minutes before zero.
    Degraded,
    /// Not busy, and work is sitting on disk. The condition the owner found by
    /// looking at a screen.
    Idle,
}

impl SupervisorVerdict {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            SupervisorVerdict::Ok => "ok",
            SupervisorVerdict::IdleNoWork => "idle_no_work",
            SupervisorVerdict::Degraded => "degraded",
            SupervisorVerdict::Idle => "idle",
        }
    }
}

pub(super) fn judge(
    args: &Args,
    util: &Utilization,
    work: &WorkOnDisk,
    stages: &[Stage],
) -> SupervisorVerdict {
    let busy = if args.simulate_idle {
        false
    } else {
        util.scq_reachable && util.core_pct() >= args.idle_core_pct
    };
    let producers_ok = stages.iter().all(|stage| stage.health.healthy());
    let under_supplied = util.queue_depth() < effective_queue_target(args, util);
    if !busy {
        return if work.total() > 0 {
            SupervisorVerdict::Idle
        } else {
            SupervisorVerdict::IdleNoWork
        };
    }
    if !producers_ok && under_supplied {
        return SupervisorVerdict::Degraded;
    }
    if !producers_ok {
        // Busy AND deeply queued: a broken producer has not bitten yet, but it
        // is still a defect worth counting toward escalation.
        return SupervisorVerdict::Degraded;
    }
    SupervisorVerdict::Ok
}

pub(super) fn effective_queue_target(args: &Args, util: &Utilization) -> u64 {
    let slot_target = util.slots_total.saturating_mul(5).saturating_add(3) / 4;
    args.queue_floor_jobs.max(slot_target).min(4096)
}

// ---------------------------------------------------------------------------
// REPAIR — bounded, receipted, backed off
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub(super) struct State {
    pub(super) consecutive_idle: u32,
    pub(super) consecutive_degraded: u32,
    pub(super) escalated: bool,
    pub(super) escalated_at_ms: u64,
    pub(super) last_notify_ms: u64,
    /// action key -> (consecutive attempts, last attempt ms)
    pub(super) attempts: BTreeMap<String, (u32, u64)>,
    /// epoch-ms of every action taken, trimmed to the last hour.
    pub(super) action_log: Vec<u64>,
}

pub(super) fn load_state(path: &Path) -> State {
    let Some(value) = read_json(path) else {
        return State::default();
    };
    let mut state = State {
        consecutive_idle: u64_at(&value, "consecutive_idle_cycles").unwrap_or(0) as u32,
        consecutive_degraded: u64_at(&value, "consecutive_degraded_cycles").unwrap_or(0) as u32,
        escalated: value
            .get("escalated")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        escalated_at_ms: u64_at(&value, "escalated_at_ms").unwrap_or(0),
        last_notify_ms: u64_at(&value, "last_notify_ms").unwrap_or(0),
        ..State::default()
    };
    if let Some(map) = value.get("attempts").and_then(Value::as_object) {
        for (key, entry) in map {
            state.attempts.insert(
                key.clone(),
                (
                    u64_at(entry, "attempts").unwrap_or(0) as u32,
                    u64_at(entry, "last_ms").unwrap_or(0),
                ),
            );
        }
    }
    if let Some(list) = value.get("action_log").and_then(Value::as_array) {
        state.action_log = list.iter().filter_map(Value::as_u64).collect();
    }
    state
}

pub(super) fn save_state(path: &Path, state: &State) -> Result<()> {
    let attempts: serde_json::Map<String, Value> = state
        .attempts
        .iter()
        .map(|(key, (attempts, last))| {
            (
                key.clone(),
                json!({ "attempts": attempts, "last_ms": last }),
            )
        })
        .collect();
    write_json_atomic(
        path,
        &json!({
            "schema": "JopeUtilizationSupervisorState.v1",
            "consecutive_idle_cycles": state.consecutive_idle,
            "consecutive_degraded_cycles": state.consecutive_degraded,
            "escalated": state.escalated,
            "escalated_at_ms": state.escalated_at_ms,
            "last_notify_ms": state.last_notify_ms,
            "attempts": Value::Object(attempts),
            "action_log": state.action_log,
        }),
    )
}

pub(super) fn backoff_ms(args: &Args, attempts: u32) -> u64 {
    if attempts == 0 {
        return 0;
    }
    let shift = (attempts - 1).min(16);
    let delay = args.backoff_base_seconds.saturating_mul(1_u64 << shift);
    delay.min(args.backoff_max_seconds) * 1000
}

/// Choose at most ONE repair per cycle. One action per cycle is the whole
/// bounding argument: with a 120 s timer the ceiling is 30 actions/hour before
/// backoff, and `--max-actions-per-hour` clamps it to 6 after.
pub(super) fn plan_repair<'a>(
    args: &Args,
    verdict: SupervisorVerdict,
    stages: &'a [Stage],
    state: &State,
    now: u64,
) -> (Option<&'a Stage>, Option<String>) {
    if matches!(
        verdict,
        SupervisorVerdict::Ok | SupervisorVerdict::IdleNoWork
    ) {
        return (None, Some("objective met; no repair considered".into()));
    }
    let recent = state
        .action_log
        .iter()
        .filter(|ms| now.saturating_sub(**ms) < HOUR_MS)
        .count();
    if recent >= args.max_actions_per_hour {
        return (
            None,
            Some(format!(
                "suppressed: {recent} actions in the last hour >= --max-actions-per-hour {}",
                args.max_actions_per_hour
            )),
        );
    }
    let mut deferred: Vec<String> = Vec::new();
    for stage in stages {
        if stage.health.healthy() {
            continue;
        }
        let Some(repair) = stage.repair.as_ref() else {
            deferred.push(format!("{}: no safe automatic repair", stage.name));
            continue;
        };
        // Re-triggering expansion is only worth an action when the cluster is
        // actually idle. While work is flowing, a guardrail block is a policy
        // decision, not an outage.
        if stage.name == "expansion" && verdict != SupervisorVerdict::Idle {
            deferred.push("expansion: blocked but cluster is not idle".into());
            continue;
        }
        let (attempts, last) = state.attempts.get(&repair.key).copied().unwrap_or((0, 0));
        let wait = backoff_ms(args, attempts);
        if last > 0 && now.saturating_sub(last) < wait {
            deferred.push(format!(
                "{}: backoff {}s remaining after {attempts} attempts",
                repair.key,
                (wait - now.saturating_sub(last)) / 1000
            ));
            continue;
        }
        return (
            Some(stage),
            (!deferred.is_empty()).then(|| deferred.join("; ")),
        );
    }
    (
        None,
        Some(if deferred.is_empty() {
            "no unhealthy stage carries a repair".into()
        } else {
            deferred.join("; ")
        }),
    )
}

pub(super) fn run_repair(args: &Args, repair: &Repair) -> (bool, String) {
    if args.dry_run {
        return (
            false,
            format!(
                "dry-run: would systemctl --user {} {}",
                repair.verb, repair.unit
            ),
        );
    }
    let output = Command::new("systemctl")
        .arg("--user")
        .arg(repair.verb)
        .arg(repair.unit)
        .output();
    match output {
        Ok(o) if o.status.success() => (
            true,
            format!("systemctl --user {} {} ok", repair.verb, repair.unit),
        ),
        Ok(o) => (
            false,
            format!(
                "systemctl --user {} {} exit {:?}: {}",
                repair.verb,
                repair.unit,
                o.status.code(),
                String::from_utf8_lossy(&o.stderr).trim()
            ),
        ),
        Err(error) => (false, format!("systemctl spawn failed: {error}")),
    }
}

// ---------------------------------------------------------------------------
// ESCALATE
// ---------------------------------------------------------------------------

/// What is actually available on this box, verified 2026-07-28:
///   - NO mail transport, NO ntfy/pushover/telegram/slack config, and no
///     webhook credential anywhere in the tree.
///   - `notify-send` is installed and DBUS_SESSION_BUS_ADDRESS is set, but no
///     process owns org.freedesktop.Notifications, so desktop toasts are
///     swallowed. Attempted anyway and recorded as delivered=false.
///   - `wall` reaches all logged-in ptys (47 at the time of writing). This is
///     the loudest thing that exists here and it is what the owner will see
///     from any terminal on the box, not just the one showing the TUI.
///   - `logger` -> journald, greppable after the fact.
///   - the TUI's alert panel, fed from ops/utilization-alarms.v1.json.
///
/// Every notification runs under `timeout` so no delivery channel can stall a
/// cycle. A supervisor blocked inside its own alarm is indistinguishable from
/// the daemon it was watching.
pub(super) fn notify(argv: &[&str]) -> bool {
    Command::new("timeout")
        .arg("10")
        .args(argv)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub(super) fn escalate(args: &Args, summary: &str, detail: &str) -> Vec<Value> {
    let mut channels = Vec::new();
    if args.dry_run {
        channels.push(
            json!({"channel": "dry-run", "delivered": false, "note": "no notification sent"}),
        );
        return channels;
    }

    // journald, critical priority.
    let logged = notify(&[
        "logger",
        "-p",
        "user.crit",
        "-t",
        "jope-utilization",
        summary,
    ]);
    channels.push(json!({
        "channel": "journald",
        "delivered": logged,
        "note": "journalctl --user -t jope-utilization -p crit"
    }));

    // `wall` was removed by operator request 2026-07-28: it interrupts EVERY
    // logged-in terminal on the box, including interactive work unrelated to
    // the cluster, and it fires on a 120 s timer so a sustained degradation
    // spams continuously. Escalation now goes to journald (above), the desktop
    // toast (below), the health-alarms file and the TUI banner — all of which
    // are visible without hijacking a shell. Do not reintroduce a broadcast
    // channel here; if escalation needs to be louder, make it push to a device
    // the operator carries, not to every pty.
    let _ = &detail;

    // desktop toast, best effort.
    let notified = notify(&[
        "notify-send",
        "-u",
        "critical",
        "-a",
        "jope-flywheel",
        "JOPE cluster idle",
        summary,
    ]);
    channels.push(json!({
        "channel": "notify-send",
        "delivered": notified,
        "note": "no org.freedesktop.Notifications owner on this box; expected false"
    }));

    // tmux banner into the live TUI session.
    let tmuxed = notify(&[
        "tmux",
        "-L",
        "jope",
        "display-message",
        "-d",
        "0",
        &format!("JOPE UTILIZATION ALARM: {summary}"),
    ]);
    channels.push(json!({"channel": "tmux:jope-tui", "delivered": tmuxed, "note": "sticky status-line banner"}));

    // On-disk banner, for anyone who lands in ops/ after the fact.
    let banner = args.banner();
    let wrote = fs::write(
        &banner,
        format!("{}\n\n{summary}\n\n{detail}\n", iso_utc(now_ms())),
    )
    .is_ok();
    channels.push(
        json!({"channel": "banner_file", "delivered": wrote, "note": banner.display().to_string()}),
    );

    channels
}

pub(super) fn write_alarms(
    args: &Args,
    verdict: SupervisorVerdict,
    escalated: bool,
    summary: &str,
    detail: &str,
) -> Result<()> {
    // DELIBERATELY NOT ops/health-alarms.v1.json. jope_selfplay_expansion_driver.rs
    // :116-141 halts dataset activation on any critical row in that file. A
    // critical `cluster_idle` row there would stop the flywheel activating
    // datasets *because the cluster is idle*, which is precisely the starvation
    // loop that emptied all 1,052 slots on 2026-07-28.
    let alarms = if escalated {
        vec![json!({
            "alarm": "cluster_idle",
            "severity": "critical",
            "observed": summary,
            "threshold": format!("core budget >= {:.0}% with producers healthy", args.idle_core_pct),
            "detail": detail,
            "at_ms": now_ms(),
        })]
    } else if verdict == SupervisorVerdict::Degraded {
        vec![json!({
            "alarm": "cluster_supply_degraded",
            "severity": "warning",
            "observed": summary,
            "threshold": "all producer stages producing",
            "detail": detail,
            "at_ms": now_ms(),
        })]
    } else {
        Vec::new()
    };
    write_json_atomic(
        &args.alarms(),
        &json!({
            "schema": "JopeUtilizationAlarms.v1",
            "generated_at_ms": now_ms(),
            "alarm_count": alarms.len(),
            "alarms": alarms,
        }),
    )
}
