use super::*;

pub(super) fn load_state(
    path: &Path,
    effective_cores: i64,
    baseline: PressureBaseline,
) -> Result<RampState> {
    if path.is_file() {
        // Version gate FIRST, on a Value, before the strict struct sees the
        // bytes — the pattern at jope_selfplay_supply.rs:423-430. The inverted
        // form (deserialize, then check `schema`) is unreachable for exactly
        // the documents it is meant to reject.
        let value = read_json(path)?;
        let schema = value.get("schema").and_then(Value::as_str);
        if let Some(schema) = schema {
            ensure!(
                schema == CAPACITY_STATE_SCHEMA,
                "capacity ramp state {} carries schema {schema:?}, expected {CAPACITY_STATE_SCHEMA:?}",
                path.display(),
            );
        }
        let stored: StoredRampState = serde_json::from_value(value)?;
        let state = stored.core()?;
        if state.effective_cores != effective_cores {
            bail!("xbabe4 effective cores changed; re-baseline explicitly");
        }
        return Ok(state);
    }
    RampState::baseline(effective_cores, baseline).map_err(Into::into)
}

pub(super) fn stage_name(stage: RampStage) -> &'static str {
    match stage {
        RampStage::Baseline => "baseline",
        RampStage::Ramp30 => "ramp_30",
        RampStage::Target40 => "target_40",
        RampStage::RolledBack => "rolled_back",
    }
}

pub(super) fn parse_stage(value: &str) -> Result<RampStage> {
    match value {
        "baseline" => Ok(RampStage::Baseline),
        "ramp_30" => Ok(RampStage::Ramp30),
        "target_40" => Ok(RampStage::Target40),
        "rolled_back" => Ok(RampStage::RolledBack),
        _ => bail!("unknown xbabe4 ramp stage {value}"),
    }
}

pub(super) fn fresh_queue_depth(path: &Path) -> Result<i64> {
    if !path.is_file() {
        return Ok(0);
    }
    let value = read_json(path)?;
    if value.get("admission_allowed").and_then(Value::as_bool) != Some(true) {
        return Ok(0);
    }
    Ok(value
        .get("queued_starting")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0))
}
