use super::*;

pub(crate) fn recovery_component_check(name: &str, value: &Value) -> (bool, String) {
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    match name {
        "queue" => {
            let fail_closed = value
                .get("fail_closed_reason")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let healthy = error.is_none()
                && fail_closed.is_none()
                && value.get("admission_allowed").and_then(Value::as_bool) == Some(true)
                && value.get("degraded").and_then(Value::as_bool) != Some(true)
                && useful_queue_depth(value) > 0;
            (
                healthy,
                if healthy {
                    "fresh useful queue depth is nonzero".to_owned()
                } else {
                    error
                        .or(fail_closed)
                        .unwrap_or("queue admission is closed, degraded, or empty")
                        .to_owned()
                },
            )
        }
        "supply" => {
            let blocked = value
                .get("hold_reason")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let useful_work = value
                .get("useful_work")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let target = value
                .get("useful_work_target")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX);
            let healthy = error.is_none()
                && blocked.is_none()
                && value.get("admission_allowed").and_then(Value::as_bool) == Some(true)
                && value.get("performance_fresh").and_then(Value::as_bool) == Some(true)
                && value.get("target_feasible").and_then(Value::as_bool) == Some(true)
                && value
                    .get("complete_reserve_waves")
                    .and_then(Value::as_u64)
                    .unwrap_or_default()
                    >= 2
                && useful_work >= target;
            (
                healthy,
                if healthy {
                    format!("useful work {useful_work} meets target {target}")
                } else {
                    error
                        .or(blocked)
                        .unwrap_or("supply admission is closed")
                        .to_owned()
                },
            )
        }
        "ingestion" => {
            let errors_empty = value.get("errors").is_none_or(|errors| {
                errors.is_null() || errors.as_array().is_some_and(Vec::is_empty)
            });
            let healthy =
                value.get("failures").and_then(Value::as_u64).unwrap_or(0) == 0 && errors_empty;
            (healthy, "ingestion reconcile failure state".to_owned())
        }
        "freezer" => {
            let healthy = error.is_none()
                && value.get("degraded").and_then(Value::as_bool) != Some(true)
                && value.get("timed_out").and_then(Value::as_bool) != Some(true);
            (healthy, "freezer health state".to_owned())
        }
        "controller" => {
            let detail = value
                .get("detail")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            (
                error.is_none() && detail.is_none(),
                error
                    .or(detail)
                    .unwrap_or("controller state is current")
                    .to_owned(),
            )
        }
        "trainer" => {
            let action = value.get("action").and_then(Value::as_str);
            let healthy = matches!(
                action,
                Some(
                    "active_trainer_preserved"
                        | "trainer_submitted"
                        | "gpu_admission_requested"
                        | "trainer_result_harvested"
                )
            );
            (
                healthy,
                action
                    .unwrap_or("trainer state is not proven active or recoverable")
                    .to_owned(),
            )
        }
        "promotion_executor" => {
            let result_error = value
                .get("result")
                .and_then(|result| result.get("error"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            (
                error.is_none() && result_error.is_none(),
                error
                    .or(result_error)
                    .unwrap_or("promotion executor is current")
                    .to_owned(),
            )
        }
        "repair" => {
            let healthy = error.is_none()
                && value
                    .get("permanent_failures_are_imputed")
                    .and_then(Value::as_bool)
                    == Some(false);
            (
                healthy,
                "repair evidence must remain non-imputed".to_owned(),
            )
        }
        "causal_evidence" => {
            let healthy = error.is_none()
                && value.get("authoritative").and_then(Value::as_bool) == Some(true);
            (healthy, "causal evidence authority state".to_owned())
        }
        "freshness" => {
            let healthy = value.get("healthy").and_then(Value::as_bool) == Some(true)
                && value
                    .get("stale_epoch_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    == 0
                && value.get("admission_paused").and_then(Value::as_bool) != Some(true);
            (healthy, "freshness authority state".to_owned())
        }
        "liveness_watchdog" => {
            let healthy = value
                .get("alarms")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty);
            (healthy, "liveness alarm state".to_owned())
        }
        "effective_unit_audit" => {
            let healthy = value.get("healthy").and_then(Value::as_bool) == Some(true)
                && value
                    .get("violations")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty);
            (healthy, "effective unit inventory state".to_owned())
        }
        _ => (true, "native document is current".to_owned()),
    }
}
