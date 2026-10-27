use super::{is_hex64, Label, LabelEligibility, Observation, RuntimeLabel, TimeoutCause};
use crate::wire_contract::{ExactF64, MissingReason};
use anyhow::{ensure, Result};

fn available<T>(value: T, unit: &str, denominator: u64) -> Label<T> {
    Label::Available {
        value,
        unit: unit.into(),
        denominator,
    }
}
pub(super) fn serialize_missing_reason<S: serde::Serializer>(
    reason: &MissingReason,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(reason.as_str())
}
fn missing<T>(reason: MissingReason, detail: &'static str) -> Label<T> {
    Label::Missing { reason, detail }
}

pub(super) fn labels(observation: &Observation) -> Result<LabelEligibility> {
    let mut out = LabelEligibility {
        quality: missing(MissingReason::Unscheduled, "not_measured"),
        runtime: missing(MissingReason::Unscheduled, "not_measured"),
        failure: missing(MissingReason::Unscheduled, "not_measured"),
        policy: missing(MissingReason::NotApplicable, "no_measured_policy_target"),
    };
    match observation {
        Observation::Proposed => {}
        Observation::Completed {
            quality,
            elapsed_ms,
            policy_target_sha256,
        } => {
            out.quality = missing(MissingReason::FailedClosed, "quality_unavailable");
            out.runtime = missing(MissingReason::FailedClosed, "duration_unavailable");
            if let Some(q) = quality {
                ExactF64::from_hex("/quality/f64_bits", &q.value.f64_bits)?;
                ensure!(
                    !q.metric.trim().is_empty() && !q.unit.trim().is_empty() && q.denominator > 0,
                    "HELD: invalid quality unit/denominator"
                );
                out.quality = available(q.clone(), &q.unit, q.denominator);
            }
            if let Some(ms) = elapsed_ms {
                out.runtime = available(RuntimeLabel::Exact(*ms), "ms", 1);
            }
            out.failure = available(false, "attempt", 1);
            if let Some(target) = policy_target_sha256 {
                ensure!(is_hex64(target), "HELD: malformed measured policy target");
                out.policy = available(target.clone(), "target", 1);
            }
        }
        Observation::Timeout {
            lower_bound_ms,
            cause,
        } => {
            out.quality = missing(MissingReason::BudgetExhausted, "timeout_has_no_quality");
            out.policy = missing(
                MissingReason::BudgetExhausted,
                "timeout_has_no_policy_target",
            );
            out.runtime = missing(MissingReason::FailedClosed, "timeout_bound_unavailable");
            if let Some(ms) = lower_bound_ms {
                ensure!(*ms > 0, "HELD: invalid timeout bound");
                out.runtime = available(RuntimeLabel::RightCensored(*ms), "ms", 1);
            }
            if matches!(cause, TimeoutCause::ProgramDeadline) {
                out.failure = available(true, "attempt", 1);
            } else {
                out.failure = missing(
                    MissingReason::FailedClosed,
                    "no_program_failure_observation",
                );
            }
        }
        Observation::Failed { elapsed_ms } => {
            out.quality = missing(MissingReason::FailedClosed, "failed_attempt_has_no_quality");
            out.runtime = missing(MissingReason::FailedClosed, "duration_unavailable");
            if let Some(ms) = elapsed_ms {
                out.runtime = available(RuntimeLabel::Exact(*ms), "ms", 1);
            }
            out.failure = available(true, "attempt", 1);
        }
    }
    Ok(out)
}
