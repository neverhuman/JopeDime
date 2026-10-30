//! Spec §9.3 catalog and per-warning hysteresis primitives.
//!
//! Callers supply thresholds, cooldown, monotonic elapsed time, context-specific
//! messages/remedy commands and the condition samples. This module emits API
//! warnings; it does not sample hardware, pause workers or publish events.

use jope_api::{Severity, Warning, WarningState};
use std::fmt;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarningDefinition {
    pub code: &'static str,
    pub severity: Severity,
    pub host: bool,
    /// Escalation is explicit; the resource governor owns its triggering policy.
    pub escalated_severity: Option<Severity>,
}

const fn definition(code: &'static str, severity: Severity, host: bool) -> WarningDefinition {
    WarningDefinition {
        code,
        severity,
        host,
        escalated_severity: None,
    }
}

pub const CATALOG: &[WarningDefinition] = &[
    definition("GPU_NOT_DETECTED", Severity::Info, true),
    definition("BACKEND_UNAVAILABLE", Severity::Error, true),
    definition("GPU_SENSOR_UNAVAILABLE", Severity::Info, true),
    definition("MEMORY_PRESSURE", Severity::Warn, true),
    WarningDefinition {
        escalated_severity: Some(Severity::Error),
        ..definition("DISK_LOW", Severity::Warn, true)
    },
    definition("DATASET_LARGE", Severity::Info, false),
    definition("WORKER_LIMIT", Severity::Info, true),
    definition("LANGUAGE_CONSTRAINT", Severity::Info, false),
    definition("GUIDANCE_HEURISTIC", Severity::Info, false),
    definition("OBJECTIVE_SURROGATE_MISMATCH", Severity::Info, false),
    definition("MODEL_INTEGRITY_FAILED", Severity::Error, false),
    definition("NO_VALID_MODEL", Severity::Error, false),
    definition("EVENT_LAG", Severity::Warn, true),
];

pub fn catalog(code: &str) -> Option<&'static WarningDefinition> {
    CATALOG.iter().find(|entry| entry.code == code)
}

impl WarningDefinition {
    pub fn event(
        &self,
        state: WarningState,
        value: Option<f64>,
        limit: Option<f64>,
        message: impl Into<String>,
        remedy: impl Into<String>,
    ) -> Result<Warning, WarningInputError> {
        if value.is_some_and(|v| !v.is_finite()) || limit.is_some_and(|v| !v.is_finite()) {
            return Err(WarningInputError::NonFiniteValue);
        }
        Ok(Warning {
            code: self.code.to_owned(),
            severity: self.severity,
            state,
            host: self.host,
            value,
            limit,
            message: message.into(),
            remedy: remedy.into(),
        })
    }

    pub fn escalated_event(
        &self,
        state: WarningState,
        value: Option<f64>,
        limit: Option<f64>,
        message: impl Into<String>,
        remedy: impl Into<String>,
    ) -> Result<Warning, WarningInputError> {
        let severity = self
            .escalated_severity
            .ok_or(WarningInputError::NotEscalatable)?;
        let mut event = self.event(state, value, limit, message, remedy)?;
        event.severity = severity;
        Ok(event)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdDirection {
    Below,
    Above,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningInputError {
    InvalidThresholds,
    NonFiniteValue,
    ClockWentBackwards,
    NotEscalatable,
}

impl fmt::Display for WarningInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidThresholds => {
                "raise_at and clear_at must be finite and ordered for the direction"
            }
            Self::NonFiniteValue => "warning value and limit must be finite or missing",
            Self::ClockWentBackwards => "warning observation time went backwards",
            Self::NotEscalatable => "warning code has no escalated severity",
        })
    }
}

impl std::error::Error for WarningInputError {}

/// Separate thresholds and a minimum interval between transitions prevent flapping.
/// Initial raise is immediate; both raise/clear boundaries are strict. Observations
/// inside the band, at either boundary, or during cooldown preserve current state.
#[derive(Clone, Debug)]
pub struct Hysteresis {
    direction: ThresholdDirection,
    raise_at: f64,
    clear_at: f64,
    cooldown: Duration,
    state: WarningState,
    last_event_at: Option<Duration>,
    last_observed_at: Option<Duration>,
}

impl Hysteresis {
    pub fn new(
        direction: ThresholdDirection,
        raise_at: f64,
        clear_at: f64,
        cooldown: Duration,
    ) -> Result<Self, WarningInputError> {
        let ordered = match direction {
            ThresholdDirection::Below => raise_at < clear_at,
            ThresholdDirection::Above => raise_at > clear_at,
        };
        if !raise_at.is_finite() || !clear_at.is_finite() || !ordered {
            return Err(WarningInputError::InvalidThresholds);
        }
        Ok(Self {
            direction,
            raise_at,
            clear_at,
            cooldown,
            state: WarningState::Cleared,
            last_event_at: None,
            last_observed_at: None,
        })
    }

    pub fn state(&self) -> WarningState {
        self.state
    }

    pub fn limit(&self, state: WarningState) -> f64 {
        match state {
            WarningState::Raised => self.raise_at,
            WarningState::Cleared => self.clear_at,
        }
    }

    /// A missing reading holds state; it never becomes a zero/recovery sample.
    /// Invalid readings or backwards time return an error without mutating state.
    pub fn observe(
        &mut self,
        now: Duration,
        value: Option<f64>,
    ) -> Result<Option<WarningState>, WarningInputError> {
        if self.last_observed_at.is_some_and(|previous| now < previous) {
            return Err(WarningInputError::ClockWentBackwards);
        }
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(WarningInputError::NonFiniteValue);
        }
        self.last_observed_at = Some(now);
        let Some(value) = value else { return Ok(None) };
        let next = match (self.direction, self.state) {
            (ThresholdDirection::Below, WarningState::Cleared) if value < self.raise_at => {
                WarningState::Raised
            }
            (ThresholdDirection::Below, WarningState::Raised) if value > self.clear_at => {
                WarningState::Cleared
            }
            (ThresholdDirection::Above, WarningState::Cleared) if value > self.raise_at => {
                WarningState::Raised
            }
            (ThresholdDirection::Above, WarningState::Raised) if value < self.clear_at => {
                WarningState::Cleared
            }
            _ => return Ok(None),
        };
        if self
            .last_event_at
            .is_some_and(|previous| now.saturating_sub(previous) < self.cooldown)
        {
            return Ok(None);
        }
        self.state = next;
        self.last_event_at = Some(now);
        Ok(Some(next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn lifecycle(code: &str, severity: Severity, host: bool) {
        let entry = catalog(code).unwrap();
        assert!(jope_api::WARNING_CODES.contains(&code));
        assert_eq!((entry.severity, entry.host), (severity, host));
        // A normalized condition flag exercises the common monitor for every code.
        let mut guard = Hysteresis::new(
            ThresholdDirection::Above,
            0.75,
            0.25,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(guard.observe(Duration::ZERO, Some(0.75)), Ok(None));
        for (seconds, value, expected) in [
            (1, 1.0, Some(WarningState::Raised)),
            (2, 0.0, None), // Recovery before the cooldown must not clear.
            (6, 0.5, None), // Band and boundary must retain active state.
            (6, 0.25, None),
            (6, 0.0, Some(WarningState::Cleared)),
            (7, 1.0, None), // Raise is also suppressed during cooldown.
            (11, 1.0, Some(WarningState::Raised)),
        ] {
            let actual = guard
                .observe(Duration::from_secs(seconds), Some(value))
                .unwrap();
            assert_eq!(actual, expected, "{code}, t={seconds}, value={value}");
            if let Some(state) = actual {
                let event = entry
                    .event(
                        state,
                        Some(value),
                        Some(guard.limit(state)),
                        "condition changed",
                        "jopedime doctor",
                    )
                    .unwrap();
                assert_eq!(
                    (event.code.as_str(), event.severity, event.host, event.state),
                    (code, severity, host, state)
                );
                assert_eq!(event.value, Some(value));
                assert_eq!(event.limit, Some(guard.limit(state)));
            }
        }
        assert_eq!(guard.state(), WarningState::Raised);
    }

    macro_rules! code_test {
        ($name:ident, $code:literal, $severity:ident, $host:literal) => {
            #[test]
            fn $name() {
                lifecycle($code, Severity::$severity, $host);
            }
        };
    }
    code_test!(gpu_not_detected, "GPU_NOT_DETECTED", Info, true);
    code_test!(backend_unavailable, "BACKEND_UNAVAILABLE", Error, true);
    code_test!(gpu_sensor_unavailable, "GPU_SENSOR_UNAVAILABLE", Info, true);
    code_test!(memory_pressure, "MEMORY_PRESSURE", Warn, true);
    code_test!(disk_low, "DISK_LOW", Warn, true);
    code_test!(dataset_large, "DATASET_LARGE", Info, false);
    code_test!(worker_limit, "WORKER_LIMIT", Info, true);
    code_test!(language_constraint, "LANGUAGE_CONSTRAINT", Info, false);
    code_test!(guidance_heuristic, "GUIDANCE_HEURISTIC", Info, false);
    code_test!(
        objective_surrogate_mismatch,
        "OBJECTIVE_SURROGATE_MISMATCH",
        Info,
        false
    );
    code_test!(
        model_integrity_failed,
        "MODEL_INTEGRITY_FAILED",
        Error,
        false
    );
    code_test!(no_valid_model, "NO_VALID_MODEL", Error, false);
    code_test!(event_lag, "EVENT_LAG", Warn, true);

    #[test]
    fn catalog_matches_the_exact_api_vocabulary() {
        let actual: BTreeSet<_> = CATALOG.iter().map(|entry| entry.code).collect();
        let api: BTreeSet<_> = jope_api::WARNING_CODES.iter().copied().collect();
        assert_eq!(actual, api);
        assert_eq!(actual.len(), 13);
        assert_eq!(CATALOG.len(), actual.len());
        assert_eq!(jope_api::WARNING_CODES.len(), api.len());
        assert!(catalog("UNKNOWN_CODE").is_none());
    }

    #[test]
    fn memory_pressure_uses_strict_ten_twenty_percent_boundaries() {
        let mut guard = Hysteresis::new(
            ThresholdDirection::Below,
            0.10,
            0.20,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(guard.observe(Duration::ZERO, Some(0.10)), Ok(None));
        assert_eq!(
            guard.observe(Duration::from_secs(1), Some(0.09)),
            Ok(Some(WarningState::Raised))
        );
        assert_eq!(guard.observe(Duration::from_secs(6), Some(0.20)), Ok(None));
        assert_eq!(
            guard.observe(Duration::from_secs(6), Some(0.21)),
            Ok(Some(WarningState::Cleared))
        );
    }

    #[test]
    fn disk_low_escalation_is_explicit_and_other_codes_refuse_it() {
        let disk = catalog("DISK_LOW").unwrap();
        assert_eq!(
            disk.event(
                WarningState::Raised,
                None,
                None,
                "disk reserve breached",
                "jopedime gc"
            )
            .unwrap()
            .severity,
            Severity::Warn
        );
        assert_eq!(
            disk.escalated_event(
                WarningState::Raised,
                None,
                None,
                "disk critically low",
                "jopedime gc"
            )
            .unwrap()
            .severity,
            Severity::Error
        );
        for entry in CATALOG.iter().filter(|entry| entry.code != "DISK_LOW") {
            assert_eq!(
                entry.escalated_event(WarningState::Raised, None, None, "test", "jopedime doctor"),
                Err(WarningInputError::NotEscalatable)
            );
        }
    }

    #[test]
    fn missing_invalid_and_backwards_observations_preserve_state() {
        let mut guard = Hysteresis::new(
            ThresholdDirection::Below,
            10.0,
            20.0,
            Duration::from_secs(5),
        )
        .unwrap();
        assert_eq!(
            guard.observe(Duration::from_secs(10), Some(9.0)),
            Ok(Some(WarningState::Raised))
        );
        assert_eq!(guard.observe(Duration::from_secs(20), None), Ok(None));
        assert_eq!(guard.state(), WarningState::Raised);
        assert_eq!(
            guard.observe(Duration::from_secs(19), Some(30.0)),
            Err(WarningInputError::ClockWentBackwards)
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                guard.observe(Duration::from_secs(21), Some(value)),
                Err(WarningInputError::NonFiniteValue)
            );
        }
        assert_eq!(guard.state(), WarningState::Raised);
        assert_eq!(
            guard.observe(Duration::from_secs(20), Some(30.0)),
            Ok(Some(WarningState::Cleared))
        );
    }

    #[test]
    fn invalid_thresholds_and_nonfinite_wire_values_are_refused() {
        for (direction, raise, clear) in [
            (ThresholdDirection::Below, 20.0, 10.0),
            (ThresholdDirection::Above, 10.0, 20.0),
            (ThresholdDirection::Below, 10.0, 10.0),
            (ThresholdDirection::Above, f64::NAN, 0.0),
            (ThresholdDirection::Below, 0.0, f64::INFINITY),
        ] {
            assert!(matches!(
                Hysteresis::new(direction, raise, clear, Duration::ZERO),
                Err(WarningInputError::InvalidThresholds)
            ));
        }
        let entry = catalog("MEMORY_PRESSURE").unwrap();
        assert_eq!(
            entry.event(WarningState::Raised, Some(f64::NAN), None, "test", "test"),
            Err(WarningInputError::NonFiniteValue)
        );
        assert_eq!(
            entry.event(
                WarningState::Raised,
                None,
                Some(f64::INFINITY),
                "test",
                "test"
            ),
            Err(WarningInputError::NonFiniteValue)
        );
    }
}
