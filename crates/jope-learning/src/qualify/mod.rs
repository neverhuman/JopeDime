//! Phase 5 qualification preparation: the sealed-family roster (G23, J17) and the champion
//! promotion gate (G36, J18). Expert review CLOVER §8.1 and §13.6 (Gate F).
//!
//! Two owner rulings are pending, so this module prepares both decisions and performs neither.
//!
//! | Request | Gap | What it checks | What it can return |
//! |---|---|---|---|
//! | [`request_seal`] | G23 / J17 | [`roster::check`]: the four §8.1 roles, family disjointness, the closure proof, and the decided alpha budget | a [`Refusal`] |
//! | [`request_promotion`] | G36 / J18 | [`promotion::decide`]: a predeclared confirmation-family arena, the selected program's outer result, the look schedule, holds, failure rate and the L6 safety panel | a [`Refusal`] |
//!
//! Each request needs two keys, and this build supplies neither:
//!
//! 1. [`gate::FLAG`] (`JOPE_SEAL_GATE_ENABLED`) set to exactly `1`. It is unset by default. While
//!    it is not `1`, a request refuses before it reads its input.
//! 2. The owner ruling for its gap. Neither ruling is in force, so a request that clears the gate
//!    and every check still refuses with [`Refusal::RulingPending`]. The success type of both
//!    requests is [`Infallible`]: no seal and no promotion can be produced by this build.
//!
//! [`roster::check`] and [`promotion::decide`] are read-only evaluators and run with the gate off.
//! They write nothing, spend no seal and promote nothing; a verdict is a report, not an action.

pub mod gate;
pub mod promotion;
pub mod roster;

use std::convert::Infallible;
use std::fmt;

use serde::Serialize;

pub use gate::Gate;

/// A request this module refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Adopt a sealed-family roster and create its seal.
    Seal,
    /// Promote a challenger to champion.
    Promote,
}

impl Action {
    /// The gap whose owner ruling the action waits for.
    #[must_use]
    pub fn gap(self) -> &'static str {
        match self {
            Self::Seal => "G23",
            Self::Promote => "G36",
        }
    }

    /// The scorecard item the gap closes.
    #[must_use]
    pub fn item(self) -> &'static str {
        match self {
            Self::Seal => "J17",
            Self::Promote => "J18",
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Seal => "seal",
            Self::Promote => "promote",
        })
    }
}

/// Why a request was refused. Every request ends in one of these.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(tag = "refusal", rename_all = "snake_case")]
pub enum Refusal {
    /// The gate is not on. Nothing was read.
    #[error(
        "{action} refused: {} is {gate}, so the input was not read",
        gate::FLAG
    )]
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    GateOff {
        /// Action recorded for this outcome.
        action: Action,
        /// Gate recorded for this outcome.
        gate: Gate,
    },
    /// The input could not be read as the declared schema.
    #[error("{action} refused: invalid input: {reason}")]
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Invalid {
        /// Action recorded for this outcome.
        action: Action,
        /// Explanation accompanying this outcome.
        reason: String,
    },
    /// The input was read and is not ready: defects, undecided owner parameters, or a verdict
    /// other than promote-eligible.
    #[error("{action} refused: not ready: {}", reasons.join("; "))]
    NotReady {
        /// Action recorded for this refusal.
        action: Action,
        /// Collection of reasons values carried by this refusal.
        reasons: Vec<String>,
    },
    /// The gate is on and every check passed, but the owner ruling is not in force.
    #[error(
        "{action} refused: owner ruling {} ({}) is pending; this build has no {action} path",
        action.gap(),
        action.item()
    )]
    /// Field value recorded by this public API type.
    RulingPending {
        /// Action recorded for this outcome.
        action: Action,
    },
}

/// Ask to adopt the sealed-family roster in `proposal` (a `jope.rlfw-sealed-roster-proposal/v1`
/// document) and create its seal.
///
/// # Errors
///
/// Always. [`Refusal::GateOff`] unless `gate` is on, before `proposal` is parsed;
/// [`Refusal::Invalid`] or [`Refusal::NotReady`] from [`roster::check`]; otherwise
/// [`Refusal::RulingPending`].
pub fn request_seal(gate: &Gate, proposal: &[u8]) -> Result<Infallible, Refusal> {
    let action = Action::Seal;
    if !gate.is_on() {
        return Err(Refusal::GateOff {
            action,
            gate: gate.clone(),
        });
    }
    let report = roster::check(proposal).map_err(|reason| Refusal::Invalid { action, reason })?;
    if !report.adoptable() {
        return Err(Refusal::NotReady {
            action,
            reasons: report.reasons(),
        });
    }
    Err(Refusal::RulingPending { action })
}

/// Ask to promote the challenger of the arena `manifest` on the evidence in `results`.
///
/// # Errors
///
/// Always. [`Refusal::GateOff`] unless `gate` is on, before either input is parsed;
/// [`Refusal::Invalid`] from [`promotion::decide`]; [`Refusal::NotReady`] for any verdict other
/// than [`promotion::Verdict::PromoteEligible`]; otherwise [`Refusal::RulingPending`].
pub fn request_promotion(
    gate: &Gate,
    manifest: &[u8],
    results: &[u8],
) -> Result<Infallible, Refusal> {
    let action = Action::Promote;
    if !gate.is_on() {
        return Err(Refusal::GateOff {
            action,
            gate: gate.clone(),
        });
    }
    let report = promotion::decide(manifest, results)
        .map_err(|reason| Refusal::Invalid { action, reason })?;
    if report.verdict != promotion::Verdict::PromoteEligible {
        let mut reasons = vec![format!("verdict {}", report.verdict)];
        reasons.extend(report.reasons);
        return Err(Refusal::NotReady { action, reasons });
    }
    Err(Refusal::RulingPending { action })
}
