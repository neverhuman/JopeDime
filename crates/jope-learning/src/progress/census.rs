//! W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
//!
//! [`reconcile`] is a pure function of the run manifest and the delivered events and HWM
//! decisions. It never trusts arrival order and never infers "completed" from the existence of a
//! file or "improved" from the presence of a score: a unit is completed only by exactly one
//! `completed` event, failed only by a terminal `failed` event, and improved only by exactly one
//! [`HwmDecision`] bound to that completion. Every contradiction is a [`Violation`] — reported,
//! counted and never resolved by picking a winner — and a census with any violation or any
//! non-terminal unit is not reconciled.
//!
//! The properties that make attrition visible (S08-G03/K03, S01-G7/K7):
//!
//! * a failure stays counted after its retry, because the retry names it: dropping the failure
//!   orphans the retry (`retry-without-failure`) instead of silently shrinking the denominator;
//! * dropping a terminal failure turns the unit back into in-flight work, which lowers coverage —
//!   losing an event can never raise it;
//! * the same bytes delivered again are counted as a duplicate delivery and have no other effect;
//!   an explicit new attempt is a `retry` and is counted as one;
//! * a logical unit has at most one completion and at most one decision, whatever the number of
//!   attempts, so a retried attempt cannot become a second logical result or improvement;
//! * an empty denominator is reported as unavailable ([`Fraction::rate`] is `None`), never as a
//!   rate.

mod model;
mod reconcile;

pub use model::{
    violation, Census, Counts, Disposition, ProgressCensusDisposition, UnitStatus, Violation,
};
pub use reconcile::reconcile;
