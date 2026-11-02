use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Where one logical unit stands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
pub enum ProgressCensusDisposition {
    /// Terminal: exactly one attempt completed.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Completed {
        /// Attempt number associated with this outcome.
        attempt: u32,
        /// Completion recorded for this outcome.
        completion: String,
    },
    /// Terminal: the last attempt failed and either was not retryable or exhausted the budget.
    Failed {
        /// Attempts recorded for this disposition.
        attempts: u32,
        /// Class recorded for this disposition.
        class: FailureClass,
        /// Exhausted recorded for this disposition.
        exhausted: bool,
    },
    /// Terminal: excluded without being scheduled.
    /// Field value recorded by this public API type.
    Excluded {
        /// Explanation accompanying this outcome.
        reason: String,
    },
    /// Not terminal: the latest attempt has no outcome.
    /// Field value recorded by this public API type.
    InFlight {
        /// Attempt number associated with this outcome.
        attempt: u32,
    },
    /// Not terminal: the latest attempt failed, may be retried, and has not been.
    /// Field value recorded by this public API type.
    AwaitingRetry {
        /// Attempt number associated with this outcome.
        attempt: u32,
    },
    /// Not terminal: nothing was recorded for the unit.
    Unscheduled,
}

/// Compatibility alias for the domain-specific `ProgressCensusDisposition` type.
pub type Disposition = ProgressCensusDisposition;

impl ProgressCensusDisposition {
    /// Performs the terminal operation.
    pub fn terminal(&self) -> bool {
        matches!(
            self,
            ProgressCensusDisposition::Completed { .. }
                | ProgressCensusDisposition::Failed { .. }
                | ProgressCensusDisposition::Excluded { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `UnitStatus` is a record for the unit status contract in this module: W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
pub struct UnitStatus {
    /// Identifier for the associated work.
    pub work_id: String,
    /// ProgressCensusDisposition recorded for this unit status.
    pub disposition: ProgressCensusDisposition,
    /// Attempts recorded for this unit status.
    pub attempts: u32,
    /// Retries recorded for this unit status.
    pub retries: u32,
    /// Number of attempts with a measured failure disposition.
    pub failures: u32,
    /// The one HWM decision counted for this unit, if any.
    pub decision: Option<String>,
}

/// One contradiction. The census reports it rather than resolving it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Violation {
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Identifier for the associated work.
    pub work_id: Option<String>,
    /// Optional object value; `None` means it is not available.
    pub object: Option<String>,
    /// Detail recorded for this violation.
    pub detail: String,
}

/// Violation kinds.
pub mod violation {
    /// Constant identifying the foreign run.
    pub const FOREIGN_RUN: &str = "foreign-run";
    /// Constant identifying the unplanned work.
    pub const UNPLANNED_WORK: &str = "unplanned-work";
    /// Constant identifying the attempt without schedule.
    pub const ATTEMPT_WITHOUT_SCHEDULE: &str = "attempt-without-schedule";
    /// Constant identifying the conflicting schedules.
    pub const CONFLICTING_SCHEDULES: &str = "conflicting-schedules";
    /// Constant identifying the retry without failure.
    pub const RETRY_WITHOUT_FAILURE: &str = "retry-without-failure";
    /// Constant identifying the retry after terminal failure.
    pub const RETRY_AFTER_TERMINAL_FAILURE: &str = "retry-after-terminal-failure";
    /// Constant identifying the attempt budget exceeded.
    pub const ATTEMPT_BUDGET_EXCEEDED: &str = "attempt-budget-exceeded";
    /// Constant identifying the conflicting starts.
    pub const CONFLICTING_STARTS: &str = "conflicting-starts";
    /// Constant identifying the conflicting outcomes.
    pub const CONFLICTING_OUTCOMES: &str = "conflicting-outcomes";
    /// Constant identifying the second completion.
    pub const SECOND_COMPLETION: &str = "second-completion";
    /// Constant identifying the work after completion.
    pub const WORK_AFTER_COMPLETION: &str = "work-after-completion";
    /// Constant identifying the conflicting exclusions.
    pub const CONFLICTING_EXCLUSIONS: &str = "conflicting-exclusions";
    /// Constant identifying the excluded work was scheduled.
    pub const EXCLUDED_WORK_WAS_SCHEDULED: &str = "excluded-work-was-scheduled";
    /// Constant identifying the decision without completion.
    pub const DECISION_WITHOUT_COMPLETION: &str = "decision-without-completion";
    /// Constant identifying the second decision.
    pub const SECOND_DECISION: &str = "second-decision";
}

/// The aggregate view: every numerator with its denominator and every exclusion with its reason.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    /// Planned recorded for this counts.
    pub planned: u64,
    /// Completed recorded for this counts.
    pub completed: u64,
    /// Failed recorded for this counts.
    pub failed: u64,
    /// Excluded recorded for this counts.
    pub excluded: u64,
    /// In flight recorded for this counts.
    pub in_flight: u64,
    /// Awaiting retry recorded for this counts.
    pub awaiting_retry: u64,
    /// Unscheduled recorded for this counts.
    pub unscheduled: u64,
    /// Distinct attempts opened, across all units.
    pub attempts: u64,
    /// Retries recorded for this counts.
    pub retries: u64,
    /// Every `failed` event by class, including failures that were later retried.
    pub failures: BTreeMap<String, u64>,
    /// Exclusions by reason.
    pub exclusions: BTreeMap<String, u64>,
    /// Counted HWM decisions by transition.
    pub decisions: BTreeMap<String, u64>,
    /// Logical units whose one counted decision is `improved`.
    pub improvements: u64,
    /// Events and decisions delivered, and how many of them were byte-identical repeats.
    pub deliveries: u64,
    /// Duplicate deliveries recorded for this counts.
    pub duplicate_deliveries: u64,
    /// Violations recorded for this counts.
    pub violations: u64,
    /// Terminal units over planned units.
    pub coverage: Fraction,
    /// Completed units over planned units that were not excluded.
    pub completion: Fraction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `Census` is a record for the census contract in this module: W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
pub struct Census {
    /// Identifier for the associated run.
    pub run_id: String,
    /// Run manifest recorded for this census.
    pub run_manifest: String,
    /// Counts recorded for this census.
    pub counts: Counts,
    /// One per planned unit, in manifest order.
    pub units: Vec<UnitStatus>,
    /// Collection of violations values carried by this census.
    pub violations: Vec<Violation>,
}

impl Census {
    /// Every planned unit is terminal and nothing contradicts anything.
    pub fn reconciled(&self) -> bool {
        self.violations.is_empty() && self.units.iter().all(|u| u.disposition.terminal())
    }

    /// Performs the unit operation.
    pub fn unit(&self, work_id: &str) -> Option<&UnitStatus> {
        self.units.iter().find(|u| u.work_id == work_id)
    }

    /// Performs the violation kinds operation.
    pub fn violation_kinds(&self) -> BTreeSet<&str> {
        self.violations.iter().map(|v| v.kind.as_str()).collect()
    }
}
