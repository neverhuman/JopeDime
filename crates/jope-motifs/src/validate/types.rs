use super::*;

/// Private label of the validation report. There is no schema file for it.
pub const VALIDATION_SCHEMA: &str = "jope.motif-validation/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
/// Options controlling how many recorded hosts are fitted for each fragment.
pub struct ValidateOptions {
    /// Distinct recorded hosts fitted per fragment, in provenance order.
    pub hosts_per_fragment: u32,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            hosts_per_fragment: 3,
        }
    }
}

/// The private validation report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Private validation result bound to a table and, optionally, a fresh source read.
pub struct Validation {
    /// Validation report schema identifier.
    pub schema: String,
    /// Digest of the table that was checked.
    pub table_sha256: String,
    /// True only when a fresh read of the source (`validate --state`) confirmed every recorded
    /// occurrence ([`verify_source`]). False when the source was not read, when nothing was
    /// examined, or when any occurrence is no longer current.
    pub source_verified: bool,
    /// `Some` exactly when the source was read: whether its digest is unchanged. The digest
    /// covers the source's exclusion reasons too, so it is recorded, never relied on.
    pub source_digest_equal: Option<bool>,
    /// Results for each fragment in table order.
    pub motifs: Vec<MotifResult>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Validation outcomes for one fragment.
pub struct MotifResult {
    /// Zero-based index of the fragment in table order.
    pub index: u32,
    /// Whether rewriting and re-expanding its recorded host is exact.
    pub round_trip: CheckResult,
    /// Whether repeated evaluation is deterministic.
    pub determinism: CheckResult,
    /// Whether independent compatible contexts were fitted successfully.
    pub context_compatibility: CheckResult,
    /// Whether emitted artifacts omit the reserved fragment reference name.
    pub name_absence: CheckResult,
    /// Hosts whose fit was refused on every table. Never counted as a pass.
    pub not_exercised: u32,
    /// Backend capability observations for the attempted contexts and tasks.
    pub backend: Vec<BackendRow>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Aggregate outcome for one validation check class.
pub struct CheckResult {
    /// True only when at least one case was exercised and none failed.
    pub passed: bool,
    /// Number of cases examined by the check.
    pub exercised: u32,
    /// Fixed-code failures observed while running the check.
    pub failures: Vec<Failure>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A fixed validation failure and its optional local context.
pub struct Failure {
    /// Fixed failure code; it contains no private source value.
    pub code: String,
    /// Host recipe digest associated with the failure, when available.
    pub recipe_sha256: Option<String>,
    /// Context label associated with the failure, when available.
    pub context: Option<String>,
    /// Table label associated with the failure, when available.
    pub table: Option<String>,
    /// Task associated with the failure, when available.
    pub task: Option<TaskLabel>,
}

/// Backend coverage of one context and task. Recorded, never gating: coverage is not parity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendRow {
    /// Synthetic context label used for this observation.
    pub context: String,
    /// Task evaluated in the context.
    pub task: TaskLabel,
    /// False when the context text, as fitted or with the task's default head written out,
    /// equals a recorded host. Such a fit repeats a host and never satisfies context
    /// compatibility.
    pub independent: bool,
    /// The legacy planner accepts the context text (every row).
    pub legacy_plan: bool,
    /// Q3 only: whether the legacy fit on S2's numeric rows succeeded. `None` when no legacy
    /// fit was attempted: on the Q1 and Q2 rows, or when the planner refused the text.
    pub legacy_fit: Option<bool>,
    /// Whether the legacy C++17 emitter accepted the context.
    pub legacy_cpp17: bool,
    /// Whether the legacy JavaScript emitter accepted the context.
    pub legacy_javascript: bool,
    /// Whether the bound artifact emitter accepted the context.
    pub bound_emit: bool,
    /// Whether fitted C++17 emission succeeded.
    pub fitted_cpp17: bool,
}

impl CheckResult {
    /// Seal a check. A check that examined nothing never passes: it records `unexercised`
    /// as a failure instead.
    pub(super) fn finish(mut self, unexercised: &str) -> Self {
        if self.exercised == 0 {
            self.failures.push(Failure::new(unexercised));
        }
        self.passed = self.exercised > 0 && self.failures.is_empty();
        self
    }
}

impl Failure {
    pub(super) fn new(code: &str) -> Self {
        Self {
            code: code.to_owned(),
            recipe_sha256: None,
            context: None,
            table: None,
            task: None,
        }
    }

    pub(super) fn recipe(mut self, recipe_sha256: &str) -> Self {
        self.recipe_sha256 = Some(recipe_sha256.to_owned());
        self
    }

    pub(super) fn context(mut self, context: &str) -> Self {
        self.context = Some(context.to_owned());
        self
    }

    pub(super) fn table(mut self, shape: Shape) -> Self {
        self.table = Some(shape.as_str().to_owned());
        self
    }

    pub(super) fn task(mut self, task: TaskLabel) -> Self {
        self.task = Some(task);
        self
    }
}

impl MotifResult {
    /// V0 passed (or no result would exist) and C1 to C4 passed. Each of them passed only
    /// after examining at least one occurrence, fit, context or artifact.
    pub fn passed(&self) -> bool {
        self.round_trip.passed
            && self.determinism.passed
            && self.context_compatibility.passed
            && self.name_absence.passed
    }
}

impl Validation {
    /// At least one fragment was checked, every fragment passed, and, when the source was read,
    /// it confirmed every occurrence. An empty table never passes: nothing was examined.
    pub fn passed(&self) -> bool {
        !self.motifs.is_empty()
            && self.motifs.iter().all(MotifResult::passed)
            && (self.source_digest_equal.is_none() || self.source_verified)
    }
}
