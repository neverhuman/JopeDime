use super::*;

pub const SCHEMA: &str = "jope.regression-finding-dispositions/v1";
pub const EVIDENCE_SCHEMA: &str = "jope.regression-finding-evidence/v1";
pub const DISPOSITIONS: [&str; 6] = [
    "confirmed-defect",
    "verified-safeguard",
    "missing-execution-evidence",
    "historical-unknown",
    "research-result",
    "deferred-non-regression",
];
/// Dispositions that record an honest open state; they can never close a finding.
pub const NEVER_CLOSED: [&str; 3] = [
    "missing-execution-evidence",
    "historical-unknown",
    "deferred-non-regression",
];
pub const STAGES: [&str; 6] = [
    "source_implementation",
    "fixture_proof",
    "hosted_ci",
    "live_qualification",
    "measurement_complete",
    "promotion",
];
/// Packages whose acceptance requires live operation on managed hosts or real storage.
pub const LIVE_PACKAGES: [&str; 4] = ["W7", "W8", "W10", "W11"];
pub const TEST_ROLES: [&str; 4] = ["pre-fix-failure", "negative", "positive", "regression"];
pub const DEFERRABLE: [&str; 3] = ["binary", "multiclass", "time-series"];
pub const SCOPE: &str =
    "evidence binding only; reviewed evidence, not this validator, carries truth";
/// Kinds that evidence carrying the conformance synthetic identity can never declare (W12-T07):
/// it may trace source or be a synthetic fixture, but never stand for real data, hosted CI,
/// live operation, measurement or a decision, so it can never close a finding on its own.
pub const NOT_SYNTHETIC_KINDS: [&str; 5] = [
    "real-corpus-fixture",
    "hosted-ci",
    "live-host",
    "measurement",
    "decision",
];

/// Evidence kinds each stage accepts. Synthetic fixtures never back live, measurement or
/// promotion stages, whatever their bytes or names say.
pub fn stage_kinds(stage: &str) -> &'static [&'static str] {
    match stage {
        "source_implementation" => &["source-trace"],
        "fixture_proof" => &["synthetic-fixture", "real-corpus-fixture"],
        "hosted_ci" => &["hosted-ci"],
        "live_qualification" => &["live-host"],
        "measurement_complete" => &["measurement"],
        "promotion" => &["decision"],
        _ => &[],
    }
}

/// The summary `verify-dispositions` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub entries: usize,
    pub with_disposition: usize,
    pub closed: usize,
    pub open: usize,
    pub closed_findings: Vec<String>,
    pub dispositions: BTreeMap<String, usize>,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("entries", self.entries.into());
        dict.insert("with_disposition", self.with_disposition.into());
        dict.insert("closed", self.closed.into());
        dict.insert("open", self.open.into());
        dict.insert(
            "closed_findings",
            Value::List(
                self.closed_findings
                    .iter()
                    .map(|f| f.as_str().into())
                    .collect(),
            ),
        );
        dict.insert(
            "dispositions",
            Value::Dict(
                self.dispositions
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}
