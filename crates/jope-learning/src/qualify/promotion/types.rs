use super::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Metric` is a record for the metric contract in this module: G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
pub struct Metric {
    /// Name recorded for this metric.
    pub name: String,
    /// Lower is better recorded for this metric.
    pub lower_is_better: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Contender` is a record for the contender contract in this module: G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
pub struct Contender {
    /// Stable identifier for this value.
    pub id: String,
    /// Sha-256 recorded for this contender.
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Look` is a record for the look contract in this module: G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
pub struct Look {
    /// Index recorded for this look.
    pub index: u32,
    /// Alpha recorded for this look.
    pub alpha: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Bootstrap` is a record for the bootstrap contract in this module: G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
pub struct Bootstrap {
    /// Resamples recorded for this bootstrap.
    pub resamples: u32,
    /// Seed recorded for this bootstrap.
    pub seed: u64,
}

/// The L6 regression panel. L6 is a validation loss: lower is better, and a challenger above the
/// frozen reference by more than `max_regression` regresses.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Safety {
    /// Panel recorded for this safety.
    pub panel: String,
    /// Reference recorded for this safety.
    pub reference: String,
    /// Upper bound on regression.
    pub max_regression: f64,
}

/// A predeclared arena.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated arena.
    pub arena_id: String,
    /// Metric used to evaluate the represented result.
    pub metric: Metric,
    /// Existing comparison score, when an incumbent is available.
    pub incumbent: Contender,
    /// Challenger recorded for this arena manifest.
    pub challenger: Contender,
    /// Roster role recorded for this arena manifest.
    pub roster_role: String,
    /// Collection of families values carried by this arena manifest.
    pub families: Vec<String>,
    /// How family effects are averaged. Only `"equal"` is accepted.
    pub family_weighting: String,
    /// Lower bound on families.
    pub min_families: usize,
    /// Incomplete families a decisive look tolerates; more makes the look inconclusive.
    #[serde(default)]
    pub max_incomplete_families: usize,
    /// Practical margin recorded for this arena manifest.
    pub practical_margin: f64,
    /// Alpha total recorded for this arena manifest.
    pub alpha_total: f64,
    /// Collection of looks values carried by this arena manifest.
    pub looks: Vec<Look>,
    /// Contenders recorded for this arena manifest.
    pub contenders: u32,
    /// Bootstrap recorded for this arena manifest.
    pub bootstrap: Bootstrap,
    /// Upper bound on failure rate.
    pub max_failure_rate: f64,
    /// Safety recorded for this arena manifest.
    pub safety: Safety,
}

/// Which program a row measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    /// The program the contender selected without seeing outer labels.
    Selected,
    /// The best outer score over all candidates: a diagnostic that leaks outer labels.
    Oracle,
}

/// One context of one family.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRow {
    /// Identifier for the associated family.
    pub family_id: String,
    /// Identifier for the associated context.
    pub context_id: String,
    /// Selection recorded for this context row.
    pub selection: Selection,
    /// The challenger's outer metric; `null` when the outcome is unknown.
    pub challenger: Option<f64>,
    /// The incumbent's outer metric; `null` when the outcome is unknown.
    pub incumbent: Option<f64>,
    /// Attempts recorded for this context row.
    pub attempts: u64,
    /// Number of attempts with a measured failure disposition.
    pub failures: u64,
}

/// A look already taken, as the caller's look ledger records it.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PriorLook {
    /// Index recorded for this prior look.
    pub index: u32,
    /// Verdict recorded for this prior look.
    pub verdict: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Cost` is a record for the cost contract in this module: G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
pub struct Cost {
    /// Cpu seconds recorded for this cost.
    pub cpu_seconds: f64,
    /// Wall seconds recorded for this cost.
    pub wall_seconds: f64,
}

/// The safety panel values: the challenger's and the frozen reference's L6.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyResult {
    /// Challenger recorded for this safety result.
    pub challenger: f64,
    /// Reference recorded for this safety result.
    pub reference: f64,
}

/// The results of one look.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaResults {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// Look recorded for this arena results.
    pub look: u32,
    /// Collection of prior looks values carried by this arena results.
    pub prior_looks: Vec<PriorLook>,
    /// Number of rows represented by this value.
    pub rows: Vec<ContextRow>,
    /// Collection of holds values carried by this arena results.
    pub holds: Vec<String>,
    /// Cost recorded for this arena results.
    pub cost: Cost,
    /// Safety recorded for this arena results.
    pub safety: SafetyResult,
}

/// The outcome of one look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The lower bound clears the practical margin. Eligible is not promoted.
    PromoteEligible,
    /// The evidence cannot tell a useful lift from a negligible one. The study may continue.
    Inconclusive,
    /// Even the upper bound is below the practical margin.
    NoPracticalLift,
    /// The challenger's L6 regresses beyond the safety limit.
    Regression,
    /// A hold, a failure rate above the limit, or absent attempt accounting.
    Hold,
}

impl Verdict {
    /// Whether the study stops at this verdict. Only an inconclusive look lets it continue.
    #[must_use]
    pub fn stops(self) -> bool {
        self != Self::Inconclusive
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PromoteEligible => "promote_eligible",
            Self::Inconclusive => "inconclusive",
            Self::NoPracticalLift => "no_practical_lift",
            Self::Regression => "regression",
            Self::Hold => "hold",
        })
    }
}

/// What [`decide`] concluded at one look.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChampionReport {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// SHA-256 digest identifying the associated results.
    pub results_sha256: String,
    /// Identifier for the associated arena.
    pub arena_id: String,
    /// Metric used to evaluate the represented result.
    pub metric: String,
    /// Existing comparison score, when an incumbent is available.
    pub incumbent: String,
    /// Challenger recorded for this champion report.
    pub challenger: String,
    /// Roster role recorded for this champion report.
    pub roster_role: String,
    /// Look recorded for this champion report.
    pub look: u32,
    /// Alpha look recorded for this champion report.
    pub alpha_look: f64,
    /// Alpha per comparison recorded for this champion report.
    pub alpha_per_comparison: f64,
    /// Alpha spent recorded for this champion report.
    pub alpha_spent: f64,
    /// Families declared recorded for this champion report.
    pub families_declared: usize,
    /// Families complete recorded for this champion report.
    pub families_complete: usize,
    /// Declared families left out: no rows, or at least one unknown context.
    pub families_incomplete: Vec<String>,
    /// Contexts recorded for this champion report.
    pub contexts: usize,
    /// Contexts unknown recorded for this champion report.
    pub contexts_unknown: usize,
    /// Families whose mean effect favours the challenger, is zero, or favours the incumbent.
    pub wins: usize,
    /// Ties recorded for this champion report.
    pub ties: usize,
    /// Losses recorded for this champion report.
    pub losses: usize,
    /// Mean of the family effects; positive favours the challenger. The decision statistic.
    pub family_effect_mean: Option<f64>,
    /// Optional family effect median value; `None` means it is not available.
    pub family_effect_median: Option<f64>,
    /// Mean over complete-family contexts, each context weighted once. Sensitivity only.
    pub context_effect_mean: Option<f64>,
    /// Family weighting recorded for this champion report.
    pub family_weighting: String,
    /// Upper bound on incomplete families.
    pub max_incomplete_families: usize,
    /// Optional lower value; `None` means it is not available.
    pub lower: Option<f64>,
    /// Optional upper value; `None` means it is not available.
    pub upper: Option<f64>,
    /// Practical margin recorded for this champion report.
    pub practical_margin: f64,
    /// Attempts recorded for this champion report.
    pub attempts: u64,
    /// Number of attempts with a measured failure disposition.
    pub failures: u64,
    /// Optional failure rate value; `None` means it is not available.
    pub failure_rate: Option<f64>,
    /// Cost recorded for this champion report.
    pub cost: Cost,
    /// Safety panel recorded for this champion report.
    pub safety_panel: String,
    /// Safety reference recorded for this champion report.
    pub safety_reference: String,
    /// Challenger L6 minus reference L6; positive is worse.
    pub safety_regression: f64,
    /// Verdict recorded for this champion report.
    pub verdict: Verdict,
    /// Collection of reasons values carried by this champion report.
    pub reasons: Vec<String>,
    /// Generalization claim recorded for this champion report.
    pub generalization_claim: &'static str,
    /// Always false: a report promotes nothing.
    pub promoted: bool,
}
