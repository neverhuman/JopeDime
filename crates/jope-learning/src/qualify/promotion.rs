//! G36 / J18: Gate F, champion promotion (expert review CLOVER §13.6, §8.7).
//!
//! [`decide`] reads two documents:
//!
//! - an **arena manifest** (`jope.qualify-arena-manifest/v1`), fixed before any result exists:
//!   the incumbent and challenger, the independent families, the metric and its direction, the
//!   practical margin, the alpha budget and its predeclared looks, the number of contenders
//!   sharing each look, the bootstrap, the failure-rate limit and the L6 safety panel;
//! - the **results** of one look (`jope.qualify-arena-results/v1`), bound to the exact manifest
//!   bytes by SHA-256: one row per context with the selected program's outer metric for both
//!   contenders, attempts and failures, holds, cost, and the safety panel values.
//!
//! The independent unit is the family, and every family weighs the same (`family_weighting:
//! "equal"`, the only weighting accepted). Context effects are averaged within their family, and
//! a family with any unknown context is left out whole rather than imputed. Leaving families out
//! can favour either side, so the bounds decide nothing while more families are incomplete than
//! the manifest's `max_incomplete_families` (default 0) allows: the look is inconclusive. The
//! challenger is promote-eligible only when the one-sided lower bound of the mean family effect
//! clears the practical margin, not merely zero. That bound comes from a percentile bootstrap
//! over families at this look's alpha divided by the contenders. A percentile bootstrap
//! under-covers with few families; `min_families` is the guard, and its value belongs to the
//! G36 ruling.
//!
//! Some inputs are refused as invalid, not given a verdict: an outer-oracle row (a diagnostic,
//! never evidence), a development or meta-training roster (not independent confirmation), a
//! sealed-final roster (no roster is adopted in this build, so no seal look exists), a look out
//! of schedule, and a look after the study stopped. A negative or inconclusive arena is a
//! verdict, not an error. The report never promotes and never claims unseen-family
//! generalization.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::canon::{is_hex64, sha256_hex};

/// The arena manifest schema.
pub const MANIFEST_SCHEMA: &str = "jope.qualify-arena-manifest/v1";
/// The look results schema.
pub const RESULTS_SCHEMA: &str = "jope.qualify-arena-results/v1";
/// The report schema [`decide`] writes.
pub const REPORT_SCHEMA: &str = "jope.qualify-champion-report/v1";
/// Fewest bootstrap resamples a manifest may declare.
pub const MIN_RESAMPLES: u32 = 1000;
/// What every report says about unseen-family generalization.
pub const GENERALIZATION_CLAIM: &str = "none: no adopted sealed-family boundary in this build";

/// Tolerance for the alpha-spending sum, so that a schedule like 0.01 + 0.04 = 0.05 is exact.
const ALPHA_SLACK: f64 = 1e-12;

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

fn finite(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} is not finite"))
    }
}

fn validate_manifest(manifest: &ArenaManifest) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!(
            "manifest schema {:?} is not {MANIFEST_SCHEMA}",
            manifest.schema
        ));
    }
    if manifest.arena_id.is_empty() || manifest.metric.name.is_empty() {
        return Err("arena_id and metric.name are required".to_owned());
    }
    for (role, contender) in [
        ("incumbent", &manifest.incumbent),
        ("challenger", &manifest.challenger),
    ] {
        if contender.id.is_empty() || !is_hex64(&contender.sha256) {
            return Err(format!("{role} needs an id and a lowercase hex sha256"));
        }
    }
    if manifest.challenger.sha256 == manifest.incumbent.sha256 {
        return Err(
            "the challenger is the incumbent; keep the champion and the learning branch separate"
                .to_owned(),
        );
    }
    match manifest.roster_role.as_str() {
        "confirmation" => {}
        "sealed_final" => {
            return Err(
                "sealed_final looks need an adopted sealed roster (G23); none exists in this build"
                    .to_owned(),
            )
        }
        "development" | "meta_training" => {
            return Err(format!(
                "{} families are not independent confirmation evidence",
                manifest.roster_role
            ))
        }
        other => return Err(format!("unknown roster role {other:?}")),
    }
    let unique: BTreeSet<&str> = manifest.families.iter().map(String::as_str).collect();
    if unique.len() != manifest.families.len() || unique.contains("") {
        return Err("families must be unique non-empty ids".to_owned());
    }
    if manifest.family_weighting != "equal" {
        return Err(format!(
            "family_weighting {:?} is not \"equal\", the only weighting this gate estimates",
            manifest.family_weighting
        ));
    }
    if manifest.min_families < 2 || manifest.min_families > manifest.families.len() {
        return Err(format!(
            "min_families {} must be at least 2 and at most the {} declared families",
            manifest.min_families,
            manifest.families.len()
        ));
    }
    if manifest.max_incomplete_families > manifest.families.len() - manifest.min_families {
        return Err(format!(
            "max_incomplete_families {} leaves fewer than min_families {} of {} families",
            manifest.max_incomplete_families,
            manifest.min_families,
            manifest.families.len()
        ));
    }
    if finite("practical_margin", manifest.practical_margin)? < 0.0 {
        return Err("practical_margin is negative".to_owned());
    }
    let alpha_total = finite("alpha_total", manifest.alpha_total)?;
    if alpha_total <= 0.0 || alpha_total >= 1.0 {
        return Err("alpha_total must lie in (0, 1)".to_owned());
    }
    if manifest.looks.is_empty() {
        return Err("at least one look must be declared".to_owned());
    }
    if manifest.contenders == 0 {
        return Err("contenders must be at least 1".to_owned());
    }
    if manifest.bootstrap.resamples < MIN_RESAMPLES {
        return Err(format!(
            "bootstrap.resamples must be at least {MIN_RESAMPLES}"
        ));
    }
    let mut spent = 0.0;
    for (position, look) in manifest.looks.iter().enumerate() {
        if usize::try_from(look.index).ok() != Some(position + 1) {
            return Err("looks must be numbered 1, 2, ... in order".to_owned());
        }
        let alpha = finite("look alpha", look.alpha)?;
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(format!("look {} alpha must lie in (0, 1)", look.index));
        }
        let per_comparison = alpha / f64::from(manifest.contenders);
        if per_comparison * f64::from(manifest.bootstrap.resamples) < 1.0 {
            return Err(format!(
                "look {} alpha per comparison {per_comparison} is finer than {} resamples resolve",
                look.index, manifest.bootstrap.resamples
            ));
        }
        spent += alpha;
    }
    if spent > alpha_total + ALPHA_SLACK {
        return Err(format!(
            "looks spend alpha {spent}, more than alpha_total {alpha_total}"
        ));
    }
    let max_failure_rate = finite("max_failure_rate", manifest.max_failure_rate)?;
    if !(0.0..=1.0).contains(&max_failure_rate) {
        return Err("max_failure_rate must lie in [0, 1]".to_owned());
    }
    if manifest.safety.panel.is_empty() || manifest.safety.reference.is_empty() {
        return Err("safety.panel and safety.reference are required".to_owned());
    }
    if finite("safety.max_regression", manifest.safety.max_regression)? < 0.0 {
        return Err("safety.max_regression is negative".to_owned());
    }
    Ok(())
}

/// SplitMix64: a small, fully specified generator, so a bootstrap replays from its seed alone.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` without modulo bias; `n` is positive.
    fn below(&mut self, n: u64) -> u64 {
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let draw = self.next();
            if draw < limit {
                return draw % n;
            }
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    }
}

/// The inverse empirical CDF (type 1) of `sorted` at `p` in (0, 1).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The percentile-bootstrap bounds of the mean of `effects` at one-sided level `alpha`.
fn bootstrap_bounds(effects: &[f64], bootstrap: Bootstrap, alpha: f64) -> (f64, f64) {
    let mut rng = SplitMix64(bootstrap.seed);
    let n = effects.len() as u64;
    let mut means = Vec::with_capacity(bootstrap.resamples as usize);
    let mut draw = vec![0.0; effects.len()];
    for _ in 0..bootstrap.resamples {
        for slot in &mut draw {
            *slot = effects[usize::try_from(rng.below(n)).unwrap_or(0)];
        }
        means.push(mean(&draw));
    }
    means.sort_by(f64::total_cmp);
    (quantile(&means, alpha), quantile(&means, 1.0 - alpha))
}

/// Decide one look of a predeclared arena. Read-only; promotes nothing.
///
/// # Errors
///
/// Either document is not valid JSON of its schema, the manifest is not a usable predeclared
/// confirmation arena, the results do not bind its exact bytes, the look is out of schedule or
/// after a stop, or a row is invalid (an oracle row, an undeclared family, a repeated context, a
/// non-finite metric, more failures than attempts).
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn decide(manifest_bytes: &[u8], results_bytes: &[u8]) -> Result<ChampionReport, String> {
    let manifest: ArenaManifest =
        serde_json::from_slice(manifest_bytes).map_err(|e| format!("manifest: {e}"))?;
    validate_manifest(&manifest)?;
    let results: ArenaResults =
        serde_json::from_slice(results_bytes).map_err(|e| format!("results: {e}"))?;
    if results.schema != RESULTS_SCHEMA {
        return Err(format!(
            "results schema {:?} is not {RESULTS_SCHEMA}",
            results.schema
        ));
    }
    let manifest_sha256 = sha256_hex(manifest_bytes);
    if results.manifest_sha256 != manifest_sha256 {
        return Err("results do not bind the exact manifest bytes".to_owned());
    }

    // The look schedule: this look is declared, every earlier look was taken in order, and none
    // of them stopped the study.
    let look = manifest
        .looks
        .iter()
        .find(|item| item.index == results.look)
        .copied()
        .ok_or_else(|| format!("look {} is not declared", results.look))?;
    let prior: Vec<u32> = results.prior_looks.iter().map(|item| item.index).collect();
    if prior != (1..results.look).collect::<Vec<_>>() {
        return Err(format!(
            "look {} needs exactly the prior looks 1..{}, got {prior:?}",
            results.look, results.look
        ));
    }
    if let Some(stop) = results.prior_looks.iter().find(|item| item.verdict.stops()) {
        return Err(format!(
            "the study stopped at look {} ({})",
            stop.index, stop.verdict
        ));
    }
    let alpha_spent: f64 = manifest
        .looks
        .iter()
        .take_while(|item| item.index <= look.index)
        .map(|item| item.alpha)
        .sum();
    let alpha_per_comparison = look.alpha / f64::from(manifest.contenders);

    // Rows: declared families, unique contexts, selected programs only.
    let declared: BTreeSet<&str> = manifest.families.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut by_family: BTreeMap<&str, Vec<Option<f64>>> = BTreeMap::new();
    let (mut attempts, mut failures) = (0_u64, 0_u64);
    for row in &results.rows {
        if row.selection == Selection::Oracle {
            return Err(format!(
                "context {} is an outer-oracle row: a diagnostic, never promotion evidence",
                row.context_id
            ));
        }
        if !declared.contains(row.family_id.as_str()) {
            return Err(format!("family {} is not declared", row.family_id));
        }
        if !seen.insert((row.family_id.as_str(), row.context_id.as_str())) {
            return Err(format!("context {} is repeated", row.context_id));
        }
        if row.failures > row.attempts {
            return Err(format!(
                "context {} has more failures than attempts",
                row.context_id
            ));
        }
        attempts = attempts.saturating_add(row.attempts);
        failures = failures.saturating_add(row.failures);
        let effect = match (row.challenger, row.incumbent) {
            (Some(challenger), Some(incumbent)) => {
                finite("challenger metric", challenger)?;
                finite("incumbent metric", incumbent)?;
                Some(if manifest.metric.lower_is_better {
                    incumbent - challenger
                } else {
                    challenger - incumbent
                })
            }
            _ => None,
        };
        by_family
            .entry(row.family_id.as_str())
            .or_default()
            .push(effect);
    }
    let cost = Cost {
        cpu_seconds: finite("cost.cpu_seconds", results.cost.cpu_seconds)?,
        wall_seconds: finite("cost.wall_seconds", results.cost.wall_seconds)?,
    };
    if cost.cpu_seconds < 0.0 || cost.wall_seconds < 0.0 {
        return Err("cost is negative".to_owned());
    }
    let safety_regression = finite("safety.challenger", results.safety.challenger)?
        - finite("safety.reference", results.safety.reference)?;

    // Family effects. A family with no rows or an unknown context is incomplete, never imputed.
    let mut family_effects = Vec::new();
    let mut context_effects = Vec::new();
    let mut families_incomplete = Vec::new();
    let contexts_unknown = by_family
        .values()
        .flatten()
        .filter(|effect| effect.is_none())
        .count();
    for family in &manifest.families {
        let known: Option<Vec<f64>> = by_family
            .get(family.as_str())
            .filter(|effects| !effects.is_empty())
            .and_then(|effects| effects.iter().copied().collect());
        match known {
            Some(effects) => {
                family_effects.push(mean(&effects));
                context_effects.extend(effects);
            }
            None => families_incomplete.push(family.clone()),
        }
    }
    let wins = family_effects
        .iter()
        .filter(|effect| **effect > 0.0)
        .count();
    let losses = family_effects
        .iter()
        .filter(|effect| **effect < 0.0)
        .count();
    let ties = family_effects.len() - wins - losses;
    let mut sorted = family_effects.clone();
    sorted.sort_by(f64::total_cmp);
    let (family_effect_mean, family_effect_median, context_effect_mean) = if sorted.is_empty() {
        (None, None, None)
    } else {
        (
            Some(mean(&family_effects)),
            Some(median(&sorted)),
            Some(mean(&context_effects)),
        )
    };
    let failure_rate = (attempts > 0).then(|| failures as f64 / attempts as f64);

    // The verdict, in precedence order; every applicable reason is kept.
    let mut reasons = Vec::new();
    let mut verdict = None;
    if !families_incomplete.is_empty() {
        reasons.push(format!(
            "{} declared families incomplete and left out, not imputed",
            families_incomplete.len()
        ));
    }
    if !results.holds.is_empty() {
        reasons.push(format!("holds in force: {}", results.holds.join(", ")));
        verdict = verdict.or(Some(Verdict::Hold));
    }
    match failure_rate {
        None => {
            reasons.push("no attempts recorded: attempt accounting is absent".to_owned());
            verdict = verdict.or(Some(Verdict::Hold));
        }
        Some(rate) if rate > manifest.max_failure_rate => {
            reasons.push(format!(
                "failure rate {rate} exceeds the limit {}",
                manifest.max_failure_rate
            ));
            verdict = verdict.or(Some(Verdict::Hold));
        }
        Some(_) => {}
    }
    if safety_regression > manifest.safety.max_regression {
        reasons.push(format!(
            "{} regresses by {safety_regression} against {} (lower is better; limit {})",
            manifest.safety.panel, manifest.safety.reference, manifest.safety.max_regression
        ));
        verdict = verdict.or(Some(Verdict::Regression));
    }
    let mut bounds = (None, None);
    if family_effects.len() < manifest.min_families {
        reasons.push(format!(
            "{} complete families, fewer than the {} required",
            family_effects.len(),
            manifest.min_families
        ));
        verdict = verdict.or(Some(Verdict::Inconclusive));
    } else {
        let (lower, upper) =
            bootstrap_bounds(&family_effects, manifest.bootstrap, alpha_per_comparison);
        bounds = (Some(lower), Some(upper));
        let margin = manifest.practical_margin;
        if families_incomplete.len() > manifest.max_incomplete_families {
            // Leaving families out can favour either contender, so the bounds decide nothing.
            reasons.push(format!(
                "{} incomplete families exceed the predeclared {}: selection on missingness",
                families_incomplete.len(),
                manifest.max_incomplete_families
            ));
            verdict = verdict.or(Some(Verdict::Inconclusive));
        } else if lower > margin {
            verdict = verdict.or(Some(Verdict::PromoteEligible));
        } else if upper < margin {
            reasons.push(format!(
                "upper bound {upper} is below the practical margin {margin}"
            ));
            verdict = verdict.or(Some(Verdict::NoPracticalLift));
        } else {
            reasons.push(format!(
                "bounds [{lower}, {upper}] straddle the practical margin {margin}"
            ));
            verdict = verdict.or(Some(Verdict::Inconclusive));
        }
    }

    Ok(ChampionReport {
        schema: REPORT_SCHEMA,
        manifest_sha256,
        results_sha256: sha256_hex(results_bytes),
        arena_id: manifest.arena_id,
        metric: manifest.metric.name,
        incumbent: manifest.incumbent.id,
        challenger: manifest.challenger.id,
        roster_role: manifest.roster_role,
        look: look.index,
        alpha_look: look.alpha,
        alpha_per_comparison,
        alpha_spent,
        families_declared: manifest.families.len(),
        families_complete: family_effects.len(),
        families_incomplete,
        contexts: results.rows.len(),
        contexts_unknown,
        wins,
        ties,
        losses,
        family_effect_mean,
        family_effect_median,
        context_effect_mean,
        family_weighting: manifest.family_weighting,
        max_incomplete_families: manifest.max_incomplete_families,
        lower: bounds.0,
        upper: bounds.1,
        practical_margin: manifest.practical_margin,
        attempts,
        failures,
        failure_rate,
        cost,
        safety_panel: manifest.safety.panel,
        safety_reference: manifest.safety.reference,
        safety_regression,
        verdict: verdict.unwrap_or(Verdict::Inconclusive),
        reasons,
        generalization_claim: GENERALIZATION_CLAIM,
        promoted: false,
    })
}
