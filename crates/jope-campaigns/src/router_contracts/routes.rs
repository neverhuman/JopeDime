use super::{canonical_sha256, validate_sha256, ROUTER_FOLDS_V2, ROUTER_SEEDS_V2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewV2 {
    Raw,
    HyperionCellOnly,
    HyperionCellAppend,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PredictorRouteV2 {
    FeatureView { view: FeatureViewV2 },
    RawCellBlend { alpha: f64 },
}

impl PredictorRouteV2 {
    pub fn raw() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::Raw,
        }
    }

    pub fn cell_only() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::HyperionCellOnly,
        }
    }

    pub fn append() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::HyperionCellAppend,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::FeatureView { .. } => Ok(()),
            Self::RawCellBlend { alpha } if alpha.is_finite() && (0.0..=0.5).contains(alpha) => {
                Ok(())
            }
            Self::RawCellBlend { .. } => Err("blend alpha must be finite and in [0, 0.5]".into()),
        }
    }

    pub fn is_raw(&self) -> bool {
        matches!(
            self,
            Self::FeatureView {
                view: FeatureViewV2::Raw
            }
        )
    }

    pub fn input_width(&self, raw_width: usize) -> Result<usize, String> {
        self.validate()?;
        if raw_width == 0 {
            return Err("raw feature width must be positive".into());
        }
        match self {
            Self::FeatureView {
                view: FeatureViewV2::HyperionCellAppend,
            } => raw_width
                .checked_mul(2)
                .ok_or_else(|| "append feature width overflowed".into()),
            _ => Ok(raw_width),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ValidationSchemeV2 {
    IidGrouped,
    TimeSeries {
        time_column: String,
        seasonal_period: usize,
        purge_groups: usize,
        embargo_groups: usize,
    },
}

impl ValidationSchemeV2 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::IidGrouped => Ok(()),
            Self::TimeSeries {
                time_column,
                seasonal_period,
                ..
            } if !time_column.trim().is_empty() && *seasonal_period > 0 => Ok(()),
            Self::TimeSeries { .. } => {
                Err("time-series validation needs a time column and positive seasonality".into())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedRouteMetricV2 {
    pub seed: u64,
    pub raw_loss: f64,
    pub routed_loss: f64,
    pub relative_improvement: f64,
    pub fold_wins: u32,
    pub valid: bool,
}

impl SeedRouteMetricV2 {
    fn validate(&self) -> Result<(), String> {
        if !ROUTER_SEEDS_V2.contains(&self.seed)
            || !self.raw_loss.is_finite()
            || !self.routed_loss.is_finite()
            || self.raw_loss < 0.0
            || self.routed_loss < 0.0
            || !self.relative_improvement.is_finite()
            || self.fold_wins > ROUTER_FOLDS_V2 as u32
        {
            return Err("seed route metric is invalid".into());
        }
        let expected = if self.raw_loss > 0.0 {
            (self.raw_loss - self.routed_loss) / self.raw_loss
        } else {
            0.0
        };
        if (self.relative_improvement - expected).abs() > 1e-12 {
            return Err("seed route improvement does not match its losses".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRouteDecisionV2 {
    pub schema: String,
    pub family_id: String,
    pub selected_route: PredictorRouteV2,
    pub matched_raw_fallback: PredictorRouteV2,
    pub per_seed_metrics: Vec<SeedRouteMetricV2>,
    pub blend_coefficient: Option<f64>,
    pub latency_ms: f64,
    pub valid: bool,
    pub receipt_sha256: String,
}

impl FamilyRouteDecisionV2 {
    pub const SCHEMA: &'static str = "JainFamilyRouteDecision.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.family_id.trim().is_empty()
            || !self.latency_ms.is_finite()
            || self.latency_ms < 0.0
            || self.matched_raw_fallback != PredictorRouteV2::raw()
        {
            return Err("family route decision is malformed".into());
        }
        validate_sha256("family receipt", &self.receipt_sha256)?;
        self.selected_route.validate()?;
        if !self.valid && !self.selected_route.is_raw() {
            return Err("invalid family decisions must select raw".into());
        }
        match (&self.selected_route, self.blend_coefficient) {
            (PredictorRouteV2::RawCellBlend { alpha }, Some(coefficient))
                if (*alpha - coefficient).abs() <= 1e-12 => {}
            (PredictorRouteV2::RawCellBlend { .. }, _) => {
                return Err("blend decision has no matching coefficient".into());
            }
            (_, None) => {}
            (_, Some(_)) => return Err("nonblend decision carries a coefficient".into()),
        }
        if self.per_seed_metrics.len() != ROUTER_SEEDS_V2.len() {
            return Err("family decision does not contain both router seeds".into());
        }
        for (index, metric) in self.per_seed_metrics.iter().enumerate() {
            metric.validate()?;
            if metric.seed != ROUTER_SEEDS_V2[index] {
                return Err("seed metrics are not in canonical order".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberInputPlanV2 {
    pub schema: String,
    pub member_id: String,
    pub family_id: String,
    pub route: PredictorRouteV2,
    pub raw_width: usize,
    pub predictor_sha256: String,
}

impl MemberInputPlanV2 {
    pub const SCHEMA: &'static str = "JainMemberInputPlan.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.member_id.trim().is_empty()
            || self.family_id.trim().is_empty()
            || self.raw_width == 0
        {
            return Err("member input plan identity or width is invalid".into());
        }
        self.route.validate()?;
        validate_sha256("member predictor", &self.predictor_sha256)
    }

    pub fn input_width(&self) -> Result<usize, String> {
        self.validate()?;
        self.route.input_width(self.raw_width)
    }
}

pub fn validate_family_ensemble_v2(
    decisions: &[FamilyRouteDecisionV2],
    members: &[MemberInputPlanV2],
) -> Result<(), String> {
    let mut selected = BTreeMap::new();
    for decision in decisions {
        decision.validate()?;
        if selected
            .insert(decision.family_id.as_str(), &decision.selected_route)
            .is_some()
        {
            return Err("family decisions contain a duplicate family".into());
        }
    }
    let mut member_ids = BTreeSet::new();
    for member in members {
        member.validate()?;
        if !member_ids.insert(member.member_id.as_str()) {
            return Err("ensemble member IDs are duplicated".into());
        }
        let route = selected
            .get(member.family_id.as_str())
            .ok_or_else(|| format!("member family {} has no route", member.family_id))?;
        if *route != &member.route {
            return Err(format!(
                "member {} does not use its family route",
                member.member_id
            ));
        }
    }
    Ok(())
}
