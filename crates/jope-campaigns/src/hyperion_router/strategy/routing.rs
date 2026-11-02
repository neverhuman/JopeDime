use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendFitV2 {
    pub alpha: f64,
    pub denominator: f64,
    pub stable: bool,
}

/// Fit the raw-biased blend from inner OOF predictions only.
pub fn fit_raw_biased_blend_v2(
    truth: &[f64],
    raw_oof: &[f64],
    cell_oof: &[f64],
) -> Result<BlendFitV2, String> {
    if truth.is_empty() || truth.len() != raw_oof.len() || truth.len() != cell_oof.len() {
        return Err("blend OOF vectors are empty or have different lengths".into());
    }
    require_finite("blend truth", truth)?;
    require_finite("blend raw predictions", raw_oof)?;
    require_finite("blend cell predictions", cell_oof)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for ((truth, raw), cell) in truth.iter().zip(raw_oof).zip(cell_oof) {
        let direction = cell - raw;
        numerator += direction * (truth - raw);
        denominator += direction * direction;
    }
    let stability_floor = 1e-12 * truth.len() as f64;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= stability_floor {
        return Ok(BlendFitV2 {
            alpha: 0.0,
            denominator,
            stable: false,
        });
    }
    Ok(BlendFitV2 {
        alpha: (numerator / denominator).clamp(0.0, 0.5),
        denominator,
        stable: true,
    })
}

pub fn apply_raw_biased_blend_v2(
    raw_predictions: &[f64],
    cell_predictions: &[f64],
    fit: BlendFitV2,
) -> Result<Vec<f64>, String> {
    if raw_predictions.len() != cell_predictions.len()
        || raw_predictions.is_empty()
        || !fit.alpha.is_finite()
        || !(0.0..=0.5).contains(&fit.alpha)
    {
        return Err("blend prediction inputs or coefficient are invalid".into());
    }
    require_finite("blend raw predictions", raw_predictions)?;
    require_finite("blend cell predictions", cell_predictions)?;
    Ok(raw_predictions
        .iter()
        .zip(cell_predictions)
        .map(|(raw, cell)| raw + fit.alpha * (cell - raw))
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteFamilyV2 {
    Raw,
    CellOnly,
    Append,
    RawCellBlend,
}

impl RouteFamilyV2 {
    fn preference(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::CellOnly => 1,
            Self::Append => 2,
            Self::RawCellBlend => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationThresholdsV2 {
    pub minimum_seed_relative_improvement_micros: u32,
    pub minimum_seed_normalized_improvement_micros: u32,
    pub minimum_fold_wins: u32,
    pub grouped_bootstrap_confidence_micros: u32,
    pub maximum_fold_regression_micros: u32,
    pub minimum_two_seed_median_improvement_micros: u32,
    pub bootstrap_repetitions: u32,
}

impl Default for ActivationThresholdsV2 {
    fn default() -> Self {
        Self {
            minimum_seed_relative_improvement_micros: 5_000,
            minimum_seed_normalized_improvement_micros: 5_000,
            minimum_fold_wins: 5,
            grouped_bootstrap_confidence_micros: 950_000,
            maximum_fold_regression_micros: 20_000,
            minimum_two_seed_median_improvement_micros: 10_000,
            bootstrap_repetitions: GROUPED_BOOTSTRAP_REPETITIONS as u32,
        }
    }
}
