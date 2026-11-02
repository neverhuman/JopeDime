use super::*;

/// Global denominators of every term, computed from labels alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Denominators {
    /// `Σ (0.05 + p⁴)` over all rows (row weights formed in f32, as on device).
    pub sequence_weight: f64,
    /// `Σ |Δv|·boost` over all non-tied pairs; shared by the ablation term.
    pub pair_weight: f64,
    /// Listwise-eligible comparisons (every validated comparison).
    pub listwise_groups: usize,
    pub runtime_rows: usize,
}

impl Denominators {
    /// Denominators in [`objective::TERM_NAMES`] order.
    pub fn by_term(&self) -> [f64; 5] {
        [
            self.sequence_weight,
            self.pair_weight,
            self.listwise_groups as f64,
            self.runtime_rows as f64,
            self.pair_weight,
        ]
    }

    /// Terms with a zero global denominator: value 0, gradient 0.
    pub fn empty_terms(&self) -> Vec<String> {
        objective::TERM_NAMES
            .iter()
            .zip(self.by_term())
            .filter(|(_, d)| *d == 0.0)
            .map(|(name, _)| name.to_string())
            .collect()
    }

    /// `D_T,local / D_T` for every term. A zero global denominator gives 0 and
    /// requires a zero local one; a local denominator above the global one
    /// means the global denominators do not cover this microbatch.
    pub fn fractions_of(&self, global: &Denominators) -> Result<[f64; 5]> {
        let (local, global) = (self.by_term(), global.by_term());
        let mut fractions = [0.0; 5];
        for i in 0..5 {
            let (l, g) = (local[i], global[i]);
            ensure!(
                l.is_finite() && g.is_finite() && l >= 0.0 && g >= 0.0,
                "{} denominator is not a finite non-negative sum",
                objective::TERM_NAMES[i]
            );
            ensure!(
                l <= g * (1.0 + DENOMINATOR_SLACK),
                "{} local denominator {l} exceeds the global {g}",
                objective::TERM_NAMES[i]
            );
            fractions[i] = if g == 0.0 { 0.0 } else { (l / g).min(1.0) };
        }
        Ok(fractions)
    }
}

pub fn denominators(rows: &[RowLabels]) -> Result<Denominators> {
    objective::validate_rows(rows)?;
    let mut pair_weight = 0.0;
    for comparison in rows.chunks(COMPARISON_ROWS) {
        for (a, left) in comparison.iter().enumerate() {
            for right in &comparison[a + 1..] {
                if left.value == right.value {
                    continue;
                }
                // Exactly `neural::ranking_losses`: f64 values, boost on the
                // larger f32 percentile.
                let boost =
                    if left.percentile.max(right.percentile) >= crate::neural::BOOST_THRESHOLD {
                        crate::neural::BOOST_FACTOR
                    } else {
                        1.0
                    };
                pair_weight += (f64::from(left.value) - f64::from(right.value)).abs() * boost;
            }
        }
    }
    Ok(Denominators {
        sequence_weight: rows
            .iter()
            .map(|r| f64::from(row_weight(r.percentile)))
            .sum(),
        pair_weight,
        listwise_groups: rows.len() / COMPARISON_ROWS,
        runtime_rows: rows.len(),
    })
}

/// The v2 objective of one rank-local microbatch and its contribution to each
/// global raw term value. Summed over every microbatch of every rank, the
/// objectives give `L` and the contributions give the global term values.
///
/// The term arithmetic is [`objective::terms`], shared with v1; each
/// microbatch-local mean `N_T,mb / D_T,mb` is scaled by `D_T,mb / D_T`.
pub fn composite<'a>(
    outputs: &Outputs<'_>,
    loss_log_vars: impl IntoIterator<Item = &'a Tensor>,
    tokens: &Tensor,
    pad: &Tensor,
    legal: &Tensor,
    rows: &[RowLabels],
    global: &Denominators,
) -> Result<(Tensor, [f64; 5])> {
    let fractions = denominators(rows)?.fractions_of(global)?;
    let objective::Terms { active, stability } =
        objective::terms(outputs, tokens, pad, legal, rows)?;
    let c = COEFFICIENTS;
    let coefficients = [
        c.sequence_nll,
        c.ranking_bt,
        c.ranking_listwise,
        c.runtime_huber,
        c.contextfree_bt,
    ];
    let mut loss = stability * c.stability;
    let mut contributions = [0.0; 5];
    for (i, term) in active.iter().enumerate() {
        let value = term.double_value(&[]);
        ensure!(
            value.is_finite(),
            "nonfinite {} term",
            objective::TERM_NAMES[i]
        );
        loss += term * (coefficients[i] * fractions[i]);
        contributions[i] = value * fractions[i];
    }
    // Inactive optional objectives stay connected for identical packed gradient layouts.
    for x in loss_log_vars {
        loss += x * c.loss_log_vars;
    }
    let total = loss.double_value(&[]);
    ensure!(
        total.is_finite(),
        "nonfinite training objective: total {total}, contributions {contributions:?}"
    );
    Ok((loss, contributions))
}

/// Objective of a rank without comparisons this step: an exact zero
/// connected to every given parameter, so backward defines a zero gradient
/// for each of them.
pub fn empty_rank_objective<'a>(
    parameters: impl IntoIterator<Item = &'a Tensor>,
) -> Result<Tensor> {
    parameters
        .into_iter()
        .map(|p| p.sum(Kind::Float) * 0.0)
        .reduce(|a, b| a + b)
        .context("an empty-rank objective needs the trainable parameters")
}

/// Whether a partitioned objective is within the frozen policy.
pub fn loss_within_policy(actual: f64, reference: f64) -> bool {
    actual.is_finite()
        && reference.is_finite()
        && (actual - reference).abs() <= LOSS_RTOL * reference.abs().max(1.0)
}

/// Whether one output gradient is within the frozen policy.
pub fn gradient_within_policy(actual: &[f64], reference: &[f64]) -> bool {
    let scale = reference.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    actual.len() == reference.len()
        && actual
            .iter()
            .zip(reference)
            .all(|(a, r)| a.is_finite() && (a - r).abs() <= GRADIENT_RTOL * scale)
}

/// Digest of a packed f32 gradient's little-endian bytes.
pub fn gradient_sha256(gradient: &Tensor) -> Result<String> {
    let values = Vec::<f32>::try_from(
        gradient
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Float)
            .flatten(0, -1),
    )?;
    Ok(crate::manifest::hash(
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    ))
}
