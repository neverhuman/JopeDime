use super::Trie;
use crate::canon::is_hex64;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `LogitsRow` is a public logits row record.
pub struct LogitsRow {
    /// Collection of prefix values carried by this logits row.
    pub prefix: Vec<u16>,
    /// Collection of logits values carried by this logits row.
    pub logits: Vec<f64>,
}

#[derive(Debug, Serialize)]
/// `Distribution` is a public distribution record.
pub struct Distribution {
    /// Collection of choices values carried by this distribution.
    pub choices: Vec<u16>,
    /// Collection of probabilities values carried by this distribution.
    pub probabilities: Vec<f64>,
    /// Collection of log probabilities values carried by this distribution.
    pub log_probabilities: Vec<f64>,
}

#[derive(Debug, Serialize)]
/// `Likelihood` is a public likelihood record.
pub struct Likelihood {
    /// SHA-256 digest identifying the associated support.
    pub support_sha256: String,
    /// Aggregation recorded for this likelihood.
    pub aggregation: &'static str,
    /// Log probability recorded for this likelihood.
    pub log_probability: f64,
    /// Collection of per token values carried by this likelihood.
    pub per_token: Vec<f64>,
}

/// Immutable CLOVER sampler/scorer settings pinned to the exact support and vocabulary lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct CloverBehaviorLaw {
    support_sha256: String,
    vocabulary_sha256: String,
    temperature: f64,
}

impl CloverBehaviorLaw {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(trie: &Trie<'_>, vocabulary_sha256: &str, temperature: f64) -> Result<Self> {
        ensure!(
            is_hex64(trie.support_sha256)
                && is_hex64(vocabulary_sha256)
                && trie.vocabulary_sha256 == vocabulary_sha256,
            "CLOVER support/vocabulary identity binding required"
        );
        ensure!(
            temperature.is_finite() && (0.1..=2.0).contains(&temperature),
            "CLOVER decoder temperature outside .1..2"
        );
        Ok(Self {
            support_sha256: trie.support_sha256.to_owned(),
            vocabulary_sha256: vocabulary_sha256.to_owned(),
            temperature,
        })
    }

    /// Performs the distribution operation and returns an error if it cannot be completed.
    pub fn distribution(
        &self,
        trie: &Trie<'_>,
        prefix: &[u16],
        logits: &[f64],
    ) -> Result<Distribution> {
        self.bind_trie(trie)?;
        Distribution::from_logits(trie, prefix, logits, self.temperature)
    }

    /// Performs the teacher forced operation and returns an error if it cannot be completed.
    pub fn teacher_forced(
        &self,
        trie: &Trie<'_>,
        tokens: &[u16],
        logits: &[LogitsRow],
    ) -> Result<Likelihood> {
        self.bind_trie(trie)?;
        Likelihood::teacher_forced(trie, tokens, logits, self.temperature)
    }

    fn bind_trie(&self, trie: &Trie<'_>) -> Result<()> {
        ensure!(
            trie.support_sha256 == self.support_sha256
                && trie.vocabulary_sha256 == self.vocabulary_sha256,
            "CLOVER behavior law support/vocabulary changed"
        );
        Ok(())
    }
}

impl Distribution {
    /// Performs the from logits operation and returns an error if it cannot be completed.
    pub fn from_logits(
        trie: &Trie<'_>,
        prefix: &[u16],
        logits: &[f64],
        temperature: f64,
    ) -> Result<Self> {
        ensure!(
            temperature.is_finite() && (0.1..=2.).contains(&temperature),
            "decoder temperature outside .1..2"
        );
        ensure!(logits.len() == 282, "deployed logit geometry required");
        let choices = trie.children(prefix)?;
        ensure!(!choices.is_empty(), "terminal EOS has no distribution");
        let supported: Vec<_> = choices
            .iter()
            .map(|&t| logits[t as usize] / temperature)
            .collect();
        ensure!(
            supported.iter().all(|v| v.is_finite()),
            "nonfinite supported logits"
        );
        let max = supported.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<_> = supported.iter().map(|v| v - max).collect();
        ensure!(
            shifted.iter().all(|v| v.is_finite()),
            "logit difference overflow"
        );
        let sum: f64 = shifted.iter().map(|v| v.exp()).sum();
        let log_sum = sum.ln();
        let log_probabilities: Vec<_> = shifted.iter().map(|v| v - log_sum).collect();
        let probabilities = shifted.iter().map(|v| v.exp() / sum).collect();
        Ok(Self {
            choices,
            probabilities,
            log_probabilities,
        })
    }

    /// The caller supplies a per-row uniform RNG stream. Sampling and teacher
    /// forcing use this same mask and normalization, including EOS.
    pub fn select(&self, uniform: f64) -> Result<(u16, f64)> {
        ensure!(
            uniform.is_finite() && (0.0..1.0).contains(&uniform),
            "uniform must be in [0,1)"
        );
        let mut sum = 0.;
        for (i, p) in self.probabilities.iter().enumerate() {
            sum += p;
            if uniform < sum {
                return Ok((self.choices[i], self.log_probabilities[i]));
            }
        }
        let i = self.choices.len() - 1;
        Ok((self.choices[i], self.log_probabilities[i]))
    }
}

impl Likelihood {
    /// Performs the teacher forced operation and returns an error if it cannot be completed.
    pub fn teacher_forced(
        trie: &Trie<'_>,
        tokens: &[u16],
        logits: &[LogitsRow],
        temperature: f64,
    ) -> Result<Self> {
        trie.validate_leaf(tokens)?;
        ensure!(
            logits.len() == tokens.len() - 1,
            "one logit row per predicted token required"
        );
        let mut per_token = Vec::new();
        for (pos, row) in logits.iter().enumerate() {
            ensure!(row.prefix == tokens[..pos + 1], "teacher prefix mismatch");
            let distribution =
                Distribution::from_logits(trie, &row.prefix, &row.logits, temperature)?;
            let chosen = distribution
                .choices
                .binary_search(&tokens[pos + 1])
                .map_err(|_| anyhow::anyhow!("unsupported raw action"))?;
            per_token.push(distribution.log_probabilities[chosen]);
        }
        let log_probability: f64 = per_token.iter().sum();
        ensure!(
            log_probability.is_finite(),
            "finite sequence likelihood required"
        );
        Ok(Self {
            support_sha256: trie.support_sha256.into(),
            aggregation: "sum_including_eos",
            log_probability,
            per_token,
        })
    }
}

/// Performs the weighted archive CE operation and returns an error if it cannot be completed.
pub fn weighted_archive_ce(log_probabilities: &[f64], weights: &[f64]) -> Result<f64> {
    ensure!(
        !weights.is_empty() && weights.len() == log_probabilities.len(),
        "aligned archive weights required"
    );
    ensure!(
        weights.iter().all(|x| x.is_finite() && *x >= 0.)
            && log_probabilities.iter().all(|x| x.is_finite() && *x <= 0.),
        "finite probabilities and nonnegative fixed weights required"
    );
    ensure!(
        (weights.iter().sum::<f64>() - 1.).abs() <= 1e-12,
        "unit mass archive weights required"
    );
    Ok(-weights
        .iter()
        .zip(log_probabilities)
        .map(|(w, p)| w * p)
        .sum::<f64>())
}

/// Exact KL decomposition for support-expanding (`N -> N+1`) policy updates (`J13` / `EI-1b`).
///
/// By the KL chain rule over the partition `S = S_base U S_new`:
/// `total_kl = bernoulli_allocation_kl + base_support_mass * conditional_base_kl + new_support_mass * conditional_new_kl`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecomposedSupportKl {
    /// KL(pi_theta(. | S_base) || pi_ref(. | S_base)) over the base catalog leaves.
    pub conditional_base_kl: f64,
    /// KL(pi_theta(. | S_new) || pi_ref(. | S_new)) over the newly admitted leaves (0 when |S_new| <= 1).
    pub conditional_new_kl: f64,
    /// Total probability mass allocated by pi_theta to base catalog leaves `S_base`.
    pub base_support_mass: f64,
    /// Total probability mass allocated by pi_theta to newly admitted `N+1` leaves `S_new`.
    pub new_support_mass: f64,
    /// Total probability mass allocated by pi_ref to newly admitted `N+1` leaves `S_new`.
    pub ref_new_support_mass: f64,
    /// Binary KL(Ber(new_support_mass) || Ber(ref_new_support_mass)).
    pub bernoulli_allocation_kl: f64,
    /// Exact total KL(pi_theta || pi_ref) over the full `N+1` support.
    pub total_kl: f64,
    /// True iff `conditional_base_kl <= max_conditional_base_kl + 1e-12` and `new_support_mass <= max_new_support_mass + 1e-12`.
    pub admissible: bool,
}

const DECOMPOSED_KL_BOUND_TOL: f64 = 1e-12;

/// Performs the decomposed support KL operation and returns an error if it cannot be completed.
pub fn decomposed_support_kl(
    policy_leaf_logprobs: &[f64],
    ref_leaf_logprobs: &[f64],
    is_base_leaf: &[bool],
    max_conditional_base_kl: f64,
    max_new_support_mass: f64,
) -> Result<DecomposedSupportKl> {
    let n = policy_leaf_logprobs.len();
    ensure!(
        n > 0 && ref_leaf_logprobs.len() == n && is_base_leaf.len() == n,
        "aligned non-empty policy, reference, and base-mask arrays required"
    );
    ensure!(
        is_base_leaf.iter().any(|&b| b),
        "at least one base support leaf required"
    );
    ensure!(
        policy_leaf_logprobs.iter().all(|x| x.is_finite())
            && ref_leaf_logprobs.iter().all(|x| x.is_finite()),
        "finite leaf log-probabilities required"
    );
    ensure!(
        max_conditional_base_kl.is_finite()
            && max_conditional_base_kl >= 0.0
            && max_new_support_mass.is_finite()
            && (0.0..=1.0).contains(&max_new_support_mass),
        "valid KL and new-support mass bounds required"
    );

    let log_sum_exp = |xs: &[f64]| -> f64 {
        let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
    };
    let z_p = log_sum_exp(policy_leaf_logprobs);
    let z_q = log_sum_exp(ref_leaf_logprobs);

    let mut base_raw_p = Vec::new();
    let mut base_raw_q = Vec::new();
    let mut new_raw_p = Vec::new();
    let mut new_raw_q = Vec::new();
    let mut total_kl = 0.0_f64;

    for i in 0..n {
        let lp = policy_leaf_logprobs[i] - z_p;
        let lq = ref_leaf_logprobs[i] - z_q;
        let p = lp.exp();
        if p > 0.0 {
            total_kl += p * (lp - lq);
        }
        if is_base_leaf[i] {
            base_raw_p.push(policy_leaf_logprobs[i]);
            base_raw_q.push(ref_leaf_logprobs[i]);
        } else {
            new_raw_p.push(policy_leaf_logprobs[i]);
            new_raw_q.push(ref_leaf_logprobs[i]);
        }
    }
    total_kl = total_kl.max(0.0);

    let z_p_base = log_sum_exp(&base_raw_p);
    let z_q_base = log_sum_exp(&base_raw_q);
    let mut conditional_base_kl = 0.0_f64;
    for (&rp, &rq) in base_raw_p.iter().zip(base_raw_q.iter()) {
        let cond_lp = rp - z_p_base;
        let cond_lq = rq - z_q_base;
        let cond_p = cond_lp.exp();
        if cond_p > 0.0 {
            conditional_base_kl += cond_p * (cond_lp - cond_lq);
        }
    }
    conditional_base_kl = conditional_base_kl.max(0.0);

    let (conditional_new_kl, p_base, p_new, q_new, bernoulli_allocation_kl) =
        if new_raw_p.is_empty() {
            (0.0, 1.0, 0.0, 0.0, 0.0)
        } else {
            let z_p_new = log_sum_exp(&new_raw_p);
            let z_q_new = log_sum_exp(&new_raw_q);
            let mut cond_new_kl = 0.0_f64;
            for (&rp, &rq) in new_raw_p.iter().zip(new_raw_q.iter()) {
                let cond_lp = rp - z_p_new;
                let cond_lq = rq - z_q_new;
                let cond_p = cond_lp.exp();
                if cond_p > 0.0 {
                    cond_new_kl += cond_p * (cond_lp - cond_lq);
                }
            }
            cond_new_kl = cond_new_kl.max(0.0);

            let log_p_base = z_p_base - z_p;
            let log_q_base = z_q_base - z_q;
            let log_p_new = z_p_new - z_p;
            let log_q_new = z_q_new - z_q;

            let p_b = log_p_base.exp().clamp(0.0, 1.0);
            let p_n = log_p_new.exp().clamp(0.0, 1.0);
            let q_n = log_q_new.exp().clamp(0.0, 1.0);

            let term_base = if p_b > 0.0 {
                p_b * (log_p_base - log_q_base)
            } else {
                0.0
            };
            let term_new = if p_n > 0.0 {
                p_n * (log_p_new - log_q_new)
            } else {
                0.0
            };
            let bern_kl = (term_base + term_new).max(0.0);
            (cond_new_kl, p_b, p_n, q_n, bern_kl)
        };

    let admissible = conditional_base_kl <= max_conditional_base_kl + DECOMPOSED_KL_BOUND_TOL
        && p_new <= max_new_support_mass + DECOMPOSED_KL_BOUND_TOL;

    Ok(DecomposedSupportKl {
        conditional_base_kl,
        conditional_new_kl,
        base_support_mass: p_base,
        new_support_mass: p_new,
        ref_new_support_mass: q_new,
        bernoulli_allocation_kl,
        total_kl,
        admissible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposed_kl_distinguishes_new_winner_allocation_from_base_drift() {
        let p_base_each: f64 = 0.96 / 4.0;
        let p_new: f64 = 0.04;
        let q_base_each: f64 = 0.9999 / 4.0;
        let q_new: f64 = 0.0001;

        let policy_lp = vec![
            p_base_each.ln(),
            p_base_each.ln(),
            p_base_each.ln(),
            p_base_each.ln(),
            p_new.ln(),
        ];
        let ref_lp = vec![
            q_base_each.ln(),
            q_base_each.ln(),
            q_base_each.ln(),
            q_base_each.ln(),
            q_new.ln(),
        ];
        let is_base = vec![true, true, true, true, false];

        let report = decomposed_support_kl(&policy_lp, &ref_lp, &is_base, 0.05, 0.15).unwrap();
        assert!(report.conditional_base_kl < 1e-12);
        assert!(report.conditional_new_kl < 1e-12);
        assert!((report.base_support_mass - 0.96).abs() < 1e-12);
        assert!((report.new_support_mass - 0.04).abs() < 1e-12);
        assert!(report.total_kl > 0.15);
        assert!((report.total_kl - report.bernoulli_allocation_kl).abs() < 1e-12);
        assert!(report.admissible);

        // Verify exact chain rule with multiple non-uniform new leaves and boundary tolerance:
        let policy_multi: Vec<f64> = vec![
            (0.85_f64 * 0.6).ln(),
            (0.85_f64 * 0.4).ln(),
            (0.15_f64 / 3.0).ln(),
            (0.15_f64 / 3.0).ln(),
            (0.15_f64 / 3.0).ln(),
        ];
        let ref_multi: Vec<f64> = vec![
            (0.99_f64 * 0.5).ln(),
            (0.99_f64 * 0.5).ln(),
            (0.01_f64 * 0.2).ln(),
            (0.01_f64 * 0.3).ln(),
            (0.01_f64 * 0.5).ln(),
        ];
        let is_base_multi = vec![true, true, false, false, false];
        let multi =
            decomposed_support_kl(&policy_multi, &ref_multi, &is_base_multi, 0.05, 0.15).unwrap();
        let chain_sum = multi.bernoulli_allocation_kl
            + multi.base_support_mass * multi.conditional_base_kl
            + multi.new_support_mass * multi.conditional_new_kl;
        assert!((multi.total_kl - chain_sum).abs() < 1e-12);
        assert!(multi.admissible);
    }

    #[test]
    fn decomposed_kl_rejects_excessive_base_drift_or_excessive_new_mass() {
        let policy_lp = vec![0.85_f64.ln(), 0.05_f64.ln(), 0.10_f64.ln()];
        let ref_lp = vec![0.45_f64.ln(), 0.45_f64.ln(), 0.10_f64.ln()];
        let is_base = vec![true, true, false];

        let report = decomposed_support_kl(&policy_lp, &ref_lp, &is_base, 0.05, 0.15).unwrap();
        assert!(report.conditional_base_kl > 0.20);
        assert!(!report.admissible);
    }
}
