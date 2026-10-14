use super::Trie;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogitsRow {
    pub prefix: Vec<u16>,
    pub logits: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub choices: Vec<u16>,
    pub probabilities: Vec<f64>,
    pub log_probabilities: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct Likelihood {
    pub support_sha256: String,
    pub aggregation: &'static str,
    pub log_probability: f64,
    pub per_token: Vec<f64>,
}

impl Distribution {
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
    pub fn teacher_forced(
        trie: &Trie<'_>,
        tokens: &[u16],
        logits: &[LogitsRow],
        temperature: f64,
    ) -> Result<Self> {
        ensure!(
            trie.contains(tokens) && tokens.last() == Some(&2),
            "complete registered EOS leaf required"
        );
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
