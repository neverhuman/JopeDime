use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GamTerm {
    pub(crate) feature: usize,
    pub(crate) edges: Vec<f64>,
    pub(crate) values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct GamBoostModel {
    pub(crate) base: f64,
    pub(crate) learning_rate: f64,
    pub(crate) binary: bool,
    pub(crate) terms: Vec<GamTerm>,
}

impl GamBoostModel {
    pub(crate) fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let rounds = stage.get_usize("rounds", 100).clamp(1, 500);
        let bins = stage.get_usize("bins", 16).clamp(4, 64);
        let learning_rate = stage.get_f64("lr", 0.05).clamp(0.001, 1.0);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let base = if binary {
            (mean.clamp(1e-6, 1.0 - 1e-6) / (1.0 - mean.clamp(1e-6, 1.0 - 1e-6))).ln()
        } else {
            mean
        };
        let mut raw = vec![base; x.len()];
        let mut terms = Vec::new();
        let mut all_edges = Vec::new();
        for feature in 0..x[0].len() {
            let mut sorted: Vec<f64> = x.iter().map(|row| row[feature]).collect();
            sorted.sort_by(f64::total_cmp);
            let edges: Vec<f64> = (1..bins)
                .map(|b| quantile_sorted(&sorted, b as f64 / bins as f64))
                .collect();
            all_edges.push(edges);
        }
        for _ in 0..rounds {
            let residuals: Vec<f64> = (0..x.len())
                .map(|r| {
                    if binary {
                        y[r] - sigmoid(raw[r])
                    } else {
                        y[r] - raw[r]
                    }
                })
                .collect();
            let mut best: Option<(f64, GamTerm)> = None;
            for (feature, edges) in all_edges.iter().enumerate() {
                let mut sums = vec![0.0; bins];
                let mut counts = vec![0usize; bins];
                for r in 0..x.len() {
                    let bin = edges
                        .partition_point(|edge| *edge < x[r][feature])
                        .min(bins - 1);
                    sums[bin] += residuals[r];
                    counts[bin] += 1;
                }
                let values: Vec<f64> = sums
                    .iter()
                    .zip(&counts)
                    .map(|(sum, count)| sum / (*count as f64 + 1.0))
                    .collect();
                let gain = (0..x.len())
                    .map(|r| {
                        let bin = edges
                            .partition_point(|edge| *edge < x[r][feature])
                            .min(bins - 1);
                        let before = residuals[r] * residuals[r];
                        let after = residuals[r] - values[bin];
                        before - after * after
                    })
                    .sum::<f64>();
                if best.as_ref().is_none_or(|entry| gain > entry.0) {
                    best = Some((
                        gain,
                        GamTerm {
                            feature,
                            edges: edges.clone(),
                            values,
                        },
                    ));
                }
            }
            let Some((gain, term)) = best else { break };
            if gain <= 0.0 {
                break;
            }
            for (row, prediction) in x.iter().zip(&mut raw) {
                let bin = term
                    .edges
                    .partition_point(|edge| *edge < row[term.feature])
                    .min(term.values.len() - 1);
                *prediction += learning_rate * term.values[bin];
            }
            terms.push(term);
        }
        Ok(Self {
            base,
            learning_rate,
            binary,
            terms,
        })
    }

    pub(crate) fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut prediction = self.base;
                for term in &self.terms {
                    let bin = term
                        .edges
                        .partition_point(|edge| *edge < row[term.feature])
                        .min(term.values.len() - 1);
                    prediction += self.learning_rate * term.values[bin];
                }
                if self.binary {
                    sigmoid(prediction)
                } else {
                    finite(prediction)
                }
            })
            .collect()
    }
}
