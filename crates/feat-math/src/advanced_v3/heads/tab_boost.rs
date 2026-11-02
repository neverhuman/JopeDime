use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BoostStump {
    pub(super) feature: usize,
    pub(super) threshold: f64,
    pub(super) left: f64,
    pub(super) right: f64,
    pub(super) missing_left: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TabBoostModel {
    pub(super) base: f64,
    pub(super) learning_rate: f64,
    pub(super) binary: bool,
    pub(super) stumps: Vec<BoostStump>,
}

impl TabBoostModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let rounds = stage.get_usize("rounds", 100).clamp(1, 500);
        let bins = stage.get_usize("bins", 32).clamp(8, 256);
        let min_leaf = stage.get_usize("min_leaf", 5).clamp(1, 100);
        let learning_rate = stage.get_f64("lr", 0.05).clamp(0.001, 1.0);
        let lambda = stage.get_f64("lambda", 1.0).clamp(0.0, 1e6);
        let loss = str_param(stage, "loss", if binary { "log_loss" } else { "squared" })?;
        if (binary && loss != "log_loss") || (!binary && !matches!(loss, "squared" | "huber")) {
            bail!("tab_boost: loss is log_loss for binary and squared|huber for regression");
        }
        let huber_delta = if loss == "huber" {
            stage.get_f64("delta", 1.0).clamp(1e-6, 1e6)
        } else {
            f64::INFINITY
        };
        let mean = y.iter().take(x.len()).sum::<f64>() / x.len() as f64;
        let base = if binary {
            ((mean.clamp(1e-6, 1.0 - 1e-6)) / (1.0 - mean.clamp(1e-6, 1.0 - 1e-6))).ln()
        } else {
            mean
        };
        let mut raw = vec![base; x.len()];
        let mut stumps = Vec::new();
        for _ in 0..rounds {
            let mut best: Option<(f64, BoostStump)> = None;
            for feature in 0..x[0].len() {
                let mut sorted: Vec<f64> = x
                    .iter()
                    .map(|row| row[feature])
                    .filter(|v| v.is_finite())
                    .collect();
                sorted.sort_by(f64::total_cmp);
                if sorted.len() < 2 * min_leaf {
                    continue;
                }
                for b in 1..bins {
                    let threshold = quantile_sorted(&sorted, b as f64 / bins as f64);
                    for missing_left in [false, true] {
                        let mut gl = 0.0;
                        let mut gr = 0.0;
                        let mut hl = 0.0;
                        let mut hr = 0.0;
                        let mut nl = 0usize;
                        let mut nr = 0usize;
                        for r in 0..x.len() {
                            let probability = sigmoid(raw[r]);
                            let (gradient, hessian) = if binary {
                                (y[r] - probability, probability * (1.0 - probability))
                            } else {
                                ((y[r] - raw[r]).clamp(-huber_delta, huber_delta), 1.0)
                            };
                            let left = if x[r][feature].is_finite() {
                                x[r][feature] <= threshold
                            } else {
                                missing_left
                            };
                            if left {
                                gl += gradient;
                                hl += hessian;
                                nl += 1;
                            } else {
                                gr += gradient;
                                hr += hessian;
                                nr += 1;
                            }
                        }
                        if nl < min_leaf || nr < min_leaf {
                            continue;
                        }
                        let left = gl / (hl + lambda).max(1e-12);
                        let right = gr / (hr + lambda).max(1e-12);
                        let gain =
                            gl * gl / (hl + lambda).max(1e-12) + gr * gr / (hr + lambda).max(1e-12);
                        if best.as_ref().is_none_or(|entry| gain > entry.0) {
                            best = Some((
                                gain,
                                BoostStump {
                                    feature,
                                    threshold,
                                    left,
                                    right,
                                    missing_left,
                                },
                            ));
                        }
                    }
                }
            }
            let Some((_, stump)) = best else { break };
            for (row, prediction) in x.iter().zip(&mut raw) {
                let left = if row[stump.feature].is_finite() {
                    row[stump.feature] <= stump.threshold
                } else {
                    stump.missing_left
                };
                *prediction += learning_rate * if left { stump.left } else { stump.right };
            }
            stumps.push(stump);
        }
        Ok(Self {
            base,
            learning_rate,
            binary,
            stumps,
        })
    }

    pub(super) fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut value = self.base;
                for stump in &self.stumps {
                    let left = if row[stump.feature].is_finite() {
                        row[stump.feature] <= stump.threshold
                    } else {
                        stump.missing_left
                    };
                    value += self.learning_rate * if left { stump.left } else { stump.right };
                }
                if self.binary {
                    sigmoid(value)
                } else {
                    finite(value)
                }
            })
            .collect()
    }
}
