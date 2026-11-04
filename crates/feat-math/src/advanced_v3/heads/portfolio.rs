use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PlsHeadModel {
    pub(crate) projection: V3Op,
    pub(crate) ridge: RidgeModel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ClassificationPortfolioModel {
    pub(crate) members: Vec<V3Head>,
    pub(crate) weights: Vec<f64>,
}

impl ClassificationPortfolioModel {
    pub(crate) fn fit(x: &Rows, y: &[f64]) -> Result<Self> {
        validate_binary_fit_target(y, x.len())?;
        let stages = [
            Stage::new("logit_ridge"),
            Stage::new("tab_boost"),
            Stage::new("gam_boost"),
        ];
        let positives = y.iter().filter(|value| **value == 1.0).count();
        let minority = positives.min(y.len() - positives);
        if minority < 2 {
            let members = stages
                .iter()
                .map(|stage| fit_v3_head(stage, x, y, TaskKind::Binary))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Self {
                weights: vec![1.0 / members.len() as f64; members.len()],
                members,
            });
        }
        let folds = 5.min(minority);
        let assignments = stratified_fold_assignments(y, folds);
        let mut columns = vec![vec![0.5; x.len()]; stages.len()];
        for fold in 0..folds {
            let fit_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] != fold).collect();
            let valid_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] == fold).collect();
            let x_fit: Rows = fit_rows.iter().map(|&r| x[r].clone()).collect();
            let y_fit: Vec<f64> = fit_rows.iter().map(|&r| y[r]).collect();
            let x_valid: Rows = valid_rows.iter().map(|&r| x[r].clone()).collect();
            for (member, stage) in stages.iter().enumerate() {
                let fitted = fit_v3_head(stage, &x_fit, &y_fit, TaskKind::Binary)?;
                for (&row, value) in valid_rows.iter().zip(fitted.predict(&x_valid)?) {
                    columns[member][row] = value;
                }
            }
        }
        let mut weights = vec![1.0 / stages.len() as f64; stages.len()];
        for _ in 0..200 {
            let mut gradient = vec![0.0; stages.len()];
            for row in 0..x.len() {
                let blend = (0..stages.len())
                    .map(|member| weights[member] * columns[member][row])
                    .sum::<f64>()
                    .clamp(1e-6, 1.0 - 1e-6);
                let common = (blend - y[row]) / (blend * (1.0 - blend) * x.len() as f64);
                for member in 0..stages.len() {
                    gradient[member] += common * columns[member][row];
                }
            }
            for member in 0..weights.len() {
                weights[member] *= (-0.05 * gradient[member].clamp(-20.0, 20.0)).exp();
            }
            let sum = weights.iter().sum::<f64>().max(1e-24);
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        let members = stages
            .iter()
            .map(|stage| fit_v3_head(stage, x, y, TaskKind::Binary))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { members, weights })
    }

    pub(crate) fn predict(&self, x: &Rows) -> Result<Vec<f64>> {
        let columns: Vec<Vec<f64>> = self
            .members
            .iter()
            .map(|member| member.predict(x))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..x.len())
            .map(|row| {
                self.weights
                    .iter()
                    .zip(&columns)
                    .map(|(weight, predictions)| weight * predictions[row])
                    .sum::<f64>()
                    .clamp(0.0, 1.0)
            })
            .collect())
    }
}

impl PlsHeadModel {
    pub(crate) fn fit(x: &Rows, y: &[f64], k: usize) -> Self {
        let (means, weights) = fit_pls_weights(x, y, k);
        let projection = V3Op::PlsProject {
            means,
            weights,
            append: false,
        };
        let projected = projection.apply(x);
        let ridge = RidgeModel::fit(&projected, y, 0.02);
        Self { projection, ridge }
    }

    pub(crate) fn predict(&self, x: &Rows) -> Vec<f64> {
        self.ridge.predict(&self.projection.apply(x))
    }
}
