use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassRidgeModel {
    pub(super) classes: Vec<i64>,
    pub(super) feature_count: usize,
    pub(super) means: Vec<f64>,
    pub(super) scales: Vec<f64>,
    pub(super) weights: Vec<Vec<f64>>,
}

impl MulticlassRidgeModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        if x.first().is_none_or(Vec::is_empty) || x.iter().any(|row| row.len() != x[0].len()) {
            bail!("multiclass fits require a nonempty rectangular feature matrix");
        }
        let class_weight = str_param(stage, "class_weight", "balanced")?;
        if class_weight != "balanced" {
            bail!("multiclass_ridge: class_weight must be balanced");
        }
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut means = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                means[feature] += finite(row[feature]) / x.len() as f64;
            }
        }
        let mut scales = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - means[feature];
                scales[feature] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / x.len() as f64).sqrt().max(1e-12);
        }

        let mut class_counts = vec![0usize; classes.len()];
        let class_indices: Vec<usize> = y
            .iter()
            .map(|value| {
                classes
                    .binary_search(&(*value as i64))
                    .expect("validated class is present")
            })
            .collect();
        for &class in &class_indices {
            class_counts[class] += 1;
        }

        let width = feature_count + 1;
        let mut gram = vec![vec![0.0; width]; width];
        let mut rhs = vec![vec![0.0; width]; classes.len()];
        for (row, &class) in x.iter().zip(&class_indices) {
            let sample_weight = x.len() as f64 / (classes.len() * class_counts[class]) as f64;
            let mut values = Vec::with_capacity(width);
            values.push(1.0);
            values.extend((0..feature_count).map(|feature| {
                ((finite(row[feature]) - means[feature]) / scales[feature]).clamp(-30.0, 30.0)
            }));
            // Lower-triangular Gram accumulation: `right` runs to `left` inclusive, and both
            // index `values` as well as `gram`. The triangle is the point.
            #[allow(clippy::needless_range_loop)]
            for left in 0..width {
                for right in 0..=left {
                    gram[left][right] += sample_weight * values[left] * values[right];
                }
                rhs[class][left] += sample_weight * values[left];
            }
        }
        // Mirror the lower triangle into the upper one. Reading and writing two different
        // cells of the same matrix is what this does; an iterator form cannot express it.
        #[allow(clippy::needless_range_loop)]
        for left in 0..width {
            for right in 0..left {
                gram[right][left] = gram[left][right];
            }
        }
        let l2 = stage.get_f64("l2", 0.02).clamp(1e-9, 1e6) * x.len() as f64;
        gram[0][0] += 1e-9;
        for (feature, row) in gram.iter_mut().enumerate().take(width).skip(1) {
            row[feature] += l2;
        }
        let weights = rhs
            .iter()
            .map(|target| {
                solve_positive_definite(&gram, target).ok_or_else(|| {
                    anyhow::anyhow!("multiclass ridge normal equations are singular")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            classes,
            feature_count,
            means,
            scales,
            weights,
        })
    }

    pub(super) fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let mut scores = Vec::with_capacity(self.classes.len());
                for weights in &self.weights {
                    let mut score = weights[0];
                    for feature in 0..self.feature_count.min(row.len()) {
                        let value = ((finite(row[feature]) - self.means[feature])
                            / self.scales[feature])
                            .clamp(-30.0, 30.0);
                        score += weights[feature + 1] * value;
                    }
                    scores.push(finite(score));
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut probabilities: Vec<f64> = scores
                    .iter()
                    .map(|score| (score - maximum).clamp(-40.0, 0.0).exp())
                    .collect();
                let sum = probabilities.iter().sum::<f64>().max(1e-300);
                for probability in &mut probabilities {
                    *probability /= sum;
                }
                probabilities
            })
            .collect()
    }

    pub(super) fn labels(&self, x: &Rows) -> Vec<i64> {
        self.probabilities(x)
            .iter()
            .map(|row| {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map_or(0, |(index, _)| index);
                self.classes[index]
            })
            .collect()
    }
}

fn multiclass_feature_budget(stage: &Stage) -> Result<usize> {
    let budget = stage.get_usize("feature_budget", 64);
    if !MULTICLASS_FEATURE_BUDGETS.contains(&budget) {
        bail!("{}: feature_budget must be one of 32|64|128|256", stage.op);
    }
    Ok(budget)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassOvrLogisticModel {
    pub(super) classes: Vec<i64>,
    pub(super) models: Vec<LogitRidgeModel>,
}

impl MulticlassOvrLogisticModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let bounded = x
            .iter()
            .map(|row| row[..feature_count].to_vec())
            .collect::<Rows>();
        let l2 = stage.get_f64("l2", 0.02).clamp(1e-9, 1e6);
        let models = classes
            .iter()
            .map(|class| {
                let binary = y
                    .iter()
                    .map(|value| f64::from(*value as i64 == *class))
                    .collect::<Vec<_>>();
                LogitRidgeModel::fit_ovr(&bounded, &binary, l2)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { classes, models })
    }

    pub(super) fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        let columns = self
            .models
            .iter()
            .map(|model| model.predict(x))
            .collect::<Vec<_>>();
        (0..x.len())
            .map(|row| {
                let mut values = columns
                    .iter()
                    .map(|column| finite(column[row]).max(0.0))
                    .collect::<Vec<_>>();
                let sum = values.iter().sum::<f64>();
                if !sum.is_finite() || sum <= 0.0 {
                    let uniform = 1.0 / values.len() as f64;
                    values.fill(uniform);
                } else {
                    for value in &mut values {
                        *value /= sum;
                    }
                }
                values
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassDiagonalLdaModel {
    pub(super) classes: Vec<i64>,
    pub(super) feature_count: usize,
    pub(super) means: Vec<Vec<f64>>,
    pub(super) variances: Vec<f64>,
    pub(super) log_priors: Vec<f64>,
}

impl MulticlassDiagonalLdaModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut means = vec![vec![0.0; feature_count]; classes.len()];
        let mut counts = vec![0usize; classes.len()];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            counts[class] += 1;
            for feature in 0..feature_count {
                means[class][feature] += finite(row[feature]);
            }
        }
        for (mean, count) in means.iter_mut().zip(&counts) {
            for value in mean {
                *value /= *count as f64;
            }
        }
        let mut variances = vec![0.0; feature_count];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - means[class][feature];
                variances[feature] += delta * delta;
            }
        }
        let shrinkage = stage.get_f64("shrinkage", 0.01).clamp(0.0, 1.0);
        let global = (variances.iter().sum::<f64>()
            / (feature_count.max(1) * x.len().max(1)) as f64)
            .max(1e-12);
        for variance in &mut variances {
            *variance =
                ((1.0 - shrinkage) * (*variance / x.len() as f64) + shrinkage * global).max(1e-12);
        }
        let log_priors = counts
            .iter()
            .map(|count| (*count as f64 / x.len() as f64).ln())
            .collect();
        Ok(Self {
            classes,
            feature_count,
            means,
            variances,
            log_priors,
        })
    }

    pub(super) fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let scores = self
                    .means
                    .iter()
                    .zip(&self.log_priors)
                    .map(|(mean, prior)| {
                        *prior
                            - 0.5
                                * (0..self.feature_count)
                                    .map(|feature| {
                                        let delta = finite(row[feature]) - mean[feature];
                                        delta * delta / self.variances[feature]
                                    })
                                    .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                softmax(&scores)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassNearestCentroidModel {
    pub(super) classes: Vec<i64>,
    pub(super) feature_count: usize,
    pub(super) centroids: Vec<Vec<f64>>,
    pub(super) scales: Vec<f64>,
}

impl MulticlassNearestCentroidModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut centroids = vec![vec![0.0; feature_count]; classes.len()];
        let mut counts = vec![0usize; classes.len()];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            counts[class] += 1;
            for feature in 0..feature_count {
                centroids[class][feature] += finite(row[feature]);
            }
        }
        for (centroid, count) in centroids.iter_mut().zip(&counts) {
            for value in centroid {
                *value /= *count as f64;
            }
        }
        let mut global = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                global[feature] += finite(row[feature]) / x.len() as f64;
            }
        }
        let mut scales = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - global[feature];
                scales[feature] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / x.len() as f64).sqrt().max(1e-12);
        }
        Ok(Self {
            classes,
            feature_count,
            centroids,
            scales,
        })
    }

    pub(super) fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let scores = self
                    .centroids
                    .iter()
                    .map(|centroid| {
                        -(0..self.feature_count)
                            .map(|feature| {
                                let delta = (finite(row[feature]) - centroid[feature])
                                    / self.scales[feature];
                                delta * delta
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                softmax(&scores)
            })
            .collect()
    }
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = scores
        .iter()
        .map(|score| (finite(*score) - maximum).clamp(-40.0, 0.0).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>().max(1e-300);
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

fn solve_positive_definite(matrix: &[Vec<f64>], target: &[f64]) -> Option<Vec<f64>> {
    let size = matrix.len();
    if size == 0 || target.len() != size {
        return None;
    }
    for attempt in 0..6 {
        let jitter = if attempt == 0 {
            0.0
        } else {
            10f64.powi(attempt - 12)
        };
        let mut lower = vec![vec![0.0; size]; size];
        let mut valid = true;
        for row in 0..size {
            for column in 0..=row {
                let mut value = matrix[row][column];
                if row == column {
                    value += jitter;
                }
                // Cholesky inner product: `prior` indexes two different rows of the same
                // triangular factor. This is the textbook recurrence, written as such.
                #[allow(clippy::needless_range_loop)]
                for prior in 0..column {
                    value -= lower[row][prior] * lower[column][prior];
                }
                if row == column {
                    if !value.is_finite() || value <= 1e-18 {
                        valid = false;
                        break;
                    }
                    lower[row][column] = value.sqrt();
                } else {
                    lower[row][column] = value / lower[column][column];
                }
            }
            if !valid {
                break;
            }
        }
        if !valid {
            continue;
        }
        let mut forward = vec![0.0; size];
        for row in 0..size {
            let residual = target[row]
                - (0..row)
                    .map(|column| lower[row][column] * forward[column])
                    .sum::<f64>();
            forward[row] = residual / lower[row][row];
        }
        let mut solution = vec![0.0; size];
        for row in (0..size).rev() {
            let residual = forward[row]
                - ((row + 1)..size)
                    .map(|column| lower[column][row] * solution[column])
                    .sum::<f64>();
            solution[row] = residual / lower[row][row];
        }
        if solution.iter().all(|value| value.is_finite()) {
            return Some(solution);
        }
    }
    None
}
