use super::*;

fn validation_error(data: &Prepared, predictions: &[f64]) -> f64 {
    let mut loss = 0.0;
    for i in data.fit_n..data.n {
        let delta = data.y[i] - predictions[i];
        loss += delta * delta;
    }
    loss / (data.n - data.fit_n) as f64
}

pub(super) fn choose(data: &Prepared, seed: u64) -> Choice {
    let penalties = [0.0001, 0.01, 1.0, 100.0];
    let mut best_penalty = penalties[0];
    let mut best_loss = f64::INFINITY;
    let mut baseline = Vec::new();
    for penalty in penalties {
        let model = Linear::learn(data, penalty);
        let predictions = model.prepared_predictions(data);
        let loss = validation_error(data, &predictions);
        if loss < best_loss {
            best_loss = loss;
            best_penalty = penalty;
            baseline = predictions;
        }
    }
    let mut choice = Choice {
        penalty: best_penalty,
        trees: 0,
        mixture: 0.0,
    };
    if baseline.is_empty() {
        return choice;
    }
    let mut predictions = baseline.clone();
    let mut residual = vec![0.0; data.fit_n];
    let mut rng = Random::new(seed ^ 0x94d049bb133111eb);
    let config = growth(data.fit_n);
    let limit = tree_limit(data);
    let mut last_improvement = 0;
    for t in 0..limit {
        for i in 0..data.fit_n {
            residual[i] = data.y[i] - predictions[i];
        }
        let tree = Tree::learn(data, &residual, &mut rng, &config);
        for (i, prediction) in predictions.iter_mut().enumerate() {
            *prediction += tree.predict_prepared(data, i);
        }
        if (t + 1) % 8 != 0 && t + 1 != limit {
            continue;
        }
        let mut dot = 0.0;
        let mut norm = 0.0;
        for i in data.fit_n..data.n {
            let delta = predictions[i] - baseline[i];
            dot += delta * (data.y[i] - baseline[i]);
            norm += delta * delta;
        }
        let mixture = (dot / (norm + 1e-24)).clamp(0.0, 1.0);
        let mut loss = 0.0;
        for i in data.fit_n..data.n {
            let prediction = baseline[i] + mixture * (predictions[i] - baseline[i]);
            let delta = data.y[i] - prediction;
            loss += delta * delta;
        }
        loss /= (data.n - data.fit_n) as f64;
        if loss < best_loss * (1.0 - 1e-5) {
            best_loss = loss;
            choice.trees = t + 1;
            choice.mixture = mixture;
            last_improvement = t + 1;
        }
        if t + 1 >= 96 && t + 1 >= last_improvement + 64 {
            break;
        }
    }
    choice
}

pub(super) fn fit_final(data: Prepared, width: usize, choice: Choice, seed: u64) -> Fitted {
    let linear = Linear::learn(&data, choice.penalty);
    let mut predictions = linear.prepared_predictions(&data);
    let mut residual = vec![0.0; data.fit_n];
    let config = growth(data.fit_n);
    let mut rng = Random::new(seed ^ 0x94d049bb133111eb);
    let mut trees = Vec::new();
    for _ in 0..choice.trees {
        for i in 0..data.fit_n {
            residual[i] = data.y[i] - predictions[i];
        }
        let tree = Tree::learn(&data, &residual, &mut rng, &config);
        for (i, prediction) in predictions.iter_mut().enumerate() {
            *prediction += tree.predict_prepared(&data, i);
        }
        trees.push(tree);
    }
    Fitted {
        width,
        features: data.features,
        scale: data.scale,
        linear,
        trees,
        mixture: choice.mixture,
    }
}
