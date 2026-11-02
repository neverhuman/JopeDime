use super::*;

pub fn fit_elo_arrays(
    index_a: &[usize],
    index_b: &[usize],
    outcomes: &[f64],
    method_count: usize,
    initial_ratings: Option<&[f64]>,
    center: f64,
    iterations: usize,
    regularization: f64,
    tolerance: f64,
) -> Fit {
    assert_eq!(index_a.len(), index_b.len());
    assert_eq!(index_a.len(), outcomes.len());
    let mut ratings = match initial_ratings {
        Some(initial) => {
            assert_eq!(initial.len(), method_count);
            initial.to_vec()
        }
        None => vec![center; method_count],
    };
    let scale = elo_scale();
    let mut iterations_run = 0;
    let mut converged = false;
    let mut residual_a = vec![0.0; method_count];
    let mut residual_b = vec![0.0; method_count];
    let mut curvature_a = vec![0.0; method_count];
    let mut curvature_b = vec![0.0; method_count];
    let mut updates = vec![0.0; method_count];
    for _ in 0..iterations {
        residual_a.iter_mut().for_each(|x| *x = 0.0);
        residual_b.iter_mut().for_each(|x| *x = 0.0);
        curvature_a.iter_mut().for_each(|x| *x = 0.0);
        curvature_b.iter_mut().for_each(|x| *x = 0.0);
        // Elementwise numpy: delta, clip, exp, probability, residual, curvature; then
        // `np.bincount(index, weights)` adds sequentially in match order.
        for ((&a, &b), &outcome) in index_a.iter().zip(index_b).zip(outcomes) {
            let delta = (scale * (ratings[a] - ratings[b])).clamp(-700.0, 700.0);
            let probability_a = 1.0 / (1.0 + (-delta).exp());
            let residual = outcome - probability_a;
            let curvature = probability_a * (1.0 - probability_a);
            residual_a[a] += residual;
            residual_b[b] += residual;
            curvature_a[a] += curvature;
            curvature_b[b] += curvature;
        }
        let scale_squared = scale * scale;
        let mut largest_update = 0.0_f64;
        for method in 0..method_count {
            let gradient = scale * (residual_a[method] - residual_b[method]);
            let information = scale_squared * (curvature_a[method] + curvature_b[method]);
            let regularized_gradient = gradient - regularization * (ratings[method] - center);
            let update = (regularized_gradient / (information + regularization))
                .clamp(-UPDATE_CLAMP, UPDATE_CLAMP);
            updates[method] = update;
            largest_update = largest_update.max(update.abs());
        }
        for (rating, update) in ratings.iter_mut().zip(&updates) {
            *rating += update;
        }
        iterations_run += 1;
        if largest_update < tolerance {
            converged = true;
            break;
        }
    }
    Fit {
        ratings,
        iterations_run,
        converged,
    }
}

/// numpy's `pairwise_sum` for a contiguous `float64` array (`np.add.reduce`, hence `np.mean`):
/// blocks of up to 128 elements are summed with eight unrolled partial sums, larger arrays are
/// split recursively at a multiple of eight.
pub fn numpy_pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut result = 0.0;
        for value in values {
            result += value;
        }
        result
    } else if n <= BLOCK {
        let mut partial = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut i = 8;
        while i < n - (n % 8) {
            for (lane, slot) in partial.iter_mut().enumerate() {
                *slot += values[i + lane];
            }
            i += 8;
        }
        let mut result = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        while i < n {
            result += values[i];
            i += 1;
        }
        result
    } else {
        let mut half = n / 2;
        half -= half % 8;
        numpy_pairwise_sum(&values[..half]) + numpy_pairwise_sum(&values[half..])
    }
}

/// `fit_elo` from `sotarena/elo.py`: sorted method universe, the default constants, and a final
/// mean-centring at [`CENTER`] (numpy's pairwise mean).
pub fn fit_elo(matches: &[Match]) -> BTreeMap<String, f64> {
    let methods: Vec<&str> = {
        let mut set: Vec<&str> = matches
            .iter()
            .flat_map(|m| [m.method_a.as_str(), m.method_b.as_str()])
            .collect();
        set.sort_unstable();
        set.dedup();
        set
    };
    if methods.is_empty() {
        return BTreeMap::new();
    }
    let index: BTreeMap<&str, usize> = methods.iter().enumerate().map(|(i, m)| (*m, i)).collect();
    let index_a: Vec<usize> = matches.iter().map(|m| index[m.method_a.as_str()]).collect();
    let index_b: Vec<usize> = matches.iter().map(|m| index[m.method_b.as_str()]).collect();
    let outcomes: Vec<f64> = matches.iter().map(|m| m.outcome_a).collect();
    let mut fit = fit_elo_arrays(
        &index_a,
        &index_b,
        &outcomes,
        methods.len(),
        None,
        CENTER,
        ITERATIONS,
        REGULARIZATION,
        TOLERANCE,
    );
    let mean = numpy_pairwise_sum(&fit.ratings) / fit.ratings.len() as f64;
    let shift = CENTER - mean;
    for rating in &mut fit.ratings {
        *rating += shift;
    }
    methods
        .into_iter()
        .zip(fit.ratings)
        .map(|(method, rating)| (method.to_owned(), rating))
        .collect()
}
