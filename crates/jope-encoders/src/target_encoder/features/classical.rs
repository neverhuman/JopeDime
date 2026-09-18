use super::*;

pub(in crate::target_encoder) fn classical_summary(
    values: &[f32],
    task: TargetTask,
    original_length: usize,
    effective_count: usize,
    replacement_fraction: f64,
) -> Vec<f32> {
    let values = values
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    let n = values.len() as f64;
    let mut features = Vec::with_capacity(FEATURE_DIM);
    let mut add = |value: f64| {
        features.push(if value.is_finite() { value as f32 } else { 0.0 });
    };

    add(f64::from(task == TargetTask::Binary));
    add(f64::from(task == TargetTask::Regression));
    add((original_length as f64).ln_1p());
    add((effective_count as f64).ln_1p());
    add(replacement_fraction);
    add(n);
    add(n / WINDOW_SIZE as f64);

    let mean = mean_f64(&values);
    let variance = mean_f64(
        &values
            .iter()
            .map(|value| (value - mean).powi(2))
            .collect::<Vec<_>>(),
    );
    let std = variance.max(0.0).sqrt();
    let median = quantile(&sorted, 0.5);
    let mad = quantile(
        &{
            let mut values = values
                .iter()
                .map(|value| (value - median).abs())
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values
        },
        0.5,
    );
    let mean_abs_dev = mean_f64(
        &values
            .iter()
            .map(|value| (value - mean).abs())
            .collect::<Vec<_>>(),
    );
    add(mean);
    add(std);
    add(variance);
    add(safe_div(std, n.sqrt()));
    add(sorted[0]);
    add(sorted[sorted.len() - 1]);
    add(sorted[sorted.len() - 1] - sorted[0]);
    add(median);
    add(mean - median);
    add(mad);
    add(mean_abs_dev);
    add(safe_div(std, mean.abs() + 1e-6));
    add(((mean + 1e-6) / (1.0 - mean + 1e-6)).ln());

    for power in 2..=12 {
        add(mean_f64(
            &values
                .iter()
                .map(|value| value.powi(power))
                .collect::<Vec<_>>(),
        ));
    }
    for power in 2..=10 {
        let central = mean_f64(
            &values
                .iter()
                .map(|value| (value - mean).powi(power))
                .collect::<Vec<_>>(),
        );
        add(central);
        if power >= 3 {
            add(safe_div(central, (std + 1e-6).powi(power)));
        }
    }

    const QUANTILE_POSITIONS: [f64; 25] = [
        0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.125, 0.20, 0.25, 0.333, 0.375, 0.50, 0.625,
        0.667, 0.75, 0.80, 0.875, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0,
    ];
    let quantiles = QUANTILE_POSITIONS
        .iter()
        .map(|&position| quantile(&sorted, position))
        .collect::<Vec<_>>();
    for &value in &quantiles {
        add(value);
    }
    let q = |position: f64| {
        let index = QUANTILE_POSITIONS
            .iter()
            .position(|candidate| candidate.to_bits() == position.to_bits())
            .expect("requested classical quantile is declared");
        quantiles[index]
    };
    for (left, right) in [
        (0.25, 0.75),
        (0.10, 0.90),
        (0.05, 0.95),
        (0.01, 0.99),
        (0.001, 0.999),
    ] {
        add(q(right) - q(left));
    }
    for pair in quantiles.windows(2) {
        add(pair[1] - pair[0]);
    }
    let iqr = q(0.75) - q(0.25);
    add(iqr);
    add(safe_div(q(0.75) + q(0.25) - 2.0 * q(0.50), iqr + 1e-6));
    add(safe_div(
        q(0.90) + q(0.10) - 2.0 * q(0.50),
        q(0.90) - q(0.10) + 1e-6,
    ));
    add(safe_div(
        (q(0.875) - q(0.625)) + (q(0.375) - q(0.125)),
        iqr + 1e-6,
    ));

    for trim in [0.01, 0.05, 0.10, 0.20, 0.25] {
        let offset = (n * trim).floor() as usize;
        let trimmed = if offset >= sorted.len() - offset {
            mean
        } else {
            mean_f64(&sorted[offset..sorted.len() - offset])
        };
        add(trimmed);
        let low = quantile(&sorted, trim);
        let high = quantile(&sorted, 1.0 - trim);
        add(mean_f64(
            &values
                .iter()
                .map(|value| value.clamp(low, high))
                .collect::<Vec<_>>(),
        ));
    }

    const THRESHOLDS: [f64; 21] = [
        0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
        0.95, 0.98, 0.99, 0.995, 0.999, 1.0,
    ];
    for threshold in THRESHOLDS {
        add(values.iter().filter(|&&value| value <= threshold).count() as f64 / n);
        add(values.iter().filter(|&&value| value >= threshold).count() as f64 / n);
    }
    add(values.iter().filter(|&&value| value <= 1e-12).count() as f64 / n);
    add(values.iter().filter(|&&value| value >= 1.0 - 1e-12).count() as f64 / n);
    add(values
        .iter()
        .filter(|&&value| (0.40..=0.60).contains(&value))
        .count() as f64
        / n);
    add(values
        .iter()
        .filter(|&&value| value <= 0.05 || value >= 0.95)
        .count() as f64
        / n);

    for bins in [4, 8, 16, 32, 64] {
        let histogram = hard_histogram(&values, bins);
        add(entropy(&histogram));
        add(1.0 - histogram.iter().map(|value| value * value).sum::<f64>());
        add(histogram.iter().copied().fold(0.0, f64::max));
        add(histogram.iter().filter(|&&value| value <= 0.0).count() as f64 / bins as f64);
        for value in histogram {
            add(value);
        }
    }

    let histogram = hard_histogram(&values, 64);
    let histogram_mean = mean_f64(&histogram);
    for frequency in 1..=16 {
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (index, value) in histogram.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * frequency as f64 * index as f64 / 64.0;
            real += (value - histogram_mean) * angle.cos();
            imaginary -= (value - histogram_mean) * angle.sin();
        }
        add(real);
        add(imaginary);
        add(real.hypot(imaginary));
    }

    let mut gaps = Vec::with_capacity(sorted.len() + 1);
    gaps.push(sorted[0]);
    gaps.extend(sorted.windows(2).map(|pair| pair[1] - pair[0]));
    gaps.push(1.0 - sorted[sorted.len() - 1]);
    let gap_mean = mean_f64(&gaps);
    let gap_variance = mean_f64(
        &gaps
            .iter()
            .map(|gap| (gap - gap_mean).powi(2))
            .collect::<Vec<_>>(),
    );
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(f64::total_cmp);
    add(gap_mean);
    add(gap_variance.sqrt());
    add(sorted_gaps[0]);
    add(sorted_gaps[sorted_gaps.len() - 1]);
    add(sorted[0]);
    add(1.0 - sorted[sorted.len() - 1]);
    for position in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99] {
        add(quantile(&sorted_gaps, position));
    }

    let sum = values.iter().sum::<f64>();
    let gini = if sum <= 1e-12 {
        0.0
    } else {
        2.0 * sorted
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>()
            / (n * sum)
            - (n + 1.0) / n
    };
    add(gini);
    let positive = values
        .iter()
        .map(|value| value.max(1e-12))
        .collect::<Vec<_>>();
    let positive_mean = mean_f64(&positive);
    add(mean_f64(
        &positive
            .iter()
            .map(|value| {
                let ratio = value / positive_mean;
                ratio * ratio.ln()
            })
            .collect::<Vec<_>>(),
    ));
    add(values
        .iter()
        .flat_map(|left| values.iter().map(move |right| (left - right).abs()))
        .sum::<f64>()
        / (n * n));
    let mut ks: f64 = 0.0;
    let mut cvm = 1.0 / (12.0 * n);
    let mut wasserstein = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        let rank = (index + 1) as f64;
        ks = ks.max(
            (rank / n - value)
                .abs()
                .max((value - index as f64 / n).abs()),
        );
        cvm += (value - (2.0 * rank - 1.0) / (2.0 * n)).powi(2);
        wasserstein += (value - (rank - 0.5) / n).abs();
    }
    add(ks);
    add(cvm);
    add(wasserstein / n);

    for decimals in [0, 1, 2, 3, 4, 6] {
        let scale = 10f64.powi(decimals);
        let mut rounded = values
            .iter()
            .map(|value| (value * scale).round_ties_even() / scale)
            .collect::<Vec<_>>();
        rounded.sort_by(f64::total_cmp);
        let mut counts = Vec::new();
        for value in rounded {
            if counts.last().is_some_and(|(last, _)| *last == value) {
                counts.last_mut().expect("count exists").1 += 1usize;
            } else {
                counts.push((value, 1usize));
            }
        }
        let probabilities = counts
            .iter()
            .map(|(_, count)| *count as f64 / n)
            .collect::<Vec<_>>();
        add(counts.len() as f64 / n);
        add(probabilities.iter().copied().fold(0.0, f64::max));
        add(entropy(&probabilities));
    }

    for degree in 1..=12 {
        add(mean_f64(
            &values
                .iter()
                .map(|value| legendre(degree, 2.0 * value - 1.0))
                .collect::<Vec<_>>(),
        ));
    }
    for frequency in 1..=16 {
        let sin_mean = mean_f64(
            &values
                .iter()
                .map(|value| (2.0 * std::f64::consts::PI * frequency as f64 * value).sin())
                .collect::<Vec<_>>(),
        );
        let cos_mean = mean_f64(
            &values
                .iter()
                .map(|value| (2.0 * std::f64::consts::PI * frequency as f64 * value).cos())
                .collect::<Vec<_>>(),
        );
        add(sin_mean);
        add(cos_mean);
        add(sin_mean.hypot(cos_mean));
    }
    for bandwidth in [0.025, 0.05, 0.10, 0.20] {
        let denominator = 2.0 * bandwidth * bandwidth;
        for index in 0..33 {
            let center = index as f64 / 32.0;
            add(mean_f64(
                &values
                    .iter()
                    .map(|value| (-((value - center).powi(2)) / denominator).exp())
                    .collect::<Vec<_>>(),
            ));
        }
    }
    let binary_rate = values.iter().filter(|&&value| value >= 0.5).count() as f64 / n;
    let binary_entropy = if (0.0..1.0).contains(&binary_rate) && binary_rate > 0.0 {
        -(binary_rate * binary_rate.ln() + (1.0 - binary_rate) * (1.0 - binary_rate).ln())
            / 2.0f64.ln()
    } else {
        0.0
    };
    add(binary_rate);
    add(1.0 - (2.0 * binary_rate - 1.0).abs());
    add(binary_entropy);
    add(binary_rate * (1.0 - binary_rate));

    debug_assert_eq!(features.len(), FEATURE_DIM);
    features
}
