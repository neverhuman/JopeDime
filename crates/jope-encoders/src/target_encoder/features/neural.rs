use super::*;

pub(in crate::target_encoder) fn neural_explicit_stats(
    values: &[f32],
    task: TargetTask,
    original_length: usize,
) -> Vec<f32> {
    let count = values.len() as f32;
    let mean = values.iter().sum::<f32>() / count;
    let centered = values.iter().map(|value| value - mean).collect::<Vec<_>>();
    let variance = centered.iter().map(|value| value * value).sum::<f32>() / count;
    let std = (variance.max(0.0) + 1e-6).sqrt();
    let skew =
        centered.iter().map(|value| value.powi(3)).sum::<f32>() / count / (std.powi(3) + 1e-6);
    let kurtosis =
        centered.iter().map(|value| value.powi(4)).sum::<f32>() / count / (std.powi(4) + 1e-6);
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let nearest = |position: f32| {
        let index = ((count - 1.0) * position).round_ties_even() as usize;
        sorted[index.min(sorted.len() - 1)]
    };

    let mut output = Vec::with_capacity(442);
    output.extend(if task == TargetTask::Binary {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    });
    output.extend([
        (original_length as f32).ln_1p() / 16.0,
        (original_length.min(WINDOW_SIZE) as f32).ln_1p() / 16.0,
        if original_length < WINDOW_SIZE {
            1.0 - original_length as f32 / WINDOW_SIZE as f32
        } else {
            0.0
        },
        mean,
        std,
        sorted[0],
        sorted[sorted.len() - 1],
        skew,
        kurtosis,
    ]);
    output.extend(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            .into_iter()
            .map(nearest),
    );
    let histogram = soft_histogram(values, 16);
    output.extend(&histogram);
    output.push(
        -histogram
            .iter()
            .map(|value| value * (value + 1e-8).ln())
            .sum::<f32>()
            / 16.0f32.ln(),
    );
    output.push(values.iter().filter(|&&value| value <= 0.05).count() as f32 / count);
    output.push(values.iter().filter(|&&value| value >= 0.95).count() as f32 / count);

    const POSITIONS: [f32; 25] = [
        0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.125, 0.20, 0.25, 0.333, 0.375, 0.50, 0.625,
        0.667, 0.75, 0.80, 0.875, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0,
    ];
    let rich_quantiles = POSITIONS.into_iter().map(nearest).collect::<Vec<_>>();
    output.extend(&rich_quantiles);
    output.extend(rich_quantiles.windows(2).map(|pair| pair[1] - pair[0]));
    let lookup = |position: f32| {
        rich_quantiles[POSITIONS
            .iter()
            .position(|candidate| candidate.to_bits() == position.to_bits())
            .expect("rich quantile exists")]
    };
    for (left, right) in [
        (0.25, 0.75),
        (0.10, 0.90),
        (0.05, 0.95),
        (0.01, 0.99),
        (0.001, 0.999),
    ] {
        output.push(lookup(right) - lookup(left));
    }
    const THRESHOLDS: [f32; 21] = [
        0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
        0.95, 0.98, 0.99, 0.995, 0.999, 1.0,
    ];
    output.extend(
        THRESHOLDS.iter().map(|threshold| {
            values.iter().filter(|value| *value <= threshold).count() as f32 / count
        }),
    );
    output.extend(
        THRESHOLDS.iter().map(|threshold| {
            values.iter().filter(|value| *value >= threshold).count() as f32 / count
        }),
    );
    output.extend([
        values.iter().filter(|&&value| value <= 1e-12).count() as f32 / count,
        values.iter().filter(|&&value| value >= 1.0 - 1e-12).count() as f32 / count,
        values
            .iter()
            .filter(|&&value| (0.40..=0.60).contains(&value))
            .count() as f32
            / count,
        values
            .iter()
            .filter(|&&value| value <= 0.05 || value >= 0.95)
            .count() as f32
            / count,
    ]);

    for bins in [4, 8, 16, 32] {
        let histogram = soft_histogram(values, bins);
        let entropy = -histogram
            .iter()
            .map(|value| value * (value + 1e-8).ln())
            .sum::<f32>()
            / (bins as f32).ln();
        let simpson = 1.0 - histogram.iter().map(|value| value * value).sum::<f32>();
        let max_bin = histogram.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let empty = histogram.iter().filter(|&&value| value <= 1e-8).count() as f32 / bins as f32;
        output.extend(histogram);
        output.extend([entropy, simpson, max_bin, empty]);
    }

    let mut sin_means = Vec::with_capacity(16);
    let mut cos_means = Vec::with_capacity(16);
    for frequency in 1..=16 {
        sin_means.push(
            values
                .iter()
                .map(|value| (2.0 * std::f32::consts::PI * frequency as f32 * value).sin())
                .sum::<f32>()
                / count,
        );
        cos_means.push(
            values
                .iter()
                .map(|value| (2.0 * std::f32::consts::PI * frequency as f32 * value).cos())
                .sum::<f32>()
                / count,
        );
    }
    output.extend(&sin_means);
    output.extend(&cos_means);
    output.extend(
        sin_means
            .iter()
            .zip(&cos_means)
            .map(|(sin, cos)| (sin * sin + cos * cos + 1e-12).sqrt()),
    );

    for bandwidth in [0.025f32, 0.05, 0.10, 0.20] {
        for center in (0..33).map(|index| index as f32 / 32.0) {
            output.push(
                values
                    .iter()
                    .map(|value| {
                        (-((value - center).powi(2)) / (2.0 * bandwidth * bandwidth)).exp()
                    })
                    .sum::<f32>()
                    / count,
            );
        }
    }
    for power in 2..=12 {
        output.push(values.iter().map(|value| value.powi(power)).sum::<f32>() / count);
    }
    let central_moments = (2..=10)
        .map(|power| centered.iter().map(|value| value.powi(power)).sum::<f32>() / count)
        .collect::<Vec<_>>();
    output.extend(&central_moments);
    for power in 3..=10 {
        output.push(central_moments[power - 2] / (std.powi(power as i32) + 1e-6));
    }

    let adjacent = sorted
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0))
        .collect::<Vec<_>>();
    let gap_mean = adjacent.iter().sum::<f32>() / (count - 1.0).max(1.0);
    let gap_std = (adjacent
        .iter()
        .map(|gap| (gap - gap_mean).powi(2))
        .sum::<f32>()
        / (count - 1.0).max(1.0)
        + 1e-8)
        .sqrt();
    output.extend([
        gap_mean,
        gap_std,
        adjacent.iter().copied().fold(1.0, f32::min),
        adjacent.iter().copied().fold(0.0, f32::max),
        rich_quantiles[0],
        1.0 - rich_quantiles[rich_quantiles.len() - 1],
    ]);

    let mut ks = 0.0f32;
    let mut cvm_sum = 0.0f32;
    let mut wasserstein_sum = 0.0f32;
    for (index, value) in sorted.iter().enumerate() {
        let position = index as f32;
        ks = ks.max(
            (((position + 1.0) / count) - value)
                .abs()
                .max((value - position / count).abs()),
        );
        let expected = (position + 0.5) / count;
        cvm_sum += (value - expected).powi(2);
        wasserstein_sum += (value - expected).abs();
    }
    let cvm = (cvm_sum + 1.0 / (12.0 * count)) / count;
    let wasserstein = wasserstein_sum / count;
    let positive_sum = sorted.iter().sum::<f32>().max(1e-8);
    let gini = 2.0
        * sorted
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f32 * value)
            .sum::<f32>()
        / (count * positive_sum)
        - (count + 1.0) / count;
    let safe_values = values
        .iter()
        .map(|value| value.max(1e-12))
        .collect::<Vec<_>>();
    let safe_mean = (safe_values.iter().sum::<f32>() / count).max(1e-12);
    let theil = safe_values
        .iter()
        .map(|value| {
            let ratio = value / safe_mean;
            ratio * ratio.ln()
        })
        .sum::<f32>()
        / count;
    let pairwise = values
        .iter()
        .flat_map(|left| values.iter().map(move |right| (left - right).abs()))
        .sum::<f32>()
        / (count * count);
    output.extend([ks, cvm, wasserstein, gini, theil, pairwise]);

    for bandwidth in [0.001f32, 0.01, 0.05] {
        let counts = values
            .iter()
            .map(|left| {
                values
                    .iter()
                    .map(|right| (-((left - right).powi(2)) / (2.0 * bandwidth * bandwidth)).exp())
                    .sum::<f32>()
                    .max(1e-6)
            })
            .collect::<Vec<_>>();
        let mode_fraction = counts.iter().copied().fold(f32::NEG_INFINITY, f32::max) / count;
        let soft_unique = counts.iter().map(|value| 1.0 / value).sum::<f32>() / count;
        let probability_sum = counts.iter().sum::<f32>().max(1e-6);
        let entropy = -counts
            .iter()
            .map(|value| {
                let probability = value / probability_sum;
                probability * (probability + 1e-8).ln()
            })
            .sum::<f32>()
            / (count + 1.0).ln();
        output.extend([mode_fraction, soft_unique, entropy]);
    }

    for value in &mut output {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    debug_assert_eq!(output.len(), 442);
    output
}

fn soft_histogram(values: &[f32], bins: usize) -> Vec<f32> {
    let width = 1.0 / (bins - 1).max(1) as f32;
    let mut histogram = (0..bins)
        .map(|index| {
            let center = torch_unit_point(index, bins);
            values
                .iter()
                .map(|value| (1.0 - (value - center).abs() / width).max(0.0))
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let sum = histogram.iter().sum::<f32>().max(1e-6);
    for value in &mut histogram {
        *value /= sum;
    }
    histogram
}
