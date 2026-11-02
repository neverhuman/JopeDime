use super::*;

use super::*;

pub(super) fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("median requires finite values");
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

pub(super) fn bootstrap_interval(values: &[f64]) -> Result<[f64; 2]> {
    if values.len() != 10 {
        bail!("study bootstrap is pinned to ten datasets");
    }
    let mut rng = DeterministicRng::new(STUDY_SEED ^ 0x6273_7472_6170);
    let mut estimates = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let sample = (0..values.len())
            .map(|_| values[(rng.next_u64() as usize) % values.len()])
            .collect::<Vec<_>>();
        estimates.push(median(&sample)?);
    }
    estimates.sort_by(f64::total_cmp);
    Ok([estimates[249], estimates[9749]])
}

pub(super) fn paired_sign_test(values: &[f64]) -> f64 {
    let positive = values.iter().filter(|value| **value > 0.0).count();
    let negative = values.iter().filter(|value| **value < 0.0).count();
    let n = positive + negative;
    if n == 0 {
        return 1.0;
    }
    let tail = positive.min(negative);
    let mut coefficient = 1.0;
    let mut probability = 1.0;
    for k in 1..=tail {
        coefficient *= (n + 1 - k) as f64 / k as f64;
        probability += coefficient;
    }
    (2.0 * probability / 2_f64.powi(n as i32)).min(1.0)
}

pub(super) fn cohort_elo(datasets: &[DatasetReport]) -> Result<Vec<EloRating>> {
    if datasets.is_empty() {
        bail!("Elo requires at least one dataset");
    }
    const SYSTEMS: [&str; 5] = [
        "ours_gen9",
        "ours_top10_5fold_cv",
        "prior_aspfm",
        "tabfm_ensemble",
        "tabpfn_v3",
    ];
    const INITIAL: f64 = 1500.0;
    const K: f64 = 32.0;
    let mut ratings = [INITIAL; SYSTEMS.len()];
    let mut wins = [0usize; SYSTEMS.len()];
    let mut losses = [0usize; SYSTEMS.len()];
    let mut ties = [0usize; SYSTEMS.len()];
    for dataset in datasets {
        let rmse = [
            dataset.generation_nine_rmse,
            dataset.cv_rerank_rmse,
            dataset.prior_aspfm_rmse,
            dataset.tabfm_ensemble_rmse,
            dataset.tabpfn_v3_rmse,
        ];
        if rmse.iter().any(|value| !value.is_finite() || *value < 0.0) {
            bail!("Elo received an invalid RMSE");
        }
        let before = ratings;
        let mut delta = [0.0; SYSTEMS.len()];
        for left in 0..SYSTEMS.len() {
            for right in (left + 1)..SYSTEMS.len() {
                let outcome = match rmse[left].total_cmp(&rmse[right]) {
                    Ordering::Less => {
                        wins[left] += 1;
                        losses[right] += 1;
                        1.0
                    }
                    Ordering::Greater => {
                        losses[left] += 1;
                        wins[right] += 1;
                        0.0
                    }
                    Ordering::Equal => {
                        ties[left] += 1;
                        ties[right] += 1;
                        0.5
                    }
                };
                let expected = 1.0 / (1.0 + 10_f64.powf((before[right] - before[left]) / 400.0));
                let change = K * (outcome - expected);
                delta[left] += change;
                delta[right] -= change;
            }
        }
        for index in 0..SYSTEMS.len() {
            ratings[index] += delta[index];
        }
    }
    let mut result = SYSTEMS
        .iter()
        .enumerate()
        .map(|(index, system)| EloRating {
            system: (*system).to_string(),
            rating: ratings[index],
            wins: wins[index],
            losses: losses[index],
            ties: ties[index],
        })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| {
        right
            .rating
            .total_cmp(&left.rating)
            .then_with(|| left.system.cmp(&right.system))
    });
    Ok(result)
}

pub(super) fn panel_points(values: &[f64], left: f64, top: f64, width: f64, height: f64) -> String {
    panel_points_scaled(values, values, left, top, width, height)
}

pub(super) fn panel_points_scaled(
    values: &[f64],
    scale: &[f64],
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) -> String {
    let minimum = scale.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = scale.iter().copied().max_by(f64::total_cmp).unwrap();
    let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
    let low = minimum - padding;
    let high = maximum + padding;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = left + width * index as f64 / (values.len() - 1).max(1) as f64;
            let y = top + height * (high - value) / (high - low);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn evolution_svg(
    title: &str,
    validation: &[f64],
    test: &[f64],
    tabpfn: f64,
    tabfm: Option<f64>,
    tabfm_ensemble: f64,
) -> String {
    let mut rmse_scale = test.to_vec();
    rmse_scale.extend([tabpfn, tabfm_ensemble]);
    if let Some(tabfm) = tabfm {
        rmse_scale.push(tabfm);
    }
    let validation_points = panel_points(validation, 80.0, 70.0, 820.0, 210.0);
    let test_points = panel_points_scaled(test, &rmse_scale, 80.0, 370.0, 820.0, 210.0);
    let baseline_y = |value: f64| {
        let minimum = rmse_scale.iter().copied().min_by(f64::total_cmp).unwrap();
        let maximum = rmse_scale.iter().copied().max_by(f64::total_cmp).unwrap();
        let padding = ((maximum - minimum).abs() * 0.08).max(maximum.abs() * 1e-9 + 1e-12);
        370.0 + 210.0 * (maximum + padding - value) / (maximum + padding - (minimum - padding))
    };
    let tabfm_line = tabfm.map_or_else(
        || {
            "<text x=\"905\" y=\"350\" font-family=\"sans-serif\" font-size=\"12\">TabFM unavailable</text>"
                .to_string()
        },
        |value| {
            let y = baseline_y(value);
            format!("<line x1=\"80\" y1=\"{y:.2}\" x2=\"900\" y2=\"{y:.2}\" stroke=\"#d97706\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM</text>")
        },
    );
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"650\" viewBox=\"0 0 960 650\">\n\
<rect width=\"960\" height=\"650\" fill=\"white\"/>\n\
<text x=\"480\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"20\">{title}</text>\n\
<text x=\"20\" y=\"175\" transform=\"rotate(-90 20 175)\" font-family=\"sans-serif\" font-size=\"14\">10-split robust validation fitness</text>\n\
<line x1=\"80\" y1=\"280\" x2=\"900\" y2=\"280\" stroke=\"#999\"/><polyline points=\"{validation_points}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\"/>\n\
<text x=\"20\" y=\"490\" transform=\"rotate(-90 20 490)\" font-family=\"sans-serif\" font-size=\"14\">post-hoc test RMSE</text>\n\
<line x1=\"80\" y1=\"580\" x2=\"900\" y2=\"580\" stroke=\"#999\"/><polyline points=\"{test_points}\" fill=\"none\" stroke=\"#059669\" stroke-width=\"3\"/>\n\
<line x1=\"80\" y1=\"{pfn_y:.2}\" x2=\"900\" y2=\"{pfn_y:.2}\" stroke=\"#dc2626\" stroke-dasharray=\"8 5\"/><text x=\"905\" y=\"{pfn_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabPFN v3</text>\n\
{tabfm_line}\n\
<line x1=\"80\" y1=\"{fm_ensemble_y:.2}\" x2=\"900\" y2=\"{fm_ensemble_y:.2}\" stroke=\"#7c3aed\" stroke-dasharray=\"4 4\"/><text x=\"905\" y=\"{fm_ensemble_y:.2}\" font-family=\"sans-serif\" font-size=\"12\">TabFM ensemble</text>\n\
<text x=\"480\" y=\"625\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">generation 0–9</text>\n</svg>\n",
        pfn_y = baseline_y(tabpfn),
        fm_ensemble_y = baseline_y(tabfm_ensemble),
    )
}
