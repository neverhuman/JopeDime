use super::*;

#[test]
fn bank_prior_is_bounded_and_removal_keeps_parsing_available() {
    let p = Program::new(2, 9, vec![], 0.02).unwrap();
    let profile = GoertzelProfile {
        goertzel: true,
        prior: 0.25,
    };
    let created = (0..1024)
        .filter(|&seed| {
            !p.mutate_goertzel(seed, &profile)
                .unwrap()
                .stages()
                .is_empty()
        })
        .count();
    assert!(
        (200..=320).contains(&created),
        "prior gate created {created}/1024"
    );
    let disabled = GoertzelProfile {
        goertzel: false,
        prior: 0.25,
    };
    assert!((0..32).all(|seed| p.mutate_goertzel(seed, &disabled).unwrap() == p));
    for prior in [0.0, -0.1, 0.251, f64::NAN, f64::INFINITY] {
        assert!(GoertzelProfile {
            goertzel: true,
            prior
        }
        .validate()
        .is_err());
    }
    let cap = feat_math::kernel::capabilities::row_vector_v2_capability(61).unwrap();
    assert!(cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Experimental));
    assert!(!cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Baseline));
    // Every seeded edit keeps a valid, lossless program within the bank bounds.
    let mut current = p.clone();
    for seed in 0..64_u64 {
        let arm = [Edit::Grow, Edit::Change, Edit::Drop][(seed % 3) as usize];
        if let Ok(next) = current.edit_goertzel(seed, arm, &profile) {
            assert_eq!(
                Program::from_bytes(&next.to_bytes().unwrap()).unwrap(),
                next
            );
            for stage in next.stages() {
                if let RowStage::Goertzel(config) = stage {
                    assert!((1..=goertzel::MAX_BINS).contains(&config.bins()));
                }
            }
            current = next;
        }
    }
}

#[test]
fn six_spectrum_tasks_five_seeds_equal_budget_tone_power_ablation() {
    // Fixture qualification only: no product corpus, sealed labels, campaign worker, promotion
    // or broad search-lift claim. Both pipelines fit the same ridge head on the same 96 training
    // rows per task/seed and are scored on the same 64 held-out rows; the reference is the
    // existing integer-bin DFT magnitude stage.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/goertzel_utility.json"
    ))
    .unwrap();
    let width = fixture["width"].as_u64().unwrap() as usize;
    let mut receipt = Vec::new();
    for task in fixture["tasks"].as_array().unwrap() {
        let target = task["frequency"].as_f64().unwrap();
        let nuisance: Vec<f64> = task["nuisance"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let bins = task["reference_bins"].as_u64().unwrap();
        let reference = feat_math::GenomeIr::parse(&format!(
            "(pipe X (fft_mag k={bins}) (ridge lambda_frac=0.02))"
        ))
        .unwrap();
        let candidate = program(width, vec![RowStage::Goertzel(bank(&[target]))]);
        let (mut base_total, mut new_total) = (0.0, 0.0);
        for seed in fixture["seeds"].as_array().unwrap() {
            let seed = seed.as_u64().unwrap();
            let mut rng = feat_math::Pcg32::new(seed, 0x676f65727a656c);
            let mut draw_rows = |count: usize| -> (Vec<Vec<f64>>, Vec<f64>) {
                (0..count)
                    .map(|_| {
                        let amplitude = 0.5 + rng.next_f64() * 1.5;
                        let phase = rng.next_f64() * std::f64::consts::TAU;
                        let tones: Vec<(f64, f64, f64)> = nuisance
                            .iter()
                            .map(|&f| {
                                (
                                    f,
                                    rng.next_f64() * 1.5,
                                    rng.next_f64() * std::f64::consts::TAU,
                                )
                            })
                            .collect();
                        let row = (0..width)
                            .map(|n| {
                                let n = n as f64;
                                let mut value =
                                    amplitude * (std::f64::consts::TAU * target * n + phase).cos();
                                for &(f, a, p) in &tones {
                                    value += a * (std::f64::consts::TAU * f * n + p).cos();
                                }
                                value + (rng.next_f64() * 2.0 - 1.0) * 0.1
                            })
                            .collect::<Vec<_>>();
                        (row, amplitude * amplitude)
                    })
                    .unzip()
            };
            let (train_x, train_y) = draw_rows(96);
            let (test_x, test_y) = draw_rows(64);
            let mse = |predictions: &[f64]| {
                predictions
                    .iter()
                    .zip(&test_y)
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f64>()
                    / test_y.len() as f64
            };
            let old = feat_math::fit(&reference, &train_x, &train_y)
                .unwrap()
                .predict(&test_x)
                .unwrap();
            let new = candidate
                .fit(&train_x, &train_y)
                .unwrap()
                .predict(&test_x)
                .unwrap();
            assert!(new.1.iter().all(|&f| f == 0));
            let (base, bank_mse) = (mse(&old), mse(&new.0));
            base_total += base;
            new_total += bank_mse;
            receipt.push(serde_json::json!({"task":task["id"],"seed":seed,"train_rows":96,"heldout_rows":64,"reference_mse":base,"goertzel_mse":bank_mse}));
        }
        // No predetermined gain: the bank must not be worse than the reference on any task, and
        // the receipt records the actual ratios (about 0.12..0.69 at this fixture).
        assert!(
            new_total.is_finite() && new_total < base_total,
            "task {}: goertzel {new_total} reference {base_total}",
            task["id"]
        );
    }
    assert_eq!(receipt.len(), 30);
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("goertzel-utility-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("receipt.json"),
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
