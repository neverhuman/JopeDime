#[test]
fn experimental_prior_is_bounded_and_removal_keeps_parsing_available() {
    let p = Program::new(2, 3, vec![], 0.02).unwrap();
    let profile = ExperimentalProfile {
        savgol: true,
        prior: 0.25,
    };
    let created = (0..1024)
        .filter(|&seed| !p.mutate(seed, &profile).unwrap().stages().is_empty())
        .count();
    assert!(
        (200..=320).contains(&created),
        "prior gate created {created}/1024"
    );
    let disabled = ExperimentalProfile {
        savgol: false,
        prior: 0.25,
    };
    assert!((0..32).all(|seed| p.mutate(seed, &disabled).unwrap() == p));
    for prior in [0.0, -0.1, 0.251, f64::NAN, f64::INFINITY] {
        assert!(ExperimentalProfile {
            savgol: true,
            prior
        }
        .validate()
        .is_err());
    }
    let cap = feat_math::kernel::capabilities::row_vector_v2_capability(58).unwrap();
    assert!(cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Experimental));
    assert!(!cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Baseline));
    let created = p.edit(17, Edit::Grow, &profile).unwrap();
    assert_eq!(
        Program::from_bytes(&created.to_bytes().unwrap()).unwrap(),
        created
    );
    assert!(created.emit_recipe(RowRecipeLanguage::Cpp17).is_ok());
}

#[test]
fn six_signal_datasets_five_seeds_equal_budget_derivative_ablation() {
    // Fixture qualification only: no product corpus, sealed labels, campaign
    // worker, promotion, or broad search-lift claim. Both kernels evaluate the
    // same 64 rows per task/seed; the old reference is the two-neighbor stencil.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/savgol_utility.json"
    ))
    .unwrap();
    let width = fixture["width"].as_u64().unwrap() as usize;
    let fitted = config(17, 6, 1).fit(width).unwrap();
    let mut receipt = Vec::new();
    for task in fixture["tasks"].as_array().unwrap() {
        let degree = task["degree"].as_u64().unwrap() as usize;
        let mut base_total = 0.0;
        let mut new_total = 0.0;
        for seed in fixture["seeds"].as_array().unwrap() {
            let seed = seed.as_u64().unwrap();
            let mut rng = feat_math::Pcg32::new(seed, degree as u64 + 0x61626c617465);
            let mut base = 0.0;
            let mut new = 0.0;
            for _ in 0..64 {
                let coefficients = (0..=degree)
                    .map(|_| rng.next_f64() * 2.0 - 1.0)
                    .collect::<Vec<_>>();
                let row = (0..width)
                    .map(|j| {
                        let t = (j as f64 - (width / 2) as f64) / 8.0;
                        let signal = coefficients.iter().rev().fold(0.0, |y, c| y * t + c);
                        signal + (rng.next_f64() * 2.0 - 1.0) * 0.5
                    })
                    .collect::<Vec<_>>();
                let target = coefficients[1] / 8.0;
                let old = (row[width / 2 + 1] - row[width / 2 - 1]) / 2.0;
                let output = fitted.apply(&row).unwrap();
                assert!(output.faults.iter().all(|&f| f == 0));
                base += (old - target).powi(2);
                new += (output.values[width / 2] - target).powi(2);
            }
            base_total += base;
            new_total += new;
            receipt.push(serde_json::json!({"task":task["id"],"seed":seed,"rows_each":64,"reference_mse":base/64.0,"savgol_mse":new/64.0}));
        }
        assert!(
            new_total < base_total * 0.7,
            "task {degree}: savgol {new_total} reference {base_total}"
        );
    }
    assert_eq!(receipt.len(), 30);
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("savgol-utility-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("receipt.json"),
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
