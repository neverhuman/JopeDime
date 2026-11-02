#[test]
fn seeded_composed_expression_recipes_match_rust_python_and_evaluator_bits() {
    let programs = [
        "(pipe X (map expr=\"round(x)\" append=true) (derive expr=\"c0+c5\" append=true) (ridge))",
        "(pipe X (derive expr=\"c1/c0\" append=true) (map expr=\"clip(x,-2,2)\") (ridge))",
        "(pipe X (map expr=\"abs(x)\" append=true) (derive expr=\"row_sum-row_mean+c5\" append=true) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=30))",
    ];
    let scratch = tempfile::tempdir().unwrap();
    for seed in [41, 43, 47] {
        let (mut x, y) = toy_data_seed(seed);
        // Targets remain finite and unchanged; the pipeline sanitizes features before fit.
        for (index, value) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[index][index] = value;
            x[66 + index][index] = value;
        }
        x[64][0] = -0.0;
        x[65][1] = f64::from_bits(1);
        x[69] = vec![0.0, -0.0, 0.0, -0.0, 0.0];
        x[70][0] = f64::from_bits(1e-12_f64.to_bits() - 1);
        x[71][0] = f64::from_bits(1e-12_f64.to_bits() + 1);
        assert!(y.iter().all(|value| value.is_finite()));
        for (index, source) in programs.into_iter().enumerate() {
            let name = format!("seed-{seed}-program-{index}");
            let genome = round_trip(source);
            let expected: Vec<u64> = fit_predict(
                &genome,
                &x[..TRAINING_ROWS].to_vec(),
                &y[..TRAINING_ROWS],
                &x[TRAINING_ROWS..].to_vec(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
            let dir = scratch.path().join(&name);
            let (python, rust) = run_emitted_pair(&name, &genome, &dir, &x, &y);
            assert_eq!(python, expected, "{name}: Python bits");
            assert_eq!(rust, expected, "{name}: Rust bits");
            let recipe = emit_expression_cpp17(&genome, &EmitOptions::default()).unwrap();
            assert_eq!(recipe.profile, PROFILE);
            let cpp = run_cpp_recipe(&name, recipe.source, &dir, &x, &y);
            assert_eq!(cpp, expected, "{name}: C++17 expression recipe bits");
        }
    }
}

#[test]
fn seeded_stage_parameter_recipes_match_three_languages_and_evaluator_bits() {
    let programs = [
        "(pipe X (quantile_bins bins=2) (ridge lambda_frac=0.1))",
        "(pipe X (quantile_bins bins=16 append=true) (elasticnet alpha=0.02 l1_ratio=0.25 sweeps=17))",
        "(pipe X (cheb degree=1) (ridge lambda_frac=0.03))",
        "(pipe X (cheb degree=6 append=true) (ridge lambda_frac=2))",
        "(pipe X (step t=-0.5) (ridge lambda_frac=0.2))",
        "(pipe X (step t=0.75 append=true) (elasticnet alpha=0.03 l1_ratio=0.75 sweeps=23))",
        "(pipe X (pair_prod topk=2) (ridge lambda_frac=0.04))",
        "(pipe X (pair_prod topk=4 append=true) (quantile_bins bins=3) (elasticnet alpha=0.03 l1_ratio=0.5 sweeps=19))",
    ];
    let mutations = [
        ("quantile_bins.bins", 0, "bins=2", "bins=3"),
        ("ridge.lambda_frac", 0, "lambda_frac=0.1", "lambda_frac=0.2"),
        ("cheb.degree", 2, "degree=1", "degree=2"),
        ("step.t", 4, "t=-0.5", "t=-0.25"),
        ("pair_prod.topk", 6, "topk=2", "topk=3"),
        ("elasticnet.alpha", 1, "alpha=0.02", "alpha=0.04"),
        ("elasticnet.l1_ratio", 1, "l1_ratio=0.25", "l1_ratio=0.6"),
        ("elasticnet.sweeps", 1, "sweeps=17", "sweeps=3"),
    ];
    let scratch = tempfile::tempdir().unwrap();
    let mut cases = 0;
    let mut sensitivities = 0;
    for seed in [53, 59] {
        let (mut x, y) = toy_data_seed(seed);
        // Fixed boundary features supplement each seed. Targets remain finite;
        // the same training split feeds native fit, CSV recipes and the C++ ABI.
        x[0][0] = -0.0;
        x[1][1] = f64::from_bits(1);
        x[64][0] = -0.0;
        x[65][1] = f64::from_bits(1);
        x[66][2] = -0.5;
        x[67][2] = 0.75;
        let check = |name: &str, source: &str| {
            let genome = round_trip(source);
            assert!(supports(&genome).is_ok(), "{name}: paired admission");
            assert!(
                supports_cpp17(&genome).is_ok(),
                "{name}: default C++17 admission"
            );
            let expected: Vec<u64> = fit_predict(
                &genome,
                &x[..TRAINING_ROWS].to_vec(),
                &y[..TRAINING_ROWS],
                &x[TRAINING_ROWS..].to_vec(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
            assert_eq!(expected.len(), 16, "{name}: evaluator output count");
            let dir = scratch.path().join(name);
            let (python, rust) = run_emitted_pair(name, &genome, &dir, &x, &y);
            assert_eq!(python, expected, "{name}: Python parameter recipe bits");
            assert_eq!(rust, expected, "{name}: Rust parameter recipe bits");
            let source = emit_cpp17(&genome, &EmitOptions::default()).unwrap();
            let cpp = run_cpp_recipe(name, source, &dir, &x, &y);
            assert_eq!(cpp, expected, "{name}: C++17 parameter recipe bits");
            expected
        };
        let originals: Vec<_> = programs
            .iter()
            .enumerate()
            .map(|(index, source)| {
                check(&format!("parameters-seed-{seed}-program-{index}"), source)
            })
            .collect();
        cases += originals.len();
        for (parameter, index, from, to) in mutations {
            assert_eq!(
                programs[index].matches(from).count(),
                1,
                "{parameter}: one consumed key"
            );
            let changed = programs[index].replacen(from, to, 1);
            let words = check(
                &format!("parameters-seed-{seed}-changed-{parameter}"),
                &changed,
            );
            assert_ne!(
                words, originals[index],
                "seed{seed}/{parameter}: parameter must affect native held-out predictions"
            );
            sensitivities += 1;
            cases += 1;
        }
    }
    assert_eq!(
        cases, 32,
        "eight originals and eight mutations across two seeds"
    );
    assert_eq!(
        sensitivities, 16,
        "each consumed key changes each seed's predictions"
    );
}
