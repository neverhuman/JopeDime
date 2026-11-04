#[test]
fn every_direct_wire_witness_evaluates_to_its_frozen_model_bits() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let (x, y) = toy_data();
    for (name, source) in witness_sources() {
        let genome = GenomeIr::parse(&source).unwrap();
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        let actual: Vec<u64> = fit_predict(
            &genome,
            &x[..TRAINING_ROWS].to_vec(),
            &y[..TRAINING_ROWS],
            &x[TRAINING_ROWS..].to_vec(),
        )
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error:#}"))
        .into_iter()
        .map(f64::to_bits)
        .collect();
        assert_eq!(actual, expected, "{name}: frozen evaluator bits");
    }
}

#[test]
fn every_direct_wire_witness_runs_emitted_rust_and_python_on_the_frozen_input() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-wire-matrix-")
        .tempdir_in(scratch_root)
        .unwrap();
    for (name, source) in witness_sources() {
        let dir = scratch.path().join(&name);
        let genome = GenomeIr::parse(&source).unwrap();
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        assert_eq!(expected.len(), 16, "{name}: frozen row count");
        let (python, rust) = run_emitted_pair(&name, &genome, &dir, &x, &y);
        assert_eq!(python, expected, "{name}: Python bits");
        assert_eq!(rust, expected, "{name}: Rust bits");
    }
}

#[test]
fn standalone_cpp17_recipe_executes_every_admitted_direct_wire_witness() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let frozen_emissions = rows_by_name(emitted["programs"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-wire-cpp17-")
        .tempdir_in(scratch_root)
        .unwrap();
    let mut admitted = 0;
    let mut refused = 0;
    for (name, source) in witness_sources() {
        let genome = GenomeIr::parse(&source).unwrap();
        let support = supports_cpp17(&genome);
        let emission = emit_cpp17(&genome, &EmitOptions::default());
        assert_eq!(support.is_ok(), emission.is_ok(), "{name}: C++17 admission");
        let frozen_route = frozen_emissions[name.as_str()]["cpp17"].as_str().unwrap();
        assert_eq!(
            support.is_err(),
            frozen_route.starts_with("unsupported:"),
            "{name}: frozen C++17 route"
        );
        let Ok(driver) = emission else {
            let reasons = support
                .unwrap_err()
                .into_iter()
                .map(|reason| {
                    reason
                        .split(['(', '{'])
                        .next()
                        .unwrap()
                        .trim_end()
                        .to_owned()
                })
                .collect::<Vec<_>>()
                .join("; ");
            assert_eq!(
                reasons,
                frozen_route.strip_prefix("unsupported: ").unwrap(),
                "{name}: frozen C++17 refusal reasons"
            );
            refused += 1;
            continue;
        };
        admitted += 1;
        let actual = run_cpp_recipe(&name, driver, &scratch.path().join(&name), &x, &y);
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        assert_eq!(actual, expected, "{name}: standalone C++17 recipe bits");
    }
    assert_eq!(admitted, 6, "standalone C++17 route inventory");
    assert_eq!(refused, 13, "each other direct wire witness classified");
}

#[test]
fn expression_cpp17_recipe_executes_admitted_wire_witnesses_and_retains_refusals() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let frozen_emissions = rows_by_name(emitted["programs"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch = tempfile::tempdir().unwrap();
    let mut admitted = 0;
    let mut refused = 0;
    for (name, source) in witness_sources() {
        let genome = round_trip(&source);
        let legacy = frozen_emissions[name.as_str()]["cpp17"].as_str().unwrap();
        match emit_expression_cpp17(&genome, &EmitOptions::default()) {
            Ok(recipe) => {
                admitted += 1;
                assert_eq!(recipe.profile, PROFILE);
                assert!(recipe.source.starts_with(&format!("// {PROFILE}\n")));
                let actual =
                    run_cpp_recipe(&name, recipe.source, &scratch.path().join(&name), &x, &y);
                let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
                    .as_str()
                    .unwrap()
                    .split_whitespace()
                    .map(|word| u64::from_str_radix(word, 16).unwrap())
                    .collect();
                assert_eq!(actual, expected, "{name}: expression recipe frozen bits");
                if legacy.starts_with("unsupported:") {
                    assert!(matches!(
                        name.as_str(),
                        "golden_map_derive_ridge" | "golden_map_pow_elasticnet"
                    ));
                    assert!(emit_cpp17(&genome, &EmitOptions::default()).is_err());
                }
            }
            Err(feat_math::kernel::Error::Unsupported(reason)) => {
                refused += 1;
                // As with supports_cpp17 above, compare element kinds; debug payloads
                // add parameter context but do not change the refused route.
                let labels = reason
                    .split(", ")
                    .filter(|part| {
                        part.starts_with("head ")
                            || part.starts_with("transform ")
                            || *part == "target transforms"
                    })
                    .map(|part| part.split(['(', '{']).next().unwrap().trim_end())
                    .collect::<Vec<_>>()
                    .join("; ");
                assert_eq!(
                    labels,
                    legacy.strip_prefix("unsupported: ").unwrap(),
                    "{name}: structured expression recipe refusal: {reason}"
                );
            }
            Err(error) => panic!("{name}: unexpected expression recipe error: {error}"),
        }
    }
    assert_eq!(admitted, 8, "six legacy plus two expression wire witnesses");
    assert_eq!(refused, 11, "unsupported heads/stages/targets stay refused");
}
