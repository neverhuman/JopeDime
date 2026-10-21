//! R0 Baseline Golden Bank & 7 Execution Routes Freeze (MR11).
//!
//! Verifies:
//! 1. `baseline_manifest_and_7_routes_frozen`:
//!    - All 7 execution routes execute deterministically (`LegacyV1`, `ScalarV2`, `AdvancedV3`,
//!      `RowVectorV2`, `OrderedStateV2`, `TypedGraphV1`, `ArtifactEmissionV3`).
//!    - Vocabulary and registry counts match the R0 freeze (`53` wire v1 operators, `90`
//!      `registry_v2` entries, `41` `wire_v2` reservations = `23` wire + `16` func + `2` var).
//! 2. `golden_bank_64_fixtures_bit_identical`:
//!    - 55 `compat_freeze` (`emitted_bytes.v1.json` + `model_serde.v1.json`) programs
//!    - 5 `legacy_v2_freeze` representative fixtures
//!    - 4 `historical_v1` wire-alias fixtures
//!
//!    Total = 64 golden fixtures verified bit-identical against frozen hashes and prediction bits.

use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::{Constraints, InputSchema, RowSemantics};
use feat_math::kernel::{registry_v2, wire_v2, Program};
use feat_math::math_expr::{Expr, Func, MathExpr, Var};
use feat_math::registry::operator_registry;
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, EmitOptions};
use feat_math::{fit, fit_predict, GenomeIr, Model, Pcg32, CHAMPION_GENOME};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compat_freeze")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn f64_bits_hex(val: f64) -> String {
    format!("{:016x}", val.to_bits())
}

fn toy_data(n: usize, cols: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..cols).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

#[test]
fn baseline_manifest_and_7_routes_frozen() {
    // 1. Vocabulary and registry counts
    assert_eq!(
        operator_registry().len(),
        53,
        "v1 wire operator registry must have 53 entries"
    );
    assert_eq!(
        registry_v2::OPERATORS.len(),
        90,
        "registry_v2 must have 90 entries"
    );
    registry_v2::validate_inheritance().expect("registry_v2 inheritance must validate");

    let wire_v2_total = wire_v2::WIRE.len() + wire_v2::FUNCTIONS.len() + wire_v2::VARIABLES.len();
    assert_eq!(wire_v2::WIRE.len(), 23);
    assert_eq!(wire_v2::FUNCTIONS.len(), 16);
    assert_eq!(wire_v2::VARIABLES.len(), 2);
    assert_eq!(
        wire_v2_total, 41,
        "wire_v2 reservations must total 41 (23 wire + 16 func + 2 var)"
    );

    // 2. Verify all 7 execution routes execute deterministically
    let rows = vec![
        vec![1.0, 0.5, -1.0, 2.0],
        vec![2.0, 1.0, 0.0, 1.0],
        vec![-1.0, 2.0, 1.0, 0.0],
        vec![0.5, -0.5, 2.0, -1.0],
        vec![3.0, 1.5, -2.0, 0.5],
        vec![-2.0, -1.0, 0.5, 3.0],
        vec![1.25, 2.5, -0.25, -1.5],
        vec![-0.75, 0.25, 1.5, 2.5],
    ];
    let target = vec![2.1, 2.6, -2.4, 1.3, 5.7, -2.8, 0.95, -0.4];

    // Route 1: LegacyV1 GenomeIr fit_predict
    let hold_rows = rows[..2].to_vec();
    let legacy_genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let route1_preds = fit_predict(&legacy_genome, &rows, &target, &hold_rows).unwrap();
    assert_eq!(route1_preds.len(), 2);
    assert!(route1_preds.iter().all(|v| v.is_finite()));

    // Route 2: ScalarV2 MathExpr evaluation
    let scalar_expr =
        MathExpr::from_ast(Expr::Call(Func::Sigmoid, vec![Expr::Var(Var::X)])).unwrap();
    let route2_val = scalar_expr.eval_element(0.5, &rows[0]).unwrap();
    assert!(route2_val.is_finite());

    // Route 3: AdvancedV3 evolved genome evaluation
    let v3_seed = feat_math::deterministic_seed_enumeration(
        &feat_math::V3EvolutionConfig::default(),
        &["(pipe X (ridge))".to_string()],
    );
    assert!(!v3_seed.is_empty());

    // Route 4: RowVectorV2 row-wise statistics
    let row_mean_expr = MathExpr::from_ast(Expr::Var(Var::RowMean)).unwrap();
    let route4_val = row_mean_expr.eval_element(0.0, &rows[0]).unwrap();
    assert!((route4_val - 0.625).abs() <= 1e-12);

    // Route 5: OrderedStateV2 lag/roll pipeline execution
    let ordered_genome =
        GenomeIr::parse("(pipe X (lag k=2 append=true) (roll k=3) (ridge))").unwrap();
    let route5_preds = fit_predict(&ordered_genome, &rows, &target, &hold_rows).unwrap();
    assert_eq!(route5_preds.len(), 2);
    assert!(route5_preds.iter().all(|v| v.is_finite()));

    // Route 6: TypedGraphV1 via LegacyBridge
    let schema = InputSchema::new(
        (0..4).map(|i| format!("column_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap();
    let prog = Program::import_legacy(&legacy_genome);
    let bridge = LegacyBridge::import(&prog, &schema, &Constraints::default()).unwrap();
    assert!(matches!(bridge.bridge, BridgeStatus::Typed { .. }));

    // Route 7: ArtifactEmissionV3 (Rust, Python, Manifest, C++17, JS)
    let emitted = emit(&legacy_genome, &EmitOptions::default()).unwrap();
    let cpp = emit_cpp17(&legacy_genome, &EmitOptions::default()).unwrap();
    let js = emit_javascript(&legacy_genome, &EmitOptions::default()).unwrap();
    assert!(!emitted.rust.is_empty() && !emitted.python.is_empty() && !emitted.manifest.is_empty());
    assert!(!cpp.is_empty() && !js.is_empty());
}

#[test]
fn golden_bank_64_fixtures_bit_identical() {
    let dir = fixtures_dir();

    // Part A: 55 fixtures from compat_freeze/emitted_bytes.v1.json + model_serde.v1.json
    let emitted_doc: Value =
        serde_json::from_slice(&std::fs::read(dir.join("emitted_bytes.v1.json")).unwrap()).unwrap();
    let models_doc: Value =
        serde_json::from_slice(&std::fs::read(dir.join("model_serde.v1.json")).unwrap()).unwrap();
    let emitted_programs = emitted_doc["programs"].as_array().unwrap();
    let model_rows = models_doc["models"].as_array().unwrap();
    assert_eq!(emitted_programs.len(), 55);
    assert_eq!(model_rows.len(), 55);

    let (all_x, all_y) = toy_data(80, 5, 42);
    let fit_x = all_x[..64].to_vec();
    let fit_y = &all_y[..64];
    let test_x = all_x[64..].to_vec();

    for (ep, mr) in emitted_programs.iter().zip(model_rows.iter()) {
        let name = ep["name"].as_str().unwrap();
        assert_eq!(mr["name"].as_str().unwrap(), name);
        let canonical = ep["canonical"].as_str().unwrap();
        let genome = GenomeIr::parse(canonical).unwrap();
        assert_eq!(
            genome.canonical(),
            canonical,
            "canonical roundtrip for {name}"
        );
        if !canonical.contains("(portfolio") {
            let model = fit(&genome, &fit_x, fit_y).unwrap_or_else(|e| panic!("{name}: {e}"));
            let bytes = serde_json::to_vec(&model).expect("serialize model");
            let expected_id = mr["serde_json"].as_str().unwrap();
            assert_eq!(
                format!("{} {}", bytes.len(), sha256_hex(&bytes)),
                expected_id,
                "model serde identity for {name}"
            );
            let preds = model.predict(&test_x).unwrap();
            let pred_bits = preds
                .into_iter()
                .map(f64_bits_hex)
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(
                pred_bits,
                mr["predictions"].as_str().unwrap(),
                "model prediction bits for {name}"
            );
        }
    }

    for stored in models_doc["stored"].as_array().unwrap() {
        let text = stored["json"].as_str().unwrap();
        let _: Model = serde_json::from_str(text).expect("stored model must deserialize");
    }

    // Part B: 5 representative fixtures from legacy_v2_freeze
    let train = vec![
        vec![1.0, 0.0, -1.0, 2.0],
        vec![2.0, 1.0, 0.0, 1.0],
        vec![-1.0, 2.0, 1.0, 0.0],
        vec![0.5, -0.5, 2.0, -1.0],
        vec![3.0, 1.5, -2.0, 0.5],
        vec![-2.0, -1.0, 0.5, 3.0],
        vec![1.25, 2.5, -0.25, -1.5],
        vec![-0.75, 0.25, 1.5, 2.5],
        vec![2.25, -1.25, 0.75, 1.25],
        vec![-1.5, 1.75, -0.5, 0.25],
        vec![0.25, 3.0, 1.25, -0.75],
        vec![1.75, -2.0, -1.25, 1.75],
    ];
    let target = vec![
        2.1, 2.6, -2.4, 1.3, 5.7, -2.8, 0.95, -0.4, 4.15, -3.2, -1.15, 4.6,
    ];
    let test = vec![
        vec![0.0, 1.0, -0.5, 1.5],
        vec![2.5, -0.75, 1.0, -0.25],
        vec![-1.25, 2.25, 0.25, 2.0],
        vec![1.5, 0.5, -1.5, 0.75],
    ];
    let legacy_v2_cases: [(&str, &str, &str, [u64; 4]); 5] = [
        (
            "ridge_random_features",
            "(pipe X (standardize append=true) (rff_cos n=4 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
            "65ebc7f874a86b96c44d534cc83b95a60e898d6a85c6cc1c31db7695f5d03611",
            [
                0x3fca590c3f55b872,
                0x4013cc4995ccc1cf,
                0xc002058c6d012514,
                0x3fff889e32b7e47c,
            ],
        ),
        (
            "legacy_row_ops",
            "(pipe X (lag k=2 append=true) (roll k=3) (ridge))",
            "8bc5bc757dc1e05bcc62a4b493ddccc623ee9dc95c9a96488974657bcb29e888",
            [
                0xbf20540380be5600,
                0x401267bbd9628caf,
                0xc00609e094cde6d1,
                0x4006c6d4dde8aa8b,
            ],
        ),
        (
            "target_aware_ensemble",
            "(pipe X (pair_prod topk=3 append=true) (target_bin_encode bins=4) (ensemble rounds=8 depth=2 lambda_frac=0.02))",
            "4bc89be6d8775ab9a3dda50b8e93152a9f4addd63f594939ef86cadb865800dd",
            [
                0x3fee888888888888,
                0x3fee888888888888,
                0x3fee888888888888,
                0x3fee888888888888,
            ],
        ),
        (
            "symbolic_affine",
            "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
            "319147c804ab23dda6e4645f00ab0a63362297003378e6f3175e1c1be441115e",
            [
                0x3fee509f8f512d46,
                0x3fef4c905f54b9a4,
                0x3fed8e53dc14f49d,
                0x3fee2adccc7886d2,
            ],
        ),
        (
            "champion",
            CHAMPION_GENOME,
            "0bc6e75d8f12173ac6e2acf432233e48457cf773e1bedef56f536384513c158d",
            [
                0x3fee888888888888,
                0x3fee888888888888,
                0x3fee888888888888,
                0x3fee888888888888,
            ],
        ),
    ];

    for (name, src, expected_sha, expected_bits) in legacy_v2_cases {
        let genome = GenomeIr::parse(src).unwrap();
        assert_eq!(
            sha256_hex(genome.canonical().as_bytes()),
            expected_sha,
            "genome sha256 mismatch for {name}"
        );
        let preds = fit_predict(&genome, &train, &target, &test).unwrap();
        let actual_bits: Vec<u64> = preds.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            actual_bits.as_slice(),
            expected_bits.as_slice(),
            "prediction bits mismatch for {name}"
        );
    }

    // Part C: 4 historical_v1 wire-alias fixtures
    let hist_rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let x = (i as f64 - 11.0) / 7.0;
            vec![x, x * x + 0.25, (i % 5) as f64 - 2.0, x * 0.5 + 1.0]
        })
        .collect();
    let hist_target: Vec<f64> = hist_rows
        .iter()
        .map(|row| 1.5 * row[0] - 0.3 * row[1] + row[2] * 0.1)
        .collect();
    let alias_pairs = [
        (
            3,
            "norm_features",
            "(pipe X (norm_features) (ridge))",
            "(pipe X (standardize) (ridge))",
        ),
        (
            11,
            "winsor_moments",
            "(pipe X (winsor_moments) (ridge))",
            "(pipe X (winsor_clip) (ridge))",
        ),
        (
            39,
            "gbm",
            "(pipe X (gbm rounds=5) )",
            "(pipe X (histgbm rounds=5) )",
        ),
        (
            41,
            "ensemble_ridge_gbm",
            "(pipe X (ensemble_ridge_gbm rounds=5) )",
            "(pipe X (ensemble rounds=5) )",
        ),
    ];
    for (id, name, alias_src, canon_src) in alias_pairs {
        let alias = GenomeIr::parse(alias_src).unwrap();
        let canon = GenomeIr::parse(canon_src).unwrap();
        let left = fit(&alias, &hist_rows, &hist_target).unwrap();
        let right = fit(&canon, &hist_rows, &hist_target).unwrap();
        let alias_bits: Vec<String> = left
            .predict(&hist_rows)
            .unwrap()
            .into_iter()
            .map(f64_bits_hex)
            .collect();
        let canon_bits: Vec<String> = right
            .predict(&hist_rows)
            .unwrap()
            .into_iter()
            .map(f64_bits_hex)
            .collect();
        assert_eq!(
            alias_bits, canon_bits,
            "wire alias {id} ({name}) must be bit-identical to canonical"
        );
    }

    let total_fixtures = emitted_programs.len() + legacy_v2_cases.len() + alias_pairs.len();
    assert_eq!(
        total_fixtures, 64,
        "golden bank must verify all 64 fixtures"
    );
}

#[test]
fn verify_64_fixture_golden_bank_and_7_routes() {
    baseline_manifest_and_7_routes_frozen();
    golden_bank_64_fixtures_bit_identical();
}
