//! Fast legacy-v2 behavior freeze.
//!
//! The broad `golden` integration target remains the cross-language oracle.
//! This target pins a compact representative corpus so every advanced-v3
//! change can cheaply prove that legacy parsing, hashing, prediction, RNG,
//! folds, and emission stayed byte-compatible.

use feat_math::transpile::{emit, EmitOptions};
use feat_math::{fit_predict, fold_of, GenomeIr, Pcg32, CHAMPION_GENOME};
use sha2::{Digest, Sha256};

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn fixture_data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
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
    (train, target, test)
}

fn fixture_genomes() -> [(&'static str, &'static str); 5] {
    [
        (
            "ridge_random_features",
            "(pipe X (standardize append=true) (rff_cos n=4 seed=9 gamma=1.5 append=true) (ridge lambda_frac=0.05))",
        ),
        (
            "legacy_row_ops",
            "(pipe X (lag k=2 append=true) (roll k=3) (ridge))",
        ),
        (
            "target_aware_ensemble",
            "(pipe X (pair_prod topk=3 append=true) (target_bin_encode bins=4) (ensemble rounds=8 depth=2 lambda_frac=0.02))",
        ),
        (
            "symbolic_affine",
            "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
        ),
        ("champion", CHAMPION_GENOME),
    ]
}

#[test]
#[ignore = "prints reviewed constants for an explicitly acknowledged fixture update"]
fn regenerate_legacy_v2_freeze() {
    let (train, target, test) = fixture_data();
    for (name, source) in fixture_genomes() {
        let genome = GenomeIr::parse(source).unwrap();
        let canonical = genome.canonical();
        let emitted = emit(&genome, &EmitOptions::default()).unwrap();
        let prediction_bits: Vec<String> = fit_predict(&genome, &train, &target, &test)
            .unwrap()
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect();
        println!("FIXTURE {name}");
        println!("canonical={canonical:?}");
        println!("genome_sha256={}", sha256_hex(canonical.as_bytes()));
        println!(
            "python=({}, {}, {}, {})",
            emitted.python.len(),
            sha256_hex(emitted.python.as_bytes()),
            emitted.py_tokens.code_tokens,
            emitted.py_tokens.doc_tokens
        );
        println!(
            "rust=({}, {}, {}, {})",
            emitted.rust.len(),
            sha256_hex(emitted.rust.as_bytes()),
            emitted.rs_tokens.code_tokens,
            emitted.rs_tokens.doc_tokens
        );
        println!(
            "manifest=({}, {})",
            emitted.manifest.len(),
            sha256_hex(emitted.manifest.as_bytes())
        );
        println!("prediction_bits={prediction_bits:?}");
    }

    for source in ["", "garbage", "(pipe X (ridge)", "(portfolio (blend))"] {
        println!(
            "PARSER {source:?} => {:?}",
            GenomeIr::parse(source).unwrap_err().to_string()
        );
    }

    let mut rng = Pcg32::new(42, 54);
    let u32_trace: Vec<String> = (0..8).map(|_| format!("{:08x}", rng.next_u32())).collect();
    println!("RNG u32={u32_trace:?}");
    let mut rng = Pcg32::new(7, 1);
    let f64_trace: Vec<String> = (0..4)
        .map(|_| format!("{:016x}", rng.next_f64().to_bits()))
        .collect();
    println!("RNG f64={f64_trace:?}");
    let mut rng = Pcg32::new(7, 1);
    let gaussian_trace: Vec<String> = (0..4)
        .map(|_| format!("{:016x}", rng.gaussian().to_bits()))
        .collect();
    println!("RNG gaussian={gaussian_trace:?}");
    let folds: Vec<usize> = (0..24)
        .map(|row| fold_of("legacy-v2-freeze", row, 5))
        .collect();
    println!("FOLDS {folds:?}");
}

struct ExpectedFixture {
    name: &'static str,
    canonical: &'static str,
    genome_sha256: &'static str,
    python_len: usize,
    python_sha256: &'static str,
    python_code_tokens: usize,
    python_doc_tokens: usize,
    rust_len: usize,
    rust_sha256: &'static str,
    rust_code_tokens: usize,
    rust_doc_tokens: usize,
    manifest_len: usize,
    manifest_sha256: &'static str,
    prediction_bits: &'static [u64],
}

const EXPECTED_FIXTURES: &[ExpectedFixture] = &[
    ExpectedFixture {
        name: "ridge_random_features",
        canonical: "(pipe X (standardize append=true) (rff_cos append=true gamma=1.5 n=4 seed=9) (ridge lambda_frac=0.05))",
        genome_sha256: "65ebc7f874a86b96c44d534cc83b95a60e898d6a85c6cc1c31db7695f5d03611",
        python_len: 12_555,
        python_sha256: "14b0bedb2f7d108114d2b8911c97470221bd8fd076b719e928ab2c03fe128b6d",
        python_code_tokens: 2_087,
        python_doc_tokens: 598,
        rust_len: 15_342,
        rust_sha256: "15529624f7d1f6927138b993103892fee598a995c31f01162205c5573f95b29a",
        rust_code_tokens: 3_104,
        rust_doc_tokens: 553,
        manifest_len: 429,
        manifest_sha256: "636eeee06e2d23fce3b084624ca7e2f055d70ce6fab923e241e113a6bbfb46db",
        prediction_bits: &[
            0x3fca590c3f55b872,
            0x4013cc4995ccc1cf,
            0xc002058c6d012514,
            0x3fff889e32b7e47c,
        ],
    },
    ExpectedFixture {
        name: "legacy_row_ops",
        canonical: "(pipe X (lag append=true k=2) (roll k=3) (ridge))",
        genome_sha256: "8bc5bc757dc1e05bcc62a4b493ddccc623ee9dc95c9a96488974657bcb29e888",
        python_len: 9_542,
        python_sha256: "253c57121ec8a70c38face7a1e97cc48fa77c78fa4d3ce573dde8833b3e62b62",
        python_code_tokens: 1_641,
        python_doc_tokens: 471,
        rust_len: 11_687,
        rust_sha256: "fd9e4f4c396a4558db49f2e7ff180b9188937d603afd97d97e433f7bdb2b2a47",
        rust_code_tokens: 2_411,
        rust_doc_tokens: 455,
        manifest_len: 376,
        manifest_sha256: "1fe0423ac9a572634c669c9d499ea10483f60aedcf192de4279e473cc1a3ae64",
        prediction_bits: &[
            0xbf20540380be5600,
            0x401267bbd9628caf,
            0xc00609e094cde6d1,
            0x4006c6d4dde8aa8b,
        ],
    },
    ExpectedFixture {
        name: "target_aware_ensemble",
        canonical: "(pipe X (pair_prod append=true topk=3) (target_bin_encode bins=4) (ensemble depth=2 lambda_frac=0.02 rounds=8))",
        genome_sha256: "4bc89be6d8775ab9a3dda50b8e93152a9f4addd63f594939ef86cadb865800dd",
        python_len: 19_781,
        python_sha256: "3bd937b61d87dd3d0b2d8a8a792256dd26b5a8734ee692ece245ac63a183b61d",
        python_code_tokens: 3_893,
        python_doc_tokens: 667,
        rust_len: 25_612,
        rust_sha256: "0bc79ba5898d035327800309657a02a44812a34c19391e556224ba6b48bbbf13",
        rust_code_tokens: 5_705,
        rust_doc_tokens: 651,
        manifest_len: 438,
        manifest_sha256: "b7299a100fb66b32b7856ec1eeaad06d52320713dd930196ab40b6a63c9522af",
        prediction_bits: &[
            0x3fee888888888888,
            0x3fee888888888888,
            0x3fee888888888888,
            0x3fee888888888888,
        ],
    },
    ExpectedFixture {
        name: "symbolic_affine",
        canonical: "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
        genome_sha256: "319147c804ab23dda6e4645f00ab0a63362297003378e6f3175e1c1be441115e",
        python_len: 6_035,
        python_sha256: "c9a837866a69a8a6a7c9e449189acafb4f8ece48f22b72a889162f598c174eed",
        python_code_tokens: 857,
        python_doc_tokens: 294,
        rust_len: 7_172,
        rust_sha256: "e7e3188ed6e7af417557dd05aa724555f8ec0e0321cb92f1a75c99fc3949b5d9",
        rust_code_tokens: 1_322,
        rust_doc_tokens: 278,
        manifest_len: 372,
        manifest_sha256: "eee99bb7bc6792882c08037761b43fceda7476f4cf84256d1959c51277a44a44",
        prediction_bits: &[
            0x3fee509f8f512d46,
            0x3fef4c905f54b9a4,
            0x3fed8e53dc14f49d,
            0x3fee2adccc7886d2,
        ],
    },
    ExpectedFixture {
        name: "champion",
        canonical: "(pipe X (winsor_clip append=true q=0.1) (cheb append=true degree=6) (robust_scale) (robust_scale) (poly_interact append=true) (quantile_bins bins=16) (histgbm depth=5 lr=0.15 rounds=150))",
        genome_sha256: "0bc6e75d8f12173ac6e2acf432233e48457cf773e1bedef56f536384513c158d",
        python_len: 16_991,
        python_sha256: "0a3c2da6f86cda8d042a2d4f6e9ba6c35e08798b89ef05b63f56cb7096f85b5d",
        python_code_tokens: 2_811,
        python_doc_tokens: 725,
        rust_len: 22_188,
        rust_sha256: "512098bc4a760614a50eac24586fca5de3d44faf7ef1dc70f5af8e44d570d1f6",
        rust_code_tokens: 4_459,
        rust_doc_tokens: 709,
        manifest_len: 514,
        manifest_sha256: "c9f57b06619cd72b165128fd4b9db9d0af7146e8f78eda82ea3df09b789b2f39",
        prediction_bits: &[
            0x3fee888888888888,
            0x3fee888888888888,
            0x3fee888888888888,
            0x3fee888888888888,
        ],
    },
];

#[test]
fn legacy_v2_genomes_predictions_and_emissions_are_frozen() {
    let (train, target, test) = fixture_data();
    let actual_genomes = fixture_genomes();
    assert_eq!(actual_genomes.len(), EXPECTED_FIXTURES.len());

    for ((name, source), expected) in actual_genomes.into_iter().zip(EXPECTED_FIXTURES) {
        assert_eq!(name, expected.name);
        let genome = GenomeIr::parse(source).unwrap();
        let canonical = genome.canonical();
        assert_eq!(canonical, expected.canonical, "{name}: canonical genome");
        assert_eq!(
            sha256_hex(canonical.as_bytes()),
            expected.genome_sha256,
            "{name}: genome hash"
        );

        let predictions = fit_predict(&genome, &train, &target, &test).unwrap();
        let prediction_bits: Vec<u64> = predictions.iter().map(|value| value.to_bits()).collect();
        assert_eq!(
            prediction_bits, expected.prediction_bits,
            "{name}: predictions"
        );

        let emitted = emit(&genome, &EmitOptions::default()).unwrap();
        assert_eq!(
            emitted.python.len(),
            expected.python_len,
            "{name}: python length"
        );
        assert_eq!(
            sha256_hex(emitted.python.as_bytes()),
            expected.python_sha256,
            "{name}: python hash"
        );
        assert_eq!(
            emitted.py_tokens.code_tokens, expected.python_code_tokens,
            "{name}: python code tokens"
        );
        assert_eq!(
            emitted.py_tokens.doc_tokens, expected.python_doc_tokens,
            "{name}: python doc tokens"
        );
        assert_eq!(emitted.rust.len(), expected.rust_len, "{name}: rust length");
        assert_eq!(
            sha256_hex(emitted.rust.as_bytes()),
            expected.rust_sha256,
            "{name}: rust hash"
        );
        assert_eq!(
            emitted.rs_tokens.code_tokens, expected.rust_code_tokens,
            "{name}: rust code tokens"
        );
        assert_eq!(
            emitted.rs_tokens.doc_tokens, expected.rust_doc_tokens,
            "{name}: rust doc tokens"
        );
        assert_eq!(
            emitted.manifest.len(),
            expected.manifest_len,
            "{name}: manifest length"
        );
        assert_eq!(
            sha256_hex(emitted.manifest.as_bytes()),
            expected.manifest_sha256,
            "{name}: manifest hash"
        );
    }
}

#[test]
fn legacy_v2_parser_errors_rng_and_folds_are_frozen() {
    for (source, expected_error) in [
        ("", "unrecognized genome form: "),
        ("garbage", "unrecognized genome form: garbage"),
        ("(pipe X (ridge)", "expected '(' at token 6, got None"),
        (
            "(portfolio (blend))",
            "portfolio: member count must be 1..12, got 0",
        ),
    ] {
        assert_eq!(
            GenomeIr::parse(source).unwrap_err().to_string(),
            expected_error,
            "parser error for {source:?}"
        );
    }

    let mut rng = Pcg32::new(42, 54);
    let u32_trace: Vec<u32> = (0..8).map(|_| rng.next_u32()).collect();
    assert_eq!(
        u32_trace,
        [
            0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e, 0xbfc6a3ad,
            0x812fff6d,
        ]
    );

    let mut rng = Pcg32::new(7, 1);
    let f64_trace: Vec<u64> = (0..4).map(|_| rng.next_f64().to_bits()).collect();
    assert_eq!(
        f64_trace,
        [
            0x3fe081b3394258ea,
            0x3fdd206d083e84c4,
            0x3fdaf369a55a62ca,
            0x3fd942b135e69fe8,
        ]
    );

    let mut rng = Pcg32::new(7, 1);
    let gaussian_trace: Vec<u64> = (0..4).map(|_| rng.gaussian().to_bits()).collect();
    assert_eq!(
        gaussian_trace,
        [
            0xbff1aeb300bcc123,
            0xbff09a45b8e09993,
            0x3fb694d2892b3071,
            0x3ffe56d4561eb848,
        ]
    );

    let folds: Vec<usize> = (0..24)
        .map(|row| fold_of("legacy-v2-freeze", row, 5))
        .collect();
    assert_eq!(
        folds,
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3, 4, 0, 1, 3, 4, 0, 1,]
    );
}

#[test]
fn legacy_v2_reference_files_match_the_takeover_snapshot() {
    for (path, contents, expected_sha256) in [
        (
            // Re-pinned 2026-09-18 after `cargo fmt` over the crate (ADR-0002 step 1): the
            // golden.rs was re-pinned after `cargo fmt`: `rustfmt` over the 53ba1f8 copy is byte-identical to
            // the 4235db6 copy (sha256 91f759100481…), so the change is formatting only. Note the raw
            // `git diff -w 53ba1f8 4235db6 -- crates/feat-math/tests/golden.rs` is NOT empty (56 lines):
            // rustfmt also added trailing commas when it wrapped arguments.
            "tests/golden.rs",
            include_str!("golden.rs"),
            // Re-pinned: three `&PathBuf` parameters became `&Path` (clippy::ptr_arg). Callers
            // reach them by deref coercion, so no call site changed. A freeze update, not a
            // semantic change: `regenerate_legacy_v2_freeze` reports every genome_sha256,
            // prediction_bits, RNG stream and fold assignment unchanged, and all 83 golden
            // parity cases still agree bit-for-bit.
            // Previous: 91f759100481b3fbbf8f85d626eba63140f7d7ead6e0e778ecc4b59ac3c6d0c9
            "d8cc5d006f4ffa2e3a40840b08b13410f9ecca73f0ddad1378a3aa29296c171e",
        ),
        (
            "tests/champion_parity.rs",
            include_str!("champion_parity.rs"),
            // Re-pinned: `emitted_python_matches_interpreter_predictions` gained the
            // JOPE_PARITY_STRICT guard required by ADR-0002 section 4, so a missing python3
            // fails instead of returning `Ok` having asserted nothing. A freeze update, not a
            // semantic change: `regenerate_legacy_v2_freeze` reports every genome_sha256,
            // emission hash and prediction_bits fixture unchanged.
            // Previous: cb79da4b7f64555ba51ec60b6a6ffcf72ac9325ea03bedc8d8130472fd8d1e85
            "69d4db59e21dad0a0f05df3e463d44e76bbddb0dfcbf9d7b32ee23c7826fab53",
        ),
        (
            "tests/tokens.rs",
            include_str!("tokens.rs"),
            // Re-pinned: four `rng.next_u32() % n == 0` became `.is_multiple_of(n)`
            // (clippy::manual_is_multiple_of), applied by `cargo clippy --fix`. Identical for
            // unsigned operands, and the RNG draw order is untouched, so the generated genome
            // corpus is the same. A freeze update, not a semantic change:
            // `regenerate_legacy_v2_freeze` reports every fixture unchanged.
            // Previous: b51814b0c5e463d0f883e50fbbbc97bd7d34ab2bfa5d51de979ce6551cdafdbe
            "b95c8d69424a651cac06edbde0d940edd9fad62db36bcb8a42d2ebe3d0a23031",
        ),
        (
            "tests/bench.rs",
            include_str!("bench.rs"),
            "9522042ea826d19a784f14feaa1509de9fc64f37058488bea00156ba321af42a",
        ),
        (
            "src/rng.rs",
            include_str!("../src/rng.rs"),
            "8b947b489096a4e761044078aea863040db94d06d80598e83fc385edce145852",
        ),
        (
            "src/folds.rs",
            include_str!("../src/folds.rs"),
            "e1cf7c3d9df9f374902420de75ccd453b634ba7a3f86b5252b942fdf4cecc8a1",
        ),
    ] {
        assert_eq!(
            sha256_hex(contents.as_bytes()),
            expected_sha256,
            "{path} differs from the G-00 takeover snapshot"
        );
    }
}
