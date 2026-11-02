#[cfg(test)]
const EXPECTED_GLOBAL_INDICES: [usize; 80] = [
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108,
];

#[cfg(test)]
const PYTHON_ORACLE_GOLDENS: &str = concat!(
    include_str!("tests/oracle/rows-1.txt"),
    include_str!("tests/oracle/rows-2.txt"),
    include_str!("tests/oracle/rows-3.txt")
);

use super::super::{parse, test_support};
use super::{
    extract, numeric_parameter_token, EXPECTED_GLOBAL_INDICES, PARAMETER_KEYS,
    PYTHON_ORACLE_GOLDENS,
};

#[test]
fn numeric_parameter_tokens_follow_python_float_boolean_and_seed_rules() {
    assert_token_value("lambda_frac=1e-3", 0.000_999_500_333_083_533_1);
    assert_token_value("lambda_frac=-2", -1.098_612_288_668_109_8);
    assert_token_value("lambda_frac=true", "0.6931471805599453".parse().unwrap());
    assert_eq!(
        numeric_parameter_token("lambda_frac=false"),
        Some(("lambda_frac", 0.0))
    );
    assert_eq!(numeric_parameter_token("seed=9"), None);
    assert_eq!(numeric_parameter_token("lambda_frac=1=2"), None);
    assert_eq!(numeric_parameter_token("lambda_frac=TRUE"), None);
    assert_token_value("lambda_frac=1_000.5", 6.909_254_155_105_873);
    assert!(numeric_parameter_token("lambda_frac=+Infinity")
        .unwrap()
        .1
        .is_infinite());
}

#[test]
fn duplicates_are_counted_and_summed_in_source_order_and_absent_keys_are_zero() {
    let parsed = parse::parse("(pipe X (ridge lambda_frac=1e-3 lambda_frac=-2 seed=9))").unwrap();
    let features = extract(&parsed);

    assert_eq!(features.len(), 80);
    assert_eq!(features["parameter.ridge.lambda_frac.count"], 2.0);
    assert!((features["parameter.ridge.lambda_frac.sum"] - (-1.097_612_788_335_026)).abs() < 1e-15);
    assert!(!features.contains_key("parameter.ridge_gcv.lambda_frac.count"));
    assert!(features.keys().all(|name| name.starts_with("parameter.")));
}

#[test]
fn unknown_operations_and_unowned_parameters_do_not_collide() {
    let parsed = parse::parse("(pipe X (fake lambda_frac=2) (ridge unknown=5))").unwrap();
    let features = extract(&parsed);

    assert_eq!(features["parameter.ridge.lambda_frac.count"], 0.0);
    assert_eq!(features["parameter.ridge.lambda_frac.sum"], 0.0);
}

#[test]
fn all_1309_row_free_programs_match_the_pinned_python_oracle() {
    let records = test_support::corpus();
    let names = parameter_feature_names();
    let frozen_names = test_support::names();
    let indices: Vec<_> = names
        .iter()
        .map(|name| {
            frozen_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap()
        })
        .collect();
    let expected_rows: Vec<_> = PYTHON_ORACLE_GOLDENS
        .lines()
        .filter(|line| !line.is_empty())
        .collect();
    assert_eq!(records.len(), 1309);
    assert_eq!(expected_rows.len(), records.len());
    assert_eq!(indices.len(), 80);
    assert_eq!(indices, EXPECTED_GLOBAL_INDICES);

    let mut compared = 0usize;
    let mut max_abs_error = [0.0_f64; 80];
    for (record, golden) in records.iter().zip(expected_rows) {
        let (program_hash, sparse_values) = golden.split_once('\t').unwrap();
        assert_eq!(record.program_sha256, program_hash);
        let expected = decode_sparse_values(sparse_values);
        let parsed = parse::parse(&record.dsl).unwrap();
        let actual = extract(&parsed);
        assert_eq!(actual.len(), 80);

        for (feature_index, name) in names.iter().enumerate() {
            let actual_value = actual.get(name.as_str()).unwrap();
            let error = (actual_value - expected[feature_index]).abs();
            if name.ends_with(".count") {
                assert_eq!(actual_value, &expected[feature_index], "{}", name);
            } else {
                assert!(error <= 1e-6, "{} error {error} > 1e-6", name);
            }
            max_abs_error[feature_index] = max_abs_error[feature_index].max(error);
            compared += 1;
        }
    }

    eprintln!(
            "RNK-04-A Python oracle comparison: records={}, unique_programs=1292, coordinates={}, values_compared={compared}, max_abs_error={}",
            records.len(),
            indices.len(),
            max_abs_error.iter().copied().fold(0.0_f64, f64::max)
        );
    for (name, error) in names.iter().zip(max_abs_error) {
        eprintln!("{name} max_abs_error={error:.17}");
    }
}

/// Serving FP32 outputs for the admitted model-prior subset, not full M3.
#[test]
#[ignore = "requires pinned RES-07 program tensors and exported feature order"]
fn real_model_prior_parameter_features_match_frozen_oracle() {
    use safetensors::{Dtype, SafeTensors};
    use std::{collections::BTreeSet, env, fs, path::Path};

    let read_pinned = |path: &Path, pin: &str| {
        assert!(
            pin.len() == 64
                && pin
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "explicit lowercase SHA256 required"
        );
        assert!(path.is_absolute(), "fixture paths must be absolute");
        assert_eq!(
            fs::canonicalize(path).unwrap(),
            path,
            "no symlink traversal"
        );
        let metadata = fs::symlink_metadata(path).expect("required real fixture");
        assert!(metadata.is_file() && metadata.len() <= 64 * 1024 * 1024);
        let bytes = fs::read(path).unwrap();
        assert_eq!(jope_runtime::sha256_hex(&bytes), pin, "fixture pin differs");
        bytes
    };
    let root = env::var_os("JOPE_GOLDEN_DIR").expect("JOPE_GOLDEN_DIR is required");
    let root = Path::new(&root);
    let index_pin =
        env::var("JOPE_GOLDEN_INDEX_SHA256").expect("explicit trusted index SHA256 is required");
    let feature_path =
        env::var_os("JOPE_GOLDEN_FEATURIZER_JSON").expect("exported featurizer JSON is required");
    let feature_pin = env::var("JOPE_GOLDEN_FEATURIZER_SHA256")
        .expect("explicit trusted featurizer SHA256 is required");
    let index: serde_json::Value =
        serde_json::from_slice(&read_pinned(&root.join("index.json"), &index_pin)).unwrap();
    assert_eq!(index["schema"], "jope.golden-fixtures/v1");
    assert_eq!(index["subset"]["role"], "model_prior");
    assert_eq!(index["model_prior_programs"], 872);
    assert_eq!(index["subset"]["programs"], 872);
    assert_eq!(
        index["bundle_digest"],
        "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
    );
    let feature_json: serde_json::Value =
        serde_json::from_slice(&read_pinned(Path::new(&feature_path), &feature_pin)).unwrap();
    let names: Vec<String> = serde_json::from_value(feature_json["feature_names"].clone()).unwrap();
    let featurizer = super::super::Featurizer::from_names(names).unwrap();
    let columns: Vec<_> = featurizer
        .feature_names()
        .iter()
        .enumerate()
        .filter(|(_, name)| name.starts_with("parameter."))
        .collect();
    assert_eq!(columns.len(), 80);
    let payload = |name: &str| {
        let spec = &index["files"][name];
        let bytes = read_pinned(&root.join(name), spec["sha256"].as_str().unwrap());
        assert_eq!(bytes.len() as u64, spec["bytes"].as_u64().unwrap());
        bytes
    };
    let programs = payload("programs.jsonl");
    let programs: Vec<serde_json::Value> = std::str::from_utf8(&programs)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(programs.len(), 872);
    let oracle = payload("programs.safetensors");
    let tensors = SafeTensors::deserialize(&oracle).unwrap();
    let features = tensors.tensor("feat412").unwrap();
    assert_eq!(features.dtype(), Dtype::F32);
    assert_eq!(
        features.shape(),
        [programs.len(), super::super::FEATURE_COUNT]
    );
    let mut unique = BTreeSet::new();
    let mut maximum = [0.0_f64; 80];
    for (row, program) in programs.iter().enumerate() {
        let dsl = program["dsl"].as_str().expect("DSL required");
        let pin = program["program_sha256"]
            .as_str()
            .expect("program identity required");
        assert_eq!(jope_runtime::sha256_hex(dsl.as_bytes()), pin);
        assert!(unique.insert(pin), "duplicate program identity");
        let actual = extract(&parse::parse(dsl).unwrap());
        assert_eq!(actual.len(), 80);
        for (family_column, &(column, name)) in columns.iter().enumerate() {
            let offset = (row * super::super::FEATURE_COUNT + column) * 4;
            let expected =
                f32::from_le_bytes(features.data()[offset..offset + 4].try_into().unwrap());
            let observed = *actual.get(name).expect("missing parameter coordinate") as f32;
            assert!(expected.is_finite() && observed.is_finite());
            let error = (f64::from(observed) - f64::from(expected)).abs();
            if name.ends_with(".count") {
                assert_eq!(observed, expected, "program {row} {pin}: {name}");
            } else {
                assert!(error <= 1e-6, "program {row} {pin}: {name} error {error}");
            }
            maximum[family_column] = maximum[family_column].max(error);
        }
    }
    for ((column, name), error) in columns.iter().zip(maximum) {
        println!("{column} {name} max_abs_error={error:.17}");
    }
    println!(
        "{}",
        serde_json::json!({
            "gate":"RNK-04-A parameter serialized-oracle parity",
            "scope":"frozen model_prior subset", "programs":programs.len(),
            "coordinates_per_program":80,"comparisons":programs.len()*80,"mismatches":0,
            "max_abs_error":maximum.into_iter().fold(0.0_f64,f64::max),
            "index_sha256":index_pin,"featurizer_sha256":feature_pin,"full_historical_M3":false,
            "compiled_source_sha256":jope_runtime::sha256_hex(include_str!("../parameters.rs").as_bytes())
        })
    );
}

fn parameter_feature_names() -> Vec<String> {
    PARAMETER_KEYS
        .iter()
        .flat_map(|(operation, key)| {
            [
                format!("parameter.{operation}.{key}.count"),
                format!("parameter.{operation}.{key}.sum"),
            ]
        })
        .collect()
}

fn decode_sparse_values(encoded: &str) -> [f64; 80] {
    let mut values = [0.0; 80];
    if encoded.is_empty() || encoded == "-" {
        return values;
    }
    for item in encoded.split(',') {
        let (index, value) = item.split_once(':').unwrap();
        values[index.parse::<usize>().unwrap()] = value.parse::<f64>().unwrap();
    }
    values
}

fn assert_token_value(token: &str, expected: f64) {
    let (key, actual) = numeric_parameter_token(token).unwrap();
    assert_eq!(key, "lambda_frac");
    assert!(
        (actual - expected).abs() < 1e-15,
        "{token}: {actual} != {expected}"
    );
}
