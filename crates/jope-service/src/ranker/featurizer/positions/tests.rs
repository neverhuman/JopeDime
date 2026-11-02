use super::*;
use crate::ranker::featurizer::test_support;
use std::collections::BTreeSet;

#[test]
fn shared_position_namespace_obeys_operator_then_parameter_overwrite_order() {
    let parsed = parse::parse(
            "(pipe X (append append=true append=false seed=8) (lambda_frac ridge=2 ridge=-2) (mystery ridge=true q=1_000.5))",
        )
        .unwrap();
    let features = extract(&parsed);

    assert_eq!(features.len(), 302);
    assert_eq!(features["position0.append"], 0.0);
    assert_eq!(features["position1.lambda_frac"], 1.0);
    assert!((features["position1.ridge"] - (-1.098_612_288_668_109_6)).abs() <= 1e-15);
    assert!(
        (features["position2.ridge"] - "0.6931471805599453".parse::<f64>().unwrap()).abs() <= 1e-15
    );
    assert!((features["position2.q"] - 6.909_254_155_105_873).abs() <= 1e-15);
    assert!(!features.contains_key("position0.ridge"));
    assert_eq!(features["position0.bins"], 0.0);
    assert!(!features.contains_key("position2.mystery"));
    assert!(features.keys().all(|name| name.starts_with("position")));
}

#[test]
fn position_coordinates_past_the_frozen_namespace_project_away() {
    let dsl = format!("(pipe X {})", "(mystery)".repeat(10));
    let parsed = parse::parse(&dsl).unwrap();
    let features = extract(&parsed);
    let structure = super::super::structural::extract(&parsed);

    assert_eq!(parsed.operations().len(), 10);
    assert_eq!(features.len(), 302);
    assert!(features.values().all(|value| *value == 0.0));
    assert_eq!(structure["steps"], 10.0);
    assert!(!features.contains_key("position9.mystery"));
}

#[test]
fn all_1309_row_free_programs_match_the_pinned_python_position_oracle() {
    let records = test_support::corpus();
    let frozen_names = test_support::names();
    let expected_names: Vec<_> = frozen_names
        .iter()
        .filter(|name| name.starts_with("position"))
        .map(String::as_str)
        .collect();
    let actual_names: Vec<_> = POSITION_FEATURE_NAMES.to_vec();
    let global_indices: Vec<_> = frozen_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| name.starts_with("position").then_some(index))
        .collect();
    let expected_rows: Vec<_> = PYTHON_ORACLE_GOLDENS
        .lines()
        .filter(|line| !line.is_empty())
        .collect();

    assert_eq!(records.len(), 1309);
    assert_eq!(expected_rows.len(), records.len());
    assert_eq!(expected_names, actual_names);
    assert_eq!(global_indices, EXPECTED_GLOBAL_INDICES);
    assert_eq!(global_indices.len(), 302);
    assert_eq!(
        BTreeSet::from_iter(records.iter().map(|row| row.program_sha256.as_str())).len(),
        1292
    );

    let mut compared = 0usize;
    let mut max_abs_error = [0.0_f64; 302];
    let mut nonzero_expected = 0usize;
    for (record, golden) in records.iter().zip(expected_rows) {
        let (program_hash, sparse_values) = golden.split_once('\t').unwrap();
        assert_eq!(record.program_sha256, program_hash);
        let expected = decode_sparse_values(sparse_values);
        let parsed = parse::parse(&record.dsl).unwrap();
        let actual = extract(&parsed);
        assert_eq!(actual.len(), 302);

        for (feature_index, name) in POSITION_FEATURE_NAMES.iter().enumerate() {
            let actual_value = actual.get(*name).unwrap();
            let expected_value = expected[feature_index];
            let error = (actual_value - expected_value).abs();
            if expected_value == 0.0 || expected_value == 1.0 {
                assert_eq!(
                    *actual_value, expected_value,
                    "record {}, {name}",
                    record.program_sha256
                );
            } else {
                assert!(
                    error <= 1e-6,
                    "record {}, {name}: error {error} > 1e-6",
                    record.program_sha256
                );
            }
            max_abs_error[feature_index] = max_abs_error[feature_index].max(error);
            nonzero_expected += usize::from(expected_value != 0.0);
            compared += 1;
        }
    }

    eprintln!(
            "RNK-04-B Python oracle comparison: records={}, unique_programs=1292, coordinates={}, values_compared={compared}, nonzero_values={nonzero_expected}, max_abs_error={}",
            records.len(),
            global_indices.len(),
            max_abs_error.iter().copied().fold(0.0_f64, f64::max)
        );
    for (name, error) in POSITION_FEATURE_NAMES.iter().zip(max_abs_error) {
        eprintln!("{name} max_abs_error={error:.17}");
    }
}

#[test]
#[ignore = "requires the hash-pinned RES-07 model-prior packet; this subset is not full M3"]
fn all_872_res07_model_prior_programs_match_exported_position_coordinates() {
    use safetensors::{Dtype, SafeTensors};
    use serde::Deserialize;
    use std::{
        env, fs,
        path::{Component, Path, PathBuf},
    };

    const INDEX_SHA256: &str = "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393";
    const PROGRAMS_JSONL_SHA256: &str =
        "3a56531c37b1bceacf2c38360abffd38d5eb1fe64c6eda4beceec079f3c6dd34";
    const PROGRAMS_TENSORS_SHA256: &str =
        "3efb3b200555892a53c51c1b28f40016564bce8eec900669aea611954392a760";
    const REFERENCE_SOURCE_PATH: &str = "home/ubuntu/JopeDimeMicro/python/rank_prepare.py";
    const FEATURIZER_SHA256: &str =
        "5f2f343464559ae292817991d10ce295d23e7021d30a4f0b8d4773e19d800e52";

    fn checked_file(root: &Path, name: &str, expected_sha: &str) -> Vec<u8> {
        assert!(Path::new(name)
            .components()
            .all(|part| matches!(part, Component::Normal(_))));
        let path = root.join(name);
        assert!(!fs::symlink_metadata(&path)
            .expect("required pinned parity file")
            .file_type()
            .is_symlink());
        let canonical = path.canonicalize().expect("canonical parity file");
        assert!(
            canonical.starts_with(root),
            "parity file escaped fixture root"
        );
        assert!(fs::metadata(&canonical).unwrap().len() <= 64 * 1024 * 1024);
        let bytes = fs::read(canonical).unwrap();
        assert_eq!(jope_runtime::sha256_hex(&bytes), expected_sha, "{name}");
        bytes
    }

    #[derive(Deserialize)]
    struct ProgramRow {
        dsl: String,
        program_sha256: String,
    }

    let root = PathBuf::from(
        env::var_os("JOPE_RNK04_POSITION_FIXTURE_DIR")
            .expect("explicit RES-07 row-free fixture directory required"),
    )
    .canonicalize()
    .expect("canonical fixture directory");
    let index_bytes = checked_file(&root, "index.json", INDEX_SHA256);
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
    assert_eq!(index["schema"], "jope.golden-fixtures/v1");
    assert_eq!(
        index["bundle_digest"],
        "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
    );
    assert_eq!(index["model_prior_programs"], 872);
    assert_eq!(index["subset"]["role"], "model_prior");
    assert_eq!(index["subset"]["historical_corpus_complete"], false);
    assert_eq!(index["qualification"]["M3"], false);
    let reference_source_key = format!("/{REFERENCE_SOURCE_PATH}");
    assert_eq!(
        index["reference_source_sha256"][&reference_source_key],
        crate::ranker::featurizer::REFERENCE_SOURCE_SHA256
    );

    let rows_bytes = checked_file(&root, "programs.jsonl", PROGRAMS_JSONL_SHA256);
    assert_eq!(
        index["files"]["programs.jsonl"]["sha256"],
        PROGRAMS_JSONL_SHA256
    );
    assert_eq!(
        index["files"]["programs.jsonl"]["bytes"],
        rows_bytes.len() as u64
    );
    let rows: Vec<ProgramRow> = String::from_utf8(rows_bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 872);

    let tensor_bytes = checked_file(&root, "programs.safetensors", PROGRAMS_TENSORS_SHA256);
    assert_eq!(
        index["files"]["programs.safetensors"]["sha256"],
        PROGRAMS_TENSORS_SHA256
    );
    assert_eq!(
        index["files"]["programs.safetensors"]["bytes"],
        tensor_bytes.len() as u64
    );
    let tensors = SafeTensors::deserialize(&tensor_bytes).unwrap();
    assert_eq!(tensors.len(), 2);
    let feat412 = tensors.tensor("feat412").unwrap();
    assert_eq!(feat412.dtype(), Dtype::F32);
    assert_eq!(feat412.shape(), &[872, 412]);
    assert_eq!(tensors.tensor("proj16").unwrap().dtype(), Dtype::F64);
    assert_eq!(tensors.tensor("proj16").unwrap().shape(), &[872, 16]);

    let metadata_root = PathBuf::from(
        env::var_os("JOPE_RNK04_POSITION_METADATA_DIR")
            .expect("explicit frozen RES-06 featurizer metadata directory required"),
    )
    .canonicalize()
    .expect("canonical metadata directory");
    let metadata_bytes = checked_file(&metadata_root, "featurizer.json", FEATURIZER_SHA256);
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes).unwrap();
    let frozen_names: Vec<String> =
        serde_json::from_value(metadata["feature_names"].clone()).unwrap();
    assert_eq!(frozen_names.len(), 412);
    assert_eq!(
        frozen_names.iter().map(String::as_str).collect::<Vec<_>>(),
        super::super::FEATURE_NAMES
    );

    let expected_position_names: Vec<_> = frozen_names
        .iter()
        .filter(|name| name.starts_with("position"))
        .map(String::as_str)
        .collect();
    let actual_position_names: Vec<_> = POSITION_FEATURE_NAMES.to_vec();
    let global_indices: Vec<_> = frozen_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| name.starts_with("position").then_some(index))
        .collect();
    assert_eq!(expected_position_names, actual_position_names);
    assert_eq!(global_indices, EXPECTED_GLOBAL_INDICES);
    assert_eq!(global_indices.len(), 302);

    let mut program_hashes = BTreeSet::new();
    let mut compared = 0usize;
    let mut nonzero_expected = 0usize;
    let mut max_abs_error = 0.0_f64;
    for (program_index, row) in rows.iter().enumerate() {
        assert_eq!(
            jope_runtime::sha256_hex(row.dsl.as_bytes()),
            row.program_sha256
        );
        assert!(
            program_hashes.insert(row.program_sha256.as_str()),
            "duplicate model-prior program"
        );
        let parsed = parse::parse(&row.dsl).unwrap();
        let actual = extract(&parsed);
        assert_eq!(actual.len(), 302);

        for (name, global_index) in POSITION_FEATURE_NAMES.iter().zip(&global_indices) {
            let actual_value = *actual.get(*name).unwrap();
            let actual_fp32 = actual_value as f32;
            assert!(actual_fp32.is_finite(), "{}: {name}", row.program_sha256);
            let offset = (program_index * 412 + *global_index) * 4;
            let expected_fp32 =
                f32::from_le_bytes(feat412.data()[offset..offset + 4].try_into().unwrap());
            assert!(expected_fp32.is_finite(), "{}: {name}", row.program_sha256);
            let error = (f64::from(actual_fp32) - f64::from(expected_fp32)).abs();
            if expected_fp32 == 0.0 || expected_fp32 == 1.0 {
                assert_eq!(actual_fp32, expected_fp32, "{}: {name}", row.program_sha256);
            } else {
                assert!(
                    error <= 1e-6,
                    "{}: {name} error {error} exceeds 1e-6",
                    row.program_sha256
                );
            }
            max_abs_error = max_abs_error.max(error);
            nonzero_expected += usize::from(expected_fp32 != 0.0);
            compared += 1;
        }
    }
    assert_eq!(program_hashes.len(), 872);
    assert_eq!(compared, 872 * 302);

    let report = serde_json::json!({
        "consumer": "RNK-04-B-position-family",
        "device": "cpu",
        "index_sha256": INDEX_SHA256,
        "programs_jsonl_sha256": PROGRAMS_JSONL_SHA256,
        "programs_safetensors_sha256": PROGRAMS_TENSORS_SHA256,
        "featurizer_sha256": FEATURIZER_SHA256,
        "programs": rows.len(),
        "unique_programs": program_hashes.len(),
        "coordinates_per_program": 302,
        "values_compared": compared,
        "nonzero_expected": nonzero_expected,
        "max_absolute_error_after_fp32_rounding": max_abs_error,
        "mismatches": 0,
        "qualification": {"M3": false, "historical_corpus_complete": false}
    });
    let report_dir = PathBuf::from(
        env::var_os("JOPE_RNK04_POSITION_REPORT_DIR")
            .expect("external RNK-04 parity report directory required"),
    );
    fs::create_dir_all(&report_dir).unwrap();
    fs::write(
        report_dir.join("RNK-04-B-position-family.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
}

fn decode_sparse_values(encoded: &str) -> [f64; 302] {
    let mut values = [0.0; 302];
    if encoded.is_empty() || encoded == "-" {
        return values;
    }
    for item in encoded.split(',') {
        let (index, value) = item.split_once(':').unwrap();
        values[index.parse::<usize>().unwrap()] = value.parse::<f64>().unwrap();
    }
    values
}
