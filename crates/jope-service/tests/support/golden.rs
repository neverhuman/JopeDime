//! Helpers for loading documented golden SafeTensors fixtures in integration tests.
//!
//! The default fixture directory is `research/flywheel_selfplay/golden` in the
//! repository. Set `JOPE_GOLDEN_DIR` to use an externally staged fixture set.

use candle_core_0_11_0::{safetensors::load_buffer, DType, Device, Tensor};
use safetensors::SafeTensors;
use std::{collections::BTreeMap, env, error::Error, fmt, fs, path::Path};

#[rustfmt::skip]
#[path = "../../src/ranker/stage1b.rs"]
mod stage1b;

const GOLDEN_DIR_ENV: &str = "JOPE_GOLDEN_DIR";
const STRICT_ENV: &str = "JOPE_PARITY_STRICT";

/// A required tensor name and its expected dimensions in a fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
        }
    }
}

/// A loaded fixture, retaining Candle tensors and an FP32 flattened view of each.
#[derive(Debug)]
pub struct GoldenFixture {
    pub tensors: BTreeMap<String, Tensor>,
    pub values_f32: BTreeMap<String, Vec<f32>>,
}

/// A fixture lookup, file, SafeTensors, or schema-validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenFixtureError(String);

impl fmt::Display for GoldenFixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GoldenFixtureError {}

fn error(message: impl Into<String>) -> GoldenFixtureError {
    GoldenFixtureError(message.into())
}

/// Load a fixture by relative name, using `JOPE_GOLDEN_DIR` or the repository's
/// `research/flywheel_selfplay/golden` directory. Missing fixtures are skipped
/// with a diagnostic unless `JOPE_PARITY_STRICT=1`.
pub fn load_named(
    name: &str,
    schema: &[TensorSpec],
) -> Result<Option<GoldenFixture>, GoldenFixtureError> {
    let root = env::var_os(GOLDEN_DIR_ENV).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../research/flywheel_selfplay/golden"),
        std::path::PathBuf::from,
    );
    let strict = env::var(STRICT_ENV).is_ok_and(|value| value == "1");
    load_from_dir(&root, name, schema, strict)
}

fn load_from_dir(
    root: &Path,
    name: &str,
    schema: &[TensorSpec],
    strict: bool,
) -> Result<Option<GoldenFixture>, GoldenFixtureError> {
    let relative = Path::new(name);
    if name.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(error(format!("invalid golden fixture name `{name}`")));
    }

    let relative = if relative
        .extension()
        .is_some_and(|extension| extension == "safetensors")
    {
        relative.to_path_buf()
    } else {
        relative.with_extension("safetensors")
    };
    let path = root.join(relative);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            let message = format!(
                "missing golden fixture `{}`; set {GOLDEN_DIR_ENV} or stage the documented fixture",
                path.display()
            );
            if strict {
                return Err(error(message));
            }
            eprintln!("skipping golden fixture: {message}");
            return Ok(None);
        }
        Err(source) => {
            return Err(error(format!(
                "could not read golden fixture `{}`: {source}",
                path.display()
            )));
        }
    };

    validate_schema(&bytes, schema, &path)?;
    let loaded = load_buffer(&bytes, &Device::Cpu).map_err(|source| {
        error(format!(
            "could not decode golden fixture `{}` into Candle tensors: {source}",
            path.display()
        ))
    })?;
    let mut tensors = BTreeMap::new();
    let mut values_f32 = BTreeMap::new();
    for (name, tensor) in loaded {
        let values = tensor
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|source| {
                error(format!(
                    "could not convert tensor `{name}` in `{}` to Vec<f32>: {source}",
                    path.display()
                ))
            })?;
        tensors.insert(name.clone(), tensor);
        values_f32.insert(name, values);
    }

    Ok(Some(GoldenFixture {
        tensors,
        values_f32,
    }))
}

fn validate_schema(
    bytes: &[u8],
    schema: &[TensorSpec],
    path: &Path,
) -> Result<(), GoldenFixtureError> {
    let safe_tensors = SafeTensors::deserialize(bytes).map_err(|source| {
        error(format!(
            "invalid SafeTensors fixture `{}`: {source}",
            path.display()
        ))
    })?;
    let expected = schema
        .iter()
        .map(|spec| (spec.name.as_str(), spec.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != schema.len() {
        return Err(error(format!(
            "fixture schema for `{}` contains duplicate tensor names",
            path.display()
        )));
    }

    let actual = safe_tensors
        .names()
        .into_iter()
        .map(|name| {
            safe_tensors
                .tensor(name)
                .map(|tensor| (name, tensor.shape().to_vec()))
                .map_err(|source| {
                    error(format!(
                        "invalid tensor `{name}` in `{}`: {source}",
                        path.display()
                    ))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if actual.len() != expected.len() {
        return Err(error(format!(
            "fixture `{}` has tensor names {:?}; expected {:?}",
            path.display(),
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        )));
    }
    for (name, expected_shape) in expected {
        let actual_shape = actual.get(name).ok_or_else(|| {
            error(format!(
                "fixture `{}` is missing required tensor `{name}`",
                path.display()
            ))
        })?;
        if actual_shape.as_slice() != expected_shape {
            return Err(error(format!(
                "tensor `{name}` in `{}` has shape {actual_shape:?}; expected {expected_shape:?}",
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_service::ranker::landmarks::{
        compare_sampled_to_full, summarize_latency, LandmarkOutput, BUILDING_BLOCK_WIDTH,
        INTERACTION_WIDTH, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH,
        REPORT_WIDTH, TABLE_WIDTH, VECTOR_WIDTH,
    };
    use safetensors::tensor::{serialize, Dtype, TensorView};
    use std::{process::Command, time::Duration};
    use tempfile::tempdir;

    const TEST_PROBE_ENV: &str = "JOPE_GOLDEN_TEST_PROBE";

    fn fixture_array<const N: usize>(fixture: &GoldenFixture, name: &str) -> [f32; N] {
        fixture.values_f32[name]
            .clone()
            .try_into()
            .unwrap_or_else(|_| panic!("fixture tensor {name} must contain {N} values"))
    }

    fn fixture_bytes() -> Vec<u8> {
        let values = [1.25_f32, -2.0, 3.5, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let extra_values = [0.5_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize(
            [
                (
                    "input",
                    TensorView::new(Dtype::F32, vec![2, 2], &values).unwrap(),
                ),
                (
                    "extra",
                    TensorView::new(Dtype::F32, vec![1], &extra_values).unwrap(),
                ),
            ],
            None,
        )
        .unwrap()
    }

    fn landmark_fixture_bytes() -> Vec<u8> {
        let input_x: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let input_y = [-0.5_f32, 0.5, -1.0, 1.0];
        let input_row_ids = [10_i64, 11, 20, 21];
        let report: Vec<f32> = (0..REPORT_WIDTH).map(|value| value as f32).collect();
        let table: Vec<f32> = (0..TABLE_WIDTH)
            .map(|value| 1_000.0 + value as f32)
            .collect();
        let interactions: Vec<f32> = (0..INTERACTION_WIDTH)
            .map(|value| 2_000.0 + value as f32)
            .collect();
        let building_blocks: Vec<f32> = (0..BUILDING_BLOCK_WIDTH)
            .map(|value| 3_000.0 + value as f32)
            .collect();
        let col_full: Vec<f32> = (0..2 * RAW_COLUMN_WIDTH)
            .map(|value| 4_000.0 + value as f32)
            .collect();
        let expected_vector: Vec<f32> = report
            .iter()
            .chain(&table)
            .chain(&interactions)
            .chain(&building_blocks)
            .copied()
            .collect();
        let mut expected_columns = vec![0.0; RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH];
        for row in 0..2 {
            for column in 0..RANKER_COLUMN_WIDTH {
                expected_columns[row * RANKER_COLUMN_WIDTH + column] =
                    col_full[row * RAW_COLUMN_WIDTH + column];
            }
        }
        let sampled_rows = [2.0_f32];
        let full_rows = [4.0_f32];
        let latency_ms = [1.0_f32, 3.0, 2.0];

        let as_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let input_x = as_bytes(&input_x);
        let input_y = as_bytes(&input_y);
        let input_row_ids = input_row_ids
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let report = as_bytes(&report);
        let table = as_bytes(&table);
        let interactions = as_bytes(&interactions);
        let building_blocks = as_bytes(&building_blocks);
        let col_full = as_bytes(&col_full);
        let v256 = as_bytes(&expected_vector);
        let col = as_bytes(&expected_columns);
        let sampled_rows = as_bytes(&sampled_rows);
        let full_rows = as_bytes(&full_rows);
        let latency_ms = as_bytes(&latency_ms);

        serialize(
            [
                (
                    "input.X",
                    TensorView::new(Dtype::F32, vec![4, 2], &input_x).unwrap(),
                ),
                (
                    "input.y",
                    TensorView::new(Dtype::F32, vec![4], &input_y).unwrap(),
                ),
                (
                    "input.row_ids",
                    TensorView::new(Dtype::I64, vec![4], &input_row_ids).unwrap(),
                ),
                (
                    "report",
                    TensorView::new(Dtype::F32, vec![REPORT_WIDTH], &report).unwrap(),
                ),
                (
                    "table",
                    TensorView::new(Dtype::F32, vec![TABLE_WIDTH], &table).unwrap(),
                ),
                (
                    "interactions",
                    TensorView::new(Dtype::F32, vec![INTERACTION_WIDTH], &interactions).unwrap(),
                ),
                (
                    "building_blocks",
                    TensorView::new(Dtype::F32, vec![BUILDING_BLOCK_WIDTH], &building_blocks)
                        .unwrap(),
                ),
                (
                    "col_full",
                    TensorView::new(Dtype::F32, vec![2, RAW_COLUMN_WIDTH], &col_full).unwrap(),
                ),
                (
                    "v256",
                    TensorView::new(Dtype::F32, vec![1, VECTOR_WIDTH], &v256).unwrap(),
                ),
                (
                    "col",
                    TensorView::new(
                        Dtype::F32,
                        vec![1, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH],
                        &col,
                    )
                    .unwrap(),
                ),
                (
                    "sampled_rows",
                    TensorView::new(Dtype::F32, vec![1], &sampled_rows).unwrap(),
                ),
                (
                    "full_rows",
                    TensorView::new(Dtype::F32, vec![1], &full_rows).unwrap(),
                ),
                (
                    "latency_ms",
                    TensorView::new(Dtype::F32, vec![3], &latency_ms).unwrap(),
                ),
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn synthetic_fixture_loads_candle_tensor_and_flat_values() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("synthetic.safetensors"),
            fixture_bytes(),
        )
        .unwrap();
        let schema = [
            TensorSpec::new("input", vec![2, 2]),
            TensorSpec::new("extra", vec![1]),
        ];

        let fixture = load_from_dir(directory.path(), "synthetic", &schema, true)
            .unwrap()
            .unwrap();

        assert_eq!(fixture.tensors["input"].dims(), &[2, 2]);
        assert_eq!(fixture.values_f32["input"], [1.25, -2.0, 3.5, 4.0]);
        assert_eq!(fixture.values_f32["extra"], [0.5]);
    }

    #[test]
    #[ignore = "pre-stage scaffold; unignore only after RES-07 publishes real full-high fixtures and recipe parity is wired"]
    fn synthetic_landmark_qualification_harness_uses_res09_loader() {
        let directory = tempdir().unwrap();
        fs::create_dir_all(directory.path().join("rnk05")).unwrap();
        fs::write(
            directory
                .path()
                .join("rnk05/synthetic-full-high.safetensors"),
            landmark_fixture_bytes(),
        )
        .unwrap();

        let schema = [
            TensorSpec::new("input.X", vec![4, 2]),
            TensorSpec::new("input.y", vec![4]),
            TensorSpec::new("input.row_ids", vec![4]),
            TensorSpec::new("report", vec![REPORT_WIDTH]),
            TensorSpec::new("table", vec![TABLE_WIDTH]),
            TensorSpec::new("interactions", vec![INTERACTION_WIDTH]),
            TensorSpec::new("building_blocks", vec![BUILDING_BLOCK_WIDTH]),
            TensorSpec::new("col_full", vec![2, RAW_COLUMN_WIDTH]),
            TensorSpec::new("v256", vec![1, VECTOR_WIDTH]),
            TensorSpec::new("col", vec![1, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH]),
            TensorSpec::new("sampled_rows", vec![1]),
            TensorSpec::new("full_rows", vec![1]),
            TensorSpec::new("latency_ms", vec![3]),
        ];
        let fixture = load_from_dir(directory.path(), "rnk05/synthetic-full-high", &schema, true)
            .unwrap()
            .unwrap();
        let values = &fixture.values_f32;
        assert_eq!(fixture.tensors["input.X"].dims(), &[4, 2]);
        assert_eq!(fixture.tensors["input.y"].dims(), &[4]);
        assert_eq!(fixture.tensors["input.row_ids"].dims(), &[4]);
        assert_eq!(fixture.tensors["input.row_ids"].dtype(), DType::I64);
        assert_eq!(values["input.row_ids"], [10.0, 11.0, 20.0, 21.0]);
        let report = fixture_array::<REPORT_WIDTH>(&fixture, "report");
        let table = fixture_array::<TABLE_WIDTH>(&fixture, "table");
        let interactions = fixture_array::<INTERACTION_WIDTH>(&fixture, "interactions");
        let building_blocks = fixture_array::<BUILDING_BLOCK_WIDTH>(&fixture, "building_blocks");
        let raw_column_values = &values["col_full"];
        let raw_columns: Vec<[f32; RAW_COLUMN_WIDTH]> = raw_column_values
            .chunks_exact(RAW_COLUMN_WIDTH)
            .map(|row| row.try_into().unwrap())
            .collect();

        let full = LandmarkOutput::from_descriptor_parts(
            &report,
            &table,
            &interactions,
            &building_blocks,
            &raw_columns,
        );
        assert_eq!(
            full.vector(),
            &fixture_array::<VECTOR_WIDTH>(&fixture, "v256")
        );
        assert_eq!(
            full.columns().iter().flatten().copied().collect::<Vec<_>>(),
            values["col"]
        );

        let mut sampled_report = report;
        sampled_report[0] += 1.0;
        let sampled = LandmarkOutput::from_descriptor_parts(
            &sampled_report,
            &table,
            &interactions,
            &building_blocks,
            &raw_columns,
        );
        let sampled_rows = values["sampled_rows"][0] as usize;
        let full_rows = values["full_rows"][0] as usize;
        let agreement = compare_sampled_to_full(&sampled, &full, sampled_rows, full_rows).unwrap();
        let compared = VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH;
        assert_eq!(agreement.sampled_rows, 2);
        assert_eq!(agreement.full_rows, 4);
        assert_eq!(agreement.values_compared, compared);
        assert_eq!(agreement.exact_matches, compared - 1);
        assert_eq!(agreement.max_abs_error, 1.0);
        assert!((agreement.mean_abs_error - 1.0 / compared as f64).abs() < f64::EPSILON);

        let latency = values["latency_ms"]
            .iter()
            .map(|milliseconds| Duration::from_secs_f64(f64::from(*milliseconds) / 1_000.0))
            .collect::<Vec<_>>();
        let latency = summarize_latency(&latency).unwrap();
        assert_eq!(latency.samples, 3);
        assert_eq!(latency.min_ms, 1.0);
        assert!((latency.median_ms - 2.0).abs() < 1e-9);
        assert_eq!(latency.max_ms, 3.0);
    }

    #[test]
    fn schema_rejects_missing_extra_and_wrong_shape_tensors() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("synthetic.safetensors"),
            fixture_bytes(),
        )
        .unwrap();

        let wrong_shape = [
            TensorSpec::new("input", vec![4]),
            TensorSpec::new("extra", vec![1]),
        ];
        assert!(
            load_from_dir(directory.path(), "synthetic", &wrong_shape, true)
                .unwrap_err()
                .to_string()
                .contains("expected [4]")
        );

        let extra = [TensorSpec::new("input", vec![2, 2])];
        assert!(load_from_dir(directory.path(), "synthetic", &extra, true)
            .unwrap_err()
            .to_string()
            .contains("tensor names"));

        let missing = [
            TensorSpec::new("input", vec![2, 2]),
            TensorSpec::new("expected", vec![2, 2]),
        ];
        assert!(load_from_dir(directory.path(), "synthetic", &missing, true)
            .unwrap_err()
            .to_string()
            .contains("missing required tensor `expected`"));
    }

    #[test]
    fn missing_fixture_skips_unless_parity_is_strict() {
        let directory = tempdir().unwrap();
        let schema = [TensorSpec::new("input", vec![2, 2])];

        assert!(load_from_dir(directory.path(), "absent", &schema, false)
            .unwrap()
            .is_none());
        assert!(load_from_dir(directory.path(), "absent", &schema, true)
            .unwrap_err()
            .to_string()
            .contains("missing golden fixture"));
    }

    #[test]
    fn missing_fixture_obeys_strict_environment_and_prints_skip_diagnostic() {
        if let Some(mode) = env::var_os(TEST_PROBE_ENV) {
            let schema = [TensorSpec::new("input", vec![2, 2])];
            match mode.to_string_lossy().as_ref() {
                "strict" => assert!(load_named("absent", &schema)
                    .unwrap_err()
                    .to_string()
                    .contains("missing golden fixture")),
                "skip" => assert!(load_named("absent", &schema).unwrap().is_none()),
                other => panic!("unknown test probe mode: {other}"),
            }
            return;
        }

        let directory = tempdir().unwrap();
        for (mode, strict) in [("strict", "1"), ("skip", "0")] {
            let output = Command::new(env::current_exe().unwrap())
                .arg("--exact")
                .arg("tests::missing_fixture_obeys_strict_environment_and_prints_skip_diagnostic")
                .arg("--nocapture")
                .env(TEST_PROBE_ENV, mode)
                .env(GOLDEN_DIR_ENV, directory.path())
                .env(STRICT_ENV, strict)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{mode} environment probe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            if mode == "skip" {
                let output = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(output.contains("skipping golden fixture: missing golden fixture"));
            }
        }
    }

    #[test]
    #[ignore = "unignore only after RES-07 publishes its generated index and every fixture payload"]
    fn res07_fixture_inventory_loads_every_safetensors_shape() {
        let root = env::var_os(GOLDEN_DIR_ENV).map_or_else(
            || {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../research/flywheel_selfplay/golden")
            },
            std::path::PathBuf::from,
        );
        let index_path = root.join("index.json");
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(&index_path).unwrap_or_else(|error| {
                panic!(
                    "RES-07 fixture inventory is unavailable at {}: {error}",
                    index_path.display()
                )
            }))
            .expect("RES-07 index.json must be valid JSON");
        assert_eq!(index["schema"], "jope.golden-fixtures/v1");

        let files = index["files"]
            .as_object()
            .expect("RES-07 index.json must contain a files object");
        let fixtures = files
            .iter()
            .filter(|(name, _)| name.ends_with(".safetensors"))
            .collect::<Vec<_>>();
        assert!(
            !fixtures.is_empty(),
            "RES-07 index lists no SafeTensors fixtures"
        );

        for (name, metadata) in fixtures {
            let tensor_metadata = metadata["tensors"]
                .as_object()
                .unwrap_or_else(|| panic!("RES-07 inventory entry `{name}` has no tensor schema"));
            let schema = tensor_metadata
                .iter()
                .map(|(tensor, spec)| {
                    let shape = spec["shape"]
                        .as_array()
                        .unwrap_or_else(|| {
                            panic!("RES-07 tensor `{tensor}` in `{name}` has no shape")
                        })
                        .iter()
                        .map(|dimension| {
                            usize::try_from(
                                dimension
                                    .as_u64()
                                    .expect("tensor dimensions must be unsigned"),
                            )
                            .expect("tensor dimension must fit usize")
                        })
                        .collect::<Vec<_>>();
                    TensorSpec::new(tensor, shape)
                })
                .collect::<Vec<_>>();
            let fixture_name = name
                .strip_suffix(".safetensors")
                .expect("filtered SafeTensors name must have its extension");
            let fixture = load_named(fixture_name, &schema)
                .unwrap_or_else(|error| {
                    panic!("RES-07 fixture `{name}` failed validation: {error}")
                })
                .unwrap_or_else(|| panic!("RES-07 fixture `{name}` was skipped"));

            for (tensor, spec) in schema.iter().map(|spec| (&spec.name, &spec.shape)) {
                assert_eq!(
                    fixture.tensors[tensor].dims(),
                    spec,
                    "shape mismatch in `{name}` tensor `{tensor}`"
                );
            }
        }
    }

    fn rnk07_output_schema() -> [TensorSpec; 2] {
        [
            TensorSpec::new("tokens", vec![1, 10, 32]),
            TensorSpec::new("mask", vec![1, 10]),
        ]
    }

    #[test]
    fn rnk07_loader_seam_rejects_invalid_fixture_paths() {
        let error = load_named("../outside", &rnk07_output_schema()).unwrap_err();
        assert!(error.to_string().contains("invalid golden fixture name"));
    }

    #[test]
    #[ignore = "RES-07 fixtures not qualified; enable only after qualified tensors are published"]
    fn rnk07_fixture_schema_smoke_after_res07_qualification() {
        let context = env::var("JOPE_RNK07_GOLDEN_CONTEXT")
            .expect("select a qualified RES-07 training context");
        let fixture_name = format!("contexts/{context}");
        let fixture = load_named(&fixture_name, &rnk07_output_schema())
            .expect("qualified RES-07 fixture must satisfy the documented output schema")
            .expect("fixture-dependent check remains ignored until RES-07 qualification");

        assert_eq!(fixture.tensors["tokens"].dims(), [1, 10, 32]);
        assert_eq!(fixture.tensors["mask"].dims(), [1, 10]);
    }

    #[test]
    fn rnk07_stage1b_module_is_available_for_synthetic_tests() {
        assert_eq!(stage1b::TOKEN_COUNT, 10);
        assert_eq!(stage1b::TOKEN_WIDTH, 32);
    }
}
