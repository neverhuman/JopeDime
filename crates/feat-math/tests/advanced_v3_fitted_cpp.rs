use feat_math::{
    emit_fitted_cpp17, fit, fit_legacy_numeric_bound, fit_table, supports_fitted_cpp17, BoundModel,
    ColumnSpec, DataSchema, EncodedTable, GenomeIr, Model, Objective, RawTable, RowSemantics,
    TaskSpec,
};
use std::path::Path;
use std::process::Command;

fn numeric_schema() -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec::numeric("a"), ColumnSpec::numeric("b")],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::from_csv(
        "a,b\n0,-2\n0.25,-1\n0.5,0\n0.75,1\n1,2\n1.25,3\n1.5,4\n1.75,5\n2,6\n2.25,7\n",
    )
    .unwrap()
}

fn task() -> TaskSpec {
    TaskSpec::regression(Objective::Rmse).unwrap()
}

#[test]
fn audit_expression_failures_match_rust_under_sanitizers() {
    let train: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64 * 0.25, i as f64 - 2.0])
        .collect();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    for (index, source) in [
        "(sym raw \"c9+1\")",
        "(sym raw \"min(clip(1,(-1)^0.5,(-1)^0.5),7)+1\")",
        "(pipe X (map expr=\"c9+1\") (ridge))",
        "(pipe X (derive expr=\"c9+1\") (ridge))",
        "(pipe X (map expr=\"min(clip(1,(-1)^0.5,(-1)^0.5),7)+1\") (ridge))",
    ]
    .iter()
    .enumerate()
    {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_legacy_numeric_bound(&genome, &train, &y).unwrap();
        let expected: Vec<_> = fit(&genome, &train, &y)
            .unwrap()
            .predict(&vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]])
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let scratch = tempfile::Builder::new()
            .prefix(&format!("kernel-expression-safety-{index}-"))
            .tempdir()
            .unwrap();
        let directory = scratch.path();
        assert_eq!(
            compile_and_run(&artifact.source, directory, true),
            expected,
            "{source}"
        );
        scratch.close().unwrap();
    }
}

fn compile_and_run(source: &str, directory: &Path, sanitizer: bool) -> Vec<u64> {
    compile_typed(
        source,
        directory,
        sanitizer,
        &[0.125, -1.5, 0.0, 2.5, 2.125, 6.5],
        &[0, 0, 1, 0, 0, 0],
        3,
        2,
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_typed(
    source: &str,
    directory: &Path,
    sanitizer: bool,
    values: &[f64],
    missing: &[u8],
    rows: usize,
    columns: usize,
) -> Vec<u64> {
    let mode = std::env::var("JOPE_FITTED_CPP_SANITIZER").unwrap_or_else(|_| {
        if sanitizer {
            "address,undefined".to_string()
        } else {
            "normal".to_string()
        }
    });
    let suffix = mode.replace(',', "-");
    let cpp = directory.join(format!("model-{suffix}.cpp"));
    let binary = directory.join(format!("model-{suffix}"));
    let encoded_values = values
        .iter()
        .map(|value| format!("f64_from_bits(0x{:016x}ULL)", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let encoded_missing = missing
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let driver = format!(
        r#"
#include <cstdio>
int main(){{
  const double values[]={{ {encoded_values} }};
  const std::uint8_t missing[]={{ {encoded_missing} }};
  double output[{rows}]={{}};
  int status=jope_fitted_predict(values,{cells},missing,{cells},{rows},{columns},output,{rows});
  if(status!=0)return status;
  for(double value:output){{std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}}
  return 0;
}}
"#,
        cells = values.len()
    );
    std::fs::write(&cpp, format!("{source}\n{driver}")).unwrap();
    let mut command = feat_math::cxx::command(None);
    command.args([
        "-O2",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]);
    if mode != "normal" {
        command.arg("-O1").arg(format!("-fsanitize={mode}"));
        if mode.contains("undefined") {
            command.arg("-fno-sanitize-recover=undefined");
        }
    }
    let compiled = command.arg(&cpp).arg("-o").arg(&binary).output().unwrap();
    assert!(
        compiled.status.success(),
        "g++ failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    assert!(
        run.status.success(),
        "generated C++ failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect()
}

#[test]
fn fitted_numeric_pipeline_is_deterministic_strict_and_bit_exact() {
    assert!(
        feat_math::cxx::version().is_ok(),
        "a C++ compiler (CXX, default g++) is required"
    );
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=hybrid) (standardize) (ridge lambda_frac=0.02))",
    )
    .unwrap();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let model = fit_table(&genome, &task(), &numeric_schema(), &training_table(), &y).unwrap();
    supports_fitted_cpp17(&model).unwrap();
    let first = emit_fitted_cpp17(&model).unwrap();
    let second = emit_fitted_cpp17(&model).unwrap();
    assert_eq!(first.source, second.source);
    assert_eq!(first.source_sha256, second.source_sha256);
    assert_eq!(first.codec, second.codec);
    assert_eq!(first.codec_sha256, second.codec_sha256);
    assert!(first.codec.contains("jope-dime.fitted-cpp-typed-table.v1"));
    for banned in [
        "fstream", "ifstream", "ofstream", "fopen", "getenv", "system(", "socket(", "chrono",
    ] {
        assert!(
            !first.source.contains(banned),
            "generated source contains {banned}"
        );
    }

    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(compile_and_run(&first.source, directory, false), expected);
    assert_eq!(compile_and_run(&first.source, directory, true), expected);
    scratch.close().unwrap();
}

#[test]
fn ordered_numeric_row_semantics_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let ordered_schema = DataSchema {
        columns: vec![ColumnSpec::numeric("a"), ColumnSpec::numeric("b")],
        row_semantics: RowSemantics::Ordered {
            group_by: Vec::new(),
            order_by: vec!["a".to_string()],
        },
    };
    let plain = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let ordered = fit_table(&plain, &task(), &ordered_schema, &training_table(), &y).unwrap();
    supports_fitted_cpp17(&ordered).unwrap();
    let artifact = emit_fitted_cpp17(&ordered).unwrap();
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let expected = ordered
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-ordered-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_and_run(&artifact.source, directory, false),
        expected
    );
    scratch.close().unwrap();
}

#[test]
fn iid_live_transform_and_regression_head_corpus_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (input numeric=median missing=indicator) (robust_scale) (winsor_clip q=0.1) (sigmoid) (step t=0.4) (norm_row) (select_corr k=4) (impute strategy=median) (target_bin_encode bins=4 prior=5) (ridge))",
        "(pipe X (input numeric=median) (maxabs_scale) (ridge))",
        "(pipe X (input numeric=median) (power_transform kind=yeo_johnson) (ridge))",
        "(pipe X (input numeric=median) (piecewise_linear bins=8) (ridge))",
        "(pipe X (input numeric=median) (corr_screen k=32) (ridge))",
        "(pipe X (input numeric=median) (pls_project k=2) (ridge))",
        "(pipe X (target kind=log1p) (input numeric=median) (piecewise_linear append=true bins=8) (ridge))",
        "(pipe X (input numeric=median) (tab_boost rounds=8 bins=8 min_leaf=1))",
        "(pipe X (input numeric=median) (extra_trees trees=8 depth=3 min_leaf=1))",
        "(pipe X (input numeric=median) (gam_boost rounds=8 bins=4))",
        "(pipe X (input numeric=median) (pls k=2))",
        "(pipe X (target kind=rank_gauss) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=log1p) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=winsor q=0.05) (input numeric=median) (standardize) (ridge))",
        "(pipe X (target kind=zscore) (input numeric=median) (standardize) (ridge))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-corpus-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task(), &numeric_schema(), &training_table(), &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_and_run(&artifact.source, directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn legacy_row_and_expression_transform_corpus_matches_bits() {
    let train = training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = [vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let cases = [
        "(pipe X (row_stats) (ridge))",
        "(pipe X (fft_mag k=1) (ridge))",
        "(pipe X (row_spectral) (ridge))",
        "(pipe X (row_diff_stats) (ridge))",
        "(pipe X (map expr=\"sigmoid(x)+row_mean\") (ridge))",
        "(pipe X (derive expr=\"row_l2+col(0)\") (ridge))",
        "(sym affine \"sigmoid(col(0)) + 0.5*col(1)\")",
        "(portfolio (member (pipe X (ridge))) (blend folds=5 guard=clip kind=nnls repeats=1))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-row-corpus-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_legacy_numeric_bound(&genome, &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(
                &RawTable::new(
                    vec!["c0".to_string(), "c1".to_string()],
                    test.iter()
                        .map(|row| row.iter().map(f64::to_string).collect())
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &values,
                &missing,
                test.len(),
                2,
            ),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn complete_legacy_transform_registry_matches_bits() {
    let train = (0..40)
        .map(|index| {
            let x = index as f64 / 11.0 - 1.5;
            vec![
                x,
                (0.7 * x).sin(),
                (1.3 * x).cos(),
                (index % 5) as f64 - 2.0,
            ]
        })
        .collect::<Vec<_>>();
    let y = train
        .iter()
        .map(|row| 0.3 * row[0] - 0.4 * row[1] + row[2] * row[3] * 0.1)
        .collect::<Vec<_>>();
    let test = [
        vec![-1.7, -0.9, 0.4, -2.0],
        vec![0.2, 0.15, 0.95, 0.0],
        vec![2.3, 1.1, -0.8, 2.0],
    ];
    let cases = [
        "(pipe X (standardize append=true) (ridge))",
        "(pipe X (robust_scale append=true) (ridge))",
        "(pipe X (rank append=true) (ridge))",
        "(pipe X (blom append=true) (ridge))",
        "(pipe X (quantile_bins bins=5 append=true) (ridge))",
        "(pipe X (bin_onehot bins=4 append=true) (ridge))",
        "(pipe X (hinge knots=3 append=true) (ridge))",
        "(pipe X (winsor_clip q=0.1 append=true) (ridge))",
        "(pipe X (cheb degree=4 append=true) (ridge))",
        "(pipe X (poly_interact append=true) (ridge))",
        "(pipe X (rff_cos n=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (rff_sin n=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (rand_proj k=5 seed=11 append=true) (ridge))",
        "(pipe X (landmark_dist k=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (oblique_hinge n=5 seed=11 append=true) (ridge))",
        "(pipe X (pair_prod topk=4 append=true) (ridge))",
        "(pipe X (target_bin_encode bins=5 prior=3 append=true) (ridge))",
        "(pipe X (sigmoid append=true) (ridge))",
        "(pipe X (step t=0.2 append=true) (ridge))",
        "(pipe X (norm_row append=true) (ridge))",
        "(pipe X (polar append=true) (ridge))",
        "(pipe X (lag k=2 append=true) (ridge))",
        "(pipe X (roll k=3 append=true) (ridge))",
        "(pipe X (fft_mag k=2 append=true) (ridge))",
        "(pipe X (row_spectral append=true) (ridge))",
        "(pipe X (row_diff_stats append=true) (ridge))",
        "(pipe X (svd_project k=3 iters=8 seed=11 append=true) (ridge))",
        "(pipe X (nystrom k=5 gamma=0.7 seed=11 append=true) (ridge))",
        "(pipe X (symlog c=0.7 append=true) (ridge))",
        "(pipe X (select_corr k=3 append=true) (ridge))",
        "(pipe X (impute strategy=median append=true) (ridge))",
        "(pipe X (pair_ratio topk=5 append=true) (ridge))",
        "(pipe X (spline knots=4 append=true) (ridge))",
        "(pipe X (map expr=\"asinh(x)+row_mean\" append=true) (ridge))",
        "(pipe X (derive expr=\"row_l2+col(0)\" append=true) (ridge))",
    ];
    let raw = RawTable::new(
        (0..4).map(|column| format!("c{column}")).collect(),
        test.iter()
            .map(|row| row.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap();
    let train_schema = DataSchema {
        columns: (0..4)
            .map(|column| ColumnSpec::numeric(format!("c{column}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_raw = RawTable::new(
        (0..4).map(|column| format!("c{column}")).collect(),
        train
            .iter()
            .map(|row| row.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap();
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-all-legacy-ops-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let legacy = fit_legacy_numeric_bound(&genome, &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let model = if source.contains("(lag ") || source.contains("(roll ") {
            legacy
        } else {
            let v3 = fit_table(&genome, &task(), &train_schema, &train_raw, &y)
                .unwrap_or_else(|error| panic!("case {index} V3 fit: {error}"));
            assert_eq!(
                v3.predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                legacy
                    .predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                "case {index} V3 bridge: {source}"
            );
            v3
        };
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&raw)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &values,
                &missing,
                test.len(),
                4,
            ),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn iid_live_binary_head_and_calibration_corpus_matches_bits() {
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (input numeric=median missing=indicator) (logit_ridge class_weight=balanced l2=0.1 calibration=platt threshold=f1))",
        "(pipe X (input numeric=median missing=indicator) (tab_boost rounds=8 bins=8 min_leaf=1))",
        "(pipe X (input numeric=median missing=indicator) (extra_trees trees=8 depth=3 min_leaf=1))",
        "(pipe X (input numeric=median missing=indicator) (gam_boost rounds=8 bins=4 calibration=isotonic threshold=youden))",
        "(pipe X (input numeric=median missing=indicator) (portfolio))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-binary-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &numeric_schema(), &training_table(), &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_and_run(&artifact.source, directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn iid_categorical_codec_modes_match_bits() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let train = RawTable::from_csv(
        "amount,kind\n0,A\n0.25,A\n0.5,B\n0.75,B\n1,C\n1.25,C\n1.5,A\n1.75,B\n2,C\n2.25,A\n",
    )
    .unwrap();
    let test = RawTable::from_csv("amount,kind\n0.125,A\n,B\n2.125,unknown\n").unwrap();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let values = [0.125, 0.0, 0.0, 1.0, 2.125, -1.0];
    let missing = [0, 0, 1, 0, 0, 0];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-categorical-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for mode in ["native", "frequency", "onehot", "ordered", "hybrid"] {
        let genome = GenomeIr::parse(&format!(
            "(pipe X (input numeric=median missing=indicator categorical={mode} max_onehot=16 min_count=1 prior=10 folds=5) (ridge))"
        ))
        .unwrap();
        let model = fit_table(&genome, &task(), &schema, &train, &y).unwrap();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        assert!(artifact
            .codec
            .contains("\"vocabulary\":[\"A\",\"B\",\"C\"]"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
            expected,
            "categorical mode {mode}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn ordered_historical_categorical_state_matches_bits() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    };
    let train = RawTable::from_csv(
        "group,time,kind\n1,1,A\n1,2,A\n1,3,B\n1,4,A\n1,5,B\n2,1,A\n2,2,B\n2,3,B\n2,4,A\n2,5,B\n",
    )
    .unwrap();
    let test = RawTable::from_csv("group,time,kind\n1,6,A\n2,6,B\n3,1,A\n1,7,unknown\n").unwrap();
    let y = [0.1, 0.2, 0.8, 0.4, 0.9, 0.3, 0.7, 0.6, 0.5, 1.0];
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=ordered prior=2) (ridge))",
    )
    .unwrap();
    let model = fit_table(&genome, &task(), &schema, &train, &y).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let values = [1.0, 6.0, 0.0, 2.0, 6.0, 1.0, 3.0, 1.0, 0.0, 1.0, 7.0, -1.0];
    let missing = [0; 12];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-historical-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_typed(&artifact.source, directory, false, &values, &missing, 4, 3,),
        expected
    );
    scratch.close().unwrap();
}

#[test]
fn table_aware_lag_and_roll_match_bits_with_fitted_tails() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    };
    let train = RawTable::from_csv(
        "group,time,value\nA,1,1\nA,2,2\nA,3,3\nA,4,4\nA,5,5\nB,1,10\nB,2,11\nB,3,12\nB,4,13\nB,5,14\n",
    )
    .unwrap();
    let test = RawTable::from_csv("group,time,value\nA,7,7\nA,6,6\nB,6,15\nC,1,20\n").unwrap();
    let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let cases = [
        "(pipe X (input numeric=median categorical=native) (lag k=2 append=true) (ridge))",
        "(pipe X (input numeric=median categorical=native) (roll k=3 append=true) (ridge))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-table-ops-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for source in cases {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &task(),
            &schema,
            &train,
            &y,
        )
        .unwrap();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &encoded.values,
                &encoded.missing,
                encoded.rows,
                encoded.columns,
            ),
            expected,
            "{source}"
        );
    }
    scratch.close().unwrap();
}

fn bound_with_serialized_legacy_head(source: &str, y: &[f64]) -> BoundModel {
    let rows = training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let legacy = fit(&GenomeIr::parse(source).unwrap(), &rows, y).unwrap();
    let base = fit_table(
        &GenomeIr::parse("(pipe X (ridge))").unwrap(),
        &task(),
        &numeric_schema(),
        &training_table(),
        y,
    )
    .unwrap();
    let legacy_json = serde_json::to_value(legacy).unwrap();
    let head = legacy_json["Pipeline"]["head"].clone();
    assert!(
        !head.is_null(),
        "legacy pipeline head serialization changed"
    );
    let mut bound_json = serde_json::to_value(base).unwrap();
    bound_json["predictor"]["Pipeline"]["head"] = serde_json::json!({"Legacy": head});
    serde_json::from_value(bound_json).unwrap()
}

#[test]
fn admitted_serialized_legacy_head_corpus_matches_bits() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (ridge_gcv))",
        "(pipe X (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=5))",
        "(pipe X (histgbm rounds=3 depth=2 bins=8 min_leaf=1))",
        "(pipe X (ensemble rounds=3 depth=2 bins=8 min_leaf=1))",
        "(pipe X (histgbm2 rounds=8 leaves=4 min_leaf=1 bags=2))",
        "(pipe X (krr m=50))",
        "(pipe X (gpr m=50 steps=1))",
        "(pipe X (gpr kernel=rbf+linear m=50 steps=1))",
        "(pipe X (ksmooth k=3 gamma=1))",
        "(pipe X (hurdle2))",
        "(pipe X (stack))",
        "(pipe X (portfolio folds=5 repeats=1))",
        "(pipe X (portfolio preset=none add1=group_mean folds=5 repeats=1))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-legacy-heads-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let model = bound_with_serialized_legacy_head(source, &y);
        model.validate_binding().unwrap();
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_and_run(&artifact.source, directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn nontrivial_fitted_legacy_head_states_match_bits() {
    let train = (0..96)
        .map(|index| {
            let x = index as f64 / 17.0 - 2.0;
            vec![x, x.sin(), (x * 0.7).cos(), (index % 7) as f64 / 3.0]
        })
        .collect::<Vec<_>>();
    let y = train
        .iter()
        .enumerate()
        .map(|(index, row)| {
            if index % 9 == 0 {
                0.0
            } else {
                0.4 * row[0] - 0.7 * row[1] + 0.2 * row[2] * row[3]
            }
        })
        .collect::<Vec<_>>();
    let test = [
        vec![-1.7, (-1.7_f64).sin(), (-1.19_f64).cos(), 0.0],
        vec![-0.2, (-0.2_f64).sin(), (-0.14_f64).cos(), 1.0],
        vec![1.3, 1.3_f64.sin(), 0.91_f64.cos(), 2.0],
    ];
    let cases = [
        "(pipe X (histgbm2 rounds=12 leaves=6 min_leaf=2 bags=2))",
        "(pipe X (gpr kernel=rbf+linear m=24 steps=2))",
        "(pipe X (gpr kernel=periodic m=24 steps=2))",
        "(pipe X (hurdle2))",
        "(pipe X (stack))",
        "(pipe X (portfolio folds=5 repeats=1))",
    ];
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-nontrivial-heads-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let model = fit_legacy_numeric_bound(&GenomeIr::parse(source).unwrap(), &train, &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let raw = RawTable::new(
            (0..4).map(|column| format!("c{column}")).collect(),
            test.iter()
                .map(|row| row.iter().map(f64::to_string).collect())
                .collect(),
        )
        .unwrap();
        let expected = model
            .predict_table(&raw)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        if source.contains("(portfolio ") {
            let legacy = fit(&GenomeIr::parse(source).unwrap(), &train, &y).unwrap();
            let Model::Pipeline { head, .. } = legacy else {
                unreachable!()
            };
            let feat_math::heads::FittedHead::Portfolio(portfolio) = *head else {
                unreachable!()
            };
            for member in portfolio.members {
                let feat_math::portfolio::FittedMemberState::Fitted(member_model) = member.state
                else {
                    continue;
                };
                let mut member_json = serde_json::to_value(&model).unwrap();
                member_json["predictor"] = serde_json::json!({"Legacy": *member_model});
                let member_bound: BoundModel = serde_json::from_value(member_json).unwrap();
                let member_artifact = emit_fitted_cpp17(&member_bound).unwrap();
                let member_expected = member_bound
                    .predict_table(&raw)
                    .unwrap()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>();
                assert_eq!(
                    compile_typed(
                        &member_artifact.source,
                        directory,
                        false,
                        &values,
                        &missing,
                        test.len(),
                        4,
                    ),
                    member_expected,
                    "portfolio member {}",
                    member.name
                );
            }
        }
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &values,
                &missing,
                test.len(),
                4,
            ),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn frozen_legacy_pipeline_can_cross_the_bound_cpp_bridge() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let train = training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();
    let test_rows = vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let genome = GenomeIr::parse(
        "(pipe X (target kind=log1p) (quantile_bins append=true bins=8) (blom append=true) (pair_prod append=true topk=4) (elasticnet alpha=0.01 l1_ratio=0.1 sweeps=12))",
    )
    .unwrap();
    let model = fit_legacy_numeric_bound(&genome, &train, &y).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let expected = fit(&genome, &train, &y)
        .unwrap()
        .predict(&test_rows)
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let values = test_rows.into_iter().flatten().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-legacy-bridge-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
        expected
    );
    scratch.close().unwrap();
}

// ---------------------------------------------------------------------------------------------
// Emitter revision 3 (kernel2 X3): fitted index checks, empty rows and tied intervals
// ---------------------------------------------------------------------------------------------

/// The status `jope_fitted_predict` returns when a stage's fitted state does not fit the row it
/// reads (emitter revision 3).
const FITTED_WIDTH_STATUS: i32 = 8;

/// Compiles `program`, an emitted source followed by a driver, and runs it; returns its stdout.
/// The process must exit 0 and write nothing to stderr, so a sanitizer report, an abort or a
/// signal fails the test. `_GLIBCXX_ASSERTIONS` makes an unchecked `operator[]` past a vector's
/// end abort even inside the vector's allocation, where AddressSanitizer sees nothing. With
/// `sanitizer` the mode defaults to `address,undefined`; `JOPE_FITTED_CPP_SANITIZER` overrides
/// it, as for `compile_typed`.
fn build_and_run(program: &str, directory: &Path, name: &str, sanitizer: bool) -> String {
    let mode = std::env::var("JOPE_FITTED_CPP_SANITIZER").unwrap_or_else(|_| {
        if sanitizer {
            "address,undefined".to_string()
        } else {
            "normal".to_string()
        }
    });
    let suffix = mode.replace(',', "-");
    let cpp = directory.join(format!("{name}-{suffix}.cpp"));
    let binary = directory.join(format!("{name}-{suffix}"));
    std::fs::write(&cpp, program).unwrap();
    let mut command = feat_math::cxx::command(None);
    command.args([
        "-O2",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-D_GLIBCXX_ASSERTIONS",
    ]);
    if mode != "normal" {
        command.arg("-O1").arg(format!("-fsanitize={mode}"));
        if mode.contains("undefined") {
            command.arg("-fno-sanitize-recover=undefined");
        }
    }
    let compiled = command.arg(&cpp).arg("-o").arg(&binary).output().unwrap();
    assert!(
        compiled.status.success(),
        "{name}: g++ failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success() && stderr.is_empty(),
        "{name}: the generated C++ did not exit cleanly ({}):\n{stderr}",
        run.status
    );
    String::from_utf8(run.stdout).unwrap()
}

/// Compiles `source` with a driver that prints `status=N` and, when the status is 0, the bits of
/// each prediction, as `build_and_run` builds and runs it. Unlike `compile_typed`, a nonzero
/// status is a result here, not a failure.
#[allow(clippy::too_many_arguments)]
fn compile_status(
    source: &str,
    directory: &Path,
    name: &str,
    sanitizer: bool,
    values: &[f64],
    missing: &[u8],
    rows: usize,
    columns: usize,
) -> (i32, Vec<u64>) {
    let encoded_values = values
        .iter()
        .map(|value| format!("f64_from_bits(0x{:016x}ULL)", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let encoded_missing = missing
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let driver = format!(
        r#"
#include <cstdio>
int main(){{
  const double values[]={{ {encoded_values} }};
  const std::uint8_t missing[]={{ {encoded_missing} }};
  double output[{rows}]={{}};
  const int status=jope_fitted_predict(values,{cells},missing,{cells},{rows},{columns},output,{rows});
  std::printf("status=%d\n",status);
  if(status==0)for(double value:output){{std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}}
  return 0;
}}
"#,
        cells = values.len()
    );
    let stdout = build_and_run(&format!("{source}\n{driver}"), directory, name, sanitizer);
    let mut lines = stdout.lines();
    let status = lines
        .next()
        .and_then(|line| line.strip_prefix("status="))
        .unwrap_or_else(|| panic!("{name}: no status line in {stdout:?}"))
        .parse()
        .unwrap();
    let bits = lines
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    (status, bits)
}

/// A copy of `model` whose serialized form `edit` changed: fitted state that no fit produces.
/// The binding check does not hash the stages, so the copy loads.
fn tampered(model: &BoundModel, edit: impl FnOnce(&mut serde_json::Value)) -> BoundModel {
    let mut json = serde_json::to_value(model).unwrap();
    edit(&mut json);
    let edited: BoundModel = serde_json::from_value(json).unwrap();
    edited.validate_binding().unwrap();
    edited
}

/// Stage `index` of a schema-bound pipeline, in its serialized form.
fn stage(json: &mut serde_json::Value, index: usize) -> &mut serde_json::Value {
    let stage = &mut json["predictor"]["Pipeline"]["ops"][index];
    assert!(stage.is_object(), "stage {index} is fitted state: {stage}");
    stage
}

/// The training table's cells as numbers, for the legacy predictors.
fn training_rows() -> Vec<Vec<f64>> {
    training_table()
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

/// `rows` as a table of the columns `c0`, `c1`, ..., which legacy predictors are bound to.
fn legacy_table(rows: &[Vec<f64>]) -> RawTable {
    RawTable::new(
        (0..rows[0].len())
            .map(|column| format!("c{column}"))
            .collect(),
        rows.iter()
            .map(|row| row.iter().map(f64::to_string).collect())
            .collect(),
    )
    .unwrap()
}

/// The serialized form of the first stage of a legacy pipeline fitted on the training rows.
fn legacy_stage(source: &str, rows: &[Vec<f64>], y: &[f64]) -> serde_json::Value {
    let model = fit_legacy_numeric_bound(&GenomeIr::parse(source).unwrap(), rows, y).unwrap();
    let stage =
        serde_json::to_value(&model).unwrap()["predictor"]["Legacy"]["Pipeline"]["ops"][0].clone();
    assert!(stage.is_object(), "{source}: {stage}");
    stage
}

#[test]
fn fitted_state_that_does_not_fit_the_row_answers_status_8() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    // Column b of the first row is exactly zero, a cell that `impute` fills.
    let test = RawTable::from_csv("a,b\n0.125,0\n0.5,2.5\n2.125,6.5\n").unwrap();
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, &str, Edit); 7] = [
        (
            "select_corr",
            "(pipe X (select_corr k=2) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([0, 99]);
            },
        ),
        // The row is narrower than the state: a fill and an indicator for six columns.
        (
            "impute_narrower",
            "(pipe X (impute add_indicator=true) (ridge))",
            |json| {
                let impute = &mut stage(json, 0)["Legacy"]["Impute"];
                impute["fill"]
                    .as_array_mut()
                    .unwrap()
                    .extend(std::iter::repeat_n(serde_json::json!(0.5), 4));
                impute["indicator_cols"] = serde_json::json!([5]);
            },
        ),
        // The row is wider than the state, with a zero cell past the fill: that cell reads it.
        ("impute_wider", "(pipe X (impute) (ridge))", |json| {
            stage(json, 0)["Legacy"]["Impute"]["fill"]
                .as_array_mut()
                .unwrap()
                .truncate(1);
        }),
        // One guard short: revision 2 read past the guards.
        (
            "pair_ratio",
            "(pipe X (pair_ratio topk=2) (ridge))",
            |json| {
                let eps = stage(json, 0)["Legacy"]["PairRatio"]["eps"]
                    .as_array_mut()
                    .unwrap();
                assert!(!eps.is_empty(), "pair_ratio fitted a pair");
                eps.pop();
            },
        ),
        ("pair_prod", "(pipe X (pair_prod topk=2) (ridge))", |json| {
            stage(json, 0)["Legacy"]["PairProd"]["pairs"] = serde_json::json!([[0, 99]]);
        }),
        (
            "poly_interact",
            "(pipe X (poly_interact) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["PolyInteract"]["pairs"] = serde_json::json!([[99, 1]]);
            },
        ),
        (
            "corr_screen",
            "(pipe X (corr_screen k=32) (ridge))",
            |json| {
                stage(json, 0)["CorrScreen"]["indices"] = serde_json::json!([0, 7]);
            },
        ),
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-width-status-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    let mut checked = Vec::new();
    let mut check = |name: &str, model: &BoundModel, test: &RawTable| {
        let encoded = EncodedTable::from_bound_model(model, test).unwrap();
        let artifact = emit_fitted_cpp17(model).unwrap();
        let (status, _) = compile_status(
            &artifact.source,
            directory,
            name,
            true,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        );
        assert_eq!(status, FITTED_WIDTH_STATUS, "{name}");
        checked.push(name.to_string());
    };
    for (name, source, edit) in cases {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &task(),
            &numeric_schema(),
            &training_table(),
            &y,
        )
        .unwrap_or_else(|error| panic!("{name} fit: {error}"));
        assert!(
            model.predict_table(&test).is_ok(),
            "{name}: the fitted model"
        );
        let model = tampered(&model, edit);
        // The runtime refuses the edited state with an error, and the C++ answers status 8.
        assert!(model.predict_table(&test).is_err(), "{name}: the runtime");
        check(name, &model, &test);
    }

    // `roll` on an empty row, which only a hand edit builds: `select_corr` keeps no column and a
    // legacy `roll` follows it. The runtime refuses it; revision 2 divided by zero there.
    let roll = legacy_stage("(pipe X (roll k=1) (ridge))", &training_rows(), &y);
    let model = fit_table(
        &GenomeIr::parse("(pipe X (select_corr k=1) (ridge))").unwrap(),
        &task(),
        &numeric_schema(),
        &training_table(),
        &y,
    )
    .unwrap();
    let model = tampered(&model, |json| {
        stage(json, 0)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([]);
        json["predictor"]["Pipeline"]["ops"]
            .as_array_mut()
            .unwrap()
            .insert(1, serde_json::json!({ "Legacy": roll }));
    });
    assert!(model.predict_table(&test).is_err(), "roll: the runtime");
    check("roll_empty_row", &model, &test);

    // The table ABI, of a table-aware pipeline, answers the same status.
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    };
    let train = RawTable::from_csv(
        "group,time,value\nA,1,1\nA,2,2\nA,3,3\nA,4,4\nA,5,5\nB,1,10\nB,2,11\nB,3,12\nB,4,13\nB,5,14\n",
    )
    .unwrap();
    let ordered_test =
        RawTable::from_csv("group,time,value\nA,7,7\nA,6,6\nB,6,15\nC,1,20\n").unwrap();
    let model = fit_table(
        &GenomeIr::parse(
            "(pipe X (input numeric=median categorical=native) (lag k=2 append=true) (select_corr k=2) (ridge))",
        )
        .unwrap(),
        &task(),
        &schema,
        &train,
        &y,
    )
    .unwrap();
    assert!(
        model.predict_table(&ordered_test).is_ok(),
        "table: the fitted model"
    );
    let model = tampered(&model, |json| {
        stage(json, 1)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([0, 99]);
    });
    assert!(
        model.predict_table(&ordered_test).is_err(),
        "table: the runtime"
    );
    check("table_select_corr", &model, &ordered_test);

    // A legacy predictor, the path compat_freeze pins, answers the same status. Its runtime,
    // `Model::predict`, gives no prediction there either: it panics.
    let rows = training_rows();
    let model = fit_legacy_numeric_bound(
        &GenomeIr::parse("(pipe X (select_corr k=2) (ridge))").unwrap(),
        &rows,
        &y,
    )
    .unwrap();
    let model = tampered(&model, |json| {
        json["predictor"]["Legacy"]["Pipeline"]["ops"][0]["SelectCorr"]["indices"] =
            serde_json::json!([0, 99]);
    });
    let legacy_test = legacy_table(&[vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]]);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.predict_table(&legacy_test)
    }));
    assert!(!matches!(outcome, Ok(Ok(_))), "legacy: the runtime");
    check("legacy_select_corr", &model, &legacy_test);

    assert_eq!(checked.len(), 10, "{checked:?}");
    scratch.close().unwrap();
}

#[test]
fn empty_stage_outputs_match_the_runtime_bits_on_the_legacy_paths() {
    // One column leaves `pair_ratio` no pair, so its output rows are empty (the compat program
    // `empty_stage_output`), and `ops::finish` pads each to [0.0]. Revision 2 kept them empty.
    let rows = training_rows();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = [vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let raw = legacy_table(&test);
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-empty-stage-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for source in [
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))",
        // Revision 2 divided by zero in `roll` here.
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (roll k=1) (ridge))",
        // Revision 2 read the head's first column past the end of the empty row here.
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=5))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_legacy_numeric_bound(&genome, &rows, &y)
            .unwrap_or_else(|error| panic!("{source} fit: {error}"));
        let expected = bits(model.predict_table(&raw).unwrap());
        assert_eq!(
            bits(fit(&genome, &rows, &y).unwrap().predict(&test.to_vec())),
            expected,
            "{source}: the bound model is the legacy model"
        );
        let artifact = emit_fitted_cpp17(&model).unwrap();
        assert_eq!(
            compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
            expected,
            "{source}"
        );
    }

    // A portfolio member's stages, and the portfolio's projection, pad as well: member 0 sees the
    // two raw columns and empties them as above; member 1 sees the projection, which keeps no
    // column and so is padded to [0.0]. Revision 2 divided by zero in both members' `roll`.
    let one_column = rows.iter().map(|row| vec![row[0]]).collect::<Vec<_>>();
    let member = fit(
        &GenomeIr::parse("(pipe X (select_corr k=1) (pair_ratio topk=4) (roll k=1) (ridge))")
            .unwrap(),
        &rows,
        &y,
    )
    .unwrap();
    let projected = fit(
        &GenomeIr::parse("(pipe X (roll k=1) (ridge))").unwrap(),
        &one_column,
        &y,
    )
    .unwrap();
    let mut select_none = legacy_stage("(pipe X (select_corr k=1) (ridge))", &rows, &y);
    select_none["SelectCorr"]["indices"] = serde_json::json!([]);
    let portfolio = tampered(
        &bound_with_serialized_legacy_head("(pipe X (portfolio folds=5 repeats=1))", &y),
        |json| {
            let portfolio = &mut json["predictor"]["Pipeline"]["head"]["Legacy"]["Portfolio"];
            assert!(portfolio.is_object(), "{portfolio}");
            assert_eq!(
                portfolio["mean_only"],
                serde_json::json!(false),
                "{portfolio}"
            );
            assert_eq!(portfolio["keep"], serde_json::json!([0, 1]), "{portfolio}");
            portfolio["proj"] = select_none;
            portfolio["members"] = serde_json::json!([
                { "name": "empty_stage_member", "proj_view": false, "state": { "Fitted": member } },
                { "name": "projected_member", "proj_view": true, "state": { "Fitted": projected } },
            ]);
            portfolio["weights"] = serde_json::json!([0.5, 0.5]);
        },
    );
    let portfolio_test = RawTable::from_csv("a,b\n0.125,-1.5\n0,2.5\n2.125,6.5\n").unwrap();
    let encoded = EncodedTable::from_bound_model(&portfolio, &portfolio_test).unwrap();
    let artifact = emit_fitted_cpp17(&portfolio).unwrap();
    assert_eq!(
        compile_typed(
            &artifact.source,
            directory,
            false,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        ),
        bits(portfolio.predict_table(&portfolio_test).unwrap()),
        "portfolio member and projection"
    );
    scratch.close().unwrap();
}

#[test]
fn empty_row_reductions_match_the_runtime_evaluator() {
    // No fitted model hands a stage an empty row: the legacy paths pad it, a V3 fit refuses a
    // stage that empties the block, and the V3 runtime refuses a generated column without a
    // source. So the emitted stage runs here on an empty row directly: a driver in the same
    // translation unit calls `transform` with no column, and its bits are compared with
    // `ops::apply_op` on an empty row. The mean and the standard deviation of an empty row
    // divide by one, as the runtime's evaluator does, so each feature is 1; revision 2 divided by
    // zero, and the guard made it 0. (`+ 1` leaves out the sign of an empty sum, which the
    // runtime starts at -0.0 and the emitted loop at +0.0.)
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-empty-row-")
        .tempdir()
        .unwrap();
    for (index, expression) in ["row_mean + 1", "row_std + 1", "row_mean + row_std + 1"]
        .into_iter()
        .enumerate()
    {
        let model = fit_table(
            &GenomeIr::parse(&format!("(pipe X (derive expr=\"{expression}\") (ridge))")).unwrap(),
            &task(),
            &numeric_schema(),
            &training_table(),
            &y,
        )
        .unwrap();
        let json = serde_json::to_value(&model).unwrap();
        let op: feat_math::ops::FittedOp =
            serde_json::from_value(json["predictor"]["Pipeline"]["ops"][0]["Legacy"].clone())
                .unwrap();
        let expected = feat_math::ops::apply_op(&op, &vec![Vec::new()])
            .remove(0)
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(expected, [1.0f64.to_bits()], "{expression}");
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let driver = r#"
#include <cstdio>
int main(){
  const V out=transform(V{},V{});
  for(double value:out){std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}
  return 0;
}
"#;
        let actual = build_and_run(
            &format!("{}\n{driver}", artifact.source),
            scratch.path(),
            &format!("empty-row-{index}"),
            false,
        )
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{expression}");
    }
    scratch.close().unwrap();
}

#[test]
fn tied_piecewise_edges_match_the_runtime_bits() {
    // A binary column gives tied quantile edges: an interval of width 0 encodes the step
    // `value > lo`, as `piecewise_row` does. Revision 2 encoded it as 0.
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let train =
        RawTable::from_csv("a,b\n0,-2\n1,-1\n0,0\n1,1\n1,2\n0,3\n1,4\n0,5\n1,6\n1,7\n").unwrap();
    let model = fit_table(
        &GenomeIr::parse("(pipe X (input numeric=median) (piecewise_linear bins=8) (ridge))")
            .unwrap(),
        &task(),
        &numeric_schema(),
        &train,
        &y,
    )
    .unwrap();
    let json = serde_json::to_value(&model).unwrap();
    let edges = &json["predictor"]["Pipeline"]["ops"][0]["PiecewiseLinear"]["edges"][0];
    let tied = edges
        .as_array()
        .unwrap()
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .count();
    assert!(tied > 0, "the binary column has tied edges: {edges}");
    let test = RawTable::from_csv("a,b\n0,-1.5\n1,2.5\n0.5,6.5\n1,-3\n").unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-tied-edges-")
        .tempdir()
        .unwrap();
    assert_eq!(
        compile_typed(
            &artifact.source,
            scratch.path(),
            false,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        ),
        expected
    );
    scratch.close().unwrap();
}
