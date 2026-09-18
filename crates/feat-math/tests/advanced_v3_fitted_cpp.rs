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
        let directory = std::env::temp_dir().join(format!(
            "kernel-expression-safety-{}-{index}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        assert_eq!(
            compile_and_run(&artifact.source, &directory, true),
            expected,
            "{source}"
        );
        std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert_eq!(compile_and_run(&first.source, &directory, false), expected);
    assert_eq!(compile_and_run(&first.source, &directory, true), expected);
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-ordered-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert_eq!(
        compile_and_run(&artifact.source, &directory, false),
        expected
    );
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-corpus-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
            compile_and_run(&artifact.source, &directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
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
    let test = vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-row-corpus-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
                &directory,
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
    std::fs::remove_dir_all(directory).unwrap();
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
    let test = vec![
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-all-legacy-ops-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
                &directory,
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
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-binary-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
            compile_and_run(&artifact.source, &directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-categorical-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
            compile_typed(&artifact.source, &directory, false, &values, &missing, 3, 2),
            expected,
            "categorical mode {mode}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-historical-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert_eq!(
        compile_typed(&artifact.source, &directory, false, &values, &missing, 4, 3,),
        expected
    );
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-table-ops-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
                &directory,
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
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-legacy-heads-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
            compile_and_run(&artifact.source, &directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
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
    let test = vec![
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-nontrivial-heads-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
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
                        &directory,
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
                &directory,
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
    std::fs::remove_dir_all(directory).unwrap();
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
    let directory = std::env::temp_dir().join(format!(
        "feat-math-advanced-v3-fitted-cpp-legacy-bridge-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    assert_eq!(
        compile_typed(&artifact.source, &directory, false, &values, &missing, 3, 2),
        expected
    );
    std::fs::remove_dir_all(directory).unwrap();
}
