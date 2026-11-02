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
            .unwrap()
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

#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_1.rs"]
mod part_1;
#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_2.rs"]
mod part_2;
#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_3.rs"]
mod part_3;
#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_4.rs"]
mod part_4;
#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_5.rs"]
mod part_5;
#[cfg(test)]
#[path = "advanced_v3_fitted_cpp/part_6.rs"]
mod part_6;
