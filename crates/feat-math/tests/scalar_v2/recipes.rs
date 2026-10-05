//! Executed native, complete-recipe and fitted-artifact scalar-v2 parity.
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::math_expr::{BinOp, Var};
use serde_json::Value;
use std::{fmt::Write, fs, path::Path, process::Command};

type RecipeCase = (
    ScalarExpressionProgram,
    ScalarCarrier,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
);

fn cases() -> Vec<RecipeCase> {
    let mut cases = Vec::new();
    let training = (0..16)
        .map(|i| vec![0.02 + i as f64 * 0.05, -0.02 - i as f64 * 0.05])
        .collect::<Vec<_>>();
    let y = (0..16).map(|i| i as f64 * 0.125 - 1.0).collect::<Vec<_>>();
    let probe = vec![
        vec![0.0, -0.0],
        vec![-0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, -0.0],
        vec![1e200, -1e200],
        vec![1e-300, -1e-300],
        vec![-1e-300, 1e-300],
        vec![f64::NAN, f64::INFINITY],
        vec![0.5, -0.5],
    ];
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            for id in [13, 19, 20, 21, 22] {
                let argument = if carrier == ScalarCarrier::Row {
                    Var::Col(usize::from(id == 22))
                } else {
                    Var::X
                };
                let node = ScalarNode::Binary(
                    BinOp::Mul,
                    Box::new(ScalarNode::Const(0.5)),
                    Box::new(ScalarNode::Call(
                        ScalarFunction::from_versioned_id(2, id).unwrap(),
                        vec![ScalarNode::Var(argument)],
                    )),
                );
                cases.push((
                    ScalarExpressionProgram::new(node)
                        .unwrap()
                        .with_numerical_policy(policy),
                    carrier,
                    training.clone(),
                    y.clone(),
                    probe.clone(),
                ));
            }
        }
        // Every retained function is also evaluated below a new call: hidden
        // NaN, signed-zero or protected branch differences remain observable.
        for id in 0..=18 {
            let mut args = vec![ScalarNode::Var(Var::Col(0))];
            if matches!(id, 7 | 8 | 10) {
                args.push(ScalarNode::Var(Var::Col(1)));
            }
            if id == 9 {
                args.extend([ScalarNode::Const(-1.0), ScalarNode::Const(1.0)]);
            }
            let old = ScalarNode::Call(ScalarFunction::from_versioned_id(2, id).unwrap(), args);
            let node =
                ScalarNode::Call(ScalarFunction::from_versioned_id(2, 21).unwrap(), vec![old]);
            cases.push((
                ScalarExpressionProgram::new(node)
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        for variable in [Var::RowMin, Var::RowMax, Var::RowStd] {
            cases.push((
                ScalarExpressionProgram::new(ScalarNode::Call(
                    ScalarFunction::from_versioned_id(2, 21).unwrap(),
                    vec![ScalarNode::Var(variable)],
                ))
                .unwrap()
                .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        // Old protected operators nested below new calls must preserve their faults.
        let function = |id, argument| {
            ScalarNode::Call(
                ScalarFunction::from_versioned_id(2, id).unwrap(),
                vec![argument],
            )
        };
        let col = |index| ScalarNode::Var(Var::Col(index));
        let mixed = [
            function(
                21,
                ScalarNode::Call(
                    ScalarFunction::from_versioned_id(2, 9).unwrap(),
                    vec![
                        col(0),
                        col(1),
                        ScalarNode::Unary(feat_math::math_expr::UnOp::Neg, Box::new(col(1))),
                    ],
                ),
            ),
            function(
                19,
                function(
                    1,
                    ScalarNode::Unary(feat_math::math_expr::UnOp::Neg, Box::new(col(0))),
                ),
            ),
            function(21, function(12, col(0))),
            function(20, function(2, col(1))),
            function(
                22,
                ScalarNode::Binary(BinOp::Div, Box::new(col(1)), Box::new(col(0))),
            ),
            function(
                19,
                ScalarNode::Binary(
                    BinOp::Pow,
                    Box::new(col(1)),
                    Box::new(ScalarNode::Const(9.0)),
                ),
            ),
            function(20, function(6, function(21, col(0)))),
        ];
        for node in mixed {
            cases.push((
                ScalarExpressionProgram::new(node)
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                training.clone(),
                y.clone(),
                probe.clone(),
            ));
        }
        for var in [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2] {
            let rows = vec![
                vec![1e20, 0.25, -1e20],
                vec![1e20, -0.5, -1e20],
                vec![-0.0, -0.0, -0.0],
                vec![1.0, 1e-16, 1.0],
            ];
            let probe = vec![
                vec![-0.0; 3],
                vec![3e200, 4e200, 0.0],
                vec![3e-200, 4e-200, 0.0],
                vec![f64::NAN, f64::INFINITY, 0.0],
            ];
            cases.push((
                ScalarExpressionProgram::new(ScalarNode::Var(var))
                    .unwrap()
                    .with_numerical_policy(policy),
                ScalarCarrier::Row,
                rows,
                vec![0.0, 1.0, 2.0, 3.0],
                probe,
            ));
        }
        // Signed zero in fitted state is observable even when final prediction adds +0.
        cases.push((
            ScalarExpressionProgram::new(ScalarNode::Var(Var::RowSum))
                .unwrap()
                .with_numerical_policy(policy),
            ScalarCarrier::Row,
            vec![vec![-0.0; 2]; 4],
            vec![-0.0; 4],
            vec![vec![-0.0; 2], vec![0.0; 2]],
        ));
    }
    cases.extend(signed_zero_label_cases());
    cases.extend(pow_literal_base_cases());
    cases
}

fn pow_literal_base_cases() -> Vec<RecipeCase> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/pow_literal_bases.v1.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "scalar_v2.pow_literal_bases.v1");
    let rows = |value: &Value| -> Vec<Vec<f64>> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f64::from_bits(v.as_u64().unwrap()))
                    .collect()
            })
            .collect()
    };
    let training = rows(&fixture["training_bits"]);
    let probe = rows(&fixture["probe_bits"]);
    let labels: Vec<_> = fixture["labels_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect();
    let mut cases = vec![];
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
            let argument = if carrier == ScalarCarrier::Row {
                Var::Col(0)
            } else {
                Var::X
            };
            for base in fixture["base_bits"].as_array().unwrap() {
                let root = ScalarNode::Binary(
                    BinOp::Pow,
                    Box::new(ScalarNode::Const(f64::from_bits(base.as_u64().unwrap()))),
                    Box::new(ScalarNode::Var(argument)),
                );
                cases.push((
                    ScalarExpressionProgram::new(root)
                        .unwrap()
                        .with_numerical_policy(policy),
                    carrier,
                    training.clone(),
                    labels.clone(),
                    probe.clone(),
                ));
            }
        }
    }
    assert_eq!(cases.len(), 16);
    cases
}

fn signed_zero_label_cases() -> Vec<RecipeCase> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/scalar_v2/ridge_signed_zero_labels.v1.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema"], "scalar_v2.ridge_signed_zero_labels.v1");
    let floats = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f64::from_bits(v.as_u64().unwrap()))
            .collect::<Vec<_>>()
    };
    let rows = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(&floats)
            .collect::<Vec<_>>()
    };
    let training = rows(&fixture["training_bits"]);
    let probes = rows(&fixture["probe_bits"]);
    assert!(training
        .iter()
        .flatten()
        .all(|v| v.is_finite() && *v != 0.0));
    let mut cases = Vec::new();
    for labels in fixture["cases"].as_array().unwrap() {
        let y = floats(&labels["labels_bits"]);
        assert!(y.iter().all(|v| *v == 0.0));
        for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
            for (carrier, variable) in [
                (ScalarCarrier::Row, Var::Col(0)),
                (ScalarCarrier::Row, Var::Col(2)),
                (ScalarCarrier::Elementwise, Var::X),
            ] {
                let p = ScalarExpressionProgram::new(ScalarNode::Var(variable))
                    .unwrap()
                    .with_numerical_policy(policy);
                let model = p.fit_training(&training, &y, carrier, 0.02).unwrap();
                let state: Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
                let weights = state["ridge_bits"][2].as_array().unwrap();
                assert_eq!(weights[0], labels["intercept_bits"], "{}", labels["name"]);
                assert!(weights[1..].iter().all(|v| v.as_u64() == Some(0)));
                cases.push((p, carrier, training.clone(), y.clone(), probes.clone()));
            }
        }
    }
    cases
}

fn expected(
    case: usize,
    p: &ScalarExpressionProgram,
    c: ScalarCarrier,
    x: &Vec<Vec<f64>>,
    y: &[f64],
    probe: &Vec<Vec<f64>>,
) -> String {
    let m = p.fit_training(x, y, c, 0.02).unwrap();
    let (f, q) = m.transform(probe).unwrap();
    let (pred, faults) = m.predict(probe).unwrap();
    assert_eq!(q, faults);
    let state = m.to_json().unwrap();
    let wire: Value = serde_json::from_str(&state).unwrap();
    let mut out = format!("C {case}\n");
    for row in f {
        for v in row {
            writeln!(out, "F {:016x}", v.to_bits()).unwrap();
        }
    }
    for v in q {
        writeln!(out, "Q {v}").unwrap();
    }
    for row in wire["ridge_bits"].as_array().unwrap() {
        for v in row.as_array().unwrap() {
            writeln!(out, "S {:016x}", v.as_u64().unwrap()).unwrap();
        }
    }
    for v in pred {
        writeln!(out, "P {:016x}", v.to_bits()).unwrap();
    }
    let _ = m.predict(&vec![vec![1e200; x[0].len()]]).unwrap();
    assert_eq!(m.to_json().unwrap(), state);
    out
}

fn literal(value: f64, language: ScalarLanguage) -> String {
    match language {
        ScalarLanguage::Rust => format!("f64::from_bits({}u64)", value.to_bits()),
        ScalarLanguage::Python => format!("scalar_bits_v2({})", value.to_bits()),
        ScalarLanguage::Cpp17 => format!("f64_from_bits({}ULL)", value.to_bits()),
    }
}
fn vector(values: &[f64], language: ScalarLanguage) -> String {
    let inner = values
        .iter()
        .map(|&v| literal(v, language))
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{inner}]"),
        ScalarLanguage::Python => format!("[{inner}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<double>{{{inner}}}"),
    }
}
fn matrix(rows: &[Vec<f64>], language: ScalarLanguage) -> String {
    let inner = rows
        .iter()
        .map(|r| vector(r, language))
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{inner}]"),
        ScalarLanguage::Python => format!("[{inner}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<std::vector<double>>{{{inner}}}"),
    }
}
fn run(cmd: &mut Command) -> String {
    let result = cmd.output().unwrap();
    assert!(
        result.status.success(),
        "{cmd:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

fn assert_output(observed: String, expected: &str, label: &str, path: &Path) {
    fs::write(path, &observed).unwrap();
    let actual = observed.lines().collect::<Vec<_>>();
    let expected = expected.lines().collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len(), "{label}: output count");
    let mut case = "";
    for (line, (a, b)) in actual.iter().zip(&expected).enumerate() {
        if b.starts_with("C ") {
            case = b;
        }
        assert_eq!(a, b, "{label}: {case}, line {}", line + 1);
    }
}

#[test]
fn complete_scalar_recipes_match_native_features_faults_state_and_predictions() {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("scalar-v2-recipes");
    fs::create_dir_all(&directory).unwrap();
    let mut reference = String::new();
    let mut rust = String::from("#![allow(dead_code,unused_imports,unused_variables)]\n");
    let mut python = String::new();
    let mut cpp = String::new();
    let mut cpp_includes = std::collections::BTreeSet::from(["#include <cstdio>".to_string()]);
    let all = cases();
    for (i, (p, c, x, y, probe)) in all.iter().enumerate() {
        reference.push_str(&expected(i, p, *c, x, y, probe));
        for lang in [
            ScalarLanguage::Rust,
            ScalarLanguage::Python,
            ScalarLanguage::Cpp17,
        ] {
            let source = p.emit_recipe(lang, *c, x[0].len(), 0.02).unwrap();
            let xx = matrix(x, lang);
            let yy = vector(y, lang);
            let pp = matrix(probe, lang);
            match lang {
                ScalarLanguage::Rust => {
                    let source = source.replace("\nfn main(", "\nfn scalar_cli_main(");
                    rust.push_str(&format!("mod case{i}{{\n{source}\npub fn probe(){{let x={xx};let y={yy};let rows={pp};let m=Model::fit(&x,&y);let (f,q)=scalar_stage1_values(&rows);let state=||m.head.means.iter().chain(&m.head.scales).chain(&m.head.w).map(|v|v.to_bits()).collect::<Vec<_>>();let before=state();let (pred,faults)=m.predict_with_faults(&rows);assert_eq!(q,faults);println!(\"C {i}\");for row in f{{for v in row{{println!(\"F {{:016x}}\",v.to_bits());}}}}for v in q{{println!(\"Q {{}}\",v);}}for v in &before{{println!(\"S {{:016x}}\",v);}}for v in pred{{println!(\"P {{:016x}}\",v.to_bits());}}let _=m.predict(&vec![vec![1e200;{}]]);assert_eq!(before,state());assert_eq!(m.predict_with_faults(&vec![]),(vec![],vec![]));assert!(std::panic::catch_unwind(||m.predict(&vec![vec![0.0;{}]])).is_err());assert!(std::panic::catch_unwind(||Model::fit(&x,&vec![f64::NAN;x.len()])).is_err());}}\n}}\n",x[0].len(),x[0].len()+1));
                }
                ScalarLanguage::Python => {
                    // Every recipe retains its own module globals; the standalone CLI is idle.
                    python.push_str(&format!("ns={{'__name__':'scalar_test_{i}'}}\nexec({source:?},ns)\nexec(\"\"\"\nx={xx}\ny={yy}\nrows={pp}\nm=Model().fit(x,y)\nf,q=scalar_stage1_values(rows)\nstate=lambda: [struct.unpack('<Q',struct.pack('<d',v))[0] for part in m.head for v in part]\nbefore=state()\npred,faults=m.predict_with_faults(rows)\nassert q==faults\nprint('C {i}')\nfor row in f:\n    for v in row: print('F %016x'%struct.unpack('<Q',struct.pack('<d',v))[0])\nfor v in q: print('Q %d'%v)\nfor v in before: print('S %016x'%v)\nfor v in pred: print('P %016x'%struct.unpack('<Q',struct.pack('<d',v))[0])\nm.predict([[1e200]*{}])\nassert before==state()\nassert m.predict_with_faults([])==([],[])\nfor action in [lambda:m.predict([[0.0]*{}]),lambda:Model().fit(x,[float('nan')]*len(x))]:\n    try: action()\n    except ValueError: pass\n    else: raise AssertionError('missing scalar refusal')\n\"\"\",ns)\n",x[0].len(),x[0].len()+1));
                }
                ScalarLanguage::Cpp17 => {
                    let mut body = String::new();
                    for line in source.lines() {
                        if line.starts_with("#include ") {
                            cpp_includes.insert(line.to_string());
                        } else {
                            writeln!(body, "{line}").unwrap();
                        }
                    }
                    // A namespace gives each C ABI symbol a unique assembler name in this bundled harness.
                    body = body.replace("extern \"C\"", "extern \"C++\"");
                    cpp.push_str(&format!("namespace case{i}{{\n{body}\nstatic void probe(){{const auto x={xx};const auto y={yy};const auto rows={pp};const auto m=Model::fit(x,y);const auto features=scalar_stage1_values(rows);auto state=[&](){{std::vector<std::uint64_t> out;for(const auto* part:{{&m.head.means,&m.head.scales,&m.head.w}})for(double v:*part){{std::uint64_t b;std::memcpy(&b,&v,8);out.push_back(b);}}return out;}};const auto before=state();const auto predictions=m.predict_with_faults(rows);if(features.second!=predictions.second)throw std::runtime_error(\"fault mismatch\");std::printf(\"C {i}\\n\");for(const auto& row:features.first)for(double v:row){{std::uint64_t b;std::memcpy(&b,&v,8);std::printf(\"F %016llx\\n\",static_cast<unsigned long long>(b));}}for(auto v:features.second)std::printf(\"Q %u\\n\",v);for(auto v:before)std::printf(\"S %016llx\\n\",static_cast<unsigned long long>(v));for(double v:predictions.first){{std::uint64_t b;std::memcpy(&b,&v,8);std::printf(\"P %016llx\\n\",static_cast<unsigned long long>(b));}}m.predict(std::vector<std::vector<double>>{{std::vector<double>({},1e200)}});if(before!=state())throw std::runtime_error(\"prediction changed state\");if(!m.predict_with_faults({{}}).first.empty())throw std::runtime_error(\"empty prediction\");bool refused=false;try{{m.predict(std::vector<std::vector<double>>{{std::vector<double>({},0.0)}});}}catch(const std::invalid_argument&){{refused=true;}}if(!refused)throw std::runtime_error(\"missing width refusal\");refused=false;try{{Model::fit(x,std::vector<double>(x.size(),std::numeric_limits<double>::quiet_NaN()));}}catch(const std::invalid_argument&){{refused=true;}}if(!refused)throw std::runtime_error(\"missing label refusal\");}}\n}}\n",x[0].len(),x[0].len()+1));
                }
            }
        }
    }
    rust.push_str("fn main(){");
    cpp.push_str("int main(){");
    for i in 0..all.len() {
        write!(rust, "case{i}::probe();").unwrap();
        write!(cpp, "case{i}::probe();").unwrap();
    }
    rust.push_str("}\n");
    cpp.push_str("return 0;}\n");
    let cpp = format!(
        "{}\n{cpp}",
        cpp_includes.into_iter().collect::<Vec<_>>().join("\n")
    );
    fs::write(directory.join("native-reference.txt"), &reference).unwrap();
    fs::write(directory.join("recipes.rs"), rust).unwrap();
    fs::write(directory.join("recipes.py"), python).unwrap();
    fs::write(directory.join("recipes.cpp"), cpp).unwrap();
    let rs_binary = directory.join("recipes-rust");
    run(Command::new("rustc")
        .args(["--edition=2021", "-C", "opt-level=2"])
        .arg(directory.join("recipes.rs"))
        .arg("-o")
        .arg(&rs_binary));
    assert_output(
        run(&mut Command::new(rs_binary)),
        &reference,
        "Rust scalar recipe bits",
        &directory.join("rust-observed.txt"),
    );
    assert_output(
        run(Command::new("python3").arg(directory.join("recipes.py"))),
        &reference,
        "Python scalar recipe bits",
        &directory.join("python-observed.txt"),
    );
    for (compiler, standard) in [("g++", "-std=c++17"), ("clang++", "-std=c++20")] {
        let binary = directory.join(compiler);
        let mut cmd = Command::new(compiler);
        cmd.args([standard, "-O2", "-ffp-contract=off", "-fno-fast-math"]);
        if let Ok(sanitizer) = std::env::var("JOPE_FITTED_CPP_SANITIZER") {
            cmd.args([
                "-O1",
                "-fno-omit-frame-pointer",
                "-fno-pie",
                "-no-pie",
                "-fno-sanitize-recover=all",
            ])
            .arg(format!("-fsanitize={sanitizer}"));
        }
        run(cmd
            .arg(directory.join("recipes.cpp"))
            .arg("-o")
            .arg(&binary));
        assert_output(
            run(&mut Command::new(binary)),
            &reference,
            &format!("{compiler} scalar recipe bits"),
            &directory.join(format!("{compiler}-observed.txt")),
        );
    }
}

#[test]
fn fitted_scalar_cpp_replays_native_predictions_faults_and_refusals() {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("scalar-v2-fitted");
    fs::create_dir_all(&directory).unwrap();
    let mut includes = std::collections::BTreeSet::from(["#include <cstdio>".to_string()]);
    let mut cpp = String::new();
    let mut reference = String::new();
    let all = cases();
    for (i, (p, c, x, y, probe)) in all.iter().enumerate() {
        let m = p.fit_training(x, y, *c, 0.02).unwrap();
        let frozen = m.to_json().unwrap();
        let (pred, faults) = m.predict(probe).unwrap();
        writeln!(reference, "C {i}").unwrap();
        for v in pred {
            writeln!(reference, "P {:016x}", v.to_bits()).unwrap();
        }
        for v in faults {
            writeln!(reference, "Q {v}").unwrap();
        }
        let source = m.emit_fitted_cpp17().unwrap();
        let mut body = String::new();
        for line in source.lines() {
            if line.starts_with("#include ") {
                includes.insert(line.to_string());
            } else {
                writeln!(body, "{line}").unwrap();
            }
        }
        let rows = matrix(probe, ScalarLanguage::Cpp17);
        cpp.push_str(&format!("namespace case{i}{{\n{body}\nstatic void probe(){{const auto rows={rows};const auto before=scalar_predict_v2(rows);std::printf(\"C {i}\\n\");for(double v:before.first){{std::uint64_t b;std::memcpy(&b,&v,8);std::printf(\"P %016llx\\n\",static_cast<unsigned long long>(b));}}for(auto v:before.second)std::printf(\"Q %u\\n\",v);scalar_predict_v2(std::vector<std::vector<double>>{{std::vector<double>({},1e200)}});const auto after=scalar_predict_v2(rows);if(before!=after)throw std::runtime_error(\"mutable fitted prediction\");const auto empty=scalar_predict_v2({{}});if(!empty.first.empty()||!empty.second.empty())throw std::runtime_error(\"empty fitted prediction\");bool refused=false;try{{scalar_predict_v2(std::vector<std::vector<double>>{{std::vector<double>({},0.0)}});}}catch(const std::invalid_argument&){{refused=true;}}if(!refused)throw std::runtime_error(\"missing fitted width refusal\");}}\n}}\n",x[0].len(),x[0].len()+1));
        let _ = m.predict(&vec![vec![1e200; x[0].len()]]).unwrap();
        assert_eq!(m.to_json().unwrap(), frozen);
    }
    cpp.push_str("int main(){");
    for i in 0..all.len() {
        write!(cpp, "case{i}::probe();").unwrap();
    }
    cpp.push_str("return 0;}\n");
    let cpp = format!(
        "{}\n{cpp}",
        includes.into_iter().collect::<Vec<_>>().join("\n")
    );
    fs::write(directory.join("native-reference.txt"), &reference).unwrap();
    fs::write(directory.join("fitted.cpp"), cpp).unwrap();
    for (compiler, standard) in [("g++", "-std=c++17"), ("clang++", "-std=c++20")] {
        let binary = directory.join(compiler);
        let mut cmd = Command::new(compiler);
        cmd.args([standard, "-O2", "-ffp-contract=off", "-fno-fast-math"]);
        if let Ok(sanitizer) = std::env::var("JOPE_FITTED_CPP_SANITIZER") {
            cmd.args([
                "-O1",
                "-fno-omit-frame-pointer",
                "-fno-pie",
                "-no-pie",
                "-fno-sanitize-recover=all",
            ])
            .arg(format!("-fsanitize={sanitizer}"));
        }
        run(cmd.arg(directory.join("fitted.cpp")).arg("-o").arg(&binary));
        assert_output(
            run(&mut Command::new(binary)),
            &reference,
            &format!("{compiler} fitted scalar bits"),
            &directory.join(format!("{compiler}-observed.txt")),
        );
    }
}

#[test]
fn scalar_recipe_admission_refuses_bad_scope_shape_and_labels() {
    let variable = ScalarExpressionProgram::new(ScalarNode::Var(Var::X)).unwrap();
    assert!(variable.plan_training(ScalarCarrier::Row, 2, 0.02).is_err());
    let p = ScalarExpressionProgram::new(ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 21).unwrap(),
        vec![ScalarNode::Var(Var::Col(0))],
    ))
    .unwrap();
    for width in [0, 513] {
        assert!(p.plan_training(ScalarCarrier::Row, width, 0.02).is_err());
    }
    for lambda in [f64::NAN, f64::INFINITY, 0.0, 2.001] {
        assert!(p.plan_training(ScalarCarrier::Row, 1, lambda).is_err());
    }
    let source = p
        .emit_recipe(ScalarLanguage::Python, ScalarCarrier::Row, 1, 0.02)
        .unwrap();
    let source = format!("__name__='scalar_admission'\n{source}\nfor action in [lambda:Model().fit([],[]),lambda:Model().fit([[0.0]],[1e13]),lambda:Model().fit([[0.0]],[]),lambda:Model().fit([[0.0,1.0]],[0.0])]:\n    try: action()\n    except ValueError: pass\n    else: raise AssertionError('missing fit refusal')\n");
    run(Command::new("python3").arg("-c").arg(source));
}

#[test]
fn nested_clip_nan_bounds_refuse_before_parent_can_hide_the_failure() {
    let clip = ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 9).unwrap(),
        vec![
            ScalarNode::Var(Var::Col(0)),
            ScalarNode::Var(Var::Col(1)),
            ScalarNode::Var(Var::Col(2)),
        ],
    );
    let p = ScalarExpressionProgram::new(ScalarNode::Call(
        ScalarFunction::from_versioned_id(2, 21).unwrap(),
        vec![clip],
    ))
    .unwrap();
    let model = p
        .fit_training(
            &vec![vec![0.0, -1.0, 1.0], vec![0.5, -1.0, 1.0]],
            &[0.0, 1.0],
            ScalarCarrier::Row,
            0.02,
        )
        .unwrap();
    assert!(model.predict(&vec![vec![0.0, f64::NAN, f64::NAN]]).is_err());
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("scalar-v2-clip-refusal");
    fs::create_dir_all(&directory).unwrap();
    for lang in [
        ScalarLanguage::Rust,
        ScalarLanguage::Python,
        ScalarLanguage::Cpp17,
    ] {
        let source = p.emit_recipe(lang, ScalarCarrier::Row, 3, 0.02).unwrap();
        let (source,file) = match lang {
            ScalarLanguage::Rust => (format!("{}\nfn main(){{assert!(std::panic::catch_unwind(||scalar_stage1_values(&vec![vec![0.0,f64::NAN,f64::NAN]])).is_err());}}",source.replace("\nfn main(","\nfn scalar_cli_main(")), "refusal.rs"),
            ScalarLanguage::Python => (format!("__name__='scalar_refusal'\n{source}\ntry: scalar_stage1_values([[0.0,float('nan'),float('nan')]])\nexcept ArithmeticError: pass\nelse: raise AssertionError('hidden clip failure')\n"),"refusal.py"),
            ScalarLanguage::Cpp17 => (format!("{source}\nint main(){{bool refused=false;try{{scalar_stage1_values({{{{0.0,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()}}}});}}catch(const std::domain_error&){{refused=true;}}return refused?0:1;}}"),"refusal.cpp"),
        };
        let path = directory.join(file);
        fs::write(&path, source).unwrap();
        match lang {
            ScalarLanguage::Rust => {
                let binary = directory.join("rust");
                run(Command::new("rustc")
                    .args(["--edition=2021", "-C", "opt-level=2"])
                    .arg(&path)
                    .arg("-o")
                    .arg(&binary));
                run(&mut Command::new(binary));
            }
            ScalarLanguage::Python => {
                run(Command::new("python3").arg(&path));
            }
            ScalarLanguage::Cpp17 => {
                let binary = directory.join("cpp");
                run(Command::new("g++")
                    .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
                    .arg(&path)
                    .arg("-o")
                    .arg(&binary));
                run(&mut Command::new(binary));
            }
        }
    }
}

#[test]
fn scalar_recipe_and_fitted_sidecars_bind_the_executed_program_policy_and_state() {
    use feat_math::kernel::wire_v2::VersionedProgram;
    use sha2::{Digest, Sha256};
    let hash = |value: &[u8]| format!("{:x}", Sha256::digest(value));
    for policy in [NumericalPolicy::LegacyV1, NumericalPolicy::StableV2] {
        let p = ScalarExpressionProgram::new(ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, 13).unwrap(),
            vec![ScalarNode::Var(Var::Col(0))],
        ))
        .unwrap()
        .with_numerical_policy(policy);
        let (paired, cpp) = p.emit_recipes(ScalarCarrier::Row, 1, 0.02).unwrap();
        let wire: Value = serde_json::from_str(&paired.manifest).unwrap();
        let bytes: Vec<u8> = serde_json::from_value(wire["program"].clone()).unwrap();
        assert_eq!(bytes, p.to_bytes().unwrap());
        assert_eq!(wire["program_sha256"], hash(&bytes));
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(wire["min_reader_version"], 2);
        assert_eq!(wire["numerical_policy"], policy.name());
        assert_eq!(
            wire["emitter_feature_identity"],
            "feat-math.scalar.ridge.recipe.v2"
        );
        assert_eq!(
            wire["profile"],
            feat_math::kernel::scalar_program::MODEL_PROFILE
        );
        for (field, source) in [
            ("py_sha256", paired.python.as_str()),
            ("rs_sha256", paired.rust.as_str()),
            ("cpp17_sha256", cpp.source.as_str()),
        ] {
            assert_eq!(wire[field], hash(source.as_bytes()));
            assert!(source.contains("feat-math.scalar.ridge.recipe.v2"));
            assert!(source.contains(&format!("policy={}", policy.name())));
        }
        assert_eq!(cpp.tokens.bytes, cpp.source.len());
        assert_eq!(paired.py_tokens.bytes, paired.python.len());
        assert_eq!(paired.rs_tokens.bytes, paired.rust.len());
        let second = p.emit_recipes(ScalarCarrier::Row, 1, 0.02).unwrap();
        assert_eq!(paired.manifest, second.0.manifest);
        let model = p
            .fit_training(
                &vec![vec![-1.0], vec![1.0]],
                &[-2.0, 2.0],
                ScalarCarrier::Row,
                0.02,
            )
            .unwrap();
        let artifact = model.emit_fitted_artifact().unwrap();
        assert_eq!(artifact.codec, model.to_json().unwrap());
        assert_eq!(artifact.source_sha256, hash(artifact.source.as_bytes()));
        assert_eq!(artifact.codec_sha256, hash(artifact.codec.as_bytes()));
        assert!(artifact.source.contains(&model.identity().unwrap()));
        assert_eq!(artifact.source, model.emit_fitted_cpp17().unwrap());
        let restored =
            feat_math::kernel::scalar_program::ScalarModel::from_json(&artifact.codec).unwrap();
        assert_eq!(restored.emit_fitted_cpp17().unwrap(), artifact.source);
    }
}
