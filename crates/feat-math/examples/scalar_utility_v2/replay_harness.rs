//! Strict F/Q/S/P parity and refusal replay, reused for bound utility artifacts.
//! Tests and utility reuse the same exact F/Q/S/P comparisons and emitted mechanisms.
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarLanguage, ScalarModel,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fmt::Write, fs, path::Path, process::Command};
pub(super) type RecipeCase = (
    ScalarExpressionProgram,
    ScalarCarrier,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
    f64,
    ScalarModel,
);

fn expected(
    case: usize,
    p: &ScalarExpressionProgram,
    c: ScalarCarrier,
    x: &Vec<Vec<f64>>,
    y: &[f64],
    probe: &Vec<Vec<f64>>,
    lambda: f64,
) -> String {
    let m = p.fit_training(x, y, c, lambda).unwrap();
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
    // Preserve the existing 240-second command budget, including compiler work.
    let mut bounded = Command::new("timeout");
    bounded
        .arg("240")
        .arg(cmd.get_program())
        .args(cmd.get_args())
        .envs(cmd.get_envs().filter_map(|(k, v)| v.map(|v| (k, v))));
    if let Some(directory) = cmd.get_current_dir() {
        bounded.current_dir(directory);
    }
    let result = bounded.output().unwrap();
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

pub(super) fn check_complete(all: &[RecipeCase], directory: &Path) {
    assert!(!all.is_empty(), "no complete recipe cases");
    fs::create_dir_all(directory).unwrap();
    let mut emitted = vec![];
    let mut reference = String::new();
    let mut rust = String::from("#![allow(dead_code,unused_imports,unused_variables)]\n");
    let mut python = String::new();
    let mut cpp = String::new();
    let mut cpp_includes = std::collections::BTreeSet::from(["#include <cstdio>".to_string()]);
    for (i, (p, c, x, y, probe, lambda, model)) in all.iter().enumerate() {
        assert_eq!(model.program().to_bytes().unwrap(), p.to_bytes().unwrap());
        assert_eq!(model.carrier(), *c);
        assert_eq!(model.lambda().to_bits(), lambda.to_bits());
        let native_refit = p.fit_training(x, y, *c, *lambda).unwrap();
        let unbound: Value = serde_json::from_str(&native_refit.to_json().unwrap()).unwrap();
        let bound: Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
        assert_eq!(unbound["ridge_bits"], bound["ridge_bits"]);
        reference.push_str(&expected(i, p, *c, x, y, probe, *lambda));
        for lang in [
            ScalarLanguage::Rust,
            ScalarLanguage::Python,
            ScalarLanguage::Cpp17,
        ] {
            let source = model.emit_recipe(lang).unwrap();
            emitted.push(json!({"case":i,"language":format!("{lang:?}"),"source_sha256":format!("{:x}",Sha256::digest(source.as_bytes())),"source_bytes":source.len(),"model_sha256":model.identity().unwrap()}));
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
    fs::write(
        directory.join("executed-source-identities.json"),
        serde_json::to_vec_pretty(&emitted).unwrap(),
    )
    .unwrap();
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

pub(super) fn check_fitted(all: &[(ScalarModel, Vec<Vec<f64>>)], directory: &Path) {
    assert!(!all.is_empty(), "no fitted recipe cases");
    fs::create_dir_all(directory).unwrap();
    let mut includes = std::collections::BTreeSet::from(["#include <cstdio>".to_string()]);
    let mut cpp = String::new();
    let mut reference = String::new();
    for (i, (m, probe)) in all.iter().enumerate() {
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
        cpp.push_str(&format!("namespace case{i}{{\n{body}\nstatic void probe(){{const auto rows={rows};const auto before=scalar_predict_v2(rows);std::printf(\"C {i}\\n\");for(double v:before.first){{std::uint64_t b;std::memcpy(&b,&v,8);std::printf(\"P %016llx\\n\",static_cast<unsigned long long>(b));}}for(auto v:before.second)std::printf(\"Q %u\\n\",v);scalar_predict_v2(std::vector<std::vector<double>>{{std::vector<double>({},1e200)}});const auto after=scalar_predict_v2(rows);if(before!=after)throw std::runtime_error(\"mutable fitted prediction\");const auto empty=scalar_predict_v2({{}});if(!empty.first.empty()||!empty.second.empty())throw std::runtime_error(\"empty fitted prediction\");bool refused=false;try{{scalar_predict_v2(std::vector<std::vector<double>>{{std::vector<double>({},0.0)}});}}catch(const std::invalid_argument&){{refused=true;}}if(!refused)throw std::runtime_error(\"missing fitted width refusal\");}}\n}}\n",m.input_width(),m.input_width()+1));
        let _ = m.predict(&vec![vec![1e200; m.input_width()]]).unwrap();
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
