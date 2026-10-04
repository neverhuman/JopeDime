use feat_math::kernel::ordered_axis::Axis;
use feat_math::kernel::row_vector::{
    emit::{self, Language},
    Edit, ExperimentalProfile, FittedStage, Model, Program, Stage,
};
use feat_math::kernel::savgol::{Config, BOUNDARY_DEGREE};
use feat_math::kernel::wire_v2::{VersionedModel, VersionedProgram};
use feat_math::math_expr::FaultSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn config(w: u8, p: u8, r: u8) -> Config {
    Config::new(2, 58, Axis::FeatureWithinRow, w, p, r, "shrink").unwrap()
}
fn program(width: usize, c: Config) -> Program {
    Program::new(2, width, vec![Stage::Savgol(c)], 0.02).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn bounded_config_rejects_versions_axes_parameters_and_legacy_ingress() {
    for (v, id, axis, w, p, r, b) in [
        (1, 58, Axis::FeatureWithinRow, 5, 2, 1, "shrink"),
        (2, 59, Axis::FeatureWithinRow, 5, 2, 1, "shrink"),
        (2, 58, Axis::SampleTime, 5, 2, 1, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 2, 1, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 4, 1, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 131, 2, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 3, 3, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 9, 7, 0, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 5, 2, 3, "shrink"),
        (2, 58, Axis::FeatureWithinRow, 5, 2, 0, "mirror"),
    ] {
        assert!(Config::new(v, id, axis, w, p, r, b).is_err());
    }
    assert!(config(5, 2, 0).fit(feat_math::MAX_WIDTH + 1).is_err());
    let old = feat_math::GenomeIr::parse("(pipe X (savgol w=5 p=2 r=0) (ridge lambda_frac=0.02))");
    if let Ok(old) = old {
        assert!(feat_math::fit(&old, &vec![vec![1.0; 5]; 4], &[1.0; 4]).is_err());
        assert!(feat_math::transpile::emit(&old, &Default::default()).is_err());
    }
    assert!(Program::new(1, 5, vec![], 0.02).is_err());
    assert!(Program::new(2, 0, vec![], 0.02).is_err());
    assert!(Program::new(2, 5, vec![Stage::Savgol(config(5, 2, 0)); 9], 0.02).is_err());
}

#[test]
fn polynomial_and_derivative_reproduction_covers_clipped_and_maximum_windows() {
    for w in [3, 5, 9, 17, 129] {
        for p in 0..=6.min(w - 1) {
            for r in 0..=p {
                let n = usize::from(w) + 7;
                let row = (0..n)
                    .map(|j| (j as f64 / n as f64).powi(i32::from(p)))
                    .collect::<Vec<_>>();
                let result = config(w, p, r).fit(n).unwrap().apply(&row).unwrap();
                for (j, &actual) in result.values.iter().enumerate() {
                    let left = j.min(usize::from(w / 2));
                    let right = (n - 1 - j).min(usize::from(w / 2));
                    // Degree is preserved when the clipped window has p+1 samples.
                    if left + right < usize::from(p) {
                        continue;
                    }
                    let factorial = (0..r).fold(1.0, |a, k| a * f64::from(p - k));
                    let expected = factorial * (j as f64 / n as f64).powi(i32::from(p - r))
                        / (n as f64).powi(i32::from(r));
                    assert!(
                        (actual - expected).abs() < 2e-9,
                        "w={w} p={p} r={r} j={j} actual={actual} expected={expected}"
                    );
                    assert_eq!(result.faults[j], 0);
                }
            }
        }
    }
    // Independent analytic five-tap least-squares smoothing weights.
    let state = config(5, 2, 0).fit(5).unwrap();
    for (i, numerator) in [-3.0, 12.0, 17.0, 12.0, -3.0].iter().enumerate() {
        let mut impulse = vec![0.0; 5];
        impulse[i] = 1.0;
        let actual = state.apply(&impulse).unwrap().values[2];
        assert!((actual - numerator / 35.0).abs() < 4e-16);
    }
    // A nonzero sixth derivative prevents a loose near-zero check from
    // admitting a zero implementation at high derivative orders.
    let sixth = config(129, 6, 6)
        .fit(13)
        .unwrap()
        .apply(&(0..13).map(|i| (i as f64).powi(6)).collect::<Vec<_>>())
        .unwrap();
    assert!(
        sixth.values.iter().all(|v| (v - 720.0).abs() < 1e-5),
        "{:?}",
        sixth.values
    );
}

#[test]
fn shrink_short_empty_rows_and_faults_have_defined_bits() {
    for width in [0, 1, 2, 3, 4, 129, feat_math::MAX_WIDTH] {
        let row = vec![2.0; width];
        for r in 0..=6 {
            let out = config(129, 6, r).fit(width).unwrap().apply(&row).unwrap();
            assert_eq!(out.values.len(), width);
            assert_eq!(out.faults.len(), width);
            if usize::from(r) >= width {
                assert!(out.values.iter().all(|v| v.to_bits() == 0));
                assert!(out.faults.iter().all(|&f| f == BOUNDARY_DEGREE));
            }
        }
    }
    let row = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
        0.0,
        -0.0,
    ];
    let output = config(3, 0, 0).fit(row.len()).unwrap().apply(&row).unwrap();
    assert!(output
        .values
        .iter()
        .all(|x| x.is_finite() && x.abs() <= 1e12));
    assert!(output.faults[0] & FaultSet::NONFINITE_INPUT.bits() != 0);
    assert!(output.faults[3] & FaultSet::MAGNITUDE_CLAMPED.bits() != 0);
    assert!(config(5, 2, 0).fit(5).unwrap().apply(&[1.0]).is_err());
    assert_eq!(FaultSet::NONFINITE_INPUT.bits(), 8192);
    assert_eq!(FaultSet::MAGNITUDE_CLAMPED.bits(), 512);
    assert_eq!(FaultSet::NONFINITE_SANITIZED.bits(), 256);
}

#[test]
fn coefficient_model_and_program_state_roundtrip_without_losing_bits() {
    let p = program(7, config(5, 2, 1));
    let bytes = p.to_bytes().unwrap();
    let loaded = Program::from_bytes(&bytes).unwrap();
    assert_eq!(loaded, p);
    let ingress = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(matches!(ingress, VersionedProgram::RowVectorV2(_)));
    assert_eq!(ingress.to_bytes().unwrap(), bytes);
    assert_eq!(
        VersionedProgram::parse_lossless_text(&p.canonical().unwrap())
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    assert!(feat_math::kernel::Program::from_bytes(&bytes).is_err());
    for index in [7, 8, 9, 10] {
        let mut bad = bytes.clone();
        bad[index] = 255;
        assert!(Program::from_bytes(&bad).is_err());
    }
    let (x, y) = training(7);
    let model = p.fit(&x, &y).unwrap();
    let json = model.to_json().unwrap();
    let restored = Model::from_json(&json).unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    assert_eq!(model.identity().unwrap(), restored.identity().unwrap());
    assert_eq!(model.predict(&x).unwrap(), restored.predict(&x).unwrap());
    assert!(matches!(
        VersionedModel::from_json(&json).unwrap(),
        VersionedModel::RowVectorV2(_)
    ));
    assert!(serde_json::from_str::<feat_math::Model>(&json).is_err());
    let mut wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    wire["stages"][0]["state"]["stencils"][0]["bits"][0] = serde_json::json!(0);
    assert!(Model::from_json(&wire.to_string()).is_err());
    assert!(model.predict(&vec![vec![]]).is_err());
    assert!(model.predict(&vec![vec![1.0; 6]]).is_err());
    assert_eq!(model.predict(&vec![]).unwrap(), (vec![], vec![]));
    // Holding out any rows/labels cannot enter this API's training closure.
    let before = model.to_json().unwrap();
    let _ = model.predict(&vec![vec![f64::NAN; 7]; 3]).unwrap();
    assert_eq!(before, model.to_json().unwrap());
    assert_eq!(before, p.fit(&x, &y).unwrap().to_json().unwrap());
}

fn training(width: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = (0..24)
        .map(|i| {
            (0..width)
                .map(|j| ((i * 13 + j * 7) % 37) as f64 / 9.0 - 2.0)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| r[0] * 0.5 + r[width - 1] * 0.2 - 0.3)
        .collect();
    (x, y)
}

#[test]
fn experimental_prior_is_bounded_and_removal_keeps_parsing_available() {
    let p = Program::new(2, 3, vec![], 0.02).unwrap();
    let profile = ExperimentalProfile {
        savgol: true,
        prior: 0.25,
    };
    let created = (0..1024)
        .filter(|&seed| !p.mutate(seed, &profile).unwrap().stages().is_empty())
        .count();
    assert!(
        (200..=320).contains(&created),
        "prior gate created {created}/1024"
    );
    let disabled = ExperimentalProfile {
        savgol: false,
        prior: 0.25,
    };
    assert!((0..32).all(|seed| p.mutate(seed, &disabled).unwrap() == p));
    for prior in [0.0, -0.1, 0.251, f64::NAN, f64::INFINITY] {
        assert!(ExperimentalProfile {
            savgol: true,
            prior
        }
        .validate()
        .is_err());
    }
    let cap = feat_math::kernel::capabilities::row_vector_v2_capability(58).unwrap();
    assert!(cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Experimental));
    assert!(!cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Baseline));
    let created = p.edit(17, Edit::Grow, &profile).unwrap();
    assert_eq!(
        Program::from_bytes(&created.to_bytes().unwrap()).unwrap(),
        created
    );
    assert!(created.emit_recipe(Language::Cpp17).is_ok());
}

#[test]
fn six_signal_datasets_five_seeds_equal_budget_derivative_ablation() {
    // Fixture qualification only: no product corpus, sealed labels, campaign
    // worker, promotion, or broad search-lift claim. Both kernels evaluate the
    // same 64 rows per task/seed; the old reference is the two-neighbor stencil.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/savgol_utility.json"
    ))
    .unwrap();
    let width = fixture["width"].as_u64().unwrap() as usize;
    let fitted = config(17, 6, 1).fit(width).unwrap();
    let mut receipt = Vec::new();
    for task in fixture["tasks"].as_array().unwrap() {
        let degree = task["degree"].as_u64().unwrap() as usize;
        let mut base_total = 0.0;
        let mut new_total = 0.0;
        for seed in fixture["seeds"].as_array().unwrap() {
            let seed = seed.as_u64().unwrap();
            let mut rng = feat_math::Pcg32::new(seed, degree as u64 + 0x61626c617465);
            let mut base = 0.0;
            let mut new = 0.0;
            for _ in 0..64 {
                let coefficients = (0..=degree)
                    .map(|_| rng.next_f64() * 2.0 - 1.0)
                    .collect::<Vec<_>>();
                let row = (0..width)
                    .map(|j| {
                        let t = (j as f64 - (width / 2) as f64) / 8.0;
                        let signal = coefficients.iter().rev().fold(0.0, |y, c| y * t + c);
                        signal + (rng.next_f64() * 2.0 - 1.0) * 0.5
                    })
                    .collect::<Vec<_>>();
                let target = coefficients[1] / 8.0;
                let old = (row[width / 2 + 1] - row[width / 2 - 1]) / 2.0;
                let output = fitted.apply(&row).unwrap();
                assert!(output.faults.iter().all(|&f| f == 0));
                base += (old - target).powi(2);
                new += (output.values[width / 2] - target).powi(2);
            }
            base_total += base;
            new_total += new;
            receipt.push(serde_json::json!({"task":task["id"],"seed":seed,"rows_each":64,"reference_mse":base/64.0,"savgol_mse":new/64.0}));
        }
        assert!(
            new_total < base_total * 0.7,
            "task {degree}: savgol {new_total} reference {base_total}"
        );
    }
    assert_eq!(receipt.len(), 30);
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("savgol-utility-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("receipt.json"),
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();
}

fn run(command: &mut Command) -> Output {
    let output = command
        .env("LC_ALL", "C")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("required compiler/interpreter missing");
    assert!(
        output.status.success(),
        "{command:?}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn cpp(root: &Path, file: &str, bin: &str, compiler: &str, standard: &str, extra: &[&str]) {
    let mut cmd = Command::new(compiler);
    cmd.args(["-O2", "-ffp-contract=off", "-fno-fast-math", standard])
        .args(extra)
        .arg(root.join(file))
        .arg("-o")
        .arg(root.join(bin));
    run(&mut cmd);
}

#[test]
fn three_language_boundary_values_and_faults_are_bit_equal() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("savgol-boundary-v2");
    fs::create_dir_all(&root).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/math_boundary/v2/savgol.json")).unwrap();
    let mut rs = format!("{}\nfn main(){{\n", emit::RUST_HELPER);
    let mut py = format!("import math\n{}\n", emit::PYTHON_HELPER);
    let mut cc=format!("#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <cstdint>\n#include <stdexcept>\n#include <vector>\n#include <utility>\n#include <iostream>\n#include <iomanip>\n{}\nint main(){{\n",emit::CPP_HELPER);
    let mut expected = String::new();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let row = case["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| f64::from_bits(u64::from_str_radix(b.as_str().unwrap(), 16).unwrap()))
            .collect::<Vec<_>>();
        let state = config(
            case["w"].as_u64().unwrap() as u8,
            case["p"].as_u64().unwrap() as u8,
            case["r"].as_u64().unwrap() as u8,
        )
        .fit(row.len())
        .unwrap();
        let out = state.apply(&row).unwrap();
        let pinned = case["expected_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| u64::from_str_radix(b.as_str().unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let pinned_faults = case["expected_faults"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(bits(&out.values), pinned);
        assert_eq!(out.faults, pinned_faults);
        for (v, f) in out.values.iter().zip(out.faults) {
            expected.push_str(&format!("{:016x} {f:08x}\n", v.to_bits()));
            count += 1;
        }
        let stage = FittedStage::Savgol(state);
        let values = bits(&row);
        let rsrow = values
            .iter()
            .map(|b| format!("f64::from_bits({b})"))
            .collect::<Vec<_>>()
            .join(",");
        let pyrow = values
            .iter()
            .map(|b| format!("struct.unpack('<d',struct.pack('<Q',{b}))[0]"))
            .collect::<Vec<_>>()
            .join(",");
        let ccrow = values
            .iter()
            .map(|b| format!("row_vector_v2_bits({b}ULL)"))
            .collect::<Vec<_>>()
            .join(",");
        rs.push_str(&format!("{{let(v,f)=row_vector_v2_values(&[vec![{rsrow}]],{});for (a,b) in v[0].iter().zip(&f[0]){{println!(\"{{:016x}} {{:08x}}\",a.to_bits(),b);}}}}\n",emit::arguments(&stage,Language::Rust)));
        py.push_str(&format!("v,f=row_vector_v2_values([[{pyrow}]],{})\nfor a,b in zip(v[0],f[0]): print('%016x %08x' % (struct.unpack('<Q',struct.pack('<d',a))[0],b))\n",emit::arguments(&stage,Language::Python)));
        cc.push_str(&format!("{{auto out=row_vector_v2_values({{{{{ccrow}}}}},{});for(std::size_t i=0;i<out.first[0].size();++i){{std::uint64_t b;double a=out.first[0][i];std::memcpy(&b,&a,8);std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<b<<' '<<std::setw(8)<<out.second[0][i]<<'\\n';}}}}\n",emit::arguments(&stage,Language::Cpp17)));
    }
    assert!(count >= 150);
    rs.push_str("}\n");
    cc.push_str("}\n");
    fs::write(root.join("boundary.rs"), rs).unwrap();
    fs::write(root.join("boundary.py"), py).unwrap();
    fs::write(root.join("boundary.cpp"), cc).unwrap();
    fs::write(root.join("expected.txt"), &expected).unwrap();
    run(Command::new("rustc")
        .arg("-O")
        .arg(root.join("boundary.rs"))
        .arg("-o")
        .arg(root.join("rust")));
    assert_eq!(
        run(&mut Command::new(root.join("rust"))).stdout,
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3").arg(root.join("boundary.py"))).stdout,
        expected.as_bytes()
    );
    cpp(&root, "boundary.cpp", "cpp17", "g++", "-std=c++17", &[]);
    assert_eq!(
        run(&mut Command::new(root.join("cpp17"))).stdout,
        expected.as_bytes()
    );
    if std::env::var_os("JOPE_ROW_CPP_QUALIFY").is_some() {
        for (name, compiler, std, extra) in [
            ("clang17", "clang++", "-std=c++17", vec![]),
            ("cpp20", "g++", "-std=c++20", vec![]),
            (
                "sanitizer",
                "g++",
                "-std=c++17",
                vec![
                    "-O1",
                    "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer",
                    "-no-pie",
                ],
            ),
        ] {
            cpp(&root, "boundary.cpp", name, compiler, std, &extra);
            assert_eq!(
                run(&mut Command::new(root.join(name))).stdout,
                expected.as_bytes()
            );
        }
    }
}

#[test]
fn gp_create_change_remove_roundtrip_and_compiled_recipe_closure() {
    let profile = ExperimentalProfile {
        savgol: true,
        prior: 0.25,
    };
    let empty = Program::new(2, 7, vec![], 0.02).unwrap();
    let grow = empty.edit(41, Edit::Grow, &profile).unwrap();
    let change = grow.edit(43, Edit::Change, &profile).unwrap();
    let drop = change.edit(47, Edit::Drop, &profile).unwrap();
    assert_eq!(drop, empty);
    assert_ne!(grow, change);
    assert!(empty
        .edit(
            41,
            Edit::Grow,
            &ExperimentalProfile {
                savgol: false,
                prior: 0.25
            }
        )
        .is_err());
    assert_eq!(
        Program::from_bytes(&grow.to_bytes().unwrap()).unwrap(),
        grow
    );
    let mut costs = Vec::new();
    for language in [Language::Rust, Language::Python, Language::Cpp17] {
        let (_, base) = empty.emit_recipe_counted(language).unwrap();
        let (_, added) = grow.emit_recipe_counted(language).unwrap();
        costs.push(serde_json::json!({"language":format!("{language:?}"),"base":base,"with_filter":added,"bytes_delta":added.bytes as i64-base.bytes as i64,"code_tokens_delta":added.code_tokens as i64-base.code_tokens as i64}));
    }
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("savgol-recipes-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("costs.json"),
        serde_json::to_string_pretty(&costs).unwrap(),
    )
    .unwrap();
    let (x, y) = training(7);
    let train = x
        .iter()
        .zip(&y)
        .map(|(r, y)| {
            format!(
                "{},{}\n",
                r.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                y
            )
        })
        .collect::<String>();
    let test = x
        .iter()
        .map(|r| {
            format!(
                "{}\n",
                r.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<String>();
    fs::write(root.join("train.csv"), train).unwrap();
    fs::write(root.join("test.csv"), test).unwrap();
    for (index, p) in [grow, change, drop].iter().enumerate() {
        let model = p.fit(&x, &y).unwrap();
        let expected = bits(&model.predict(&x).unwrap().0);
        for (language, name) in [
            (Language::Rust, "model.rs"),
            (Language::Python, "model.py"),
            (Language::Cpp17, "model.cpp"),
        ] {
            fs::write(root.join(name), p.emit_recipe(language).unwrap()).unwrap();
        }
        run(Command::new("rustc")
            .arg("-O")
            .arg(root.join("model.rs"))
            .arg("-o")
            .arg(root.join("rs-model")));
        let mut recipe = fs::read_to_string(root.join("model.cpp")).unwrap();
        recipe.push_str("#include <fstream>\n#include <iomanip>\nstatic double test_bits(std::uint64_t b){double v;std::memcpy(&v,&b,8);return v;}\nint main(int argc,char** argv){if(argc!=4)return 1;std::vector<double> x=");
        recipe.push_str(&format!(
            "{{{}}}",
            x.iter()
                .flatten()
                .map(|v| format!("test_bits({}ULL)", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        ));
        recipe.push_str(&format!(
            ";std::vector<double> y={{{}}};",
            y.iter()
                .map(|v| format!("test_bits({}ULL)", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        ));
        recipe.push_str("model_handle* h=nullptr;if(model_fit_numeric(x.data(),x.size(),24,7,y.data(),y.size(),&h)!=0)return 2;std::vector<double> out(24);if(model_predict_numeric_batch(h,x.data(),x.size(),24,7,nullptr,0,out.data(),out.size())!=0)return 3;std::ofstream f(argv[3]);for(double v:out)f<<std::setprecision(17)<<v<<'\\n';model_unload(h);}");
        fs::write(root.join("model.cpp"), recipe).unwrap();
        cpp(&root, "model.cpp", "cpp-model", "g++", "-std=c++17", &[]);
        for (name, mut command) in [
            ("rust", Command::new(root.join("rs-model"))),
            ("python", {
                let mut c = Command::new("python3");
                c.arg(root.join("model.py"));
                c
            }),
            ("cpp17", Command::new(root.join("cpp-model"))),
        ] {
            let output = root.join(format!("{index}-{name}.csv"));
            run(command
                .arg(root.join("train.csv"))
                .arg(root.join("test.csv"))
                .arg(&output));
            let actual = fs::read_to_string(output)
                .unwrap()
                .lines()
                .map(|s| s.parse::<f64>().unwrap().to_bits())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{name} recipe {index}");
        }
        fs::copy(
            root.join("model.cpp"),
            root.join(format!("{index}-recipe.cpp")),
        )
        .unwrap();
        let mut fitted = model.emit_fitted_cpp17().unwrap();
        fitted.push_str("#include <iostream>\n#include <iomanip>\nint main(){std::vector<std::vector<double>> x=");
        fitted.push_str(&format!(
            "{{{}}}",
            x.iter()
                .map(|r| format!(
                    "{{{}}}",
                    r.iter()
                        .map(|v| format!("row_vector_v2_bits({}ULL)", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                ))
                .collect::<Vec<_>>()
                .join(",")
        ));
        fitted.push_str(";for(double v:row_vector_predict(x)){std::uint64_t bits;std::memcpy(&bits,&v,8);std::cout<<std::hex<<bits<<'\\n';}}");
        fs::write(root.join("fitted.cpp"), fitted).unwrap();
        cpp(&root, "fitted.cpp", "fitted", "g++", "-std=c++17", &[]);
        let actual = String::from_utf8(run(&mut Command::new(root.join("fitted"))).stdout)
            .unwrap()
            .lines()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "fitted {index}");
        fs::copy(
            root.join("fitted.cpp"),
            root.join(format!("{index}-fitted.cpp")),
        )
        .unwrap();
        if std::env::var_os("JOPE_ROW_CPP_QUALIFY").is_some() {
            for (name, compiler, standard, extra) in [
                ("clang17", "clang++", "-std=c++17", vec![]),
                ("cpp20", "g++", "-std=c++20", vec![]),
                (
                    "sanitizer",
                    "g++",
                    "-std=c++17",
                    vec![
                        "-O1",
                        "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer",
                        "-no-pie",
                    ],
                ),
            ] {
                let recipe_name = format!("{index}-{name}-recipe");
                cpp(&root, "model.cpp", &recipe_name, compiler, standard, &extra);
                let output = root.join(format!("{recipe_name}.csv"));
                run(Command::new(root.join(&recipe_name))
                    .arg(root.join("train.csv"))
                    .arg(root.join("test.csv"))
                    .arg(&output));
                let actual = fs::read_to_string(output)
                    .unwrap()
                    .lines()
                    .map(|s| s.parse::<f64>().unwrap().to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{name} recipe {index}");
                let fitted_name = format!("{index}-{name}-fitted");
                cpp(
                    &root,
                    "fitted.cpp",
                    &fitted_name,
                    compiler,
                    standard,
                    &extra,
                );
                let actual =
                    String::from_utf8(run(&mut Command::new(root.join(&fitted_name))).stdout)
                        .unwrap()
                        .lines()
                        .map(|s| u64::from_str_radix(s, 16).unwrap())
                        .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{name} fitted {index}");
            }
        }
    }
}
