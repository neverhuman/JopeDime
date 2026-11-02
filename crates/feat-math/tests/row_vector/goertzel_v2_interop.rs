use super::*;

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

fn qualifier_builds() -> Vec<(&'static str, &'static str, &'static str, Vec<&'static str>)> {
    vec![
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
    ]
}

#[test]
fn three_language_boundary_values_and_faults_are_bit_equal() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("goertzel-boundary-v2");
    fs::create_dir_all(&root).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/math_boundary/v2/goertzel.json")).unwrap();
    let mut rs = format!("{}\nfn main(){{\n", emit::RUST_GOERTZEL_HELPER);
    let mut py = format!("import math\n{}\n", emit::PYTHON_GOERTZEL_HELPER);
    let mut cc = format!("#include <algorithm>\n#include <cmath>\n#include <cstring>\n#include <cstdint>\n#include <stdexcept>\n#include <vector>\n#include <utility>\n#include <iostream>\n#include <iomanip>\n{}\nint main(){{\n", emit::CPP_GOERTZEL_HELPER);
    let mut expected = String::new();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let row = hex_words(&case["bits"])
            .into_iter()
            .map(f64::from_bits)
            .collect::<Vec<_>>();
        let frequencies = hex_words(&case["frequency_bits"])
            .into_iter()
            .map(f64::from_bits)
            .collect::<Vec<_>>();
        let state = bank(&frequencies).fit(row.len()).unwrap();
        let out = state.apply(&row).unwrap();
        assert_eq!(
            bits(&out.values),
            hex_words(&case["expected_bits"]),
            "{}",
            case["name"]
        );
        let pinned_faults = case["expected_faults"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(out.faults, pinned_faults, "{}", case["name"]);
        for (v, f) in out.values.iter().zip(&out.faults) {
            expected.push_str(&format!("{:016x} {f:08x}\n", v.to_bits()));
            count += 1;
        }
        let stage = FittedStage::Goertzel(state);
        let words = bits(&row);
        let rsrow = words
            .iter()
            .map(|b| format!("f64::from_bits({b})"))
            .collect::<Vec<_>>()
            .join(",");
        let pyrow = words
            .iter()
            .map(|b| format!("struct.unpack('<d',struct.pack('<Q',{b}))[0]"))
            .collect::<Vec<_>>()
            .join(",");
        let ccrow = words
            .iter()
            .map(|b| format!("row_vector_goertzel_v2_bits({b}ULL)"))
            .collect::<Vec<_>>()
            .join(",");
        rs.push_str(&format!("{{let row:Vec<f64>=vec![{rsrow}];let(v,f)=row_vector_goertzel_v2_values(&[row],{});for (a,b) in v[0].iter().zip(&f[0]){{println!(\"{{:016x}} {{:08x}}\",a.to_bits(),b);}}}}\n",emit::arguments(&stage,RowRecipeLanguage::Rust)));
        py.push_str(&format!("v,f=row_vector_goertzel_v2_values([[{pyrow}]],{})\nfor a,b in zip(v[0],f[0]): print('%016x %08x' % (struct.unpack('<Q',struct.pack('<d',a))[0],b))\n",emit::arguments(&stage,RowRecipeLanguage::Python)));
        cc.push_str(&format!("{{std::vector<std::vector<double>> x{{std::vector<double>{{{ccrow}}}}};auto out=row_vector_goertzel_v2_values(x,{});for(std::size_t i=0;i<out.first[0].size();++i){{std::uint64_t b;double a=out.first[0][i];std::memcpy(&b,&a,8);std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<b<<' '<<std::setw(8)<<out.second[0][i]<<'\\n';}}}}\n",emit::arguments(&stage,RowRecipeLanguage::Cpp17)));
    }
    assert!(count >= 150, "boundary corpus has {count} values");
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
        for (name, compiler, standard, extra) in qualifier_builds() {
            cpp(&root, "boundary.cpp", name, compiler, standard, &extra);
            assert_eq!(
                run(&mut Command::new(root.join(name))).stdout,
                expected.as_bytes()
            );
        }
    }
}

#[test]
fn gp_create_change_remove_roundtrip_and_compiled_recipe_closure() {
    let profile = GoertzelProfile {
        goertzel: true,
        prior: 0.25,
    };
    let base = Program::new(2, 7, vec![RowStage::Savgol(smoothing())], 0.02).unwrap();
    let grow = base.edit_goertzel(41, Edit::Grow, &profile).unwrap();
    let change = grow.edit_goertzel(43, Edit::Change, &profile).unwrap();
    let drop = change.edit_goertzel(47, Edit::Drop, &profile).unwrap();
    assert_eq!(drop, base);
    assert_ne!(grow, change);
    assert_eq!(
        grow.required_features(),
        REQUIRED_FEATURES | GOERTZEL_FEATURE
    );
    assert_eq!(drop.required_features(), REQUIRED_FEATURES);
    // Disabling generation keeps removal and parsing available.
    let disabled = GoertzelProfile {
        goertzel: false,
        prior: 0.25,
    };
    assert!(base.edit_goertzel(41, Edit::Grow, &disabled).is_err());
    assert!(change.edit_goertzel(47, Edit::Drop, &disabled).is_ok());
    assert_eq!(
        Program::from_bytes(&grow.to_bytes().unwrap()).unwrap(),
        grow
    );
    // The savgol profile edits only savgol stages, and never a bank.
    let savgol_profile = ExperimentalProfile {
        savgol: true,
        prior: 0.25,
    };
    let bank_only = program(6, vec![RowStage::Goertzel(bank(&[0.0, 0.3]))]);
    assert!(bank_only.edit(5, Edit::Change, &savgol_profile).is_err());
    assert!(bank_only.edit(5, Edit::Drop, &savgol_profile).is_err());
    assert!(bank_only.edit_goertzel(5, Edit::Drop, &disabled).is_ok());
    let mut costs = Vec::new();
    for language in [
        RowRecipeLanguage::Rust,
        RowRecipeLanguage::Python,
        RowRecipeLanguage::Cpp17,
    ] {
        let (_, without) = base.emit_recipe_counted(language).unwrap();
        let (_, with) = grow.emit_recipe_counted(language).unwrap();
        costs.push(serde_json::json!({"language":format!("{language:?}"),"base":without,"with_bank":with,"bytes_delta":with.bytes as i64-without.bytes as i64,"code_tokens_delta":with.code_tokens as i64-without.code_tokens as i64}));
    }
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("goertzel-recipes-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("costs.json"),
        serde_json::to_string_pretty(&costs).unwrap(),
    )
    .unwrap();
    let (mut x, y) = training(7);
    // Protected inputs survive every edit, including huge finite values.
    for (i, row) in x.iter_mut().enumerate() {
        row[0] = ((i % 7) as f64 - 3.0) * 1e12;
    }
    let csv = |rows: &[Vec<f64>], labels: Option<&[f64]>| -> String {
        rows.iter()
            .enumerate()
            .map(|(i, r)| {
                let mut line = r
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                if let Some(labels) = labels {
                    line.push_str(&format!(",{}", labels[i]));
                }
                line.push('\n');
                line
            })
            .collect()
    };
    fs::write(root.join("train.csv"), csv(&x, Some(&y))).unwrap();
    fs::write(root.join("test.csv"), csv(&x, None)).unwrap();
    let mixed = program(
        7,
        vec![
            RowStage::Goertzel(bank(&[0.0, 0.0625, 0.125, 0.25, 0.4375, 0.5])),
            RowStage::Savgol(smoothing()),
        ],
    );
    let bank_first = program(7, vec![RowStage::Goertzel(bank(&[0.17, 0.33]))]);
    for (index, p) in [grow, change, mixed, bank_first].iter().enumerate() {
        let model = p.fit(&x, &y).unwrap();
        let expected = bits(&model.predict(&x).unwrap().0);
        for (language, name) in [
            (RowRecipeLanguage::Rust, "model.rs"),
            (RowRecipeLanguage::Python, "model.py"),
            (RowRecipeLanguage::Cpp17, "model.cpp"),
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
        if std::env::var_os("JOPE_ROW_CPP_QUALIFY").is_some() {
            for (name, compiler, standard, extra) in qualifier_builds() {
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
