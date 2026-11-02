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
    for language in [
        RowRecipeLanguage::Rust,
        RowRecipeLanguage::Python,
        RowRecipeLanguage::Cpp17,
    ] {
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
    let (mut x, y) = training(7);
    // GP Drop must retain the v2 input clamp at fit and prediction, even
    // when no filter remains. These rows reproduce reviewer R1.
    for (i, row) in x.iter_mut().enumerate() {
        row[0] = ((i % 7) as f64 - 3.0) * 1e12;
    }
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
