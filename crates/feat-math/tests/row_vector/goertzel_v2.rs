//! B13 goertzel_bank (wire 61) as a v2 FeatureWithinRow row stage beside B12's savgol.
use feat_math::kernel::goertzel::{self, Config};
use feat_math::kernel::ordered_axis::Axis;
use feat_math::kernel::row_vector::{
    emit::{self, RowRecipeLanguage},
    Edit, ExperimentalProfile, FittedStage, GoertzelProfile, Model, Program, RowStage,
    GOERTZEL_FEATURE, REQUIRED_FEATURES,
};
use feat_math::kernel::savgol;
use feat_math::kernel::wire_v2::{VersionError, VersionedModel, VersionedProgram};
use feat_math::math_expr::FaultSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn bank(frequencies: &[f64]) -> Config {
    Config::new(2, 61, Axis::FeatureWithinRow, frequencies).unwrap()
}
fn smoothing() -> savgol::Config {
    savgol::Config::new(2, 58, Axis::FeatureWithinRow, 5, 2, 0, "shrink").unwrap()
}
fn program(width: usize, stages: Vec<RowStage>) -> Program {
    Program::new(2, width, stages, 0.02).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn hex_words(values: &serde_json::Value) -> Vec<u64> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|b| u64::from_str_radix(b.as_str().unwrap(), 16).unwrap())
        .collect()
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
fn bank_contract_refuses_versions_axes_banks_and_legacy_ingress() {
    for (version, wire, axis, frequencies) in [
        (1, 61, Axis::FeatureWithinRow, vec![0.1]),
        (2, 58, Axis::FeatureWithinRow, vec![0.1]),
        (2, 61, Axis::SampleTime, vec![0.1]),
        (2, 61, Axis::FeatureWithinRow, vec![]),
        (2, 61, Axis::FeatureWithinRow, vec![0.1; 33]),
        (2, 61, Axis::FeatureWithinRow, vec![0.2, f64::NAN]),
        (2, 61, Axis::FeatureWithinRow, vec![-0.25]),
        (2, 61, Axis::FeatureWithinRow, vec![0.5000000000000001]),
    ] {
        assert!(Config::new(version, wire, axis, &frequencies).is_err());
    }
    assert!(bank(&[0.1]).fit(feat_math::MAX_WIDTH + 1).is_err());
    let old = feat_math::GenomeIr::parse("(pipe X (goertzel_bank f=0.1) (ridge lambda_frac=0.02))");
    if let Ok(old) = old {
        assert!(feat_math::fit(&old, &vec![vec![1.0; 5]; 4], &[1.0; 4]).is_err());
        assert!(feat_math::transpile::emit(&old, &Default::default()).is_err());
    }
    // The bank sets the stage width; a following stage sees the bank length.
    let p = program(
        9,
        vec![
            RowStage::Goertzel(bank(&[0.0, 0.1, 0.25])),
            RowStage::Savgol(smoothing()),
        ],
    );
    let model = p.fit(&training(9).0, &training(9).1).unwrap();
    let widths: Vec<(usize, usize)> = model
        .fitted_stages()
        .iter()
        .map(|s| (s.input_width(), s.output_width()))
        .collect();
    assert_eq!(widths, [(9, 3), (3, 3)]);
}

#[test]
fn feature_declarations_bind_exactly_to_stage_content() {
    let savgol_only = program(7, vec![RowStage::Savgol(smoothing())]);
    let with_bank = program(
        7,
        vec![
            RowStage::Savgol(smoothing()),
            RowStage::Goertzel(bank(&[0.0, 0.2])),
        ],
    );
    assert_eq!(savgol_only.required_features(), REQUIRED_FEATURES);
    assert_eq!(
        with_bank.required_features(),
        REQUIRED_FEATURES | GOERTZEL_FEATURE
    );
    for (p, declared) in [(&savgol_only, 32_u64), (&with_bank, 160)] {
        let bytes = p.to_bytes().unwrap();
        let at = bytes.len()
            - bytes
                .windows(8)
                .rev()
                .position(|w| w == declared.to_le_bytes())
                .map(|i| i + 8)
                .unwrap();
        assert_eq!(Program::from_bytes(&bytes).unwrap(), *p);
        // Swapping the declaration for the other legal value is refused, as is any other value.
        for forged in [32_u64 + 160 - declared, 0, 64, 32 | 64] {
            let mut bad = bytes.clone();
            bad[at..at + 8].copy_from_slice(&forged.to_le_bytes());
            assert!(matches!(
                Program::from_bytes(&bad),
                Err(VersionError::UnsupportedVersion(_))
            ));
        }
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
    }
    // Model envelopes carry the program's feature set and the bank's own identities.
    let (x, y) = training(7);
    let model = with_bank.fit(&x, &y).unwrap();
    let json = model.to_json().unwrap();
    let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(wire["required_features"], 160);
    assert_eq!(
        wire["emitter_feature_identity"],
        "feat-math.feature-row.recipes.v2+wire58+wire61"
    );
    let restored = Model::from_json(&json).unwrap();
    assert_eq!(restored.to_json().unwrap(), json);
    assert_eq!(model.predict(&x).unwrap(), restored.predict(&x).unwrap());
    assert!(matches!(
        VersionedModel::from_json(&json).unwrap(),
        VersionedModel::RowVectorV2(_)
    ));
    let savgol_json = savgol_only.fit(&x, &y).unwrap().to_json().unwrap();
    let savgol_wire: serde_json::Value = serde_json::from_str(&savgol_json).unwrap();
    for (field, bank_text, savgol_text) in [
        ("required_features", &wire, &savgol_wire),
        ("grammar_sha256", &wire, &savgol_wire),
        ("emitter_feature_identity", &wire, &savgol_wire),
    ] {
        // A bank model cannot claim the savgol-only identities, and vice versa.
        let mut forged = bank_text.clone();
        forged[field] = savgol_text[field].clone();
        assert!(Model::from_json(&forged.to_string()).is_err(), "{field}");
        let mut forged = savgol_text.clone();
        forged[field] = bank_text[field].clone();
        assert!(Model::from_json(&forged.to_string()).is_err(), "{field}");
    }
    // Tampered bank constants and forged frequencies refuse at model ingress.
    let mut forged = wire.clone();
    forged["stages"][1]["state"]["bins"][0]["sine_bits"] = serde_json::json!(1);
    assert!(Model::from_json(&forged.to_string()).is_err());
    let mut forged = wire.clone();
    forged["stages"][1]["state"]["config"]["frequency_bits"][0] =
        serde_json::json!((-0.0_f64).to_bits());
    assert!(Model::from_json(&forged.to_string()).is_err());
    // Holding out rows cannot change the fitted model.
    let before = model.to_json().unwrap();
    let _ = model.predict(&vec![vec![f64::NAN; 7]; 3]).unwrap();
    assert_eq!(before, model.to_json().unwrap());
    assert_eq!(before, with_bank.fit(&x, &y).unwrap().to_json().unwrap());
}

/// Savgol-only and stage-free programs keep B12's exact bytes: these digests were taken at the
/// B12 head (289a4cae) and match after the bank was added.
#[test]
fn savgol_only_artifacts_keep_their_b12_bytes() {
    use sha2::{Digest, Sha256};
    let sha = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let pins: &[(&str, &str)] = &[
        (
            "empty.canonical",
            "09687d18dcdc5b5a5691532254f2bdb83a0411a5fe67ffe00639fddef0b2a57f",
        ),
        (
            "empty.fitted_cpp17",
            "3844b4fbe7c771a95f1d4321c130681d4e27f54df0cbd677460bfe85bdb1881f",
        ),
        (
            "empty.model_json",
            "64a187e253d871d64c0e2735f24347ee719deba20376f9896bdf799ff73c4bcd",
        ),
        (
            "empty.program",
            "9fdc3311c001a99a28bf13adeaffab9522a918857dd630cdfdaff4c2ea5c57e5",
        ),
        (
            "empty.recipe.cpp17",
            "fede71b4af3bb915b8219f3cd63f03b35505eaf96e6b8cce5a180004db348c22",
        ),
        (
            "empty.recipe.python",
            "b34e9f7c56f5c1ea178b20b8b5df6fad44b3bbf2116023b77acfa659cc18289b",
        ),
        (
            "empty.recipe.rust",
            "7d28963f155739424eb35dc0fc64614225aa1b7949a6fcc3bcd2105f28bbf0b9",
        ),
        (
            "savgol5.canonical",
            "4596f1fe27cd00cf0c468f4f090d8a248097c5ce666f0523c08ae9673967a81f",
        ),
        (
            "savgol5.fitted_cpp17",
            "792b57e69978a2ff2d38935afc219535a0776a82bd1b088545c8bf0b083649e7",
        ),
        (
            "savgol5.model_json",
            "d5b2d764ff72ecc058e6a8a7927872d4fb0180e1d05b5b752e43d769f8abed99",
        ),
        (
            "savgol5.program",
            "0af38ee99c7ba8309d1fd765ce02cd8b163b6a35eec843eb11047fe527d70790",
        ),
        (
            "savgol5.recipe.cpp17",
            "c7385381ac77a7fd1e396dadfa3c35eb90a1b2152139a0ce8d492fa9346772ef",
        ),
        (
            "savgol5.recipe.python",
            "66fb375493bc03acee63824d09bf087da4a3b2548b4d09c1ed16ad7ba2c615d5",
        ),
        (
            "savgol5.recipe.rust",
            "cd9d1c6b4d099df99a87e07ae483fa7e9d92641153d3e2fdac4987ede68cd1b4",
        ),
    ];
    let mut actual = std::collections::BTreeMap::new();
    for (name, p) in [
        ("savgol5", program(7, vec![RowStage::Savgol(smoothing())])),
        ("empty", program(7, vec![])),
    ] {
        actual.insert(format!("{name}.program"), sha(&p.to_bytes().unwrap()));
        actual.insert(
            format!("{name}.canonical"),
            sha(p.canonical().unwrap().as_bytes()),
        );
        for (language, label) in [
            (RowRecipeLanguage::Rust, "rust"),
            (RowRecipeLanguage::Python, "python"),
            (RowRecipeLanguage::Cpp17, "cpp17"),
        ] {
            actual.insert(
                format!("{name}.recipe.{label}"),
                sha(p.emit_recipe(language).unwrap().as_bytes()),
            );
        }
        let (x, y) = training(7);
        let model = p.fit(&x, &y).unwrap();
        actual.insert(
            format!("{name}.model_json"),
            sha(model.to_json().unwrap().as_bytes()),
        );
        actual.insert(
            format!("{name}.fitted_cpp17"),
            sha(model.emit_fitted_cpp17().unwrap().as_bytes()),
        );
    }
    assert_eq!(actual.len(), pins.len());
    for (name, digest) in pins {
        assert_eq!(actual[*name], *digest, "{name}");
    }
}

#[test]
fn rows_reset_independently_and_bank_artifacts_pass_the_shipped_text_lint() {
    let p = program(
        8,
        vec![
            RowStage::Goertzel(bank(&[0.0, 0.125, 0.3, 0.5])),
            RowStage::Savgol(smoothing()),
        ],
    );
    let (x, y) = training(8);
    let model = p.fit(&x, &y).unwrap();
    // Each row restarts the recurrence: predicting rows together, apart or permuted agrees.
    let (all, faults) = model.predict(&x).unwrap();
    let (head, rest) = x.split_at(9);
    let mut parts = model.predict(&head.to_vec()).unwrap().0;
    parts.extend(model.predict(&rest.to_vec()).unwrap().0);
    assert_eq!(bits(&parts), bits(&all));
    let reversed: Vec<Vec<f64>> = x.iter().rev().cloned().collect();
    let mut back = model.predict(&reversed).unwrap().0;
    back.reverse();
    assert_eq!(bits(&back), bits(&all));
    assert!(faults
        .iter()
        .all(|&f| f & FaultSet::NONFINITE_SANITIZED.bits() == 0));
    // Shipped bank artifacts contain no banned vocabulary.
    for language in [
        RowRecipeLanguage::Rust,
        RowRecipeLanguage::Python,
        RowRecipeLanguage::Cpp17,
    ] {
        let recipe = p.emit_recipe(language).unwrap();
        assert!(feat_math::transpile::lint_shipped_text(&recipe).is_empty());
    }
    let fitted = model.emit_fitted_cpp17().unwrap();
    assert!(feat_math::transpile::lint_shipped_text(&fitted).is_empty());
}

#[test]
fn unknown_wires_stay_unsupported_and_savgol_ingress_is_unchanged() {
    let mut bytes = program(4, vec![RowStage::Goertzel(bank(&[0.1]))])
        .to_bytes()
        .unwrap();
    let at = bytes
        .windows(11)
        .position(|w| w == b"\"wire\":\"61\"")
        .unwrap();
    bytes[at + 8] = b'6';
    bytes[at + 9] = b'2';
    assert!(matches!(
        Program::from_bytes(&bytes),
        Err(VersionError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        VersionedProgram::from_bytes(&bytes),
        Err(VersionError::UnsupportedVersion(_))
    ));
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

#[test]
fn bank_prior_is_bounded_and_removal_keeps_parsing_available() {
    let p = Program::new(2, 9, vec![], 0.02).unwrap();
    let profile = GoertzelProfile {
        goertzel: true,
        prior: 0.25,
    };
    let created = (0..1024)
        .filter(|&seed| {
            !p.mutate_goertzel(seed, &profile)
                .unwrap()
                .stages()
                .is_empty()
        })
        .count();
    assert!(
        (200..=320).contains(&created),
        "prior gate created {created}/1024"
    );
    let disabled = GoertzelProfile {
        goertzel: false,
        prior: 0.25,
    };
    assert!((0..32).all(|seed| p.mutate_goertzel(seed, &disabled).unwrap() == p));
    for prior in [0.0, -0.1, 0.251, f64::NAN, f64::INFINITY] {
        assert!(GoertzelProfile {
            goertzel: true,
            prior
        }
        .validate()
        .is_err());
    }
    let cap = feat_math::kernel::capabilities::row_vector_v2_capability(61).unwrap();
    assert!(cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Experimental));
    assert!(!cap
        .class
        .admits(feat_math::kernel::capabilities::Tier::Baseline));
    // Every seeded edit keeps a valid, lossless program within the bank bounds.
    let mut current = p.clone();
    for seed in 0..64_u64 {
        let arm = [Edit::Grow, Edit::Change, Edit::Drop][(seed % 3) as usize];
        if let Ok(next) = current.edit_goertzel(seed, arm, &profile) {
            assert_eq!(
                Program::from_bytes(&next.to_bytes().unwrap()).unwrap(),
                next
            );
            for stage in next.stages() {
                if let RowStage::Goertzel(config) = stage {
                    assert!((1..=goertzel::MAX_BINS).contains(&config.bins()));
                }
            }
            current = next;
        }
    }
}

#[test]
fn six_spectrum_tasks_five_seeds_equal_budget_tone_power_ablation() {
    // Fixture qualification only: no product corpus, sealed labels, campaign worker, promotion
    // or broad search-lift claim. Both pipelines fit the same ridge head on the same 96 training
    // rows per task/seed and are scored on the same 64 held-out rows; the reference is the
    // existing integer-bin DFT magnitude stage.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/goertzel_utility.json"
    ))
    .unwrap();
    let width = fixture["width"].as_u64().unwrap() as usize;
    let mut receipt = Vec::new();
    for task in fixture["tasks"].as_array().unwrap() {
        let target = task["frequency"].as_f64().unwrap();
        let nuisance: Vec<f64> = task["nuisance"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let bins = task["reference_bins"].as_u64().unwrap();
        let reference = feat_math::GenomeIr::parse(&format!(
            "(pipe X (fft_mag k={bins}) (ridge lambda_frac=0.02))"
        ))
        .unwrap();
        let candidate = program(width, vec![RowStage::Goertzel(bank(&[target]))]);
        let (mut base_total, mut new_total) = (0.0, 0.0);
        for seed in fixture["seeds"].as_array().unwrap() {
            let seed = seed.as_u64().unwrap();
            let mut rng = feat_math::Pcg32::new(seed, 0x676f65727a656c);
            let mut draw_rows = |count: usize| -> (Vec<Vec<f64>>, Vec<f64>) {
                (0..count)
                    .map(|_| {
                        let amplitude = 0.5 + rng.next_f64() * 1.5;
                        let phase = rng.next_f64() * std::f64::consts::TAU;
                        let tones: Vec<(f64, f64, f64)> = nuisance
                            .iter()
                            .map(|&f| {
                                (
                                    f,
                                    rng.next_f64() * 1.5,
                                    rng.next_f64() * std::f64::consts::TAU,
                                )
                            })
                            .collect();
                        let row = (0..width)
                            .map(|n| {
                                let n = n as f64;
                                let mut value =
                                    amplitude * (std::f64::consts::TAU * target * n + phase).cos();
                                for &(f, a, p) in &tones {
                                    value += a * (std::f64::consts::TAU * f * n + p).cos();
                                }
                                value + (rng.next_f64() * 2.0 - 1.0) * 0.1
                            })
                            .collect::<Vec<_>>();
                        (row, amplitude * amplitude)
                    })
                    .unzip()
            };
            let (train_x, train_y) = draw_rows(96);
            let (test_x, test_y) = draw_rows(64);
            let mse = |predictions: &[f64]| {
                predictions
                    .iter()
                    .zip(&test_y)
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f64>()
                    / test_y.len() as f64
            };
            let old = feat_math::fit(&reference, &train_x, &train_y)
                .unwrap()
                .predict(&test_x)
                .unwrap();
            let new = candidate
                .fit(&train_x, &train_y)
                .unwrap()
                .predict(&test_x)
                .unwrap();
            assert!(new.1.iter().all(|&f| f == 0));
            let (base, bank_mse) = (mse(&old), mse(&new.0));
            base_total += base;
            new_total += bank_mse;
            receipt.push(serde_json::json!({"task":task["id"],"seed":seed,"train_rows":96,"heldout_rows":64,"reference_mse":base,"goertzel_mse":bank_mse}));
        }
        // No predetermined gain: the bank must not be worse than the reference on any task, and
        // the receipt records the actual ratios (about 0.12..0.69 at this fixture).
        assert!(
            new_total.is_finite() && new_total < base_total,
            "task {}: goertzel {new_total} reference {base_total}",
            task["id"]
        );
    }
    assert_eq!(receipt.len(), 30);
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("goertzel-utility-v2");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("receipt.json"),
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
