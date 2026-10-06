//! Repeated emission and real compilation of bounded scalar GP trees.
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::math_expr::{BinOp, UnOp, Var};
use feat_math::Pcg32;
use std::{fs, path::Path, process::Command};

const LANGUAGES: [ScalarLanguage; 3] = [
    ScalarLanguage::Python,
    ScalarLanguage::Rust,
    ScalarLanguage::Cpp17,
];
const SEED: u64 = 0x4750_454d_4954;
const WIDTH: usize = 3;
const LAMBDA: f64 = 0.02;
const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

fn function(id: u16, argument: ScalarNode) -> ScalarNode {
    let mut args = vec![argument];
    match id {
        7 | 8 | 10 => args.push(ScalarNode::Var(Var::Col(1))),
        9 => args.extend([ScalarNode::Const(-1.0), ScalarNode::Const(1.0)]),
        _ => {}
    }
    ScalarNode::Call(ScalarFunction::from_versioned_id(2, id).unwrap(), args)
}

fn random_tree(rng: &mut Pcg32, depth: usize) -> ScalarNode {
    let choice = rng.next_u32();
    if depth == 0 || choice.is_multiple_of(5) {
        return if choice.is_multiple_of(2) {
            ScalarNode::Var(Var::Col((choice as usize / 5) % WIDTH))
        } else {
            ScalarNode::Const([-0.0, 0.5, -2.0, 1e-300, 1e200][(choice as usize / 5) % 5])
        };
    }
    if choice.is_multiple_of(3) {
        let op =
            [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow][(choice as usize / 3) % 5];
        ScalarNode::Binary(
            op,
            Box::new(random_tree(rng, depth - 1)),
            Box::new(random_tree(rng, depth - 1)),
        )
    } else {
        function((choice % 23) as u16, random_tree(rng, depth - 1))
    }
}

fn corpus() -> Vec<ScalarExpressionProgram> {
    let mut nodes = Vec::new();
    for value in [
        0.0,
        -0.0,
        f64::from_bits(1),
        1e-300,
        1e200,
        f64::MAX,
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        nodes.push(ScalarNode::Const(value));
    }
    for id in 0..23 {
        nodes.push(function(id, ScalarNode::Var(Var::Col(0))));
    }
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
        nodes.push(ScalarNode::Binary(
            op,
            Box::new(ScalarNode::Var(Var::Col(0))),
            Box::new(ScalarNode::Var(Var::Col(1))),
        ));
    }
    for var in [
        Var::RowSum,
        Var::RowMean,
        Var::RowL1,
        Var::RowL2,
        Var::RowMin,
        Var::RowMax,
        Var::RowStd,
    ] {
        nodes.push(ScalarNode::Var(var));
    }
    let mut deep = ScalarNode::Var(Var::Col(0));
    for _ in 0..15 {
        deep = ScalarNode::Unary(UnOp::Neg, Box::new(deep));
    }
    nodes.push(deep);
    nodes.push(function(21, ScalarNode::Var(Var::X)));
    let mut rng = Pcg32::new(SEED, 54);
    for _ in 0..32 {
        nodes.push(random_tree(&mut rng, 4));
    }
    // Retain constant-folding regressions alongside seeded case 58: Min/Max
    // must select the native second operand for both orders of signed-zero ties.
    for id in [7, 8] {
        for zero in [0.0, -0.0] {
            nodes.push(function(id, ScalarNode::Const(zero)));
        }
    }
    // Clip's bound sorting calls the same kernels; retain both zero orders.
    for (a, b) in [(0.0, -0.0), (-0.0, 0.0)] {
        nodes.push(ScalarNode::Call(
            ScalarFunction::from_versioned_id(2, 9).unwrap(),
            vec![
                ScalarNode::Var(Var::Col(0)),
                ScalarNode::Const(a),
                ScalarNode::Const(b),
            ],
        ));
    }
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| {
            ScalarExpressionProgram::new(node)
                .unwrap()
                .with_numerical_policy(if i.is_multiple_of(2) {
                    NumericalPolicy::LegacyV1
                } else {
                    NumericalPolicy::StableV2
                })
        })
        .collect()
}

fn carrier(case: usize) -> ScalarCarrier {
    // The explicit x case and alternating random trees exercise elementwise binding.
    if case.is_multiple_of(3) || case == 45 {
        ScalarCarrier::Elementwise
    } else {
        ScalarCarrier::Row
    }
}

fn emit(case: usize, program: &ScalarExpressionProgram, language: ScalarLanguage) -> String {
    program
        .emit_recipe(language, carrier(case), WIDTH, LAMBDA)
        .unwrap()
}

fn run(command: &mut Command) -> String {
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("{command:?}: {e}"));
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
#[ignore = "subprocess entry point, explicitly executed twice by repeated_emission"]
fn emit_child() {
    let directory = std::env::var_os("JOPE_GP_EMIT_CHILD_DIR").expect("child destination");
    let directory = Path::new(&directory);
    fs::create_dir_all(directory).unwrap();
    for (case, program) in corpus().iter().enumerate() {
        for (lang, language) in LANGUAGES.into_iter().enumerate() {
            let first = emit(case, program, language);
            assert_eq!(
                first,
                emit(case, program, language),
                "case {case}, {language:?}"
            );
            fs::write(directory.join(format!("{case}-{lang}")), first).unwrap();
        }
    }
}

fn evidence_directory(label: &str) -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gp-emitter-determinism");
    fs::create_dir_all(&root).unwrap();
    let path = tempfile::Builder::new()
        .prefix(label)
        .tempdir_in(root)
        .unwrap()
        .keep();
    println!("GP emitter evidence: {}", path.display());
    path
}

#[test]
fn repeated_emission() {
    assert_eq!(corpus().len(), 84);
    let directory = evidence_directory("processes-");
    for child in ["first", "second"] {
        let output = run(Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "emit_child", "--ignored", "--nocapture"])
            .env("JOPE_GP_EMIT_CHILD_DIR", directory.join(child)));
        assert!(
            output.contains("1 passed"),
            "child test did not execute: {output}"
        );
    }
    for (case, program) in corpus().iter().enumerate() {
        for (lang, language) in LANGUAGES.into_iter().enumerate() {
            let first = emit(case, program, language);
            assert_eq!(first, emit(case, program, language));
            for child in ["first", "second"] {
                assert_eq!(
                    first.as_bytes(),
                    fs::read(directory.join(child).join(format!("{case}-{lang}"))).unwrap(),
                    "case {case}, {language:?}, {child}"
                );
            }
        }
    }
}

fn rows() -> Vec<Vec<f64>> {
    let mut rows = vec![vec![0.25, -0.75, 1.5], vec![-0.0, 0.0, -0.0]];
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -1e200,
        1e-300,
        f64::from_bits(1),
        -2.0,
        -1.0,
        -0.0,
        0.0,
        1e-12,
        f64::from_bits(1e-12_f64.to_bits() - 1),
        f64::from_bits(1e-12_f64.to_bits() + 1),
    ] {
        rows.push(vec![value, value, value]);
        rows.push(vec![value, 0.5, -0.5]);
    }
    rows
}

fn matrix(rows: &[Vec<f64>], language: ScalarLanguage) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            let cells = row
                .iter()
                .map(|v| match language {
                    ScalarLanguage::Rust => format!("f64::from_bits({}u64)", v.to_bits()),
                    ScalarLanguage::Python => format!("m.scalar_bits_v2({})", v.to_bits()),
                    ScalarLanguage::Cpp17 => format!("f64_from_bits({}ULL)", v.to_bits()),
                })
                .collect::<Vec<_>>()
                .join(",");
            match language {
                ScalarLanguage::Rust => format!("vec![{cells}]"),
                ScalarLanguage::Python => format!("[{cells}]"),
                ScalarLanguage::Cpp17 => format!("std::vector<double>{{{cells}}}"),
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{rows}]"),
        ScalarLanguage::Python => format!("[{rows}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<std::vector<double>>{{{rows}}}"),
    }
}

fn agrees(actual: f64, expected: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if !actual.is_finite() || !expected.is_finite() || actual == 0.0 || expected == 0.0 {
        return actual.to_bits() == expected.to_bits();
    }
    (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * actual.abs().max(expected.abs())
}

fn compile_case(case: usize, program: &ScalarExpressionProgram, directory: &Path) {
    let rows = rows();
    let train: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![i as f64 / 8.0 - 1.0, 0.25, -0.5])
        .collect();
    let labels: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let native = program
        .fit_training(&train, &labels, carrier(case), LAMBDA)
        .unwrap();
    let (features, faults) = native.transform(&rows).unwrap();
    let (predictions, prediction_faults) = native.predict(&rows).unwrap();
    assert_eq!(faults, prediction_faults);
    let expected: Vec<_> = features
        .iter()
        .flatten()
        .chain(&predictions)
        .copied()
        .collect();
    fs::create_dir_all(directory).unwrap();
    fs::write(directory.join("program.bin"), program.to_bytes().unwrap()).unwrap();
    fs::write(directory.join("native.json"), serde_json::to_vec(&serde_json::json!({"values_bits":expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"faults":faults})).unwrap()).unwrap();
    for language in LANGUAGES {
        let source = emit(case, program, language);
        let test_rows = matrix(&rows, language);
        let training = matrix(&train, language);
        let y = matrix(std::slice::from_ref(&labels), language);
        let observations = match language {
            ScalarLanguage::Rust => {
                fs::write(directory.join("model.rs"), &source).unwrap();
                let driver = format!("\n#[test] fn gp_probe(){{let rows={test_rows};let train={training};let y={y};let model=Model::fit(&train,&y[0]);let (f,q)=scalar_stage1_values(&rows);for v in f.into_iter().flatten().chain(model.predict(&rows)){{println!(\"V {{:016x}}\",v.to_bits());}}for v in q{{println!(\"Q {{}}\",v);}}}}\n");
                fs::write(directory.join("probe.rs"), source + &driver).unwrap();
                run(Command::new("rustc")
                    .args(["--test", "--edition=2021", "-C", "opt-level=1"])
                    .arg(directory.join("probe.rs"))
                    .arg("-o")
                    .arg(directory.join("rust-probe")));
                vec![(
                    "Rust",
                    run(Command::new(directory.join("rust-probe")).args([
                        "--exact",
                        "gp_probe",
                        "--nocapture",
                    ])),
                )]
            }
            ScalarLanguage::Python => {
                fs::write(directory.join("model.py"), source).unwrap();
                run(Command::new("python3")
                    .args(["-m", "py_compile"])
                    .arg(directory.join("model.py")));
                let driver = format!("import model as m\nimport struct\nrows={test_rows}\nmodel=m.Model().fit({training}, {y}[0])\nf,q=m.scalar_stage1_values(rows)\nfor v in [v for row in f for v in row]+model.predict(rows): print('V %016x'%struct.unpack('<Q',struct.pack('<d',v))[0])\nfor v in q: print('Q %d'%v)\n");
                fs::write(directory.join("probe.py"), driver).unwrap();
                vec![(
                    "Python",
                    run(Command::new("python3").arg(directory.join("probe.py"))),
                )]
            }
            ScalarLanguage::Cpp17 => {
                fs::write(directory.join("model.cpp"), source).unwrap();
                let driver = format!("#include \"model.cpp\"\n#include <cstdio>\nint main(){{const auto rows={test_rows};const auto y={y};const auto model=Model::fit({training},y[0]);const auto f=scalar_stage1_values(rows);auto print=[](double v){{std::uint64_t bits;std::memcpy(&bits,&v,8);std::printf(\"V %016llx\\n\",static_cast<unsigned long long>(bits));}};for(const auto& row:f.first)for(double v:row)print(v);for(double v:model.predict(rows))print(v);for(auto v:f.second)std::printf(\"Q %u\\n\",v);}}\n");
                fs::write(directory.join("probe.cpp"), driver).unwrap();
                ["g++", "clang++"]
                    .into_iter()
                    .map(|compiler| {
                        let executable = directory.join(format!("{compiler}-probe"));
                        run(Command::new(compiler)
                            .args([
                                "-std=c++17",
                                "-Wall",
                                "-Werror",
                                "-O1",
                                "-fno-fast-math",
                                "-ffp-contract=off",
                            ])
                            .arg(directory.join("probe.cpp"))
                            .arg("-o")
                            .arg(&executable));
                        (compiler, run(&mut Command::new(executable)))
                    })
                    .collect()
            }
        };
        for (implementation, observed) in observations {
            fs::write(directory.join(format!("{implementation}.out")), &observed).unwrap();
            let values: Vec<_> = observed
                .lines()
                .filter_map(|line| line.strip_prefix("V "))
                .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).unwrap()))
                .collect();
            let actual_faults: Vec<u32> = observed
                .lines()
                .filter_map(|line| line.strip_prefix("Q "))
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual_faults, faults, "case {case} {implementation} faults");
            assert_eq!(
                values.len(),
                expected.len(),
                "case {case} {implementation} value count"
            );
            for (row, (actual, expected)) in values.into_iter().zip(&expected).enumerate() {
                assert!(
                    agrees(actual, *expected),
                    "case {case} {implementation} value {row}: {actual:?} != {expected:?}"
                );
            }
        }
    }
}

fn differential(shard: usize) {
    let directory = evidence_directory(&format!("shard-{shard}-"));
    for (case, program) in corpus()
        .iter()
        .enumerate()
        .filter(|(case, _)| case % 8 == shard)
    {
        compile_case(case, program, &directory.join(case.to_string()));
    }
}

macro_rules! shards {
    ($($name:ident: $shard:literal),*) => {$(#[test] fn $name() { differential($shard); })*};
}
shards!(compile_0:0, compile_1:1, compile_2:2, compile_3:3,
    compile_4:4, compile_5:5, compile_6:6, compile_7:7);

#[test]
fn comparison_rejects_signed_zero_nonfinite_and_material_changes() {
    assert!(!agrees(-0.0, 0.0));
    assert!(!agrees(f64::NAN, 0.0));
    assert!(!agrees(f64::INFINITY, f64::NEG_INFINITY));
    assert!(!agrees(1.0, 1.001));
    assert!(agrees(1.0, f64::from_bits(1.0_f64.to_bits() + 1)));
}

// Lane A's strict conversion repro: the opt-in Symbolic route does not call
// max_num. Keep the historical/default emitter separate and byte-frozen.
#[test]
fn symbolic_cpp17_compiles_with_both_warning_strict_compilers() {
    use feat_math::{kernel::expression_recipe::emit_cpp17, transpile::EmitOptions, GenomeIr};
    let genome = GenomeIr::parse("(sym raw \"c0-(c1-c2)\")").unwrap();
    let options = EmitOptions::default();
    let source = emit_cpp17(&genome, &options).unwrap().source;
    assert_eq!(source, emit_cpp17(&genome, &options).unwrap().source);
    let directory = evidence_directory("symbolic-");
    fs::write(directory.join("model.cpp"), source).unwrap();
    fs::write(directory.join("probe.cpp"),
        "#include \"model.cpp\"\nint main(){const std::vector<std::vector<double>> x={{0.25,0.75,1.5}};const auto m=Model::fit(x,{1.0});const auto p=m.predict(x);return p.size()==1 && p[0]==1.0 ? 0 : 1;}\n").unwrap();
    for compiler in ["g++", "clang++"] {
        let executable = directory.join(format!("{compiler}-probe"));
        run(Command::new(compiler)
            .args([
                "-std=c++17",
                "-Wall",
                "-Werror",
                "-O1",
                "-ffp-contract=off",
                "-fno-fast-math",
            ])
            .arg(directory.join("probe.cpp"))
            .arg("-o")
            .arg(&executable));
        run(&mut Command::new(&executable));
    }
}
