//! Opt-in training recipes fit their own state; fitted inference is a separate route.
use feat_math::kernel::recipe_v2::{emit_cpp17, PROFILE};
use feat_math::transpile::EmitOptions;
use feat_math::{fit_predict, GenomeIr};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    process::{Command, Stdio},
};

const EXPRESSIONS: &[&str] = &[
    "sqrt(c0) + sin(c1) - cos(c2)",
    "tanh(c0) * sigmoid(c1) + log1p(c2)",
    "exp(c0) / (c1*c1+1) - log(c2)",
    "asinh(c0) + atan(c1) * erf(c2)",
    "sign(c0) * abs(c1) + floor(c1) - round(c2)",
    "min(c0,c1) - max(c1,c2) + step(c0,c2) + clip(c0,c1,c2)",
    "-(abs(c0)^c1) + abs(c2)^c0",
    "row_sum-row_l1+row_l2+row_mean*row_std+row_min-row_max",
    "x + c0 / c1",
];

fn visit(
    expr: &feat_math::math_expr::Expr,
    functions: &mut BTreeSet<u8>,
    variables: &mut BTreeSet<String>,
    binary: &mut BTreeSet<u8>,
    unary: &mut bool,
) {
    use feat_math::math_expr::Expr;
    match expr {
        Expr::Const(_) => {}
        Expr::Var(var) => {
            variables.insert(format!("{var:?}"));
        }
        Expr::Unary(_, child) => {
            *unary = true;
            visit(child, functions, variables, binary, unary);
        }
        Expr::Binary(op, a, b) => {
            binary.insert(*op as u8);
            visit(a, functions, variables, binary, unary);
            visit(b, functions, variables, binary, unary);
        }
        Expr::Call(func, args) => {
            functions.insert(*func as u8);
            for arg in args {
                visit(arg, functions, variables, binary, unary);
            }
        }
    }
}

fn rows() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let train: Vec<Vec<f64>> = (0..32)
        .map(|i| {
            vec![
                i as f64 / 8.0 - 2.0,
                (i % 7) as f64 / 3.0 - 1.0,
                (i % 11) as f64 / 5.0 - 1.0,
            ]
        })
        .collect();
    let targets = train
        .iter()
        .map(|r| 0.7 * r[0] - 0.2 * r[1] + 0.1 * r[2])
        .collect();
    let mut test = vec![
        vec![0.125, -0.75, 1.5],
        vec![-2.0, -1.0, -0.5],
        vec![0.0, -0.0, 0.0],
        vec![f64::from_bits(1), 1e-12, 1.0],
    ];
    for value in [1e-12_f64.to_bits() - 1, 1e-12_f64.to_bits() + 1] {
        test.push(vec![f64::from_bits(value), 0.5, 1.0]);
    }
    (train, targets, test)
}

fn execute(program: &str, compiler: &str, standard: &str, optimization: &str, sanitize: bool) {
    let original = GenomeIr::parse(program).unwrap();
    let canonical = original.canonical();
    let genome = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(canonical, genome.canonical());
    let bytes = feat_math::kernel::Program::import_legacy(&genome)
        .to_bytes()
        .unwrap();
    assert_eq!(
        bytes,
        feat_math::kernel::Program::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap()
    );
    let recipe = emit_cpp17(&genome, &EmitOptions::default()).unwrap();
    let (mut train, targets, mut test) = rows();
    if matches!(genome, GenomeIr::Pipeline(_)) {
        // Pipeline entry points sanitize features; raw symbolic has a different input contract.
        for (index, value) in [
            f64::from_bits(0x7ff8_1234_5678_9abc),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]
        .into_iter()
        .enumerate()
        {
            train[index][index] = value;
            let mut row = vec![0.75, -1.25, 1.5];
            row[index] = value;
            test.push(row);
        }
    }
    let expected: Vec<u64> = fit_predict(&genome, &train, &targets, &test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("recipe.cpp");
    let binary = scratch.path().join("recipe");
    fs::write(&source, recipe.source + DRIVER).unwrap();
    let mut build = Command::new(compiler);
    build.args([
        standard,
        optimization,
        "-ffp-contract=off",
        "-fno-fast-math",
    ]);
    if sanitize {
        build.args([
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-no-pie",
        ]);
    }
    let output = build.arg(&source).arg("-o").arg(&binary).output().unwrap();
    assert!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut input = format!("{} {} 3\n", train.len(), test.len());
    for value in train
        .into_iter()
        .flatten()
        .chain(targets)
        .chain(test.into_iter().flatten())
    {
        input.push_str(&format!("{:016x}\n", value.to_bits()));
    }
    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{program} {compiler} {standard}: status {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Vec<u64> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "{program}: {compiler} {standard}");
}

// Input arrives after compilation, so libm calls cannot be constant-folded by the compiler.
const DRIVER: &str = r#"
#include <iostream>
#include <iomanip>
static std::vector<double> read_values(std::size_t n) {
    std::vector<double> values; values.reserve(n);
    for(std::size_t i=0;i<n;++i){std::uint64_t word;if(!(std::cin>>std::hex>>word))throw std::runtime_error("input bits");values.push_back(f64_from_bits(word));}
    return values;
}
int main() {
    std::size_t n,m,width; if(!(std::cin>>n>>m>>width))return 30;
    const auto train=read_values(n*width),y=read_values(n),test=read_values(m*width);
    model_handle* fitted=nullptr;
    if(model_fit_numeric(train.data(),train.size(),n,width,y.data(),y.size(),&fitted)!=0 || fitted==nullptr)return 31;
    std::vector<double> out(m);
    if(model_predict_numeric_batch(fitted,test.data(),test.size(),m,width,nullptr,0,out.data(),out.size())!=0)return 32;
    if(model_predict_numeric_batch(fitted,test.data(),test.size(),m,0,nullptr,0,out.data(),out.size())!=1)return 33;
    if(model_predict_numeric_batch(fitted,test.data(),test.size(),m,width+1,nullptr,0,out.data(),out.size())!=1)return 34;
    if(model_predict_numeric_batch(fitted,test.data(),test.size()-1,m,width,nullptr,0,out.data(),out.size())!=1)return 35;
    if(model_predict_numeric_batch(fitted,nullptr,0,0,width,nullptr,0,nullptr,0)!=0)return 36;
    model_handle* rejected=fitted;
    if(model_fit_numeric(train.data(),train.size(),n,0,y.data(),y.size(),&rejected)!=1 || rejected!=nullptr)return 37;
    for(std::uint64_t bad:{0x7ff8000000000001ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL}) {
        auto invalid=y;invalid[0]=f64_from_bits(bad);rejected=fitted;
        if(model_fit_numeric(train.data(),train.size(),n,width,invalid.data(),invalid.size(),&rejected)!=1 || rejected!=nullptr)return 38;
    }
    bool fit_refused=false,predict_refused=false,ragged_refused=false;
    try{(void)Model::fit({{}},{0.0});}catch(const std::invalid_argument&){fit_refused=true;}
    try{(void)fitted->model.predict({{}});}catch(const std::invalid_argument&){predict_refused=true;}
    try{(void)Model::fit({{1.0},{1.0,2.0}},{0.0,1.0});}catch(const std::invalid_argument&){ragged_refused=true;}
    if(!fit_refused || !predict_refused || !ragged_refused)return 39;
    for(double value:out){std::uint64_t word;std::memcpy(&word,&value,8);std::cout<<std::hex<<std::setw(16)<<std::setfill('0')<<word<<'\n';}
    model_unload(fitted);
}
"#;

#[test]
fn expression_recipes_are_opt_in_and_preserve_default_refusals() {
    for source in [
        "(sym raw \"c0+c1\")",
        "(sym affine \"c0+c1\")",
        "(pipe X (map expr=\"x+c0\") (ridge))",
        "(pipe X (derive expr=\"c0+c1\") (ridge))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let refusal = feat_math::transpile::emit_cpp17(&genome, &EmitOptions::default())
            .unwrap_err()
            .to_string();
        let recipe = emit_cpp17(&genome, &EmitOptions::default()).unwrap();
        assert_eq!(recipe.profile, PROFILE);
        assert!(recipe
            .source
            .starts_with("// feat-math.cpp17-expression-recipe.v2\n"));
        assert_eq!(
            refusal,
            feat_math::transpile::emit_cpp17(&genome, &EmitOptions::default())
                .unwrap_err()
                .to_string()
        );
        assert_eq!(
            genome.canonical(),
            GenomeIr::parse(&genome.canonical()).unwrap().canonical()
        );
        let x = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, -1.0]];
        assert_eq!(
            fit_predict(&genome, &x, &[1.0, 3.0, 1.0], &x)
                .unwrap()
                .len(),
            3
        );
    }
}

#[test]
fn expression_training_recipes_execute_all_historical_scalar_forms() {
    let (mut functions, mut variables, mut binary, mut unary) =
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new(), false);
    for expr in EXPRESSIONS {
        let parsed = feat_math::math_expr::MathExpr::parse(expr).unwrap();
        visit(
            parsed.ast(),
            &mut functions,
            &mut variables,
            &mut binary,
            &mut unary,
        );
        for program in [
            format!("(sym raw \"{expr}\")"),
            format!("(sym affine \"{expr}\")"),
            format!("(pipe X (map expr=\"{expr}\") (ridge))"),
            format!("(pipe X (derive expr=\"{expr}\") (ridge))"),
        ] {
            execute(&program, "g++", "-std=c++17", "-O2", false);
        }
    }
    assert_eq!(functions, (0..19).collect());
    assert_eq!(binary, (0..5).collect());
    assert!(unary);
    assert_eq!(
        variables,
        [
            "X", "Col(0)", "Col(1)", "Col(2)", "RowMean", "RowSum", "RowL1", "RowL2", "RowMin",
            "RowMax", "RowStd"
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );
}

#[test]
fn expression_training_recipes_preserve_fault_fallback_and_stage_order() {
    for expr in [
        "min(clip(c0, (-1)^0.5, (-1)^0.5), 1)",
        "sign(clip(c0, (-1)^0.5, (-1)^0.5))",
        "c9+1",
        "c1/c0+c2",
    ] {
        for program in [
            format!("(sym raw \"{expr}\")"),
            format!("(pipe X (map expr=\"{expr}\") (ridge))"),
            format!("(pipe X (derive expr=\"{expr}\") (ridge))"),
        ] {
            execute(&program, "g++", "-std=c++17", "-O2", false);
        }
    }
    for program in [
        "(pipe X (standardize) (map expr=\"x+c0\" append=true) (derive expr=\"row_sum+c4\" append=true) (map expr=\"x/3\") (ridge))",
        "(pipe X (derive expr=\"c0+c1\" append=true) (derive expr=\"c3*c2\" append=true) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=30))",
    ] { execute(program, "g++", "-std=c++17", "-O2", false); }
}

#[test]
fn expression_training_recipes_pass_compiler_and_sanitizer_matrix() {
    for program in [
        "(sym affine \"c0+c1\")",
        "(pipe X (map expr=\"x+c0\") (ridge))",
        "(pipe X (derive expr=\"c0+c1\") (ridge))",
    ] {
        for (compiler, standard, optimization, sanitize) in [
            ("g++", "-std=c++17", "-O2", false),
            ("g++", "-std=c++17", "-O3", false),
            ("g++", "-std=c++20", "-O2", false),
            ("clang++", "-std=c++17", "-O2", false),
            ("clang++", "-std=c++20", "-O2", false),
            ("g++", "-std=c++17", "-O2", true),
        ] {
            execute(program, compiler, standard, optimization, sanitize);
        }
    }
}

#[test]
fn expression_training_recipes_refuse_other_heads_targets_and_portable_math() {
    for value in [5.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mismatched = GenomeIr::Symbolic {
            source: "c0".into(),
            expr: feat_math::math_expr::MathExpr::from_ast(feat_math::math_expr::Expr::Const(
                value,
            ))
            .unwrap(),
            affine: false,
        };
        assert!(matches!(
            emit_cpp17(&mismatched, &EmitOptions::default()),
            Err(feat_math::kernel::Error::Syntax(_) | feat_math::kernel::Error::Unsupported(_))
        ));
    }
    for (program, reason) in [
        ("(pipe X (map expr=\"x\") (gbm))", "head"),
        ("(pipe X (target kind=log1p) (ridge))", "target transforms"),
        ("(portfolio (member (pipe X (ridge))) (blend))", "portfolio"),
    ] {
        let Err(feat_math::kernel::Error::Unsupported(message)) =
            emit_cpp17(&GenomeIr::parse(program).unwrap(), &EmitOptions::default())
        else {
            panic!("{program}: expected Unsupported");
        };
        assert!(message.contains(reason), "{program}: {message}");
    }
    let opts = EmitOptions {
        math_kernel: feat_math::transpile::MathKernel::Portable,
        ..EmitOptions::default()
    };
    assert!(matches!(
        emit_cpp17(&GenomeIr::parse("(sym raw \"c0\")").unwrap(), &opts),
        Err(feat_math::kernel::Error::Unsupported(_))
    ));
}
