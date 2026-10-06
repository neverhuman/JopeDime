//! X0 conversion witnesses for nested operator order; no legacy fixture regeneration.
use feat_math::kernel::expression_recipe::emit_cpp17;
use feat_math::kernel::Program;
use feat_math::math_expr::{BinOp, Expr, MathExpr, Var};
use feat_math::transpile::{emit, EmitOptions};
use feat_math::{fit_predict, GenomeIr};
use std::{fs, path::Path, process::Command};

fn binary(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}

#[test]
fn every_nested_binary_operator_shape_preserves_tree_values_and_faults() {
    let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
    let rows = [
        [1.5, 3., 2.],
        [-0., 0., -1.],
        [1e12, -1e12, 1.],
        [1., f64::from_bits(1e-12_f64.to_bits() - 1), 2.],
        [1., 1e-12, 2.],
        [f64::from_bits(1), -2., 0.5],
    ];
    let mut shapes = 0;
    for outer in ops {
        for inner in ops {
            for left_nested in [false, true] {
                let [a, b, c] = [0, 1, 2].map(|i| Expr::Var(Var::Col(i)));
                let root = if left_nested {
                    binary(outer, binary(inner, a, b), c)
                } else {
                    binary(outer, a, binary(inner, b, c))
                };
                let original = MathExpr::from_ast(root).unwrap();
                let source = original.to_source().unwrap();
                let reparsed = MathExpr::parse(&source).unwrap();
                assert_eq!(reparsed.ast(), original.ast(), "{source}: operator tree");
                assert_eq!(
                    reparsed.to_source().unwrap(),
                    source,
                    "{source}: fixed point"
                );
                let genome = GenomeIr::Symbolic {
                    source,
                    expr: reparsed,
                    affine: false,
                };
                let program = Program::import_legacy(&genome);
                let bytes = program.to_bytes().unwrap();
                for decoded in [
                    Program::from_bytes(&bytes).unwrap(),
                    Program::parse_lossless_text(&program.to_lossless_text().unwrap()).unwrap(),
                ] {
                    assert_eq!(decoded.to_bytes().unwrap(), bytes);
                    let Program::LegacyV1(GenomeIr::Symbolic { expr, .. }) = decoded else {
                        panic!("conversion changed genome form")
                    };
                    assert_eq!(expr.ast(), original.ast());
                    for row in rows {
                        let (value, faults) = original.eval_row_traced(&row);
                        let (actual, actual_faults) = expr.compile().unwrap().eval_row_traced(&row);
                        assert_eq!(actual.to_bits(), value.to_bits());
                        assert_eq!(actual_faults, faults);
                    }
                }
                shapes += 1;
            }
        }
    }
    assert_eq!(shapes, 50);
}

fn run(command: &mut Command) {
    let output = command.output().expect("required export toolchain");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn bits(path: &Path) -> Vec<u64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| line.parse::<f64>().unwrap().to_bits())
        .collect()
}

// These hand-calculated dyadic witnesses distinguish the two parenthesizations and
// unary negation from a negative power base. Power uses the historical exponent cap of 8.
const WITNESSES: &[(&str, [f64; 2])] = &[
    ("c0-(c1-c2)", [0.5, 2.5]),
    ("(c0-c1)-c2", [-3.5, -3.5]),
    ("c0/(c1/c2)", [1., 2.25]),
    ("(c0/c1)/c2", [0.25, 0.25]),
    ("c0^(c1^c2)", [25.62890625, 25.62890625]),
    ("(c0^c1)^c2", [11.390625, 11.390625]),
    ("-(c0^c1)", [-3.375, -2.25]),
    ("(-c0)^c1", [-3.375, 2.25]),
    ("c0/(c1-c1)", [0., 0.]),
];

#[test]
fn executed_three_language_recipes_match_independent_conversion_witnesses() {
    let rows = vec![vec![1.5, 3., 2.], vec![1.5, 2., 3.]];
    let labels = [0., 1.];
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path();
    fs::write(dir.join("train.csv"), "1.5,3,2,0\n1.5,2,3,1\n").unwrap();
    fs::write(dir.join("test.csv"), "1.5,3,2\n1.5,2,3\n").unwrap();
    for (index, (source, expected)) in WITNESSES.iter().enumerate() {
        let expected = expected.map(f64::to_bits).to_vec();
        let expression = MathExpr::parse(source).unwrap();
        let canonical = expression.to_source().unwrap();
        let genome = GenomeIr::parse(&format!("(sym raw {canonical:?})")).unwrap();
        let program = Program::import_legacy(&genome);
        let Program::LegacyV1(genome) = Program::from_bytes(&program.to_bytes().unwrap()).unwrap();
        assert_eq!(
            fit_predict(&genome, &rows, &labels, &rows)
                .unwrap()
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            expected,
            "{source}: evaluator"
        );
        let artifact = emit(&genome, &EmitOptions::default()).unwrap();
        let rust = dir.join(format!("case_{index}.rs"));
        let binary = dir.join(format!("case_{index}"));
        fs::write(&rust, artifact.rust).unwrap();
        run(Command::new("rustc")
            .args(["-O", "-o"])
            .arg(&binary)
            .arg(&rust));
        let output = dir.join("rust.out");
        run(Command::new(&binary)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&output));
        assert_eq!(bits(&output), expected, "{source}: Rust recipe");
        let python = dir.join("case.py");
        fs::write(&python, artifact.python).unwrap();
        let output = dir.join("python.out");
        run(Command::new("python3")
            .arg(&python)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&output));
        assert_eq!(bits(&output), expected, "{source}: Python recipe");
        let cpp = dir.join("case.cpp");
        fs::write(
            &cpp,
            emit_cpp17(&genome, &EmitOptions::default()).unwrap().source + CPP_DRIVER,
        )
        .unwrap();
        for compiler in ["g++", "clang++"] {
            run(Command::new(compiler)
                .args([
                    "-std=c++17",
                    "-Wall",
                    "-Werror",
                    "-O2",
                    "-ffp-contract=off",
                    "-fno-fast-math",
                ])
                .arg(&cpp)
                .arg("-o")
                .arg(&binary));
            let output = dir.join("cpp.out");
            run(Command::new(&binary)
                .arg(dir.join("train.csv"))
                .arg(dir.join("test.csv"))
                .arg(&output));
            assert_eq!(bits(&output), expected, "{source}: {compiler} C++17 recipe");
        }
    }
    // Negative controls: a conversion that reassociates or moves negation cannot pass.
    for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
        assert_ne!(
            WITNESSES[a].1.map(f64::to_bits),
            WITNESSES[b].1.map(f64::to_bits)
        );
    }
}

const CPP_DRIVER: &str = r#"
#include <fstream>
#include <sstream>
#include <iomanip>
static std::vector<std::vector<double>> read_rows(const char* path) {
    std::ifstream file(path); std::string line; std::vector<std::vector<double>> rows;
    while(std::getline(file,line)){for(char& c:line)if(c==',')c=' ';std::istringstream in(line);
        std::vector<double> row;double value;while(in>>value)row.push_back(value);rows.push_back(row);}
    return rows;
}
int main(int argc,char** argv) {
    if(argc!=4){return 1;}
    auto train=read_rows(argv[1]);std::vector<double> labels;
    for(auto& row:train){labels.push_back(row.back());row.pop_back();}
    const auto model=Model::fit(train,labels);std::ofstream out(argv[3]);
    for(double value:model.predict(read_rows(argv[2])))out<<std::setprecision(17)<<value<<'\n';
    return out.good()?0:2;
}
"#;
