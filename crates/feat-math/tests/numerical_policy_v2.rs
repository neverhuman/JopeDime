//! First B07 numerical prerequisite. Helper parity does not admit complete
//! programs, fitted models, recipes or shipped generation under policy v2.
use feat_math::kernel::numerical_policy::{asinh_v2, reduce_v2, Reduction};
use feat_math::kernel::numerical_policy::{emit_helpers, NumericalLanguage, KERNEL_ID, POLICY_ID};
use feat_math::math_expr::{FaultSet, MathExpr};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn asinh_repairs_the_huge_finite_input_only_in_the_new_kernel() {
    let (value, faults) = asinh_v2(1.0e200);
    let expected = 461.2101657793691_f64;
    assert!(value.to_bits().abs_diff(expected.to_bits()) <= 2);
    assert_eq!(faults, FaultSet::EMPTY);
    let old = MathExpr::parse("asinh(c0)").unwrap();
    assert_eq!(old.eval_row(&[1.0e200]).unwrap().to_bits(), 0);
    for input in [0.0_f64, -0.0, f64::from_bits(1), -f64::from_bits(1)] {
        assert_eq!(asinh_v2(input), (input, FaultSet::EMPTY));
        assert_eq!(asinh_v2(input).0.to_bits(), input.to_bits());
    }
}

#[test]
fn neumaier_preserves_the_cancellation_term_in_input_order() {
    let row = [1.0e16, 1.0, -1.0e16];
    assert_eq!(reduce_v2(Reduction::Sum, &row), (1.0, FaultSet::EMPTY));
    assert_eq!(
        reduce_v2(Reduction::Mean, &row),
        (1.0 / 3.0, FaultSet::EMPTY)
    );
    assert_eq!(
        MathExpr::parse("row_sum")
            .unwrap()
            .eval_row(&row)
            .unwrap()
            .to_bits(),
        0
    );
}

#[test]
fn finite_extreme_means_clamp_at_non_power_of_two_widths() {
    for count in 2..=32 {
        for sign in [1.0_f64, -1.0] {
            let row = vec![sign * f64::MAX; count];
            let (value, faults) = reduce_v2(Reduction::Mean, &row);
            assert_eq!(
                value.to_bits(),
                (sign * 1.0e12).to_bits(),
                "width {count}, sign {sign}"
            );
            assert_eq!(
                faults,
                FaultSet::MAGNITUDE_CLAMPED,
                "width {count}, sign {sign}"
            );
        }
    }
}

#[test]
fn overflowing_mean_preserves_exact_cancellation_and_small_residuals() {
    let maximum = f64::MAX;
    let previous = f64::from_bits(maximum.to_bits() - 1);
    let gap = maximum - previous;
    for sign in [1.0_f64, -1.0] {
        for residual in [0.0, 1.0, f64::MIN_POSITIVE, f64::from_bits(7)] {
            // The first five binary-rational values sum to exactly zero.
            let mut row = vec![maximum, maximum, -previous, -previous, -2.0 * gap];
            if residual != 0.0 {
                row.push(residual);
            }
            for value in &mut row {
                *value *= sign;
            }
            let expected = if residual == 0.0 {
                0.0
            } else {
                (sign * residual) / row.len() as f64
            };
            let (value, faults) = reduce_v2(Reduction::Mean, &row);
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "sign {sign}, residual {residual:e}"
            );
            assert_eq!(
                faults,
                FaultSet::EMPTY,
                "sign {sign}, residual {residual:e}"
            );
        }
    }
}

#[test]
fn scaled_l2_saturates_huge_finite_norms_and_retains_tiny_norms() {
    assert_eq!(
        reduce_v2(Reduction::L2, &[3.0e200, 4.0e200]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
    let (value, faults) = reduce_v2(Reduction::L2, &[3.0e-200, 4.0e-200]);
    assert!(value.to_bits().abs_diff(5.0e-200_f64.to_bits()) <= 2);
    assert_eq!(faults, FaultSet::EMPTY);
    assert_eq!(
        reduce_v2(Reduction::L2, &[f64::MAX, f64::MAX]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
}

#[test]
fn nonfinite_reads_have_a_positive_zero_domain_fallback() {
    for input in [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8000000000042),
        f64::from_bits(0xfff8000000000042),
    ] {
        let result = asinh_v2(input);
        assert_eq!(result.0.to_bits(), 0);
        assert_eq!(result.1, FaultSet::NONFINITE_INPUT);
        for reduction in [
            Reduction::Sum,
            Reduction::L1,
            Reduction::Mean,
            Reduction::L2,
        ] {
            let result = reduce_v2(reduction, &[1.0, input, 2.0]);
            assert_eq!(result.0.to_bits(), 0);
            assert_eq!(result.1, FaultSet::NONFINITE_INPUT);
        }
    }
}

#[test]
fn empty_reductions_keep_their_defined_identity_or_fault() {
    for reduction in [Reduction::Sum, Reduction::L1, Reduction::L2] {
        assert_eq!(reduce_v2(reduction, &[]), (0.0, FaultSet::EMPTY));
    }
    assert_eq!(
        reduce_v2(Reduction::Mean, &[]),
        (0.0, FaultSet::EMPTY_ROW_REDUCTION)
    );
    assert_eq!(
        reduce_v2(Reduction::L1, &[-3.0e200, 4.0e200]),
        (1.0e12, FaultSet::MAGNITUDE_CLAMPED)
    );
}

#[derive(Deserialize)]
struct Corpus {
    asinh: Vec<ScalarCase>,
    rows: Vec<RowCase>,
    mean_cancellation: Vec<MeanCase>,
}

#[derive(Deserialize)]
struct ScalarCase {
    input_bits: String,
    oracle_bits: String,
    fault_bits: u32,
}

#[derive(Deserialize)]
struct Expected {
    oracle_bits: String,
    fault_bits: u32,
}

#[derive(Deserialize)]
struct RowCase {
    input_bits: Vec<String>,
    expected: BTreeMap<String, Expected>,
}

#[derive(Deserialize)]
struct MeanCase {
    input_bits: Vec<String>,
    oracle_bits: String,
    fault_bits: u32,
}

fn corpus() -> Corpus {
    serde_json::from_str(include_str!(
        "fixtures/math_boundary/v2_numerical_policy.json"
    ))
    .unwrap()
}

fn value(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
}

fn reductions() -> [(Reduction, &'static str); 4] {
    [
        (Reduction::Sum, "sum"),
        (Reduction::L1, "l1"),
        (Reduction::Mean, "mean"),
        (Reduction::L2, "l2"),
    ]
}

fn check_oracle(actual: f64, expected: f64, context: &str) {
    assert!(actual.is_finite(), "{context}: nonfinite output");
    if expected == 0.0 {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}: zero bits");
    } else if expected.is_normal() {
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "{context}: sign"
        );
        assert!(
            actual.abs().to_bits().abs_diff(expected.abs().to_bits()) <= 2,
            "{context}: actual={actual:?}, oracle={expected:?}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= f64::from_bits(1),
            "{context}: subnormal bound"
        );
    }
}

#[test]
fn asinh_matches_the_independent_high_precision_boundary_corpus() {
    let cases = corpus().asinh;
    assert_eq!(cases.len(), 1380);
    for case in cases {
        let (actual, faults) = asinh_v2(value(&case.input_bits));
        check_oracle(actual, value(&case.oracle_bits), &case.input_bits);
        assert_eq!(faults.bits(), case.fault_bits, "input {}", case.input_bits);
    }
}

#[test]
fn stable_reductions_match_the_independent_boundary_oracle() {
    let corpus = corpus();
    assert_eq!(corpus.rows.len(), 72);
    for case in corpus.rows {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        for (reduction, name) in reductions() {
            let (actual, faults) = reduce_v2(reduction, &row);
            let expected = &case.expected[name];
            check_oracle(
                actual,
                value(&expected.oracle_bits),
                &format!("{name} {row:?}"),
            );
            assert_eq!(faults.bits(), expected.fault_bits, "{name} {row:?}");
        }
    }
    assert_eq!(corpus.mean_cancellation.len(), 8);
    for case in corpus.mean_cancellation {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        let (actual, faults) = reduce_v2(Reduction::Mean, &row);
        assert_eq!(
            actual.to_bits(),
            value(&case.oracle_bits).to_bits(),
            "mean {row:?}"
        );
        assert_eq!(faults.bits(), case.fault_bits, "mean {row:?}");
    }
}

#[test]
fn policy_identity_is_additive_and_legacy_emitter_identity_is_preserved() {
    assert_eq!(POLICY_ID, "v2");
    assert_eq!(KERNEL_ID, "feat-math.fdlibm-neumaier-lassq.v2");
    assert_eq!(feat_math::transpile::EMITTER_VERSION, "2.2.0");
    for language in [
        NumericalLanguage::Rust,
        NumericalLanguage::Python,
        NumericalLanguage::Cpp17,
    ] {
        assert!(feat_math::transpile::lint_shipped_text(&emit_helpers(language)).is_empty());
    }
}

fn run(command: &mut Command) -> Vec<u8> {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn compile_cpp(
    root: &Path,
    compiler: &str,
    standard: &str,
    optimization: &str,
    sanitizer: bool,
) -> Vec<u8> {
    let binary = root.join(format!("{compiler}-{standard}-{optimization}-{sanitizer}"));
    let mut command = Command::new(compiler);
    command.args([
        standard,
        optimization,
        "-ffp-contract=off",
        "-fno-fast-math",
    ]);
    if sanitizer {
        command.args(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]);
    }
    run(command
        .arg("-o")
        .arg(&binary)
        .arg(root.join("numerical.cpp")));
    run(Command::new(binary).arg(root.join("bits.txt")))
}

#[test]
fn emitted_helper_value_and_fault_bits_match_in_the_three_language_matrix() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("numerical_policy_v2");
    fs::create_dir_all(&root).unwrap();
    let corpus = corpus();
    let mut inputs = String::new();
    let mut expected = String::new();
    for case in corpus.asinh {
        inputs.push_str(&format!("a {}\n", case.input_bits));
        let (v, f) = asinh_v2(value(&case.input_bits));
        expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f.bits()));
    }
    for case in corpus.rows {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        for (reduction, name) in reductions() {
            inputs.push_str(&format!("{name} {}\n", case.input_bits.join(" ")));
            let (v, f) = reduce_v2(reduction, &row);
            expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f.bits()));
        }
    }
    for case in corpus.mean_cancellation {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        inputs.push_str(&format!("mean {}\n", case.input_bits.join(" ")));
        let (v, f) = reduce_v2(Reduction::Mean, &row);
        expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f.bits()));
    }
    assert_eq!(inputs.lines().count(), 1676);
    fs::write(root.join("bits.txt"), inputs).unwrap();
    let rust = emit_helpers(NumericalLanguage::Rust)
        + r#"
fn main() {
    for line in std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap().lines() {
        let mut fields=line.split_whitespace();let kind=fields.next().unwrap();
        let row:Vec<_>=fields.map(|s|f64::from_bits(u64::from_str_radix(s,16).unwrap())).collect();
        let (v,f)=if kind=="a" {numerical_asinh_v2(row[0])} else {numerical_reduce_v2(kind,&row).unwrap()};
        println!("{:016x} {:08x}",v.to_bits(),f);
    }
    assert!(numerical_reduce_v2("unknown",&[]).is_err());
}
"#;
    let python = emit_helpers(NumericalLanguage::Python)
        + r#"
import struct,sys
for line in open(sys.argv[1]):
    fields=line.split();kind=fields[0]
    row=[struct.unpack('>d',bytes.fromhex(x))[0] for x in fields[1:]]
    v,f=numerical_asinh_v2(row[0]) if kind=='a' else numerical_reduce_v2(kind,row)
    print(struct.pack('>d',v).hex(),format(f,'08x'))
try:
    numerical_reduce_v2('unknown',[])
except ValueError:
    pass
else:
    raise AssertionError('unknown reduction accepted')
"#;
    let cpp = emit_helpers(NumericalLanguage::Cpp17)
        + r#"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream input(argv[1]);std::string line;
    while(std::getline(input,line)) {
        std::istringstream fields(line);std::string kind;fields>>kind;
        std::vector<double> row;std::uint64_t bits;
        while(fields>>std::hex>>bits) {double v;std::memcpy(&v,&bits,sizeof(v));row.push_back(v);}
        auto result=kind=="a"?numerical_asinh_v2(row.at(0)):numerical_reduce_v2(kind,row);
        std::uint64_t output;std::memcpy(&output,&result.first,sizeof(output));
        std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<output<<' '<<std::setw(8)<<result.second<<'\n';
    }
    try {numerical_reduce_v2("unknown",{});return 3;} catch(const std::invalid_argument&) {}
}
"#;
    for (file, source) in [
        ("numerical.rs", rust),
        ("numerical.py", python),
        ("numerical.cpp", cpp),
    ] {
        assert!(feat_math::transpile::lint_shipped_text(&source).is_empty());
        fs::write(root.join(file), source).unwrap();
    }
    let binary = root.join("numerical-rust");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "-o"])
        .arg(&binary)
        .arg(root.join("numerical.rs")));
    assert_eq!(
        run(Command::new(binary).arg(root.join("bits.txt"))),
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .arg(root.join("numerical.py"))
            .arg(root.join("bits.txt"))),
        expected.as_bytes()
    );
    for compiler in ["g++", "clang++"] {
        for (standard, optimization) in [
            ("-std=c++17", "-O2"),
            ("-std=c++17", "-O3"),
            ("-std=c++20", "-O3"),
        ] {
            assert_eq!(
                compile_cpp(&root, compiler, standard, optimization, false),
                expected.as_bytes()
            );
        }
    }
    assert_eq!(
        compile_cpp(&root, "g++", "-std=c++17", "-O1", true),
        expected.as_bytes()
    );
}
