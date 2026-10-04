//! Guarded portable exponential helpers; complete program admission is separate.
use feat_math::kernel::stable_exponential::{
    emit_helpers, StableExponential, StableExponentialLanguage, NEG_LN_2,
};
use feat_math::math_expr::{FaultSet, MathExpr};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Deserialize)]
struct Corpus {
    entries: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    function_id: u16,
    input_bits: String,
    oracle_bits: String,
    expected_fault_bits: String,
}

fn corpus() -> Vec<Case> {
    let cases = serde_json::from_str::<Corpus>(include_str!(
        "fixtures/math_boundary/v2/stable_exponential.json"
    ))
    .unwrap()
    .entries;
    assert_eq!(cases.len(), 3048);
    cases
}

fn bits(text: &str) -> u64 {
    u64::from_str_radix(text, 16).unwrap()
}

#[test]
fn stable_exponential_ids_are_version_two_only_and_closed() {
    for (id, function) in [
        (21, StableExponential::Expm1),
        (22, StableExponential::Log1mexp),
    ] {
        assert_eq!(
            StableExponential::from_versioned_id(2, id).unwrap(),
            function
        );
        assert_eq!(function.id(), id);
        for version in [0, 1, 3, 255] {
            assert!(StableExponential::from_versioned_id(version, id).is_err());
        }
    }
    for id in [0, 18, 19, 20, 23, 34, u16::MAX] {
        assert!(StableExponential::from_versioned_id(2, id).is_err());
    }
    for (id, name) in [(21, "expm1"), (22, "log1mexp")] {
        assert!(MathExpr::parse(&format!("{name}(c0)")).is_err());
        let row = feat_math::kernel::wire_v2::FUNCTIONS
            .iter()
            .find(|row| row.id == id)
            .unwrap();
        assert_eq!(row.name, Some(name));
        assert!(!row.generatable);
        for profile in feat_math::kernel::profile::shipped_kernel2_profiles() {
            assert!(!profile.expr.unwrap().functions.contains(&name));
        }
    }
}

#[test]
fn exponential_nonfinite_and_signed_zero_guards_are_exact() {
    for function in [StableExponential::Expm1, StableExponential::Log1mexp] {
        for raw in [
            0x7ff0000000000000,
            0xfff0000000000000,
            0x7ff8000000000001,
            0xfff8000000000001,
            0x7ff0000000000001,
            0xfff0000000000001,
        ] {
            let (result, faults) = function.evaluate(f64::from_bits(raw));
            assert_eq!(result.to_bits(), 0);
            assert_eq!(faults, FaultSet::NONFINITE_INPUT);
        }
    }
    for x in [0.0_f64, -0.0, f64::from_bits(1), -f64::from_bits(1)] {
        let (result, faults) = StableExponential::Expm1.evaluate(x);
        assert_eq!(result.to_bits(), x.to_bits());
        assert_eq!(faults, FaultSet::EMPTY);
    }
    for x in [0.0_f64, -0.0, f64::from_bits(1), 1.0, f64::MAX] {
        let (result, faults) = StableExponential::Log1mexp.evaluate(x);
        assert_eq!(result.to_bits(), 0);
        assert_eq!(faults, FaultSet::LOG_DOMAIN);
    }
}

#[test]
fn expm1_argument_and_magnitude_projections_have_distinct_faults() {
    for x in [-f64::MAX, -31.0, f64::from_bits((-30.0_f64).to_bits() + 1)] {
        let (actual, faults) = StableExponential::Expm1.evaluate(x);
        assert_eq!(
            actual.to_bits(),
            feat_math::portable_math::expm1(-30.0).to_bits()
        );
        assert_eq!(faults, FaultSet::EXP_ARG_CLAMPED);
    }
    for x in [f64::from_bits(30.0_f64.to_bits() + 1), 31.0, f64::MAX] {
        let (actual, faults) = StableExponential::Expm1.evaluate(x);
        assert_eq!(actual.to_bits(), 1e12_f64.to_bits());
        assert_eq!(
            faults,
            FaultSet::EXP_ARG_CLAMPED.union(FaultSet::MAGNITUDE_CLAMPED)
        );
    }
    assert_eq!(
        StableExponential::Expm1.evaluate(30.0),
        (1e12, FaultSet::MAGNITUDE_CLAMPED)
    );
    assert_eq!(StableExponential::Expm1.evaluate(-30.0).1, FaultSet::EMPTY);
    let small = StableExponential::Expm1.evaluate(1e-10).0;
    assert!(small.to_bits().abs_diff(1.00000000005e-10_f64.to_bits()) <= 2);
    assert_ne!(small.to_bits(), (1e-10_f64.exp() - 1.0).to_bits());
}

#[test]
fn log1mexp_preserves_the_branch_subnormal_domain_and_underflow_sign() {
    assert_eq!(NEG_LN_2.to_bits(), 0xbfe62e42fefa39ef);
    for x in [
        f64::from_bits(NEG_LN_2.to_bits() + 1),
        NEG_LN_2,
        f64::from_bits(NEG_LN_2.to_bits() - 1),
    ] {
        let (actual, faults) = StableExponential::Log1mexp.evaluate(x);
        let expected = if x < NEG_LN_2 {
            feat_math::portable_math::log1p(-feat_math::portable_math::exp(x))
        } else {
            feat_math::portable_math::ln(-feat_math::portable_math::expm1(x))
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(faults, FaultSet::EMPTY);
    }
    for x in [-f64::from_bits(1), -f64::MIN_POSITIVE, -1e-300] {
        let (actual, faults) = StableExponential::Log1mexp.evaluate(x);
        assert!(actual.is_finite() && actual < -600.0);
        assert_eq!(faults, FaultSet::EMPTY);
    }
    for x in [-746.0, -1000.0, -f64::MAX] {
        let (actual, faults) = StableExponential::Log1mexp.evaluate(x);
        assert_eq!(actual.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(faults, FaultSet::EMPTY);
    }
}

fn check_oracle(function: StableExponential) {
    for case in corpus()
        .into_iter()
        .filter(|case| case.function_id == function.id())
    {
        let input = f64::from_bits(bits(&case.input_bits));
        let expected = f64::from_bits(bits(&case.oracle_bits));
        let (actual, faults) = function.evaluate(input);
        assert!(actual.is_finite());
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "{function:?} {input:?}"
        );
        if expected.is_normal() {
            assert!(
                actual.abs().to_bits().abs_diff(expected.abs().to_bits()) <= 2,
                "{function:?} {input:?}: actual={actual:?}, expected={expected:?}"
            );
        } else {
            // One minimum-subnormal absolute step; signs and guarded zeros
            // are checked exactly by the separate guard contract.
            assert!(
                (actual - expected).abs() <= f64::from_bits(1),
                "{function:?} {input:?}"
            );
        }
        assert_eq!(
            u64::from(faults.bits()),
            bits(&case.expected_fault_bits),
            "{function:?} {input:?}"
        );
    }
}

#[test]
fn expm1_meets_the_independent_converged_high_precision_oracle() {
    check_oracle(StableExponential::Expm1);
}

#[test]
fn log1mexp_meets_the_independent_converged_high_precision_oracle() {
    check_oracle(StableExponential::Log1mexp);
}

#[test]
fn protected_exponential_functions_are_monotone_on_dense_valid_grids() {
    let mut old_exp = -f64::MAX;
    let mut old_log = 0.0;
    for index in 0..=10_000 {
        let x = -30.0 + f64::from(index) * 60.0 / 10_000.0;
        let result = StableExponential::Expm1.evaluate(x).0;
        assert!(result >= old_exp);
        old_exp = result;
        let negative = -746.0 + f64::from(index) * 745.999 / 10_000.0;
        let result = StableExponential::Log1mexp.evaluate(negative).0;
        assert!(result <= old_log);
        old_log = result;
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
        command.args([
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-pie",
            "-no-pie",
        ]);
    }
    run(command
        .arg("-o")
        .arg(&binary)
        .arg(root.join("exponentials.cpp")));
    run(Command::new(binary).arg(root.join("cases.txt")))
}

#[test]
fn exponential_helpers_match_runtime_value_and_fault_bits_in_three_languages() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("stable_exponential_v2");
    fs::create_dir_all(&root).unwrap();
    let cases = corpus();
    let mut inputs = String::new();
    let mut expected = String::new();
    for case in &cases {
        inputs.push_str(&format!("{} {}\n", case.function_id, case.input_bits));
        let function = StableExponential::from_versioned_id(2, case.function_id).unwrap();
        let (value, faults) = function.evaluate(f64::from_bits(bits(&case.input_bits)));
        expected.push_str(&format!("{:016x} {:08x}\n", value.to_bits(), faults.bits()));
    }
    fs::write(root.join("cases.txt"), inputs).unwrap();
    let rust = emit_helpers(StableExponentialLanguage::Rust).unwrap()
        + r#"
fn main() {
    let text=std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    for line in text.lines() {
        let mut fields=line.split_whitespace();
        let id=fields.next().unwrap().parse().unwrap();
        let x=f64::from_bits(u64::from_str_radix(fields.next().unwrap(),16).unwrap());
        let (v,f)=stable_exponential_v2(id,x).unwrap();
        println!("{:016x} {:08x}",v.to_bits(),f);
    }
    for id in [0,19,20,23,65535] {assert!(stable_exponential_v2(id,f64::NAN).is_err());}
}
"#;
    let python = emit_helpers(StableExponentialLanguage::Python).unwrap()
        + r#"
for line in open(sys.argv[1]):
    op,raw=line.split()
    x=struct.unpack('>d',bytes.fromhex(raw))[0]
    v,f=stable_exponential_v2(int(op),x)
    print(struct.pack('>d',v).hex(),format(f,'08x'))
for op in [0,19,20,23,65535]:
    try: stable_exponential_v2(op,float('nan'))
    except ValueError: pass
    else: raise AssertionError('unknown function accepted')
"#;
    let cpp = emit_helpers(StableExponentialLanguage::Cpp17).unwrap()
        + r#"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream input(argv[1]); unsigned id; std::uint64_t raw;
    while(input>>std::dec>>id>>std::hex>>raw) {
        const auto result=stable_exponential_v2(id,from_bits(raw));
        std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<to_bits(result.first)
                 <<' '<<std::setw(8)<<result.second<<'\n';
    }
    for(auto bad:{0,19,20,23,65535}) {
        try {stable_exponential_v2(bad,std::numeric_limits<double>::quiet_NaN()); return 3;}
        catch(const std::invalid_argument&) {}
    }
}
"#;
    for (file, source) in [
        ("exponentials.rs", rust),
        ("exponentials.py", python),
        ("exponentials.cpp", cpp),
    ] {
        assert!(feat_math::transpile::lint_shipped_text(&source).is_empty());
        fs::write(root.join(file), source).unwrap();
    }
    let binary = root.join("exponentials-rust");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "-o"])
        .arg(&binary)
        .arg(root.join("exponentials.rs")));
    assert_eq!(
        run(Command::new(binary).arg(root.join("cases.txt"))),
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .arg(root.join("exponentials.py"))
            .arg(root.join("cases.txt"))),
        expected.as_bytes()
    );
    for compiler in ["g++", "clang++"] {
        for (standard, opt) in [
            ("-std=c++17", "-O2"),
            ("-std=c++17", "-O3"),
            ("-std=c++20", "-O3"),
        ] {
            assert_eq!(
                compile_cpp(&root, compiler, standard, opt, false),
                expected.as_bytes()
            );
        }
    }
    assert_eq!(
        compile_cpp(&root, "g++", "-std=c++17", "-O2", true),
        expected.as_bytes()
    );
}
