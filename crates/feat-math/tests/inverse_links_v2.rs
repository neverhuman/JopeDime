//! Numerical prerequisite for B05. These are helper execution proofs, not
//! admission of the two functions into shipped expression search.
use feat_math::kernel::inverse_links::{emit_helpers, InverseLink, Language, MAX_P, MIN_P};
use feat_math::math_expr::{FaultClass, FaultSet, FaultSeverity, MathExpr};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct Corpus {
    entries: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    input_bits: String,
    probit_oracle_bits: String,
    logit_oracle_bits: String,
    expected_clamp: bool,
}

fn corpus() -> Vec<Case> {
    serde_json::from_str::<Corpus>(include_str!("fixtures/math_boundary/v2_inverse_links.json"))
        .unwrap()
        .entries
}

fn bits(value: &str) -> u64 {
    u64::from_str_radix(value, 16).unwrap()
}

#[test]
fn inverse_link_ids_require_version_two_and_remain_outside_legacy_search() {
    for (id, link, name) in [
        (19, InverseLink::Probit, "probit"),
        (20, InverseLink::Logit, "logit"),
    ] {
        assert_eq!(InverseLink::from_versioned_id(2, id).unwrap(), link);
        assert_eq!(link.id(), id);
        for version in [0, 1, 3, 255] {
            assert!(matches!(
                InverseLink::from_versioned_id(version, id),
                Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
            ));
        }
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
    for id in [0, 18, 21, 22, 34, u16::MAX] {
        assert!(matches!(
            InverseLink::from_versioned_id(2, id),
            Err(feat_math::kernel::wire_v2::VersionError::UnsupportedVersion(_))
        ));
    }
}

#[test]
fn finite_probability_projection_has_an_append_only_saturation_fault() {
    assert_eq!(MIN_P.to_bits(), 0x3ca0_0000_0000_0000);
    assert_eq!(MAX_P.to_bits(), 0x3fef_ffff_ffff_ffff);
    assert_eq!(FaultClass::ProbabilityClamped.index(), 16);
    assert_eq!(
        FaultClass::ProbabilityClamped.severity(),
        FaultSeverity::Saturation
    );
    for link in [InverseLink::Probit, InverseLink::Logit] {
        for (input, endpoint) in [
            (-f64::MAX, MIN_P),
            (-0.0, MIN_P),
            (0.0, MIN_P),
            (f64::from_bits(MIN_P.to_bits() - 1), MIN_P),
            (1.0, MAX_P),
            (f64::MAX, MAX_P),
        ] {
            let (value, faults) = link.evaluate(input);
            assert_eq!(value.to_bits(), link.evaluate(endpoint).0.to_bits());
            assert_eq!(faults, FaultSet::PROBABILITY_CLAMPED);
        }
        for input in [MIN_P, 0.5, MAX_P] {
            assert_eq!(link.evaluate(input).1, FaultSet::EMPTY);
        }
        assert_eq!(link.evaluate(0.5).0.to_bits(), 0.0_f64.to_bits());
    }
    assert!(InverseLink::Probit.evaluate(MIN_P).0 < -8.2);
    // Frozen Blom retains its 1e-12 clamp and its original helper result.
    assert_eq!(
        feat_math::ops::probit(MIN_P).to_bits(),
        feat_math::ops::probit(1e-12).to_bits()
    );
}

#[test]
fn nonfinite_probabilities_return_positive_zero_with_a_domain_fault() {
    for link in [InverseLink::Probit, InverseLink::Logit] {
        for input in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_0001),
        ] {
            let (value, faults) = link.evaluate(input);
            assert_eq!(value.to_bits(), 0);
            assert_eq!(faults, FaultSet::NONFINITE_INPUT);
            assert!(!faults.contains(FaultSet::PROBABILITY_CLAMPED));
        }
    }
}

fn check_oracle(link: InverseLink) {
    let cases = corpus();
    assert_eq!(cases.len(), 1128);
    for case in cases {
        let input = f64::from_bits(bits(&case.input_bits));
        let oracle = match link {
            InverseLink::Probit => case.probit_oracle_bits,
            InverseLink::Logit => case.logit_oracle_bits,
        };
        let expected = f64::from_bits(bits(&oracle));
        let (actual, faults) = link.evaluate(input);
        assert!(actual.is_finite());
        match link {
            InverseLink::Probit => {
                let bound = expected.abs() * 1.2e-9;
                assert!(
                    (actual - expected).abs() <= bound,
                    "{link:?} input={input:?} actual={actual:?} oracle={expected:?} bound={bound}"
                );
            }
            InverseLink::Logit => {
                assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                let ulps = actual.abs().to_bits().abs_diff(expected.abs().to_bits());
                assert!(
                    ulps <= 1 && (expected != 0.0 || actual.to_bits() == 0),
                    "{link:?} input={input:?} actual={actual:?} oracle={expected:?} ulps={ulps}"
                );
            }
        }
        let expected_faults = if !input.is_finite() {
            FaultSet::NONFINITE_INPUT
        } else if case.expected_clamp {
            FaultSet::PROBABILITY_CLAMPED
        } else {
            FaultSet::EMPTY
        };
        assert_eq!(faults, expected_faults);
    }
}

#[test]
fn probit_meets_the_approved_independent_high_precision_oracle() {
    check_oracle(InverseLink::Probit);
}

#[test]
fn logit_meets_the_approved_one_ulp_independent_high_precision_oracle() {
    check_oracle(InverseLink::Logit);
}

#[test]
fn inverse_links_are_monotone_on_a_dense_probability_grid() {
    let inputs: Vec<_> = std::iter::once(MIN_P)
        .chain((1..10_000).map(|i| f64::from(i) / 10_000.0))
        .chain(std::iter::once(MAX_P))
        .collect();
    for link in [InverseLink::Probit, InverseLink::Logit] {
        for pair in inputs.windows(2) {
            let a = link.evaluate(pair[0]).0;
            let b = link.evaluate(pair[1]).0;
            assert!(a <= b, "{link:?} {pair:?}: {a:?} > {b:?}");
        }
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

fn compile_cpp(root: &Path, compiler: &str, standard: &str, optimization: &str) -> Vec<u8> {
    let binary = root.join(format!("{compiler}-{standard}-{optimization}"));
    run(Command::new(compiler)
        .args([
            standard,
            optimization,
            "-ffp-contract=off",
            "-fno-fast-math",
            "-o",
        ])
        .arg(&binary)
        .arg(root.join("links.cpp")));
    run(Command::new(binary).arg(root.join("bits.txt")))
}

#[test]
fn emitted_helper_value_and_fault_bits_match_in_all_three_languages() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inverse_links_v2");
    fs::create_dir_all(&root).unwrap();
    let cases = corpus();
    let inputs = cases
        .iter()
        .map(|case| case.input_bits.clone())
        .collect::<Vec<_>>();
    fs::write(root.join("bits.txt"), inputs.join("\n") + "\n").unwrap();
    let mut expected = String::new();
    for input in &inputs {
        for link in [InverseLink::Probit, InverseLink::Logit] {
            let (value, faults) = link.evaluate(f64::from_bits(bits(input)));
            expected.push_str(&format!("{:016x} {:08x}\n", value.to_bits(), faults.bits()));
        }
    }
    let rust = emit_helpers(Language::Rust)
        + r#"
fn main() {
    let path = std::env::args().nth(1).unwrap();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let p = f64::from_bits(u64::from_str_radix(line, 16).unwrap());
        for id in [19,20] {
            let (v,f) = inverse_link_v2(id,p).unwrap();
            println!("{:016x} {:08x}",v.to_bits(),f);
        }
    }
    assert!(inverse_link_v2(21,0.5).is_err());
}
"#;
    let python = emit_helpers(Language::Python)
        + r#"
import struct, sys
for line in open(sys.argv[1]):
    p = struct.unpack('>d',bytes.fromhex(line.strip()))[0]
    for op in [19,20]:
        v,f = inverse_link_v2(op,p)
        print(struct.pack('>d',v).hex(),format(f,'08x'))
try:
    inverse_link_v2(21,0.5)
except ValueError:
    pass
else:
    raise AssertionError('unknown function accepted')
"#;
    let cpp = emit_helpers(Language::Cpp17)
        + r#"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream input(argv[1]); std::uint64_t bits;
    while(input>>std::hex>>bits) {
        double p; std::memcpy(&p,&bits,sizeof(p));
        for(auto id:{19,20}) {
            auto value=inverse_link_v2(id,p);
            std::uint64_t output; std::memcpy(&output,&value.first,sizeof(output));
            std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<output<<' '
                     <<std::setw(8)<<value.second<<'\n';
        }
    }
    try {inverse_link_v2(21,0.5); return 3;} catch(const std::invalid_argument&) {}
}
"#;
    for (file, source) in [("links.rs", rust), ("links.py", python), ("links.cpp", cpp)] {
        assert!(feat_math::transpile::lint_shipped_text(&source).is_empty());
        fs::write(root.join(file), source).unwrap();
    }
    let binary = root.join("links-rust");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "-o"])
        .arg(&binary)
        .arg(root.join("links.rs")));
    assert_eq!(
        run(Command::new(binary).arg(root.join("bits.txt"))),
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .arg(root.join("links.py"))
            .arg(root.join("bits.txt"))),
        expected.as_bytes()
    );
    for compiler in ["g++", "clang++"] {
        for (standard, opt) in [
            ("-std=c++17", "-O2"),
            ("-std=c++17", "-O3"),
            ("-std=c++20", "-O3"),
        ] {
            assert_eq!(
                compile_cpp(&root, compiler, standard, opt),
                expected.as_bytes()
            );
        }
    }
}
