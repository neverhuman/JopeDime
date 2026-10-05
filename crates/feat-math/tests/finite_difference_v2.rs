//! Causal difference kernels, native routing, ordered recipes and fitted continuation.
use feat_math::kernel::finite_difference::{
    emit_helpers, DifferenceConfig, DifferenceLanguage, DifferenceState, BOUNDARY_HISTORY,
};
use feat_math::kernel::ordered_axis::{
    FoldBoundary, FoldPhase, GroupKey, OrderedInputIdentity, TaskMode,
};
use feat_math::kernel::Error;
use feat_math::math_expr::FaultSet;
use std::fs;
use std::path::Path;
use std::process::Command;

fn config(order: u8, step: f64) -> DifferenceConfig {
    DifferenceConfig::new(2, 59, order, step).unwrap()
}

#[test]
fn difference_configuration_is_versioned_bounded_and_generation_stays_closed() {
    for order in 1..=4 {
        for step in [f64::from_bits(1), 0.5, 1.0, f64::MAX] {
            assert!(DifferenceConfig::new(2, 59, order, step).is_ok());
        }
    }
    for version in [0, 1, 3, 255] {
        assert!(matches!(
            DifferenceConfig::new(version, 59, 1, 1.0),
            Err(Error::Unsupported(_))
        ));
    }
    for wire in [0, 52, 58, 60, u16::MAX] {
        assert!(matches!(
            DifferenceConfig::new(2, wire, 1, 1.0),
            Err(Error::Unsupported(_))
        ));
    }
    for order in [0, 5, 255] {
        assert!(matches!(
            DifferenceConfig::new(2, 59, order, 1.0),
            Err(Error::Domain(_))
        ));
    }
    for step in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            DifferenceConfig::new(2, 59, 1, step),
            Err(Error::Domain(_))
        ));
    }
    let reserved = feat_math::kernel::wire_v2::WIRE
        .iter()
        .find(|entry| entry.id == 59)
        .unwrap();
    assert!(!reserved.generatable);
    assert_eq!(FaultSet::ALL.bits(), 0x1ffff);
    assert_eq!(BOUNDARY_HISTORY, 1 << 17);
    assert!(FaultSet::from_bits(BOUNDARY_HISTORY).is_none());
}

#[test]
fn integer_backward_stencils_reproduce_polynomials_for_all_four_orders() {
    for order in 1..=4 {
        for step in [0.5, 1.0, 2.0] {
            for degree in 0..=order {
                for end in order..=12 {
                    let window = (end - order..=end)
                        .map(|i| vec![(f64::from(i) * step).powi(i32::from(degree))])
                        .collect::<Vec<_>>();
                    let result = config(order, step).evaluate(&window).unwrap();
                    let expected = if degree < order {
                        0.0
                    } else {
                        (1..=order).map(f64::from).product()
                    };
                    assert_eq!(
                        result.values[0].to_bits(),
                        expected.to_bits(),
                        "m={order} h={step} degree={degree} end={end}"
                    );
                    assert_eq!(result.faults, vec![0]);
                }
            }
        }
    }
}

#[test]
fn first_m_observations_are_positive_zero_with_an_explicit_boundary_fault() {
    for order in 1..=4 {
        for count in 1..=order {
            let window = vec![vec![-0.0, 4.0]; usize::from(count)];
            let result = config(order, 1.0).evaluate(&window).unwrap();
            assert_eq!(
                result
                    .values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                vec![0, 0]
            );
            assert_eq!(result.faults, vec![BOUNDARY_HISTORY; 2]);
        }
    }
}

#[test]
fn width_bounds_and_ragged_or_excess_history_refuse_before_evaluation() {
    let kernel = config(2, 1.0);
    for bad in [
        vec![],
        vec![vec![]],
        vec![vec![1.0], vec![2.0, 3.0]],
        vec![vec![0.0]; 4],
        vec![vec![0.0; feat_math::MAX_WIDTH + 1]],
    ] {
        assert!(kernel.evaluate(&bad).is_err());
    }
    let window = vec![vec![3.0; feat_math::MAX_WIDTH]; 3];
    let result = kernel.evaluate(&window).unwrap();
    assert_eq!(result.values.len(), feat_math::MAX_WIDTH);
    assert!(result.values.iter().all(|v| v.to_bits() == 0));
    assert!(result.faults.iter().all(|v| *v == 0));
}

#[test]
fn causal_group_histories_continue_across_the_training_boundary() {
    let input = OrderedInputIdentity::new("group", "time");
    assert!(DifferenceState::new(
        config(1, 1.0),
        1,
        TaskMode::IID,
        &input,
        FoldBoundary::new("cut", 2, 6).unwrap()
    )
    .is_err());
    let mut state = DifferenceState::new(
        config(1, 1.0),
        1,
        TaskMode::Forecasting,
        &input,
        FoldBoundary::new("cut", 2, 6).unwrap(),
    )
    .unwrap();
    for (i, (group, time, value, expected)) in [
        ("a", 0.0, 1.0, 0.0_f64),
        ("b", 0.0, 20.0, 0.0),
        ("a", 1.0, 4.0, 3.0),
        ("b", 0.0, 25.0, 5.0),
        ("a", 9.0, 9.0, 5.0),
        ("b", 20.0, 24.0, -1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let row = state
            .push(GroupKey::categorical(group).unwrap(), time, &[value])
            .unwrap();
        assert_eq!(row.output.values[0].to_bits(), expected.to_bits());
        assert_eq!(
            row.output.faults,
            vec![if i < 2 { BOUNDARY_HISTORY } else { 0 }]
        );
        assert_eq!(
            row.admission.phase,
            if i < 2 {
                FoldPhase::Train
            } else {
                FoldPhase::HeldOut
            }
        );
        assert!(!row.admission.may_fit);
    }
    state.finish().unwrap();
}

#[test]
fn rejected_rows_do_not_spend_ordinals_or_mutate_group_history() {
    let mut state = DifferenceState::new(
        config(1, 1.0),
        1,
        TaskMode::Ordered,
        &OrderedInputIdentity::new("g", "t"),
        FoldBoundary::new("cut", 1, 2).unwrap(),
    )
    .unwrap();
    for (group, time, row) in [
        (GroupKey::Categorical(String::new()), 0.0, vec![1.0]),
        (GroupKey::NumericBits(f64::NAN.to_bits()), 0.0, vec![1.0]),
        (GroupKey::Categorical("a".into()), 0.0, vec![]),
        (GroupKey::Categorical("a".into()), 0.0, vec![1.0, 2.0]),
        (GroupKey::Categorical("a".into()), f64::NAN, vec![1.0]),
    ] {
        assert!(state.push(group, time, &row).is_err());
    }
    let first = state
        .push(GroupKey::categorical("a").unwrap(), 1.0, &[3.0])
        .unwrap();
    assert_eq!(first.admission.ordinal, 0);
    assert!(state
        .push(GroupKey::categorical("a").unwrap(), 0.0, &[999.0])
        .is_err());
    let second = state
        .push(GroupKey::categorical("a").unwrap(), 2.0, &[7.0])
        .unwrap();
    assert_eq!(second.admission.ordinal, 1);
    assert_eq!(second.output.values, vec![4.0]);
    assert!(state
        .push(GroupKey::categorical("a").unwrap(), 3.0, &[9.0])
        .is_err());
    state.finish().unwrap();
}

#[test]
fn held_out_poison_and_a_new_cut_cannot_change_the_training_prefix() {
    let run = |poison| {
        let mut state = DifferenceState::new(
            config(2, 1.0),
            1,
            TaskMode::Forecasting,
            &OrderedInputIdentity::new("g", "t"),
            FoldBoundary::new("cut", 3, 5).unwrap(),
        )
        .unwrap();
        let mut outputs = Vec::new();
        for (i, value) in [0.0, 1.0, 4.0, poison, poison].into_iter().enumerate() {
            outputs.push(
                state
                    .push(GroupKey::numeric(0.0).unwrap(), i as f64, &[value])
                    .unwrap()
                    .output,
            );
        }
        state.finish().unwrap();
        outputs
    };
    let normal = run(9.0);
    let poison = run(f64::MAX);
    assert_eq!(normal[..3], poison[..3]);
    assert_eq!(normal[2].values, vec![2.0]);
    let mut fresh = DifferenceState::new(
        config(2, 1.0),
        1,
        TaskMode::Forecasting,
        &OrderedInputIdentity::new("g", "t"),
        FoldBoundary::new("next-cut", 1, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        fresh
            .push(GroupKey::numeric(-0.0).unwrap(), 0.0, &[99.0])
            .unwrap()
            .output
            .faults,
        vec![BOUNDARY_HISTORY]
    );
    fresh.finish().unwrap();
}

#[test]
fn finite_difference_guards_nonfinite_cells_and_preserves_valid_zero_signs() {
    let kernel = config(1, 1.0);
    let result = kernel
        .evaluate(&[vec![0.0, 0.0], vec![f64::NAN, -0.0]])
        .unwrap();
    assert_eq!(result.values[0].to_bits(), 0);
    assert_eq!(result.faults[0], FaultSet::NONFINITE_INPUT.bits());
    assert_eq!(result.values[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(result.faults[1], 0);
    let saturated = kernel.evaluate(&[vec![0.0], vec![1e13]]).unwrap();
    assert_eq!(saturated.values, vec![1e12]);
    assert_eq!(saturated.faults, vec![FaultSet::MAGNITUDE_CLAMPED.bits()]);
    let unrepresentable = kernel.evaluate(&[vec![-f64::MAX], vec![f64::MAX]]).unwrap();
    assert_eq!(unrepresentable.values, vec![0.0]);
    assert_eq!(
        unrepresentable.faults,
        vec![FaultSet::NONFINITE_SANITIZED.bits()]
    );
    let scaled = config(1, f64::MAX)
        .evaluate(&[vec![-f64::MAX], vec![f64::MAX]])
        .unwrap();
    assert_eq!(scaled.values, vec![2.0]);
    assert_eq!(scaled.faults, vec![0]);
    for order in 1..=4 {
        let constant = vec![vec![f64::MAX]; usize::from(order) + 1];
        assert_eq!(
            config(order, 1.0).evaluate(&constant).unwrap().values,
            vec![0.0]
        );
        assert_eq!(
            config(order, f64::from_bits(1))
                .evaluate(&constant)
                .unwrap()
                .faults,
            vec![0]
        );
    }
}

fn run(command: &mut Command) -> Vec<u8> {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn cpp(root: &Path, compiler: &str, standard: &str, opt: &str, sanitizer: bool) -> Vec<u8> {
    let binary = root.join(format!("{compiler}-{standard}-{opt}-{sanitizer}"));
    let mut command = Command::new(compiler);
    command.args([standard, opt, "-ffp-contract=off", "-fno-fast-math"]);
    if sanitizer {
        command.args([
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-pie",
            "-no-pie",
        ]);
    }
    run(command
        .arg(root.join("difference.cpp"))
        .arg("-o")
        .arg(&binary));
    run(Command::new(binary).arg(root.join("cases.txt")))
}

#[test]
fn three_language_helpers_match_value_and_fault_bits_on_the_boundary_corpus() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("finite_difference_v2");
    fs::create_dir_all(&root).unwrap();
    let raw = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x000fffffffffffff,
        0x0010000000000000,
        0x3ff0000000000000,
        0x4008000000000000,
        0x7fefffffffffffff,
        0xffefffffffffffff,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff8000000000001,
        0xfff8000000000001,
        0x7ff0000000000001,
    ];
    let mut input = String::new();
    let mut expected = String::new();
    let mut cases = 0;
    for order in 1..=4 {
        for h in [f64::from_bits(1), 1e-300, 0.5, 1.0, 2.0, 1e200, f64::MAX] {
            for width in [1, 3] {
                for count in 1..=usize::from(order) + 1 {
                    for offset in 0..raw.len() {
                        let window = (0..count)
                            .map(|i| {
                                (0..width)
                                    .map(|j| f64::from_bits(raw[(offset + i * 3 + j) % raw.len()]))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        input.push_str(&format!("{order} {:016x} {count} {width}", h.to_bits()));
                        for row in &window {
                            for v in row {
                                input.push_str(&format!(" {:016x}", v.to_bits()));
                            }
                        }
                        input.push('\n');
                        let result = config(order, h).evaluate(&window).unwrap();
                        for (v, f) in result.values.iter().zip(result.faults) {
                            expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f));
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2940);
    fs::write(root.join("cases.txt"), input).unwrap();
    let rust = emit_helpers(DifferenceLanguage::Rust)
        + r#"
fn main(){let text=std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();for line in text.lines(){let mut f=line.split_whitespace();let m=f.next().unwrap().parse().unwrap();let h=f64::from_bits(u64::from_str_radix(f.next().unwrap(),16).unwrap());let n:usize=f.next().unwrap().parse().unwrap();let w:usize=f.next().unwrap().parse().unwrap();let rows=(0..n).map(|_|(0..w).map(|_|f64::from_bits(u64::from_str_radix(f.next().unwrap(),16).unwrap())).collect()).collect::<Vec<Vec<f64>>>();let(v,s)=finite_difference_v2(2,59,m,h,&rows).unwrap();for(x,b)in v.iter().zip(s){println!("{:016x} {:08x}",x.to_bits(),b);}}assert!(finite_difference_v2(1,59,1,1.0,&[vec![1.0]]).is_err());assert!(finite_difference_v2(2,59,1,1.0,&[vec![]]).is_err());}
"#;
    let python = emit_helpers(DifferenceLanguage::Python)
        + r#"
for line in open(sys.argv[1]):
    f=iter(line.split());m=int(next(f));h=struct.unpack('>d',bytes.fromhex(next(f)))[0];n=int(next(f));w=int(next(f))
    rows=[[struct.unpack('>d',bytes.fromhex(next(f)))[0] for _ in range(w)] for _ in range(n)]
    v,s=finite_difference_v2(2,59,m,h,rows)
    for x,b in zip(v,s): print(struct.pack('>d',x).hex(),format(b,'08x'))
for version,rows in [(1,[[1.0]]),(2,[[]])]:
    try: finite_difference_v2(version,59,1,1.0,rows)
    except ValueError: pass
    else: raise AssertionError('invalid request accepted')
"#;
    let cpp_source = emit_helpers(DifferenceLanguage::Cpp17)
        + r#"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){if(argc!=2)return 2;std::ifstream input(argv[1]);unsigned m;std::uint64_t hb;std::size_t n,w;while(input>>std::dec>>m>>std::hex>>hb>>std::dec>>n>>w){std::vector<std::vector<double>> rows(n,std::vector<double>(w));for(auto& row:rows)for(auto& x:row){std::uint64_t bits;input>>std::hex>>bits;std::memcpy(&x,&bits,8);}double h;std::memcpy(&h,&hb,8);auto result=finite_difference_v2(2,59,m,h,rows);for(std::size_t i=0;i<w;i++){std::uint64_t bits;std::memcpy(&bits,&result.first[i],8);std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<bits<<' '<<std::setw(8)<<result.second[i]<<'\n';}}for(unsigned version:{1,2}){try{finite_difference_v2(version,59,1,1.0,version==1?std::vector<std::vector<double>>{{1.0}}:std::vector<std::vector<double>>{{}});return 3;}catch(const std::invalid_argument&){}}}
"#;
    for (name, source) in [
        ("difference.rs", rust),
        ("difference.py", python),
        ("difference.cpp", cpp_source),
    ] {
        assert!(feat_math::transpile::lint_shipped_text(&source).is_empty());
        fs::write(root.join(name), source).unwrap();
    }
    let binary = root.join("difference-rust");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "-o"])
        .arg(&binary)
        .arg(root.join("difference.rs")));
    assert_eq!(
        run(Command::new(binary).arg(root.join("cases.txt"))),
        expected.as_bytes()
    );
    assert_eq!(
        run(Command::new("python3")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("LC_ALL", "C")
            .arg(root.join("difference.py"))
            .arg(root.join("cases.txt"))),
        expected.as_bytes()
    );
    for compiler in ["g++", "clang++"] {
        for (std, opt) in [
            ("-std=c++17", "-O2"),
            ("-std=c++17", "-O3"),
            ("-std=c++20", "-O3"),
        ] {
            assert_eq!(cpp(&root, compiler, std, opt, false), expected.as_bytes());
        }
    }
    assert_eq!(
        cpp(&root, "g++", "-std=c++17", "-O2", true),
        expected.as_bytes()
    );
}

// B08 ordered-state program/model/fit continuation.
#[path = "finite_difference/routing.rs"]
mod routing;

// B08 ordered recipes and fitted C++ continuation.
#[path = "finite_difference/recipes.rs"]
mod recipes;

// B08 bounded Experimental generation, EEC and equal-budget evidence.
#[path = "finite_difference/eec.rs"]
mod eec;
#[path = "finite_difference/profile.rs"]
mod profile;
#[path = "finite_difference/shortest.rs"]
mod shortest;
#[path = "finite_difference/utility.rs"]
mod utility;
#[path = "finite_difference/utility_exports.rs"]
mod utility_exports;
