use super::reduction_cases::mean_properties;
use super::*;

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

fn reduction_boundary_rows() -> Vec<Vec<f64>> {
    let mut rows = Vec::new();
    for width in 6..=128 {
        for exponent in [120_u64, 500, 1000] {
            let big = f64::from_bits((exponent + 1023) << 52);
            let small = f64::from_bits((exponent / 2 + 1023) << 52);
            for residual in [0.0, 1.0, f64::MIN_POSITIVE, f64::from_bits(7)] {
                if width == 6 && residual != 0.0 {
                    continue;
                }
                for sign in [1.0, -1.0] {
                    let mut row = vec![big, 1.0, small, -small, -1.0, -big];
                    row.resize(width, 0.0);
                    if width > 6 {
                        row[6] = residual;
                    }
                    row.iter_mut().for_each(|value| *value *= sign);
                    rows.push(row);
                }
            }
        }
    }
    for width in (1..=128).chain([8191, 8192, 8193, 16383, 16384, 16385]) {
        for sign in [1.0, -1.0] {
            let mut row = vec![sign * 2.0_f64.powi(-27); width];
            row[0] = sign;
            rows.push(row);
        }
    }
    for tail in [2.0_f64.powi(-15), 1.0, f64::MIN_POSITIVE, f64::from_bits(1)] {
        for sign in [1.0, -1.0] {
            for width in 2..=128 {
                let mut row = vec![sign * 1e12, sign * tail];
                row.resize(width, 0.0);
                rows.push(row);
            }
        }
        rows.push(vec![6e11, 8e11, tail]);
        rows.push(vec![f64::from_bits(1e12_f64.to_bits() - 1), tail]);
    }
    assert_eq!(rows.len(), 4226);
    rows
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
    for probe in mean_properties() {
        inputs.push_str(&format!(
            "mean {}\n",
            probe
                .row
                .iter()
                .map(|v| format!("{:016x}", v.to_bits()))
                .collect::<Vec<_>>()
                .join(" ")
        ));
        let (v, f) = reduce_v2(Reduction::Mean, &probe.row);
        expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f.bits()));
    }
    assert_eq!(inputs.lines().count(), 12898);
    for mut row in reduction_boundary_rows() {
        for _ in 0..3 {
            for (reduction, name) in reductions() {
                inputs.push_str(&format!(
                    "{name} {}\n",
                    row.iter()
                        .map(|v| format!("{:016x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                let (v, f) = reduce_v2(reduction, &row);
                expected.push_str(&format!("{:016x} {:08x}\n", v.to_bits(), f.bits()));
            }
            row.rotate_left(1);
            row.reverse();
        }
    }
    assert_eq!(inputs.lines().count(), 63610);
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
    let cpp_compilers: Vec<_> = ["g++", "clang++"]
        .into_iter()
        .filter(|compiler| {
            Command::new(*compiler)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
        })
        .collect();
    assert!(
        !cpp_compilers.is_empty(),
        "the emitted matrix needs at least one C++ compiler"
    );
    for &compiler in &cpp_compilers {
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
        compile_cpp(&root, cpp_compilers[0], "-std=c++17", "-O1", true),
        expected.as_bytes()
    );
}
