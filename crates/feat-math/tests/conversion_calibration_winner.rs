//! Persisted B09 winners execute real exports against read-only boundary rows.
use feat_math::kernel::linear_scaling::LinearScaledWinner;
use feat_math::kernel::wire_v2::VersionedProgram;
use feat_math::math_expr::MathExpr;
use serde_json::Value;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

const HELD: &str = include_str!("fixtures/compat_freeze/mathexpr_values.v1.json");

fn held_rows() -> Vec<Vec<f64>> {
    let fixture: Value = serde_json::from_str(HELD).unwrap();
    let mut rows: Vec<Vec<f64>> = fixture["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row["values"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
                .collect()
        })
        .filter(|row: &Vec<f64>| row.len() == 2)
        .collect();
    assert_eq!(rows.len(), 66);
    rows.extend([
        vec![100.0, 0.0],
        vec![-100.0, 0.0],
        vec![9_007_199_254_740_991.0, 0.0],
        vec![1e-300, -0.0],
        vec![],
        vec![0.0],
        vec![0.0; 3],
    ]);
    rows
}

fn table(rows: &[Vec<f64>], rust: bool) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            let words = row
                .iter()
                .map(|value| {
                    if rust {
                        format!("f64::from_bits(0x{:016x})", value.to_bits())
                    } else {
                        format!("from_bits(0x{:016x}ULL)", value.to_bits())
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            if rust {
                format!("vec![{words}]")
            } else {
                format!("{{{words}}}")
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    if rust {
        format!("vec![{rows}]")
    } else {
        format!("{{{rows}}}")
    }
}

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn execute(dir: &Path, rust: &str, cpp: &str, rows: &[Vec<f64>]) -> Vec<String> {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("winner.rs"), rust).unwrap();
    let rs = dir.join("driver.rs");
    fs::write(&rs, format!(r#"
#[path="winner.rs"] mod generated;
fn main() {{
    let rows: Vec<Vec<f64>> = {};
    assert!(generated::scaled_predict(&[]).unwrap().is_empty());
    for row in rows {{
        match generated::scaled_predict(&[row]) {{
            Ok(values) => {{ assert_eq!(values.len(),1); println!("{{:016x}}",values[0].to_bits()); }}
            Err(_) => println!("REFUSED"),
        }}
    }}
}}
"#, table(rows, true))).unwrap();
    let cpp_file = dir.join("driver.cpp");
    fs::write(
        &cpp_file,
        format!(
            r#"{cpp}
#include <cstdio>
static double from_bits(std::uint64_t b) {{ double v; std::memcpy(&v,&b,8); return v; }}
int main() {{
    const std::vector<std::vector<double>> rows = {};
    if (!scaled_predict({{}}).empty()) return 2;
    for (const auto& row : rows) {{
        try {{
            const auto values = scaled_predict({{row}});
            if (values.size()!=1) return 3;
            std::uint64_t bits; std::memcpy(&bits,&values[0],8);
            std::printf("%016llx\n",static_cast<unsigned long long>(bits));
        }} catch (const std::invalid_argument&) {{ std::puts("REFUSED"); }}
    }}
}}
"#,
            table(rows, false)
        ),
    )
    .unwrap();
    let mut actual = Vec::new();
    for level in ["0", "3"] {
        let binary = dir.join(format!("rust-{level}"));
        run(Command::new("rustc")
            .args(["--edition=2021", "-C"])
            .arg(format!("opt-level={level}"))
            .arg(&rs)
            .arg("-o")
            .arg(&binary));
        actual.push(run(&mut Command::new(binary)));
    }
    for optimization in ["-O2", "-O3"] {
        let binary = dir.join(format!("cpp{optimization}"));
        run(Command::new("g++")
            .args([
                "-std=c++17",
                optimization,
                "-ffp-contract=off",
                "-fno-fast-math",
            ])
            .arg(&cpp_file)
            .arg("-o")
            .arg(&binary));
        actual.push(run(&mut Command::new(binary)));
    }
    actual
}

#[test]
fn persisted_calibration_exports_match_evaluator_bits_and_refusals() {
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_TARGET_TMPDIR").into());
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::tempdir_in(root).unwrap();
    let rows = held_rows();
    for (index, (source, extreme)) in [
        ("c0", false),
        ("c0*c0+c1", false),
        ("step(c0,c1)", false),
        ("c0/(abs(c1)+1)", false),
        ("c0", true),
    ]
    .into_iter()
    .enumerate()
    {
        let expression = MathExpr::parse(source).unwrap();
        let limit = if extreme { 1 } else { 2 };
        let slope = if extreme { f64::MAX } else { 3.0 };
        let intercept = if extreme { 0.0 } else { 7.0 };
        // Symmetric raw values or balanced Step classes make the coefficient
        // witness exact without claiming exact coefficients for arbitrary fits.
        let train: Vec<_> = (-limit..=limit)
            .filter(|&x| source != "step(c0,c1)" || x != 0)
            .map(|x| {
                if source == "c0*c0+c1" {
                    vec![0.0, f64::from(x)]
                } else {
                    vec![f64::from(x), 0.0]
                }
            })
            .collect();
        let targets: Vec<_> = train
            .iter()
            .map(|row| slope * expression.eval_row(row).unwrap() + intercept)
            .collect();
        let fit = LinearScaledWinner::fit_training(&expression, &train, &targets).unwrap();
        assert_eq!(
            fit.winner.coefficients().map(f64::to_bits),
            [slope.to_bits(), intercept.to_bits()]
        );
        let dir = scratch.path().join(index.to_string());
        fs::create_dir_all(&dir).unwrap();
        let bytes = fit.winner.to_bytes().unwrap();
        fs::write(dir.join("winner.bin"), &bytes).unwrap();
        drop(fit);
        let stored = fs::read(dir.join("winner.bin")).unwrap();
        let winner = LinearScaledWinner::from_bytes(&stored).unwrap();
        let text = VersionedProgram::from_bytes(&stored)
            .unwrap()
            .to_lossless_text()
            .unwrap();
        assert_eq!(
            VersionedProgram::parse_lossless_text(&text)
                .unwrap()
                .to_bytes()
                .unwrap(),
            stored
        );
        let compiled = expression.compile().unwrap();
        let [a, b] = winner.coefficients();
        let mut expected = String::new();
        for row in &rows {
            // Preserve the documented raw-tree root fallback. Invalid input and
            // nonfinite affine output must refuse, rather than silently return 0.
            let oracle = if row.len() == 2 && row.iter().all(|v| v.is_finite()) {
                let (raw, _) = compiled.eval_row_traced(row);
                let product = std::hint::black_box(a * raw);
                let value = product + b;
                value.is_finite().then_some(value.to_bits())
            } else {
                None
            };
            match oracle {
                Some(word) => {
                    assert_eq!(
                        winner.predict(std::slice::from_ref(row)).unwrap()[0].to_bits(),
                        word,
                        "{source}: {row:?}"
                    );
                    writeln!(expected, "{word:016x}").unwrap();
                }
                None => {
                    assert!(winner.predict(std::slice::from_ref(row)).is_err());
                    writeln!(expected, "REFUSED").unwrap();
                }
            }
        }
        assert!(expected.contains("REFUSED\n"));
        assert_ne!(expected, "0000000000000000\n".repeat(rows.len()));
        assert_ne!(
            expected,
            format!("{:016x}\n", b.to_bits()).repeat(rows.len())
        );
        if source == "c0" && !extreme {
            assert_eq!(winner.predict(&[vec![100.0, 0.0]]).unwrap(), [307.0]);
        }
        if extreme {
            assert!(winner.predict(&[vec![100.0, 0.0]]).is_err());
        }
        let artifact = winner.emit().unwrap();
        assert_eq!(artifact.winner_identity, winner.identity().unwrap());
        for (backend, actual) in execute(&dir, &artifact.rust, &artifact.cpp17, &rows)
            .into_iter()
            .enumerate()
        {
            assert_eq!(actual, expected, "{source}: backend {backend}");
        }
        assert_eq!(winner.to_bytes().unwrap(), stored);
        assert_eq!(fs::read(dir.join("winner.bin")).unwrap(), bytes);
        eprintln!(
            "witness={index} source={source} extreme={extreme} rows={} ports=4 value_bits={} refusals={}",
            rows.len(),
            expected.lines().filter(|line| *line != "REFUSED").count(),
            expected.lines().filter(|line| *line == "REFUSED").count()
        );
    }
}
