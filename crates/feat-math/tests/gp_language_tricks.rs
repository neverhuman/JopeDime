//! Concrete GP-language witnesses; the shared determinism harness is not modified.
use feat_math::kernel::inverse_links::InverseLink;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::math_expr::MathExpr;
use feat_math::math_expr::Var;
use serde_json::json;
use std::{fs, path::Path, process::Command};

struct Trick {
    name: &'static str,
    tree: &'static str,
    reference: fn(&[f64]) -> f64,
    positive: bool,
}

type TrickSpec = (&'static str, &'static str, fn(&[f64]) -> f64);

fn tricks() -> Vec<Trick> {
    let cases: &[TrickSpec] = &[
        ("ols_predictor", "2*c0+1", |r| 2.0 * r[0] + 1.0),
        // Centered one-dimensional training: sum(x*x)=4, sum(x*y)=8, lambda=1.
        ("ridge_closed_form", "(8/(4+1))*c0+1", |r| 1.6 * r[0] + 1.0),
        // Soft threshold is a closed form only for the orthogonal design case.
        ("lasso_orthogonal", "sign(c1)*max(abs(c1)-0.25,0)*c0", |r| {
            r[1].signum() * (r[1].abs() - 0.25).max(0.0) * r[0]
        }),
        (
            "elastic_net_orthogonal",
            "sign(c1)*max(abs(c1)-0.25,0)/(1+0.5)*c0",
            |r| r[1].signum() * (r[1].abs() - 0.25).max(0.0) / 1.5 * r[0],
        ),
        ("glm_logistic", "sigmoid(c0)", |r| {
            1.0 / (1.0 + (-r[0]).exp())
        }),
        ("glm_log_link", "exp(c0)", |r| r[0].exp()),
        ("hinge", "max(c0-0.5,0)", |r| {
            if r[0] > 0.5 {
                r[0] - 0.5
            } else {
                0.0
            }
        }),
        ("linear_spline", "c0+0.5*max(c0-0.5,0)", |r| {
            if r[0] > 0.5 {
                1.5 * r[0] - 0.25
            } else {
                r[0]
            }
        }),
        ("cubic_spline", "c0+0.5*max(c0-0.5,0)^3", |r| {
            let h = (r[0] - 0.5).max(0.0);
            r[0] + 0.5 * h * h * h
        }),
        (
            "nested_decision_tree",
            "(1-step(c0,0))*2+step(c0,0)*((1-step(c1,0.5))*3+step(c1,0.5)*4)",
            |r| {
                if r[0] <= 0.0 {
                    2.0
                } else if r[1] <= 0.5 {
                    3.0
                } else {
                    4.0
                }
            },
        ),
        (
            "boosted_stumps",
            "0.5+0.25*step(c0,0)-0.5*step(c1,0.5)",
            |r| 0.5 + if r[0] > 0.0 { 0.25 } else { 0.0 } - if r[1] > 0.5 { 0.5 } else { 0.0 },
        ),
        ("rbf_kernel", "exp(-0.5*((c0-c2)^2+(c1-c3)^2))", |r| {
            let a = r[0] - r[2];
            let b = r[1] - r[3];
            (-0.5 * (a * a + b * b)).exp()
        }),
        ("polynomial_kernel", "(1+c0*c2+c1*c3)^3", |r| {
            let v = 1.0 + r[0] * r[2] + r[1] * r[3];
            v * v * v
        }),
        ("standardize_frozen_train_state", "(c0-0.5)/2", |r| {
            (r[0] - 0.5) / 2.0
        }),
        ("minmax_frozen_train_state", "(c0+2)/4", |r| {
            (r[0] + 2.0) / 4.0
        }),
        ("log1p_transform", "log1p(abs(c0))", |r| r[0].abs().ln_1p()),
        (
            "softmax_component",
            "exp(c0-max(c0,c1))/(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))",
            |r| {
                let m = r[0].max(r[1]);
                let a = (r[0] - m).exp();
                a / (a + (r[1] - m).exp())
            },
        ),
        (
            "log_sum_exp",
            "max(c0,c1)+log(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))",
            |r| {
                let m = r[0].max(r[1]);
                m + ((r[0] - m).exp() + (r[1] - m).exp()).ln()
            },
        ),
        ("clamp", "clip(c0,-0.5,0.5)", |r| r[0].clamp(-0.5, 0.5)),
        (
            "where_finite_branches",
            "step(c0,0)*c1+(1-step(c0,0))*c2",
            |r| {
                if r[0] > 0.0 {
                    r[1]
                } else {
                    r[2]
                }
            },
        ),
        ("interactions", "c0*c1+c2*c3", |r| r[0] * r[1] + r[2] * r[3]),
        ("rational_function", "(c0+c1)/(1+c2*c2)", |r| {
            (r[0] + r[1]) / (1.0 + r[2] * r[2])
        }),
    ];
    let mut out = cases
        .iter()
        .map(|&(name, tree, reference)| Trick {
            name,
            tree,
            reference,
            positive: false,
        })
        .collect::<Vec<_>>();
    out.extend([
        Trick {
            name: "box_cox_lambda2",
            tree: "(c0^2-1)/2",
            reference: |r| (r[0] * r[0] - 1.0) / 2.0,
            positive: true,
        },
        Trick {
            name: "box_cox_lambda0",
            tree: "log(c0)",
            reference: |r| r[0].ln(),
            positive: true,
        },
        Trick {
            name: "glm_logit_link",
            tree: "log(c0/(1-c0))",
            reference: |r| (r[0] / (1.0 - r[0])).ln(),
            positive: true,
        },
    ]);
    out.extend([
        Trick {
            name: "glm_probit_link_v2",
            tree: "probit_v2(c0)",
            positive: true,
            reference: |r| {
                // sqrt(2)*erfinv(2*p-1), independently evaluated at 100 decimal digits.
                const ORACLE: [f64; 9] = [
                    -1.5341205443525463,
                    -1.150349380376008,
                    -0.8871465590188761,
                    -0.6744897501960817,
                    -0.4887764111146695,
                    -0.31863936396437514,
                    -0.1573106846101707,
                    0.0,
                    0.1573106846101707,
                ];
                ORACLE[(r[0] * 16.0) as usize - 1]
            },
        },
        Trick {
            name: "glm_logit_link_v2",
            tree: "logit_v2(c0)",
            positive: true,
            reference: |r| r[0].ln() - (-r[0]).ln_1p(),
        },
    ]);
    out
}

fn probes(positive: bool) -> Vec<Vec<f64>> {
    [
        -2.0,
        -1.0,
        -0.0,
        0.0,
        f64::from_bits(1),
        0.5,
        f64::from_bits(0.5f64.to_bits() + 1),
        1.0,
        2.0,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, value)| {
        vec![
            if positive {
                (i as f64 + 1.0) / 16.0
            } else {
                value
            },
            if i % 2 == 0 { -0.5 } else { 0.75 },
            0.25,
            -0.125,
        ]
    })
    .collect()
}

fn run(command: &mut Command, root: &Path, name: &str) -> Result<String, String> {
    let output = command.output().map_err(|e| e.to_string())?;
    fs::write(root.join(format!("{name}.stdout")), &output.stdout).unwrap();
    fs::write(root.join(format!("{name}.stderr")), &output.stderr).unwrap();
    if !output.status.success() {
        return Err(format!(
            "{name}: {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8(output.stdout).unwrap())
}

fn check_backend(
    program: &ScalarExpressionProgram,
    language: ScalarLanguage,
    cpp_compiler: Option<&str>,
    rows: &[Vec<f64>],
    dir: &Path,
) -> Result<Vec<u64>, String> {
    let recipe = program
        .emit_recipe(language, ScalarCarrier::Row, 4, 0.02)
        .map_err(|e| e.to_string())?;
    assert_eq!(
        recipe,
        program
            .emit_recipe(language, ScalarCarrier::Row, 4, 0.02)
            .unwrap(),
        "emission must be deterministic"
    );
    let words = rows
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    // The driver receives input words after compilation; emitted arithmetic is untouched.
    let input = words
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(dir.join("input.txt"), input).unwrap();
    let output = match language {
        ScalarLanguage::Rust => {
            let driver = "\nfn main(){let path=std::env::args().nth(1).unwrap();let text=std::fs::read_to_string(path).unwrap();let v=text.split_whitespace().map(|s|f64::from_bits(s.parse::<u64>().unwrap())).collect::<Vec<_>>();let rows=v.chunks(4).map(|r|r.to_vec()).collect::<Vec<_>>();let (values,faults)=scalar_stage1_values(&rows);for (row,fault) in values.iter().zip(faults){println!(\"{:016x}:{:08x}\",row[0].to_bits(),fault);}}\n";
            fs::write(
                dir.join("case.rs"),
                recipe.replace("\nfn main(", "\nfn recipe_main(") + driver,
            )
            .unwrap();
            run(
                Command::new("rustc").current_dir(dir).args([
                    "--edition=2021",
                    "-O",
                    "case.rs",
                    "-o",
                    "case-rust",
                ]),
                dir,
                "rust-compile",
            )?;
            run(
                Command::new(dir.join("case-rust")).arg(dir.join("input.txt")),
                dir,
                "rust-run",
            )?
        }
        ScalarLanguage::Python => {
            fs::write(dir.join("case.py"), recipe).unwrap();
            let driver = "import sys,struct; ns={'__name__':'gp_probe'}; exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),ns); words=[int(v) for v in open(sys.argv[2]).read().split()]; vals=[struct.unpack('<d',struct.pack('<Q',v))[0] for v in words]; rows=[vals[i:i+4] for i in range(0,len(vals),4)]; result,faults=ns['scalar_stage1_values'](rows); print('\\n'.join('%016x:%08x'%(struct.unpack('<Q',struct.pack('<d',r[0]))[0],f) for r,f in zip(result,faults)))";
            run(
                Command::new("python3")
                    .args(["-c", driver])
                    .arg(dir.join("case.py"))
                    .arg(dir.join("input.txt")),
                dir,
                "python-run",
            )?
        }
        ScalarLanguage::Cpp17 => {
            let compiler = cpp_compiler.expect("C++17 compiler selected explicitly");
            let driver = "\n#include <fstream>\n#include <cstdio>\nint main(int argc,char** argv){if(argc!=2)return 2;std::ifstream in(argv[1]);std::uint64_t bits;std::vector<std::vector<double>> rows;while(in>>bits){std::vector<double> row(4);std::memcpy(&row[0],&bits,8);for(int j=1;j<4;++j){if(!(in>>bits))return 3;std::memcpy(&row[j],&bits,8);}rows.push_back(row);}const auto out=scalar_stage1_values(rows);for(std::size_t i=0;i<out.first.size();++i){std::uint64_t value;std::memcpy(&value,&out.first[i][0],8);std::printf(\"%016llx:%08x\\n\",static_cast<unsigned long long>(value),static_cast<unsigned int>(out.second.at(i)));}}\n";
            fs::write(dir.join("case.cpp"), recipe + driver).unwrap();
            run(
                Command::new(compiler).current_dir(dir).args([
                    "-std=c++17",
                    "-Wall",
                    "-Werror",
                    "-O2",
                    "-ffp-contract=off",
                    "-fno-fast-math",
                    "case.cpp",
                    "-o",
                    "case-cpp",
                ]),
                dir,
                &format!("{compiler}-compile"),
            )?;
            run(
                Command::new(dir.join("case-cpp")).arg(dir.join("input.txt")),
                dir,
                &format!("{compiler}-run"),
            )?
        }
    };
    let records = output.lines().collect::<Vec<_>>();
    if records.len() != rows.len() {
        return Err(format!(
            "backend returned {} rows, expected {}",
            records.len(),
            rows.len()
        ));
    }
    records
        .iter()
        .zip(rows)
        .map(|(record, row)| {
            let (value, fault) = record.split_once(':').ok_or("missing fault word")?;
            let bits = u32::from_str_radix(fault, 16).map_err(|e| e.to_string())?;
            let expected = program.evaluate_row(row).faults.bits();
            if bits != expected {
                return Err(format!(
                    "fault bits {bits:08x} differ from evaluator {expected:08x}"
                ));
            }
            u64::from_str_radix(value, 16).map_err(|e| e.to_string())
        })
        .collect()
}

#[path = "gp_language_tricks/mod.rs"]
mod cases;
