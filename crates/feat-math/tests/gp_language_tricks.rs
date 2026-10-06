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
            let driver = "\nfn main(){let path=std::env::args().nth(1).unwrap();let text=std::fs::read_to_string(path).unwrap();let v=text.split_whitespace().map(|s|f64::from_bits(s.parse::<u64>().unwrap())).collect::<Vec<_>>();let rows=v.chunks(4).map(|r|r.to_vec()).collect::<Vec<_>>();let (values,_)=scalar_stage1_values(&rows);for row in values{println!(\"{:016x}\",row[0].to_bits());}}\n";
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
            let driver = "import sys,struct; ns={'__name__':'gp_probe'}; exec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec'),ns); words=[int(v) for v in open(sys.argv[2]).read().split()]; vals=[struct.unpack('<d',struct.pack('<Q',v))[0] for v in words]; rows=[vals[i:i+4] for i in range(0,len(vals),4)]; result,_=ns['scalar_stage1_values'](rows); print('\\n'.join('%016x'%struct.unpack('<Q',struct.pack('<d',r[0]))[0] for r in result))";
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
            let driver = "\n#include <fstream>\n#include <cstdio>\nint main(int argc,char** argv){if(argc!=2)return 2;std::ifstream in(argv[1]);std::uint64_t bits;std::vector<std::vector<double>> rows;while(in>>bits){std::vector<double> row(4);std::memcpy(&row[0],&bits,8);for(int j=1;j<4;++j){if(!(in>>bits))return 3;std::memcpy(&row[j],&bits,8);}rows.push_back(row);}const auto out=scalar_stage1_values(rows);for(const auto& row:out.first){std::uint64_t value;std::memcpy(&value,&row[0],8);std::printf(\"%016llx\\n\",static_cast<unsigned long long>(value));}}\n";
            fs::write(dir.join("case.cpp"), recipe + driver).unwrap();
            run(
                Command::new("g++").current_dir(dir).args([
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
                "cpp-compile",
            )?;
            run(
                Command::new(dir.join("case-cpp")).arg(dir.join("input.txt")),
                dir,
                "cpp-run",
            )?
        }
    };
    output
        .lines()
        .map(|s| u64::from_str_radix(s, 16).map_err(|e| e.to_string()))
        .collect()
}

#[test]
fn gp_tricks_compile_and_match_independent_references() {
    let scratch = tempfile::tempdir().unwrap();
    let root = std::env::var_os("JOPE_GP_REDTEAM_EVIDENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| scratch.path().to_owned());
    fs::create_dir_all(&root).unwrap();
    let mut failures = Vec::new();
    let mut matrix = Vec::new();
    for case in tricks() {
        let dir = root.join(case.name);
        fs::create_dir_all(&dir).unwrap();
        let (program, canonical) = match case.name {
            "glm_probit_link_v2" | "glm_logit_link_v2" => (
                ScalarExpressionProgram::new(ScalarNode::Call(
                    ScalarFunction::InverseLink(if case.name == "glm_probit_link_v2" {
                        InverseLink::Probit
                    } else {
                        InverseLink::Logit
                    }),
                    vec![ScalarNode::Var(Var::Col(0))],
                ))
                .unwrap(),
                case.tree.to_owned(),
            ),
            _ => {
                let expr = MathExpr::parse(case.tree).unwrap();
                let canonical = expr.to_source().unwrap();
                assert_eq!(
                    canonical,
                    MathExpr::parse(&canonical).unwrap().to_source().unwrap()
                );
                (ScalarExpressionProgram::from_v1(&expr).unwrap(), canonical)
            }
        };
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            bytes,
            ScalarExpressionProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let rows = probes(case.positive);
        let values = rows
            .iter()
            .map(|r| program.evaluate_row(r).value.unwrap())
            .collect::<Vec<_>>();
        let words = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let references = rows.iter().map(|r| (case.reference)(r)).collect::<Vec<_>>();
        for (i, (value, reference)) in values.iter().zip(&references).enumerate() {
            // Independent algebraic forms differ in association. This bounded mathematical
            // check is separate from mandatory exact evaluator/backend word comparisons.
            let bound = if case.name == "glm_probit_link_v2" {
                1.2e-9 * reference.abs()
            } else {
                5e-14 * reference.abs().max(1.0)
            };
            if (value - reference).abs() > bound {
                failures.push(format!(
                    "{} reference row{i}: {value:?} != {reference:?}",
                    case.name
                ));
            }
        }
        let mut backends = Vec::new();
        for lang in [
            ScalarLanguage::Python,
            ScalarLanguage::Rust,
            ScalarLanguage::Cpp17,
        ] {
            match check_backend(&program, lang, &rows, &dir) {
                Ok(actual) if actual == words => backends
                    .push(json!({"language":format!("{lang:?}"),"status":"PASS","words":actual})),
                Ok(actual) => {
                    failures.push(format!(
                        "{} {lang:?} bits differ: {actual:?} != {words:?}",
                        case.name
                    ));
                    backends.push(json!({"language":format!("{lang:?}"),"status":"BITS_MISMATCH","words":actual}));
                }
                Err(error) => {
                    failures.push(format!("{} {lang:?}: {error}", case.name));
                    backends.push(
                        json!({"language":format!("{lang:?}"),"status":"BACKEND_FAILED","error":error}),
                    );
                }
            }
        }
        matrix.push(json!({"trick":case.name,"tree":case.tree,"canonical":canonical,"rows":rows,"reference":references,"native_words":words,"backends":backends}));
    }
    fs::write(
        root.join("matrix.json"),
        serde_json::to_vec_pretty(&json!({"matrix":matrix,"failures":failures})).unwrap(),
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "GP language gaps (raw evidence {}):\n{}",
        root.display(),
        failures.join("\n")
    );
}

#[test]
fn gp_language_adversarial_boundaries_are_explicit() {
    use feat_math::math_expr::FaultSet;
    // These are contract boundaries, not successful mathematical witnesses.
    // Arithmetic selection evaluates both branches and cannot hide a hard refusal.
    let eager = MathExpr::parse("(1-step(c0,0))*2+step(c0,0)*clip(c1,c2,c3)").unwrap();
    let selected = ScalarExpressionProgram::from_v1(&eager).unwrap();
    let result = selected.evaluate_row(&[-1.0, 3.0, f64::NAN, f64::NAN]);
    assert!(result.value.is_err());
    assert!(result.faults.contains(FaultSet::INVALID_CLIP_BOUNDS));

    // A ninth-degree polynomial is expressible, but a guarded pow node is not
    // a substitute for it. Multiplication retains the intended degree.
    let guarded = ScalarExpressionProgram::from_v1(&MathExpr::parse("c0^9").unwrap()).unwrap();
    let result = guarded.evaluate_row(&[2.0]);
    assert_eq!(result.value.unwrap(), 256.0);
    assert!(result.faults.contains(FaultSet::POW_EXP_CLAMPED));
    let ninth = MathExpr::parse("c0*c0*c0*c0*c0*c0*c0*c0*c0").unwrap();
    assert_eq!(ninth.eval_row(&[2.0]).unwrap(), 512.0);

    // Stabilized log-sum-exp still inherits the historical exp argument clamp.
    let lse = MathExpr::parse("max(c0,c1)+log(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))").unwrap();
    let program = ScalarExpressionProgram::from_v1(&lse).unwrap();
    let result = program.evaluate_row(&[0.0, -100.0]);
    assert!(result.value.unwrap() > 0.0);
    assert!(result.faults.contains(FaultSet::EXP_ARG_CLAMPED));
    assert_eq!((-100.0f64).exp().ln_1p(), 3.720075976020836e-44);

    // Resource refusals remain explicit; large ensembles are not silently trimmed.
    let large = std::iter::repeat_n("c0", 65).collect::<Vec<_>>().join("+");
    assert!(MathExpr::parse(&large).is_err());
    assert!(MathExpr::parse("where(c0,c1,c2)").is_err());
}
