use std::collections::BTreeMap;
use std::process::Command;

use feat_math::GenomeIr;
use jope_runtime::math_pipeline::{
    InputContract, MathPipeline, NumericColumn, PipelineState, TargetScale, INPUT_PART,
    PIPELINE_PART,
};

fn build_pipeline(dsl: &str) -> (MathPipeline, Vec<Vec<f64>>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..48 {
        let a = (i as f64) / 47.0;
        let b = ((i * 7 % 48) as f64) / 47.0;
        let c = ((i * 13 % 48) as f64) / 47.0;
        let target = 0.25 + 0.45 * a - 0.2 * b + 0.15 * (a * c).sin();
        x.push(vec![a, b, c]);
        y.push(target);
    }
    let genome = GenomeIr::parse(dsl).expect("parse DSL");
    let fitted = feat_math::fit(&genome, &x, &y).expect("fit model");
    let input = InputContract::new(
        vec![
            NumericColumn {
                name: "x0".into(),
                min: 0.0,
                max: 10.0,
                constant: false,
            },
            NumericColumn {
                name: "x1".into(),
                min: -5.0,
                max: 5.0,
                constant: false,
            },
            NumericColumn {
                name: "x2".into(),
                min: 100.0,
                max: 200.0,
                constant: false,
            },
        ],
        vec![0, 1, 2],
        TargetScale {
            name: "y".into(),
            min: 10.0,
            scale: 50.0,
        },
    )
    .expect("input contract");
    let parts = BTreeMap::from([
        (
            PIPELINE_PART.to_owned(),
            serde_json::to_vec(&PipelineState {
                dsl: genome.canonical(),
                model: fitted,
            })
            .unwrap(),
        ),
        (INPUT_PART.to_owned(), serde_json::to_vec(&input).unwrap()),
    ]);
    let raw_query = vec![
        vec![1.5, -2.0, 125.0],
        vec![7.25, 3.5, 180.0],
        vec![0.0, 0.0, 100.0],
        vec![10.0, 5.0, 200.0],
    ];
    (
        MathPipeline::decode(&parts, PIPELINE_PART).expect("decode MathPipeline"),
        raw_query,
    )
}

#[test]
fn js_and_cpp17_emitters_match_rust_for_gbm_histgbm2_blend_and_kernels() {
    let dsls = [
        "(pipe X (winsor_clip q=0.1) (cheb degree=3) (histgbm rounds=12 depth=3 lr=0.15))",
        "(pipe X (standardize) (histgbm2 rounds=10 leaves=5 lr=0.12 bags=2))",
        "(pipe X (robust_scale) (ensemble rounds=10 depth=3 lr=0.15))",
        "(pipe X (rank) (ridge_gcv))",
        "(pipe X (standardize) (krr m=24))",
        "(pipe X (standardize) (gpr m=20 steps=4))",
        "(pipe X (standardize) (gpr m=20 steps=4 kernel=\"rbf+rq\"))",
        "(pipe X (standardize) (ksmooth k=5 gamma=1.5))",
        "(pipe X (standardize) (stack))",
    ];

    let tmp = tempfile::tempdir().expect("tempdir");
    let node_bin = Command::new("node")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| "node");
    let gpp_bin = Command::new("g++")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| "g++");

    for (idx, dsl) in dsls.iter().enumerate() {
        let (pipeline, raw_rows) = build_pipeline(dsl);
        let str_rows: Vec<Vec<String>> = raw_rows
            .iter()
            .map(|r| r.iter().map(|v| format!("{v:?}")).collect())
            .collect();
        let opt_rows: Vec<Vec<Option<&str>>> = str_rows
            .iter()
            .map(|r| r.iter().map(|s| Some(s.as_str())).collect())
            .collect();
        let expected = pipeline.predict_cells(&opt_rows).expect("rust predict");

        let js_bytes = pipeline
            .export_javascript()
            .unwrap_or_else(|e| panic!("JS export failed for {dsl}: {e:#}"));
        let js_src = String::from_utf8(js_bytes).expect("utf8 js");
        if let Some(node) = node_bin {
            let js_path = tmp.path().join(format!("m_{idx}.js"));
            let mut harness = js_src;
            harness.push_str(&format!(
                "\nconsole.log(JSON.stringify(predictRows({})));\n",
                serde_json::to_string(&raw_rows).unwrap()
            ));
            std::fs::write(&js_path, harness).unwrap();
            let out = Command::new(node).arg(&js_path).output().unwrap();
            assert!(
                out.status.success(),
                "node failed for {dsl}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let actual_js: Vec<f64> = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(actual_js.len(), expected.len());
            for (a, e) in actual_js.iter().zip(&expected) {
                assert!(
                    (a - e).abs() <= 1e-9,
                    "JS mismatch on {dsl}: got {a}, expected {e}"
                );
            }
        }

        let cpp_src = pipeline
            .export_cpp17(None, None)
            .unwrap_or_else(|e| panic!("C++17 export failed for {dsl}: {e:#}"));
        if let Some(gpp) = gpp_bin {
            let cpp_path = tmp.path().join(format!("m_{idx}.cpp"));
            let bin_path = tmp.path().join(format!("m_{idx}_bin"));
            let rows_cpp: Vec<String> = raw_rows
                .iter()
                .map(|r| {
                    let inner: Vec<String> = r.iter().map(|v| format!("{v:?}")).collect();
                    format!("{{{}}}", inner.join(","))
                })
                .collect();
            let mut harness = cpp_src;
            harness.push_str(&format!(
                r#"
#include <iostream>
#include <iomanip>
int main() {{
    std::vector<std::vector<double>> q = {{{}}};
    auto p = predict_rows(q);
    std::cout << std::setprecision(17);
    for (std::size_t i = 0; i < p.size(); ++i) {{
        if (i > 0) std::cout << ",";
        std::cout << p[i];
    }}
    std::cout << "\n";
    return 0;
}}
"#,
                rows_cpp.join(",")
            ));
            std::fs::write(&cpp_path, harness).unwrap();
            let comp = Command::new(gpp)
                .args(["-std=c++17", "-O1"])
                .arg(&cpp_path)
                .arg("-o")
                .arg(&bin_path)
                .output()
                .unwrap();
            assert!(
                comp.status.success(),
                "g++ compile failed for {dsl}: {}",
                String::from_utf8_lossy(&comp.stderr)
            );
            let run = Command::new(&bin_path).output().unwrap();
            assert!(run.status.success(), "c++ binary failed for {dsl}");
            let actual_cpp: Vec<f64> = String::from_utf8_lossy(&run.stdout)
                .trim()
                .split(',')
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            assert_eq!(actual_cpp.len(), expected.len());
            for (a, e) in actual_cpp.iter().zip(&expected) {
                assert!(
                    (a - e).abs() <= 1e-9,
                    "C++17 mismatch on {dsl}: got {a}, expected {e}"
                );
            }
        }
    }
}
