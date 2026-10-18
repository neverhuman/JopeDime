use feat_math::{
    dsl::{ParamValue, Stage},
    gpr::parse_kernel,
    heads::{fit_head, FittedHead},
    target_transform::is_target_op,
    GenomeIr, Rows,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
struct BenchCase {
    id: String,
    prepared: PathBuf,
    prepared_sha256: String,
    cut: usize,
    genome: String,
    program_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchCases {
    cases: Vec<BenchCase>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn validate_cases_schema() -> BenchCases {
    let root = workspace_root();
    let path = root.join("benchmarks/gpr_cuda_bench/cases.json");
    let raw = fs::read(&path).expect("read benchmarks/gpr_cuda_bench/cases.json");
    let parsed: BenchCases = serde_json::from_slice(&raw).expect("parse cases.json");
    assert!(
        parsed.cases.len() >= 4,
        "cases.json must define at least 4 canonical GPR benchmark cases"
    );
    let mut ids = BTreeSet::new();
    for c in &parsed.cases {
        assert!(!c.id.trim().is_empty(), "empty case id");
        assert!(ids.insert(c.id.clone()), "duplicate case id {}", c.id);
        assert!(
            !c.prepared.as_os_str().is_empty(),
            "empty prepared path for {}",
            c.id
        );
        assert_eq!(
            c.prepared_sha256.len(),
            64,
            "prepared_sha256 must be 64-hex for {}",
            c.id
        );
        assert!(
            c.prepared_sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
            "non-hex prepared_sha256 for {}",
            c.id
        );
        assert_eq!(
            sha256_hex(c.genome.as_bytes()),
            c.program_sha256,
            "program_sha256 mismatch for {}",
            c.id
        );
        let GenomeIr::Pipeline(dsl) = GenomeIr::parse(&c.genome).expect("parse genome") else {
            panic!("case {} must be a pipeline genome", c.id);
        };
        let head = dsl.stages.last().expect("pipeline head");
        assert_eq!(head.op, "gpr", "case {} head must be gpr", c.id);
        for stage in &dsl.stages[..dsl.stages.len() - 1] {
            assert!(
                !is_target_op(&stage.op),
                "case {} includes target transform before gpr head",
                c.id
            );
        }
        let kernel = match head.params.get("kernel") {
            Some(ParamValue::Str(s)) => s.as_str(),
            _ => "rbf",
        };
        parse_kernel(kernel).expect("valid GPR kernel spec");
        assert_eq!(c.cut, 0, "canonical benchmark cut index must be 0");
    }
    parsed
}

fn verify_cpu_reference_envelope(cases: &BenchCases) {
    let n_train = 64usize;
    let n_val = 16usize;
    let d_feat = 16usize;
    let make_data = |n: usize, offset: usize| -> (Rows, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(d_feat);
            let mut target = 0.0f64;
            for j in 0..d_feat {
                let u = ((offset + i * 17 + j * 31) % 97) as f64 / 97.0 - 0.5;
                row.push(u);
                target += u * ((j % 3) as f64 - 0.5) * 0.4;
            }
            target += ((i as f64) * 0.15).sin() * 0.25;
            x.push(row);
            y.push(target);
        }
        (x, y)
    };
    let (x_train, y_train) = make_data(n_train, 0);
    let (x_val, y_val) = make_data(n_val, 1000);

    for c in &cases.cases {
        let GenomeIr::Pipeline(dsl) = GenomeIr::parse(&c.genome).expect("parse genome") else {
            unreachable!()
        };
        let mut head: Stage = dsl.stages.last().expect("head").clone();
        head.params.insert("m".to_string(), ParamValue::Int(64));
        head.params.insert("steps".to_string(), ParamValue::Int(12));
        let model = fit_head(&head, &x_train, &y_train).expect("fit CPU reference GPR head");
        match &model {
            FittedHead::GprKernel(m) => {
                assert!(!m.theta.is_empty() && m.theta.iter().all(|v| v.is_finite()));
            }
            FittedHead::Gpr(_) => {}
            _ => panic!("unexpected head variant for {}", c.id),
        }
        let pred = model
            .predict(&x_val)
            .expect("predict CPU reference GPR head");
        assert_eq!(pred.len(), n_val);
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "non-finite prediction for {}",
            c.id
        );
        let rmse = feat_math::rmse(&y_val, &pred);
        assert!(
            rmse.is_finite() && rmse < 5.0,
            "CPU reference RMSE out of envelope for {}: {rmse}",
            c.id
        );
    }
}

#[test]
fn gpr_cuda_bench_cases_json_valid() {
    let _ = validate_cases_schema();
}

#[test]
fn gpr_cuda_bench_cpu_reference_envelope_verified() {
    let cases = validate_cases_schema();
    verify_cpu_reference_envelope(&cases);
}

#[test]
fn validate_gpr_cuda_bench_cases_schema_and_cpu_reference() {
    let cases = validate_cases_schema();
    verify_cpu_reference_envelope(&cases);
}
