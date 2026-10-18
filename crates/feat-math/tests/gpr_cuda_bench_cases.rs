#[path = "../../jope-engine/src/gpr_cuda_args.rs"]
mod bench_args;

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
struct S3Provenance {
    dataset_family: String,
    s3_parquet_sha256: String,
    s3_curation_receipt_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchCase {
    id: String,
    prepared: PathBuf,
    prepared_sha256: String,
    cut: usize,
    #[serde(default)]
    train_rows: Option<usize>,
    #[serde(default)]
    feature_columns: Option<usize>,
    genome: String,
    program_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchCases {
    #[serde(default)]
    matern_dropped: Option<bool>,
    #[serde(default)]
    matern_dropped_disclosure: Option<String>,
    #[serde(default)]
    s3_provenance: Option<S3Provenance>,
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

fn is_64_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn validate_cases_schema() -> BenchCases {
    let root = workspace_root();
    let path = root.join("benchmarks/gpr_cuda_bench/cases.json");
    let raw = fs::read(&path).expect("read benchmarks/gpr_cuda_bench/cases.json");
    let parsed: BenchCases = serde_json::from_slice(&raw).expect("parse cases.json");
    assert!(
        parsed.cases.len() >= 9,
        "cases.json must define the 9-cell N x D GPR benchmark grid"
    );
    assert_eq!(
        parsed.matern_dropped,
        Some(true),
        "cases.json must explicitly disclose matern_dropped=true"
    );
    assert!(
        parsed
            .matern_dropped_disclosure
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty()),
        "cases.json must include non-empty matern_dropped_disclosure"
    );
    let prov = parsed
        .s3_provenance
        .as_ref()
        .expect("cases.json must record S3 dataset provenance");
    assert!(!prov.dataset_family.trim().is_empty());
    assert!(
        is_64_hex(&prov.s3_parquet_sha256),
        "s3_parquet_sha256 must be 64-hex"
    );
    assert!(
        is_64_hex(&prov.s3_curation_receipt_sha256),
        "s3_curation_receipt_sha256 must be 64-hex"
    );

    let mut ids = BTreeSet::new();
    let mut grid_cells = BTreeSet::new();
    for c in &parsed.cases {
        assert!(!c.id.trim().is_empty(), "empty case id");
        assert!(ids.insert(c.id.clone()), "duplicate case id {}", c.id);
        assert!(
            !c.prepared.as_os_str().is_empty(),
            "empty prepared path for {}",
            c.id
        );
        assert!(
            is_64_hex(&c.prepared_sha256),
            "prepared_sha256 must be 64-hex for {}",
            c.id
        );
        assert_eq!(
            sha256_hex(c.genome.as_bytes()),
            c.program_sha256,
            "program_sha256 mismatch for {}",
            c.id
        );
        if let (Some(n), Some(d)) = (c.train_rows, c.feature_columns) {
            grid_cells.insert((n, d));
        }
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

    for n in [1024usize, 4096, 16384] {
        for d in [16usize, 64, 256] {
            assert!(
                grid_cells.contains(&(n, d)),
                "cases.json missing (N={n}, D={d}) grid cell"
            );
        }
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

#[test]
fn benchmark_without_reference_is_refused_before_preparation() {
    let err = bench_args::require_reference_for_benchmark(false, None).unwrap_err();
    assert!(err.contains("--reference is required"), "{err}");
}

#[test]
fn benchmark_with_reference_or_reference_only_is_accepted() {
    bench_args::require_reference_for_benchmark(false, Some(Path::new("reference.json"))).unwrap();
    bench_args::require_reference_for_benchmark(true, None).unwrap();
}

#[test]
fn verify_benchmark_envelope_enforces_parity_and_timing_bounds() {
    let ok_rows = vec![bench_args::BenchmarkRowSummary {
        id: "case_00_n1024_d16_rbf".into(),
        status: "parity_pass".into(),
        max_absolute_prediction_error: Some(1.2e-7),
        prediction_rmse_error: Some(4.5e-8),
        amortized_backend_seconds: Some(0.042),
        end_to_end_seconds: Some(0.055),
    }];
    bench_args::verify_benchmark_envelope(true, false, 1, &ok_rows, 1e-4).unwrap();
    bench_args::verify_benchmark_envelope(false, false, 1, &[], 1e-4).unwrap();

    let ref_only_err =
        bench_args::verify_benchmark_envelope(true, true, 1, &ok_rows, 1e-4).unwrap_err();
    assert!(ref_only_err.contains("--reference-only"), "{ref_only_err}");

    let mut parity_fail = ok_rows.clone();
    parity_fail[0].status = "parity_fail".into();
    let err =
        bench_args::verify_benchmark_envelope(true, false, 1, &parity_fail, 1e-4).unwrap_err();
    assert!(err.contains("parity_fail"), "{err}");

    let mut timing_fail = ok_rows;
    timing_fail[0].amortized_backend_seconds = Some(0.0);
    let err2 =
        bench_args::verify_benchmark_envelope(true, false, 1, &timing_fail, 1e-4).unwrap_err();
    assert!(err2.contains("timing"), "{err2}");
}
