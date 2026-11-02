use feat_math::cuda::*;
use feat_math::kernel::registry_v2::OPERATORS;
use feat_math::transpile::{self, EmitOptions};
use feat_math::{fit_predict, GenomeIr};
use std::collections::BTreeSet;
use std::path::Path;

mod support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/cuda.rs"
    ));
}

fn data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let x = (0..32)
        .map(|i| {
            let v = i as f64 / 8.0 - 2.0;
            vec![v, (v * 0.7).sin(), (i % 5) as f64 - 2.0, v * v, 1.0]
        })
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| r[0] * 0.8 - r[1] * 0.3 + r[2] * 0.1)
        .collect();
    let probe = vec![x[3].clone(), x[17].clone(), vec![0.0; 5]];
    (x, y, probe)
}

fn recipe(root: &Path, source: &str) {
    let (x, y, probe) = data();
    let genome = GenomeIr::parse(source).unwrap();
    let expected = fit_predict(&genome, &x, &y, &probe).unwrap();
    let bundle = emit(&genome, &EmitOptions::default()).unwrap();
    support::run(
        root,
        &bundle,
        &serde_json::json!({"train":x,"targets":y,"test":probe}),
        &expected,
        false,
    );
}

/// This denominator comes from the same registry used to admit publication.
/// Every Rust recipe spelling either executes through the selected CUDA C++ build mode or
/// is explicitly refused by both emitters. New registry rows enter automatically.

include!("cuda_census/registry.rs");
include!("cuda_census/scalar.rs");
include!("cuda_census/fitted.rs");
include!("cuda_census/metadata.rs");
