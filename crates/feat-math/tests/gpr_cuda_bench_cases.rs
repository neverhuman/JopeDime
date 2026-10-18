#[path = "../../jope-engine/examples/gpr_cuda_bench_args.rs"]
mod bench_args;

use std::path::Path;

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
