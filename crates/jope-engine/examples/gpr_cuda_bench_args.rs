//! Argument checks for gpr_cuda_bench that run before any case is prepared. Kept free of
//! LibTorch so the regression test can compile it without the neural feature.

/// A benchmark run must name the reference bound to its cases. Refuse before preparing any case.
pub fn require_reference_for_benchmark(
    reference_only: bool,
    reference: Option<&std::path::Path>,
) -> Result<(), String> {
    if !reference_only && reference.is_none() {
        return Err(
            "--reference is required for a benchmark: build one first with --reference-only --device cpu on the same --cases file"
                .into(),
        );
    }
    Ok(())
}
