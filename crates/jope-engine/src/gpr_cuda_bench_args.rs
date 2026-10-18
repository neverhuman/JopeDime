//! Argument and envelope checks for gpr_cuda_bench that run without LibTorch so regression tests
//! can compile and exercise them without the neural feature.
#![allow(dead_code)]

use std::path::Path;

/// A benchmark run must name the reference bound to its cases. Refuse before preparing any case.
pub fn require_reference_for_benchmark(
    reference_only: bool,
    reference: Option<&Path>,
) -> Result<(), String> {
    if !reference_only && reference.is_none() {
        return Err(
            "--reference is required for a benchmark: build one first with --reference-only --device cpu on the same --cases file"
                .into(),
        );
    }
    Ok(())
}

/// Summary of one benchmark case row for `--verify-envelope` enforcement.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRowSummary {
    pub id: String,
    pub status: String,
    pub max_absolute_prediction_error: Option<f64>,
    pub prediction_rmse_error: Option<f64>,
    pub amortized_backend_seconds: Option<f64>,
    pub end_to_end_seconds: Option<f64>,
}

/// Verify that all executed benchmark rows passed CPU-vs-CUDA accuracy parity and produced finite
/// positive timing measurements when `--verify-envelope` is active.
pub fn verify_benchmark_envelope(
    verify_envelope: bool,
    reference_only: bool,
    expected_cases: usize,
    rows: &[BenchmarkRowSummary],
    max_allowed_rmse_error: f64,
) -> Result<(), String> {
    if !verify_envelope {
        return Ok(());
    }
    if reference_only {
        return Err("--verify-envelope cannot be combined with --reference-only".into());
    }
    if expected_cases == 0 || rows.len() != expected_cases {
        return Err(format!(
            "--verify-envelope expected {expected_cases} completed benchmark rows, got {}",
            rows.len()
        ));
    }
    for row in rows {
        if row.status != "parity_pass" {
            return Err(format!(
                "--verify-envelope failed for case {}: status={}",
                row.id, row.status
            ));
        }
        let max_abs = row.max_absolute_prediction_error.ok_or_else(|| {
            format!(
                "--verify-envelope missing max_absolute_prediction_error for case {}",
                row.id
            )
        })?;
        let rmse_err = row.prediction_rmse_error.ok_or_else(|| {
            format!(
                "--verify-envelope missing prediction_rmse_error for case {}",
                row.id
            )
        })?;
        let backend_s = row.amortized_backend_seconds.ok_or_else(|| {
            format!(
                "--verify-envelope missing amortized_backend_seconds for case {}",
                row.id
            )
        })?;
        let e2e_s = row.end_to_end_seconds.ok_or_else(|| {
            format!(
                "--verify-envelope missing end_to_end_seconds for case {}",
                row.id
            )
        })?;
        if !max_abs.is_finite() || !rmse_err.is_finite() || rmse_err > max_allowed_rmse_error {
            return Err(format!(
                "--verify-envelope accuracy out of bounds for case {}: max_abs={max_abs}, rmse_err={rmse_err} (limit={max_allowed_rmse_error})",
                row.id
            ));
        }
        if !backend_s.is_finite() || backend_s <= 0.0 || !e2e_s.is_finite() || e2e_s <= 0.0 {
            return Err(format!(
                "--verify-envelope non-positive or non-finite timing for case {}: backend_s={backend_s}, e2e_s={e2e_s}",
                row.id
            ));
        }
    }
    Ok(())
}
