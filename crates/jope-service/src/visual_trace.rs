//! Bounded, thread-local observations of the existing inference implementation.
//!
//! These records contain private model and candidate information. They are not a
//! public render package and do not establish product quality or output lineage.
use anyhow::{ensure, Context, Result};
use candle_core_0_11_0::Tensor;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

const MAX_RECORDS: usize = 128;
const MAX_VALUES: usize = 4_000_000;

thread_local! {
    static ACTIVE: RefCell<Option<NativeTrace>> = const { RefCell::new(None) };
}

/// Captured affine operation. All arrays are row-major; weights are `[out,in]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AffineTrace {
    pub sequence: usize,
    pub network: String,
    pub name: String,
    pub weight_layout: String,
    pub input_shape: Vec<usize>,
    pub weight_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub input: Vec<f32>,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub output: Vec<f32>,
    pub closure_max_normalized_error: f64,
}

/// A native non-affine readout, without an invented edge interpretation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorTrace {
    pub sequence: usize,
    pub network: String,
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// One synchronous inference invocation's private observations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrace {
    pub schema: String,
    pub affine: Vec<AffineTrace>,
    pub tensors: Vec<TensorTrace>,
    #[serde(skip)]
    values: usize,
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

/// Observe a synchronous native call without changing its inputs or parameters.
/// Nested captures are refused. Errors and unwinding discard incomplete records.
pub fn capture<T>(run: impl FnOnce() -> Result<T>) -> Result<(T, NativeTrace)> {
    ACTIVE.with(|active| {
        let mut state = active.borrow_mut();
        ensure!(state.is_none(), "nested visual capture is unsupported");
        *state = Some(NativeTrace {
            schema: "jope.native-affine-trace/v1".into(),
            ..NativeTrace::default()
        });
        Ok(())
    })?;
    let _reset = Reset;
    let result = run()?;
    let trace = ACTIVE
        .with(|active| active.borrow_mut().take())
        .context("capture state lost")?;
    ensure!(
        !trace.affine.is_empty(),
        "capture contains no affine observations"
    );
    Ok((result, trace))
}

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    let values = t.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite visual observation"
    );
    Ok(values)
}

fn reserve(trace: &mut NativeTrace, count: usize) -> Result<usize> {
    let sequence = trace.affine.len() + trace.tensors.len();
    ensure!(sequence < MAX_RECORDS, "visual record budget exceeded");
    let values = trace
        .values
        .checked_add(count)
        .context("visual value count overflow")?;
    ensure!(values <= MAX_VALUES, "visual value budget exceeded");
    trace.values = values;
    Ok(sequence)
}

/// Check all captured output rows against independently summed weighted terms.
pub fn validate_affine(record: &AffineTrace) -> Result<f64> {
    ensure!(
        record.weight_shape.len() == 2,
        "affine weights must be a matrix"
    );
    let (outputs, inputs) = (record.weight_shape[0], record.weight_shape[1]);
    ensure!(inputs > 0 && outputs > 0, "empty affine dimension");
    ensure!(
        record.weight_layout == "out_in",
        "unsupported affine weight layout"
    );
    ensure!(
        record.input_shape.last() == Some(&inputs),
        "affine input width differs"
    );
    ensure!(
        record.output_shape.last() == Some(&outputs),
        "affine output width differs"
    );
    let size = |shape: &[usize]| shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    ensure!(
        size(&record.input_shape) == Some(record.input.len()),
        "affine input shape differs"
    );
    ensure!(
        size(&record.output_shape) == Some(record.output.len()),
        "affine output shape differs"
    );
    ensure!(
        record.weights.len() == inputs * outputs && record.bias.len() == outputs,
        "affine parameter shape differs"
    );
    ensure!(
        record.input.len() / inputs == record.output.len() / outputs,
        "affine row count differs"
    );
    ensure!(
        record
            .input
            .iter()
            .chain(&record.weights)
            .chain(&record.bias)
            .chain(&record.output)
            .all(|v| v.is_finite()),
        "nonfinite affine observation"
    );
    let mut worst = 0.0_f64;
    for (x, y) in record
        .input
        .chunks_exact(inputs)
        .zip(record.output.chunks_exact(outputs))
    {
        for (j, value) in y.iter().enumerate() {
            let mut sum = f64::from(record.bias[j]);
            let mut magnitude = sum.abs();
            for (a, w) in x.iter().zip(&record.weights[j * inputs..(j + 1) * inputs]) {
                let term = f64::from(*a) * f64::from(*w);
                sum += term;
                magnitude += term.abs();
            }
            let error = (sum - f64::from(*value)).abs() / (magnitude + 1e-9);
            worst = worst.max(error);
        }
    }
    ensure!(
        worst <= 1e-5,
        "affine closure failed for {}: {worst:e}",
        record.name
    );
    Ok(worst)
}

pub(crate) fn affine(
    network: &str,
    name: &str,
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    y: &Tensor,
) -> Result<()> {
    ACTIVE.with(|active| {
        let mut state = active.borrow_mut();
        let Some(trace) = state.as_mut() else {
            return Ok(());
        };
        let out = w.dim(0)?;
        let sequence = reserve(
            trace,
            x.elem_count() + w.elem_count() + out + y.elem_count(),
        )?;
        let mut record = AffineTrace {
            sequence,
            network: network.into(),
            name: name.into(),
            weight_layout: "out_in".into(),
            input_shape: x.dims().to_vec(),
            weight_shape: w.dims().to_vec(),
            output_shape: y.dims().to_vec(),
            input: flat(x)?,
            weights: flat(w)?,
            bias: match b {
                Some(b) => flat(b)?,
                None => vec![0.; out],
            },
            output: flat(y)?,
            closure_max_normalized_error: 0.,
        };
        record.closure_max_normalized_error = validate_affine(&record)?;
        trace.affine.push(record);
        Ok(())
    })
}

pub(crate) fn tensor(network: &str, name: &str, value: &Tensor) -> Result<()> {
    ACTIVE.with(|active| {
        let mut state = active.borrow_mut();
        let Some(trace) = state.as_mut() else {
            return Ok(());
        };
        let sequence = reserve(trace, value.elem_count())?;
        trace.tensors.push(TensorTrace {
            sequence,
            network: network.into(),
            name: name.into(),
            shape: value.dims().to_vec(),
            values: flat(value)?,
        });
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core_0_11_0::Device;

    #[test]
    fn native_affine_capture_checks_closure_and_is_observation_only() -> Result<()> {
        let x = Tensor::new(&[[2_f32, -3.]], &Device::Cpu)?;
        let w = Tensor::new(&[[0.5_f32, 2.], [-1., 0.25]], &Device::Cpu)?;
        let b = Tensor::new(&[1_f32, -0.5], &Device::Cpu)?;
        let run = || -> Result<Tensor> {
            let y = x.matmul(&w.t()?.contiguous()?)?.broadcast_add(&b)?;
            affine("dope", "sentinel", &x, &w, Some(&b), &y)?;
            Ok(y)
        };
        let reference = flat(&run()?)?;
        let (observed, trace) = capture(run)?;
        assert_eq!(flat(&observed)?, reference);
        assert_eq!(trace.affine.len(), 1);
        let mut bad = trace.affine[0].clone();
        bad.output[0] += 0.1;
        assert!(validate_affine(&bad).is_err());
        Ok(())
    }

    #[test]
    fn failed_and_nested_capture_leave_no_active_trace() {
        assert!(capture::<()>(|| anyhow::bail!("write failed")).is_err());
        assert!(capture::<()>(|| {
            assert!(capture::<()>(|| Ok(())).is_err());
            anyhow::bail!("end")
        })
        .is_err());
        ACTIVE.with(|active| assert!(active.borrow().is_none()));
    }
}
