//! Bounded, thread-local observations of the existing inference implementation.
//!
//! These records contain private model and candidate information. They are not a
//! public render package and do not establish product quality or output lineage.
use anyhow::{ensure, Context, Result};
use candle_core_0_11_0::Tensor;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

const MAX_RECORDS: usize = 128;
const MAX_VALUES: usize = 4_000_000;

thread_local! {
    static ACTIVE: RefCell<Option<NativeTrace>> = const { RefCell::new(None) };
}

const MAX_RUN_CALLS: usize = 64;
const MAX_RUN_VALUES: usize = 64_000_000;

/// Private observations made inside one actual production run, across its host threads.
#[derive(Debug, Serialize)]
pub struct RunJournal {
    pub schema: String,
    pub run_id: Option<String>,
    pub calls: Vec<RunCall>,
    pub recorded_values: usize,
    pub dropped_records: usize,
}

#[derive(Debug, Serialize)]
pub struct RunCall {
    pub call_id: usize,
    pub identity: serde_json::Value,
    pub outputs: serde_json::Value,
    pub trace: NativeTrace,
}

fn run_journal() -> &'static Mutex<Option<RunJournal>> {
    static JOURNAL: OnceLock<Mutex<Option<RunJournal>>> = OnceLock::new();
    JOURNAL.get_or_init(|| Mutex::new(None))
}

/// Explicit trusted-host opt-in; no environment or remote request enables this.
/// Only one journal may exist in a process. A second run identity is refused.
pub fn begin_run_journal() -> Result<RunJournalSession> {
    let mut journal = run_journal().lock().unwrap_or_else(|p| p.into_inner());
    ensure!(journal.is_none(), "a visual run journal is already active");
    *journal = Some(RunJournal {
        schema: "jope.actual-run-visual-journal/v1".into(),
        run_id: None,
        calls: Vec::new(),
        recorded_values: 0,
        dropped_records: 0,
    });
    Ok(RunJournalSession { active: true })
}

pub struct RunJournalSession {
    active: bool,
}

impl RunJournalSession {
    pub fn finish(mut self) -> Result<RunJournal> {
        let result = run_journal()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .context("visual run journal disappeared")?;
        self.active = false;
        ensure!(
            !result.calls.is_empty(),
            "no actual native calls were observed"
        );
        Ok(result)
    }
}

impl Drop for RunJournalSession {
    fn drop(&mut self) {
        if self.active {
            run_journal()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
        }
    }
}

/// Observe the existing synchronous operation at its real host-thread boundary.
/// The singleton lock prevents concurrent attribution; affine taps use only TLS.
pub(crate) fn observe_run_call<T>(
    identity: serde_json::Value,
    run: impl FnOnce() -> Result<T>,
    summarize: impl FnOnce(&T) -> serde_json::Value,
) -> Result<T> {
    let mut guard = run_journal().lock().unwrap_or_else(|p| p.into_inner());
    let Some(journal) = guard.as_mut() else {
        drop(guard);
        return run();
    };
    let id = identity["run_id"]
        .as_str()
        .context("observed call has no run identity")?;
    if let Some(bound) = &journal.run_id {
        ensure!(bound == id, "concurrent visual run attribution refused");
    } else {
        journal.run_id = Some(id.into());
    }
    ensure!(
        journal.calls.len() < MAX_RUN_CALLS,
        "visual run call budget exceeded"
    );
    let (result, trace) = capture(run)?;
    let values = journal
        .recorded_values
        .checked_add(trace.values)
        .context("run trace size overflow")?;
    ensure!(values <= MAX_RUN_VALUES, "visual run value budget exceeded");
    journal.calls.push(RunCall {
        call_id: journal.calls.len(),
        identity,
        outputs: summarize(&result),
        trace,
    });
    journal.recorded_values = values;
    Ok(result)
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
    fn actual_run_journal_observes_host_threads_and_refuses_other_run_identity() -> Result<()> {
        let session = begin_run_journal()?;
        let values = std::thread::spawn(|| {
            observe_run_call(
                serde_json::json!({"run_id":"run-software-control","phase":"test"}),
                || {
                    let x = Tensor::new(&[[2_f32, -3.]], &Device::Cpu)?;
                    let w = Tensor::new(&[[0.5_f32, 2.]], &Device::Cpu)?;
                    let y = x.matmul(&w.t()?.contiguous()?)?;
                    affine("test", "native_control", &x, &w, None, &y)?;
                    flat(&y)
                },
                |values| serde_json::json!({"values":values}),
            )
        })
        .join()
        .expect("control thread panicked")?;
        assert_eq!(values, vec![-5.]);
        assert!(observe_run_call(
            serde_json::json!({"run_id":"different-run"}),
            || -> Result<()> { panic!("mismatched run must refuse before execution") },
            |_| serde_json::Value::Null,
        )
        .is_err());
        let journal = session.finish()?;
        assert_eq!(journal.run_id.as_deref(), Some("run-software-control"));
        assert_eq!(journal.calls.len(), 1);
        assert_eq!(journal.calls[0].trace.affine.len(), 1);
        assert_eq!(journal.dropped_records, 0);
        assert!(journal.recorded_values > 0);
        Ok(())
    }

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
