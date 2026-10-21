//! battle-gpu: raw-CUDA-FFI batched-ridge backend for the feat-math rung-0
//! screening seam (`feat_math::evolve::GpuRidgeScoreBatchFn`).
//!
//! Contract:
//! - `gpu_ridge_score_batch` is order-preserving and TOTAL over well-formed
//!   problems: GPU-ineligible widths or row counts are scored on the CPU
//!   with the exact reference math, malformed shapes return `Err` (the seam
//!   then falls back to its exact CPU path), and nothing here ever panics on
//!   driver failure.
//! - Scores are BIT-IDENTICAL to the CPU reference (`RidgeModel::fit` +
//!   `metrics::r2` after f32->f64 widening): the kernels are compiled with
//!   `-fmad=false` and every reduction is a per-thread sequential loop in the
//!   reference order. Determinism across runs on fixed hardware is therefore
//!   trivial, and cross-backend replay only needs score bits, which match.
//! - No CUDA crates and no build-time CUDA: the committed PTX is embedded and
//!   `libcuda.so.1` is dlopen'd at first use. Hosts without a driver probe as
//!   `None` and every call returns `Err`.

mod driver;

use driver::{Api, CUfunction, DeviceBuffer, Library};
pub use feat_math::evolve::{
    GpuRidgeScoreBatch, GpuRidgeScoreFoldBytes, GpuRidgeScoreProblem, GpuRidgeScoreTimings,
};
use feat_math::heads::RidgeModel;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Widest problem the on-device Cholesky handles; wider problems are scored
/// with the CPU reference inside this crate (still order-preserving).
/// Mirrored by `feat-math`'s `GPU_SCREEN_MAX_P` comment.
pub const MAX_RIDGE_P: usize = 96;

/// SHA-256 of the committed PTX (see scripts/build_ptx.sh); recorded so
/// screen receipts can pin the exact kernel image.
pub const PTX_SHA256: &str = "124f062f84bcb406cd36d4a3ade17f39d7ac0fe162ae62bdde325e94c9b715b5";

const PTX: &str = include_str!("../kernels/ridge_screen.ptx");

/// Fraction of currently-free device memory one wave may claim.
const WAVE_MEM_FRACTION: f64 = 0.7;
/// Fallback wave budget when cuMemGetInfo is unavailable.
const WAVE_MEM_FALLBACK_BYTES: usize = 2 << 30;

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub driver_version: i32,
    pub total_mem_bytes: usize,
    pub multiprocessors: i32,
}

struct Gpu {
    _lib: Library,
    api: Api,
    ctx: driver::CUcontext,
    stream: driver::CUstream,
    f_col_stats: CUfunction,
    f_gram_atb: CUfunction,
    f_chol_solve: CUfunction,
    f_predict_r2: CUfunction,
    info: GpuInfo,
}

// SAFETY: the context, stream and function handles are opaque driver tokens that CUDA allows
// any thread to use once the context is made current (every use calls cuCtxSetCurrent first).
// All access is serialized through the backend Mutex; raw handles never leave it.
unsafe impl Send for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Release the stream before its owning context, while the driver library
        // is still loaded. The process-lifetime BACKEND retains its usual lifetime.
        // SAFETY: `ctx` and `stream` were created by this Gpu's init and are destroyed exactly
        // once here, stream before context; `_lib` keeps libcuda loaded until after this body.
        unsafe {
            (self.api.cuCtxSetCurrent)(self.ctx);
            (self.api.cuStreamDestroy)(self.stream);
            (self.api.cuCtxDestroy_v2)(self.ctx);
        }
    }
}

static BACKEND: OnceLock<Result<Mutex<Gpu>, String>> = OnceLock::new();

fn backend() -> Result<&'static Mutex<Gpu>, String> {
    BACKEND
        .get_or_init(init_backend)
        .as_ref()
        .map_err(Clone::clone)
}

fn init_backend() -> Result<Mutex<Gpu>, String> {
    let lib = Library::open("libcuda.so.1")?;
    let api = lib.api()?;
    // SAFETY: cuInit takes flags by value (0 is the only defined value).
    api.check(unsafe { (api.cuInit)(0) }, "cuInit")?;
    let mut driver_version: c_int = 0;
    api.check(
        // SAFETY: `driver_version` is a live out-parameter.
        unsafe { (api.cuDriverGetVersion)(&mut driver_version) },
        "cuDriverGetVersion",
    )?;
    let mut device: driver::CUdevice = 0;
    // SAFETY: `device` is a live out-parameter; ordinal 0 is validated by the driver.
    api.check(unsafe { (api.cuDeviceGet)(&mut device, 0) }, "cuDeviceGet")?;
    let mut name_buf = [0 as c_char; 128];
    api.check(
        // SAFETY: the driver writes at most `len` (= the buffer's length) bytes into the
        // live local buffer.
        unsafe { (api.cuDeviceGetName)(name_buf.as_mut_ptr(), name_buf.len() as c_int, device) },
        "cuDeviceGetName",
    )?;
    // Parse within the buffer's bounds, so a driver that fills it without a NUL cannot cause an
    // out-of-bounds read (the previous CStr::from_ptr assumed termination).
    let name_bytes = name_buf.map(|c| c as u8);
    let name = match std::ffi::CStr::from_bytes_until_nul(&name_bytes) {
        Ok(name) => name.to_string_lossy().into_owned(),
        Err(_) => String::from_utf8_lossy(&name_bytes).into_owned(),
    };
    let mut multiprocessors: c_int = 0;
    api.check(
        // SAFETY: `multiprocessors` is a live out-parameter; attribute and device are values.
        unsafe {
            (api.cuDeviceGetAttribute)(
                &mut multiprocessors,
                driver::ATTR_MULTIPROCESSOR_COUNT,
                device,
            )
        },
        "cuDeviceGetAttribute",
    )?;
    let mut total_mem: usize = 0;
    api.check(
        // SAFETY: `total_mem` is a live out-parameter.
        unsafe { (api.cuDeviceTotalMem_v2)(&mut total_mem, device) },
        "cuDeviceTotalMem",
    )?;
    let mut ctx: driver::CUcontext = std::ptr::null_mut();
    api.check(
        // SAFETY: `ctx` is a live out-parameter; it is destroyed on every later failure path.
        unsafe { (api.cuCtxCreate_v2)(&mut ctx, 0, device) },
        "cuCtxCreate",
    )?;
    // From here on, failures must release the context before returning.
    let built = (|| -> Result<(driver::CUmodule, [CUfunction; 4], driver::CUstream), String> {
        let image = CString::new(PTX).map_err(|_| "PTX contains NUL".to_string())?;
        let mut module: driver::CUmodule = std::ptr::null_mut();
        api.check(
            // SAFETY: `image` is a NUL-terminated PTX string alive for the call; `module` is a
            // live out-parameter. The driver copies the image.
            unsafe { (api.cuModuleLoadData)(&mut module, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        let function = |name: &str| -> Result<CUfunction, String> {
            let cname = CString::new(name).unwrap();
            let mut f: CUfunction = std::ptr::null_mut();
            api.check(
                // SAFETY: `module` was loaded above, `cname` is NUL-terminated and alive for
                // the call, and `f` is a live out-parameter.
                unsafe { (api.cuModuleGetFunction)(&mut f, module, cname.as_ptr()) },
                &format!("cuModuleGetFunction({name})"),
            )?;
            Ok(f)
        };
        let functions = [
            function("col_stats")?,
            function("gram_atb")?,
            function("chol_solve")?,
            function("predict_r2")?,
        ];
        let mut stream: driver::CUstream = std::ptr::null_mut();
        api.check(
            // SAFETY: `stream` is a live out-parameter in the current context.
            unsafe { (api.cuStreamCreate)(&mut stream, 0) },
            "cuStreamCreate",
        )?;
        Ok((module, functions, stream))
    })();
    match built {
        Ok((_module, [f_col_stats, f_gram_atb, f_chol_solve, f_predict_r2], stream)) => {
            Ok(Mutex::new(Gpu {
                _lib: lib,
                api,
                ctx,
                stream,
                f_col_stats,
                f_gram_atb,
                f_chol_solve,
                f_predict_r2,
                info: GpuInfo {
                    name,
                    driver_version,
                    total_mem_bytes: total_mem,
                    multiprocessors,
                },
            }))
        }
        Err(error) => {
            // SAFETY: `ctx` was created above and is not stored anywhere else; destroying it
            // also releases any module or stream created in it before the failure.
            unsafe { (api.cuCtxDestroy_v2)(ctx) };
            drop(lib);
            Err(error)
        }
    }
}

/// Best-effort device discovery. `None` when no usable driver/device exists.
pub fn probe() -> Option<GpuInfo> {
    backend().ok().map(|gpu| gpu.lock().unwrap().info.clone())
}

// ---------------------------------------------------------------------------
// CPU reference (mirrors feat-math evolve.rs test exact_backend bit-for-bit)
// ---------------------------------------------------------------------------

fn cpu_reference_score(problem: &GpuRidgeScoreProblem) -> f64 {
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0; problem.p]; problem.train_n];
    for (row, values) in rows.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = problem.train_x_cols[column * problem.train_n + row] as f64;
        }
    }
    let target = problem
        .train_y
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let model = RidgeModel::fit(&rows, &target, problem.lambda_frac);
    let validation = problem
        .valid_x_rows
        .chunks_exact(problem.p)
        .map(|row| row.iter().map(|value| f64::from(*value)).collect())
        .collect::<Vec<Vec<f64>>>();
    feat_math::metrics::r2(&problem.valid_y, &model.predict(&validation))
}

fn validate_dimensions(
    index: usize,
    p: usize,
    train_n: usize,
    valid_n: usize,
) -> Result<(usize, usize), String> {
    if p == 0 || train_n == 0 {
        return Err(format!(
            "problem {index}: empty ridge problem (p={p}, n={train_n})"
        ));
    }
    let train_len = p
        .checked_mul(train_n)
        .ok_or_else(|| format!("problem {index}: ridge dimensions overflow"))?;
    let valid_len = p
        .checked_mul(valid_n)
        .ok_or_else(|| format!("problem {index}: ridge dimensions overflow"))?;
    let gram_len = p
        .checked_mul(p)
        .ok_or_else(|| format!("problem {index}: ridge Gram dimensions overflow"))?;
    // CPU fallback widens both designs to f64 and allocates the Gram and row headers.
    for len in [train_len, valid_len, gram_len] {
        std::alloc::Layout::array::<f64>(len)
            .map_err(|_| format!("problem {index}: ridge allocation layout overflow"))?;
    }
    for len in [train_n, valid_n] {
        std::alloc::Layout::array::<Vec<f64>>(len)
            .map_err(|_| format!("problem {index}: ridge row layout overflow"))?;
    }
    Ok((train_len, valid_len))
}

fn gpu_dimensions_fit(p: usize, train_n: usize, valid_n: usize) -> bool {
    p <= MAX_RIDGE_P
        && i32::try_from(p).is_ok()
        && i32::try_from(train_n).is_ok()
        && i32::try_from(valid_n).is_ok()
}

fn validate_shape(index: usize, problem: &GpuRidgeScoreProblem) -> Result<(), String> {
    let GpuRidgeScoreProblem {
        train_x_cols,
        train_y,
        train_n,
        valid_x_rows,
        valid_y,
        valid_n,
        p,
        ..
    } = problem;
    let (train_len, valid_len) = validate_dimensions(index, *p, *train_n, *valid_n)?;
    if train_x_cols.len() != train_len
        || train_y.len() != *train_n
        || valid_x_rows.len() != valid_len
        || valid_y.len() != *valid_n
    {
        return Err(format!("problem {index}: inconsistent buffer shapes"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Packing + launch
// ---------------------------------------------------------------------------

struct Packed {
    train_x: Vec<f32>,
    train_y: Vec<f32>,
    valid_x: Vec<f32>,
    valid_y: Vec<f64>,
    x_off: Vec<u64>,
    y_off: Vec<u64>,
    vx_off: Vec<u64>,
    vy_off: Vec<u64>,
    stat_off: Vec<u64>,
    gram_off: Vec<u64>,
    w_off: Vec<u64>,
    train_n: Vec<i32>,
    valid_n: Vec<i32>,
    p_dims: Vec<i32>,
    lambdas: Vec<f64>,
    stat_len: u64,
    gram_len: u64,
    w_len: u64,
}

fn pack(problems: &[&GpuRidgeScoreProblem]) -> Packed {
    let mut packed = Packed {
        train_x: Vec::new(),
        train_y: Vec::new(),
        valid_x: Vec::new(),
        valid_y: Vec::new(),
        x_off: Vec::with_capacity(problems.len()),
        y_off: Vec::with_capacity(problems.len()),
        vx_off: Vec::with_capacity(problems.len()),
        vy_off: Vec::with_capacity(problems.len()),
        stat_off: Vec::with_capacity(problems.len()),
        gram_off: Vec::with_capacity(problems.len()),
        w_off: Vec::with_capacity(problems.len()),
        train_n: Vec::with_capacity(problems.len()),
        valid_n: Vec::with_capacity(problems.len()),
        p_dims: Vec::with_capacity(problems.len()),
        lambdas: Vec::with_capacity(problems.len()),
        stat_len: 0,
        gram_len: 0,
        w_len: 0,
    };
    for problem in problems {
        packed.x_off.push(packed.train_x.len() as u64);
        packed.y_off.push(packed.train_y.len() as u64);
        packed.vx_off.push(packed.valid_x.len() as u64);
        packed.vy_off.push(packed.valid_y.len() as u64);
        packed.stat_off.push(packed.stat_len);
        packed.gram_off.push(packed.gram_len);
        packed.w_off.push(packed.w_len);
        packed.train_x.extend_from_slice(&problem.train_x_cols);
        packed.train_y.extend_from_slice(&problem.train_y);
        packed.valid_x.extend_from_slice(&problem.valid_x_rows);
        packed.valid_y.extend_from_slice(&problem.valid_y);
        packed.train_n.push(problem.train_n as i32);
        packed.valid_n.push(problem.valid_n as i32);
        packed.p_dims.push(problem.p as i32);
        // Mirrors RidgeModel::fit: lambda = max(lambda_frac * n, 1e-3).
        packed
            .lambdas
            .push((problem.lambda_frac * problem.train_n as f64).max(1e-3));
        packed.stat_len += problem.p as u64;
        packed.gram_len += (problem.p * problem.p) as u64;
        packed.w_len += (problem.p + 1) as u64;
    }
    packed
}

fn wave_bytes(problem: &GpuRidgeScoreProblem) -> usize {
    let p = problem.p;
    (problem.train_x_cols.len() + problem.train_y.len() + problem.valid_x_rows.len()) * 4
        + problem.valid_y.len() * 8
        + (4 * p + 1 + p * p + p + 1) * 8 // means/scales/atb/z + score + gram + w
        + 96 // metadata slots
}

fn logical_h2d(problem: &GpuRidgeScoreProblem) -> u64 {
    ((problem.train_x_cols.len() + problem.train_y.len() + problem.valid_x_rows.len()) * 4
        + problem.valid_y.len() * 8) as u64
}

/// Order-preserving batched rung-0 ridge scorer: the drop-in backend for
/// `feat_math::evolve::GpuRidgeScoreBatchFn`.
pub fn gpu_ridge_score_batch(
    problems: Vec<GpuRidgeScoreProblem>,
) -> Result<GpuRidgeScoreBatch, String> {
    let gpu_routes = problems
        .iter()
        .enumerate()
        .map(|(index, problem)| {
            validate_shape(index, problem)?;
            Ok(gpu_dimensions_fit(
                problem.p,
                problem.train_n,
                problem.valid_n,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let gpu = backend()?;
    let mut scores = vec![0.0f64; problems.len()];
    let mut fold_bytes = vec![GpuRidgeScoreFoldBytes::default(); problems.len()];
    let mut timings = GpuRidgeScoreTimings::default();

    // Eligibility was checked before backend setup; CUDA dimensions must fit signed int.
    let pack_start = Instant::now();
    let mut gpu_idx: Vec<usize> = Vec::with_capacity(problems.len());
    for (index, problem) in problems.iter().enumerate() {
        if gpu_routes[index] {
            gpu_idx.push(index);
        } else {
            scores[index] = cpu_reference_score(problem);
        }
    }
    timings.pack_seconds += pack_start.elapsed().as_secs_f64();

    let guard = gpu
        .lock()
        .map_err(|_| "battle-gpu mutex poisoned".to_string())?;
    let api = &guard.api;
    api.check(
        // SAFETY: `guard.ctx` is the backend's live context, used under its Mutex.
        unsafe { (api.cuCtxSetCurrent)(guard.ctx) },
        "cuCtxSetCurrent",
    )?;

    // Wave budget from free memory (fallback: fixed 2 GiB).
    let budget = {
        let (mut free, mut total): (usize, usize) = (0, 0);
        // SAFETY: both are live out-parameters; the context was made current above.
        let status = unsafe { (api.cuMemGetInfo_v2)(&mut free, &mut total) };
        if status == 0 {
            ((free as f64 * WAVE_MEM_FRACTION) as usize).max(64 << 20)
        } else {
            WAVE_MEM_FALLBACK_BYTES
        }
    };

    let mut cursor = 0usize;
    while cursor < gpu_idx.len() {
        let mut end = cursor;
        let mut bytes = 0usize;
        while end < gpu_idx.len() {
            let candidate = wave_bytes(&problems[gpu_idx[end]]);
            if end > cursor && bytes + candidate > budget {
                break;
            }
            bytes += candidate;
            end += 1;
        }
        let wave: Vec<&GpuRidgeScoreProblem> =
            gpu_idx[cursor..end].iter().map(|&i| &problems[i]).collect();
        let wave_scores = run_wave(&guard, &wave, &mut timings)?;
        for (slot, &index) in gpu_idx[cursor..end].iter().enumerate() {
            scores[index] = wave_scores[slot];
            fold_bytes[index] = GpuRidgeScoreFoldBytes {
                logical_h2d_bytes: logical_h2d(&problems[index]),
                physical_h2d_bytes: logical_h2d(&problems[index]),
                logical_d2h_bytes: 8,
                physical_d2h_bytes: 8,
            };
        }
        cursor = end;
    }
    Ok(GpuRidgeScoreBatch {
        scores,
        timings,
        fold_bytes,
    })
}

fn run_wave(
    gpu: &Gpu,
    wave: &[&GpuRidgeScoreProblem],
    timings: &mut GpuRidgeScoreTimings,
) -> Result<Vec<f64>, String> {
    let api = &gpu.api;
    let pack_start = Instant::now();
    let problems = c_int::try_from(wave.len())
        .map_err(|_| "battle-gpu wave count exceeds CUDA int".to_string())?;
    let packed = pack(wave);
    timings.pack_seconds += pack_start.elapsed().as_secs_f64();

    let h2d_start = Instant::now();
    let alloc_f64 = |len: u64| DeviceBuffer::alloc(api, (len as usize) * 8);
    let upload_f32 = |host: &[f32]| -> Result<DeviceBuffer<'_>, String> {
        let buffer = DeviceBuffer::alloc(api, host.len() * 4)?;
        buffer.upload(host)?;
        Ok(buffer)
    };
    let upload_f64 = |host: &[f64]| -> Result<DeviceBuffer<'_>, String> {
        let buffer = DeviceBuffer::alloc(api, host.len() * 8)?;
        buffer.upload(host)?;
        Ok(buffer)
    };
    let upload_u64 = |host: &[u64]| -> Result<DeviceBuffer<'_>, String> {
        let buffer = DeviceBuffer::alloc(api, host.len() * 8)?;
        buffer.upload(host)?;
        Ok(buffer)
    };
    let upload_i32 = |host: &[i32]| -> Result<DeviceBuffer<'_>, String> {
        let buffer = DeviceBuffer::alloc(api, host.len() * 4)?;
        buffer.upload(host)?;
        Ok(buffer)
    };
    let d_train_x = upload_f32(&packed.train_x)?;
    let d_train_y = upload_f32(&packed.train_y)?;
    let d_valid_x = upload_f32(&packed.valid_x)?;
    let d_valid_y = upload_f64(&packed.valid_y)?;
    let d_x_off = upload_u64(&packed.x_off)?;
    let d_y_off = upload_u64(&packed.y_off)?;
    let d_vx_off = upload_u64(&packed.vx_off)?;
    let d_vy_off = upload_u64(&packed.vy_off)?;
    let d_stat_off = upload_u64(&packed.stat_off)?;
    let d_gram_off = upload_u64(&packed.gram_off)?;
    let d_w_off = upload_u64(&packed.w_off)?;
    let d_train_n = upload_i32(&packed.train_n)?;
    let d_valid_n = upload_i32(&packed.valid_n)?;
    let d_p = upload_i32(&packed.p_dims)?;
    let d_lambda = upload_f64(&packed.lambdas)?;
    let d_means = alloc_f64(packed.stat_len)?;
    let d_scales = alloc_f64(packed.stat_len)?;
    let d_atb = alloc_f64(packed.stat_len)?;
    let d_z = alloc_f64(packed.stat_len)?;
    let d_gram = alloc_f64(packed.gram_len)?;
    let d_w = alloc_f64(packed.w_len)?;
    let d_y_means = alloc_f64(wave.len() as u64)?;
    let d_scores = alloc_f64(wave.len() as u64)?;
    timings.h2d_seconds += h2d_start.elapsed().as_secs_f64();

    let launch = |f: CUfunction,
                  grid: c_uint,
                  block: c_uint,
                  args: &mut [*mut c_void]|
     -> Result<(), String> {
        api.check(
            // `f` is one of the four functions loaded from the embedded PTX, and every call site
            // below passes `args` built by `ptr_args!` in that kernel's parameter order, each
            // element pointing at a live local of the parameter's type (u64 device pointers, then
            // the c_int problem count). The driver copies the values at launch; `extra` is null.
            // SAFETY: kernel handle and argument list match the PTX signature (see above).
            unsafe {
                (api.cuLaunchKernel)(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    gpu.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    };
    macro_rules! ptr_args {
        ($($value:expr),* $(,)?) => {{
            [$(std::ptr::addr_of!($value) as *mut c_void),*]
        }};
    }

    let solve_start = Instant::now();
    let (px, py, pvx, pvy) = (
        d_train_x.ptr(),
        d_train_y.ptr(),
        d_valid_x.ptr(),
        d_valid_y.ptr(),
    );
    let (pxo, pyo, pvxo, pvyo) = (d_x_off.ptr(), d_y_off.ptr(), d_vx_off.ptr(), d_vy_off.ptr());
    let (pso, pgo, pwo) = (d_stat_off.ptr(), d_gram_off.ptr(), d_w_off.ptr());
    let (ptn, pvn, pp, pl) = (d_train_n.ptr(), d_valid_n.ptr(), d_p.ptr(), d_lambda.ptr());
    let (pm, psc, pym) = (d_means.ptr(), d_scales.ptr(), d_y_means.ptr());
    let (pg, pb, pz, pw, psr) = (
        d_gram.ptr(),
        d_atb.ptr(),
        d_z.ptr(),
        d_w.ptr(),
        d_scores.ptr(),
    );

    let mut a1 = ptr_args![px, py, pxo, pyo, pso, ptn, pp, pm, psc, pym, problems];
    launch(gpu.f_col_stats, wave.len() as c_uint, 128, &mut a1)?;
    let mut a2 = ptr_args![px, py, pxo, pyo, pso, pgo, ptn, pp, pm, psc, pym, pl, pg, pb, problems];
    launch(gpu.f_gram_atb, wave.len() as c_uint, 256, &mut a2)?;
    let chol_grid = wave.len().div_ceil(128) as c_uint;
    let mut a3 = ptr_args![pgo, pso, pwo, pp, pym, pg, pb, pz, pw, problems];
    launch(gpu.f_chol_solve, chol_grid, 128, &mut a3)?;
    api.check(
        // SAFETY: `gpu.stream` is the backend's live stream in the current context.
        unsafe { (api.cuStreamSynchronize)(gpu.stream) },
        "cuStreamSynchronize(solve)",
    )?;
    timings.solve_seconds += solve_start.elapsed().as_secs_f64();

    let predict_start = Instant::now();
    let mut a4 = ptr_args![pvx, pvy, pvxo, pvyo, pso, pwo, pvn, pp, pm, psc, pw, psr, problems];
    launch(gpu.f_predict_r2, chol_grid, 128, &mut a4)?;
    api.check(
        // SAFETY: `gpu.stream` is the backend's live stream in the current context.
        unsafe { (api.cuStreamSynchronize)(gpu.stream) },
        "cuStreamSynchronize(predict)",
    )?;
    let mut scores = vec![0.0f64; wave.len()];
    d_scores.download(&mut scores)?;
    timings.predict_score_seconds += predict_start.elapsed().as_secs_f64();
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::Pcg32;

    /// Compile-time proof that the public fn satisfies the seam signature.
    #[test]
    fn backend_signature_matches_the_seam() {
        fn accepts(_backend: &feat_math::evolve::GpuRidgeScoreBatchFn<'_>) {}
        accepts(&gpu_ridge_score_batch);
    }

    fn synth_problem(
        rng: &mut Pcg32,
        n: usize,
        p: usize,
        lambda_frac: f64,
    ) -> GpuRidgeScoreProblem {
        let vn = (n / 3).max(2);
        let mut cols = vec![0.0f32; n * p];
        for column in 0..p {
            for row in 0..n {
                // Constant column when column == p-1 and p > 2 exercises the
                // scale floor; otherwise noisy signal.
                cols[column * n + row] = if column + 1 == p && p > 2 {
                    1.5
                } else {
                    (rng.next_f64() * 4.0 - 2.0) as f32
                };
            }
        }
        let train_y: Vec<f32> = (0..n)
            .map(|row| {
                let mut acc = 0.0f64;
                for column in 0..p.min(4) {
                    acc += (column as f64 + 1.0) * 0.3 * cols[column * n + row] as f64;
                }
                (acc + 0.1 * rng.gaussian()) as f32
            })
            .collect();
        let mut valid_x = vec![0.0f32; vn * p];
        for value in valid_x.iter_mut() {
            *value = (rng.next_f64() * 4.0 - 2.0) as f32;
        }
        let valid_y: Vec<f64> = (0..vn)
            .map(|row| {
                let mut acc = 0.0f64;
                for column in 0..p.min(4) {
                    acc += (column as f64 + 1.0) * 0.3 * valid_x[row * p + column] as f64;
                }
                acc + 0.1 * rng.gaussian()
            })
            .collect();
        GpuRidgeScoreProblem {
            train_x_cols: cols,
            train_y,
            train_n: n,
            valid_x_rows: valid_x,
            valid_y,
            valid_n: vn,
            p,
            lambda_frac,
        }
    }

    fn gpu_available() -> bool {
        match probe() {
            Some(info) => {
                println!(
                    "battle-gpu test device: {} (driver {})",
                    info.name, info.driver_version
                );
                true
            }
            None => {
                println!("battle-gpu: no usable CUDA device — hardware-gated tests skipped");
                false
            }
        }
    }

    #[test]
    fn gpu_scores_match_cpu_reference_bitwise() {
        if !gpu_available() {
            return;
        }
        let mut rng = Pcg32::new(0xB47713, 7);
        let mut problems = Vec::new();
        // Randomized shapes incl. edges: minimum rows, p=1, near-max p,
        // constant columns (baked into synth for p > 2).
        for case in 0..50 {
            let n = [4, 16, 33, 100, 257, 512][case % 6];
            let p = [1, 2, 5, 17, 64, 96][(case / 6) % 6]
                .min(n.max(2) - 1)
                .max(1);
            let lambda_frac = [0.0, 0.02, 0.5][case % 3];
            problems.push(synth_problem(&mut rng, n, p, lambda_frac));
        }
        let expected: Vec<f64> = problems.iter().map(cpu_reference_score).collect();
        let batch = gpu_ridge_score_batch(problems).expect("gpu batch");
        assert_eq!(batch.scores.len(), expected.len());
        let mut max_delta = 0.0f64;
        let mut exact = 0usize;
        for (index, (gpu, cpu)) in batch.scores.iter().zip(&expected).enumerate() {
            let delta = (gpu - cpu).abs();
            max_delta = max_delta.max(delta);
            if gpu.to_bits() == cpu.to_bits() {
                exact += 1;
            }
            assert!(
                delta <= 1e-9,
                "problem {index}: gpu={gpu} cpu={cpu} delta={delta}"
            );
        }
        println!(
            "gpu-vs-cpu: {}/{} bit-exact, max |delta| = {max_delta:e}",
            exact,
            expected.len()
        );
        // -fmad=false + sequential order should make every score bit-exact.
        assert_eq!(exact, expected.len(), "expected full bitwise agreement");
    }

    #[test]
    fn same_wave_twice_is_bit_identical() {
        if !gpu_available() {
            return;
        }
        let mut rng = Pcg32::new(0xD373C7, 11);
        let problems: Vec<GpuRidgeScoreProblem> = (0..64)
            .map(|case| synth_problem(&mut rng, 128 + case, 32, 0.02))
            .collect();
        let first = gpu_ridge_score_batch(problems.clone()).expect("first wave");
        let second = gpu_ridge_score_batch(problems).expect("second wave");
        for (a, b) in first.scores.iter().zip(&second.scores) {
            assert_eq!(a.to_bits(), b.to_bits(), "nondeterministic score");
        }
    }

    #[test]
    fn wide_problems_route_to_cpu_and_preserve_order() {
        if !gpu_available() {
            return;
        }
        let mut rng = Pcg32::new(0xA11CE, 3);
        // p = 200 > MAX_RIDGE_P must be CPU-routed; neighbors stay GPU-scored.
        let narrow_a = synth_problem(&mut rng, 64, 8, 0.02);
        let wide = synth_problem(&mut rng, 210, 200, 0.02);
        let narrow_b = synth_problem(&mut rng, 64, 8, 0.02);
        let expected: Vec<f64> = [&narrow_a, &wide, &narrow_b]
            .iter()
            .map(|problem| cpu_reference_score(problem))
            .collect();
        let batch = gpu_ridge_score_batch(vec![narrow_a, wide, narrow_b]).expect("mixed batch");
        for (index, (got, want)) in batch.scores.iter().zip(&expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "slot {index} diverged");
        }
        // CPU-routed slot reports no device traffic.
        assert_eq!(batch.fold_bytes[1], GpuRidgeScoreFoldBytes::default());
        assert!(batch.fold_bytes[0].logical_h2d_bytes > 0);
    }

    #[test]
    fn malformed_problems_error_instead_of_scoring() {
        let mut rng = Pcg32::new(0xBAD, 5);
        let mut problem = synth_problem(&mut rng, 32, 4, 0.02);
        problem.train_y.pop();
        let result = gpu_ridge_score_batch(vec![problem]);
        assert!(result.is_err(), "shape mismatch must be an Err");
        let empty = GpuRidgeScoreProblem {
            train_x_cols: vec![],
            train_y: vec![],
            train_n: 0,
            valid_x_rows: vec![],
            valid_y: vec![],
            valid_n: 0,
            p: 0,
            lambda_frac: 0.02,
        };
        assert!(gpu_ridge_score_batch(vec![empty]).is_err());
    }

    #[test]
    fn wave_timing_snapshot() {
        if !gpu_available() {
            return;
        }
        let mut rng = Pcg32::new(0x71317, 13);
        let problems: Vec<GpuRidgeScoreProblem> = (0..1024)
            .map(|_| synth_problem(&mut rng, 256, 64, 0.02))
            .collect();
        // Warm-up wave (context/module init + allocator warm).
        let _ = gpu_ridge_score_batch(problems[..32].to_vec()).expect("warmup");
        let start = Instant::now();
        let batch = gpu_ridge_score_batch(problems).expect("big wave");
        let total = start.elapsed().as_secs_f64();
        println!(
            "wave: 1024 problems n=256 p=64 total={total:.4}s pack={:.4}s h2d={:.4}s solve={:.4}s predict={:.4}s",
            batch.timings.pack_seconds,
            batch.timings.h2d_seconds,
            batch.timings.solve_seconds,
            batch.timings.predict_score_seconds
        );
        assert!(batch.scores.iter().all(|score| score.is_finite()));
    }
}

#[cfg(test)]
mod shape_bounds_tests {
    use super::*;

    fn problem() -> GpuRidgeScoreProblem {
        GpuRidgeScoreProblem {
            train_x_cols: vec![0.0; 2],
            train_y: vec![0.0],
            train_n: 1,
            valid_x_rows: vec![0.0; 2],
            valid_y: vec![0.0],
            valid_n: 1,
            p: 2,
            lambda_frac: 0.02,
        }
    }

    #[test]
    fn valid_shapes_and_empty_validation_remain_admitted() {
        let mut problem = problem();
        assert!(validate_shape(0, &problem).is_ok());
        problem.valid_n = 0;
        problem.valid_x_rows.clear();
        problem.valid_y.clear();
        assert!(validate_shape(0, &problem).is_ok());
    }

    #[test]
    fn wrapped_training_dimensions_are_refused_without_panicking() {
        let mut problem = problem();
        problem.p = 1usize << (usize::BITS - 1);
        problem.train_n = 2;
        problem.train_x_cols.clear();
        problem.train_y = vec![0.0; 2];
        problem.valid_n = 0;
        problem.valid_x_rows.clear();
        problem.valid_y.clear();
        let result = std::panic::catch_unwind(|| validate_shape(0, &problem));
        assert!(result.is_ok(), "malformed dimensions must return an error");
        assert!(
            result.unwrap().is_err(),
            "wrapped zero-size matrix is invalid"
        );
    }

    #[test]
    fn overflowing_validation_dimensions_return_error_without_panicking() {
        let mut problem = problem();
        problem.valid_n = 1usize << (usize::BITS - 1);
        problem.valid_x_rows.clear();
        problem.valid_y.clear();
        let result = std::panic::catch_unwind(|| validate_shape(0, &problem));
        assert!(result.is_ok(), "malformed dimensions must return an error");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn cuda_row_count_boundaries_route_without_narrowing() {
        let max = i32::MAX as usize;
        assert!(gpu_dimensions_fit(1, max, max));
        assert!(gpu_dimensions_fit(MAX_RIDGE_P, 1, 0));
        assert!(!gpu_dimensions_fit(1, max + 1, 1));
        assert!(!gpu_dimensions_fit(1, 1, max + 1));
        assert!(!gpu_dimensions_fit(MAX_RIDGE_P + 1, 1, 1));
        #[cfg(target_pointer_width = "64")]
        assert!(validate_dimensions(0, 1, max + 1, max + 1).is_ok());
    }

    #[test]
    fn cpu_gram_product_overflow_is_refused_without_allocating() {
        let p = 1usize << (usize::BITS / 2);
        assert!(p.checked_mul(1).is_some());
        assert!(p.checked_mul(p).is_none());
        assert!(validate_dimensions(0, p, 1, 0).is_err());
    }

    #[test]
    fn cpu_gram_byte_layout_overflow_is_refused_without_allocating() {
        let p = 1usize << ((usize::BITS - 4).div_ceil(2));
        assert!(p.checked_mul(p).is_some());
        assert!(std::alloc::Layout::array::<f64>(p * p).is_err());
        assert!(validate_dimensions(0, p, 1, 0).is_err());
        assert_eq!(
            validate_dimensions(0, MAX_RIDGE_P + 1, 2, 0),
            Ok(((MAX_RIDGE_P + 1) * 2, 0))
        );
    }

    #[test]
    fn cpu_widened_design_and_row_layouts_are_checked() {
        let max_f64 = isize::MAX as usize / std::mem::size_of::<f64>();
        let n = max_f64 / 4 + 1;
        assert!(std::alloc::Layout::array::<Vec<f64>>(n).is_ok());
        assert!(validate_dimensions(0, 4, n, 0).is_err());
        assert!(validate_dimensions(0, 4, 1, n).is_err());
        let max_rows = isize::MAX as usize / std::mem::size_of::<Vec<f64>>();
        assert!(validate_dimensions(0, 1, max_rows + 1, 0).is_err());
        assert!(validate_dimensions(0, 1, 1, max_rows + 1).is_err());
    }
}
