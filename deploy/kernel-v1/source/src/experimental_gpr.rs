//! Optional FP64 LibTorch GPR qualification backend. Not used by production GP.
//!
//! The initialization, subset recipe, parameter layout and update clamps mirror
//! feat-math's GPR heads. LibTorch reductions and Cholesky have different floating
//! point order: predictions require explicit numerical qualification. No reward
//! or fitted predictor from this module is admitted to the teaching corpus.
use anyhow::{ensure, Result};
use feat_math::gpr::{KernelPrim, KernelSpec, GPR_SEED, INIT_ROWS, LR, OPT_CAP};
use feat_math::{heads::std_fit, krr::seeded_rows, Rows};
use serde::Serialize;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

pub struct Input<'a> {
    pub x: &'a Rows,
    pub y: &'a [f64],
    pub valid: &'a Rows,
    pub m: usize,
    pub steps: usize,
    pub spec: &'a KernelSpec,
}

#[derive(Debug, Serialize)]
pub struct Timing {
    pub cpu_initialization_seconds: f64,
    pub upload_seconds: f64,
    pub optimization_seconds: f64,
    pub final_solve_seconds: f64,
    pub prediction_seconds: f64,
    pub download_seconds: f64,
    pub total_seconds: f64,
    pub batch: usize,
    pub p: usize,
    pub basis_rows: usize,
    pub optimization_rows: usize,
    pub steps: usize,
}

pub struct Output {
    pub predictions: Vec<Vec<f64>>,
    pub theta: Vec<Vec<f64>>,
    pub timing: Timing,
}

struct Initial {
    basis: Vec<f64>,
    centered: Vec<f64>,
    valid: Vec<f64>,
    theta: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    amp_indices: Vec<i64>,
    mean: f64,
}

fn initial(input: &Input<'_>) -> Result<Initial> {
    let n = input.x.len();
    let p = input.x[0].len();
    ensure!(n >= 12 && p > 0 && input.y.len() == n, "unsupported small/empty input");
    ensure!(input.x.iter().chain(input.valid.iter()).all(|r| r.len() == p && r.iter().all(|v| v.is_finite())), "ragged or non-finite features");
    ensure!(input.y.iter().all(|v| v.is_finite()), "non-finite target");
    let (means, scales) = std_fit(input.x);
    let mean = input.y.iter().sum::<f64>() / n as f64;
    let indices = if n > input.m { seeded_rows(n, input.m, GPR_SEED) } else { (0..n).collect() };
    let z: Rows = indices.iter().map(|&i| (0..p).map(|j| (input.x[i][j] - means[j]) / scales[j]).collect()).collect();
    let centered: Vec<f64> = indices.iter().map(|&i| input.y[i] - mean).collect();
    let opt_size = z.len().min(OPT_CAP);
    let n_init = opt_size.min(INIT_ROWS);
    let mut ard = Vec::with_capacity(p);
    for d in 0..p {
        let mut distances = Vec::with_capacity(n_init * (n_init - 1) / 2);
        for i in 0..n_init { for j in i+1..n_init { distances.push((z[i][d] - z[j][d]).abs()); } }
        distances.sort_by(f64::total_cmp);
        let median = if distances.is_empty() { 1. } else {
            let pos = 0.5 * (distances.len() - 1) as f64;
            let l = pos as usize;
            let h = (l + 1).min(distances.len() - 1);
            distances[l] * (1. - (pos - l as f64)) + distances[h] * (pos - l as f64)
        };
        ard.push((median.max(1e-3) * (p as f64).sqrt()).ln());
    }
    let sy = (centered.iter().take(opt_size).map(|v| v*v).sum::<f64>() / opt_size as f64).sqrt();
    let mut theta = vec![];
    let mut lo = vec![];
    let mut hi = vec![];
    let mut amp_indices = vec![];
    for term in input.spec {
        amp_indices.push(theta.len() as i64);
        theta.push(sy.max(1e-6).ln() - 0.5 * (input.spec.len() as f64).ln());
        lo.push(-15.); hi.push(15.);
        for prim in term {
            match prim {
                KernelPrim::Rbf | KernelPrim::Rq => {
                    theta.extend_from_slice(&ard);
                    lo.extend(std::iter::repeat_n(-6., p)); hi.extend(std::iter::repeat_n(6., p));
                    if *prim == KernelPrim::Rq { theta.push(0.); lo.push(-3.); hi.push(5.); }
                }
                KernelPrim::Periodic => {
                    theta.extend([2f64.ln(), 0.5 * (2. * p as f64).ln()]);
                    lo.extend([-3., -6.]); hi.extend([3., 6.]);
                }
                KernelPrim::Linear => {}
            }
        }
    }
    theta.push((0.1 * sy).max(1e-6).ln());
    // Noise is dynamically clamped to the largest updated amplitude below.
    lo.push(f64::NEG_INFINITY); hi.push(f64::INFINITY);
    let valid = input.valid.iter().flat_map(|r| (0..p).map(|j| (r[j] - means[j]) / scales[j])).collect();
    Ok(Initial { basis: z.into_iter().flatten().collect(), centered, valid, theta, lo, hi, amp_indices, mean })
}

fn synchronize(device: Device) {
    if let Device::Cuda(index) = device { tch::Cuda::synchronize(index as i64); }
}

struct PairFeatures { differences: Tensor, linear: Tensor, p: i64 }
impl PairFeatures {
    fn new(a: &Tensor, b: &Tensor, spec: &KernelSpec) -> Self {
        let p = a.size()[2];
        let differences = if spec.iter().flatten().any(|v| *v != KernelPrim::Linear) { a.unsqueeze(2) - b.unsqueeze(1) } else { Tensor::new() };
        Self { differences, linear: a.matmul(&b.transpose(1,2)) / p as f64, p }
    }
    fn kernel(&self, spec: &KernelSpec, theta: &Tensor) -> Tensor {
        let mut off = 0;
        let mut total = Tensor::zeros_like(&self.linear);
        for term in spec {
            let mut value = (theta.narrow(1, off, 1) * 2.).exp().unsqueeze(2);
            off += 1;
            for prim in term {
                let factor = match prim {
                    KernelPrim::Rbf | KernelPrim::Rq => {
                        // Decode exp(log lengthscale), then reciprocal square,
                        // matching the CPU expression (not exp(-2*log ls)).
                        let ls = theta.narrow(1, off, self.p).exp();
                        let inv = (&ls * &ls).reciprocal().unsqueeze(1).unsqueeze(1);
                        let r2 = (&self.differences * &self.differences * inv).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
                        off += self.p;
                        if *prim == KernelPrim::Rbf { (r2 * -0.5).exp() } else {
                            let alpha = theta.narrow(1, off, 1).exp().unsqueeze(2);
                            off += 1;
                            (-&alpha * (r2 / (&alpha * 2.) + 1.).log()).exp()
                        }
                    }
                    KernelPrim::Periodic => {
                        let period = theta.narrow(1, off, 1).exp().unsqueeze(2).unsqueeze(3);
                        let ls = theta.narrow(1, off+1, 1).exp();
                        let inv = (&ls * &ls).reciprocal().unsqueeze(2);
                        off += 2;
                        let sin = (&self.differences * std::f64::consts::PI / period).sin();
                        ((&sin * &sin).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double) * inv * -2.).exp()
                    }
                    KernelPrim::Linear => self.linear.shallow_clone(),
                };
                value = value * factor;
            }
            total = total + value;
        }
        total
    }
}

fn factor(matrix: &Tensor) -> Result<Tensor> {
    let (l, info) = matrix.f_linalg_cholesky_ex(false, false)?;
    ensure!(i64::try_from(info.abs().max())? == 0, "LibTorch Cholesky failed; exact CPU fallback required");
    ensure!(f64::try_from(l.diagonal(0, -2, -1).min())? > 1e-6, "CPU pivot-floor regime: exact CPU fallback required");
    Ok(l)
}

/// Same-shape batch; caller owns exclusive GPU scheduling and bounds memory.
pub fn fit_predict(inputs: &[Input<'_>], device: Device) -> Result<Output> {
    ensure!(!inputs.is_empty(), "empty batch");
    let first = &inputs[0];
    ensure!(!first.x.is_empty() && !first.x[0].is_empty(), "empty matrix");
    let p = first.x[0].len();
    let mb = first.x.len().min(first.m);
    let mo = mb.min(OPT_CAP);
    let vn = first.valid.len();
    let batch = inputs.len();
    ensure!(vn > 0, "empty heldout features");
    for x in inputs {
        ensure!(!x.x.is_empty() && x.x[0].len() == p && x.x.len().min(x.m) == mb && x.valid.len() == vn && x.steps == first.steps && x.spec == first.spec, "batch geometry/kernel mismatch");
    }
    // Conservative ceiling for pairwise tensors plus autograd intermediates.
    let pair_width = if first.spec.iter().flatten().any(|v| *v != KernelPrim::Linear) { p } else { 1 };
    let estimate = batch as u64 * (mb.max(vn) as u64) * mb as u64 * pair_width as u64 * 8 * 24;
    ensure!(estimate <= 8 * 1024u64.pow(3), "prototype batch estimate exceeds 8 GiB; reduce batch");
    let begin = Instant::now();
    let initial: Vec<Initial> = inputs.iter().map(initial).collect::<Result<_>>()?;
    let cpu_initialization_seconds = begin.elapsed().as_secs_f64();
    let t = Instant::now();
    let stack = |get: fn(&Initial) -> &[f64], shape: &[i64]| {
        Tensor::from_slice(&initial.iter().flat_map(|x| get(x).iter().copied()).collect::<Vec<f64>>()).reshape(shape).to_device(device)
    };
    let basis = stack(|x| &x.basis, &[batch as i64, mb as i64, p as i64]);
    let centered = stack(|x| &x.centered, &[batch as i64, mb as i64, 1]);
    let valid = stack(|x| &x.valid, &[batch as i64, vn as i64, p as i64]);
    let parameters = initial[0].theta.len() as i64;
    let mut theta = stack(|x| &x.theta, &[batch as i64, parameters]).set_requires_grad(true);
    let lo = stack(|x| &x.lo, &[batch as i64, parameters]);
    let hi = stack(|x| &x.hi, &[batch as i64, parameters]);
    let amp_indices = Tensor::from_slice(&initial[0].amp_indices).to_device(device);
    let means = Tensor::from_slice(&initial.iter().map(|x| x.mean).collect::<Vec<f64>>()).to_device(device).reshape([batch as i64, 1, 1]);
    let eye = Tensor::eye(mo as i64, (Kind::Double, device)).unsqueeze(0);
    synchronize(device);
    let upload_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let opt = basis.narrow(1, 0, mo as i64);
    let opt_y = centered.narrow(1, 0, mo as i64);
    let pairs = PairFeatures::new(&opt, &opt, first.spec);
    for _ in 0..first.steps {
        let k = pairs.kernel(first.spec, &theta);
        let noise = (theta.narrow(1, parameters-1, 1) * 2.).exp().unsqueeze(2);
        let ky = k + &eye * noise;
        let weight = tch::no_grad(|| -> Result<Tensor> {
            let inverse = factor(&ky.detach())?.f_cholesky_inverse(false)?;
            let alpha = inverse.matmul(&opt_y);
            Ok(alpha.matmul(&alpha.transpose(1,2)) - inverse)
        })?;
        // The detached W expression gives the same analytic marginal-
        // likelihood gradient as feat-math's explicit loops.
        let objective = (&weight * ky).sum(Kind::Double) * 0.5;
        let gradient = Tensor::f_run_backward(&[objective], &[&theta], false, false)?.remove(0);
        theta = tch::no_grad(|| {
            let updated = (&theta + (gradient * (LR / mo as f64)).clamp(-0.5, 0.5)).maximum(&lo).minimum(&hi);
            let max_amp = updated.index_select(1, &amp_indices).max_dim(1, true).0;
            let noise = updated.narrow(1, parameters-1, 1).maximum(&(&max_amp - 6.)).minimum(&max_amp);
            Tensor::cat(&[updated.narrow(1, 0, parameters-1), noise], 1).detach().set_requires_grad(true)
        });
    }
    synchronize(device);
    let optimization_seconds = t.elapsed().as_secs_f64();
    drop(pairs);
    let t = Instant::now();
    let theta = theta.detach();
    let alpha = tch::no_grad(|| -> Result<Tensor> {
        let k = PairFeatures::new(&basis, &basis, first.spec).kernel(first.spec, &theta);
        let noise = (theta.narrow(1, parameters-1, 1) * 2.).exp().unsqueeze(2);
        let ky = k + Tensor::eye(mb as i64, (Kind::Double, device)).unsqueeze(0) * noise;
        Ok(centered.f_cholesky_solve(&factor(&ky)?, false)?)
    })?;
    synchronize(device);
    let final_solve_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let predictions = tch::no_grad(|| (PairFeatures::new(&valid, &basis, first.spec).kernel(first.spec, &theta).matmul(&alpha) + means).clamp(-60., 60.).squeeze_dim(2));
    synchronize(device);
    let prediction_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let predictions: Vec<Vec<f64>> = Vec::try_from(predictions.to_device(Device::Cpu))?;
    let theta: Vec<Vec<f64>> = Vec::try_from(theta.to_device(Device::Cpu))?;
    let download_seconds = t.elapsed().as_secs_f64();
    ensure!(predictions.iter().flatten().all(|v| v.is_finite()) && theta.iter().flatten().all(|v| v.is_finite()), "non-finite prototype result");
    Ok(Output { predictions, theta, timing: Timing { cpu_initialization_seconds, upload_seconds, optimization_seconds, final_solve_seconds, prediction_seconds, download_seconds, total_seconds: begin.elapsed().as_secs_f64(), batch, p, basis_rows: mb, optimization_rows: mo, steps: first.steps } })
}
