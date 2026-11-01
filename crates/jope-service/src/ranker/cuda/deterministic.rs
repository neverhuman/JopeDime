//! CUDA versions of the frozen CPU reduction boundaries. No host arithmetic fallback.
//! The small launch boundary validates contiguous F32 layouts, offsets, extents and
//! devices before calling the existing Candle/cudarc tracked-stream interfaces.
#![allow(unsafe_code)]

use candle_core_0_11_0::{
    cuda_backend::cudarc::{
        driver::{CudaFunction, LaunchConfig, PushKernelArg},
        nvrtc::{compile_ptx_with_opts, CompileOptions},
    },
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Error, Layout, Result, Shape, Tensor,
};
use std::sync::OnceLock;

fn error(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}
fn count(n: usize) -> Result<u32> {
    if n == 0 {
        candle_core_0_11_0::bail!("nonempty CUDA extent required")
    }
    u32::try_from(n).map_err(error)
}
fn check(s: &CudaStorage, l: &Layout) -> Result<u64> {
    count(l.shape().elem_count())?;
    let end = l
        .start_offset()
        .checked_add(l.shape().elem_count())
        .ok_or_else(|| error("CUDA extent overflow"))?;
    if !l.is_contiguous() || end > s.as_cuda_slice::<f32>()?.len() {
        candle_core_0_11_0::bail!("invalid contiguous CUDA F32 extent")
    }
    Ok(l.start_offset() as u64)
}
fn function(s: &CudaStorage, name: &str) -> Result<CudaFunction> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = PTX
        .get_or_init(|| {
            compile_ptx_with_opts(
                include_str!("deterministic.cu"),
                CompileOptions {
                    ftz: Some(false),
                    prec_sqrt: Some(true),
                    prec_div: Some(true),
                    fmad: Some(false),
                    use_fast_math: Some(false),
                    ..Default::default()
                },
            )
            .map(|p| p.to_src())
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(error)?;
    s.device
        .get_or_load_custom_func(name, "rnk12_frozen_arithmetic_v1", ptx)
        .map(|function| function.into_cuda_function())
}
struct Unary {
    name: &'static str,
    width: usize,
    rows: usize,
}
impl CustomOp1 for Unary {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core_0_11_0::bail!("CUDA-only frozen arithmetic")
    }
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let offset = check(s, l)?;
        let n = count(self.rows)?;
        let width = count(self.width)?;
        let src = s.as_cuda_slice::<f32>()?;
        let mut out = s.device.alloc_zeros::<f32>(l.shape().elem_count())?;
        let f = function(s, self.name)?;
        let stream = s.device.cuda_stream();
        let mut b = stream.launch_builder(&f);
        b.arg(&n).arg(&width).arg(src).arg(&offset).arg(&mut out);
        // SAFETY: the kernel bounds each thread by n; check() proves the input
        // extent, and output spans exactly this layout. Cudarc tracks stream use.
        unsafe {
            b.launch(LaunchConfig::for_num_elems(n)).map_err(error)?;
        }
        drop(b);
        Ok((
            CudaStorage::wrap_cuda_slice(out, s.device.clone()),
            l.shape().clone(),
        ))
    }
}
struct Ternary {
    name: &'static str,
    rows: usize,
    input: usize,
    output: usize,
    shape: Shape,
}
impl CustomOp3 for Ternary {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core_0_11_0::bail!("CUDA-only frozen arithmetic")
    }
    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        la: &Layout,
        w: &CudaStorage,
        lw: &Layout,
        bias: &CudaStorage,
        lb: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (oa, ow, ob) = (check(a, la)?, check(w, lw)?, check(bias, lb)?);
        if a.device.id() != w.device.id() || a.device.id() != bias.device.id() {
            candle_core_0_11_0::bail!("CUDA operand device mismatch")
        }
        let (rows, input, output) = (count(self.rows)?, count(self.input)?, count(self.output)?);
        count(self.shape.elem_count())?;
        let threads = if self.name == "frozen_norm" {
            rows
        } else {
            count(self.shape.elem_count())?
        };
        let mut dst = a.device.alloc_zeros::<f32>(self.shape.elem_count())?;
        let f = function(a, self.name)?;
        let stream = a.device.cuda_stream();
        let mut builder = stream.launch_builder(&f);
        builder
            .arg(&rows)
            .arg(&input)
            .arg(&output)
            .arg(a.as_cuda_slice::<f32>()?)
            .arg(&oa)
            .arg(w.as_cuda_slice::<f32>()?)
            .arg(&ow)
            .arg(bias.as_cuda_slice::<f32>()?)
            .arg(&ob)
            .arg(&mut dst);
        // SAFETY: validated tensor dimensions establish rows*input input values,
        // complete weight/bias extents and rows*output (or norm width) output.
        // All offsets/extents and devices are checked; each kernel bounds threads.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(threads))
                .map_err(error)?;
        }
        drop(builder);
        Ok((
            CudaStorage::wrap_cuda_slice(dst, a.device.clone()),
            self.shape.clone(),
        ))
    }
}
struct Matmul {
    rows: usize,
    inner: usize,
    cols: usize,
}
impl CustomOp2 for Matmul {
    fn name(&self) -> &'static str {
        "frozen_matmul"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core_0_11_0::bail!("CUDA-only frozen arithmetic")
    }
    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        la: &Layout,
        b: &CudaStorage,
        lb: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (oa, ob) = (check(a, la)?, check(b, lb)?);
        if a.device.id() != b.device.id() {
            candle_core_0_11_0::bail!("CUDA operand device mismatch")
        }
        let (rows, inner, cols) = (count(self.rows)?, count(self.inner)?, count(self.cols)?);
        let shape = Shape::from((self.rows, self.cols));
        let n = count(shape.elem_count())?;
        let mut dst = a.device.alloc_zeros::<f32>(shape.elem_count())?;
        let f = function(a, "frozen_matmul")?;
        let stream = a.device.cuda_stream();
        let mut builder = stream.launch_builder(&f);
        builder
            .arg(&rows)
            .arg(&inner)
            .arg(&cols)
            .arg(a.as_cuda_slice::<f32>()?)
            .arg(&oa)
            .arg(b.as_cuda_slice::<f32>()?)
            .arg(&ob)
            .arg(&mut dst);
        // SAFETY: exact 2-D dimensions validate both operands; check() validates
        // contiguous F32 storage plus offsets; n threads write the exact output.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n))
                .map_err(error)?;
        }
        Ok((CudaStorage::wrap_cuda_slice(dst, a.device.clone()), shape))
    }
}
pub(super) fn unary(x: &Tensor, name: &'static str) -> anyhow::Result<Tensor> {
    anyhow::ensure!(
        matches!(name, "frozen_silu" | "frozen_tanh" | "frozen_softmax"),
        "unsupported frozen CUDA unary operation"
    );
    let width = x.dims().last().copied().unwrap_or(1);
    anyhow::ensure!(x.elem_count() > 0 && width > 0, "nonempty CUDA unary input");
    let rows = if name == "frozen_softmax" {
        anyhow::ensure!(x.dims2()?.1 == 8, "frozen softmax width");
        x.dims2()?.0
    } else {
        x.elem_count()
    };
    Ok(x.contiguous()?
        .apply_op1_no_bwd(&Unary { name, width, rows })?)
}
pub(super) fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
    let input = x.dim(candle_core_0_11_0::D::Minus1)?;
    let (output, k) = w.dims2()?;
    anyhow::ensure!(
        input > 0 && output > 0 && k == input && b.dims() == [output],
        "CUDA linear dimensions"
    );
    let rows = x.elem_count() / input;
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = output;
    Ok(x.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.contiguous()?,
        &Ternary {
            name: "frozen_linear",
            rows,
            input,
            output,
            shape: shape.into(),
        },
    )?)
}
pub(super) fn norm(x: &Tensor, w: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
    let width = x.dim(candle_core_0_11_0::D::Minus1)?;
    anyhow::ensure!(
        width > 0 && width <= 256 && w.dims() == [width] && b.dims() == [width],
        "frozen CUDA norm dimensions"
    );
    Ok(x.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.contiguous()?,
        &Ternary {
            name: "frozen_norm",
            rows: x.elem_count() / width,
            input: width,
            output: width,
            shape: x.shape().clone(),
        },
    )?)
}
pub(super) fn matmul(a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
    let (rows, inner) = a.dims2()?;
    let (k, cols) = b.dims2()?;
    anyhow::ensure!(
        rows > 0 && inner > 0 && cols > 0 && inner == k,
        "frozen CUDA matrix dimensions"
    );
    Ok(a.contiguous()?
        .apply_op2_no_bwd(&b.contiguous()?, &Matmul { rows, inner, cols })?)
}

#[cfg(test)]
pub(super) fn verify_frozen_primitives(device: &candle_core_0_11_0::Device) -> anyhow::Result<()> {
    use candle_core_0_11_0::Device;
    fn bits(x: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(x.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }
    for width in [116usize, 140, 256] {
        let x = Tensor::from_vec(
            (0..width)
                .map(|i| 10000f32 + ((i % 11) as f32 - 5.) * 0.125)
                .collect::<Vec<_>>(),
            (1, width),
            &Device::Cpu,
        )?;
        let w = Tensor::from_vec(
            (0..width)
                .map(|i| 1f32 + ((i % 7) as f32 - 3.) * 0.25)
                .collect::<Vec<_>>(),
            width,
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            (0..width)
                .map(|i| ((i % 5) as f32 - 2.) * 0.0625)
                .collect::<Vec<_>>(),
            width,
            &Device::Cpu,
        )?;
        anyhow::ensure!(
            bits(&norm(
                &x.to_device(device)?,
                &w.to_device(device)?,
                &b.to_device(device)?
            )?)? == bits(&super::cpu::norm(&x, &w, &b)?)?,
            "frozen CUDA norm width {width}"
        );
    }
    let x = Tensor::from_vec(vec![16_777_216f32, 1., -16_777_216.], (1, 3), &Device::Cpu)?;
    let w = Tensor::from_vec(vec![1f32, 1., 1., 1., -1., 1.], (2, 3), &Device::Cpu)?;
    let b = Tensor::from_vec(vec![1f32, 1.], 2, &Device::Cpu)?;
    anyhow::ensure!(
        bits(&linear(
            &x.to_device(device)?,
            &w.to_device(device)?,
            &b.to_device(device)?
        )?)? == bits(&super::cpu::linear(&x, &w, &b)?)?,
        "frozen CUDA dense cancellation"
    );
    let packed = Tensor::from_vec(
        (0..8 * 96)
            .map(|i| (i % 23) as f32 * 0.125 - 1.)
            .collect::<Vec<_>>(),
        (8, 96),
        &Device::Cpu,
    )?;
    let right = Tensor::from_vec(
        (0..32 * 8)
            .map(|i| (i % 17) as f32 * 0.25 - 2.)
            .collect::<Vec<_>>(),
        (32, 8),
        &Device::Cpu,
    )?;
    let a = packed.narrow(1, 32, 32)?;
    let gpu = packed.to_device(device)?.narrow(1, 32, 32)?;
    anyhow::ensure!(
        bits(&matmul(&gpu, &right.to_device(device)?)?)? == bits(&super::cpu::matmul(&a, &right)?)?,
        "frozen CUDA packed-slice matrix"
    );
    let x = Tensor::from_vec(
        vec![-100f32, -20., -1., -0., 0., 0.125, 0.5, 1., 10., 88.],
        10,
        &Device::Cpu,
    )?;
    for (name, expected) in [
        ("frozen_silu", super::cpu::silu(&x)?),
        ("frozen_tanh", super::cpu::tanh(&x)?),
    ] {
        anyhow::ensure!(
            bits(&unary(&x.to_device(device)?, name)?)? == bits(&expected)?,
            "frozen CUDA {name}"
        );
    }
    let x = Tensor::from_vec(
        vec![-20f32, -1., -0., 0., 0.125, 0.5, 1., 10.],
        (1, 8),
        &Device::Cpu,
    )?;
    anyhow::ensure!(
        bits(&unary(&x.to_device(device)?, "frozen_softmax")?)? == bits(&super::cpu::softmax(&x)?)?,
        "frozen CUDA softmax reduction"
    );
    println!("RNK12_CUDA_FROZEN_PRIMITIVES_PASS norms=3 cancellation=1 strided_matrix=1 unary=2 softmax=1");
    Ok(())
}
