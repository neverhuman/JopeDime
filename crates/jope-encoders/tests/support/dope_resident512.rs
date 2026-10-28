//! Released-checkpoint compatibility and bounded timing, not native coverage or a cold latency gate.
use super::{DopeDevice, DopeEncoder, Tensor, WEIGHTS};
use anyhow::{bail, Result};
use candle_core::DType;
use dope_core::mlc_v3::MlcV3Model;
use dope_core::prepare::{build_conditioning, prepare_table_with_ids, DopeConditioning, PrepTask};
use starforge::dope_encoder::{
    conditioning_digest, BundleScope, DopeCheckpointIdentity, LandmarkBundle, LandmarkOptions,
    LandmarkScope, DOPE512_RECIPE, DOPE_WEIGHTS_SHA256,
};
use std::hint::black_box;
use std::time::Instant;

struct Harness {
    encoder: DopeEncoder,
    oracle: MlcV3Model,
    pool: rayon::ThreadPool,
}
impl Harness {
    fn new(device: &DopeDevice) -> Result<Self> {
        let oracle = MlcV3Model::from_safetensors_bytes_on_device(WEIGHTS, device)?;
        assert_eq!(oracle.inventory.tensor_count, 56);
        assert_eq!(oracle.inventory.parameter_count, 444_168);
        Ok(Self {
            encoder: DopeEncoder::from_bytes(WEIGHTS, device)?,
            oracle,
            pool: rayon::ThreadPoolBuilder::new().num_threads(1).build()?,
        })
    }
    fn full(&self, x: &Tensor, columns: &Tensor) -> Result<Tensor> {
        let run = || self.oracle.encode_features_629_tensor(x, columns, "high");
        Ok(if self.encoder.device().is_cpu() {
            self.pool.install(run)?
        } else {
            run()?
        })
    }
    // Reproduce the old host path, including packing/upload, crop/copy, and product metadata.
    fn legacy_host(&self, input: &DopeConditioning) -> Result<LegacyHost> {
        let mut columns = vec![0f32; 64 * 17];
        for (dst, src) in columns
            .chunks_mut(17)
            .zip(input.col_landmarks.iter().take(64))
        {
            dst.copy_from_slice(src);
        }
        let device = self.encoder.device();
        let x = Tensor::from_slice(&input.vec256, (1, 256), device)?;
        let columns = Tensor::from_vec(columns, (1, 64, 17), device)?;
        Ok(LegacyHost {
            tensor: self.full(&x, &columns)?.narrow(1, 0, 512)?.contiguous()?,
            digest: conditioning_digest(input),
            source: input.bundle.source_digest.clone(),
            checkpoint: self.encoder.checkpoint_identity().clone(),
        })
    }
}
struct LegacyHost {
    tensor: Tensor,
    digest: String,
    source: String,
    checkpoint: DopeCheckpointIdentity,
}
fn inputs(batch: usize, device: &DopeDevice) -> Result<(Tensor, Tensor)> {
    // Synthetic neural inputs make no claim of observed native-landmark evidence.
    let x = (0..batch * 256)
        .map(|i| ((i % 151) as f32 - 75.) * 0.021)
        .collect::<Vec<_>>();
    let columns = (0..batch * 64 * 17)
        .map(|i| ((i % 97) as f32 - 48.) * 0.01)
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_vec(x, (batch, 256), device)?,
        Tensor::from_vec(columns, (batch, 64, 17), device)?,
    ))
}
fn bits(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor
        .to_device(&DopeDevice::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}
fn assert_projection(h: &Harness, x: &Tensor, columns: &Tensor) -> Result<Tensor> {
    let full = h.full(x, columns)?;
    let got = h.encoder.encode_tensors(x, columns)?;
    assert_eq!(full.dims(), [x.dim(0)?, 629]);
    assert_eq!(got.dims(), [x.dim(0)?, 512]);
    assert_eq!(got.dtype(), DType::F32);
    assert!(got.device().same_device(h.encoder.device()));
    assert_eq!(bits(&got)?, bits(&full.narrow(1, 0, 512)?)?);
    Ok(got)
}
fn resident_contract(device: &DopeDevice) -> Result<()> {
    let h = Harness::new(device)?;
    for batch in [1, 8, 64, 256] {
        let (x, columns) = inputs(batch, device)?;
        assert_projection(&h, &x, &columns)?;
    }
    let (x, c) = inputs(1, device)?;
    let (x2, _) = inputs(2, device)?;
    let mut bad = vec![
        (x.reshape((256,))?, c.clone()),
        (x.reshape((1, 1, 256))?, c.clone()),
        (x.narrow(1, 0, 255)?, c.clone()),
        (x.clone(), c.reshape((64, 17))?),
        (x.clone(), c.narrow(1, 0, 63)?),
        (x.clone(), c.narrow(2, 0, 16)?),
        (x2, c.clone()),
        (x.to_dtype(DType::F64)?, c.clone()),
        (x.clone(), c.to_dtype(DType::F64)?),
    ];
    if device.is_cpu() {
        bad.push((
            Tensor::zeros((0, 256), DType::F32, device)?,
            Tensor::zeros((0, 64, 17), DType::F32, device)?,
        ));
    }
    for (bad_x, bad_c) in bad {
        assert!(h.full(&bad_x, &bad_c).is_err());
        assert!(h.encoder.encode_tensors(&bad_x, &bad_c).is_err());
    }
    if device.is_cuda() {
        let (cpu_x, cpu_c) = inputs(1, &DopeDevice::Cpu)?;
        for (bad_x, bad_c) in [(cpu_x, c.clone()), (x.clone(), cpu_c)] {
            assert!(h.full(&bad_x, &bad_c).is_err());
            assert!(h.encoder.encode_tensors(&bad_x, &bad_c).is_err());
        }
    }
    let mut dirty = vec![0f32; 256];
    dirty[..4].copy_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.]);
    let dirty = Tensor::from_vec(dirty, (1, 256), device)?;
    let clean = assert_projection(&h, &dirty, &c)?;
    let cleaned_bits = bits(&clean)?;
    assert_eq!(&cleaned_bits[..4], &[0, 0, 0, (-0f32).to_bits()]);
    assert!(cleaned_bits.iter().all(|v| f32::from_bits(*v).is_finite()));
    let dirty_c = Tensor::from_vec(vec![f32::NAN; 64 * 17], (1, 64, 17), device)?;
    assert_eq!(
        bits(&assert_projection(&h, &x, &dirty_c)?)?,
        bits(&assert_projection(&h, &x, &c)?)?
    );
    let input_bits = (bits(&x)?, bits(&c)?);
    let a = h.encoder.encode_tensors(&x, &c)?;
    let expected_a = bits(&a)?;
    let b = h.encoder.encode_tensors(&x.affine(1., 0.5)?, &c)?;
    let expected_b = bits(&b)?;
    assert_ne!(expected_a, expected_b);
    let again = h.encoder.encode_tensors(&x, &c)?;
    assert_eq!(bits(&a)?, expected_a);
    assert_eq!(bits(&again)?, expected_a);
    assert_eq!(bits(&b)?, expected_b);
    a.zero_set()?;
    assert_eq!((bits(&x)?, bits(&c)?), input_bits);
    assert_eq!(bits(&again)?, expected_a);
    assert_eq!(bits(&h.encoder.encode_tensors(&x, &c)?)?, expected_a);
    x.zero_set()?;
    c.zero_set()?;
    assert_eq!(bits(&again)?, expected_a);
    assert_eq!(bits(&b)?, expected_b);
    Ok(())
}
fn none_conditioning(task: PrepTask) -> Result<DopeConditioning> {
    use dope_core::landmark::bundle::LandmarkMode;
    let x = (0..12)
        .map(|i| vec![i as f64, (i % 3) as f64])
        .collect::<Vec<_>>();
    let y = (0..12)
        .map(|i| match task {
            PrepTask::Regression => (i * i) as f64,
            PrepTask::Classification => (i % 2) as f64,
            PrepTask::Multiclass => (i % 3) as f64,
        })
        .collect::<Vec<_>>();
    let table = prepare_table_with_ids(&x, &y, task, (0..12).map(|i| 100 + i * 3).collect())?;
    let scope = LandmarkScope::for_source_digest(
        &LandmarkOptions::default(),
        None,
        table.source_rows_digest(),
    );
    let zero =
        LandmarkBundle::from_squashed(&[0.; 23], &[false; 23], BundleScope::new(scope.digest()));
    let input = build_conditioning(&table, &zero, None, LandmarkMode::None)?;
    assert!(input.bundle.mask.iter().all(|v| !v));
    Ok(input)
}
fn assert_host(h: &Harness, input: &DopeConditioning) -> Result<Vec<u32>> {
    let old = h.legacy_host(input)?;
    let got = h.encoder.encode_conditioning(input)?;
    let tensor = got.tensor_copy()?;
    assert_eq!(tensor.dims(), [1, 512]);
    assert!(tensor.device().same_device(h.encoder.device()));
    assert_eq!(bits(&tensor)?, bits(&old.tensor)?);
    assert_eq!(got.source_digest(), old.source.as_str());
    assert_eq!(got.conditioning_digest(), old.digest.as_str());
    assert_eq!(got.checkpoint_identity(), &old.checkpoint);
    assert_eq!(
        got.checkpoint_identity(),
        &DopeCheckpointIdentity::ReleasedV1
    );
    assert_eq!(got.checkpoint_identity().embedding_recipe(), DOPE512_RECIPE);
    assert_eq!(
        got.checkpoint_identity().weights_sha256(),
        DOPE_WEIGHTS_SHA256
    );
    bits(&tensor)
}
fn host_contract(device: &DopeDevice) -> Result<()> {
    let h = Harness::new(device)?;
    for task in [
        PrepTask::Regression,
        PrepTask::Classification,
        PrepTask::Multiclass,
    ] {
        let input = none_conditioning(task)?;
        let original = input.clone();
        for rows in [0, 3, 64, 70] {
            let mut variant = input.clone();
            variant.col_landmarks = (0..rows).map(|i| [i as f32 * 0.01; 17]).collect();
            assert_host(&h, &variant)?;
        }
        let mut empty = input.clone();
        empty.col_landmarks.clear();
        let mut padded = empty.clone();
        padded.col_landmarks = vec![[0.; 17]; 64];
        let expected = assert_host(&h, &empty)?;
        assert_eq!(expected, assert_host(&h, &padded)?);
        assert_eq!(empty.digest(), padded.digest());
        let mut changed = padded.clone();
        changed.col_landmarks[0][0] = 0.5;
        assert_eq!(expected, assert_host(&h, &changed)?);
        assert_ne!(padded.digest(), changed.digest());
        let mut truncated = padded.clone();
        truncated.col_landmarks.extend([[1.; 17], [2.; 17]]);
        assert_eq!(expected, assert_host(&h, &truncated)?);
        assert_eq!(padded.digest(), truncated.digest());
        assert_eq!(input, original);
    }
    Ok(())
}
#[test]
fn released_cpu_resident512_matches_full629_contract() -> Result<()> {
    resident_contract(&DopeDevice::Cpu)
}
#[test]
fn released_cpu_host_conditioning_preserves_legacy_pack_and_identity() -> Result<()> {
    host_contract(&DopeDevice::Cpu)
}
#[cfg(feature = "dope-cuda")]
#[test]
fn released_cuda_resident512_matches_full629_contract() -> Result<()> {
    let device = DopeEncoder::cuda(0)?;
    assert!(device.is_cuda());
    resident_contract(&device)
}
#[cfg(feature = "dope-cuda")]
#[test]
fn released_cuda_host_conditioning_preserves_legacy_pack_and_identity() -> Result<()> {
    let device = DopeEncoder::cuda(0)?;
    assert!(device.is_cuda());
    host_contract(&device)
}
fn micros(device: &DopeDevice, run: &mut impl FnMut() -> Result<()>) -> Result<f64> {
    device.synchronize()?;
    let start = Instant::now();
    run()?;
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6)
}
fn summary(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    serde_json::json!({"p50_us":sorted[(sorted.len()*50).div_ceil(100)-1],
        "p95_us":sorted[(sorted.len()*95).div_ceil(100)-1],"samples_us":samples})
}
fn bench_pair(
    device: &DopeDevice,
    case: &str,
    batch: usize,
    mut old: impl FnMut() -> Result<()>,
    mut current: impl FnMut() -> Result<()>,
) -> Result<()> {
    for _ in 0..3 {
        old()?;
        current()?;
    }
    device.synchronize()?;
    let (mut baseline, mut product) = (Vec::new(), Vec::new());
    for i in 0..21 {
        if i % 2 == 0 {
            baseline.push(micros(device, &mut old)?);
            product.push(micros(device, &mut current)?);
        } else {
            product.push(micros(device, &mut current)?);
            baseline.push(micros(device, &mut old)?);
        }
    }
    println!(
        "\n{}",
        serde_json::json!({"schema":"jope.dope-resident512-benchmark/v1",
        "case":case,"device":format!("{:?}",device.location()),"batch":batch,
        "cpu_pool_threads":if device.is_cpu() { 1 } else { 0 },
        "host_math_threads":1,"trials":21,"warmup_pairs":3,"paired_order":"alternating",
        "timing":"pre-synchronize excluded; call and completion synchronization timed",
        "weights_sha256":DOPE_WEIGHTS_SHA256,"checkpoint":"released-v1",
        "baseline_method":"retained full629 oracle + crop; host case also packs/uploads and binds metadata",
        "current":"public DopeEncoder API at the source under test",
        "loading_and_preparation_included":false,"cold_5x_gate_claimed":false,
        "native_evidence_observed":false,"baseline":summary(&baseline),"product":summary(&product)})
    );
    Ok(())
}
#[test]
#[ignore = "explicit bounded benchmark; choose JOPE_DOPE_BENCH_DEVICE=cpu|cuda and reserve hardware"]
fn benchmark_released_resident512_matched() -> Result<()> {
    let device = match std::env::var("JOPE_DOPE_BENCH_DEVICE").as_deref() {
        Ok("cpu") => DopeDevice::Cpu,
        Ok("cuda") => DopeEncoder::cuda(0)?,
        _ => bail!("set JOPE_DOPE_BENCH_DEVICE=cpu or cuda explicitly"),
    };
    let h = Harness::new(&device)?;
    for batch in [1, 8, 64, 256] {
        let (x, c) = inputs(batch, &device)?;
        assert_projection(&h, &x, &c)?;
        bench_pair(
            &device,
            "resident",
            batch,
            || {
                black_box(h.full(&x, &c)?.narrow(1, 0, 512)?.contiguous()?);
                Ok(())
            },
            || {
                black_box(h.encoder.encode_tensors(&x, &c)?);
                Ok(())
            },
        )?;
    }
    let input = none_conditioning(PrepTask::Regression)?;
    assert_host(&h, &input)?;
    bench_pair(
        &device,
        "host-conditioning-public-call",
        1,
        || {
            black_box(h.legacy_host(&input)?);
            Ok(())
        },
        || {
            black_box(h.encoder.encode_conditioning(&input)?);
            Ok(())
        },
    )
}
