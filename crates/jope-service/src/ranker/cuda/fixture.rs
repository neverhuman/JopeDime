//! Strict real-payload replay. No recapture, evaluator fits or optimizer updates.
use super::{
    compare_admission, compare_trace, forward::FusedForward, with_gpu_qualification, TensorView,
    Trace,
};
use crate::ranker::{
    adapter::{
        real_cpu_parity::{Fixtures, Stats},
        CpuAdapter, PredictCostsInputs,
    },
    bundle::FROZEN_DOPE_COMPOSITE,
};
use candle_core_0_11_0::Device;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::PathBuf, time::Instant};

fn view<'a>(shape: &'a [usize], values: &'a [f32]) -> TensorView<'a> {
    TensorView { shape, values }
}

fn replay(device: Device, authenticated: bool) {
    let started = Instant::now();
    let source = env::var("JOPE_SOURCE_SHA").expect("exact tested source SHA required");
    assert!(
        source.len() == 40
            && source
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    let fixtures = Fixtures::open();
    let out = PathBuf::from(
        env::var_os("JOPE_PARITY_REPORT_DIR").expect("external report directory required"),
    );
    fs::create_dir_all(&out).unwrap();
    let bundle = PathBuf::from(env::var_os("JOPE_PARITY_BUNDLE_DIR").unwrap());
    let dope = fs::read(bundle.join("dope.safetensors")).unwrap();
    let dope_sha = format!("{:x}", Sha256::digest(&dope));
    assert_eq!(
        dope_sha,
        "0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d"
    );
    let selected = match device.location() {
        candle_core_0_11_0::DeviceLocation::Cpu => "cpu".to_owned(),
        candle_core_0_11_0::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        other => panic!("unsupported replay device: {other:?}"),
    };
    let graph = if authenticated {
        let bundle = super::authenticated::load();
        assert_eq!(bundle.asset("dope.safetensors").unwrap(), dope);
        assert_eq!(
            bundle.asset("jope.safetensors").unwrap(),
            fixtures.weights()
        );
        let request = match device.location() {
            candle_core_0_11_0::DeviceLocation::Cpu => jope_runtime::backend::DeviceRequest::Cpu,
            candle_core_0_11_0::DeviceLocation::Cuda { gpu_id } => {
                jope_runtime::backend::DeviceRequest::Cuda(gpu_id as u32)
            }
            other => panic!("unsupported replay device: {other:?}"),
        };
        FusedForward::from_bundle(&bundle, request, 8).unwrap()
    } else {
        FusedForward::from_bytes(&dope, &fixtures.weights(), device, 8).unwrap()
    };
    assert_eq!(
        serde_json::to_value(graph.placement()).unwrap(),
        serde_json::json!({
            "dope": selected, "ranker": selected, "evaluation": "cpu", "workers": 8
        }),
        "placement must describe the instantiated neural device and CPU evaluation"
    );
    let cpu_reference =
        CpuAdapter::from_bytes_on_device(&fixtures.weights(), &Device::Cpu).unwrap();
    let mut stages: BTreeMap<&str, Stats> = BTreeMap::new();
    let (mut slates, mut ties, mut decisive) = (0, 0, 0);
    let mut trace_errors = Vec::new();
    let mut admission_failures = 0;
    let mut max_kl = 0f64;
    for cid in &fixtures.contexts {
        let reference = fixtures.load(&format!("contexts/{cid}.safetensors"));
        let v = reference.f32("v256");
        let col = reference.f32("col");
        let actual = graph
            .stages(&v, &col)
            .unwrap_or_else(|e| panic!("{cid} forward: {e:#}"));
        fs::write(
            out.join(format!("stages-{cid}.json")),
            serde_json::to_vec(&serde_json::json!({
                "source": &source, "context": cid, "v256": &actual.v256, "col": &actual.col,
                "dope629": &actual.dope629, "lm48": &actual.lm48, "lm48_z": &actual.lm48_z,
                "tokens": &actual.tokens, "mask": &actual.mask,
            }))
            .unwrap(),
        )
        .unwrap();
        let dope = reference.f32("dope629");
        let lm48 = reference.f32("lm48");
        let lm48_z = reference.f32("lm48_z");
        let tokens = reference.f32("tokens");
        let mask = reference.f32("mask");
        for (name, a, b) in [
            ("v256", &actual.v256, &v),
            ("col", &actual.col, &col),
            ("dope629", &actual.dope629, &dope),
            ("lm48", &actual.lm48, &lm48),
            ("lm48_z", &actual.lm48_z, &lm48_z),
            ("tokens", &actual.tokens, &tokens),
            ("mask", &actual.mask, &mask),
        ] {
            stages
                .entry(name)
                .or_default()
                .compare(a, &b.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
        }
        for width in [12, 20] {
            let f = fixtures.load(&format!("slates/{cid}-{width}.safetensors"));
            let prog = f.f32("input.prog_412");
            let proj = f.f32("input.proj_16");
            let indices = f.i64("input.revealed_indices");
            let fitness = f.f32("input.revealed_raw_f");
            let gates = f.f32("input.revealed_gate");
            let ctx = f.f32("input.ctx_feat");
            let prior = f.f32("input.base_prior_cost");
            let sos = f.f32("input.sos_prior");
            let scale = f.f32("input.scale");
            let inputs = PredictCostsInputs {
                batch: 1,
                candidates: width,
                prog_412: &prog,
                proj_16: &proj,
                tokens: &actual.tokens,
                token_mask: &actual.mask,
                revealed_indices: &indices,
                revealed_fitness: &fitness,
                revealed_gates: &gates,
                ctx_feat: &ctx,
                base_prior_cost: &prior,
                sos_prior: &sos,
                scale: scale[0],
            };
            let output = graph.predict_costs(inputs).unwrap();
            let cpu = cpu_reference
                .predict_costs(PredictCostsInputs {
                    tokens: &tokens,
                    token_mask: &mask,
                    ..inputs
                })
                .unwrap();
            stages.entry("anchor_costs").or_default().compare(
                &output.anchor_cost,
                &cpu.anchor_cost
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect::<Vec<_>>(),
            );
            stages.entry("proposal_KL").or_default().compare(
                &output
                    .proposal_kl_to_anchor
                    .iter()
                    .map(|v| *v as f32)
                    .collect::<Vec<_>>(),
                &cpu.proposal_kl_to_anchor,
            );
            let costs = f.f32("output.total_cost");
            stages.entry("costs").or_default().compare(
                &output.total_cost,
                &costs.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            );
            let cost_shape = [1, width];
            let reference_trace = Trace {
                context_id: cid,
                v256: view(&[1, 256], &v),
                col: view(&[1, 64, 17], &col),
                dope629: view(&[1, 629], &dope),
                lm48: view(&[1, 48], &lm48),
                tokens: view(&[1, 10, 32], &tokens),
                mask: view(&[1, 10], &mask),
                costs: view(&cost_shape, &costs),
            };
            let actual_trace = Trace {
                context_id: cid,
                v256: view(&[1, 256], &actual.v256),
                col: view(&[1, 64, 17], &actual.col),
                dope629: view(&[1, 629], &actual.dope629),
                lm48: view(&[1, 48], &actual.lm48),
                tokens: view(&[1, 10, 32], &actual.tokens),
                mask: view(&[1, 10], &actual.mask),
                costs: view(&cost_shape, &output.total_cost),
            };
            if let Err(error) = compare_trace(reference_trace, actual_trace) {
                trace_errors.push(format!("{cid}/{width}: {error:#}"));
            }
            match compare_admission(&costs, &output.total_cost, width == 20, [42; 32]) {
                Ok(admission) if admission.ambiguous_boundary => ties += 1,
                Ok(_) => decisive += 1,
                Err(_) => admission_failures += 1,
            }
            max_kl = max_kl.max(output.proposal_kl_to_anchor[0]);
            slates += 1;
        }
    }
    graph.device_sync_for_test();
    let stage_rows:serde_json::Value=stages.iter().map(|(name,s)|(name.to_string(),serde_json::json!({"values":s.values,"failed_values":s.failures,"max_absolute_error":s.max_abs,"max_tolerance_ratio":s.max_tolerance_ratio}))).collect();
    let failed: usize = stages.values().map(|s| s.failures).sum();
    let receipt = serde_json::json!({"source":source,"consumer":"RNK-12","authenticated_release":authenticated,"placement":graph.placement(),"dope_logical_composite":FROZEN_DOPE_COMPOSITE,"dope_weights_sha256":dope_sha,"fixture_index_sha256":env::var("JOPE_GOLDEN_INDEX_SHA256").unwrap(),"contexts":fixtures.contexts.len(),"slates":slates,"stages":stage_rows,"atol":1e-5,"rtol":1e-4,"failed_values":failed,"trace_errors":trace_errors,"decisive_admission_slates":decisive,"tie_band_slates":ties,"admitted_set_mismatches_outside_tie_band":admission_failures,"max_proposal_KL":max_kl,"elapsed_seconds":started.elapsed().as_secs_f64(),"passed":failed==0 && trace_errors.is_empty() && admission_failures==0,"native_evaluator_calls":0,"optimizer_updates":0,"scope":"recorded real DL-Train inputs/reference replay; no Q1/default promotion"});
    fs::write(
        out.join("RNK-12.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    println!("{receipt}");
    assert_eq!(slates, 100);
    assert!(trace_errors.is_empty());
    assert_eq!(admission_failures, 0);
    assert_eq!(failed, 0, "stage parity failed; receipt retained");
}

#[test]
#[ignore = "requires hash-pinned external RES-07 real inputs, references and frozen bundle; execute explicitly"]
fn real_full_forward_cpu_fixture_replay() {
    replay(Device::Cpu, false);
}

#[test]
#[ignore = "requires the exact release-verified S1 r782 assets and pinned real M2 capture"]
fn authenticated_real_full_forward_cpu_fixture_replay() {
    replay(Device::Cpu, true);
}

#[cfg(unix)]
pub(crate) fn inherited_lease<T>(
    lease: fs::File,
    lock: &std::path::Path,
    run: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    use std::os::unix::fs::MetadataExt;
    let actual = lease.metadata()?;
    let expected = fs::metadata(lock)?;
    anyhow::ensure!(
        actual.is_file() && (actual.dev(), actual.ino()) == (expected.dev(), expected.ino()),
        "inherited qualification descriptor differs from existing lock"
    );
    lease
        .try_lock()
        .map_err(|e| anyhow::anyhow!("exclusive inherited qualification lease unavailable: {e}"))?;
    let contender = fs::OpenOptions::new().read(true).write(true).open(lock)?;
    anyhow::ensure!(
        contender.try_lock().is_err(),
        "qualification flock is not exclusive"
    );
    run()
}

#[cfg(unix)]
#[test]
fn inherited_lease_covers_callback_failure_without_relocking_a_new_description() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().lock().unwrap();
    let result: anyhow::Result<()> =
        inherited_lease(file.as_file().try_clone().unwrap(), file.path(), || {
            let contender = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.path())
                .unwrap();
            assert!(contender.try_lock().is_err());
            anyhow::bail!("owned callback control failure")
        });
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("callback control failure"));
    file.as_file().unlock().unwrap();
    let contender = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(file.path())
        .unwrap();
    assert!(contender.try_lock().is_ok());
}

#[cfg(unix)]
#[test]
fn inherited_lease_rejects_another_inode_before_callback() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let other = tempfile::NamedTempFile::new().unwrap();
    let invoked = std::cell::Cell::new(false);
    assert!(
        inherited_lease(other.as_file().try_clone().unwrap(), file.path(), || {
            invoked.set(true);
            Ok(())
        })
        .is_err()
    );
    assert!(!invoked.get());
}

#[test]
#[ignore = "requires fused-cuda, qualified GPU and exclusive existing qualification.lock; execute explicitly after merge"]
fn real_cuda_m2_fixture_replay() {
    cuda_replay(false);
}

#[test]
#[ignore = "requires release-verified S1 r782, fused-cuda and the exclusive existing qualification.lock"]
fn authenticated_real_cuda_m2_fixture_replay() {
    cuda_replay(true);
}

fn cuda_replay(authenticated: bool) {
    let lock = PathBuf::from(
        env::var_os("JOPE_QUALIFICATION_LOCK").expect("existing qualification lock path required"),
    );
    let run = || {
        let device =
            Device::new_cuda(0).map_err(|e| anyhow::anyhow!("BACKEND_UNAVAILABLE: {e}"))?;
        #[cfg(feature = "fused-cuda")]
        super::deterministic::verify_frozen_primitives(&device)?;
        replay(device, authenticated);
        Ok(())
    };
    if env::var_os("JOPE_GPU_LEASE_STDIN").is_some() {
        // The documented outer flock locks inherited stdin. Clone that same
        // open-file description; opening the path again would self-deadlock.
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            let lease = fs::File::from(std::io::stdin().as_fd().try_clone_to_owned().unwrap());
            inherited_lease(lease, &lock, run).unwrap();
        }
        #[cfg(not(unix))]
        panic!("inherited flock qualification requires the declared Unix GPU host");
    } else {
        with_gpu_qualification(&lock, run).unwrap();
    }
}
