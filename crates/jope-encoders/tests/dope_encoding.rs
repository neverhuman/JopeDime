use candle_core::Tensor;
use starforge::dope_encoder::{DopeDevice, DopeEncoder, FusionComponents};

const WEIGHTS: &[u8] =
    include_bytes!("../../../deploy/appliance/assets/dope-mlc-v3/model.safetensors");

#[path = "support/dope_consumers.rs"]
mod consumers;

#[path = "support/dope_product_sessions.rs"]
mod product_sessions;

#[path = "support/dope_learned_product.rs"]
mod learned_product;

#[path = "support/dope_projection.rs"]
mod projection;

#[path = "support/dope_resident512.rs"]
mod resident512;

#[test]
fn whole_checkpoint_checksum_and_cpu_reference_projection() {
    assert!(DopeEncoder::from_bytes(b"wrong checkpoint", &DopeDevice::Cpu).is_err());
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    reference_projection(&encoder);
}

fn reference_projection(encoder: &DopeEncoder) {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dope_cuda_reference.json")).unwrap();
    let desc: Vec<Vec<f32>> = serde_json::from_value(fixture["vec"].clone()).unwrap();
    let cols: Vec<Vec<Vec<f32>>> = serde_json::from_value(fixture["columns"].clone()).unwrap();
    let expected: Vec<Vec<f32>> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let descriptor = Tensor::from_vec(
        desc.into_iter().flatten().collect::<Vec<_>>(),
        (4, 256),
        encoder.device(),
    )
    .unwrap();
    let columns = Tensor::from_vec(
        cols.into_iter().flatten().flatten().collect::<Vec<_>>(),
        (4, 64, 17),
        encoder.device(),
    )
    .unwrap();
    let output = encoder.encode_tensors(&descriptor, &columns).unwrap();
    assert_eq!(output.device().location(), encoder.device().location());
    assert_eq!(output.dims(), [4, 512]);
    let values = output.to_vec2::<f32>().unwrap();
    let max = values
        .iter()
        .zip(expected.iter())
        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
        .fold(0f32, f32::max);
    eprintln!(
        "Dope512 parity device={:?} max_abs_diff={max:.9e}",
        encoder.device().location()
    );
    assert!(max < 3e-5, "Python CUDA parity difference {max}");
    assert!(values.iter().flatten().all(|v| v.is_finite()));
}

#[cfg(feature = "dope-cuda")]
#[test]
fn actual_cuda_projection_stays_on_device_and_matches_whole_model_reference() {
    let device = DopeEncoder::cuda(0).expect("CUDA qualification must execute on a GPU");
    assert!(device.is_cuda());
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &device).unwrap();
    reference_projection(&encoder);
}

#[test]
fn alternative_fusion_uses_its_own_identity_and_checks_scope() {
    use starforge::dope_encoder::{
        prepare_table_with_ids, LandmarkOptions, LandmarkScope, PrepTask,
    };
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let x = (0..8).map(|i| vec![i as f64]).collect::<Vec<_>>();
    let y = (0..8).map(|i| (i * i) as f64).collect::<Vec<_>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, (0..8).collect()).unwrap();
    let scope = LandmarkScope::for_source_digest(
        &LandmarkOptions::default(),
        None,
        prepared.source_rows_digest(),
    );
    let embedding = encoder.encode(&prepared, None, &scope).unwrap();
    let mut components = FusionComponents {
        source_digest: embedding.source_digest().into(),
        recipe: "supplied-foundation-target/v1".into(),
        foundation1024: vec![1.; 1024],
        target512: vec![2.; 512],
    };
    let resident = embedding.assemble(&components).unwrap();
    assert_eq!(resident.dataset1536_copy().unwrap().dims(), [1, 1536]);
    let fused = resident.to_export().unwrap();
    assert_eq!(fused.dataset1536.len(), 1536);
    assert_eq!(fused.policy2048.len(), 2048);
    assert_eq!(fused.dataset1536[..512], embedding.to_vec().unwrap());
    assert!(fused.policy2048[512..1536].iter().all(|v| *v == 1.));
    assert!(fused.policy2048[1536..].iter().all(|v| *v == 2.));
    assert!(fused.recipe.starts_with("dope512-"));
    // A Tensor clone shares mutable storage; only independent copies cross the public API.
    embedding.tensor_copy().unwrap().zero_set().unwrap();
    resident.dataset1536_copy().unwrap().zero_set().unwrap();
    resident.policy2048_copy().unwrap().zero_set().unwrap();
    assert_eq!(embedding.to_vec().unwrap(), fused.dataset1536[..512]);
    assert_eq!(resident.to_export().unwrap().policy2048, fused.policy2048);
    components.source_digest = "c".repeat(64);
    assert!(embedding.assemble(&components).is_err());
}

#[test]
fn shared_bundle_serialization_fixture_uses_the_upstream_contract() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/landmark_bundle_v1.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let bundle: starforge::dope_encoder::LandmarkBundle =
            serde_json::from_value(case["bundle"].clone()).unwrap();
        assert_eq!(
            bundle.conditioning_digest(),
            case["conditioning_digest"].as_str().unwrap()
        );
        let roundtrip: starforge::dope_encoder::LandmarkBundle =
            serde_json::from_value(serde_json::to_value(&bundle).unwrap()).unwrap();
        assert_eq!(bundle, roundtrip);
    }
}

#[test]
fn progressive_terminal_equals_full_and_publishes_supplied_fusion() {
    use starforge::dope_encoder::{
        collect, prepare_table_with_ids, start_training_with_options, Head, LandmarkArgs,
        LandmarkMode, LandmarkOptions, LandmarkScope, PrepTask, SessionOptions, SessionOutcome,
        SubsampleSpec, TrainingInput,
    };
    let x = (0..80)
        .map(|i| vec![(i as f64 * 0.31).sin(), (i % 7) as f64, i as f64 / 80.])
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| 2. * r[0] + 0.03 * r[1] + r[2] * r[2])
        .collect::<Vec<_>>();
    let ids = (0..80).map(|i| 1000 + i * 3).collect::<Vec<u64>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, ids.clone()).unwrap();
    let opts = LandmarkOptions {
        work_budget: 2_000_000_000_000,
        ..Default::default()
    };
    let scope = LandmarkScope::for_source_digest(
        &opts,
        Some(SubsampleSpec::new(40, opts.seed)),
        prepared.source_rows_digest(),
    );
    let input = TrainingInput::numeric(x, y, ids, Head::Regression);
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::Progressive),
        landmark_subsample: Some(40),
        ..Default::default()
    };
    let components = FusionComponents {
        source_digest: scope.digest(),
        recipe: "provided-reference-components/v1".into(),
        foundation1024: vec![1.; 1024],
        target512: vec![2.; 512],
    };
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let session = start_training_with_options(
        input.clone(),
        encoder.clone(),
        &args,
        SessionOptions {
            cancel: None,
            components: Some(components),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(session.initial().receipt.version, 0);
    assert!(session.initial().receipt.mask.iter().all(|v| !*v));
    let mut observed = Vec::new();
    let progressive = collect(session, |s| {
        assert!(s.fusion.is_some());
        assert_eq!(
            s.fusion.as_ref().unwrap().policy2048_copy().unwrap().dims(),
            [1, 2048]
        );
        observed.push(s.receipt.conditioning_digest.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(progressive.outcome, SessionOutcome::Complete);
    assert!(observed.len() >= 2);
    assert!(observed.windows(2).all(|pair| pair[0] != pair[1]));
    let full_args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::Full),
        ..args
    };
    let full =
        start_training_with_options(input, encoder, &full_args, SessionOptions::default()).unwrap();
    let full = collect(full, |_| Ok(())).unwrap();
    assert_eq!(full.outcome, SessionOutcome::Complete);
    assert_eq!(
        progressive.final_snapshot.receipt.conditioning_digest,
        full.final_snapshot.receipt.conditioning_digest
    );
    assert_eq!(
        progressive.final_snapshot.embedding.to_vec().unwrap(),
        full.final_snapshot.embedding.to_vec().unwrap()
    );
    assert!(progressive.final_snapshot.receipt.embedding.is_some());
}

#[test]
fn none_and_fast_retain_small_table_descriptors_without_native_admission() {
    use starforge::dope_encoder::{
        collect, start_training, Head, LandmarkArgs, LandmarkMode, SessionOutcome, TrainingInput,
    };
    let x = (0..8)
        .map(|i| vec![i as f64, (i % 2) as f64])
        .collect::<Vec<_>>();
    let input = TrainingInput::numeric(
        x,
        vec![0., 1., 2., 3., 4., 6., 10., 30.],
        (0..8).map(|i| 100 + i).collect(),
        Head::Regression,
    );
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let mut results = Vec::new();
    for mode in [LandmarkMode::None, LandmarkMode::Fast] {
        let args = LandmarkArgs {
            landmark_mode: Some(mode),
            ..Default::default()
        };
        let session = start_training(input.clone(), encoder.clone(), &args).unwrap();
        let result = collect(session, |_| Ok(())).unwrap();
        assert_eq!(result.outcome, SessionOutcome::Complete);
        assert_eq!(result.receipts.len(), 1);
        let vector = result.final_snapshot.embedding.to_vec().unwrap();
        assert!(vector.iter().all(|v| v.is_finite()));
        assert!(vector[..85].iter().all(|v| *v == 0.));
        assert!(vector[85..87].iter().any(|v| *v != 0.));
        assert!(vector[91..116].iter().all(|v| *v == 0.));
        assert!(vector[140..256].iter().all(|v| *v == 0.));
        results.push(result);
    }
    assert_eq!(
        results[0].final_snapshot.receipt.conditioning_digest,
        results[1].final_snapshot.receipt.conditioning_digest
    );
    assert_eq!(
        results[0].final_snapshot.embedding.to_vec().unwrap(),
        results[1].final_snapshot.embedding.to_vec().unwrap()
    );
}

#[test]
fn external_csv_identity_does_not_erase_task_or_row_scope() {
    use starforge::dope_encoder::{
        start_training, Head, LandmarkArgs, LandmarkMode, TrainingInput,
    };
    let x = (0..8)
        .map(|i| vec![i as f64, (i % 3) as f64])
        .collect::<Vec<_>>();
    let y = (0..8).map(|i| (i % 2) as f64).collect::<Vec<_>>();
    let mut input = TrainingInput::numeric(x, y, (0..8).collect(), Head::Regression);
    input.source_digest = Some("a".repeat(64));
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::None),
        ..Default::default()
    };
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let regression = start_training(input.clone(), encoder.clone(), &args).unwrap();
    input.head = Head::Binary;
    let binary = start_training(input.clone(), encoder.clone(), &args).unwrap();
    input.ids[0] = 99;
    let different_rows = start_training(input, encoder, &args).unwrap();
    assert_ne!(
        regression.initial().receipt.source_digest,
        binary.initial().receipt.source_digest
    );
    assert_ne!(
        binary.initial().receipt.source_digest,
        different_rows.initial().receipt.source_digest
    );
}

#[test]
fn checkpoint_pin_is_verified_before_loading_and_legacy_identity_is_preserved() {
    use starforge::dope_encoder::DopeCheckpointIdentity;
    let wrong =
        DopeCheckpointIdentity::unqualified_candidate("a".repeat(64), "b".repeat(64)).unwrap();
    assert!(DopeEncoder::from_pinned_bytes(WEIGHTS, &DopeDevice::Cpu, wrong).is_err());
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    assert_eq!(
        encoder.checkpoint_identity(),
        &DopeCheckpointIdentity::ReleasedV1
    );
}
