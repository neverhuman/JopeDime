//! Explicit selected-asset execution proof; this is not a predictive quality qualification.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use serde_json::Value;
use starforge::bundle::VerifiedJopePolicyAssets;
use starforge::dope_encoder::{
    collect, prepare_product_training_input, start_training_with_options, ComponentInputs,
    DopeConsumer, DopeConsumerProduct, DopeEncoder, EncodingSnapshot, FitScope, Head, LandmarkArgs,
    LandmarkConsumer, LandmarkMode, SessionOptions, SessionOutcome, TrainingInput,
};
use starforge::jope_policy::JopePolicyModel;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

#[path = "../../../../test-support/dope_components.rs"]
mod components;
use components::{install_for_scope, ComponentFixture};

fn fixture_root(label: &str) -> Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/landmarks");
    std::fs::create_dir_all(&base)?;
    let root = base.join(format!(
        "p4-learned-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root)?;
    Ok(root)
}
fn model() -> Result<Arc<JopePolicyModel>> {
    let root = std::env::var_os("JOPE_MODELS_ROOT")
        .context("explicit learned-product qualification requires JOPE_MODELS_ROOT")?;
    // This loader reads the selected closure only, including for fetched v2 directory layouts.
    let assets = VerifiedJopePolicyAssets::load(PathBuf::from(root))?;
    let model = Arc::new(JopePolicyModel::load_policy_assets(&assets, Device::Cpu)?);
    ensure!(
        model.projection_identity() == Some(assets.identity()),
        "missing projection identity"
    );
    frozen_oracle(&model)?;
    Ok(model)
}
fn frozen_oracle(model: &JopePolicyModel) -> Result<()> {
    let bytes = starforge::JOPE_GEN31_PARITY_FIXTURE_JSON;
    assert_eq!(
        jope_runtime::sha256_hex(bytes),
        "9f4d3d4ee12828eb8ba0f6312615e1e807f622894123ba2f7bd52efc8c11f619"
    );
    assert_eq!(
        model.weights_sha256(),
        "2c4c5531c5f1d834653fb1fef8b2b51cfd9d90044ff15cc98b67cd1b5aaae871"
    );
    let fixture: Value = serde_json::from_slice(bytes)?;
    let embedding: Vec<f32> = serde_json::from_value(fixture["embedding_1536"].clone())?;
    let target: Vec<f32> = serde_json::from_value(fixture["target_512"].clone())?;
    let expected: Vec<f32> = serde_json::from_value(fixture["z_512"].clone())?;
    let actual = model.context_values(&model.context(&embedding, &target)?)?;
    assert_eq!(actual.len(), 512);
    assert_eq!(expected.len(), 512);
    let mut maximum = 0f32;
    for (a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        maximum = maximum.max((a - b).abs());
    }
    assert!(
        maximum <= 1e-5,
        "selected policy/frozen context error {maximum}"
    );
    Ok(())
}
fn input(head: Head) -> TrainingInput {
    let x = (0..48)
        .map(|i| vec![(i as f64 * 0.31).sin(), (i % 7) as f64, i as f64 / 48.])
        .collect::<Vec<_>>();
    let y = (0..48)
        .map(|i| match head {
            Head::Regression => 2. * x[i][0] + x[i][2].powi(2),
            Head::Binary => (i % 2) as f64,
            Head::Multiclass(_) => [11., 23., 47.][i % 3],
        })
        .collect();
    let mut input = TrainingInput::numeric(x, y, (0..48).map(|i| 900 + i * 3).collect(), head);
    input.fold = Some(format!("p4-fit-{head:?}"));
    input
}
fn fixture(input: &TrainingInput, label: &str) -> Result<ComponentFixture> {
    let prepared = prepare_product_training_input(input)?;
    install_for_scope(
        &fixture_root(label)?,
        FitScope::from_prepared(input, &prepared)?,
    )
}
fn args(fixture: &ComponentFixture, mode: LandmarkMode) -> LandmarkArgs {
    LandmarkArgs {
        landmark_mode: Some(mode),
        landmark_subsample: Some(40),
        landmark_consumer: Some(LandmarkConsumer::JopeDl),
        landmark_components: Some(fixture.requests.clone()),
        ..Default::default()
    }
}
fn options(
    fixture: &ComponentFixture,
    consumer: &Arc<DopeConsumer>,
    gate: Option<Arc<Mutex<()>>>,
) -> Result<SessionOptions> {
    Ok(SessionOptions {
        consumer: Some(consumer.clone()),
        component_inputs: Some(Arc::new(ComponentInputs::load_with_controller(
            &fixture.requests,
            &fixture.controller,
        )?)),
        native_gate: gate,
        ..Default::default()
    })
}
fn oracle(
    snapshot: &EncodingSnapshot,
    model: &JopePolicyModel,
    fixture: &ComponentFixture,
) -> Result<()> {
    let raw = DopeConsumer::encoding_only().consume_snapshot(snapshot)?;
    assert!(matches!(raw.product(), DopeConsumerProduct::Direct(_)));
    assert_eq!(raw.to_vec()?, snapshot.embedding.to_vec()?);
    let learned = snapshot
        .consumer
        .as_ref()
        .context("missing learned product")?;
    let DopeConsumerProduct::JopeDl(context) = learned.product() else {
        anyhow::bail!("learned session returned raw features");
    };
    let values = learned.to_vec()?;
    assert_eq!(values.len(), 512);
    let fusion = snapshot
        .fusion
        .as_ref()
        .context("missing admitted fusion")?
        .to_export()?;
    assert_eq!(&fusion.dataset1536[512..], fixture.foundation1024);
    assert_eq!(&fusion.policy2048[1536..], fixture.target512);
    let expected =
        model.context_values(&model.context(&fusion.dataset1536, &fusion.policy2048[1536..])?)?;
    if snapshot.embedding.tensor_copy()?.device().is_cuda() {
        assert_eq!(values.len(), expected.len());
        let mut maximum = 0f32;
        for (actual, expected) in values.iter().zip(&expected) {
            assert!(actual.is_finite() && expected.is_finite());
            maximum = maximum.max((actual - expected).abs());
        }
        assert!(
            maximum < 3e-5,
            "learned CUDA/CPU context error {maximum:.9e}"
        );
        eprintln!("P4 admitted CUDA context max_abs={maximum:.9e}");
    } else {
        assert_eq!(values, expected);
    }
    let decoder = context.decoder_context(model)?;
    assert_eq!(model.context_values(&decoder)?, values);
    // A second pass over the derived coordinates is a different computation and must not occur.
    let mut wrong = fusion.dataset1536.clone();
    wrong[..512].copy_from_slice(&values);
    let double = model.context_values(&model.context(&wrong, &fusion.policy2048[1536..])?)?;
    assert_ne!(values, double);
    let receipt = learned.receipt();
    assert_eq!(
        receipt.input_conditioning_digest,
        snapshot.receipt.conditioning_digest
    );
    assert_eq!(
        receipt.checkpoint_identity,
        *snapshot.embedding.checkpoint_identity()
    );
    assert_eq!(
        receipt.projection_identity.as_ref(),
        model.projection_identity()
    );
    assert_eq!(
        receipt.jope_weights_sha256.as_deref(),
        Some(model.weights_sha256())
    );
    let admitted = receipt
        .component_admission
        .as_ref()
        .context("missing component admission")?;
    assert_eq!(admitted.subject_sha256, fixture.subject_sha256);
    assert_eq!(
        admitted.exposure_receipt_sha256,
        fixture.exposure_receipt_sha256
    );
    assert_eq!(snapshot.receipt.consumer.as_ref(), Some(receipt));
    Ok(())
}

#[test]
#[ignore = "requires selected governed JOPE assets in JOPE_MODELS_ROOT; no full bundle loading"]
fn admitted_learned_product_three_tasks_matches_oracle_and_terminal_full() -> Result<()> {
    let model = model()?;
    let consumer = Arc::new(DopeConsumer::jope_dl(model.clone())?);
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &Device::Cpu)?;
    for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
        let input = input(head);
        let fixture = fixture(&input, &format!("{head:?}"))?;
        let progressive = start_training_with_options(
            input.clone(),
            encoder.clone(),
            &args(&fixture, LandmarkMode::Progressive),
            options(&fixture, &consumer, None)?,
        )?;
        oracle(progressive.initial(), &model, &fixture)?;
        let initial = progressive.initial().embedding.to_vec()?;
        let direct = collect(
            start_training_with_options(
                input.clone(),
                encoder.clone(),
                &LandmarkArgs {
                    landmark_mode: Some(LandmarkMode::None),
                    landmark_subsample: Some(40),
                    landmark_consumer: Some(LandmarkConsumer::Direct),
                    ..Default::default()
                },
                SessionOptions {
                    consumer: Some(Arc::new(DopeConsumer::encoding_only())),
                    ..Default::default()
                },
            )?,
            |_| Ok(()),
        )?;
        assert_eq!(direct.outcome, SessionOutcome::Complete);
        assert_eq!(
            direct.final_snapshot.consumer.as_ref().unwrap().to_vec()?,
            initial
        );
        let mut masks = Vec::new();
        let mut versions = Vec::new();
        let run = collect(progressive, |snapshot| {
            oracle(snapshot, &model, &fixture)?;
            masks.push(snapshot.receipt.mask.iter().filter(|&&v| v).count());
            versions.push(snapshot.receipt.version);
            Ok(())
        })?;
        assert_eq!(run.outcome, SessionOutcome::Complete);
        assert_eq!(masks.first(), Some(&0));
        assert!(masks.iter().any(|&n| n > 0 && n < 23));
        assert!(versions.windows(2).all(|v| v[0] <= v[1]));
        let full = collect(
            start_training_with_options(
                input,
                encoder.clone(),
                &args(&fixture, LandmarkMode::Full),
                options(&fixture, &consumer, None)?,
            )?,
            |s| oracle(s, &model, &fixture),
        )?;
        assert_eq!(full.outcome, SessionOutcome::Complete);
        assert_eq!(
            run.final_snapshot.receipt.mask,
            full.final_snapshot.receipt.mask
        );
        assert_eq!(
            run.final_snapshot.receipt.conditioning_digest,
            full.final_snapshot.receipt.conditioning_digest
        );
        assert_eq!(
            run.final_snapshot.embedding.to_vec()?,
            full.final_snapshot.embedding.to_vec()?
        );
        assert_eq!(
            run.final_snapshot.consumer.as_ref().unwrap().receipt(),
            full.final_snapshot.consumer.as_ref().unwrap().receipt()
        );
        assert_eq!(
            run.final_snapshot.receipt.consumer_embedding,
            full.final_snapshot.receipt.consumer_embedding
        );
        eprintln!("P4 admitted {head:?}: native masks={masks:?}, versions={versions:?}, terminal/full equal");
    }
    Ok(())
}

#[test]
#[ignore = "requires selected governed JOPE assets; fixture authority changes only"]
fn admitted_authority_change_before_native_work_keeps_last_valid_learned_product() -> Result<()> {
    let model = model()?;
    let consumer = Arc::new(DopeConsumer::jope_dl(model)?);
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &Device::Cpu)?;
    let input = input(Head::Regression);
    let fixture = fixture(&input, "gate-revocation")?;
    let gate = Arc::new(Mutex::new(()));
    let guard = gate.lock().unwrap();
    let session = start_training_with_options(
        input,
        encoder,
        &args(&fixture, LandmarkMode::Progressive),
        options(&fixture, &consumer, Some(gate.clone()))?,
    )?;
    let initial = session.initial().clone();
    fixture.mutate_authority()?;
    drop(guard);
    let mut versions = Vec::new();
    let run = collect(session, |s| {
        versions.push(s.receipt.version);
        Ok(())
    })?;
    assert!(matches!(run.outcome, SessionOutcome::Failed { .. }));
    assert!(versions.iter().all(|&version| version == 0));
    assert_eq!(run.final_snapshot.receipt.version, 0);
    assert_eq!(
        run.final_snapshot.embedding.to_vec()?,
        initial.embedding.to_vec()?
    );
    assert_eq!(
        run.final_snapshot.consumer.as_ref().unwrap().to_vec()?,
        initial.consumer.as_ref().unwrap().to_vec()?
    );
    Ok(())
}

#[test]
#[ignore = "requires selected governed JOPE assets; checks authority after none-mode callback"]
fn admitted_none_callback_authority_change_cannot_finalize_complete() -> Result<()> {
    let model = model()?;
    let consumer = Arc::new(DopeConsumer::jope_dl(model)?);
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &Device::Cpu)?;
    let input = input(Head::Binary);
    let fixture = fixture(&input, "none-revocation")?;
    let session = start_training_with_options(
        input,
        encoder,
        &args(&fixture, LandmarkMode::None),
        options(&fixture, &consumer, None)?,
    )?;
    let initial = session.initial().clone();
    let mut count = 0;
    let run = collect(session, |_| {
        count += 1;
        fixture.mutate_authority()
    })?;
    assert_eq!(count, 1);
    assert!(matches!(run.outcome, SessionOutcome::Failed { .. }));
    assert_eq!(run.final_snapshot.receipt.version, 0);
    assert_eq!(
        run.final_snapshot.consumer.as_ref().unwrap().to_vec()?,
        initial.consumer.as_ref().unwrap().to_vec()?
    );
    assert_eq!(
        run.final_snapshot.embedding.to_vec()?,
        initial.embedding.to_vec()?
    );
    Ok(())
}

#[cfg(feature = "dope-cuda")]
#[test]
#[ignore = "requires selected governed JOPE assets and an explicitly admitted GPU window"]
fn admitted_cuda_product_three_tasks_matches_cpu_oracle_and_terminal_full() -> Result<()> {
    let source = std::env::var_os("JOPE_MODELS_ROOT")
        .context("explicit CUDA product qualification requires JOPE_MODELS_ROOT")?;
    let assets = VerifiedJopePolicyAssets::load(PathBuf::from(source))?;
    let cpu = JopePolicyModel::load_policy_assets(&assets, Device::Cpu)?;
    frozen_oracle(&cpu)?;
    // Exactly one owned stream is shared by the Dope encoder and learned policy context.
    let device = DopeEncoder::cuda(0)?;
    assert!(device.is_cuda());
    let gpu = Arc::new(JopePolicyModel::load_policy_assets_on_device(
        &assets,
        device.clone(),
    )?);
    assert!(gpu.device().same_device(&device));
    assert_eq!(gpu.weights_sha256(), cpu.weights_sha256());
    let consumer = Arc::new(DopeConsumer::jope_dl(gpu)?);
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &device)?;
    for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
        let input = input(head);
        let fixture = fixture(&input, &format!("cuda-{head:?}"))?;
        let check = |snapshot: &EncodingSnapshot| -> Result<()> {
            assert!(snapshot
                .embedding
                .tensor_copy()?
                .device()
                .same_device(&device));
            assert!(snapshot
                .consumer
                .as_ref()
                .context("missing CUDA consumer")?
                .tensor_copy()?
                .device()
                .same_device(&device));
            let fusion = snapshot.fusion.as_ref().context("missing CUDA fusion")?;
            assert!(fusion.dataset1536_copy()?.device().same_device(&device));
            assert!(fusion.policy2048_copy()?.device().same_device(&device));
            // Only the test oracle explicitly serializes resident tensors back to the CPU.
            oracle(snapshot, &cpu, &fixture)
        };
        let session = start_training_with_options(
            input.clone(),
            encoder.clone(),
            &args(&fixture, LandmarkMode::Progressive),
            options(&fixture, &consumer, None)?,
        )?;
        check(session.initial())?;
        let progressive = collect(session, |snapshot| check(snapshot))?;
        assert_eq!(progressive.outcome, SessionOutcome::Complete);
        let full = collect(
            start_training_with_options(
                input,
                encoder.clone(),
                &args(&fixture, LandmarkMode::Full),
                options(&fixture, &consumer, None)?,
            )?,
            |snapshot| check(snapshot),
        )?;
        assert_eq!(full.outcome, SessionOutcome::Complete);
        assert_eq!(
            progressive.final_snapshot.receipt.mask,
            full.final_snapshot.receipt.mask
        );
        assert_eq!(
            progressive.final_snapshot.receipt.conditioning_digest,
            full.final_snapshot.receipt.conditioning_digest
        );
        assert_eq!(
            progressive.final_snapshot.embedding.to_vec()?,
            full.final_snapshot.embedding.to_vec()?
        );
        let a = progressive.final_snapshot.consumer.as_ref().unwrap();
        let b = full.final_snapshot.consumer.as_ref().unwrap();
        assert_eq!(a.receipt(), b.receipt());
        assert_eq!(a.to_vec()?, b.to_vec()?);
        assert!(a.receipt().recipe.contains("cuda-f32"));
        eprintln!("P4 admitted CUDA {head:?}: resident raw512/context512/fusion, CPU oracle, decoder bypass and terminal/full equality");
    }
    Ok(())
}
