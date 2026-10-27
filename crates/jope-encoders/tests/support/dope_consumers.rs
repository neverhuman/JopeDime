//! Execution and parity proof, independent of the held-out consumer quality ablation.
use candle_core::Device;
use starforge::bundle::LoadedModelBundle;
use starforge::dope_encoder::{
    collect, prepare_table_with_ids, start_training_with_options, write_snapshot, DopeConsumer,
    DopeConsumerMode, DopeEncoder, FusionComponents, Head, LandmarkArgs, LandmarkMode,
    LandmarkOptions, LandmarkScope, PrepTask, SessionOptions, SessionOutcome, SubsampleSpec,
    TrainingInput,
};
use starforge::jope_policy::JopePolicyModel;
use std::io::Write;
use std::sync::Arc;

#[path = "dope_candidate.rs"]
mod candidate;
#[path = "models_root.rs"]
mod models_root;

#[cfg(feature = "dope-cuda")]
#[test]
fn actual_cuda_consumer_context_matches_cpu_and_exports_independent_storage() -> anyhow::Result<()>
{
    let Some(root) = models_root::bundle_root("dope_cuda_jope_consumer") else {
        return Ok(());
    };
    let bundle = LoadedModelBundle::load(&root)?;
    let device = DopeEncoder::cuda(0)?;
    let gpu_model = Arc::new(JopePolicyModel::load_bundle_on_device(
        &bundle,
        device.clone(),
    )?);
    let cpu_model = JopePolicyModel::load_bundle(&bundle, Device::Cpu)?;
    let consumer = DopeConsumer::jope_dl(gpu_model)?;
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &device)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
        let x = (0..12).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y = (0..12)
            .map(|i| match head {
                Head::Regression => (i * i) as f64,
                Head::Binary => (i % 2) as f64,
                Head::Multiclass(_) => [11., 23., 47.][i % 3],
            })
            .collect();
        let session = start_training_with_options(
            TrainingInput::numeric(x, y, (0..12).collect(), head),
            encoder.clone(),
            &LandmarkArgs {
                landmark_mode: Some(LandmarkMode::None),
                ..Default::default()
            },
            SessionOptions::default(),
        )?;
        let snapshot = session.initial().clone();
        let components = FusionComponents {
            source_digest: snapshot.embedding.source_digest().into(),
            recipe: "cuda-consumer-execution-fixture/v1".into(),
            foundation1024: (0..1024).map(|i| (i as f32 * 0.01).sin()).collect(),
            target512: (0..512).map(|i| (i as f32 * 0.02).cos()).collect(),
        };
        let snapshot = snapshot.with_components(&components)?;
        let output = consumer.consume_snapshot(&snapshot)?;
        let before = output.to_vec()?;
        assert!(output.tensor_copy()?.device().same_device(&device));
        assert!(output.receipt().recipe.contains("cuda-f32"));
        let export = snapshot.fusion.as_ref().unwrap().to_export()?;
        let expected = pool.install(|| -> anyhow::Result<_> {
            let context = cpu_model.context(&export.dataset1536, &export.policy2048[1536..])?;
            cpu_model.context_values(&context)
        })?;
        let max = before
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("actual CUDA JOPE context task={head:?} max_abs={max:.9e}");
        assert!(
            max < 3e-5,
            "CUDA context differs from scalar CPU oracle: {max}"
        );
        snapshot.embedding.tensor_copy()?.zero_set()?;
        snapshot
            .fusion
            .as_ref()
            .unwrap()
            .dataset1536_copy()?
            .zero_set()?;
        snapshot
            .fusion
            .as_ref()
            .unwrap()
            .policy2048_copy()?
            .zero_set()?;
        output.tensor_copy()?.zero_set()?;
        assert_eq!(consumer.consume_snapshot(&snapshot)?.to_vec()?, before);
        assert_eq!(output.to_vec()?, before);
        assert_eq!(
            collect(session, |_| Ok(()))?.outcome,
            SessionOutcome::Complete
        );
    }
    Ok(())
}

#[test]
fn both_consumers_execute_three_tasks_and_preserve_scope_and_pins() -> anyhow::Result<()> {
    let Some(root) = models_root::bundle_root("both_dope_consumers") else {
        return Ok(());
    };
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &Device::Cpu)?;
    run_consumers(encoder, &root)
}

fn run_consumers(encoder: DopeEncoder, root: &std::path::Path) -> anyhow::Result<()> {
    let bundle = LoadedModelBundle::load(root)?;
    let model = Arc::new(JopePolicyModel::load_bundle_on_device(
        &bundle,
        encoder.device().clone(),
    )?);
    let oracle = if encoder.device().is_cuda() {
        Arc::new(JopePolicyModel::load_bundle(&bundle, Device::Cpu)?)
    } else {
        model.clone()
    };
    let learned = DopeConsumer::jope_dl(model.clone())?;
    let direct = DopeConsumer::encoding_only();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    for (head, task) in [
        (Head::Regression, PrepTask::Regression),
        (Head::Binary, PrepTask::Classification),
        (Head::Multiclass(3), PrepTask::Multiclass),
    ] {
        let x = (0..48)
            .map(|i| vec![(i as f64 * 0.31).sin(), (i % 7) as f64, i as f64 / 48.])
            .collect::<Vec<_>>();
        let y = (0..48)
            .map(|i| match head {
                Head::Regression => 2. * x[i][0] + x[i][2].powi(2),
                Head::Binary => (i % 2) as f64,
                Head::Multiclass(_) => [11., 23., 47.][i % 3],
            })
            .collect::<Vec<_>>();
        let ids = (0..48).map(|i| 900 + i * 3).collect::<Vec<u64>>();
        let prepared = prepare_table_with_ids(&x, &y, task, ids.clone())?;
        let options = LandmarkOptions {
            work_budget: 2_000_000_000_000,
            ..Default::default()
        };
        let scope = LandmarkScope::for_source_digest(
            &options,
            Some(SubsampleSpec::new(40, options.seed)),
            prepared.source_rows_digest(),
        );
        // Supplied components exercise the execution interface; they are not retention evidence.
        let components = FusionComponents {
            source_digest: scope.digest(),
            recipe: "consumer-execution-fixture/v1".into(),
            foundation1024: (0..1024).map(|i| (i as f32 * 0.01).sin()).collect(),
            target512: (0..512).map(|i| (i as f32 * 0.02).cos()).collect(),
        };
        let args = LandmarkArgs {
            landmark_mode: Some(LandmarkMode::Progressive),
            landmark_subsample: Some(40),
            ..Default::default()
        };
        let input = TrainingInput::numeric(x, y, ids, head);
        let task_name = match head {
            Head::Regression => "regression",
            Head::Binary => "binary",
            Head::Multiclass(_) => "multiclass",
        };
        // Explicit export freezes admission fixtures from actual Rust forwards.
        let mut evidence = std::env::var_os("JOPE_DOPE_EVIDENCE_DIR").map(|path| -> anyhow::Result<_> {
            std::fs::create_dir_all(&path)?;
            let mut file = std::fs::OpenOptions::new().create_new(true).write(true)
                .open(std::path::Path::new(&path).join(format!("{task_name}.jsonl")))?;
            writeln!(file, "{}", serde_json::json!({"schema":"dope-consumer-execution-input/v1", "task": task_name, "x":input.x,"y":input.y,"physical_row_ids":input.ids,"subsample_n":40}))?;
            Ok(file)
        }).transpose()?;
        let session = start_training_with_options(
            input.clone(),
            encoder.clone(),
            &args,
            SessionOptions {
                components: Some(components.clone()),
                ..Default::default()
            },
        )?;
        let mut masks = Vec::new();
        let mut previous = None;
        let run = collect(session, |snapshot| {
            if let Some(file) = &mut evidence {
                write_snapshot(file, snapshot)?;
            }
            let a = direct.consume_snapshot(snapshot)?;
            let b = learned.consume_snapshot(snapshot)?;
            assert!(a.tensor_copy()?.device().same_device(encoder.device()));
            assert!(b.tensor_copy()?.device().same_device(encoder.device()));
            assert_eq!(a.receipt().mode, DopeConsumerMode::EncodingOnly);
            assert_eq!(b.receipt().mode, DopeConsumerMode::JopeDl);
            assert_eq!(a.to_vec()?, snapshot.embedding.to_vec()?);
            assert_eq!(
                a.receipt().checkpoint_identity,
                b.receipt().checkpoint_identity
            );
            assert_ne!(
                a.receipt().conditioning_digest,
                b.receipt().conditioning_digest
            );
            assert_eq!(
                b.receipt().jope_weights_sha256.as_deref(),
                Some(model.weights_sha256())
            );
            let exported = snapshot.fusion.as_ref().unwrap().to_export()?;
            let expected = pool.install(|| -> anyhow::Result<_> {
                let context =
                    oracle.context(&exported.dataset1536, &exported.policy2048[1536..])?;
                oracle.context_values(&context)
            })?;
            let before = b.to_vec()?;
            let starforge::dope_encoder::DopeConsumerProduct::JopeDl(derived) = b.product() else {
                anyhow::bail!("learned consumer returned raw features");
            };
            // The decoder seam copies the already-derived context; it must not run the MLP again.
            let decoder_context = derived.decoder_context(&oracle)?;
            assert_eq!(oracle.context_values(&decoder_context)?, before);
            assert_eq!(before.len(), expected.len());
            if encoder.device().is_cuda() {
                let error = before
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "candidate native {head:?} mask={:?} CUDA context max_abs={error:.9e}",
                    snapshot.receipt.status
                );
                assert!(error < 3e-5, "native CUDA context/CPU oracle: {error}");
            } else {
                assert_eq!(before, expected);
            }
            a.tensor_copy()?.zero_set()?;
            b.tensor_copy()?.zero_set()?;
            assert_eq!(a.to_vec()?, snapshot.embedding.to_vec()?);
            assert_eq!(b.to_vec()?, before);
            assert!(learned.consume(&snapshot.embedding, None).is_err());
            let mut changed = components.clone();
            changed.target512[0] += 1.;
            let changed_fusion = snapshot.embedding.assemble(&changed)?;
            let changed = learned.consume(&snapshot.embedding, Some(&changed_fusion))?;
            assert_ne!(
                b.receipt().conditioning_digest,
                changed.receipt().conditioning_digest
            );
            assert_ne!(b.to_vec()?, changed.to_vec()?);
            if let Some((digest, values)) = previous.take() {
                if digest != snapshot.receipt.conditioning_digest {
                    assert_ne!(values, a.to_vec()?);
                }
            }
            previous = Some((snapshot.receipt.conditioning_digest.clone(), a.to_vec()?));
            masks.push(snapshot.receipt.mask.iter().filter(|v| **v).count());
            Ok(())
        })?;
        assert_eq!(run.outcome, SessionOutcome::Complete);
        assert_eq!(masks.first(), Some(&0));
        assert!(masks.iter().any(|n| *n > 0 && *n < 23));
        assert!(masks.windows(2).all(|pair| pair[0] <= pair[1]));
        // A completed native panel may still have unavailable measurements. Compare actual
        // terminal conditioning rather than inventing observed coverage for those cells.
        let full = start_training_with_options(
            input,
            encoder.clone(),
            &LandmarkArgs {
                landmark_mode: Some(LandmarkMode::Full),
                ..args
            },
            SessionOptions::default(),
        )?;
        let full = collect(full, |_| Ok(()))?;
        assert_eq!(full.outcome, SessionOutcome::Complete);
        assert_eq!(
            full.final_snapshot.receipt.mask,
            run.final_snapshot.receipt.mask
        );
        assert_eq!(
            full.final_snapshot.embedding.to_vec()?,
            run.final_snapshot.embedding.to_vec()?
        );
        assert_eq!(
            full.final_snapshot.receipt.conditioning_digest,
            run.final_snapshot.receipt.conditioning_digest
        );
        eprintln!("actual native task={head:?} progressive observed counts={masks:?}; both consumers and full parity passed");
    }
    Ok(())
}
