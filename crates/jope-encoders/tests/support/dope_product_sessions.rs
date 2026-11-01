//! Explicit product selection and serialization; no native probes or external assets are needed.
use starforge::dope_encoder::{
    collect, prepare_product_training_input, start_training_with_options, write_snapshot,
    DopeConsumer, DopeConsumerProduct, DopeDevice, DopeEncoder, FitScope, Head, LandmarkArgs,
    LandmarkConsumer, LandmarkMode, SessionOptions, SessionOutcome, TrainingInput,
    LOSSLESS_PREPARATION_RECIPE,
};
use std::sync::Arc;

fn input() -> TrainingInput {
    TrainingInput::numeric(
        (0..12).map(|i| vec![i as f64, (i % 3) as f64]).collect(),
        (0..12).map(|i| (i * i) as f64).collect(),
        (0..12).collect(),
        Head::Regression,
    )
}

#[test]
fn direct_session_emits_raw_product_and_companion_values() -> anyhow::Result<()> {
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::None),
        landmark_consumer: Some(LandmarkConsumer::Direct),
        ..Default::default()
    };
    let session = start_training_with_options(
        input(),
        encoder,
        &args,
        SessionOptions {
            consumer: Some(Arc::new(DopeConsumer::encoding_only())),
            ..Default::default()
        },
    )?;
    assert!(matches!(
        session.initial().consumer.as_ref().unwrap().product(),
        DopeConsumerProduct::Direct(_)
    ));
    let mut events = Vec::new();
    let run = collect(session, |snapshot| write_snapshot(&mut events, snapshot))?;
    assert_eq!(run.outcome, SessionOutcome::Complete);
    let receipt = &run.final_snapshot.receipt;
    assert_eq!(receipt.embedding, receipt.consumer_embedding);
    let events = serde_json::Deserializer::from_slice(&events)
        .into_iter::<serde_json::Value>()
        .collect::<serde_json::Result<Vec<_>>>()?;
    assert_eq!(events.len(), 2);
    for event in &events {
        assert_eq!(event["embedding"], event["consumer"]["embedding"]);
        assert_eq!(
            event["consumer"]["receipt"]["input_conditioning_digest"],
            event["receipt"]["conditioning_digest"]
        );
        assert_eq!(event["receipt"]["version"], 0);
    }
    assert!(events[0]["receipt"]["outcome"].is_null());
    assert_eq!(events[1]["receipt"]["outcome"], "complete");
    Ok(())
}

#[test]
fn explicit_consumer_and_bound_consumer_must_match() -> anyhow::Result<()> {
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    for (selected, bound) in [
        (None, true),
        (Some(LandmarkConsumer::Direct), false),
        (Some(LandmarkConsumer::JopeDl), true),
    ] {
        let args = LandmarkArgs {
            landmark_mode: Some(LandmarkMode::None),
            landmark_consumer: selected,
            landmark_components: (selected == Some(LandmarkConsumer::JopeDl))
                .then(|| "controller-reference.json".into()),
            ..Default::default()
        };
        let options = SessionOptions {
            consumer: bound.then(|| Arc::new(DopeConsumer::encoding_only())),
            ..Default::default()
        };
        assert!(start_training_with_options(input(), encoder.clone(), &args, options).is_err());
    }
    Ok(())
}

#[test]
fn omitted_consumer_keeps_historical_event_and_receipt_fields() -> anyhow::Result<()> {
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::Fast),
        ..Default::default()
    };
    let session = start_training_with_options(input(), encoder, &args, SessionOptions::default())?;
    let mut events = Vec::new();
    let run = collect(session, |snapshot| write_snapshot(&mut events, snapshot))?;
    let events = serde_json::Deserializer::from_slice(&events)
        .into_iter::<serde_json::Value>()
        .collect::<serde_json::Result<Vec<_>>>()?;
    assert_eq!(events.len(), 2);
    for event in &events {
        assert!(event.get("consumer").is_none());
    }
    assert_eq!(events[1]["receipt"]["outcome"], "complete");
    let receipt = serde_json::to_value(run.final_snapshot.receipt)?;
    assert!(receipt.get("consumer").is_none());
    assert!(receipt.get("consumer_embedding").is_none());
    Ok(())
}

fn multiclass_input(labels: &[f64]) -> TrainingInput {
    let rows = labels.len() * 4;
    TrainingInput::numeric(
        (0..rows).map(|i| vec![i as f64, (i % 5) as f64]).collect(),
        (0..rows).map(|i| labels[i % labels.len()]).collect(),
        (0..rows).map(|i| 1000 + i as u64 * 7).collect(),
        Head::Multiclass(labels.len()),
    )
}

#[test]
fn explicit_direct_none_k33_preserves_full_fit_vocabulary_and_emits_finite512() -> anyhow::Result<()>
{
    let labels = (0..33).map(|n| n as f64).collect::<Vec<_>>();
    let input = multiclass_input(&labels);
    let prepared = prepare_product_training_input(&input)?;
    assert_eq!(prepared.classes, 33);
    assert_eq!(prepared.transform.target.class_labels, labels);
    assert_eq!(prepared.row_ids, input.ids);
    for (i, code) in prepared.y_scaled.iter().enumerate() {
        assert_eq!(*code, (i % 33) as f64);
    }
    let expected_scope = FitScope::from_prepared(&input, &prepared)?;
    assert_eq!(
        expected_scope.preparation_recipe.as_deref(),
        Some(LOSSLESS_PREPARATION_RECIPE)
    );
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::None),
        landmark_consumer: Some(LandmarkConsumer::Direct),
        ..Default::default()
    };
    let session = start_training_with_options(
        input,
        encoder,
        &args,
        SessionOptions {
            consumer: Some(Arc::new(DopeConsumer::encoding_only())),
            ..Default::default()
        },
    )?;
    let run = collect(session, |_| Ok(()))?;
    assert_eq!(run.outcome, SessionOutcome::Complete);
    let snapshot = &run.final_snapshot;
    assert!(matches!(
        snapshot.consumer.as_ref().unwrap().product(),
        DopeConsumerProduct::Direct(_)
    ));
    let values = snapshot.embedding.to_vec()?;
    assert_eq!(values.len(), 512);
    assert!(values.iter().all(|v| v.is_finite()));
    assert_eq!(snapshot.receipt.embedding.as_ref(), Some(&values));
    assert_eq!(snapshot.receipt.consumer_embedding.as_ref(), Some(&values));
    assert_eq!(snapshot.receipt.fit_scope.as_ref(), Some(&expected_scope));
    assert_eq!(
        snapshot.receipt.fit_scope.as_ref().unwrap().task,
        "Multiclass(33)"
    );
    assert_eq!(snapshot.receipt.mask, [false; 23]);
    assert_eq!(snapshot.receipt.version, 0);
    Ok(())
}

#[test]
fn explicit_products_refuse_rounded_or_colliding_classes_before_component_admission(
) -> anyhow::Result<()> {
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    for labels in [
        vec![0., 0.0000001, 1.],
        vec![0.1000001, 0.2000001, 0.3000001],
    ] {
        for selected in [LandmarkConsumer::Direct, LandmarkConsumer::JopeDl] {
            let args = LandmarkArgs {
                landmark_mode: Some(LandmarkMode::None),
                landmark_consumer: Some(selected),
                landmark_components: (selected == LandmarkConsumer::JopeDl)
                    .then(|| "unopened-components.json".into()),
                ..Default::default()
            };
            // The learned selection deliberately lacks projected assets/admitted components.
            // Class refusal must happen before any consumer binding or component admission.
            let result = start_training_with_options(
                multiclass_input(&labels),
                encoder.clone(),
                &args,
                SessionOptions {
                    consumer: Some(Arc::new(DopeConsumer::encoding_only())),
                    ..Default::default()
                },
            );
            let error = match result {
                Ok(_) => anyhow::bail!("lossy product preparation admitted"),
                Err(e) => e,
            };
            assert!(
                error
                    .to_string()
                    .contains("exact uncapped FIT class vocabulary"),
                "{selected:?}: {error}"
            );
        }
    }
    Ok(())
}
