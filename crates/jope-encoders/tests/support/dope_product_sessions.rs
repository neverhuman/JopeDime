//! Explicit product selection and serialization; no native probes or external assets are needed.
use starforge::dope_encoder::{
    collect, start_training_with_options, write_snapshot, DopeConsumer, DopeConsumerProduct,
    DopeDevice, DopeEncoder, Head, LandmarkArgs, LandmarkConsumer, LandmarkMode, SessionOptions,
    SessionOutcome, TrainingInput,
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
    let event: serde_json::Value = serde_json::from_slice(&events)?;
    assert_eq!(event["embedding"], event["consumer"]["embedding"]);
    assert_eq!(
        event["consumer"]["receipt"]["input_conditioning_digest"],
        event["receipt"]["conditioning_digest"]
    );
    assert_eq!(event["receipt"]["version"], 0);
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
    let event: serde_json::Value = serde_json::from_slice(&events)?;
    assert!(event.get("consumer").is_none());
    let receipt = serde_json::to_value(run.final_snapshot.receipt)?;
    assert!(receipt.get("consumer").is_none());
    assert!(receipt.get("consumer_embedding").is_none());
    Ok(())
}

#[test]
fn explicit_products_refuse_class_capping_or_label_rounding_before_component_admission(
) -> anyhow::Result<()> {
    let encoder = DopeEncoder::from_bytes(super::WEIGHTS, &DopeDevice::Cpu)?;
    for labels in [
        (0..33).map(|n| n as f64).collect::<Vec<_>>(),
        vec![0., 0.0000001, 1.],
    ] {
        let k = labels.len();
        let input = TrainingInput::numeric(
            (0..k).map(|n| vec![n as f64]).collect(),
            labels,
            (0..k as u64).collect(),
            Head::Multiclass(k),
        );
        let args = LandmarkArgs {
            landmark_mode: Some(LandmarkMode::None),
            landmark_consumer: Some(LandmarkConsumer::Direct),
            ..Default::default()
        };
        let result = start_training_with_options(
            input,
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
            error.to_string().contains("lossless FIT class preparation"),
            "{error}"
        );
    }
    Ok(())
}
