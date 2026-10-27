use super::*;
use crate::dope_encoder::{
    LandmarkStatus, DOPE512_RECIPE, DOPE_WEIGHTS_SHA256, ENCODING_RECEIPT_SCHEMA,
};
use candle_core::{DType, Device, Tensor};

fn snapshot(digest: &str) -> EncodingSnapshot {
    EncodingSnapshot {
        embedding: Dope512 {
            checkpoint_identity: Default::default(),
            tensor: Tensor::zeros((1, 512), DType::F32, &Device::Cpu).unwrap(),
            source_digest: "a".repeat(64),
            conditioning_digest: digest.into(),
        },
        receipt: LandmarkEncodingReceipt {
            schema: ENCODING_RECEIPT_SCHEMA.into(),
            status: LandmarkStatus::None,
            mask: [false; 23],
            subsample_n: Some(300),
            source_digest: "a".repeat(64),
            conditioning_digest: digest.into(),
            options_digest: None,
            work_budget: None,
            weights_sha256: DOPE_WEIGHTS_SHA256.into(),
            training_receipt_sha256: None,
            embedding_identity: DOPE512_RECIPE.into(),
            device: "cpu".into(),
            version: 0,
            outcome: None,
            embedding: None,
            consumer: None,
            consumer_embedding: None,
            fit_scope: None,
        },
        fusion: None,
        consumer: None,
    }
}

#[test]
fn initial_precedes_worker_completion_and_updates_are_ordered_idempotent() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let worker_barrier = barrier.clone();
    let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
        worker_barrier.wait();
        assert!(publisher.publish(snapshot("macro"))?);
        assert!(!publisher.publish(snapshot("macro"))?);
        assert!(publisher.publish(snapshot("full"))?);
        Ok(())
    });
    assert_eq!(session.initial().receipt.version, 0);
    assert_eq!(session.recv().unwrap().receipt.conditioning_digest, "zero");
    barrier.wait();
    let mut versions = Vec::new();
    let (last, outcome) = session
        .finish(|s| {
            versions.push(s.receipt.version);
            Ok(())
        })
        .unwrap();
    assert_eq!(versions, vec![1, 2]);
    assert_eq!(last.receipt.version, 2);
    assert_eq!(outcome, SessionOutcome::Complete);
}

#[test]
fn new_coverage_is_published_without_advancing_an_unchanged_encoding() {
    let session = EncodingSession::spawn(snapshot("zero"), |publisher| {
        let mut macro_snapshot = snapshot("macro");
        macro_snapshot.receipt.mask[0] = true;
        macro_snapshot.receipt.status = LandmarkStatus::Partial(1);
        assert!(publisher.publish(macro_snapshot.clone())?);
        macro_snapshot.receipt.mask[17] = true;
        macro_snapshot.receipt.status = LandmarkStatus::Partial(2);
        assert!(publisher.publish(macro_snapshot.clone())?);
        assert!(!publisher.publish(macro_snapshot)?);
        Ok(())
    });
    let mut versions = Vec::new();
    let (last, outcome) = session
        .finish(|s| {
            versions.push(s.receipt.version);
            Ok(())
        })
        .unwrap();
    assert_eq!(versions, vec![0, 1, 1]);
    assert_eq!(last.receipt.status, LandmarkStatus::Partial(2));
    assert_eq!(last.receipt.version, 1);
    assert_eq!(outcome, SessionOutcome::Complete);
}

#[test]
fn checkpoint_switch_and_forged_receipt_fail_before_duplicate_elision() {
    for consistent in [false, true] {
        let session = EncodingSession::spawn(snapshot("same"), move |publisher| {
            let mut next = snapshot("same");
            let pin = super::super::DopeCheckpointIdentity::unqualified_candidate(
                "c".repeat(64),
                "d".repeat(64),
            )?;
            next.receipt.weights_sha256 = pin.weights_sha256().into();
            next.receipt.training_receipt_sha256 = pin.training_receipt_sha256().map(str::to_owned);
            next.receipt.embedding_identity = pin.embedding_recipe().into();
            if consistent {
                next.embedding.checkpoint_identity = pin;
            }
            publisher.publish(next)?;
            Ok(())
        });
        let (last, outcome) = session.finish(|_| Ok(())).unwrap();
        assert_eq!(last.receipt.version, 0);
        assert_eq!(last.receipt.weights_sha256, DOPE_WEIGHTS_SHA256);
        assert!(matches!(outcome, SessionOutcome::Failed { .. }));
    }
}

#[test]
fn incompatible_scope_and_worker_panic_preserve_last_valid_snapshot() {
    for panic in [false, true] {
        let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
            publisher.publish(snapshot("macro"))?;
            if panic {
                panic!("injected worker failure");
            }
            let mut wrong = snapshot("other");
            wrong.receipt.source_digest = "b".repeat(64);
            publisher.publish(wrong)?;
            Ok(())
        });
        let (last, outcome) = session.finish(|_| Ok(())).unwrap();
        assert_eq!(last.receipt.conditioning_digest, "macro");
        assert_eq!(last.receipt.version, 1);
        assert!(matches!(outcome, SessionOutcome::Failed { .. }));
    }
}

#[test]
fn cancellation_keeps_last_published_embedding() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let worker_barrier = barrier.clone();
    let session = EncodingSession::spawn(snapshot("zero"), move |publisher| {
        worker_barrier.wait();
        assert!(publisher.cancelled());
        assert!(!publisher.publish(snapshot("late"))?);
        Ok(())
    });
    session.cancel();
    barrier.wait();
    let (last, outcome) = session.finish(|_| Ok(())).unwrap();
    assert_eq!(last.receipt.version, 0);
    assert_eq!(outcome, SessionOutcome::Cancelled);
}
#[test]
fn external_cancellation_is_observed_before_late_publication() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let worker_barrier = barrier.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let session = EncodingSession::spawn_with_cancel(
        snapshot("zero"),
        Some(cancel.clone()),
        move |publisher| {
            worker_barrier.wait();
            assert!(publisher.cancelled());
            assert!(!publisher.publish(snapshot("late"))?);
            Ok(())
        },
    );
    assert_eq!(session.recv().unwrap().receipt.version, 0);
    cancel.store(true, Ordering::Release);
    barrier.wait();
    let (last, outcome) = session.finish(|_| Ok(())).unwrap();
    assert_eq!(last.receipt.version, 0);
    assert_eq!(outcome, SessionOutcome::Cancelled);
}

fn direct_snapshot(digest: &str) -> EncodingSnapshot {
    let mut s = snapshot(digest);
    let output = super::super::DopeConsumer::encoding_only()
        .consume_snapshot(&s)
        .unwrap();
    s.receipt.consumer = Some(output.receipt().clone());
    s.consumer = Some(output);
    s
}

#[test]
fn consumer_switch_and_receipt_forgery_fail_before_duplicate_elision() {
    for remove in [false, true] {
        let session = EncodingSession::spawn(direct_snapshot("same"), move |publisher| {
            let mut next = direct_snapshot("same");
            if remove {
                next.consumer = None;
                next.receipt.consumer = None;
            } else {
                next.receipt.consumer.as_mut().unwrap().recipe = "forged".into();
            }
            publisher.publish(next)?;
            Ok(())
        });
        let (last, outcome) = session.finish(|_| Ok(())).unwrap();
        assert_eq!(last.receipt.version, 0);
        assert!(last.consumer.is_some());
        assert!(matches!(outcome, SessionOutcome::Failed { .. }));
    }
}

#[test]
fn direct_consumer_stays_bound_and_duplicate_is_idempotent() {
    let session = EncodingSession::spawn(direct_snapshot("zero"), |publisher| {
        assert!(publisher.publish(direct_snapshot("macro"))?);
        assert!(!publisher.publish(direct_snapshot("macro"))?);
        Ok(())
    });
    let (last, outcome) = session.finish(|_| Ok(())).unwrap();
    assert_eq!(outcome, SessionOutcome::Complete);
    assert_eq!(last.receipt.version, 1);
    let output = last.consumer.unwrap();
    assert_eq!(output.to_vec().unwrap(), last.embedding.to_vec().unwrap());
    output.tensor_copy().unwrap().one_set().unwrap();
    assert_eq!(output.to_vec().unwrap(), last.embedding.to_vec().unwrap());
}

#[test]
fn failed_consumer_update_keeps_last_valid_product() {
    let session = EncodingSession::spawn(direct_snapshot("zero"), |publisher| {
        publisher.publish(direct_snapshot("macro"))?;
        anyhow::bail!("consumer forward failed before publication")
    });
    let (last, outcome) = session.finish(|_| Ok(())).unwrap();
    assert!(matches!(outcome, SessionOutcome::Failed { .. }));
    assert_eq!(last.receipt.version, 1);
    assert_eq!(last.receipt.conditioning_digest, "macro");
    assert_eq!(
        last.consumer
            .as_ref()
            .unwrap()
            .receipt()
            .input_conditioning_digest,
        "macro"
    );
    assert_eq!(
        last.receipt.consumer.as_ref(),
        Some(last.consumer.as_ref().unwrap().receipt())
    );
}
