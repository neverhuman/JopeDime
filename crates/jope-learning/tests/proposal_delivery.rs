use anyhow::Result;
use jope_learning::canon::{canonical_json, sha256_hex};
use jope_learning::proposals::outbox::{
    AcceptedProposalInputs, DeliveryStatus, DrawRef, Inserted, PendingLimits, ProposalBundle,
    ProposalOutbox, RequestContext, TerminalProofVerifier, STATUS_SCHEMA,
};
use jope_learning::proposals::receiving::{dispatch, FencedRelease, ReleaseOutcome};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn scratch() -> PathBuf {
    let root = PathBuf::from("target/task-evidence/I26/proposal-delivery-tests");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("case-")
        .tempdir_in(root)
        .unwrap()
        .keep()
}

fn digest(byte: u8) -> String {
    hex::encode([byte; 32])
}

fn context(context: u8, recipe: u8) -> RequestContext {
    RequestContext {
        context_sha256: digest(context),
        recipe_sha256: digest(recipe),
        fidelity_sha256: digest(3),
        source_sha256: digest(4),
        original_window_sha256: digest(5),
        reservation_id: "reserve-1".into(),
        deadline_unix_ms: 50_000,
    }
}

fn bundle() -> ProposalBundle {
    ProposalBundle {
        ledger_sha256: digest(6),
        draws: vec![
            DrawRef {
                draw_id: "draw-001".into(),
                slot_record_sha256: digest(7),
                canonical_draw_sha256: digest(8),
                multiplicity: 2,
                denominator: 10,
            },
            DrawRef {
                draw_id: "draw-002".into(),
                slot_record_sha256: digest(9),
                canonical_draw_sha256: digest(8),
                multiplicity: 1,
                denominator: 10,
            },
        ],
    }
}

struct FixtureAccepted;

impl AcceptedProposalInputs for FixtureAccepted {
    fn verify_request_context(&self, context: &RequestContext) -> Result<()> {
        context.validate()
    }

    fn verify_draw_bundle(&self, bundle: &ProposalBundle) -> Result<()> {
        anyhow::ensure!(
            bundle.ledger_sha256 == digest(6),
            "fixture is not from the accepted draw ledger"
        );
        anyhow::ensure!(
            bundle.draws == crate::bundle().draws,
            "fixture draw references do not match the ledger"
        );
        Ok(())
    }
}

struct FixtureTerminalProof;

impl TerminalProofVerifier for FixtureTerminalProof {
    fn verify_terminal_or_reap(
        &self,
        request_key: &str,
        generation: u64,
        receipt_sha256: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            jope_learning::canon::is_hex64(request_key),
            "fixture request key invalid"
        );
        anyhow::ensure!(
            generation > 0 && jope_learning::canon::is_hex64(receipt_sha256),
            "fixture terminal proof invalid"
        );
        Ok(())
    }
}

struct RejectTerminalProof;

impl TerminalProofVerifier for RejectTerminalProof {
    fn verify_terminal_or_reap(
        &self,
        _request_key: &str,
        _generation: u64,
        _receipt_sha256: &str,
    ) -> Result<()> {
        anyhow::bail!("no authenticated terminal proof")
    }
}

#[derive(Default)]
struct FakeRelease {
    calls: u64,
    remote: BTreeMap<String, Vec<u8>>,
    crash_after_upload: bool,
}

impl FencedRelease for FakeRelease {
    fn authorize_and_release(
        &mut self,
        key: &str,
        _lease: &jope_learning::proposals::outbox::Lease,
    ) -> Result<ReleaseOutcome> {
        self.calls += 1;
        let bytes = format!("proposal:{key}").into_bytes();
        match self.remote.get(key) {
            Some(existing) if existing != &bytes => anyhow::bail!("remote content conflict"),
            Some(_) => {}
            None => {
                self.remote.insert(key.to_string(), bytes);
            }
        }
        if self.crash_after_upload {
            return Ok(ReleaseOutcome::Unknown);
        }
        Ok(ReleaseOutcome::Released {
            physical_attempt_id: format!("fit-{}", self.calls),
            physical_cost_units: 7,
        })
    }
}

#[test]
fn ten_thousand_duplicate_transports_keep_draws_and_authorize_one_fit() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let context = context(1, 2);
    let bundle = bundle();
    let first = outbox.insert(context.clone(), bundle.clone(), &FixtureAccepted)?;
    assert!(matches!(first, Inserted::Added { .. }));
    for _ in 1..10_000 {
        assert!(matches!(
            outbox.insert(context.clone(), bundle.clone(), &FixtureAccepted)?,
            Inserted::Existing { .. }
        ));
    }
    let key = match first {
        Inserted::Added { request_key } => request_key,
        _ => unreachable!(),
    };
    assert_eq!(outbox.record(&key)?.bundle.draws, bundle.draws);
    assert_eq!(outbox.recover()?.requests, 1);
    let lease = outbox.lease(&key, "worker-a", 100, 1_000)?;
    let mut release = FakeRelease::default();
    dispatch(&mut outbox, &lease, 101, &mut release)?;
    assert!(dispatch(&mut outbox, &lease, 102, &mut release).is_err());
    assert_eq!(release.calls, 1);
    assert_eq!(outbox.status(&key)?.attempts.len(), 1);
    assert_eq!(outbox.status(&key)?.attempts[0].physical_cost_units, 7);
    assert_eq!(outbox.total_physical_cost(&key)?, 7);
    assert!(outbox.status(&key)?.attempts[0]
        .ack_sha256
        .as_deref()
        .is_some_and(jope_learning::canon::is_hex64));
    drop(outbox);
    let mut recovered = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    assert_eq!(recovered.status(&key)?.attempts[0].physical_cost_units, 7);
    assert!(recovered.lease(&key, "worker-b", 200, 100).is_err());
    Ok(())
}

#[test]
fn crash_before_record_commit_replays_into_same_bounded_row() -> Result<()> {
    let root = scratch();
    let request = context(5, 6);
    let proposal = bundle();
    let key = request.physical_request_key()?;
    let row = ProposalOutbox::dir(&root).join("requests").join(&key);
    fs::create_dir_all(&row)?;
    let status = DeliveryStatus {
        schema: STATUS_SCHEMA.to_string(),
        request_key: key.clone(),
        generation: 0,
        lease: None,
        attempts: Vec::new(),
    };
    fs::write(row.join("status.json"), canonical_json(&status)?)?;
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    assert_eq!(outbox.recover()?.requests, 0);
    assert!(matches!(
        outbox.insert(request, proposal, &FixtureAccepted)?,
        Inserted::Added { .. }
    ));
    assert_eq!(outbox.recover()?.requests, 1);
    Ok(())
}

#[test]
fn crash_after_lease_waits_for_expiry_then_advances_fence() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let key = match outbox.insert(context(15, 16), bundle(), &FixtureAccepted)? {
        Inserted::Added { request_key } => request_key,
        _ => unreachable!(),
    };
    let first = outbox.lease(&key, "worker-a", 100, 100)?;
    drop(outbox);
    let mut recovered = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    assert!(recovered.lease(&key, "worker-b", 150, 100).is_err());
    let second = recovered.lease(&key, "worker-b", 201, 100)?;
    assert!(second.generation > first.generation);
    assert!(recovered.start(&first, 202).is_err());
    Ok(())
}

#[test]
fn upload_before_ack_recovers_exact_content_without_refit() -> Result<()> {
    let root = scratch();
    let context = context(11, 12);
    let bundle = bundle();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let key = match outbox.insert(context.clone(), bundle.clone(), &FixtureAccepted)? {
        Inserted::Added { request_key } => request_key,
        _ => unreachable!(),
    };
    let lease = outbox.lease(&key, "worker-a", 100, 100)?;
    let mut remote = FakeRelease {
        crash_after_upload: true,
        ..FakeRelease::default()
    };
    assert_eq!(
        dispatch(&mut outbox, &lease, 101, &mut remote)?,
        ReleaseOutcome::Unknown
    );
    let remote_bytes = remote.remote.get(&key).cloned().unwrap();
    drop(outbox);

    let mut recovered = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    assert!(matches!(
        recovered.insert(context, bundle, &FixtureAccepted)?,
        Inserted::Existing { .. }
    ));
    assert_eq!(remote.remote.get(&key), Some(&remote_bytes));
    assert!(recovered.lease(&key, "worker-b", 1_000, 100).is_err());
    assert!(recovered
        .resolve_unknown(
            &key,
            &sha256_hex(&remote_bytes),
            7,
            false,
            &RejectTerminalProof
        )
        .is_err());
    recovered.resolve_unknown(
        &key,
        &sha256_hex(&remote_bytes),
        7,
        false,
        &FixtureTerminalProof,
    )?;
    assert_eq!(recovered.status(&key)?.attempts.len(), 1);
    assert_eq!(remote.calls, 1);
    Ok(())
}

#[test]
fn stale_lease_cannot_start_or_charge_after_expiry() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let key = match outbox.insert(context(21, 22), bundle(), &FixtureAccepted)? {
        Inserted::Added { request_key } => request_key,
        _ => unreachable!(),
    };
    let stale = outbox.lease(&key, "worker-old", 100, 10)?;
    let current = outbox.lease(&key, "worker-current", 111, 100)?;
    assert!(outbox.start(&stale, 112).is_err());
    outbox.start(&current, 112)?;
    assert!(outbox
        .finish(
            &stale,
            jope_learning::proposals::outbox::TerminalKind::Failed {
                receipt_sha256: digest(31),
                retryable: false
            },
            99
        )
        .is_err());
    outbox.finish(
        &current,
        jope_learning::proposals::outbox::TerminalKind::Released {
            physical_attempt_id: "fit-1".into(),
        },
        7,
    )?;
    assert_eq!(outbox.status(&key)?.attempts.len(), 1);
    assert_eq!(outbox.status(&key)?.attempts[0].physical_cost_units, 7);
    Ok(())
}

#[test]
fn request_key_binds_context_and_capacity_backpressures() -> Result<()> {
    let root = scratch();
    let limits = PendingLimits {
        max_requests: 2,
        max_pending: 1,
        max_total_bytes: 1 << 20,
        max_startup_requests: 2,
        max_draws_per_request: 4,
        max_attempts: 2,
        max_record_bytes: 1 << 16,
        max_status_bytes: 1 << 12,
    };
    let mut outbox = ProposalOutbox::open(&root, limits, &FixtureAccepted)?;
    outbox.insert(context(41, 42), bundle(), &FixtureAccepted)?;
    assert!(outbox
        .insert(context(43, 42), bundle(), &FixtureAccepted)
        .is_err());
    assert_eq!(outbox.recover()?.requests, 1);
    Ok(())
}

#[test]
fn terminal_headroom_is_reserved_before_admission() -> Result<()> {
    let root = scratch();
    let limits = PendingLimits {
        max_requests: 2,
        max_pending: 2,
        max_total_bytes: 4 << 10,
        max_startup_requests: 2,
        max_draws_per_request: 4,
        max_attempts: 2,
        max_record_bytes: 1 << 16,
        max_status_bytes: 4 << 10,
    };
    let mut outbox = ProposalOutbox::open(&root, limits, &FixtureAccepted)?;
    assert!(outbox
        .insert(context(46, 42), bundle(), &FixtureAccepted)
        .is_err());
    assert_eq!(outbox.recover()?.requests, 0);
    Ok(())
}

#[test]
fn startup_index_refuses_history_above_admitted_bound() -> Result<()> {
    let root = scratch();
    let limits = PendingLimits {
        max_requests: 2,
        max_pending: 2,
        max_total_bytes: 1 << 20,
        max_startup_requests: 1,
        max_draws_per_request: 4,
        max_attempts: 2,
        max_record_bytes: 1 << 16,
        max_status_bytes: 1 << 12,
    };
    let mut outbox = ProposalOutbox::open(&root, limits, &FixtureAccepted)?;
    outbox.insert(context(47, 42), bundle(), &FixtureAccepted)?;
    outbox.insert(context(48, 42), bundle(), &FixtureAccepted)?;
    drop(outbox);
    assert!(ProposalOutbox::open(&root, limits, &FixtureAccepted).is_err());
    Ok(())
}

#[test]
fn same_recipe_in_distinct_contexts_keeps_distinct_requests() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let first = outbox.insert(context(44, 42), bundle(), &FixtureAccepted)?;
    let second = outbox.insert(context(45, 42), bundle(), &FixtureAccepted)?;
    let key = |inserted| match inserted {
        Inserted::Added { request_key } => request_key,
        Inserted::Existing { request_key } => request_key,
    };
    assert_ne!(key(first), key(second));
    assert_eq!(outbox.recover()?.requests, 2);
    Ok(())
}

#[test]
fn unknown_started_attempt_requires_terminal_proof_before_retry_and_charges_both() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let key = match outbox.insert(context(51, 52), bundle(), &FixtureAccepted)? {
        Inserted::Added { request_key } => request_key,
        _ => unreachable!(),
    };
    let first = outbox.lease(&key, "worker-a", 100, 10)?;
    outbox.start(&first, 101)?;
    drop(outbox);
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    assert!(outbox.status(&key)?.attempts[0].unknown);
    assert!(outbox.lease(&key, "worker-b", 1_000, 10).is_err());
    assert!(outbox
        .resolve_unknown(&key, &digest(61), 5, true, &RejectTerminalProof)
        .is_err());
    outbox.resolve_unknown(&key, &digest(61), 5, true, &FixtureTerminalProof)?;
    let retry = outbox.lease(&key, "worker-b", 1_001, 10)?;
    outbox.start(&retry, 1_002)?;
    outbox.finish(
        &retry,
        jope_learning::proposals::outbox::TerminalKind::Released {
            physical_attempt_id: "fit-retry".into(),
        },
        7,
    )?;
    let attempts = &outbox.status(&key)?.attempts;
    assert_eq!(attempts.len(), 2);
    assert_eq!(outbox.total_physical_cost(&key)?, 12);
    Ok(())
}

#[test]
fn changed_content_under_same_physical_key_is_rejected() -> Result<()> {
    let root = scratch();
    let mut outbox = ProposalOutbox::open(&root, PendingLimits::DRAFT, &FixtureAccepted)?;
    let context = context(71, 72);
    outbox.insert(context.clone(), bundle(), &FixtureAccepted)?;
    let mut changed = bundle();
    changed.draws[0].slot_record_sha256 = digest(73);
    assert!(outbox.insert(context, changed, &FixtureAccepted).is_err());
    Ok(())
}

#[test]
fn pinned_request_and_draw_bundle_fixtures_decode() -> Result<()> {
    let context: RequestContext =
        serde_json::from_str(include_str!("fixtures/proposal-delivery-v1/request.json"))?;
    let fixture: ProposalBundle =
        serde_json::from_str(include_str!("fixtures/proposal-delivery-v1/bundle.json"))?;
    context.validate()?;
    fixture.validate(PendingLimits::DRAFT)?;
    assert_eq!(context.context_sha256, digest(1));
    assert_eq!(fixture.draws.len(), 2);
    assert_eq!(
        fixture.draws[0].canonical_draw_sha256,
        fixture.draws[1].canonical_draw_sha256
    );
    assert_ne!(fixture.draws[0].draw_id, fixture.draws[1].draw_id);
    let ledger: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/proposal-delivery-v1/draw-ledger.json"
    ))?;
    assert_eq!(ledger["ledger_sha256"], fixture.ledger_sha256);
    let cost: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/proposal-delivery-v1/terminal-cost.json"
    ))?;
    assert_eq!(cost["physical_cost_units"], 7);
    assert_eq!(cost["committed_update_credit"], 0);
    let fence: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/proposal-delivery-v1/stale-fence.json"
    ))?;
    assert!(fence["stale_generation"].as_u64() < fence["current_generation"].as_u64());
    let conflict: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/proposal-delivery-v1/negative-conflict.json"
    ))?;
    assert_eq!(conflict["same_key_different_content"], true);
    let ack: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/proposal-delivery-v1/terminal-ack.json"
    ))?;
    assert_eq!(ack["schema"], "jope.proposal-delivery.ack/v1");
    assert_eq!(ack["content_addressed"], true);
    assert_eq!(ack["committed_update_credit"], 0);
    Ok(())
}
