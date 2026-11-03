use super::*;

/// Performs the record checkpoint commit operation and returns an error if it cannot be completed.
pub fn record_checkpoint_commit(
    store: &impl PublicationStore,
    commit: CheckpointCommit,
) -> Result<()> {
    require_digest(&commit.checkpoint_sha256, "checkpoint")?;
    require_digest(&commit.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&commit.closure_sha256, "checkpoint closure")?;
    store.append_event(PublicationEvent::CheckpointCommitted { checkpoint: commit })
}

/// Performs the adopt research parent operation and returns an error if it cannot be completed.
pub fn adopt_research_parent(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    decision: ResearchParentDecision,
    now_ms: u64,
) -> Result<AdoptionResult> {
    validate_research_decision(&decision)?;
    let decision_sha256 = crate::canon::canonical_sha256(&decision)?;
    // The assessment and evidence pin are durably recorded before any attempted pointer move.
    store.append_event(PublicationEvent::ResearchParentAssessed {
        decision_sha256: decision_sha256.clone(),
        decision: decision.clone(),
    })?;
    validate_grant(
        grant,
        GrantPurpose::ResearchParentAdoption,
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
        now_ms,
    )?;
    let observed = store.read_pointer(PointerKind::ResearchParent)?;
    ensure_expected_parent(
        observed.as_ref(),
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
    )?;
    if decision.assessment != Assessment::Qualified {
        return Ok(AdoptionResult::Retained {
            current: observed.map(|v| v.pointer),
            evidence_sha256: decision.candidate_evidence_sha256,
            assessment: decision.assessment,
        });
    }
    validate_closure(&decision.closure, &decision.compatible_closure_sha256)?;
    ensure!(
        decision
            .rollback
            .required_objects
            .is_subset(&decision.closure.required_objects),
        "compatible closure omits rollback pins"
    );
    ensure!(
        decision.expected_parent_sha256.as_deref() == Some(decision.rollback.root_sha256.as_str()),
        "rollback root is not the exact current parent"
    );
    let next_generation = decision
        .expected_generation
        .checked_add(1)
        .context("research-parent generation exhausted")?;
    decision.rollback.validate_before(next_generation)?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: decision.candidate_root_sha256,
        parent_root_sha256: decision.expected_parent_sha256,
        decision_sha256,
        closure_sha256: decision.compatible_closure_sha256,
        rollback: decision.rollback,
    };
    compare_and_advance(store, PointerKind::ResearchParent, observed, next)
}

/// Performs the promote serving operation and returns an error if it cannot be completed.
pub fn promote_serving(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    decision: ServingPromotionDecision,
    now_ms: u64,
) -> Result<AdoptionResult> {
    validate_serving_decision(&decision)?;
    let decision_sha256 = crate::canon::canonical_sha256(&decision)?;
    store.append_event(PublicationEvent::ServingAssessed {
        decision_sha256: decision_sha256.clone(),
        decision: decision.clone(),
    })?;
    validate_grant(
        grant,
        GrantPurpose::ServingPromotion,
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
        now_ms,
    )?;
    let observed = store.read_pointer(PointerKind::Serving)?;
    ensure_expected_parent(
        observed.as_ref(),
        decision.expected_generation,
        decision.expected_parent_sha256.as_deref(),
    )?;
    let research = store.read_pointer(PointerKind::ResearchParent)?;
    ensure!(
        research
            .as_ref()
            .is_some_and(|r| r.pointer.root_sha256 == decision.expected_research_parent_sha256),
        "research parent is not the independently adopted root"
    );
    if decision.assessment != Assessment::Qualified {
        return Ok(AdoptionResult::Retained {
            current: observed.map(|v| v.pointer),
            evidence_sha256: decision.candidate_evidence_sha256,
            assessment: decision.assessment,
        });
    }
    ensure!(
        decision.qualification_sha256.is_some(),
        "serving promotion needs independent qualification evidence"
    );
    validate_closure(&decision.closure, &decision.compatible_closure_sha256)?;
    ensure!(
        decision
            .rollback
            .required_objects
            .is_subset(&decision.closure.required_objects),
        "compatible closure omits rollback pins"
    );
    ensure!(
        decision.expected_parent_sha256.as_deref() == Some(decision.rollback.root_sha256.as_str()),
        "rollback root is not the exact current parent"
    );
    let next_generation = decision
        .expected_generation
        .checked_add(1)
        .context("serving generation exhausted")?;
    decision.rollback.validate_before(next_generation)?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: decision.candidate_root_sha256,
        parent_root_sha256: decision.expected_parent_sha256,
        decision_sha256,
        closure_sha256: decision.compatible_closure_sha256,
        rollback: decision.rollback,
    };
    compare_and_advance(store, PointerKind::Serving, observed, next)
}

/// Performs the rollback root operation and returns an error if it cannot be completed.
pub fn rollback_root(
    store: &impl PublicationStore,
    grant: &PublisherGrant,
    kind: PointerKind,
    rollback: RollbackRoot,
    closure: ClosureProof,
    decision_evidence_sha256: String,
    now_ms: u64,
) -> Result<RootPointer> {
    require_digest(&decision_evidence_sha256, "rollback decision evidence")?;
    let observed = store.read_pointer(kind)?;
    let current = observed
        .as_ref()
        .context("cannot roll back an absent root")?;
    validate_grant(
        grant,
        GrantPurpose::Rollback,
        current.pointer.generation,
        Some(&current.pointer.root_sha256),
        now_ms,
    )?;
    rollback.validate_before(current.pointer.generation)?;
    validate_closure(&closure, &rollback.closure_sha256)?;
    ensure!(
        rollback
            .required_objects
            .is_subset(&closure.required_objects),
        "rollback proof omits retained rollback pins"
    );
    let decision_sha256 = crate::canon::canonical_sha256(&(
        &kind,
        &current.pointer.root_sha256,
        &rollback,
        &decision_evidence_sha256,
    ))?;
    store.append_event(PublicationEvent::RootRolledBack {
        decision_sha256: decision_sha256.clone(),
        kind,
        previous_root: current.pointer.root_sha256.clone(),
        rollback: rollback.clone(),
    })?;
    let next_generation = current
        .pointer
        .generation
        .checked_add(1)
        .context("pointer generation exhausted")?;
    let next = RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation: next_generation,
        root_sha256: rollback.root_sha256.clone(),
        parent_root_sha256: Some(current.pointer.root_sha256.clone()),
        decision_sha256,
        closure_sha256: rollback.closure_sha256.clone(),
        rollback,
    };
    match store.compare_exchange(kind, Some(&current.store_version), next.clone())? {
        true => Ok(next),
        false => bail!("rollback CAS lost; accepted root remains unchanged"),
    }
}

fn validate_research_decision(d: &ResearchParentDecision) -> Result<()> {
    require_digest(&d.candidate_root_sha256, "candidate root")?;
    require_digest(&d.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&d.compatible_closure_sha256, "compatible closure")?;
    if let Some(parent) = &d.expected_parent_sha256 {
        require_digest(parent, "expected parent")?;
    }
    Ok(())
}

fn validate_serving_decision(d: &ServingPromotionDecision) -> Result<()> {
    require_digest(&d.candidate_root_sha256, "candidate root")?;
    require_digest(&d.candidate_evidence_sha256, "candidate evidence")?;
    require_digest(&d.expected_research_parent_sha256, "research parent")?;
    require_digest(&d.compatible_closure_sha256, "compatible closure")?;
    if let Some(parent) = &d.expected_parent_sha256 {
        require_digest(parent, "expected serving parent")?;
    }
    if let Some(qualification) = &d.qualification_sha256 {
        require_digest(qualification, "serving qualification")?;
    }
    Ok(())
}

fn validate_closure(closure: &ClosureProof, expected_sha256: &str) -> Result<()> {
    closure.validate()?;
    ensure!(
        closure.closure_sha256 == expected_sha256,
        "compatible closure snapshot changed"
    );
    Ok(())
}

fn validate_grant(
    grant: &PublisherGrant,
    purpose: GrantPurpose,
    generation: u64,
    parent: Option<&str>,
    now_ms: u64,
) -> Result<()> {
    require_digest(&grant.verified_grant_sha256, "verified grant")?;
    ensure!(grant.purpose == purpose, "grant purpose mismatch");
    ensure!(
        grant.expires_at_ms > now_ms,
        "publisher grant is stale/expired"
    );
    ensure!(
        grant.expected_generation == generation
            && grant.expected_parent_sha256.as_deref() == parent,
        "grant is not bound to expected parent/generation"
    );
    Ok(())
}

fn ensure_expected_parent(
    observed: Option<&VersionedPointer>,
    generation: u64,
    parent: Option<&str>,
) -> Result<()> {
    match observed {
        Some(value) => ensure!(
            value.pointer.generation == generation
                && Some(value.pointer.root_sha256.as_str()) == parent,
            "accepted root changed; stale parent/generation"
        ),
        None => ensure!(
            generation == 0 && parent.is_none(),
            "expected root is absent or stale"
        ),
    }
    Ok(())
}

fn compare_and_advance(
    store: &impl PublicationStore,
    kind: PointerKind,
    observed: Option<VersionedPointer>,
    next: RootPointer,
) -> Result<AdoptionResult> {
    let expected_version = observed.as_ref().map(|v| v.store_version.as_str());
    match store.compare_exchange(kind, expected_version, next.clone())? {
        true => Ok(AdoptionResult::Advanced(next)),
        false => bail!("publisher CAS lost; accepted root remains unchanged"),
    }
}

pub(super) fn require_digest(value: &str, name: &str) -> Result<()> {
    ensure!(is_digest(value), "{name} must be a lowercase SHA256 digest");
    Ok(())
}

pub(super) fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
