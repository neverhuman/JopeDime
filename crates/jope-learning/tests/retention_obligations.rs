use jope_learning::canon::sha256_hex;
use jope_learning::retention::obligations::{
    Candidate, Disposition, LifecycleAssessment, ObligationPlanV2, PinGeneration, PromiseKind,
    ProvenanceRef, RecoveryPromise, RetiredVersion, RuntimeRestoreProof, StrongPhysicalRef,
    TagAssessment, VersionId,
};
use std::collections::{BTreeMap, BTreeSet};

fn physical(key: &str, version: &str, digest: &str) -> StrongPhysicalRef {
    StrongPhysicalRef {
        account: "714044927654".into(),
        bucket: "learning-use2".into(),
        key: key.into(),
        version_id: VersionId::Explicit(version.into()),
        sha256: digest.into(),
        bytes: 128,
    }
}

fn candidate(object: StrongPhysicalRef) -> Candidate {
    Candidate {
        object,
        tags: TagAssessment::Verified {
            required_tags_present: true,
        },
        lifecycle: LifecycleAssessment {
            configuration_sha256: Some("a".repeat(64)),
            complete: true,
            matching_current_expiration_rules: Vec::new(),
            matching_noncurrent_expiration_rules: Vec::new(),
            unsupported_fields: Vec::new(),
        },
        unknowns: Vec::new(),
    }
}

fn snapshot() -> PinGeneration {
    PinGeneration {
        root_generation: "root-12".into(),
        pin_generation: "pin-12".into(),
        observed_root_generation: "root-12".into(),
        observed_pin_generation: "pin-12".into(),
        closure_complete: true,
        registered_publishers: BTreeSet::from(["trainer-a".into()]),
        observed_publishers: BTreeSet::from(["trainer-a".into()]),
        unresolved: Vec::new(),
        strong_dependencies: BTreeMap::new(),
        provenance: BTreeMap::new(),
        promises: Vec::new(),
        retired_versions: BTreeMap::new(),
    }
}

fn promise(kind: PromiseKind, dependencies: Vec<StrongPhysicalRef>) -> RecoveryPromise {
    RecoveryPromise {
        id: format!("promise-{kind:?}"),
        kind,
        root_id: "accepted-root-12".into(),
        active: true,
        dependencies,
        restore_proofs: Vec::new(),
        released_by_transition: None,
        released_at_generation: None,
    }
}

#[test]
fn ancestor_checkpoint_corpus_and_optimizer_remain_pinned_by_resume_promise() {
    let checkpoint = physical("checkpoints/ancestor", "cp-v4", &"1".repeat(64));
    let corpus = physical("corpus/ancestor", "co-v8", &"2".repeat(64));
    let optimizer = physical("optimizer/ancestor", "opt-v4", &"3".repeat(64));
    let rollback = physical("checkpoints/rollback", "cp-v2", &"4".repeat(64));
    let retrain_corpus = physical("corpus/retrain", "co-v3", &"5".repeat(64));
    let claim_artifact = physical("claims/evidence", "claim-v6", &"6".repeat(64));
    let mut state = snapshot();
    state.promises.push(promise(
        PromiseKind::ExactResume,
        vec![checkpoint.clone(), corpus.clone(), optimizer.clone()],
    ));
    state
        .promises
        .push(promise(PromiseKind::Rollback, vec![rollback.clone()]));
    state.promises.push(promise(
        PromiseKind::ExactRetraining,
        vec![retrain_corpus.clone()],
    ));
    state.promises.push(promise(
        PromiseKind::ClaimReproduction,
        vec![claim_artifact.clone()],
    ));

    let plan = ObligationPlanV2::build(
        &state,
        vec![
            candidate(checkpoint.clone()),
            candidate(corpus),
            candidate(optimizer),
            candidate(rollback),
            candidate(retrain_corpus),
            candidate(claim_artifact),
        ],
    )
    .unwrap();
    assert!(plan
        .rows
        .iter()
        .all(|row| row.disposition == Disposition::Keep));
    assert_eq!(
        plan.rows
            .iter()
            .map(|row| row.promises[0].clone())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "promise-ClaimReproduction".into(),
            "promise-ExactResume".into(),
            "promise-ExactRetraining".into(),
            "promise-Rollback".into(),
        ])
    );
    assert_eq!(
        plan.rows
            .iter()
            .find(|row| row.object == checkpoint)
            .unwrap()
            .strong_roots,
        ["accepted-root-12"]
    );
}

#[test]
fn byte_equality_does_not_replace_runtime_restore_evidence() {
    let source = physical("checkpoints/old", "v1", &"a".repeat(64));
    let same_bytes = physical("packs/copy", "v3", &"a".repeat(64));
    let mut state = snapshot();
    let mut active = promise(PromiseKind::ExactResume, vec![source.clone()]);
    active.restore_proofs.push(RuntimeRestoreProof {
        source: source.clone(),
        replacement: same_bytes,
        capability: "resume".into(),
        runtime_tested: false,
        receipt_sha256: "b".repeat(64),
        pin_generation: "pin-12".into(),
        readers_migrated: true,
    });
    state.promises.push(active);
    let plan = ObligationPlanV2::build(&state, vec![candidate(source.clone())]).unwrap();
    assert_eq!(plan.rows[0].disposition, Disposition::Keep);
    assert_eq!(plan.rows[0].object, source);
}

#[test]
fn tested_capability_specific_restore_can_move_promise_to_exact_replacement() {
    let source = physical("checkpoints/old", "v1", &"a".repeat(64));
    let replacement = physical("packs/checkpoint", "v2", &"a".repeat(64));
    let mut state = snapshot();
    let mut active = promise(PromiseKind::ExactResume, vec![source.clone()]);
    active.restore_proofs.push(RuntimeRestoreProof {
        source: source.clone(),
        replacement: replacement.clone(),
        capability: "resume".into(),
        runtime_tested: true,
        receipt_sha256: "b".repeat(64),
        pin_generation: "pin-12".into(),
        readers_migrated: true,
    });
    state.promises.push(active);
    let plan = ObligationPlanV2::build(
        &state,
        vec![candidate(source.clone()), candidate(replacement.clone())],
    )
    .unwrap();
    assert_eq!(
        plan.rows
            .iter()
            .find(|r| r.object == source)
            .unwrap()
            .disposition,
        Disposition::ReviewAfterRootTransition
    );
    assert_eq!(
        plan.rows
            .iter()
            .find(|r| r.object == replacement)
            .unwrap()
            .disposition,
        Disposition::Keep
    );
}

#[test]
fn provenance_only_metadata_does_not_pin_bytes() {
    let object = physical("history/old", "v2", &"4".repeat(64));
    let mut state = snapshot();
    state.provenance.insert(
        object.clone(),
        BTreeSet::from([ProvenanceRef {
            schema: "jope.policy-lineage/v1".into(),
            identity: "lineage-14".into(),
            sha256: "5".repeat(64),
            description: "historical parent identity".into(),
        }]),
    );
    let plan = ObligationPlanV2::build(&state, vec![candidate(object)]).unwrap();
    assert_eq!(
        plan.rows[0].disposition,
        Disposition::ReviewAfterRootTransition
    );
    assert_eq!(plan.rows[0].provenance.len(), 1);
}

#[test]
fn stale_publisher_cannot_reintroduce_a_retired_version() {
    let retired = physical("checkpoints/retired", "v7", &"6".repeat(64));
    let mut state = snapshot();
    state.retired_versions.insert(
        retired.clone(),
        RetiredVersion {
            object: retired.clone(),
            retired_at_pin_generation: "pin-11".into(),
        },
    );
    assert!(state
        .validate_publish("trainer-a", "pin-11", &[retired.clone()])
        .is_err());
    assert!(state
        .validate_publish("trainer-a", "pin-12", &[retired.clone()])
        .is_err());
    assert!(state.validate_publish("trainer-a", "pin-11", &[]).is_err());
    assert!(state
        .validate_publish("unknown-publisher", "pin-12", &[])
        .is_err());
    let plan = ObligationPlanV2::build(&state, vec![candidate(retired)]).unwrap();
    assert_eq!(plan.rows[0].disposition, Disposition::Hold);
    assert!(plan.rows[0].unresolved[0].contains("retired"));
}

#[test]
fn promise_release_requires_a_recorded_transition_at_the_current_root() {
    let object = physical("checkpoints/released", "v3", &"5".repeat(64));
    let mut state = snapshot();
    let mut released = promise(PromiseKind::Rollback, vec![object.clone()]);
    released.active = false;
    released.released_by_transition = Some("root-transition-13".into());
    released.released_at_generation = Some("root-11".into());
    state.promises.push(released);
    assert!(ObligationPlanV2::build(&state, vec![candidate(object.clone())]).is_err());

    state.promises[0].released_at_generation = Some("root-12".into());
    let plan = ObligationPlanV2::build(&state, vec![candidate(object)]).unwrap();
    assert_eq!(
        plan.rows[0].disposition,
        Disposition::ReviewAfterRootTransition
    );
}

#[test]
fn unknown_lifecycle_tag_bypass_and_incomplete_root_closure_hold() {
    let object = physical("research/raw", "v3", &"7".repeat(64));
    let mut item = candidate(object);
    item.tags = TagAssessment::Unknown {
        reason: "tag inventory unavailable".into(),
    };
    item.lifecycle.complete = false;
    item.lifecycle.matching_noncurrent_expiration_rules = vec!["broad-rule".into()];
    let mut state = snapshot();
    state.closure_complete = false;
    state.observed_pin_generation = "pin-11".into();
    state.registered_publishers.insert("publisher-b".into());
    let plan = ObligationPlanV2::build(&state, vec![item]).unwrap();
    assert_eq!(plan.rows[0].disposition, Disposition::Hold);
    let gaps = plan.rows[0].unresolved.join(" ");
    assert!(gaps.contains("lifecycle configuration is incomplete"));
    assert!(gaps.contains("tag configuration is unknown"));
    assert!(gaps.contains("root closure is incomplete"));
    assert!(gaps.contains("publisher-b"));
    assert!(gaps.contains("matching noncurrent-version lifecycle expiration"));

    let mut bypass = candidate(physical("research/tagged", "v4", &"9".repeat(64)));
    bypass.tags = TagAssessment::Verified {
        required_tags_present: false,
    };
    bypass.lifecycle.matching_current_expiration_rules = vec!["tag-bypass-rule".into()];
    let clean_snapshot = snapshot();
    let plan = ObligationPlanV2::build(&clean_snapshot, vec![bypass]).unwrap();
    assert_eq!(plan.rows[0].disposition, Disposition::Hold);
    let gaps = plan.rows[0].unresolved.join(" ");
    assert!(gaps.contains("required object tags are missing"));
    assert!(gaps.contains("matching current-version lifecycle expiration"));
}

#[test]
fn plan_is_stable_and_rejects_null_or_unknown_versions_for_review() {
    let unknown = StrongPhysicalRef {
        account: "714044927654".into(),
        bucket: "learning-use2".into(),
        key: "legacy/object".into(),
        version_id: VersionId::Unknown,
        sha256: "8".repeat(64),
        bytes: 10,
    };
    let first = ObligationPlanV2::build(&snapshot(), vec![candidate(unknown.clone())]).unwrap();
    let second = ObligationPlanV2::build(&snapshot(), vec![candidate(unknown)]).unwrap();
    assert_eq!(first.plan_sha256, second.plan_sha256);
    assert_eq!(first.rows[0].disposition, Disposition::Hold);
    assert_eq!(first.schema, "jope.retention-plan/v2");
}

#[test]
fn source_has_no_storage_or_process_mutation_path() {
    let source = include_str!("../src/retention/obligations.rs");
    for forbidden in [
        "object_store",
        "jope_datasets",
        "remove_file",
        "remove_dir",
        "process::Command",
        "DeleteObject",
        "delete_object",
        "PutObject",
        "put_object",
        "PutBucketLifecycle",
        "CopyObject",
    ] {
        assert!(
            !source.contains(forbidden),
            "obligations.rs contains {forbidden}"
        );
    }
    assert_eq!(sha256_hex(source.as_bytes()).len(), 64);
}
