//! Non-authoritative CPU receipts. No trainer, authority service or grant issuance.
use anyhow::Result;
use jain_smartcluster::{JobId, ResourceRequest, Timestamp};
use jope_parallel::{
    manifest::hash,
    native_genome::Arm,
    native_graph::Artifact,
    native_loop::{self, lineage::*, warm_start, Config, RunCompletion},
    task_contract::ContinuationMode,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

fn h(s: &str) -> String {
    hash(s.as_bytes())
}
fn put<T: Serialize>(bundle: &mut LineageBundle, schema: &str, value: &T) -> String {
    let bytes = canonical_bytes(value).unwrap();
    let id = object_id(schema, &bytes).unwrap();
    bundle
        .evidence
        .insert(id.clone(), String::from_utf8(bytes).unwrap());
    id
}
fn edge(bundle: &LineageBundle, index: usize) -> LineageEdge {
    serde_json::from_str(&bundle.evidence[&bundle.lineage.edges[index]]).unwrap()
}
fn replace(bundle: &mut LineageBundle, index: usize, edge: &LineageEdge) {
    bundle.lineage.edges[index] = put(bundle, EDGE_SCHEMA, edge);
}
fn append(bundle: &mut LineageBundle, initial: u64, target: u64) {
    append_with_inputs(bundle, initial, target, h("world1"), h("sampler"));
}
fn append_with_inputs(
    bundle: &mut LineageBundle,
    initial: u64,
    target: u64,
    partition_sha256: String,
    sampler_sha256: String,
) {
    let prior = bundle
        .lineage
        .edges
        .len()
        .checked_sub(1)
        .map(|i| edge(bundle, i));
    let parent = prior
        .as_ref()
        .map_or(bundle.lineage.anchor.checkpoint_sha256.clone(), |p| {
            p.checkpoint_sha256.clone()
        });
    let reset = prior.is_none() || initial == 0;
    let epoch = if reset {
        OptimizerEpoch {
            schema: EPOCH_SCHEMA.into(),
            origin_checkpoint_sha256: parent.clone(),
            reset_nonce_sha256: h(&format!("fixture reset{}", bundle.lineage.edges.len())),
            recipe_sha256: bundle.lineage.protocol.recipe_sha256.clone(),
        }
    } else {
        prior.as_ref().unwrap().epoch.clone()
    };
    let intent = TrainingIntent {
        schema: INTENT_SCHEMA.into(),
        parent_checkpoint_sha256: parent.clone(),
        protocol: bundle.lineage.protocol.clone(),
        corpus_sha256: h("corpus"),
        optimizer_epoch_sha256: epoch.sha256().unwrap(),
        initial_step: initial,
        target_step: target,
        execution_partition_sha256: partition_sha256,
        sampler_sha256,
    };
    let intent_id = object_id(INTENT_SCHEMA, &canonical_bytes(&intent).unwrap()).unwrap();
    let eligibility = TrainingEligibility {
        schema: ELIGIBILITY_SCHEMA.into(),
        purpose: "training".into(),
        intent_sha256: intent_id.clone(),
        corpus_sha256: intent.corpus_sha256.clone(),
        role_exposure_closure_sha256: h("fixture role closure"),
        source_roles: BTreeSet::from(["train".into()]),
    };
    let eligible_id = put(bundle, ELIGIBILITY_SCHEMA, &eligibility);
    let checkpoint = h(&format!("checkpoint{target}"));
    let credit = CommittedUpdateCredit {
        schema: CREDIT_SCHEMA.into(),
        commit_sha256: h(&format!("commit{target}")),
        parent_checkpoint_sha256: parent.clone(),
        checkpoint_sha256: checkpoint.clone(),
        optimizer_epoch_sha256: intent.optimizer_epoch_sha256.clone(),
        intent_sha256: intent_id,
        eligibility_sha256: eligible_id,
        rank: 0,
        world: 1,
        initial_update: initial,
        target_update: target,
    };
    let credit_id = put(bundle, CREDIT_SCHEMA, &credit);
    let edge = LineageEdge {
        schema: EDGE_SCHEMA.into(),
        previous_edge_sha256: bundle.lineage.edges.last().cloned(),
        parent_checkpoint_sha256: parent,
        checkpoint_sha256: checkpoint,
        weights_sha256: h(&format!("weights{target}")),
        optimizer_sha256: h(&format!("optimizer{target}")),
        mode: if reset {
            ContinuationMode::WeightWarmStart
        } else {
            ContinuationMode::NewCorpusTransition
        },
        epoch,
        intent,
        committed_credit_sha256: vec![credit_id],
    };
    let id = put(bundle, EDGE_SCHEMA, &edge);
    bundle.lineage.edges.push(id);
}
fn fixture() -> LineageBundle {
    let protocol = ProtocolIdentity {
        objective_sha256: h("objective"),
        recipe_sha256: h("recipe"),
        architecture_sha256: h("architecture"),
        vocabulary_sha256: h("vocabulary"),
        support_sha256: h("support"),
    };
    let anchor = PolicyAnchor {
        checkpoint_sha256: h("accepted60000"),
        weights_sha256: h("accepted weights"),
        optimizer_step: 60000,
        objective_sha256: None,
        committed_updates: None,
        architecture_sha256: protocol.architecture_sha256.clone(),
        vocabulary_sha256: protocol.vocabulary_sha256.clone(),
        support_sha256: protocol.support_sha256.clone(),
    };
    let mut b = LineageBundle {
        lineage: PolicyLineage {
            schema: LINEAGE_SCHEMA.into(),
            anchor,
            protocol,
            edges: vec![],
        },
        evidence: BTreeMap::new(),
    };
    append(&mut b, 0, 3000);
    append(&mut b, 3000, 6000);
    b
}
fn artifact(d: String) -> Artifact {
    Artifact {
        path: format!("fixture/{d}.json").into(),
        sha256: d,
    }
}
fn configs(b: &LineageBundle) -> (Config, Config, RunCompletion) {
    let a = artifact(h("artifact"));
    let previous = Config {
        schema: warm_start::LOOP_V2.into(),
        run: "reset run".into(),
        accepted_checkpoint: artifact(b.lineage.anchor.checkpoint_sha256.clone()),
        initial_corpus: a.clone(),
        accepted_decision: a.clone(),
        scientific_policy: a.clone(),
        label_admission_policy: a.clone(),
        fit_policy: "fit.json".into(),
        run_approval: "approval.json".into(),
        resident_job: JobId::new(),
        workers: 48,
        cycles: 2,
        initial_optimizer_step: 60000,
        predecessor_run: None,
        updates_per_block: 3000,
        global_batch: 2048,
        microbatch: 256,
        direct_seconds: 30,
        continuation_evaluations: 24,
        continuation_seconds: 54,
        cpu_resources: ResourceRequest::cpu_and_ram(1000, 1024 * 1024 * 1024),
        task_deadline: Timestamp::from_unix_millis(u64::MAX),
        qualification: false,
        preparation_overlay: None,
        cuts: 1,
        cut_start: 0,
        materialization: None,
    };
    let mut next = previous.clone();
    next.schema = warm_start::LOOP_V3.into();
    next.run = "descendant run".into();
    next.predecessor_run = Some(a.clone());
    next.initial_optimizer_step = 6000;
    next.accepted_checkpoint = artifact(edge(b, 1).checkpoint_sha256);
    next.initial_corpus = artifact(h("corpus"));
    let completed = RunCompletion {
        schema: warm_start::COMPLETION_V2.into(),
        run: previous.run.clone(),
        config: a.clone(),
        initial_optimizer_step: 0,
        target_optimizer_step: 6000,
        checkpoint: next.accepted_checkpoint.clone(),
        corpus: next.initial_corpus.clone(),
        accepted_parent_decision: a.clone(),
        final_proposals: BTreeMap::from([(Arm::Model, a.clone()), (Arm::Shuffled, a)]),
    };
    (previous, next, completed)
}
fn native(b: &LineageBundle) -> Result<Option<VerifiedLineage>> {
    let (p, n, c) = configs(b);
    native_loop::verify_versioned_successor(&n, &p, &c, Some(b), None)
}

#[path = "policy_lineage/acceptance.rs"]
mod acceptance;
#[path = "policy_lineage/refusals.rs"]
mod refusals;
