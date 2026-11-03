//! W3-T02, producer side: a real teacher is admissible through the product path.
//!
//! - The freeze preregisters the W2 producer exactly as its W2 qualification registers it
//!   (`jope_policy::context_producer::ProducerRegistration`, parsed field for field as
//!   `ProducerPlan`; the W2 side pins that record and shows it is the producer that hands off to
//!   teacher admission, in
//!   `the_preregistered_producer_is_the_producer_that_hands_off_to_teacher_admission`).
//! - The producer run hands its context off to teacher admission.
//! - `retain_handed_off_context`, which the live `jope-bench` driver calls, retains that run where
//!   teacher admission looks for it and issues the context's receipt under the freeze's scope.
//! - `qualify` admits the teacher and records the handoff; `hwm::ingest`, `freeze_teaching` and
//!   `revalidate_qualification` agree.
//!
//! The producer run is a CPU fixture: synthetic pool receipts and a synthetic context, as the live
//! driver retains them. Nothing is encoded or executed, and this is not a live W2 run.
use super::provenance::{assert_refused, authority, Authority};
use super::*;

/// The registration a W2 qualification yields, as `ProducerRegistration` serializes it.
pub(super) fn registration() -> serde_json::Value {
    let plan = producer_plan();
    json!({
        "release": plan.release,
        "transformations": plan.transformations,
        "qualification_sha256": plan.qualification_sha256,
        "qualification_execution_sha256": plan.qualification_execution_sha256,
    })
}

/// The freeze's context plan as the live driver builds it: the preregistered context encoder and
/// the W2 producer parsed field for field from its registration.
pub(super) fn preregistered() -> ContextPlan {
    ContextPlan {
        producer: Some(serde_json::from_value(registration()).unwrap()),
        ..context_plan()
    }
}

/// The context the producer run hands off; distinct from the fixture's own context, so nothing
/// of this run is retained before the product path retains it.
pub(super) fn teaching_context() -> Vec<u8> {
    context_output(2048, 0.625)
}

/// The W2 producer run of `context` for `f`'s dataset inside `window`, as the live driver retains
/// it: the handoff record the run emitted for teacher admission, the context job and its
/// supervisor receipt, the experiment and window, the artifact the job wrote and the context's
/// bytes. `tamper` changes the run first. Nothing is retained.
pub(super) fn producer_run(
    f: &Fixture,
    context: &[u8],
    window: &ExecutionWindow,
    tamper: impl FnOnce(&mut HandoffParts),
) -> ContextRun {
    producer_run_in(f, context, &f.experiment, window, tamper)
}

/// [`producer_run`] in `experiment` instead of `f`'s: the context job's request and the retained
/// experiment are that experiment's.
pub(super) fn producer_run_in(
    f: &Fixture,
    context: &[u8],
    experiment: &Experiment,
    window: &ExecutionWindow,
    tamper: impl FnOnce(&mut HandoffParts),
) -> ContextRun {
    let mut parts = handoff_parts(&f.state, context, &f.plan.comparison, experiment, window);
    tamper(&mut parts);
    ContextRun {
        handoff: serde_json::to_vec(&parts.handoff).unwrap(),
        job: canonical_json(&parts.job).unwrap(),
        outcome: parts.outcome_bytes(),
        experiment: canonical_json(experiment).unwrap(),
        window: serde_json::to_vec(window).unwrap(),
        artifact: parts.artifact,
        context: context.to_vec(),
    }
}

/// `f`'s teacher, submitted with the context the product path issued for `run`.
pub(super) fn submission(f: &Fixture, run: &ContextRun) -> OofEvidence {
    OofEvidence {
        context: Some(retain_handed_off_context(&f.state, &f.evidence.freeze, run).unwrap()),
        ..f.evidence.clone()
    }
}

pub(super) fn ingest(f: &Fixture) -> hwm::IngestReport {
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: candidate_blob(""),
            chain_from_source: false,
        }],
        json!({"kind":"w3-t02-producer"}),
    )
    .unwrap()
}

pub(super) fn nothing() -> Authority {
    Authority {
        qualified: vec![],
        pointers: vec![],
        accepted: vec![],
        teaching: vec![],
    }
}

/// Positive control: freeze with the preregistered producer, run it, hand its context off,
/// retain the run and admit the teacher. The qualification records the handoff.
/// A producer run whose artifact the job wrote as `change` edits it: the job's supervisor receipt,
/// the handoff and the controller's admission pins all name the edited bytes.
pub(super) fn with_changed_artifact(
    f: &Fixture,
    change: impl FnOnce(&mut serde_json::Value),
) -> ContextRun {
    producer_run(f, &teaching_context(), &f.window, |p| {
        let mut artifact: serde_json::Value = serde_json::from_slice(&p.artifact).unwrap();
        change(&mut artifact);
        p.with_artifact(serde_json::to_vec(&artifact).unwrap());
    })
}

/// PR #114 review round 1, P1 (the reviewer's probe): the producer artifact, the context job and
/// its supervisor outcome stay byte-identical, and only the submitted 2048-f32 context (finite,
/// clipped values) and the handoff's context digest are substituted. The artifact still holds the
/// original context, so the substitute is refused and no authority changes.
mod admission;
#[allow(unused_imports)]
use admission::*;
mod artifact;
#[allow(unused_imports)]
use artifact::*;
mod records;
#[allow(unused_imports)]
use records::*;
mod window;
#[allow(unused_imports)]
use window::*;
