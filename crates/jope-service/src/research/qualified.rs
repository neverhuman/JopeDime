//! Source-bound measurement and provenance for the admitted r782 v2 evaluator.
//!
//! The adapter consumes only the existing worker request and response. It
//! verifies the exact attempt, v2 full-TRAIN folds, evaluator source identity,
//! and output invariants before decoding the separate gate. It does not run a
//! model or create qualification evidence from a score shape.
// The trusted admission/recording consumer is a separate reviewed task. Keep
// these crate-private hooks available without treating this staged provider as
// a production qualification path before that consumer is connected.
#![allow(dead_code)]

use super::{
    capture::{digest, Binding, CaptureConfig, CaptureContext, CaptureProvider, SlateObservation},
    hex, NativeCaptureInputs, RecordEvidence, RecordIdentity, RecordTask, RecordedProgram,
    RecordedReveal, Rung0Measurement, DOPE_SHA256,
};
use crate::{
    discover::{
        propose::{Proposal, ProposalFamily},
        r#loop::Plan,
    },
    ranker::bundle::VerifiedBundle,
    ranker::stage1b::TokenProjection,
    run::learned::LearnedAdmission,
    worker::{Request, Response},
};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const R782: &str = crate::ranker::cuda::release::R782;
const SEED_PROTOCOL: &[u8] =
    b"jope.discover.evaluation-seed/v1:sha256(seed_le,round_le,slot_le,rung_u8),le64(digest[0..8])";
const GATE_RULE: &str = "jope.r782.surrogate.v2:F<100;fitness-enforces-(F<100)==gate";

/// Typed origin from the actual seed generator or proposer. The provider never
/// infers a family from DSL syntax, model-prior features, or a hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperatorOrigin {
    Seed,
    Proposal(ProposalFamily),
}

impl OperatorOrigin {
    fn family(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::Proposal(family) => family.name(),
        }
    }
}

/// Supplied by the admitted proposer/seed path, with its roster and Plan pins
/// checked before capture. No implementation is installed by default.
pub(crate) trait OperatorProvenance: Send + Sync {
    fn validate_plan(&self, plan: &Plan) -> Result<()>;
    fn origin(&self, round: u64, dsl_sha256: &str) -> Result<OperatorOrigin>;
}

/// Hash-bound provenance collected directly from the frozen seed roster and
/// proposer `Proposal` objects. Missing or duplicate origins fail closed.
pub(crate) struct AdmittedOperatorProvenance {
    plan_sha256: String,
    origins: BTreeMap<(u64, String), OperatorOrigin>,
}

impl AdmittedOperatorProvenance {
    pub(crate) fn from_seed_roster(plan: &Plan, seeds: &[String]) -> Result<Self> {
        ensure!(!seeds.is_empty(), "admitted seed roster is empty");
        let mut origins = BTreeMap::new();
        for dsl in seeds {
            ensure!(!dsl.is_empty(), "empty seed program");
            let sha = digest(dsl.as_bytes());
            ensure!(
                origins.insert((0, sha), OperatorOrigin::Seed).is_none(),
                "duplicate seed program"
            );
        }
        Ok(Self {
            plan_sha256: plan_sha256(plan)?,
            origins,
        })
    }

    pub(crate) fn record_proposals(
        &mut self,
        plan: &Plan,
        round: u64,
        proposals: &[Proposal],
    ) -> Result<()> {
        ensure!(round > 0, "proposal provenance requires a positive round");
        ensure!(
            self.plan_sha256 == plan_sha256(plan)?,
            "operator provenance Plan changed"
        );
        let mut pending = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            ensure!(
                proposal.program.program_sha256 == digest(proposal.program.dsl.as_bytes()),
                "proposer program digest mismatch"
            );
            pending.push((
                (round, proposal.program.program_sha256.clone()),
                OperatorOrigin::Proposal(proposal.family),
            ));
        }
        ensure!(
            pending
                .iter()
                .map(|(key, _)| key)
                .collect::<BTreeSet<_>>()
                .len()
                == pending.len(),
            "duplicate proposal provenance"
        );
        ensure!(
            pending
                .iter()
                .all(|(key, _)| !self.origins.contains_key(key)),
            "proposal provenance already recorded"
        );
        self.origins.extend(pending);
        Ok(())
    }
}

impl OperatorProvenance for AdmittedOperatorProvenance {
    fn validate_plan(&self, plan: &Plan) -> Result<()> {
        ensure!(
            self.plan_sha256 == plan_sha256(plan)?,
            "operator provenance Plan differs"
        );
        Ok(())
    }

    fn origin(&self, round: u64, dsl_sha256: &str) -> Result<OperatorOrigin> {
        self.origins
            .get(&(round, dsl_sha256.to_owned()))
            .copied()
            .context("program has no admitted seed/proposer origin")
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V2FoldSpec {
    version: u32,
    matrix_sha256: String,
    target: String,
    features: Vec<String>,
    folds: Vec<V2Fold>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V2Fold {
    repeat: u8,
    train: Vec<usize>,
    validation: Vec<usize>,
}

/// Provider over actual admitted r782 state. Construction is crate-private and
/// requires a verified bundle, the exact Plan, and actual bound FIT conditioning.
pub(crate) struct QualifiedCaptureProvider {
    plan: Plan,
    inputs: NativeCaptureInputs,
    conditioning_sha256: String,
    evaluator_sha256: String,
    tokens: Vec<Vec<f32>>,
    mask: Vec<bool>,
    provenance: Arc<dyn OperatorProvenance>,
}

impl QualifiedCaptureProvider {
    pub(crate) fn from_admitted(
        bundle: &VerifiedBundle,
        plan: &Plan,
        learned: &LearnedAdmission,
        provenance: Arc<dyn OperatorProvenance>,
    ) -> Result<Self> {
        ensure!(
            bundle.digest().to_string() == R782,
            "qualified provider requires admitted r782 bundle"
        );
        ensure!(
            format!("sha256:{}", plan.bundle_digest) == R782,
            "Plan does not pin admitted r782 bundle"
        );
        ensure!(
            plan.objective == "rmse",
            "r782 v2 evaluator supports rmse only"
        );
        ensure!(
            hex(&plan.matrix_sha256, 64),
            "Plan FIT matrix identity is invalid"
        );
        provenance.validate_plan(plan)?;
        let conditioning = learned.research_conditioning(plan)?;
        let conditioning_sha256 = conditioning_sha256(plan, conditioning)?;
        let evaluator_sha256 = evaluator_sha256();
        let inputs =
            NativeCaptureInputs::from_bundle(bundle, plan, &conditioning_sha256, conditioning)?;
        Ok(Self {
            plan: plan.clone(),
            inputs,
            conditioning_sha256,
            evaluator_sha256,
            tokens: conditioning.tokens.iter().map(|row| row.to_vec()).collect(),
            mask: conditioning
                .mask
                .iter()
                .map(|value| *value == 1.0)
                .collect(),
            provenance,
        })
    }

    /// Create an Engine context only from accepted metadata supplied by the
    /// trusted job loader. Dataset/corpus hashes are validated by that loader;
    /// this layer binds them to the exact execution inputs and frozen DOPE.
    pub(crate) fn engine_context(
        self: &Arc<Self>,
        identity: RecordIdentity,
        config: CaptureConfig,
    ) -> Result<Arc<CaptureContext>> {
        ensure!(
            identity.evidence == RecordEvidence::Engine,
            "Engine provider requires Engine identity"
        );
        ensure!(
            identity.task == RecordTask::Regression,
            "r782 provider supports regression only"
        );
        ensure!(
            identity.partition == "train",
            "research capture is TRAIN-only"
        );
        ensure!(
            identity.run_id == self.plan.run_id.to_string(),
            "research run identity mismatch"
        );
        ensure!(
            identity.problem_sha256 == self.plan.problem_digest,
            "research problem identity mismatch"
        );
        ensure!(
            identity.bundle_sha256 == self.plan.bundle_digest,
            "research bundle identity mismatch"
        );
        ensure!(
            identity.conditioning_sha256 == self.conditioning_sha256,
            "FIT conditioning identity mismatch"
        );
        ensure!(
            identity.evaluator_sha256 == self.evaluator_sha256,
            "evaluator source identity mismatch"
        );
        ensure!(
            identity.dope_weights_sha256 == DOPE_SHA256,
            "frozen DOPE identity mismatch"
        );
        ensure!(
            hex(&identity.corpus_manifest_sha256, 64),
            "corpus manifest identity missing"
        );
        ensure!(
            hex(&identity.dataset_sha256, 64),
            "dataset identity missing"
        );
        ensure!(hex(&identity.family_sha256, 64), "family identity missing");
        ensure!(
            hex(&identity.source_commit, 40),
            "source commit identity missing"
        );
        ensure!(
            config.seed_protocol_sha256 == seed_protocol_sha256(),
            "capture seed protocol mismatch"
        );
        let binding = Binding {
            config,
            identity,
            seed: self.plan.seed,
            tokens: self.tokens.clone(),
            mask: self.mask.clone(),
            matrix_sha256: self.plan.matrix_sha256.clone(),
        };
        Ok(Arc::new(CaptureContext::engine(
            binding,
            Arc::clone(self) as Arc<dyn CaptureProvider>,
            &self.plan,
        )?))
    }
}

impl CaptureProvider for QualifiedCaptureProvider {
    fn native_inputs(&self) -> Option<&NativeCaptureInputs> {
        Some(&self.inputs)
    }

    fn validate_binding(&self, binding: &Binding, plan: &Plan) -> Result<()> {
        ensure!(plan == &self.plan, "qualified provider Plan changed");
        ensure!(
            binding.identity.evidence == RecordEvidence::Engine,
            "synthetic input cannot use qualified provider"
        );
        ensure!(
            binding.identity.task == RecordTask::Regression,
            "qualified provider requires regression"
        );
        ensure!(
            binding.identity.evaluator_sha256 == self.evaluator_sha256,
            "evaluator identity changed"
        );
        ensure!(
            binding.identity.conditioning_sha256 == self.conditioning_sha256,
            "conditioning identity changed"
        );
        ensure!(
            binding.config.seed_protocol_sha256 == seed_protocol_sha256(),
            "seed protocol identity changed"
        );
        self.provenance.validate_plan(plan)?;
        self.inputs.validate_binding(binding, plan)
    }

    fn observe(
        &self,
        round: u64,
        dsls: &[String],
        measured: &[RecordedReveal],
    ) -> Result<SlateObservation> {
        let programs = dsls
            .iter()
            .map(|dsl| {
                let dsl_sha256 = digest(dsl.as_bytes());
                let origin = self.provenance.origin(round, &dsl_sha256)?;
                self.inputs.program(dsl, origin.family())
            })
            .collect::<Result<Vec<RecordedProgram>>>()?;
        let revealed_indices = if round == 0 {
            ensure!(
                measured.is_empty(),
                "seed round cannot have measured reveals"
            );
            Vec::new()
        } else {
            super::capture::select_od13_indices(round, measured)?
        };
        Ok(SlateObservation {
            programs,
            revealed_indices,
        })
    }

    fn measurement(&self, request: &Request, response: &Response) -> Result<Rung0Measurement> {
        self.provenance
            .origin(request.attempt.round, &digest(request.dsl.as_bytes()))?;
        decode_v2_measurement(&self.plan, request, response)
    }

    fn label_seed(&self, round: u64, proposal_index: usize) -> Result<u64> {
        let slot =
            u32::try_from(proposal_index).context("proposal index exceeds worker slot range")?;
        Ok(crate::discover::r#loop::evaluation_seed(
            self.plan.seed,
            round,
            slot,
            0,
        ))
    }
}

fn decode_v2_measurement(
    plan: &Plan,
    request: &Request,
    response: &Response,
) -> Result<Rung0Measurement> {
    ensure!(
        request.attempt == response.attempt,
        "worker Attempt identity mismatch"
    );
    ensure!(
        request.attempt.run == plan.run_id,
        "request run differs from bound Plan"
    );
    ensure!(
        request.attempt.round <= plan.evaluations,
        "request round exceeds bound Plan"
    );
    ensure!(
        request.matrix_path == plan.matrix_path,
        "request FIT path differs from bound Plan"
    );
    ensure!(
        request.objective.as_str() == Some(plan.objective.as_str()),
        "request objective differs from bound Plan"
    );
    ensure!(
        request.limits.wall_ms == plan.evaluation_wall_ms
            && request.limits.rss_bytes == plan.evaluation_rss_bytes,
        "request limits differ from bound Plan"
    );
    ensure!(
        request.seed
            == crate::discover::r#loop::evaluation_seed(
                plan.seed,
                request.attempt.round,
                request.attempt.slot,
                0,
            ),
        "request seed differs from frozen evaluation protocol"
    );
    let spec: V2FoldSpec = serde_json::from_value(request.fold_spec.clone())
        .context("invalid frozen r782 v2 fold specification")?;
    validate_v2_request(plan, &spec)?;
    if let Some(code) = response.failure {
        ensure!(
            response.scores.is_empty()
                && response.fold_scores.is_empty()
                && response.predictions_digest.is_empty(),
            "failed response carries scores"
        );
        return Ok(Rung0Measurement::Failed { code });
    }
    ensure!(
        response.scores.len() == 3,
        "r782 evaluator must return three values"
    );
    ensure!(
        response.scores.iter().all(|score| score.is_finite()),
        "nonfinite r782 score"
    );
    ensure!(
        response.fold_scores.len() == 2,
        "r782 evaluator must return two fold losses"
    );
    ensure!(
        response
            .fold_scores
            .iter()
            .all(|score| score.is_finite() && *score >= 0.0),
        "invalid r782 fold loss"
    );
    let objective = 0.5 * (response.fold_scores[0] + response.fold_scores[1]);
    ensure!(
        response.scores[1].to_bits() == objective.to_bits(),
        "objective differs from exact fold aggregate"
    );
    ensure!(
        response.scores[2].to_bits() == objective.to_bits(),
        "worker output differs from frozen v2 contract"
    );
    ensure!(
        hex(&response.predictions_digest, 64),
        "prediction digest missing"
    );
    let surrogate = response.scores[0];
    // The exact evaluator source enforces (F < 100) == gate; F is score 0.
    // Score 2 is user RMSE and never participates in gate decoding.
    Ok(Rung0Measurement::Success {
        surrogate,
        objective,
        gate: surrogate < 100.0,
    })
}

fn validate_v2_request(plan: &Plan, spec: &V2FoldSpec) -> Result<()> {
    ensure!(spec.version == 2, "qualified evaluator requires frozen v2");
    ensure!(
        spec.matrix_sha256 == plan.matrix_sha256,
        "fold FIT identity differs from Plan"
    );
    ensure!(
        spec.target == plan.target && spec.features == plan.features,
        "fold columns differ from Plan"
    );
    ensure!(
        plan.objective == "rmse" && spec.folds.len() == 2,
        "unsupported r782 evaluator request"
    );
    let mut validation = BTreeSet::new();
    let mut universe: Option<BTreeSet<usize>> = None;
    let mut fold_sets = Vec::new();
    for fold in &spec.folds {
        ensure!(fold.repeat == 0, "r782 v2 supports one FIT fold repeat");
        let train: BTreeSet<_> = fold.train.iter().copied().collect();
        let valid: BTreeSet<_> = fold.validation.iter().copied().collect();
        ensure!(
            train.len() == fold.train.len() && valid.len() == fold.validation.len(),
            "duplicate fold row"
        );
        ensure!(
            train.len() >= 2 && valid.len() >= 2 && train.is_disjoint(&valid),
            "invalid fold partition"
        );
        let rows: BTreeSet<_> = train.union(&valid).copied().collect();
        ensure!(
            rows.iter().copied().eq(0..rows.len()),
            "fold does not cover a contiguous FIT roster"
        );
        if let Some(expected) = &universe {
            ensure!(&rows == expected, "folds use different FIT rosters");
        } else {
            universe = Some(rows.clone());
        }
        ensure!(valid.is_disjoint(&validation), "validation folds overlap");
        validation.extend(valid.iter().copied());
        fold_sets.push((train, valid));
    }
    let all_rows = universe.context("missing FIT folds")?;
    ensure!(
        validation == all_rows,
        "validation folds do not cover the full FIT roster"
    );
    ensure!(
        fold_sets[0].0 == fold_sets[1].1 && fold_sets[0].1 == fold_sets[1].0,
        "r782 folds are not complementary"
    );
    Ok(())
}

fn plan_sha256(plan: &Plan) -> Result<String> {
    Ok(digest(&serde_json::to_vec(plan)?))
}

fn evaluator_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"jope.research.r782-v2-evaluator/v1\0");
    hash.update(include_bytes!("../worker/evaluate.rs"));
    hash.update(include_bytes!("../run/surrogate.rs"));
    hash.update(GATE_RULE.as_bytes());
    format!("{:x}", hash.finalize())
}

fn seed_protocol_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"jope.research.evaluation-seed/v1\0");
    hash.update(SEED_PROTOCOL);
    hash.update(include_bytes!("../discover/loop.rs"));
    format!("{:x}", hash.finalize())
}

fn conditioning_sha256(plan: &Plan, conditioning: &TokenProjection) -> Result<String> {
    ensure!(
        conditioning
            .tokens
            .iter()
            .flatten()
            .all(|value| value.is_finite()),
        "nonfinite FIT conditioning"
    );
    ensure!(
        conditioning
            .mask
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0),
        "invalid FIT conditioning mask"
    );
    let mut hash = Sha256::new();
    hash.update(b"jope.research.fit-conditioning/v1\0");
    hash.update(serde_json::to_vec(plan)?);
    for value in conditioning.tokens.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
    for value in conditioning.mask {
        hash.update([u8::from(value == 1.0)]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discover::{propose::Program, r#loop::Plan},
        worker::{Attempt, Limits, Response, WorkerFailure},
    };
    use jope_api::{ids::RunId, Effort};
    use serde_json::json;

    fn plan() -> Plan {
        Plan {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse::<RunId>().unwrap(),
            dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
            problem_digest: "a".repeat(64),
            bundle_digest: crate::ranker::cuda::release::R782
                .trim_start_matches("sha256:")
                .to_owned(),
            binary_version: "test".into(),
            matrix_path: "train.jmat".into(),
            matrix_sha256: "b".repeat(64),
            target: "y".into(),
            features: vec!["x".into()],
            objective: "rmse".into(),
            seed: 4,
            planned_workers: 1,
            effort: Effort::Fast,
            evaluations: 4,
            wall_seconds: 5,
            evaluation_wall_ms: 1000,
            evaluation_rss_bytes: 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        }
    }

    fn request(plan: &Plan) -> Request {
        Request {
            attempt: Attempt {
                run: plan.run_id.clone(),
                round: 1,
                slot: 0,
                attempt: 8,
            },
            matrix_path: plan.matrix_path.clone(),
            fold_spec: json!({"version":2,"matrix_sha256":plan.matrix_sha256,"target":"y","features":["x"],
                "folds":[{"repeat":0,"train":[0,1],"validation":[2,3]},
                         {"repeat":0,"train":[2,3],"validation":[0,1]}]}),
            dsl: "(pipe X (ridge))".into(),
            objective: json!("rmse"),
            seed: crate::discover::r#loop::evaluation_seed(plan.seed, 1, 0, 0),
            limits: Limits {
                wall_ms: plan.evaluation_wall_ms,
                rss_bytes: 1024,
            },
        }
    }

    fn response(request: &Request, fitness: f64) -> Response {
        Response {
            attempt: request.attempt.clone(),
            scores: vec![fitness, 1.5, 1.5],
            fold_scores: vec![1.0, 2.0],
            predictions_digest: "c".repeat(64),
            wall_ms: 2,
            failure: None,
        }
    }

    #[test]
    fn v2_measurement_decodes_gate_from_f_and_binds_attempt_and_fold_receipt() {
        let plan = plan();
        let req = request(&plan);
        let pass = decode_v2_measurement(&plan, &req, &response(&req, 0.5)).unwrap();
        assert!(matches!(pass, Rung0Measurement::Success { gate: true, .. }));
        let failed_f = decode_v2_measurement(&plan, &req, &response(&req, 100.0)).unwrap();
        assert!(matches!(
            failed_f,
            Rung0Measurement::Success { gate: false, .. }
        ));
        assert_ne!(
            response(&req, 0.5).scores[2] < 100.0,
            matches!(failed_f, Rung0Measurement::Success { gate: true, .. })
        );
        let mut bad = response(&req, 0.5);
        bad.attempt.slot = 1;
        assert!(decode_v2_measurement(&plan, &req, &bad).is_err());
    }

    #[test]
    fn v2_measurement_rejects_wrong_fold_matrix_objective_and_heldout_fields() {
        let plan = plan();
        let mut req = request(&plan);
        req.fold_spec["matrix_sha256"] = json!("d".repeat(64));
        assert!(decode_v2_measurement(&plan, &req, &response(&req, 0.5)).is_err());
        req = request(&plan);
        req.objective = json!("mae");
        assert!(decode_v2_measurement(&plan, &req, &response(&req, 0.5)).is_err());
        req = request(&plan);
        req.limits.wall_ms += 1;
        assert!(decode_v2_measurement(&plan, &req, &response(&req, 0.5)).is_err());
        req = request(&plan);
        req.seed += 1;
        assert!(decode_v2_measurement(&plan, &req, &response(&req, 0.5)).is_err());
        req = request(&plan);
        req.fold_spec["test_rows"] = json!([4, 5]);
        assert!(decode_v2_measurement(&plan, &req, &response(&req, 0.5)).is_err());
        req = request(&plan);
        let failure = Response::failed(req.attempt.clone(), WorkerFailure::Timeout, 3);
        assert!(matches!(
            decode_v2_measurement(&plan, &req, &failure).unwrap(),
            Rung0Measurement::Failed {
                code: WorkerFailure::Timeout
            }
        ));
    }
    #[test]
    fn operator_provenance_requires_actual_seed_or_proposer_records() {
        let plan = plan();
        let seed = "(pipe X (ridge))".to_string();
        let mut provenance =
            AdmittedOperatorProvenance::from_seed_roster(&plan, &[seed.clone()]).unwrap();
        assert_eq!(
            provenance.origin(0, &digest(seed.as_bytes())).unwrap(),
            OperatorOrigin::Seed
        );
        let proposal = Proposal {
            program: Program::from_dsl("(pipe X (linear))"),
            memo_key: "memo".into(),
            family: ProposalFamily::Insert,
            parents: vec![digest(seed.as_bytes())],
        };
        let proposal_sha = proposal.program.program_sha256.clone();
        provenance.record_proposals(&plan, 1, &[proposal]).unwrap();
        assert_eq!(
            provenance.origin(1, &proposal_sha).unwrap(),
            OperatorOrigin::Proposal(ProposalFamily::Insert)
        );
        assert!(provenance.origin(1, &"f".repeat(64)).is_err());
        let mut changed = plan.clone();
        changed.seed += 1;
        assert!(provenance.validate_plan(&changed).is_err());
    }

    #[test]
    fn conditioning_identity_binds_plan_tokens_and_mask() {
        let plan = plan();
        let projection = TokenProjection {
            tokens: [[0.25; 32]; 10],
            mask: [1.0; 10],
        };
        let first = conditioning_sha256(&plan, &projection).unwrap();
        let mut changed = projection.clone();
        changed.tokens[0][0] = 0.5;
        assert_ne!(first, conditioning_sha256(&plan, &changed).unwrap());
        changed = projection.clone();
        changed.mask[0] = 0.0;
        assert_ne!(first, conditioning_sha256(&plan, &changed).unwrap());
        let mut changed_plan = plan.clone();
        changed_plan.seed += 1;
        assert_ne!(
            first,
            conditioning_sha256(&changed_plan, &projection).unwrap()
        );
    }
}
