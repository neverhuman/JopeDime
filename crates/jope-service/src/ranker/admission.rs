//! CPU admission-policy primitives for spec §5.5 and §5.8.
//!
//! Explicit CPU round composition and admission-policy primitives. Composition
//! invokes the real adapter; it does not grant qualification, choose OD13 or
//! enable guidance defaults. M5/S4/S2 and quality gates remain separate evidence.

use super::{
    adapter::{self, CpuAdapter, PredictCostsInputs, PredictedCosts},
    bundle::VerifiedBundle,
    stage1b::TokenProjection,
};
use crate::discover::r#loop::{Plan, RoundAdmission, RoundContext};
use crate::{discover::elites::ELITE_CAPACITY, warnings};
use anyhow::{ensure, Result};
use jope_api::{error::ErrorCode, Guidance, LandmarkMode, ResolvedGuidance, Warning, WarningState};
use rand_chacha::{rand_core::RngCore, ChaCha12Rng};
use std::{collections::BTreeSet, fmt};

#[cfg(feature = "research")]
use crate::{
    discover::r#loop::RecordedAdmissionContext,
    research::{BehaviorPolicy, RecordedPolicyKind},
};
#[cfg(feature = "research")]
use sha2::{Digest, Sha256};

const REVEALED: usize = 4;
const ADMITTED: usize = 4;

/// Provider-owned feature channels, aligned to canonical programs. Fitness is
/// taken from the scheduler, never supplied or replaced by this provider.
pub struct RoundProgram {
    pub dsl: String,
    pub features412: [f32; adapter::PROG_FEATURES],
    pub projection16: [f32; adapter::PROJ_FEATURES],
}

impl RoundProgram {
    fn borrowed(&self) -> AdmissionProgram<'_> {
        AdmissionProgram {
            key: &self.dsl,
            features412: &self.features412,
            projection16: &self.projection16,
        }
    }
}

pub struct RoundInputs {
    pub pool: Vec<RoundProgram>,
    pub pool_gates: Vec<bool>,
    pub proposals: Vec<RoundProgram>,
    pub conditioning: TokenProjection,
}

/// Featurization/landmark providers must use the bound problem and the actual
/// rung-0 gate measurements. This interface accepts no precomputed model costs.
pub trait RoundInputProvider {
    fn observe(&self, context: &RoundContext<'_>, proposals: &[String]) -> Result<RoundInputs>;
}

/// Explicit composition primitive for CPU replay/integration. The qualified
/// guidance resolver remains responsible for accepted capability/quality evidence;
/// constructing this object alone must never enable Auto or Learned serving.
pub struct CpuRoundAdmission<P> {
    adapter: CpuAdapter,
    provider: P,
    plan: Plan,
    rule: RevealedRule,
}

pub struct RoundReadout {
    pub admitted: Vec<String>,
    pub admitted_indices: [usize; ADMITTED],
    pub costs: PredictedCosts,
}

impl<P: RoundInputProvider> CpuRoundAdmission<P> {
    /// Preserve the signed release boundary. A declaration in a bundle is not a
    /// parity or quality receipt; callers still need the qualified resolver.
    pub fn from_bundle(
        bundle: &VerifiedBundle,
        plan: &Plan,
        provider: P,
        rule: RevealedRule,
    ) -> Result<Self> {
        ensure!(
            format!("sha256:{}", plan.bundle_digest) == bundle.digest().to_string(),
            "admission bundle does not match discovery plan"
        );
        Ok(Self {
            adapter: CpuAdapter::from_bundle(bundle)?,
            provider,
            plan: plan.clone(),
            rule,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_cpu_replay(
        adapter: CpuAdapter,
        plan: Plan,
        provider: P,
        rule: RevealedRule,
    ) -> Self {
        Self {
            adapter,
            provider,
            plan,
            rule,
        }
    }

    /// Construct the same CPU replay adapter from a caller-verified test fixture.
    /// Production bundle loading continues to require `from_bundle`.
    #[cfg(test)]
    pub(crate) fn for_cpu_replay_weights(
        weights: &[u8],
        plan: Plan,
        provider: P,
        rule: RevealedRule,
    ) -> Result<Self> {
        Ok(Self {
            adapter: CpuAdapter::from_bytes_on_device(weights, &candle_core_0_11_0::Device::Cpu)?,
            provider,
            plan,
            rule,
        })
    }

    /// Run once on the complete revealed/proposal slate, then select using the
    /// caller's round RNG. All binding, alignment and model errors precede RNG use.
    pub fn evaluate_round(
        &self,
        context: &RoundContext<'_>,
        proposals: &[String],
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<RoundReadout> {
        #[cfg(feature = "research")]
        {
            self.evaluate_round_inner(context, proposals, explore, rng, None)
        }
        #[cfg(not(feature = "research"))]
        {
            self.evaluate_round_inner(context, proposals, explore, rng)
        }
    }

    fn evaluate_round_inner(
        &self,
        context: &RoundContext<'_>,
        proposals: &[String],
        explore: bool,
        rng: &mut ChaCha12Rng,
        #[cfg(feature = "research")] recorded: Option<&RecordedAdmissionContext<'_>>,
    ) -> Result<RoundReadout> {
        ensure!(
            context.plan == &self.plan,
            "admission plan identity changed"
        );
        ensure!(
            context.round > 0,
            "seed round does not use learned admission"
        );
        let observed = self.provider.observe(context, proposals)?;
        ensure!(
            observed.pool.len() == context.measured.len()
                && observed.pool_gates.len() == context.measured.len()
                && observed
                    .pool
                    .iter()
                    .zip(&context.measured)
                    .all(|(p, m)| p.dsl == m.dsl)
                && observed.proposals.len() == proposals.len()
                && observed
                    .proposals
                    .iter()
                    .zip(proposals)
                    .all(|(p, dsl)| &p.dsl == dsl),
            "admission feature/gate programs do not match committed round"
        );
        let measured: Vec<_> = context
            .measured
            .iter()
            .zip(&observed.pool_gates)
            .map(|(m, &gate)| MeasuredAlgorithm {
                id: m.dsl,
                surrogate_fitness: m.surrogate_fitness,
                gate,
            })
            .collect();
        let pool: Vec<_> = observed.pool.iter().map(RoundProgram::borrowed).collect();
        let candidates: Vec<_> = observed
            .proposals
            .iter()
            .map(RoundProgram::borrowed)
            .collect();
        let slate = ServingAdmissionSlate::prepare(
            &measured,
            &pool,
            &candidates,
            &observed.conditioning,
            self.rule,
            explore,
        )?;
        #[cfg(feature = "research")]
        if let Some(recorded) = recorded {
            validate_recorded_inputs(&slate, context, recorded)?;
        }
        let costs = self.adapter.predict_costs(slate.inputs()?)?;
        let values: Vec<_> = costs.total_cost.iter().map(|&v| f64::from(v)).collect();
        let indices = slate.select(&values, rng)?;
        Ok(RoundReadout {
            admitted: indices.into_iter().map(|i| proposals[i].clone()).collect(),
            admitted_indices: indices,
            costs,
        })
    }
}

impl<P: RoundInputProvider> RoundAdmission for CpuRoundAdmission<P> {
    #[cfg(feature = "research")]
    fn admit_recorded_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
        recorded: &RecordedAdmissionContext<'_>,
    ) -> Result<BehaviorPolicy> {
        let readout =
            self.evaluate_round_inner(context, &proposals, explore, rng, Some(recorded))?;
        Ok(BehaviorPolicy {
            kind: RecordedPolicyKind::Learned,
            explore,
            slate_costs: Some(
                readout
                    .costs
                    .total_cost
                    .into_iter()
                    .map(f64::from)
                    .collect(),
            ),
            admitted: readout.admitted_indices.into(),
        })
    }

    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        Ok(self
            .evaluate_round(context, &proposals, explore, rng)?
            .admitted)
    }
}

/// Check the channels that the engine will persist against the exact prepared
/// graph inputs. This grants no Engine evidence or capture capability itself.
#[cfg(feature = "research")]
fn validate_recorded_inputs(
    slate: &ServingAdmissionSlate<'_>,
    context: &RoundContext<'_>,
    recorded: &RecordedAdmissionContext<'_>,
) -> Result<()> {
    fn same_f32(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
    }
    let observed = recorded.slate;
    ensure!(
        observed.revealed.len() == REVEALED
            && observed.programs.len() == slate.proposal_keys().len(),
        "recorded admission slate shape changed"
    );
    let input = slate.inputs()?;
    let tokens: Vec<_> = recorded
        .conditioning
        .tokens
        .iter()
        .flatten()
        .copied()
        .collect();
    ensure!(
        same_f32(input.tokens, &tokens) && same_f32(input.token_mask, &recorded.conditioning.mask),
        "recorded conditioning differs from model invocation"
    );
    for (i, program) in observed
        .revealed
        .iter()
        .map(|r| &r.program)
        .chain(&observed.programs)
        .enumerate()
    {
        ensure!(
            program.program_sha256
                == format!("{:x}", Sha256::digest(slate.program_keys[i].as_bytes()))
                && same_f32(
                    &program.feat412,
                    &input.prog_412[i * adapter::PROG_FEATURES..(i + 1) * adapter::PROG_FEATURES]
                )
                && program.proj16.len() == adapter::PROJ_FEATURES
                && program.proj16.iter().all(|v| v.is_finite()),
            "recorded program identity/features differ from model invocation"
        );
        // Recorded projections retain the original f64 values; the adapter graph
        // consumes their documented f32 conversion, not a second projection.
        let projection: Vec<_> = program.proj16.iter().map(|&v| v as f32).collect();
        ensure!(
            same_f32(
                &projection,
                &input.proj_16[i * adapter::PROJ_FEATURES..(i + 1) * adapter::PROJ_FEATURES]
            ),
            "recorded projection differs from model invocation"
        );
    }
    for (i, reveal) in observed.revealed.iter().enumerate() {
        ensure!(
            reveal.surrogate.to_bits()
                == context.measured[slate.revealed_pool_indices[i]]
                    .surrogate_fitness
                    .to_bits()
                && slate.revealed_gates[i] == if reveal.gate { 1.0 } else { 0.0 },
            "recorded revealed fitness/gate differs from committed context"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidSlate,
    NonFiniteCost,
    InvalidRevealed,
    InvalidProgram,
    InvalidConditioning,
    InvalidAdapter,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidSlate => "expected four revealed costs and eight or sixteen proposals",
            Self::NonFiniteCost => "all slate costs must be finite",
            Self::InvalidRevealed => "expected four to forty-eight distinct measured algorithms",
            Self::InvalidProgram => {
                "program identities, features and projections must be aligned and finite"
            }
            Self::InvalidConditioning => {
                "conditioning must have finite tokens and binary mask values"
            }
            Self::InvalidAdapter => "adapter shape contract does not accept canonical round inputs",
        })
    }
}

impl std::error::Error for AdmissionError {}

/// Indices into proposals, not into the combined revealed/proposal slate.
/// Equal costs (including signed zero) retain proposal order. Invalid input and
/// non-exploring selection leave the caller's persisted RNG unchanged.
pub fn select_costs(
    slate_costs: &[f64],
    explore: bool,
    rng: &mut ChaCha12Rng,
) -> Result<[usize; ADMITTED], AdmissionError> {
    let proposals = if explore { 16 } else { 8 };
    if slate_costs.len() != REVEALED + proposals {
        return Err(AdmissionError::InvalidSlate);
    }
    if !slate_costs.iter().all(|value| value.is_finite()) {
        return Err(AdmissionError::NonFiniteCost);
    }
    let costs = &slate_costs[REVEALED..];
    let mut order: Vec<_> = (0..proposals).collect();
    order.sort_by(|&a, &b| {
        costs[a]
            .partial_cmp(&costs[b])
            .expect("costs were checked finite")
            .then_with(|| a.cmp(&b))
    });
    let mut selected = [order[0], order[1], order[2], order[3]];
    if explore {
        // Preserve proposal order in the random population. Neither a sampled
        // proposal nor one of the two ranked choices can be selected twice.
        let mut remaining: Vec<_> = (0..proposals)
            .filter(|index| *index != selected[0] && *index != selected[1])
            .collect();
        for slot in &mut selected[2..] {
            *slot = remaining.remove(sample_index(rng, remaining.len()));
        }
    }
    Ok(selected)
}

fn sample_index(rng: &mut ChaCha12Rng, len: usize) -> usize {
    // Same rejection rule as the existing heuristic Admission. The caller has
    // validated a sixteen-proposal slate, so len is fourteen, then thirteen.
    let limit = u64::MAX - u64::MAX % len as u64;
    loop {
        let value = rng.next_u64();
        if value < limit {
            return (value % len as u64) as usize;
        }
    }
}

/// No Default: the OWNER's OD13 decision must be supplied explicitly. This
/// policy utility's ties retain input order; historical NumPy tie parity still
/// requires fixtures and must not be inferred from these arithmetic controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevealedRule {
    StorageOrder,
    SurrogateFitness,
}

/// A rung-0 surrogate measurement. User-objective scores have no field here.
/// Fitness and gate remain paired; selecting an index never normalizes or caps
/// either value. The real adapter owns its documented dtype/channel conversion.
#[derive(Clone, Copy, Debug)]
pub struct MeasuredAlgorithm<'a> {
    pub id: &'a str,
    pub surrogate_fitness: f64,
    pub gate: bool,
}

pub fn select_revealed(
    measured: &[MeasuredAlgorithm<'_>],
    rule: RevealedRule,
) -> Result<[usize; REVEALED], AdmissionError> {
    if !(REVEALED..=ELITE_CAPACITY).contains(&measured.len())
        || measured
            .iter()
            .any(|entry| entry.id.is_empty() || !entry.surrogate_fitness.is_finite())
        || measured
            .iter()
            .map(|entry| entry.id)
            .collect::<BTreeSet<_>>()
            .len()
            != measured.len()
    {
        return Err(AdmissionError::InvalidRevealed);
    }
    let mut order: Vec<_> = (0..measured.len()).collect();
    if rule == RevealedRule::SurrogateFitness {
        order.sort_by(|&a, &b| {
            measured[a]
                .surrogate_fitness
                .partial_cmp(&measured[b].surrogate_fitness)
                .expect("measurements were checked finite")
                .then_with(|| a.cmp(&b))
        });
    }
    Ok([order[0], order[1], order[2], order[3]])
}

/// One already-featurized program. Keys must share the measured pool's canonical
/// identity namespace; this boundary does not canonicalize DSL or verify assets.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionProgram<'a> {
    pub key: &'a str,
    pub features412: &'a [f32; adapter::PROG_FEATURES],
    pub projection16: &'a [f32; adapter::PROJ_FEATURES],
}

impl AdmissionProgram<'_> {
    fn valid(&self) -> bool {
        !self.key.is_empty()
            && self
                .features412
                .iter()
                .chain(self.projection16)
                .all(|v| v.is_finite())
    }
}

/// Owned F12 channels for one round, with explicit reveal/proposal alignment.
///
/// Preparation does not invoke a model, authenticate provenance or establish any
/// qualification. The adapter's real forward pass and the durable discovery
/// context handoff are separate dependencies; this is not Discovery Admission.
#[derive(Debug)]
pub struct ServingAdmissionSlate<'a> {
    program_keys: Vec<&'a str>,
    revealed_pool_indices: [usize; REVEALED],
    prog_412: Vec<f32>,
    proj_16: Vec<f32>,
    tokens: Vec<f32>,
    token_mask: Vec<f32>,
    revealed_fitness: [f32; REVEALED],
    revealed_gates: [f32; REVEALED],
    base_prior_cost: Vec<f32>,
    sos_prior: Vec<f32>,
    explore: bool,
}

impl<'a> ServingAdmissionSlate<'a> {
    /// Preserve measured storage order when binding features, then prepend the
    /// four explicitly selected reveals. No caller randomness is consumed.
    pub fn prepare(
        measured: &[MeasuredAlgorithm<'a>],
        pool: &[AdmissionProgram<'a>],
        proposals: &[AdmissionProgram<'a>],
        conditioning: &TokenProjection,
        rule: RevealedRule,
        explore: bool,
    ) -> Result<Self, AdmissionError> {
        if proposals.len() != if explore { 16 } else { 8 } {
            return Err(AdmissionError::InvalidSlate);
        }
        let indices = select_revealed(measured, rule)?;
        if measured.len() != pool.len()
            || measured.iter().zip(pool).any(|(m, p)| m.id != p.key)
            || !pool.iter().chain(proposals).all(AdmissionProgram::valid)
            || pool
                .iter()
                .chain(proposals)
                .map(|p| p.key)
                .collect::<BTreeSet<_>>()
                .len()
                != pool.len() + proposals.len()
        {
            return Err(AdmissionError::InvalidProgram);
        }
        if !conditioning.tokens.iter().flatten().all(|v| v.is_finite())
            || !conditioning.mask.iter().all(|&v| v == 0.0 || v == 1.0)
        {
            return Err(AdmissionError::InvalidConditioning);
        }
        let revealed_fitness = indices.map(|i| measured[i].surrogate_fitness as f32);
        if !revealed_fitness.iter().all(|v| v.is_finite()) {
            return Err(AdmissionError::InvalidRevealed);
        }
        let rows: Vec<_> = indices
            .iter()
            .map(|&i| pool[i])
            .chain(proposals.iter().copied())
            .collect();
        let candidates = rows.len();
        Ok(Self {
            program_keys: rows.iter().map(|p| p.key).collect(),
            revealed_pool_indices: indices,
            prog_412: rows
                .iter()
                .flat_map(|p| p.features412.iter().copied())
                .collect(),
            proj_16: rows
                .iter()
                .flat_map(|p| p.projection16.iter().copied())
                .collect(),
            tokens: conditioning.tokens.iter().flatten().copied().collect(),
            token_mask: conditioning.mask.to_vec(),
            revealed_fitness,
            revealed_gates: indices.map(|i| if measured[i].gate { 1.0 } else { 0.0 }),
            base_prior_cost: vec![0.0; candidates],
            sos_prior: vec![0.0; candidates * adapter::SOS_FEATURES],
            explore,
        })
    }

    fn canonical_inputs(&self) -> PredictCostsInputs<'_> {
        PredictCostsInputs {
            batch: 1,
            candidates: self.program_keys.len(),
            prog_412: &self.prog_412,
            proj_16: &self.proj_16,
            tokens: &self.tokens,
            token_mask: &self.token_mask,
            revealed_indices: &[0, 1, 2, 3],
            revealed_fitness: &self.revealed_fitness,
            revealed_gates: &self.revealed_gates,
            ctx_feat: &[0.0; adapter::CONTEXT_FEATURES],
            base_prior_cost: &self.base_prior_cost,
            sos_prior: &self.sos_prior,
            scale: adapter::SCALE,
        }
    }

    /// Export only when the adapter accepts the frozen per-context layout.
    /// Refuse an incompatible contract rather than repeat or reshape context
    /// to satisfy it. Shape acceptance alone is not qualification.
    pub fn inputs(&self) -> Result<PredictCostsInputs<'_>, adapter::InputShapeError> {
        let inputs = self.canonical_inputs();
        inputs.validate_shapes()?;
        Ok(inputs)
    }

    pub fn revealed_pool_indices(&self) -> [usize; REVEALED] {
        self.revealed_pool_indices
    }

    pub fn revealed_keys(&self) -> &[&'a str] {
        &self.program_keys[..REVEALED]
    }

    pub fn proposal_keys(&self) -> &[&'a str] {
        &self.program_keys[REVEALED..]
    }

    /// Return original proposal indices using the prepared round's explore flag.
    /// The caller must obtain costs from the real forward on `inputs()`; this
    /// method cannot authenticate that association or qualify a learned result.
    pub fn select(
        &self,
        costs: &[f64],
        rng: &mut ChaCha12Rng,
    ) -> Result<[usize; ADMITTED], AdmissionError> {
        self.inputs().map_err(|_| AdmissionError::InvalidAdapter)?;
        select_costs(costs, self.explore, rng)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidanceUnavailableReason {
    Integrity,
    Parity,
    Quality,
    Capability,
    RevealedDecision,
}

impl GuidanceUnavailableReason {
    fn message(self) -> &'static str {
        match self {
            Self::Integrity => "The ranker bundle failed integrity verification.",
            Self::Parity => "Serving-shaped ranker parity has not been qualified.",
            Self::Quality => "Learned guidance has not passed the required quality comparison.",
            Self::Capability => "Learned guidance is unavailable for this task, mode or backend.",
            Self::RevealedDecision => "The revealed-context rule awaits an OWNER decision.",
        }
    }
}

/// The unqualified branch of plan resolution only. This helper cannot return
/// learned guidance or turn a manifest declaration into empirical evidence.
/// Qualified matrix/prior_only resolution belongs to the later integration.
pub fn resolve_unqualified(
    requested: Guidance,
    landmarks: LandmarkMode,
    reason: GuidanceUnavailableReason,
) -> Result<(ResolvedGuidance, Vec<Warning>), ErrorCode> {
    if landmarks == LandmarkMode::Progressive {
        return Err(ErrorCode::CapabilityUnavailable);
    }
    if requested == Guidance::Learned {
        return Err(if reason == GuidanceUnavailableReason::Integrity {
            ErrorCode::ModelIntegrityFailed
        } else {
            ErrorCode::CapabilityUnavailable
        });
    }
    let explicit = requested == Guidance::Heuristic;
    let message = if explicit {
        "Heuristic guidance was requested."
    } else {
        reason.message()
    };
    let mut warnings = if explicit {
        vec![]
    } else {
        vec![warnings::catalog("GUIDANCE_HEURISTIC")
            .expect("warning is catalogued")
            .event(
                WarningState::Raised,
                None,
                None,
                message,
                "Use heuristic guidance while learned guidance is unavailable.",
            )
            .expect("the warning has no numeric fields")]
    };
    if !explicit && reason == GuidanceUnavailableReason::Integrity {
        warnings.push(
            warnings::catalog("MODEL_INTEGRITY_FAILED")
                .expect("warning is catalogued")
                .event(
                    WarningState::Raised,
                    None,
                    None,
                    reason.message(),
                    "Verify the installed bundle before requesting learned guidance.",
                )
                .expect("the warning has no numeric fields"),
        );
    }
    Ok((
        ResolvedGuidance {
            value: Guidance::Heuristic,
            prior_only: false,
            reason: message.to_owned(),
        },
        warnings,
    ))
}

#[cfg(test)]
#[path = "admission/tests.rs"]
mod tests;
