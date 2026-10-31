//! CPU admission-policy primitives for spec §5.5 and §5.8.
//!
//! These functions do not invoke or qualify a ranker, choose OD13, or replace the
//! discovery loop's admission implementation. Real adapter/fixture integration
//! must separately establish M5, S4, S2 and the preregistered quality gates.

use super::{
    adapter::{self, PredictCostsInputs},
    stage1b::TokenProjection,
};
use crate::{discover::elites::ELITE_CAPACITY, warnings};
use jope_api::{error::ErrorCode, Guidance, LandmarkMode, ResolvedGuidance, Warning, WarningState};
use rand_chacha::{rand_core::RngCore, ChaCha12Rng};
use std::{collections::BTreeSet, fmt};

const REVEALED: usize = 4;
const ADMITTED: usize = 4;

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
mod tests {
    use super::*;
    use jope_api::Severity;
    use rand_chacha::rand_core::SeedableRng;

    fn rng(seed: u64) -> ChaCha12Rng {
        ChaCha12Rng::seed_from_u64(seed)
    }

    fn slate(costs: &[f64]) -> Vec<f64> {
        [vec![-1e100; REVEALED], costs.to_vec()].concat()
    }

    #[test]
    fn normal_selection_excludes_revealed_and_preserves_rng() {
        let costs = slate(&[8., 7., 6., 5., 4., 3., 2., 1.]);
        let mut actual = rng(7);
        assert_eq!(select_costs(&costs, false, &mut actual), Ok([7, 6, 5, 4]));
        assert_eq!(actual.next_u64(), rng(7).next_u64());
    }

    #[test]
    fn ties_and_signed_zero_keep_proposal_order() {
        let costs = slate(&[1., -0., 0., -0., 0., 1., 1., 1.]);
        assert_eq!(select_costs(&costs, false, &mut rng(0)), Ok([1, 2, 3, 4]));
        assert_eq!(
            select_costs(&slate(&[0.; 8]), false, &mut rng(0)),
            Ok([0, 1, 2, 3])
        );
    }

    #[test]
    fn exploration_has_two_best_and_two_distinct_random_choices() {
        let costs = slate(&(0..16).map(|v| -(v as f64)).collect::<Vec<_>>());
        let mut sampled = BTreeSet::<usize>::new();
        for seed in 0..100 {
            let mut actual = rng(seed);
            let chosen = select_costs(&costs, true, &mut actual).unwrap();
            assert_eq!(chosen[..2], [15, 14]);
            assert_eq!(chosen.into_iter().collect::<BTreeSet<_>>().len(), 4);
            assert!(chosen[2..].iter().all(|index| *index < 14));
            sampled.extend(chosen[2..].iter().copied());
            let mut replay = rng(seed);
            assert_eq!(select_costs(&costs, true, &mut replay).unwrap(), chosen);
            assert_eq!(actual.next_u64(), replay.next_u64());
        }
        assert_eq!(sampled, (0..14).collect());
    }

    #[test]
    fn selection_refuses_shapes_and_all_nonfinite_positions_before_rng_use() {
        for len in [0, 4, 8, 11, 13, 19, 21] {
            let mut actual = rng(12);
            assert_eq!(
                select_costs(&vec![0.; len], false, &mut actual),
                Err(AdmissionError::InvalidSlate)
            );
            assert_eq!(actual.next_u64(), rng(12).next_u64());
        }
        for explore in [false, true] {
            let len = if explore { 20 } else { 12 };
            for index in 0..len {
                for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let mut costs = vec![0.; len];
                    costs[index] = invalid;
                    let mut actual = rng(12);
                    assert_eq!(
                        select_costs(&costs, explore, &mut actual),
                        Err(AdmissionError::NonFiniteCost)
                    );
                    assert_eq!(actual.next_u64(), rng(12).next_u64());
                }
            }
        }
        assert_eq!(
            select_costs(&[0.; 12], true, &mut rng(0)),
            Err(AdmissionError::InvalidSlate)
        );
        assert_eq!(
            select_costs(&[0.; 20], false, &mut rng(0)),
            Err(AdmissionError::InvalidSlate)
        );
    }

    fn measurements() -> Vec<MeasuredAlgorithm<'static>> {
        ["a", "b", "c", "d", "e", "f"]
            .into_iter()
            .enumerate()
            .map(|(i, id)| MeasuredAlgorithm {
                id,
                surrogate_fitness: 5. - i as f64,
                gate: i % 2 == 0,
            })
            .collect()
    }

    #[test]
    fn reveal_choice_is_explicit_and_keeps_original_measurements_paired() {
        let measured = measurements();
        assert_eq!(
            select_revealed(&measured, RevealedRule::StorageOrder),
            Ok([0, 1, 2, 3])
        );
        let sorted = select_revealed(&measured, RevealedRule::SurrogateFitness).unwrap();
        assert_eq!(sorted, [5, 4, 3, 2]);
        assert_eq!(sorted.map(|i| measured[i].gate), [false, true, false, true]);
        assert_eq!(
            sorted.map(|i| measured[i].surrogate_fitness),
            [0., 1., 2., 3.]
        );
        let mut tied = measured;
        tied.iter_mut().for_each(|v| v.surrogate_fitness = 0.);
        assert_eq!(
            select_revealed(&tied, RevealedRule::SurrogateFitness),
            Ok([0, 1, 2, 3])
        );
    }

    #[test]
    fn reveal_refuses_short_large_duplicate_empty_and_nonfinite_pools() {
        let ids: Vec<_> = (0..ELITE_CAPACITY)
            .map(|i| format!("algorithm-{i}"))
            .collect();
        let full: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| MeasuredAlgorithm {
                id,
                surrogate_fitness: i as f64,
                gate: true,
            })
            .collect();
        assert_eq!(
            select_revealed(&full, RevealedRule::SurrogateFitness),
            Ok([0, 1, 2, 3])
        );
        for rule in [RevealedRule::StorageOrder, RevealedRule::SurrogateFitness] {
            assert_eq!(
                select_revealed(&[], rule),
                Err(AdmissionError::InvalidRevealed)
            );
            let mut measured = measurements();
            assert_eq!(
                select_revealed(&measured[..3], rule),
                Err(AdmissionError::InvalidRevealed)
            );
            measured[5].id = measured[0].id;
            assert_eq!(
                select_revealed(&measured, rule),
                Err(AdmissionError::InvalidRevealed)
            );
            measured[5].id = "";
            assert_eq!(
                select_revealed(&measured, rule),
                Err(AdmissionError::InvalidRevealed)
            );
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut measured = measurements();
                measured[5].surrogate_fitness = invalid;
                assert_eq!(
                    select_revealed(&measured, rule),
                    Err(AdmissionError::InvalidRevealed)
                );
            }
            assert_eq!(
                select_revealed(&vec![measurements()[0]; 49], rule),
                Err(AdmissionError::InvalidRevealed)
            );
        }
    }

    #[test]
    fn unqualified_auto_is_heuristic_with_catalogued_warning_and_real_reason() {
        for reason in [
            GuidanceUnavailableReason::Integrity,
            GuidanceUnavailableReason::Parity,
            GuidanceUnavailableReason::Quality,
            GuidanceUnavailableReason::Capability,
            GuidanceUnavailableReason::RevealedDecision,
        ] {
            let (plan, warnings) =
                resolve_unqualified(Guidance::Auto, LandmarkMode::Full, reason).unwrap();
            assert_eq!(plan.value, Guidance::Heuristic);
            assert!(!plan.prior_only);
            assert_eq!(plan.reason, reason.message());
            assert_eq!(
                warnings.len(),
                if reason == GuidanceUnavailableReason::Integrity {
                    2
                } else {
                    1
                }
            );
            assert_eq!(warnings[0].code, "GUIDANCE_HEURISTIC");
            assert_eq!(warnings[0].severity, Severity::Info);
            assert!(!warnings[0].host);
            assert_eq!(warnings[0].message, plan.reason);
            if reason == GuidanceUnavailableReason::Integrity {
                assert_eq!(warnings[1].code, "MODEL_INTEGRITY_FAILED");
                assert_eq!(warnings[1].severity, Severity::Error);
            }
        }
    }

    #[test]
    fn learned_is_refused_and_progressive_remains_unavailable_for_every_request() {
        assert_eq!(
            resolve_unqualified(
                Guidance::Learned,
                LandmarkMode::Full,
                GuidanceUnavailableReason::Integrity
            )
            .unwrap_err(),
            ErrorCode::ModelIntegrityFailed
        );
        for reason in [
            GuidanceUnavailableReason::Parity,
            GuidanceUnavailableReason::Quality,
            GuidanceUnavailableReason::Capability,
            GuidanceUnavailableReason::RevealedDecision,
        ] {
            assert_eq!(
                resolve_unqualified(Guidance::Learned, LandmarkMode::Full, reason).unwrap_err(),
                ErrorCode::CapabilityUnavailable
            );
        }
        for requested in [Guidance::Auto, Guidance::Learned, Guidance::Heuristic] {
            assert_eq!(
                resolve_unqualified(
                    requested,
                    LandmarkMode::Progressive,
                    GuidanceUnavailableReason::Integrity
                )
                .unwrap_err(),
                ErrorCode::CapabilityUnavailable
            );
        }
        let (plan, warnings) = resolve_unqualified(
            Guidance::Heuristic,
            LandmarkMode::None,
            GuidanceUnavailableReason::Integrity,
        )
        .unwrap();
        assert_eq!(plan.value, Guidance::Heuristic);
        assert!(!plan.prior_only);
        assert!(warnings.is_empty());
    }

    struct AdmissionSlateFixture {
        keys: Vec<String>,
        features: Vec<[f32; adapter::PROG_FEATURES]>,
        projections: Vec<[f32; adapter::PROJ_FEATURES]>,
        conditioning: TokenProjection,
    }

    impl AdmissionSlateFixture {
        fn new() -> Self {
            Self {
                keys: (0..22).map(|i| format!("program-{i}")).collect(),
                features: (0..22)
                    .map(|i| [i as f32; adapter::PROG_FEATURES])
                    .collect(),
                projections: (0..22)
                    .map(|i| [100.0 + i as f32; adapter::PROJ_FEATURES])
                    .collect(),
                conditioning: TokenProjection {
                    tokens: std::array::from_fn(|i| {
                        std::array::from_fn(|j| (i * 32 + j) as f32 / 1000.0)
                    }),
                    mask: [1., 1., 1., 0., 1., 1., 0., 1., 0., 1.],
                },
            }
        }

        fn measured(&self) -> Vec<MeasuredAlgorithm<'_>> {
            self.keys[..6]
                .iter()
                .enumerate()
                .map(|(i, key)| MeasuredAlgorithm {
                    id: key,
                    surrogate_fitness: 5.0 - i as f64,
                    gate: i % 2 == 0,
                })
                .collect()
        }

        fn programs(&self) -> Vec<AdmissionProgram<'_>> {
            (0..22)
                .map(|i| AdmissionProgram {
                    key: &self.keys[i],
                    features412: &self.features[i],
                    projection16: &self.projections[i],
                })
                .collect()
        }

        fn prepare(
            &self,
            rule: RevealedRule,
            explore: bool,
        ) -> Result<ServingAdmissionSlate<'_>, AdmissionError> {
            let programs = self.programs();
            ServingAdmissionSlate::prepare(
                &self.measured(),
                &programs[..6],
                &programs[6..if explore { 22 } else { 14 }],
                &self.conditioning,
                rule,
                explore,
            )
        }
    }

    #[test]
    fn assembly_aligns_rows_and_frozen_context_channels_for_both_shapes() {
        let fixture = AdmissionSlateFixture::new();
        for explore in [false, true] {
            let slate = fixture
                .prepare(RevealedRule::SurrogateFitness, explore)
                .unwrap();
            assert_eq!(slate.revealed_pool_indices(), [5, 4, 3, 2]);
            assert_eq!(
                slate.revealed_keys(),
                ["program-5", "program-4", "program-3", "program-2"]
            );
            let expected_proposals: Vec<_> = fixture.keys[6..if explore { 22 } else { 14 }]
                .iter()
                .map(String::as_str)
                .collect();
            assert_eq!(slate.proposal_keys(), expected_proposals);
            let inputs = slate.canonical_inputs();
            assert_eq!(
                (inputs.batch, inputs.candidates),
                (1, if explore { 20 } else { 12 })
            );
            // Frozen cpu_exact_fused.py:1084 / serving caller:412-413:
            // one conditioning row per context batch, independent of N.
            assert_eq!(inputs.tokens, fixture.conditioning.tokens.as_flattened());
            assert_eq!(inputs.token_mask, fixture.conditioning.mask);
            assert_eq!(inputs.revealed_indices, [0, 1, 2, 3]);
            assert_eq!(inputs.revealed_fitness, [0., 1., 2., 3.]);
            assert_eq!(inputs.revealed_gates, [0., 1., 0., 1.]);
            assert_eq!(inputs.scale.to_bits(), adapter::SCALE.to_bits());
            assert!(inputs
                .ctx_feat
                .iter()
                .chain(inputs.base_prior_cost)
                .chain(inputs.sos_prior)
                .all(|&v| v == 0.));
            for (row, source) in [5, 4, 3, 2]
                .into_iter()
                .chain(6..if explore { 22 } else { 14 })
                .enumerate()
            {
                assert_eq!(
                    &inputs.prog_412
                        [row * adapter::PROG_FEATURES..(row + 1) * adapter::PROG_FEATURES],
                    &fixture.features[source]
                );
                assert_eq!(
                    &inputs.proj_16
                        [row * adapter::PROJ_FEATURES..(row + 1) * adapter::PROJ_FEATURES],
                    &fixture.projections[source]
                );
            }
        }
    }

    #[test]
    fn assembly_uses_explicit_reveal_rule_without_rewriting_measurements() {
        let fixture = AdmissionSlateFixture::new();
        let stored = fixture.prepare(RevealedRule::StorageOrder, false).unwrap();
        let sorted = fixture
            .prepare(RevealedRule::SurrogateFitness, false)
            .unwrap();
        assert_eq!(stored.revealed_pool_indices(), [0, 1, 2, 3]);
        assert_eq!(stored.canonical_inputs().revealed_fitness, [5., 4., 3., 2.]);
        assert_eq!(stored.canonical_inputs().revealed_gates, [1., 0., 1., 0.]);
        assert_eq!(stored.proposal_keys(), sorted.proposal_keys());
        assert_ne!(stored.revealed_keys(), sorted.revealed_keys());
    }

    #[test]
    fn assembly_refuses_partial_or_wrong_exploration_slates() {
        let fixture = AdmissionSlateFixture::new();
        let programs = fixture.programs();
        for (count, explore) in [
            (0, false),
            (7, false),
            (9, false),
            (16, false),
            (8, true),
            (15, true),
        ] {
            assert_eq!(
                ServingAdmissionSlate::prepare(
                    &fixture.measured(),
                    &programs[..6],
                    &programs[6..6 + count],
                    &fixture.conditioning,
                    RevealedRule::StorageOrder,
                    explore
                )
                .unwrap_err(),
                AdmissionError::InvalidSlate
            );
        }
    }

    #[test]
    fn assembly_refuses_unaligned_empty_duplicate_or_already_measured_keys() {
        let fixture = AdmissionSlateFixture::new();
        for case in 0..5 {
            let mut programs = fixture.programs();
            match case {
                0 => programs[0].key = "wrong-measurement",
                1 => programs[6].key = "",
                2 => programs[7].key = programs[6].key,
                3 => programs[6].key = programs[5].key,
                _ => {}
            }
            let pool_len = if case == 4 { 5 } else { 6 };
            assert_eq!(
                ServingAdmissionSlate::prepare(
                    &fixture.measured(),
                    &programs[..pool_len],
                    &programs[6..14],
                    &fixture.conditioning,
                    RevealedRule::StorageOrder,
                    false
                )
                .unwrap_err(),
                AdmissionError::InvalidProgram
            );
        }
    }

    #[test]
    fn assembly_refuses_nonfinite_program_channels_and_preserves_finite_values() {
        for row in [0, 5, 6, 13] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for projection in [false, true] {
                    let mut fixture = AdmissionSlateFixture::new();
                    if projection {
                        fixture.projections[row][15] = invalid;
                    } else {
                        fixture.features[row][411] = invalid;
                    }
                    assert_eq!(
                        fixture
                            .prepare(RevealedRule::StorageOrder, false)
                            .unwrap_err(),
                        AdmissionError::InvalidProgram
                    );
                }
            }
        }
        let mut fixture = AdmissionSlateFixture::new();
        fixture.features[0][0] = -0.0;
        fixture.projections[6][0] = f32::MAX;
        let slate = fixture.prepare(RevealedRule::StorageOrder, false).unwrap();
        assert_eq!(
            slate.canonical_inputs().prog_412[0].to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            slate.canonical_inputs().proj_16[4 * adapter::PROJ_FEATURES],
            f32::MAX
        );
    }

    #[test]
    fn assembly_refuses_nonfinite_tokens_and_nonbinary_masks() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., 0.5, 2.] {
            let mut fixture = AdmissionSlateFixture::new();
            fixture.conditioning.mask[9] = invalid;
            assert_eq!(
                fixture
                    .prepare(RevealedRule::StorageOrder, false)
                    .unwrap_err(),
                AdmissionError::InvalidConditioning
            );
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut fixture = AdmissionSlateFixture::new();
            fixture.conditioning.tokens[9][31] = invalid;
            assert_eq!(
                fixture
                    .prepare(RevealedRule::StorageOrder, false)
                    .unwrap_err(),
                AdmissionError::InvalidConditioning
            );
        }
    }

    #[test]
    fn assembly_refuses_surrogate_overflow_in_adapter_fp32_conversion() {
        let fixture = AdmissionSlateFixture::new();
        let programs = fixture.programs();
        for overflow in [f64::MAX, -f64::MAX] {
            let mut measured = fixture.measured();
            measured[0].surrogate_fitness = overflow;
            assert_eq!(
                ServingAdmissionSlate::prepare(
                    &measured,
                    &programs[..6],
                    &programs[6..14],
                    &fixture.conditioning,
                    RevealedRule::StorageOrder,
                    false
                )
                .unwrap_err(),
                AdmissionError::InvalidRevealed
            );
        }
    }

    #[test]
    fn assembled_selection_checks_adapter_compatibility_before_rng_use() {
        let fixture = AdmissionSlateFixture::new();
        for explore in [false, true] {
            let prepared = fixture
                .prepare(RevealedRule::StorageOrder, explore)
                .unwrap();
            let costs = slate(
                &(0..prepared.proposal_keys().len())
                    .map(|i| -(i as f64))
                    .collect::<Vec<_>>(),
            );
            let mut actual = rng(71);
            let mut control = rng(71);
            let selection = prepared.select(&costs, &mut actual);
            if let Err(error) = prepared.inputs() {
                assert!(matches!(error.field, "tokens" | "token_mask"));
                assert!(error.expected > error.actual);
                assert_eq!(selection, Err(AdmissionError::InvalidAdapter));
                assert_eq!(actual.next_u64(), control.next_u64());
                continue;
            }
            let chosen = selection.unwrap();
            assert_eq!(chosen, select_costs(&costs, explore, &mut control).unwrap());
            assert_eq!(actual.next_u64(), control.next_u64());
            assert_eq!(
                prepared.proposal_keys()[chosen[0]],
                if explore { "program-21" } else { "program-13" }
            );
            let mut untouched = rng(71);
            assert_eq!(
                prepared.select(&costs[..costs.len() - 1], &mut untouched),
                Err(AdmissionError::InvalidSlate)
            );
            assert_eq!(untouched.next_u64(), rng(71).next_u64());
        }
    }
}
