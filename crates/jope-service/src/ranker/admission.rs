//! CPU admission-policy primitives for spec §5.5 and §5.8.
//!
//! These functions do not invoke or qualify a ranker, choose OD13, or replace the
//! discovery loop's admission implementation. Real adapter/fixture integration
//! must separately establish M5, S4, S2 and the preregistered quality gates.

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
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidSlate => "expected four revealed costs and eight or sixteen proposals",
            Self::NonFiniteCost => "all slate costs must be finite",
            Self::InvalidRevealed => "expected four to forty-eight distinct measured algorithms",
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
}
