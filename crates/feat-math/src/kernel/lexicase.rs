//! B10's opt-in epsilon-lexicase parent selection.
//!
//! Engine-internal: no operator ID, wire stage, program byte or model byte. The default parent
//! selection stays tournament-3, and nothing here runs unless a caller selects it explicitly.
//!
//! Cases are training rows only. The caller supplies each candidate's residual on each training
//! case; held-out rows and labels never enter the matrix, so a selection cannot read them.
//!
//! Semantics, the semi-dynamic form of La Cava, Spector and Danai (2016) and La Cava et al. (2019):
//! - an error is `|residual|`; a non-finite residual is the error `+inf`, so it never passes a
//!   filter that a finite error fails;
//! - `epsilon_j` is the median absolute deviation of case `j`'s finite errors over the whole
//!   candidate population, fixed for the call (`0` when the case has no finite error);
//! - each selection event filters the remaining pool case by case, keeping every candidate whose
//!   error is at most the pool's current best error plus `epsilon_j`, and stops when one candidate
//!   remains or the cases run out; a remaining tie is broken by a uniform draw.
//!
//! Draw order, frozen by [`PROFILE`] and pinned by a trace test: a down-sampled call first draws
//! its case subset by a partial Fisher-Yates over `0..cases`, one draw per sampled case, and keeps
//! the subset ascending. Each event then draws its cases lazily by a partial Fisher-Yates over the
//! call's case list, restarted from ascending order for every event, one draw per case it examines.
//! It draws a tie last, and only when more than one candidate remains. A population of one
//! candidate therefore draws nothing. Every bounded draw is `next_u32() % n`, exactly as the legacy
//! tournament draws its contenders.
use super::{Error, Result};
use crate::Pcg32;

/// The selection identity recorded in traces and receipts. Any change to the rules above needs a
/// new identity, never an edit of this one.
pub const PROFILE: &str = "feat-math.parent-selection.epsilon-lexicase.v1";

/// The most candidates one call accepts.
pub const MAX_CANDIDATES: usize = 1 << 16;
/// The most training cases one call accepts.
pub const MAX_CASES: usize = 1 << 20;
/// The most matrix entries (`candidates * cases`) one call accepts.
pub const MAX_ENTRIES: usize = 1 << 25;

/// How each case's tolerance is set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EpsilonRule {
    /// The median absolute deviation of the case's finite errors over the population.
    #[default]
    MedianAbsoluteDeviation,
    /// No tolerance: plain lexicase, which keeps only exact error ties.
    Zero,
}

/// Which training cases a call uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CaseSampling {
    /// Every training case.
    #[default]
    All,
    /// One subset of `ceil(cases * per_mille / 1000)` cases, at least one, drawn once per call and
    /// shared by all of the call's events (down-sampled lexicase). `per_mille` is in `1..=1000`.
    DownSampled { per_mille: u16 },
}

/// A complete selection configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LexicaseConfig {
    pub epsilon: EpsilonRule,
    pub sampling: CaseSampling,
}

/// Absolute training-case errors, row-major by candidate: entry `(i, j)` is candidate `i`'s error
/// on case `j`. Built only from residuals, so every entry is `+inf` or finite and non-negative.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseErrors {
    candidates: usize,
    cases: usize,
    values: Vec<f64>,
}

impl CaseErrors {
    /// `residuals[i * cases + j]` is candidate `i`'s residual on training case `j`.
    pub fn from_residuals(candidates: usize, cases: usize, residuals: &[f64]) -> Result<Self> {
        if candidates == 0 || cases == 0 {
            return Err(Error::Shape(
                "lexicase needs at least one candidate and one case".into(),
            ));
        }
        let entries = candidates
            .checked_mul(cases)
            .filter(|&entries| {
                candidates <= MAX_CANDIDATES && cases <= MAX_CASES && entries <= MAX_ENTRIES
            })
            .ok_or_else(|| Error::Resource("lexicase case-error matrix is too large".into()))?;
        if residuals.len() != entries {
            return Err(Error::Shape(
                "lexicase residuals do not match candidates x cases".into(),
            ));
        }
        let values = residuals
            .iter()
            .map(|residual| {
                if residual.is_finite() {
                    residual.abs()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        Ok(Self {
            candidates,
            cases,
            values,
        })
    }

    pub fn candidates(&self) -> usize {
        self.candidates
    }

    pub fn cases(&self) -> usize {
        self.cases
    }

    /// Candidate `candidate`'s error on case `case`.
    pub fn error(&self, candidate: usize, case: usize) -> f64 {
        self.values[candidate * self.cases + case]
    }
}

/// One selection event: the cases it examined in order, the pool size after each filter, the
/// selected candidate, and whether a final tie draw was needed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventTrace {
    pub cases: Vec<u32>,
    pub pool_sizes: Vec<u32>,
    pub selected: u32,
    pub tie_draw: bool,
}

/// Everything needed to replay and audit one call.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LexicaseTrace {
    pub profile: String,
    pub config: LexicaseConfig,
    /// The cases the call used, ascending.
    pub cases: Vec<u32>,
    /// `epsilon_j` for each entry of `cases`, in the same order.
    pub epsilons: Vec<f64>,
    pub events: Vec<EventTrace>,
}

/// The selected parents, by candidate index, with the call's trace.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub parents: Vec<usize>,
    pub trace: LexicaseTrace,
}

/// Selects `parents` parents by independent epsilon-lexicase events.
pub fn select(
    errors: &CaseErrors,
    config: LexicaseConfig,
    parents: usize,
    rng: &mut Pcg32,
) -> Result<Selection> {
    select_with(errors, config, parents, rng)
}

/// `epsilon_j` for each listed case, in the listed order.
pub fn epsilons(errors: &CaseErrors, cases: &[u32], rule: EpsilonRule) -> Vec<f64> {
    cases
        .iter()
        .map(|&case| match rule {
            EpsilonRule::Zero => 0.0,
            EpsilonRule::MedianAbsoluteDeviation => {
                let mut finite: Vec<f64> = (0..errors.candidates)
                    .map(|candidate| errors.error(candidate, case as usize))
                    .filter(|error| error.is_finite())
                    .collect();
                median_absolute_deviation(&mut finite)
            }
        })
        .collect()
}

/// The fraction of selection events that chose a distinct parent: `distinct / events`, and `0`
/// when there is no event.
pub fn distinct_parent_fraction(parents: &[usize]) -> f64 {
    if parents.is_empty() {
        return 0.0;
    }
    let mut distinct = parents.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len() as f64 / parents.len() as f64
}

/// A source of bounded draws, so tests can script exact draw sequences.
trait Draws {
    /// A value in `0..bound`, for `bound >= 1`.
    fn draw(&mut self, bound: usize) -> usize;
}

impl Draws for Pcg32 {
    fn draw(&mut self, bound: usize) -> usize {
        self.next_u32() as usize % bound
    }
}

fn select_with<D: Draws>(
    errors: &CaseErrors,
    config: LexicaseConfig,
    parents: usize,
    draws: &mut D,
) -> Result<Selection> {
    let cases = sample_cases(errors.cases, config.sampling, draws)?;
    let tolerances = epsilons(errors, &cases, config.epsilon);
    let mut pool = Vec::with_capacity(errors.candidates);
    let mut order = Vec::with_capacity(cases.len());
    let mut selected = Vec::with_capacity(parents);
    let mut events = Vec::with_capacity(parents);
    for _ in 0..parents {
        let event = select_one(errors, &cases, &tolerances, &mut pool, &mut order, draws);
        selected.push(event.selected as usize);
        events.push(event);
    }
    Ok(Selection {
        parents: selected,
        trace: LexicaseTrace {
            profile: PROFILE.into(),
            config,
            cases,
            epsilons: tolerances,
            events,
        },
    })
}

fn sample_cases<D: Draws>(cases: usize, sampling: CaseSampling, draws: &mut D) -> Result<Vec<u32>> {
    // `cases <= MAX_CASES`, so every index fits in u32 and `cases * 1000` cannot overflow.
    let mut all: Vec<u32> = (0..cases as u32).collect();
    match sampling {
        CaseSampling::All => Ok(all),
        CaseSampling::DownSampled { per_mille } => {
            if per_mille == 0 || per_mille > 1000 {
                return Err(Error::Domain(
                    "lexicase down-sampling must keep 1..=1000 per mille of the cases".into(),
                ));
            }
            let take = (cases * usize::from(per_mille)).div_ceil(1000).max(1);
            for slot in 0..take {
                let pick = slot + draws.draw(cases - slot);
                all.swap(slot, pick);
            }
            all.truncate(take);
            all.sort_unstable();
            Ok(all)
        }
    }
}

/// One event. `pool` and `order` are scratch buffers reset here, so every event starts from the
/// whole population and the ascending case list.
fn select_one<D: Draws>(
    errors: &CaseErrors,
    cases: &[u32],
    epsilons: &[f64],
    pool: &mut Vec<u32>,
    order: &mut Vec<usize>,
    draws: &mut D,
) -> EventTrace {
    pool.clear();
    // `candidates <= MAX_CANDIDATES`, so every index fits in u32.
    pool.extend(0..errors.candidates as u32);
    order.clear();
    order.extend(0..cases.len());
    let mut examined = Vec::new();
    let mut pool_sizes = Vec::new();
    let mut slot = 0;
    while pool.len() > 1 && slot < order.len() {
        let pick = slot + draws.draw(order.len() - slot);
        order.swap(slot, pick);
        let position = order[slot];
        let case = cases[position] as usize;
        let best = pool
            .iter()
            .map(|&candidate| errors.error(candidate as usize, case))
            .fold(f64::INFINITY, f64::min);
        // Adding a non-negative tolerance never rounds below `best`, so the best candidate always
        // survives and the pool never empties; an infinite best keeps every infinite error.
        let threshold = best + epsilons[position];
        pool.retain(|&candidate| errors.error(candidate as usize, case) <= threshold);
        examined.push(cases[position]);
        pool_sizes.push(pool.len() as u32);
        slot += 1;
    }
    let tie_draw = pool.len() > 1;
    let selected = if tie_draw {
        pool[draws.draw(pool.len())]
    } else {
        pool[0]
    };
    EventTrace {
        cases: examined,
        pool_sizes,
        selected,
        tie_draw,
    }
}

/// The median absolute deviation of finite, non-negative values, or `0` for no value. Sorts and
/// overwrites `values`.
fn median_absolute_deviation(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let center = sorted_median(values);
    for value in values.iter_mut() {
        *value = (*value - center).abs();
    }
    values.sort_unstable_by(f64::total_cmp);
    sorted_median(values)
}

/// The median of a nonempty ascending slice of finite non-negative values. An even count takes
/// the midpoint `low + (high - low) / 2`, which cannot overflow.
fn sorted_median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n.is_multiple_of(2) {
        let low = sorted[n / 2 - 1];
        let high = sorted[n / 2];
        low + (high - low) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draws replayed from a script; each value must lie below its bound.
    struct Script {
        values: Vec<usize>,
        next: usize,
    }

    impl Script {
        fn new(values: &[usize]) -> Self {
            Self {
                values: values.to_vec(),
                next: 0,
            }
        }

        fn used(&self) -> usize {
            self.next
        }
    }

    impl Draws for Script {
        fn draw(&mut self, bound: usize) -> usize {
            let value = self.values[self.next];
            assert!(value < bound, "scripted draw {value} is not below {bound}");
            self.next += 1;
            value
        }
    }

    /// Four candidates on three cases: B is the generalist, A/C/D are one-case specialists.
    fn specialists() -> CaseErrors {
        #[rustfmt::skip]
        let residuals = [
            0.0, 5.0, -5.0, // A
            1.0, -1.0, 1.0, // B
            5.0, 0.0, 5.0,  // C
            -9.0, 9.0, 0.0, // D
        ];
        CaseErrors::from_residuals(4, 3, &residuals).unwrap()
    }

    const MAD: LexicaseConfig = LexicaseConfig {
        epsilon: EpsilonRule::MedianAbsoluteDeviation,
        sampling: CaseSampling::All,
    };
    const PLAIN: LexicaseConfig = LexicaseConfig {
        epsilon: EpsilonRule::Zero,
        sampling: CaseSampling::All,
    };

    #[test]
    fn median_absolute_deviation_matches_hand_values() {
        // Case 0 errors {0, 1, 5, 9}: median 3, deviations {3, 2, 2, 6}, MAD (2 + 3) / 2 = 2.5.
        // Case 2 errors {5, 1, 5, 0}: median 3, deviations {2, 2, 2, 3}, MAD 2.
        assert_eq!(
            epsilons(&specialists(), &[0, 1, 2], MAD.epsilon),
            [2.5, 2.5, 2.0]
        );
        assert_eq!(
            epsilons(&specialists(), &[2, 0], EpsilonRule::Zero),
            [0.0, 0.0]
        );
        assert_eq!(median_absolute_deviation(&mut [7.0]), 0.0);
        assert_eq!(median_absolute_deviation(&mut []), 0.0);
        assert_eq!(
            median_absolute_deviation(&mut [1.0, 2.0, 3.0, 4.0, 100.0]),
            1.0
        );
        // The even-count midpoint is overflow-free at the top of the range.
        assert_eq!(sorted_median(&[f64::MAX, f64::MAX]), f64::MAX);
    }

    #[test]
    fn nonfinite_residuals_are_infinite_errors_outside_the_tolerance() {
        let errors =
            CaseErrors::from_residuals(3, 2, &[f64::NAN, 1.0, f64::NEG_INFINITY, 2.0, -3.0, -0.0])
                .unwrap();
        assert_eq!(errors.error(0, 0), f64::INFINITY);
        assert_eq!(errors.error(1, 0), f64::INFINITY);
        assert_eq!(errors.error(2, 1).to_bits(), 0.0_f64.to_bits());
        // Only the finite errors {3} of case 0 and {1, 2, 0} of case 1 enter the tolerance.
        assert_eq!(epsilons(&errors, &[0, 1], MAD.epsilon), [0.0, 1.0]);
        // A candidate whose residuals are all non-finite can never be selected against finite
        // competitors: the first filter of every event removes it.
        let errors =
            CaseErrors::from_residuals(2, 2, &[f64::NAN, f64::INFINITY, 4.0, 4.0]).unwrap();
        let mut rng = Pcg32::new(17, 3);
        let selection = select(&errors, MAD, 64, &mut rng).unwrap();
        assert!(selection.parents.iter().all(|&parent| parent == 1));
    }

    #[test]
    fn scripted_events_follow_the_documented_semi_dynamic_rule() {
        let errors = specialists();
        // Order 0, 1: case 0 keeps {A, B} (best 0, tolerance 2.5); case 1 keeps {B} (best 1).
        let mut draws = Script::new(&[0, 0]);
        let selection = select_with(&errors, MAD, 1, &mut draws).unwrap();
        assert_eq!(draws.used(), 2);
        assert_eq!(selection.parents, [1]);
        assert_eq!(
            selection.trace.events,
            [EventTrace {
                cases: vec![0, 1],
                pool_sizes: vec![2, 1],
                selected: 1,
                tie_draw: false,
            }]
        );
        // Plain lexicase on the same order keeps only the case-0 specialist A.
        let mut draws = Script::new(&[0]);
        let plain = select_with(&errors, PLAIN, 1, &mut draws).unwrap();
        assert_eq!((plain.parents, draws.used()), (vec![0], 1));
        // Draw 2 first swaps case 2 to the front: {B, D} survive (best 0, tolerance 2), then the
        // remaining order is [1, 0] and draw 0 examines case 1, which keeps {B}.
        let mut draws = Script::new(&[2, 0]);
        let event = select_with(&errors, MAD, 1, &mut draws)
            .unwrap()
            .trace
            .events;
        assert_eq!(
            (event[0].cases.clone(), event[0].pool_sizes.clone()),
            (vec![2, 1], vec![2, 1])
        );
        // Plain lexicase reaches each specialist when its case comes first.
        for (first, specialist) in [(0, 0), (1, 2), (2, 3)] {
            let mut draws = Script::new(&[first]);
            let selection = select_with(&errors, PLAIN, 1, &mut draws).unwrap();
            assert_eq!(selection.parents, [specialist]);
        }
    }

    #[test]
    fn remaining_ties_take_one_final_uniform_draw() {
        let errors = CaseErrors::from_residuals(3, 2, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap();
        // Both cases are examined without separating anyone, then draw 2 picks the third tie.
        let mut draws = Script::new(&[1, 0, 2]);
        let selection = select_with(&errors, PLAIN, 1, &mut draws).unwrap();
        assert_eq!(draws.used(), 3);
        assert_eq!(selection.parents, [2]);
        let event = &selection.trace.events[0];
        assert_eq!((event.cases.clone(), event.tie_draw), (vec![1, 0], true));
    }

    #[test]
    fn a_single_candidate_draws_nothing() {
        let errors = CaseErrors::from_residuals(1, 3, &[4.0, -1.0, 0.5]).unwrap();
        let mut draws = Script::new(&[]);
        let selection = select_with(&errors, MAD, 5, &mut draws).unwrap();
        assert_eq!((selection.parents, draws.used()), (vec![0; 5], 0));
        assert!(selection
            .trace
            .events
            .iter()
            .all(|event| event.cases.is_empty()));
    }

    #[test]
    fn plain_lexicase_never_selects_a_candidate_dominated_on_every_case() {
        // Candidate 2 is strictly worse than candidate 0 on every case.
        #[rustfmt::skip]
        let residuals = [
            0.5, 3.0, 1.0, 2.0,
            2.0, 0.1, 4.0, 0.0,
            0.6, 3.5, 1.5, 2.5,
        ];
        let errors = CaseErrors::from_residuals(3, 4, &residuals).unwrap();
        let mut rng = Pcg32::new(0xB10, 0x6c657869);
        let selection = select(&errors, PLAIN, 256, &mut rng).unwrap();
        assert!(!selection.parents.contains(&2));
        assert!(selection.parents.contains(&0) && selection.parents.contains(&1));
    }

    #[test]
    fn down_sampling_draws_one_shared_ascending_subset_first() {
        let errors = CaseErrors::from_residuals(2, 5, &[0.0; 10]).unwrap();
        // ceil(5 * 400 / 1000) = 2 cases: draw 3 swaps case 3 into slot 0, draw 0 keeps case 1.
        let config = LexicaseConfig {
            epsilon: EpsilonRule::MedianAbsoluteDeviation,
            sampling: CaseSampling::DownSampled { per_mille: 400 },
        };
        let mut draws = Script::new(&[3, 0, 1, 0, 1]);
        let selection = select_with(&errors, config, 1, &mut draws).unwrap();
        assert_eq!(selection.trace.cases, [1, 3]);
        assert_eq!(selection.trace.epsilons, [0.0, 0.0]);
        // The event examines both sampled cases (order [3, 1] after draw 1), then ties on draw 1.
        assert_eq!(selection.trace.events[0].cases, [3, 1]);
        assert_eq!((selection.parents, draws.used()), (vec![1], 5));
        // A rate rounding below one case still keeps one.
        let config = LexicaseConfig {
            sampling: CaseSampling::DownSampled { per_mille: 1 },
            ..config
        };
        let mut rng = Pcg32::new(1, 1);
        assert_eq!(
            select(&errors, config, 1, &mut rng)
                .unwrap()
                .trace
                .cases
                .len(),
            1
        );
    }

    #[test]
    fn invalid_shapes_and_rates_are_structured_refusals() {
        assert!(matches!(
            CaseErrors::from_residuals(0, 3, &[]),
            Err(Error::Shape(_))
        ));
        assert!(matches!(
            CaseErrors::from_residuals(2, 0, &[]),
            Err(Error::Shape(_))
        ));
        assert!(matches!(
            CaseErrors::from_residuals(2, 2, &[1.0; 3]),
            Err(Error::Shape(_))
        ));
        assert!(matches!(
            CaseErrors::from_residuals(MAX_CANDIDATES + 1, 1, &[]),
            Err(Error::Resource(_))
        ));
        assert!(matches!(
            CaseErrors::from_residuals(1 << 13, 1 << 13, &[]),
            Err(Error::Resource(_))
        ));
        let errors = specialists();
        for per_mille in [0, 1001] {
            let config = LexicaseConfig {
                epsilon: EpsilonRule::Zero,
                sampling: CaseSampling::DownSampled { per_mille },
            };
            let mut rng = Pcg32::new(1, 1);
            assert!(matches!(
                select(&errors, config, 1, &mut rng),
                Err(Error::Domain(_))
            ));
        }
    }

    #[test]
    fn seeded_selection_replays_exactly_and_traces_every_event() {
        let errors = specialists();
        let run = |seed| select(&errors, MAD, 24, &mut Pcg32::new(seed, 0x6c657869)).unwrap();
        let first = run(0xB10);
        assert_eq!(first, run(0xB10));
        assert_eq!(first.trace.profile, PROFILE);
        assert_eq!(first.trace.events.len(), 24);
        for (event, &parent) in first.trace.events.iter().zip(&first.parents) {
            assert_eq!(event.selected as usize, parent);
            assert!(parent < errors.candidates());
            assert_eq!(event.cases.len(), event.pool_sizes.len());
            assert!(event.pool_sizes.windows(2).all(|pair| pair[1] <= pair[0]));
            assert_eq!(
                event.tie_draw,
                event.pool_sizes.last().is_none_or(|&n| n > 1)
            );
        }
        let json = serde_json::to_string(&first.trace).unwrap();
        assert_eq!(
            serde_json::from_str::<LexicaseTrace>(&json).unwrap(),
            first.trace
        );
    }

    /// The frozen draw order of [`PROFILE`]: any change to sampling, case order, filtering or the
    /// tie draw changes this trace and needs a new profile identity, not a new pin.
    #[test]
    fn pinned_draw_trace_for_a_fixed_seed() {
        #[rustfmt::skip]
        let residuals = [
            0.3, -1.2, 2.5, 0.0, 4.1, -0.7,
            1.1, 0.4, -0.2, 3.3, -0.5, 2.2,
            -2.0, 0.9, 0.1, 0.8, 1.5, f64::NAN,
            0.3, -1.2, 2.5, 0.0, 4.1, -0.7,
            5.0, 5.0, -5.0, 0.05, 0.6, 0.2,
        ];
        let errors = CaseErrors::from_residuals(5, 6, &residuals).unwrap();
        let config = LexicaseConfig {
            epsilon: EpsilonRule::MedianAbsoluteDeviation,
            sampling: CaseSampling::DownSampled { per_mille: 500 },
        };
        let mut rng = Pcg32::new(0xB10, 0x6c657869);
        let selection = select(&errors, config, 6, &mut rng).unwrap();
        let trace = serde_json::to_string(&selection.trace).unwrap();
        assert_eq!(trace, PINNED_TRACE);
        assert_eq!(
            selection.parents,
            selection
                .trace
                .events
                .iter()
                .map(|event| event.selected as usize)
                .collect::<Vec<_>>()
        );
    }

    /// Hand check: half of six cases keeps {1, 2, 5}; their tolerances are the MADs of
    /// {1.2, 0.4, 0.9, 1.2, 5}, {2.5, 0.2, 0.1, 2.5, 5} and the finite {0.7, 2.2, 0.7, 0.2}; e.g.
    /// event 3 examines case 2 (best 0.1, so {1, 2} survive within 2.3) and then case 1 (best 0.4,
    /// so only candidate 1 survives within 0.3).
    const PINNED_TRACE: &str = concat!(
        r#"{"profile":"feat-math.parent-selection.epsilon-lexicase.v1","#,
        r#""config":{"epsilon":"MedianAbsoluteDeviation","#,
        r#""sampling":{"DownSampled":{"per_mille":500}}},"#,
        r#""cases":[1,2,5],"#,
        r#""epsilons":[0.29999999999999993,2.3,0.24999999999999997],"#,
        r#""events":["#,
        r#"{"cases":[5],"pool_sizes":[1],"selected":4,"tie_draw":false},"#,
        r#"{"cases":[1],"pool_sizes":[1],"selected":1,"tie_draw":false},"#,
        r#"{"cases":[2,1],"pool_sizes":[2,1],"selected":1,"tie_draw":false},"#,
        r#"{"cases":[1],"pool_sizes":[1],"selected":1,"tie_draw":false},"#,
        r#"{"cases":[1],"pool_sizes":[1],"selected":1,"tie_draw":false},"#,
        r#"{"cases":[2,5],"pool_sizes":[2,1],"selected":1,"tie_draw":false}]}"#,
    );

    #[test]
    fn distinct_parent_fraction_counts_distinct_choices() {
        assert_eq!(distinct_parent_fraction(&[]), 0.0);
        assert_eq!(distinct_parent_fraction(&[3, 3, 3, 3]), 0.25);
        assert_eq!(distinct_parent_fraction(&[2, 0, 1, 0]), 0.75);
        // Plain lexicase spreads selection over all four specialists of the fixture.
        let mut rng = Pcg32::new(0xB10, 0x6c657869);
        let parents = select(&specialists(), PLAIN, 64, &mut rng).unwrap().parents;
        let mut distinct = parents.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct, [0, 2, 3]);
    }
}
