//! Experimental measured-leaf Gumbel root search.
//!
//! This module selects among caller-supplied legal actions and schedules measured leaves. It is
//! not wired into a default strategy or product driver. Surrogate completion estimates and visits
//! are retained separately from measured, fidelity-specific values. No board-game improvement
//! guarantee is claimed for noisy, adaptive, multi-fidelity search.

use crate::Pcg32;
use std::collections::{BTreeMap, BTreeSet};

pub const PROFILE: &str = "feat-math.root-search.measured-leaf-gumbel.v1";
const RNG_STREAM: u64 = 0x4755_4d42_454c_0001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingLaw {
    /// Gumbel top-k over caller-supplied legal actions; each action can be selected once.
    GumbelTopKWithoutReplacementV1,
    WeightedWithReplacement,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionPrior {
    pub action_id: String,
    /// Positive relative prior weight; normalization is not required.
    pub prior_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledAction {
    pub action_id: String,
    pub prior_weight: f64,
    pub gumbel_noise: f64,
    pub perturbed_log_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleTrace {
    pub profile: &'static str,
    pub law: SamplingLaw,
    pub seed: u64,
    pub requested: usize,
    pub legal_count: usize,
    /// Original caller order and weights, so the exact sampling input can be replayed.
    pub legal_actions: Vec<ActionPrior>,
    /// Unique actions in descending Gumbel perturbed-weight order.
    pub selected: Vec<SampledAction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootSearchError {
    NoLegalActions,
    EmptySample,
    DuplicateAction(String),
    InvalidPrior(String),
    InvalidSamplingTrace,
    InvalidFidelityLadder,
    UnsupportedSamplingLaw,
    RoundAlreadyPending,
    NoPendingRound,
    UnexpectedAction(String),
    DuplicateMeasurement(String),
    DuplicateFitOrdinal(u64),
    WrongFidelity(String),
    NonFiniteMeasurement(String),
    InvalidCompletedEstimate(String),
    UnknownAction(String),
    CounterOverflow,
}

/// Samples `min(requested, legal.len())` unique actions using a domain-separated PCG stream and
/// Gumbel top-k. If `requested` is smaller than the legal set, the remainder is retained in
/// `legal_count` as absent probability mass; it is never silently replaced or sampled twice.
pub fn sample_legal(
    legal: &[ActionPrior],
    requested: usize,
    seed: u64,
) -> Result<SampleTrace, RootSearchError> {
    if legal.is_empty() {
        return Err(RootSearchError::NoLegalActions);
    }
    if requested == 0 {
        return Err(RootSearchError::EmptySample);
    }
    let mut ids = BTreeSet::new();
    for action in legal {
        if !ids.insert(action.action_id.as_str()) {
            return Err(RootSearchError::DuplicateAction(action.action_id.clone()));
        }
        if !action.prior_weight.is_finite() || action.prior_weight <= 0.0 {
            return Err(RootSearchError::InvalidPrior(action.action_id.clone()));
        }
    }

    let mut rng = Pcg32::new(seed, RNG_STREAM);
    let mut ranked: Vec<_> = legal
        .iter()
        .map(|action| {
            let uniform = rng.next_f64().max(f64::MIN_POSITIVE);
            let noise = -(-uniform.ln()).ln();
            SampledAction {
                action_id: action.action_id.clone(),
                prior_weight: action.prior_weight,
                gumbel_noise: noise,
                perturbed_log_weight: action.prior_weight.ln() + noise,
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.perturbed_log_weight
            .total_cmp(&a.perturbed_log_weight)
            .then_with(|| a.action_id.cmp(&b.action_id))
    });
    ranked.truncate(requested.min(legal.len()));

    Ok(SampleTrace {
        profile: PROFILE,
        law: SamplingLaw::GumbelTopKWithoutReplacementV1,
        seed,
        requested,
        legal_count: legal.len(),
        legal_actions: legal.to_vec(),
        selected: ranked,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fidelity {
    pub rank: u16,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredQ {
    pub action_id: String,
    pub fidelity: Fidelity,
    /// Measured loss; lower is better. These values alone drive elimination and final choice.
    pub loss: f64,
    pub fit_ordinal: u64,
}

/// A completed estimate for an action not measured at `target_fidelity`. It is never accepted as
/// a `MeasuredQ` and cannot eliminate an action or certify a root choice.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletedQ {
    pub action_id: String,
    pub target_fidelity: Fidelity,
    pub estimate: f64,
    pub completion_model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundPlan {
    pub round: usize,
    pub fidelity: Fidelity,
    pub action_ids: Vec<String>,
    pub unscheduled_action_ids: Vec<String>,
}

impl RoundPlan {
    pub fn is_complete(&self) -> bool {
        self.unscheduled_action_ids.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HalvingTerminal {
    Running,
    Incomplete { round: usize, reason: &'static str },
    Complete { winner: String, fidelity: Fidelity },
}

/// Stateful sequential halving over the sampled root actions. A partial round is terminal and
/// cannot halve away actions whose measured leaf was never completed.
#[derive(Clone, Debug)]
pub struct GumbelRoot {
    sample: SampleTrace,
    fidelities: Vec<Fidelity>,
    current_actions: Vec<String>,
    round: usize,
    pending: Option<RoundPlan>,
    terminal: HalvingTerminal,
    measured_q: Vec<MeasuredQ>,
    completed_q: Vec<CompletedQ>,
    visits: BTreeMap<String, u64>,
}

impl GumbelRoot {
    pub fn new(sample: SampleTrace, fidelities: Vec<Fidelity>) -> Result<Self, RootSearchError> {
        if sample.selected.is_empty() {
            return Err(RootSearchError::EmptySample);
        }
        if sample.profile != PROFILE || sample.law != SamplingLaw::GumbelTopKWithoutReplacementV1 {
            return Err(RootSearchError::UnsupportedSamplingLaw);
        }
        if sample.legal_count != sample.legal_actions.len() {
            return Err(RootSearchError::InvalidSamplingTrace);
        }
        if sample.requested == 0
            || sample.selected.len() != sample.requested.min(sample.legal_count)
        {
            return Err(RootSearchError::InvalidSamplingTrace);
        }
        let mut legal_ids = BTreeMap::new();
        for action in &sample.legal_actions {
            if !action.prior_weight.is_finite() || action.prior_weight <= 0.0 {
                return Err(RootSearchError::InvalidPrior(action.action_id.clone()));
            }
            if legal_ids
                .insert(action.action_id.as_str(), action.prior_weight)
                .is_some()
            {
                return Err(RootSearchError::DuplicateAction(action.action_id.clone()));
            }
        }
        let mut sampled_ids = BTreeSet::new();
        if sample.selected.iter().any(|action| {
            !sampled_ids.insert(action.action_id.as_str())
                || legal_ids.get(action.action_id.as_str()) != Some(&action.prior_weight)
                || !action.gumbel_noise.is_finite()
                || !action.perturbed_log_weight.is_finite()
                || (action.prior_weight.ln() + action.gumbel_noise).to_bits()
                    != action.perturbed_log_weight.to_bits()
        }) {
            return Err(RootSearchError::DuplicateAction(
                "sample trace contains duplicate actions".to_owned(),
            ));
        }
        if sample.selected.windows(2).any(|pair| {
            pair[0]
                .perturbed_log_weight
                .total_cmp(&pair[1].perturbed_log_weight)
                .is_lt()
                || (pair[0]
                    .perturbed_log_weight
                    .total_cmp(&pair[1].perturbed_log_weight)
                    .is_eq()
                    && pair[0].action_id > pair[1].action_id)
        }) {
            return Err(RootSearchError::InvalidSamplingTrace);
        }
        let mut fidelity_ids = BTreeSet::new();
        if fidelities.is_empty()
            || fidelities
                .iter()
                .any(|fidelity| !fidelity_ids.insert(&fidelity.id))
            || fidelities
                .windows(2)
                .any(|pair| pair[0].rank >= pair[1].rank)
        {
            return Err(RootSearchError::InvalidFidelityLadder);
        }
        let current_actions = sample
            .selected
            .iter()
            .map(|action| action.action_id.clone())
            .collect();
        Ok(Self {
            sample,
            fidelities,
            current_actions,
            round: 0,
            pending: None,
            terminal: HalvingTerminal::Running,
            measured_q: Vec::new(),
            completed_q: Vec::new(),
            visits: BTreeMap::new(),
        })
    }

    pub fn sample(&self) -> &SampleTrace {
        &self.sample
    }

    pub fn terminal(&self) -> &HalvingTerminal {
        &self.terminal
    }

    pub fn measured_q(&self) -> &[MeasuredQ] {
        &self.measured_q
    }

    pub fn completed_q(&self) -> &[CompletedQ] {
        &self.completed_q
    }

    pub fn visits(&self) -> &BTreeMap<String, u64> {
        &self.visits
    }

    /// Plans this fidelity round in sample order. If the measured-fit budget cannot cover all
    /// remaining actions, the plan explicitly lists unscheduled actions; recording it terminates
    /// as incomplete without survivor selection.
    pub fn plan_round(
        &mut self,
        available_fit_slots: usize,
    ) -> Result<Option<RoundPlan>, RootSearchError> {
        if self.pending.is_some() {
            return Err(RootSearchError::RoundAlreadyPending);
        }
        if self.terminal != HalvingTerminal::Running {
            return Ok(None);
        }
        if self.round >= self.fidelities.len() {
            return Ok(None);
        }
        let scheduled_count = available_fit_slots.min(self.current_actions.len());
        let plan = RoundPlan {
            round: self.round,
            fidelity: self.fidelities[self.round].clone(),
            action_ids: self.current_actions[..scheduled_count].to_vec(),
            unscheduled_action_ids: self.current_actions[scheduled_count..].to_vec(),
        };
        self.pending = Some(plan.clone());
        Ok(Some(plan))
    }

    /// Records only measured leaf values. Missing results or a partially scheduled round end the
    /// search as incomplete; completed surrogate estimates are stored through another method.
    pub fn record_round(
        &mut self,
        plan: RoundPlan,
        measured: Vec<MeasuredQ>,
    ) -> Result<(), RootSearchError> {
        let Some(pending) = self.pending.as_ref() else {
            return Err(RootSearchError::NoPendingRound);
        };
        if &plan != pending {
            return Err(RootSearchError::NoPendingRound);
        }
        let allowed: BTreeSet<_> = plan.action_ids.iter().map(String::as_str).collect();
        let mut seen = BTreeSet::new();
        let mut fit_ordinals: BTreeSet<_> = self
            .measured_q
            .iter()
            .map(|value| value.fit_ordinal)
            .collect();
        for value in &measured {
            if !allowed.contains(value.action_id.as_str()) {
                return Err(RootSearchError::UnexpectedAction(value.action_id.clone()));
            }
            if !seen.insert(value.action_id.as_str()) {
                return Err(RootSearchError::DuplicateMeasurement(
                    value.action_id.clone(),
                ));
            }
            if !fit_ordinals.insert(value.fit_ordinal) {
                return Err(RootSearchError::DuplicateFitOrdinal(value.fit_ordinal));
            }
            if value.fidelity != plan.fidelity {
                return Err(RootSearchError::WrongFidelity(value.action_id.clone()));
            }
            if !value.loss.is_finite() {
                return Err(RootSearchError::NonFiniteMeasurement(
                    value.action_id.clone(),
                ));
            }
        }
        let complete_measurements = seen.len() == self.current_actions.len();
        self.pending = None;
        self.measured_q.extend(measured);

        if !plan.is_complete() || !complete_measurements {
            self.terminal = HalvingTerminal::Incomplete {
                round: self.round,
                reason: if !plan.is_complete() {
                    "fit budget did not schedule every action"
                } else {
                    "one or more measured leaves did not complete"
                },
            };
            return Ok(());
        }

        let mut round_values: Vec<_> = self
            .measured_q
            .iter()
            .filter(|value| value.fidelity == plan.fidelity)
            .cloned()
            .collect();
        round_values.sort_by(|a, b| {
            a.loss
                .total_cmp(&b.loss)
                .then_with(|| a.action_id.cmp(&b.action_id))
        });
        let keep = if self.round + 1 == self.fidelities.len() {
            1
        } else {
            self.current_actions.len().div_ceil(2).max(1)
        };
        self.current_actions = round_values
            .into_iter()
            .take(keep)
            .map(|value| value.action_id)
            .collect();
        self.round += 1;
        if self.round == self.fidelities.len() {
            let winner = self.current_actions[0].clone();
            self.terminal = HalvingTerminal::Complete {
                winner,
                fidelity: plan.fidelity,
            };
        }
        Ok(())
    }

    pub fn record_completed_q(&mut self, value: CompletedQ) -> Result<(), RootSearchError> {
        if !self
            .sample
            .selected
            .iter()
            .any(|action| action.action_id == value.action_id)
        {
            return Err(RootSearchError::UnknownAction(value.action_id));
        }
        if !self.fidelities.contains(&value.target_fidelity) {
            return Err(RootSearchError::WrongFidelity(value.action_id));
        }
        if !value.estimate.is_finite() || value.completion_model.is_empty() {
            return Err(RootSearchError::InvalidCompletedEstimate(value.action_id));
        }
        self.completed_q.push(value);
        Ok(())
    }

    pub fn record_visit(&mut self, action_id: &str) -> Result<(), RootSearchError> {
        if !self
            .sample
            .selected
            .iter()
            .any(|action| action.action_id == action_id)
        {
            return Err(RootSearchError::UnknownAction(action_id.to_owned()));
        }
        let visits = self.visits.entry(action_id.to_owned()).or_default();
        *visits = visits
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        Ok(())
    }

    pub fn teacher_target(
        &self,
        law: SamplingLaw,
        fidelity: &Fidelity,
    ) -> Result<TeacherTarget, TeacherTargetError> {
        if law != SamplingLaw::GumbelTopKWithoutReplacementV1 || self.sample.law != law {
            return Err(TeacherTargetError::UnsupportedSamplingLaw);
        }
        if !matches!(self.terminal, HalvingTerminal::Complete { .. }) {
            return Err(TeacherTargetError::IncompleteSearch);
        }
        let mut actions: Vec<_> = self
            .measured_q
            .iter()
            .filter(|value| &value.fidelity == fidelity)
            .map(|value| value.action_id.clone())
            .collect();
        actions.sort();
        actions.dedup();
        if actions.len() != self.sample.selected.len() {
            return Err(TeacherTargetError::MissingMeasuredTarget);
        }
        Ok(TeacherTarget {
            profile: PROFILE,
            law,
            fidelity: fidelity.clone(),
            measured_action_ids: actions,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitPurpose {
    PriorArgmaxControl,
    MeasuredLeaf,
    BestOfNControl,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub fits_charged: u64,
    pub cpu_ns_charged: u64,
    pub failed_fits_charged: u64,
    pub visits: u64,
    pub by_purpose: BTreeMap<FitPurpose, u64>,
    pub over_budget: bool,
}

/// Counts the prior-argmax control, every measured leaf, best-of-N evaluations and failed fits.
/// Visits are a separate counter and never substitute for measured fits.
#[derive(Clone, Debug)]
pub struct MeasuredFitBudget {
    max_fits: u64,
    max_cpu_ns: u64,
    snapshot: BudgetSnapshot,
}

impl MeasuredFitBudget {
    pub fn new(max_fits: u64, max_cpu_ns: u64) -> Self {
        Self {
            max_fits,
            max_cpu_ns,
            snapshot: BudgetSnapshot::default(),
        }
    }

    pub fn can_start_fit(&self, estimated_cpu_ns: u64) -> bool {
        estimated_cpu_ns > 0
            && self.snapshot.fits_charged < self.max_fits
            && self
                .snapshot
                .cpu_ns_charged
                .checked_add(estimated_cpu_ns)
                .is_some_and(|total| total <= self.max_cpu_ns)
    }

    /// Conservative slot count using a caller-supplied per-fit CPU estimate. Actual CPU is
    /// charged on completion; any underestimate remains visible as an over-budget result.
    pub fn available_fit_slots(&self, estimated_cpu_ns: u64) -> usize {
        if estimated_cpu_ns == 0 {
            return 0;
        }
        let fit_slots = self.max_fits.saturating_sub(self.snapshot.fits_charged);
        let cpu_slots =
            self.max_cpu_ns.saturating_sub(self.snapshot.cpu_ns_charged) / estimated_cpu_ns;
        fit_slots.min(cpu_slots).min(usize::MAX as u64) as usize
    }

    /// Charges a fit after completion. Failed attempts and actual CPU overrun remain recorded.
    pub fn charge_fit(
        &mut self,
        purpose: FitPurpose,
        cpu_ns: u64,
        succeeded: bool,
    ) -> Result<(), RootSearchError> {
        let fits_charged = self
            .snapshot
            .fits_charged
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        let cpu_ns_charged = self
            .snapshot
            .cpu_ns_charged
            .checked_add(cpu_ns)
            .ok_or(RootSearchError::CounterOverflow)?;
        let failed_fits_charged = if succeeded {
            self.snapshot.failed_fits_charged
        } else {
            self.snapshot
                .failed_fits_charged
                .checked_add(1)
                .ok_or(RootSearchError::CounterOverflow)?
        };
        let purpose_count = self
            .snapshot
            .by_purpose
            .get(&purpose)
            .copied()
            .unwrap_or_default()
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        self.snapshot.fits_charged = fits_charged;
        self.snapshot.cpu_ns_charged = cpu_ns_charged;
        self.snapshot.failed_fits_charged = failed_fits_charged;
        self.snapshot.by_purpose.insert(purpose, purpose_count);
        self.snapshot.over_budget =
            fits_charged > self.max_fits || cpu_ns_charged > self.max_cpu_ns;
        Ok(())
    }

    pub fn record_visit(&mut self) -> Result<(), RootSearchError> {
        self.snapshot.visits = self
            .snapshot
            .visits
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        Ok(())
    }

    pub fn snapshot(&self) -> &BudgetSnapshot {
        &self.snapshot
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TeacherTargetError {
    UnsupportedSamplingLaw,
    IncompleteSearch,
    MissingMeasuredTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherTarget {
    pub profile: &'static str,
    pub law: SamplingLaw,
    pub fidelity: Fidelity,
    pub measured_action_ids: Vec<String>,
}
