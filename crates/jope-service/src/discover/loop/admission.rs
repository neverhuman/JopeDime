use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub algorithm_id: String,
    pub dsl: String,
}

/// Context-free policies remain supported through the `RoundAdmission` bridge.
pub trait Admission {
    fn admit(&self, proposals: Vec<String>, explore: bool, rng: &mut ChaCha12Rng) -> Vec<String>;

    #[cfg(feature = "research")]
    fn admit_recorded(
        &self,
        _proposals: Vec<String>,
        _explore: bool,
        _rng: &mut ChaCha12Rng,
        _observed: &ObservedSlate,
    ) -> Result<BehaviorPolicy> {
        bail!("same-invocation admission observations are unavailable")
    }
}

/// Committed rung-0 context. User-objective values are deliberately absent;
/// objective ordering still determines elite membership and search parents.
pub struct RoundMeasurement<'a> {
    pub algorithm_id: &'a str,
    pub dsl: &'a str,
    pub surrogate_fitness: f64,
}

pub struct RoundContext<'a> {
    pub plan: &'a Plan,
    pub round: u64,
    pub measured: Vec<RoundMeasurement<'a>>,
}

/// The immutable capture observation and conditioning that will be persisted
/// with this same invocation. Learned policies must verify both against their
/// actual model inputs before recording costs or consuming selection randomness.
#[cfg(feature = "research")]
pub struct RecordedAdmissionContext<'a> {
    pub slate: &'a ObservedSlate,
    pub conditioning: crate::ranker::stage1b::TokenProjection,
}

/// Fallible admission against the same committed context for every sibling in a
/// wave. Failure aborts planning before any scheduler checkpoint is committed.
pub trait RoundAdmission {
    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>>;

    #[cfg(feature = "research")]
    fn admit_recorded_round(
        &self,
        _context: &RoundContext<'_>,
        _proposals: Vec<String>,
        _explore: bool,
        _rng: &mut ChaCha12Rng,
        _recorded: &RecordedAdmissionContext<'_>,
    ) -> Result<BehaviorPolicy> {
        bail!("same-invocation admission observations are unavailable")
    }
}

impl<T: Admission> RoundAdmission for T {
    fn admit_round(
        &self,
        _context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        Ok(self.admit(proposals, explore, rng))
    }

    #[cfg(feature = "research")]
    fn admit_recorded_round(
        &self,
        _context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
        recorded: &RecordedAdmissionContext<'_>,
    ) -> Result<BehaviorPolicy> {
        self.admit_recorded(proposals, explore, rng, recorded.slate)
    }
}

pub struct HeuristicAdmission;
impl Admission for HeuristicAdmission {
    #[cfg(feature = "research")]
    fn admit_recorded(
        &self,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
        _observed: &ObservedSlate,
    ) -> Result<BehaviorPolicy> {
        let admitted = self.admit(proposals.clone(), explore, rng);
        let indices = admitted
            .iter()
            .map(|dsl| {
                proposals
                    .iter()
                    .position(|p| p == dsl)
                    .context("admission returned an unknown proposal")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BehaviorPolicy {
            kind: RecordedPolicyKind::Heuristic,
            explore,
            slate_costs: None,
            admitted: indices,
        })
    }

    fn admit(
        &self,
        mut proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Vec<String> {
        if !explore || proposals.len() <= 4 {
            proposals.truncate(4);
            return proposals;
        }
        let mut rest = proposals.split_off(2);
        for _ in 0..2 {
            let index = sample_index(rng, rest.len());
            proposals.push(rest.remove(index));
        }
        proposals
    }
}

fn sample_index(rng: &mut ChaCha12Rng, len: usize) -> usize {
    let limit = u64::MAX - u64::MAX % len as u64;
    loop {
        let value = rng.next_u64();
        if value < limit {
            return (value % len as u64) as usize;
        }
    }
}
