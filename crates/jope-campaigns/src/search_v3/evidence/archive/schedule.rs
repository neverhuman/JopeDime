use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionScheduleV1 {
    pub schema: String,
    pub fold_count: u32,
    pub fold_one_slots: Vec<u32>,
    pub promoted_thirty_slots: Vec<u32>,
    pub fold_two_slots: Vec<u32>,
    pub finalist_eight_slots: Vec<u32>,
    pub complete_fold_slots: BTreeMap<u32, Vec<u32>>,
    pub fold_evaluation_count: u64,
}

impl PromotionScheduleV1 {
    pub const SCHEMA: &'static str = "SearchPromotionSchedule.v1";

    pub fn build(
        plan: &SearchCandidatePlanV1,
        evidence: &[CandidateEvidenceV1],
        fold_count: u32,
    ) -> Result<Self, String> {
        plan.validate()?;
        if fold_count < 3 || evidence.len() != 60 {
            return Err(
                "promotion schedule requires 60 candidates and at least three folds".into(),
            );
        }
        for (slot, candidate) in evidence.iter().enumerate() {
            candidate.validate_structural()?;
            if candidate.candidate != plan.candidates[slot] {
                return Err("candidate evidence must be in exact plan slot order".into());
            }
        }
        let all: Vec<_> = evidence.iter().collect();
        let fold_one_slots = rank_on_common_evidence(&all, &[0])?;
        let promoted_thirty_slots = fold_one_slots[..30].to_vec();
        let promoted: Vec<_> = promoted_thirty_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let fold_two_slots = rank_on_common_evidence(&promoted, &[0, 1])?;
        let finalist_eight_slots = fold_two_slots[..8].to_vec();
        let finalists: Vec<_> = finalist_eight_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let complete_folds: Vec<_> = (0..fold_count).collect();
        let final_rank = rank_on_common_evidence(&finalists, &complete_folds)?;

        let promoted_set: BTreeSet<_> = promoted_thirty_slots.iter().copied().collect();
        let finalist_set: BTreeSet<_> = finalist_eight_slots.iter().copied().collect();
        for candidate in evidence {
            let expected: BTreeSet<u32> = if finalist_set.contains(&candidate.candidate.slot) {
                (0..fold_count).collect()
            } else if promoted_set.contains(&candidate.candidate.slot) {
                [0, 1].into_iter().collect()
            } else {
                [0].into_iter().collect()
            };
            if candidate.evidence_bitmap() != expected {
                return Err(format!(
                    "candidate {} has evidence {:?}, expected {:?}",
                    candidate.candidate.slot,
                    candidate.evidence_bitmap(),
                    expected
                ));
            }
        }
        let complete_fold_slots = (0..fold_count)
            .map(|fold| {
                let slots = if fold < 2 {
                    if fold == 0 {
                        (0..60).collect()
                    } else {
                        promoted_thirty_slots.clone()
                    }
                } else {
                    finalist_eight_slots.clone()
                };
                (fold, slots)
            })
            .collect();
        Ok(Self {
            schema: Self::SCHEMA.into(),
            fold_count,
            fold_one_slots,
            promoted_thirty_slots,
            fold_two_slots,
            finalist_eight_slots: final_rank,
            complete_fold_slots,
            fold_evaluation_count: 60 + 30 + u64::from(fold_count - 2) * 8,
        })
    }
}
