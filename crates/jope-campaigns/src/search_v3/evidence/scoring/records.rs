use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRecordV2 {
    pub schema: String,
    pub candidate_slot: u32,
    pub fold_index: u32,
    pub prediction: PredictionObjectV1,
    pub scoring_tape: ScoringTapeV2,
    pub score_key: ScoreKeyV2,
    pub fitness: FitnessVectorV1,
}

impl EvaluationRecordV2 {
    pub const SCHEMA: &'static str = "SearchEvaluationRecord.v2";

    pub fn validate_structural(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        self.prediction.validate()?;
        self.scoring_tape.validate()?;
        self.score_key.validate()?;
        self.fitness.validate()?;
        if self.fold_index != self.prediction.fold_index
            || self.fitness.candidate_semantic_sha256 != self.prediction.candidate_semantic_sha256
            || self.scoring_tape.prediction_object_sha256 != self.prediction.identity()?
            || self.score_key.scoring_tape_sha256 != self.scoring_tape.identity()?
        {
            return Err("evaluation record identities or fold binding disagree".into());
        }
        Ok(())
    }

    pub fn verify_score(
        &self,
        registered: &RegisteredFitnessV1,
        complexity: u64,
    ) -> Result<(), String> {
        self.validate_structural()?;
        let (key, fitness) =
            score_prediction(&self.prediction, &self.scoring_tape, registered, complexity)?;
        if key != self.score_key || fitness != self.fitness {
            return Err(
                "evaluation score does not recompute from prediction and scoring tape".into(),
            );
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate_structural()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidenceV1 {
    pub candidate: SearchCandidateIdentityV1,
    pub evaluations: BTreeMap<u32, EvaluationRecordV2>,
}

impl CandidateEvidenceV1 {
    pub fn validate_structural(&self) -> Result<(), String> {
        self.candidate.validate()?;
        for (fold, evaluation) in &self.evaluations {
            evaluation.validate_structural()?;
            if *fold != evaluation.fold_index
                || evaluation.candidate_slot != self.candidate.slot
                || evaluation.prediction.candidate_semantic_sha256 != self.candidate.semantic_sha256
            {
                return Err(
                    "candidate evidence has a mismatched slot, fold, or semantic identity".into(),
                );
            }
        }
        Ok(())
    }

    pub fn evidence_bitmap(&self) -> BTreeSet<u32> {
        self.evaluations.keys().copied().collect()
    }
}
