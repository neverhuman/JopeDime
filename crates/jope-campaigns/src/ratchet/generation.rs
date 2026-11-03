use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GprKernelFamilyV3 {
    Rbf,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Periodic,
    Linear,
    WhiteComposite,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "source", rename_all = "snake_case")]
pub(crate) enum CandidateProvenanceV3 {
    Archive {
        priority: usize,
    },
    Model {
        bucket: ModelRetentionBucketV2,
        bucket_rank: usize,
    },
    Gpr {
        kernel_family: GprKernelFamilyV3,
    },
    Mutation {
        seed: u64,
        parent_semantic_sha256: String,
    },
    Crossover {
        seed: u64,
        left_parent_semantic_sha256: String,
        right_parent_semantic_sha256: String,
    },
    Random {
        seed: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct GenerationCandidateV3 {
    pub(crate) position: usize,
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) provenance: CandidateProvenanceV3,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationRoleV3 {
    ConfirmedCvChampion,
    LatestExactTrack1Frontier,
    CheckpointRewardTop1,
    LatestModelPrediction,
    LatestCorrectorPrediction,
    LatestTrack1Winner,
    LatestTrack2Winner,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct GenerationRoleBindingV3 {
    pub(crate) role: GenerationRoleV3,
    pub(crate) semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Generation0ReceiptV3 {
    pub(crate) schema: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_state_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) seed_request_sha256: String,
    pub(crate) seed_response_sha256: String,
    pub(crate) model_retention_receipt_sha256: String,
    pub(crate) assembly_seed: u64,
    pub(crate) candidates: Vec<GenerationCandidateV3>,
    pub(crate) role_bindings: Vec<GenerationRoleBindingV3>,
}

impl Generation0ReceiptV3 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != GENERATION_ZERO_V3_SCHEMA || self.candidates.len() != GENERATION_ZERO_SIZE
        {
            bail!("generation zero is not the promoted v3 population of 256");
        }
        validate_hashes([
            (
                "generation learning wave",
                self.learning_wave_sha256.as_str(),
            ),
            (
                "generation dataset state",
                self.dataset_state_sha256.as_str(),
            ),
            ("generation dataset", self.dataset_sha256.as_str()),
            ("generation seed request", self.seed_request_sha256.as_str()),
            (
                "generation seed response",
                self.seed_response_sha256.as_str(),
            ),
            (
                "generation model retention receipt",
                self.model_retention_receipt_sha256.as_str(),
            ),
        ])?;

        let mut semantics = BTreeSet::new();
        let mut archive = 0;
        let mut model = 0;
        let mut gpr = 0;
        let mut mutation = 0;
        let mut crossover = 0;
        let mut random = 0;
        let mut archive_priorities = BTreeSet::new();
        let mut model_buckets = BTreeMap::<ModelRetentionBucketV2, usize>::new();
        let mut gpr_families = BTreeSet::new();
        let mut mutation_seeds = BTreeSet::new();
        let mut crossover_seeds = BTreeSet::new();
        let mut random_seeds = BTreeSet::new();
        for (position, member) in self.candidates.iter().enumerate() {
            if member.position != position {
                bail!("generation zero physical positions are not contiguous");
            }
            member.candidate.validate()?;
            if !semantics.insert(member.candidate.semantic_sha256.as_str()) {
                bail!("generation zero contains a semantic duplicate");
            }
            match &member.provenance {
                CandidateProvenanceV3::Archive { priority } => {
                    archive += 1;
                    if *priority >= 32 || !archive_priorities.insert(*priority) {
                        bail!("archive priority is out of range or duplicated");
                    }
                }
                CandidateProvenanceV3::Model {
                    bucket,
                    bucket_rank,
                } => {
                    model += 1;
                    let expected = match bucket {
                        ModelRetentionBucketV2::Reward => 32,
                        ModelRetentionBucketV2::Disagreement => 16,
                        ModelRetentionBucketV2::Novelty => 8,
                        ModelRetentionBucketV2::SeededTemperature => 8,
                    };
                    if *bucket_rank >= expected {
                        bail!("model retention bucket rank exceeds its quota");
                    }
                    *model_buckets.entry(bucket.clone()).or_default() += 1;
                }
                CandidateProvenanceV3::Gpr { kernel_family } => {
                    gpr += 1;
                    if !gpr_families.insert(kernel_family.clone()) {
                        bail!("GPR kernel family is duplicated");
                    }
                }
                CandidateProvenanceV3::Mutation {
                    seed,
                    parent_semantic_sha256,
                } => {
                    mutation += 1;
                    validate_sha256("mutation parent", parent_semantic_sha256)?;
                    if !mutation_seeds.insert(*seed) {
                        bail!("mutation seed is duplicated");
                    }
                }
                CandidateProvenanceV3::Crossover {
                    seed,
                    left_parent_semantic_sha256,
                    right_parent_semantic_sha256,
                } => {
                    crossover += 1;
                    validate_hashes([
                        (
                            "crossover left parent",
                            left_parent_semantic_sha256.as_str(),
                        ),
                        (
                            "crossover right parent",
                            right_parent_semantic_sha256.as_str(),
                        ),
                    ])?;
                    if left_parent_semantic_sha256 == right_parent_semantic_sha256
                        || !crossover_seeds.insert(*seed)
                    {
                        bail!("crossover parents or seed are invalid");
                    }
                }
                CandidateProvenanceV3::Random { seed } => {
                    random += 1;
                    if !random_seeds.insert(*seed) {
                        bail!("random seed is duplicated");
                    }
                }
            }
        }
        if (archive, model, gpr, mutation, crossover, random) != (32, 64, 8, 60, 60, 32) {
            bail!("generation zero physical source topology drifted from 32/64/8/60/60/32");
        }
        for (bucket, count) in [
            (ModelRetentionBucketV2::Reward, 32),
            (ModelRetentionBucketV2::Disagreement, 16),
            (ModelRetentionBucketV2::Novelty, 8),
            (ModelRetentionBucketV2::SeededTemperature, 8),
        ] {
            if model_buckets.get(&bucket).copied() != Some(count) {
                bail!("generation zero model source quotas drifted");
            }
        }
        if gpr_families.len() != 8 {
            bail!("generation zero does not contain eight distinct GPR families");
        }

        self.validate_roles(&semantics)
    }

    pub(crate) fn validate_roles(&self, semantics: &BTreeSet<&str>) -> Result<()> {
        require_unique(
            "generation role bindings",
            self.role_bindings
                .iter()
                .map(|binding| binding.role.clone()),
        )?;
        for binding in &self.role_bindings {
            validate_sha256("generation role candidate", &binding.semantic_sha256)?;
            if !semantics.contains(binding.semantic_sha256.as_str()) {
                bail!("generation role refers to a candidate outside the physical population");
            }
        }
        let roles = self
            .role_bindings
            .iter()
            .map(|binding| (binding.role.clone(), binding.semantic_sha256.as_str()))
            .collect::<BTreeMap<_, _>>();
        for required in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
            GenerationRoleV3::CheckpointRewardTop1,
            GenerationRoleV3::LatestModelPrediction,
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            if !roles.contains_key(&required) {
                bail!("generation zero is missing a mandatory role");
            }
        }
        for anchor_role in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
        ] {
            let semantic = roles[&anchor_role];
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(member.provenance, CandidateProvenanceV3::Archive { .. })
            }) {
                bail!("known-best anchor is not physically retained by the archive");
            }
        }
        for archive_role in [
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            let semantic = roles[&archive_role];
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(member.provenance, CandidateProvenanceV3::Archive { .. })
            }) {
                bail!("mandatory archive role is not physically retained by the archive");
            }
        }
        let checkpoint_top_1 = roles[&GenerationRoleV3::CheckpointRewardTop1];
        if !self.candidates.iter().any(|member| {
            member.candidate.semantic_sha256 == checkpoint_top_1
                && matches!(
                    member.provenance,
                    CandidateProvenanceV3::Model {
                        bucket: ModelRetentionBucketV2::Reward,
                        bucket_rank: 0
                    }
                )
        }) {
            bail!("checkpoint reward top-1 is absent from the model quota");
        }
        Ok(())
    }

    pub(crate) fn receipt_sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }

    pub(crate) fn validate_against_state(&self, state: &DatasetLearningStateV2) -> Result<()> {
        self.validate()?;
        state.validate()?;
        if self.dataset_sha256 != state.dataset_sha256
            || self.dataset_state_sha256 != state.snapshot_sha256()?
        {
            bail!("generation zero does not bind the exact dataset-state snapshot");
        }
        let roles = self
            .role_bindings
            .iter()
            .map(|binding| (binding.role.clone(), binding.semantic_sha256.as_str()))
            .collect::<BTreeMap<_, _>>();
        let confirmed = state
            .confirmed_cv_champion
            .as_ref()
            .context("dataset has no confirmed-CV anchor; confirmation work is required")?;
        let exact = state
            .latest_exact_track_1_frontier
            .as_ref()
            .context("dataset has no exact Track-1 anchor; confirmation work is required")?;
        if roles[&GenerationRoleV3::ConfirmedCvChampion] != confirmed.semantic_sha256
            || roles[&GenerationRoleV3::LatestExactTrack1Frontier] != exact.semantic_sha256
        {
            bail!("generation zero known-best roles are stale or test-selected");
        }
        let latest_track_1 = state
            .track_1_history
            .last()
            .context("dataset has no Track-1 history")?;
        let latest_track_2 = state
            .track_2_history
            .last()
            .context("dataset has no Track-2 history")?;
        if roles[&GenerationRoleV3::LatestTrack1Winner] != latest_track_1.candidate.semantic_sha256
            || roles[&GenerationRoleV3::LatestTrack2Winner]
                != latest_track_2.candidate.semantic_sha256
        {
            bail!("generation zero latest-track archive roles are stale");
        }
        let latest_model = state
            .latest_model_prediction
            .as_ref()
            .context("dataset has no latest model prediction")?;
        let latest_corrector = state
            .latest_corrector_prediction
            .as_ref()
            .context("dataset has no latest corrector prediction")?;
        if roles[&GenerationRoleV3::LatestModelPrediction] != latest_model.candidate.semantic_sha256
            || roles[&GenerationRoleV3::LatestCorrectorPrediction]
                != latest_corrector.candidate.semantic_sha256
        {
            bail!("generation zero prediction roles are stale");
        }
        let mut required_archive = Vec::new();
        for role in [
            GenerationRoleV3::ConfirmedCvChampion,
            GenerationRoleV3::LatestExactTrack1Frontier,
            GenerationRoleV3::LatestCorrectorPrediction,
            GenerationRoleV3::LatestTrack1Winner,
            GenerationRoleV3::LatestTrack2Winner,
        ] {
            let semantic = roles[&role];
            if !required_archive.contains(&semantic) {
                required_archive.push(semantic);
            }
        }
        for (expected_priority, semantic) in required_archive.into_iter().enumerate() {
            if !self.candidates.iter().any(|member| {
                member.candidate.semantic_sha256 == semantic
                    && matches!(
                        member.provenance,
                        CandidateProvenanceV3::Archive { priority }
                            if priority == expected_priority
                    )
            }) {
                bail!("mandatory archive roles violate deterministic priority ordering");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod pools;

#[cfg(test)]
pub(crate) use pools::{
    assemble_generation_zero_v3, build_bounded_offspring_pools, deterministic_archive_32,
};
