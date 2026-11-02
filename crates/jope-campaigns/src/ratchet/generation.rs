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

    pub(super) fn validate_roles(&self, semantics: &BTreeSet<&str>) -> Result<()> {
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

/// Deterministically builds the physical generation from already validated
/// source pools. The fixed concatenation order is part of the Rust authority.
// Keep each independently validated generation pool and identity binding explicit.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn assemble_generation_zero_v3(
    learning_wave_sha256: String,
    dataset_state_sha256: String,
    dataset_sha256: String,
    seed_request_sha256: String,
    seed_response_sha256: String,
    model_retention_receipt_sha256: String,
    assembly_seed: u64,
    archive: Vec<CandidateIdentityV1>,
    model: Vec<RetainedModelProposalV2>,
    gpr: Vec<(CandidateIdentityV1, GprKernelFamilyV3)>,
    mutation: Vec<(CandidateIdentityV1, u64, String)>,
    crossover: Vec<(CandidateIdentityV1, u64, String, String)>,
    random: Vec<(CandidateIdentityV1, u64)>,
    role_bindings: Vec<GenerationRoleBindingV3>,
) -> Result<Generation0ReceiptV3> {
    if archive.len() != 32
        || model.len() != 64
        || gpr.len() != 8
        || mutation.len() != 60
        || crossover.len() != 60
        || random.len() != 32
    {
        bail!("generation source pool exhausted before its exact physical quota");
    }
    let mut candidates = Vec::with_capacity(GENERATION_ZERO_SIZE);
    for (priority, candidate) in archive.into_iter().enumerate() {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Archive { priority },
        );
    }
    for retained in model {
        push_generation_candidate(
            &mut candidates,
            retained.proposal.candidate,
            CandidateProvenanceV3::Model {
                bucket: retained.bucket,
                bucket_rank: retained.bucket_rank,
            },
        );
    }
    for (candidate, kernel_family) in gpr {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Gpr { kernel_family },
        );
    }
    for (candidate, seed, parent_semantic_sha256) in mutation {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Mutation {
                seed,
                parent_semantic_sha256,
            },
        );
    }
    for (candidate, seed, left_parent_semantic_sha256, right_parent_semantic_sha256) in crossover {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Crossover {
                seed,
                left_parent_semantic_sha256,
                right_parent_semantic_sha256,
            },
        );
    }
    for (candidate, seed) in random {
        push_generation_candidate(
            &mut candidates,
            candidate,
            CandidateProvenanceV3::Random { seed },
        );
    }
    let receipt = Generation0ReceiptV3 {
        schema: GENERATION_ZERO_V3_SCHEMA.into(),
        learning_wave_sha256,
        dataset_state_sha256,
        dataset_sha256,
        seed_request_sha256,
        seed_response_sha256,
        model_retention_receipt_sha256,
        assembly_seed,
        candidates,
        role_bindings,
    };
    receipt.validate()?;
    Ok(receipt)
}

#[cfg(test)]
pub(super) fn push_generation_candidate(
    candidates: &mut Vec<GenerationCandidateV3>,
    candidate: CandidateIdentityV1,
    provenance: CandidateProvenanceV3,
) {
    candidates.push(GenerationCandidateV3 {
        position: candidates.len(),
        candidate,
        provenance,
    });
}

#[cfg(test)]
pub(crate) fn deterministic_archive_32(
    state: &DatasetLearningStateV2,
    supplemental_priority: &[CandidateIdentityV1],
) -> Result<Vec<CandidateIdentityV1>> {
    state.validate()?;
    let mut result = Vec::with_capacity(32);
    let mut seen = BTreeSet::new();
    push_unique_identity(
        &mut result,
        &mut seen,
        state
            .confirmed_cv_champion
            .as_ref()
            .context("archive requires confirmed-CV champion")?,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        state
            .latest_exact_track_1_frontier
            .as_ref()
            .context("archive requires exact Track-1 frontier")?,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .latest_corrector_prediction
            .as_ref()
            .context("archive requires latest corrector prediction")?
            .candidate,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .track_1_history
            .last()
            .context("archive requires latest Track-1 winner")?
            .candidate,
    )?;
    push_unique_identity(
        &mut result,
        &mut seen,
        &state
            .track_2_history
            .last()
            .context("archive requires latest Track-2 winner")?
            .candidate,
    )?;
    if let Some(prediction) = &state.latest_model_prediction {
        push_unique_identity(&mut result, &mut seen, &prediction.candidate)?;
    }
    let mut frontier = state.global_cv_frontier.iter().collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        right
            .robust_fitness
            .total_cmp(&left.robust_fitness)
            .then_with(|| {
                left.candidate
                    .semantic_sha256
                    .cmp(&right.candidate.semantic_sha256)
            })
    });
    for member in frontier {
        push_unique_identity(&mut result, &mut seen, &member.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for entry in state.track_1_history.iter().rev() {
        push_unique_identity(&mut result, &mut seen, &entry.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for entry in state.track_2_history.iter().rev() {
        push_unique_identity(&mut result, &mut seen, &entry.candidate)?;
        if result.len() == 32 {
            return Ok(result);
        }
    }
    for candidate in supplemental_priority {
        push_unique_identity(&mut result, &mut seen, candidate)?;
        if result.len() == 32 {
            break;
        }
    }
    if result.len() != 32 {
        bail!("deterministic archive subpool exhausted before 32 unique candidates");
    }
    Ok(result)
}

#[cfg(test)]
pub(super) fn push_unique_identity(
    result: &mut Vec<CandidateIdentityV1>,
    seen: &mut BTreeSet<String>,
    candidate: &CandidateIdentityV1,
) -> Result<()> {
    candidate.validate()?;
    if result.len() < 32 && seen.insert(candidate.semantic_sha256.clone()) {
        result.push(candidate.clone());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) struct OffspringPoolsV3 {
    pub(crate) mutation: Vec<(CandidateIdentityV1, u64, String)>,
    pub(crate) crossover: Vec<(CandidateIdentityV1, u64, String, String)>,
    pub(crate) random: Vec<(CandidateIdentityV1, u64)>,
}

/// Existing canonical feat-math operators are used, but Rust owns parent
/// order, seeds, semantic deduplication, quotas, and bounded failure.
#[cfg(test)]
pub(crate) fn build_bounded_offspring_pools(
    parents: &[CandidateIdentityV1],
    occupied_semantics: &BTreeSet<String>,
    assembly_seed: u64,
    max_transforms: usize,
    max_attempts_per_pool: usize,
) -> Result<OffspringPoolsV3> {
    if parents.len() < 2 || max_attempts_per_pool == 0 {
        bail!("offspring construction requires two parents and a positive attempt bound");
    }
    for parent in parents {
        parent.validate()?;
    }
    let mut seen = occupied_semantics.clone();
    for parent in parents {
        seen.insert(parent.semantic_sha256.clone());
    }

    let mut mutation = Vec::with_capacity(60);
    for attempt in 0..max_attempts_per_pool {
        if mutation.len() == 60 {
            break;
        }
        let parent = &parents[attempt % parents.len()];
        let seed = derived_seed(assembly_seed, b"mutation", attempt as u64);
        let Ok(canonical) =
            mutate_canonical_pipeline(&parent.canonical_genome, seed, max_transforms)
        else {
            continue;
        };
        let candidate = CandidateIdentityV1::from_canonical(canonical)?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            mutation.push((candidate, seed, parent.semantic_sha256.clone()));
        }
    }
    if mutation.len() != 60 {
        bail!("mutation subpool exhausted its bounded refill");
    }

    let mut crossover = Vec::with_capacity(60);
    for attempt in 0..max_attempts_per_pool {
        if crossover.len() == 60 {
            break;
        }
        let left_index = attempt % parents.len();
        let mut right_index =
            (attempt + 1 + (assembly_seed as usize % (parents.len() - 1))) % parents.len();
        if right_index == left_index {
            right_index = (right_index + 1) % parents.len();
        }
        let left = &parents[left_index];
        let right = &parents[right_index];
        let seed = derived_seed(assembly_seed, b"crossover", attempt as u64);
        let Ok(canonical) = crossover_canonical_pipelines(
            &left.canonical_genome,
            &right.canonical_genome,
            seed,
            max_transforms,
        ) else {
            continue;
        };
        let candidate = CandidateIdentityV1::from_canonical(canonical)?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            crossover.push((
                candidate,
                seed,
                left.semantic_sha256.clone(),
                right.semantic_sha256.clone(),
            ));
        }
    }
    if crossover.len() != 60 {
        bail!("crossover subpool exhausted its bounded refill");
    }

    let mut random = Vec::with_capacity(32);
    for attempt in 0..max_attempts_per_pool {
        if random.len() == 32 {
            break;
        }
        let seed = derived_seed(assembly_seed, b"random", attempt as u64);
        let candidate =
            CandidateIdentityV1::from_canonical(random_canonical_pipeline(seed, max_transforms))?;
        if seen.insert(candidate.semantic_sha256.clone()) {
            random.push((candidate, seed));
        }
    }
    if random.len() != 32 {
        bail!("random subpool exhausted its bounded refill");
    }
    Ok(OffspringPoolsV3 {
        mutation,
        crossover,
        random,
    })
}

#[cfg(test)]
pub(super) fn derived_seed(base: u64, lane: &[u8], attempt: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"jope-dime/generation-zero-offspring/v3\0");
    digest.update(base.to_le_bytes());
    digest.update(lane);
    digest.update(attempt.to_le_bytes());
    let digest = digest.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"))
}
