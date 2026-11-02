use super::*;

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
pub(in crate::ratchet) fn push_generation_candidate(
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
pub(in crate::ratchet) fn push_unique_identity(
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
pub(in crate::ratchet) fn derived_seed(base: u64, lane: &[u8], attempt: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"jope-dime/generation-zero-offspring/v3\0");
    digest.update(base.to_le_bytes());
    digest.update(lane);
    digest.update(attempt.to_le_bytes());
    let digest = digest.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"))
}
