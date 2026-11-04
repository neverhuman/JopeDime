use super::*;

pub(super) fn checkpoint_blobs(state: &Checkpoint) -> Result<Vec<BlobRef>> {
    let roots: Vec<_> = state
        .input_roots
        .iter()
        .cloned()
        .chain(state.champion.iter().map(|c| c.model.clone()))
        .collect();
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        let mut unique = BTreeMap::new();
        for reference in roots.into_iter().chain(capture.roots()) {
            if let Some(old) = unique.insert(reference.sha256, reference.clone()) {
                ensure!(old == reference, "conflicting CAS root metadata");
            }
        }
        return Ok(unique.into_values().collect());
    }
    Ok(roots)
}

pub(super) fn rung0_budget_used(state: &Checkpoint) -> u64 {
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        return capture.charged;
    }
    state.admitted
}

pub(super) fn rung0_budget_limit(state: &Checkpoint) -> u64 {
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        return capture.config.max_attempts;
    }
    state.plan.evaluations
}

pub(super) fn hex_blob(blob: &BlobRef) -> String {
    blob.sha256.iter().map(|b| format!("{b:02x}")).collect()
}
pub(super) fn projector(plan: &Plan) -> Result<project::Projector> {
    let mut bytes = [0_u8; 32];
    for (i, value) in bytes.iter_mut().enumerate() {
        *value = u8::from_str_radix(&plan.problem_digest[i * 2..i * 2 + 2], 16)?;
    }
    Ok(project::Projector::new(
        plan.run_id.clone(),
        project::ProblemDigest(bytes),
    ))
}

pub(super) fn public_metric(plan: &Plan) -> Result<String> {
    let name = jope_runtime::fitness::Fitness::parse(&plan.objective)?.name();
    Ok(
        if jope_runtime::fitness::Fitness::catalog()
            .iter()
            .any(|entry| entry.name == name)
        {
            name
        } else {
            "custom objective".into()
        },
    )
}

pub(super) fn experiment(plan: &Plan, round: u64, slot: usize, rung: u8) -> Result<String> {
    ensure!(
        slot < 65536 && rung < 4,
        "evaluation ordinal is out of range"
    );
    let ordinal = evaluation_ordinal(round, u32::from(rung) * 65536 + slot as u32)?;
    Ok(projector(plan)?.experiment_id(project::EvaluationOrdinal(ordinal)))
}

pub(super) fn evaluation_ordinal(round: u64, encoded_slot: u32) -> Result<u64> {
    ensure!(encoded_slot < 262144, "evaluation ordinal is out of range");
    round
        .checked_mul(262144)
        .and_then(|n| n.checked_add(u64::from(encoded_slot)))
        .context("evaluation ordinal exhausted")
}

pub(super) fn projected_summary(plan: &Plan, claim: project::Claim<'_>) -> Result<String> {
    let EventBody::Reasoning(body) = projector(plan)?.project(claim)? else {
        bail!("unexpected public projection body");
    };
    Ok(body.summary)
}

pub(super) fn accepted_summary(
    plan: &Plan,
    candidate: &Candidate,
    value: f64,
    previous: Option<f64>,
) -> Result<String> {
    let key = feat_math::GenomeIr::parse(&candidate.dsl)
        .ok()
        .and_then(|genome| match genome {
            feat_math::GenomeIr::Pipeline(dsl) => dsl.stages.last().map(|stage| stage.op.clone()),
            feat_math::GenomeIr::Portfolio(_) => Some("portfolio".into()),
            _ => None,
        });
    let Some(head) = key.as_deref().and_then(|key| {
        feat_math::public_names::heads::HEADS
            .iter()
            .find(|h| h.key == key)
    }) else {
        return projected_summary(plan, project::Claim::Unknown);
    };
    let context = project::ScoreContext {
        problem: project::ProblemDigest({
            let mut b = [0_u8; 32];
            for (i, v) in b.iter_mut().enumerate() {
                *v = u8::from_str_radix(&plan.problem_digest[i * 2..i * 2 + 2], 16)?;
            }
            b
        }),
        folds: project::FoldsDigest(
            Sha256::digest(serde_json::to_vec(&plan.fold_spec(1)?)?).into(),
        ),
        metric: public_metric(plan)?,
        scope: format!("cv_fit_{}fold", plan.limits().rung1_folds),
    };
    // This describes the atomic post-commit view. Neither body nor model
    // reference becomes observable unless the containing transaction commits.
    projected_summary(
        plan,
        project::Claim::Accepted {
            champion: project::Champion {
                algorithm_id: candidate.algorithm_id.parse()?,
                head: phrase::CatalogEntry {
                    public_name: head.public_name,
                    purpose: head.purpose,
                },
                committed: true,
                score: project::Score {
                    context: context.clone(),
                    value,
                },
            },
            previous: previous.map(|value| project::Score { context, value }),
        },
    )
}
pub(super) fn round_rng(seed: u64, round: u64) -> ChaCha12Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    ChaCha12Rng::from_seed(hash.finalize().into())
}
pub(crate) fn evaluation_seed(seed: u64, round: u64, slot: u32, rung: u8) -> u64 {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    hash.update(slot.to_le_bytes());
    hash.update([rung]);
    u64::from_le_bytes(
        hash.finalize()[..8]
            .try_into()
            .expect("SHA256 contains eight bytes"),
    )
}
