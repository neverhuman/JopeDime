use super::*;

pub(super) fn decode_v2_measurement(
    plan: &Plan,
    request: &Request,
    response: &Response,
) -> Result<Rung0Measurement> {
    ensure!(
        request.attempt == response.attempt,
        "worker Attempt identity mismatch"
    );
    ensure!(
        request.attempt.run == plan.run_id,
        "request run differs from bound Plan"
    );
    ensure!(
        request.attempt.round <= plan.evaluations,
        "request round exceeds bound Plan"
    );
    ensure!(
        request.matrix_path == plan.matrix_path,
        "request FIT path differs from bound Plan"
    );
    ensure!(
        request.objective.as_str() == Some(plan.objective.as_str()),
        "request objective differs from bound Plan"
    );
    ensure!(
        request.limits.wall_ms == plan.evaluation_wall_ms
            && request.limits.rss_bytes == plan.evaluation_rss_bytes,
        "request limits differ from bound Plan"
    );
    ensure!(
        request.seed
            == crate::discover::r#loop::evaluation_seed(
                plan.seed,
                request.attempt.round,
                request.attempt.slot,
                0,
            ),
        "request seed differs from frozen evaluation protocol"
    );
    let spec: V2FoldSpec = serde_json::from_value(request.fold_spec.clone())
        .context("invalid frozen r782 v2 fold specification")?;
    validate_v2_request(plan, &spec)?;
    if let Some(code) = response.failure {
        ensure!(
            response.scores.is_empty()
                && response.fold_scores.is_empty()
                && response.predictions_digest.is_empty(),
            "failed response carries scores"
        );
        return Ok(Rung0Measurement::Failed { code });
    }
    ensure!(
        response.scores.len() == 3,
        "r782 evaluator must return three values"
    );
    ensure!(
        response.scores.iter().all(|score| score.is_finite()),
        "nonfinite r782 score"
    );
    ensure!(
        response.fold_scores.len() == 2,
        "r782 evaluator must return two fold losses"
    );
    ensure!(
        response
            .fold_scores
            .iter()
            .all(|score| score.is_finite() && *score >= 0.0),
        "invalid r782 fold loss"
    );
    let objective = 0.5 * (response.fold_scores[0] + response.fold_scores[1]);
    ensure!(
        response.scores[1].to_bits() == objective.to_bits(),
        "objective differs from exact fold aggregate"
    );
    ensure!(
        response.scores[2].to_bits() == objective.to_bits(),
        "worker output differs from frozen v2 contract"
    );
    ensure!(
        hex(&response.predictions_digest, 64),
        "prediction digest missing"
    );
    let surrogate = response.scores[0];
    // The exact evaluator source enforces (F < 100) == gate; F is score 0.
    // Score 2 is user RMSE and never participates in gate decoding.
    Ok(Rung0Measurement::Success {
        surrogate,
        objective,
        gate: surrogate < 100.0,
    })
}

pub(super) fn validate_v2_request(plan: &Plan, spec: &V2FoldSpec) -> Result<()> {
    ensure!(spec.version == 2, "qualified evaluator requires frozen v2");
    ensure!(
        spec.matrix_sha256 == plan.matrix_sha256,
        "fold FIT identity differs from Plan"
    );
    ensure!(
        spec.target == plan.target && spec.features == plan.features,
        "fold columns differ from Plan"
    );
    ensure!(
        plan.objective == "rmse" && spec.folds.len() == 2,
        "unsupported r782 evaluator request"
    );
    let mut validation = BTreeSet::new();
    let mut universe: Option<BTreeSet<usize>> = None;
    let mut fold_sets = Vec::new();
    for fold in &spec.folds {
        ensure!(fold.repeat == 0, "r782 v2 supports one FIT fold repeat");
        let train: BTreeSet<_> = fold.train.iter().copied().collect();
        let valid: BTreeSet<_> = fold.validation.iter().copied().collect();
        ensure!(
            train.len() == fold.train.len() && valid.len() == fold.validation.len(),
            "duplicate fold row"
        );
        ensure!(
            train.len() >= 2 && valid.len() >= 2 && train.is_disjoint(&valid),
            "invalid fold partition"
        );
        let rows: BTreeSet<_> = train.union(&valid).copied().collect();
        ensure!(
            rows.iter().copied().eq(0..rows.len()),
            "fold does not cover a contiguous FIT roster"
        );
        if let Some(expected) = &universe {
            ensure!(&rows == expected, "folds use different FIT rosters");
        } else {
            universe = Some(rows.clone());
        }
        ensure!(valid.is_disjoint(&validation), "validation folds overlap");
        validation.extend(valid.iter().copied());
        fold_sets.push((train, valid));
    }
    let all_rows = universe.context("missing FIT folds")?;
    ensure!(
        validation == all_rows,
        "validation folds do not cover the full FIT roster"
    );
    ensure!(
        fold_sets[0].0 == fold_sets[1].1 && fold_sets[0].1 == fold_sets[1].0,
        "r782 folds are not complementary"
    );
    Ok(())
}

pub(super) fn evaluator_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"jope.research.r782-v2-evaluator/v1\0");
    hash.update(include_bytes!("../worker/evaluate.rs"));
    hash.update(include_bytes!("../run/surrogate.rs"));
    hash.update(GATE_RULE.as_bytes());
    format!("{:x}", hash.finalize())
}

pub(super) fn seed_protocol_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(b"jope.research.evaluation-seed/v1\0");
    hash.update(SEED_PROTOCOL);
    hash.update(include_bytes!("../discover/loop.rs"));
    format!("{:x}", hash.finalize())
}

pub(super) fn conditioning_sha256(plan: &Plan, conditioning: &TokenProjection) -> Result<String> {
    ensure!(
        conditioning
            .tokens
            .iter()
            .flatten()
            .all(|value| value.is_finite()),
        "nonfinite FIT conditioning"
    );
    ensure!(
        conditioning
            .mask
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0),
        "invalid FIT conditioning mask"
    );
    let mut hash = Sha256::new();
    hash.update(b"jope.research.fit-conditioning/v1\0");
    hash.update(serde_json::to_vec(plan)?);
    for value in conditioning.tokens.iter().flatten() {
        hash.update(value.to_bits().to_le_bytes());
    }
    for value in conditioning.mask {
        hash.update([u8::from(value == 1.0)]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
