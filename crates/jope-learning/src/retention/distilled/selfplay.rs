use super::*;

use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Orientation of a registered `gp_distilled/v1` score identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDirection {
    /// Loss or error metric where smaller finite values are better
    /// (`cv_geomean_rmse_ratio`, `best_rust_rmse`, `validation_rmse`, `l6`).
    LowerIsBetter,
    /// Reward or utility metric where larger finite values are better.
    HigherIsBetter,
}

/// Returns the explicit orientation of a registered score identity, refusing any identity
/// without a declared orientation.
pub fn score_direction(score_name: &str) -> Result<ScoreDirection> {
    match score_name {
        "cv_geomean_rmse_ratio" | "best_rust_rmse" | "validation_rmse" | "l6" => {
            Ok(ScoreDirection::LowerIsBetter)
        }
        "shadow_pool_q" => Ok(ScoreDirection::HigherIsBetter),
        other => bail!("score identity {other:?} has no declared ScoreDirection"),
    }
}

/// One self-play or GP inner-cut observation to be distilled into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillObservation {
    /// Program sha recorded for this self play distill observation.
    pub program_sha: String,
    /// Optional program value; `None` means it is not available.
    pub program: Option<String>,
    /// Distilled outcome recorded for this self play distill observation.
    pub outcome: DistilledOutcome,
    /// Measured score under `spec.score_name`, oriented according to
    /// [`score_direction`]`(&spec.score_name)` (required iff `outcome == DistilledOutcome::Ok`).
    pub score: Option<f64>,
    /// Seed provenance label (e.g. `"ordinary_gp"`, `"bootstrap_warm_start"`, `"selfplay_archive_r1"`).
    pub seed_packet: Option<String>,
    /// Source key recorded for this self play distill observation.
    pub source_key: String,
    /// Identifier for the associated source version.
    pub source_version_id: Option<String>,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
}

/// Context-level metadata for distilling a self-play / GP evaluation batch into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillSpec {
    /// Dataset recorded for this self play distill spec.
    pub dataset: String,
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// Optional fold value; `None` means it is not available.
    pub fold: Option<i64>,
    /// Split role recorded for this self play distill spec.
    pub split_role: SplitRole,
    /// Score name recorded for this self play distill spec.
    pub score_name: String,
    /// Optional policylm ckpt value; `None` means it is not available.
    pub policylm_ckpt: Option<String>,
    /// Optional campaign value; `None` means it is not available.
    pub campaign: Option<String>,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Family recorded for this self play distill spec.
    pub family: String,
    /// Source bucket recorded for this self play distill spec.
    pub source_bucket: String,
}

/// Distills a batch of self-play / GP inner-cut observations for one context into validated
/// `gp_distilled/v1` rows, assigning deterministic within-context rank (`rank_ctx = 1..=M`)
/// according to [`score_direction`]`(&spec.score_name)` and recording the rank-1 `champion_id`
/// (with `qualified = false` and `superseded_by = None` because inner-cut self-play measurements
/// are never paired L6 qualifications).
pub fn distill_selfplay_observations(
    spec: &SelfPlayDistillSpec,
    observations: &[SelfPlayDistillObservation],
    known_scores: &BTreeSet<String>,
) -> Result<Vec<DistilledRow>> {
    ensure!(
        !observations.is_empty(),
        "cannot distill an empty self-play observation batch"
    );
    ensure!(
        crate::canon::is_hex64(&spec.context_sha256),
        "context_sha256 {:?} must be lowercase 64-hex",
        spec.context_sha256
    );
    ensure!(
        known_scores.contains(&spec.score_name),
        "unknown score_name {:?}",
        spec.score_name
    );
    ensure!(
        spec.score_name != "l6",
        "self-play inner-cut distillation may not claim score_name \"l6\""
    );
    let direction = score_direction(&spec.score_name)?;

    // Rank all successful observations by best score under `direction`, breaking ties by ascending program_sha.
    let mut ok_indices: Vec<usize> = Vec::new();
    for (idx, obs) in observations.iter().enumerate() {
        match (obs.outcome, obs.score) {
            (DistilledOutcome::Ok, Some(s)) => {
                ensure!(s.is_finite(), "observation {idx}: non-finite score");
                ok_indices.push(idx);
            }
            (DistilledOutcome::Ok, None) => {
                bail!("observation {idx}: outcome ok requires finite score")
            }
            (_, Some(_)) => {
                bail!("observation {idx}: failed/timeout observation must not carry score")
            }
            (_, None) => {}
        }
    }
    ok_indices.sort_by(|&a, &b| {
        let sa = observations[a].score.unwrap();
        let sb = observations[b].score.unwrap();
        let primary = match direction {
            ScoreDirection::LowerIsBetter => sa.total_cmp(&sb),
            ScoreDirection::HigherIsBetter => sb.total_cmp(&sa),
        };
        primary.then_with(|| {
            observations[a]
                .program_sha
                .cmp(&observations[b].program_sha)
        })
    });

    let champion_id = ok_indices
        .first()
        .map(|&i| observations[i].program_sha.clone());
    let mut rank_by_idx = vec![None; observations.len()];
    for (rank_zero, &obs_idx) in ok_indices.iter().enumerate() {
        rank_by_idx[obs_idx] = Some((rank_zero + 1) as i64);
    }

    let mut out = Vec::with_capacity(observations.len());
    for (idx, obs) in observations.iter().enumerate() {
        let row = DistilledRow {
            program_sha: obs.program_sha.clone(),
            program: obs.program.clone(),
            dataset: spec.dataset.clone(),
            context_sha256: Some(spec.context_sha256.clone()),
            fold: spec.fold,
            split_role: spec.split_role,
            outcome: obs.outcome,
            score: obs.score,
            score_name: obs.score.map(|_| spec.score_name.clone()),
            rank_ctx: rank_by_idx[idx],
            champion_id: obs.score.as_ref().and(champion_id.clone()),
            superseded_by: None,
            seed_packet: obs.seed_packet.clone(),
            policylm_ckpt: spec.policylm_ckpt.clone(),
            campaign: spec.campaign.clone(),
            run_id: spec.run_id.clone(),
            family: spec.family.clone(),
            qualified: false,
            l6_delta: None,
            source_bucket: spec.source_bucket.clone(),
            source_key: obs.source_key.clone(),
            source_version_id: obs.source_version_id.clone(),
            source_sha256: obs.source_sha256.clone(),
            pack_key: None,
            pack_entry: None,
        };
        validate(&row, known_scores).with_context(|| format!("self-play distilled row {idx}"))?;
        out.push(row);
    }
    Ok(out)
}

fn program_expr_tokens(expr: &str) -> Vec<u32> {
    let mut tokens = vec![1u32];
    for part in expr
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|s| !s.is_empty())
    {
        let mut h = 2166136261u32;
        for b in part.bytes() {
            h ^= u32::from(b);
            h = h.wrapping_mul(16777619);
        }
        tokens.push(3 + (h % 61));
    }
    tokens.push(2u32);
    tokens
}

/// Bridges validated `gp_distilled/v1` rows for a single non-`l6` score identity into
/// [`crate::pools::pool::MeasurementRow`] records (where `q` is higher-is-better) for shadow pool
/// freezing and cross-fitted surrogate (`ContextualSurrogateQHat`) training.
///
/// Lower-is-better identities (`cv_geomean_rmse_ratio`, `best_rust_rmse`, `validation_rmse`) are
/// negated (`q = -score`) so lower loss/RMSE maps strictly monotonically to higher pool reward `q`.
/// Refuses `"l6"`, unknown/unoriented score identities, mixed score identities, mixed `fold`
/// values within one `context_sha256`, missing `context_sha256`, or `SplitRole::None` / `Unknown`.
pub fn distilled_rows_to_measurement_rows(
    rows: &[DistilledRow],
    score_name: &str,
    protocol_sha256: &str,
    known_scores: &BTreeSet<String>,
) -> Result<Vec<crate::pools::pool::MeasurementRow>> {
    ensure!(
        !rows.is_empty(),
        "cannot convert empty distilled row slice to measurement rows"
    );
    ensure!(
        crate::canon::is_hex64(protocol_sha256),
        "protocol_sha256 must be lowercase 64-hex"
    );
    ensure!(
        known_scores.contains(score_name),
        "unknown score identity {score_name:?}"
    );
    ensure!(
        score_name != "l6",
        "held-out l6 validation loss rows may not be bridged into shadow training pools"
    );
    let direction = score_direction(score_name)?;
    let mut fold_by_ctx: BTreeMap<&str, Option<i64>> = BTreeMap::new();
    let mut out = Vec::with_capacity(rows.len());
    for (idx, r) in rows.iter().enumerate() {
        validate(r, known_scores).with_context(|| format!("distilled row {idx}"))?;
        if let Some(name) = &r.score_name {
            ensure!(
                name == score_name,
                "row {idx}: cannot mix score_name {name:?} with requested {score_name:?}"
            );
        }
        let ctx_sha = r
            .context_sha256
            .as_ref()
            .with_context(|| format!("row {idx}: context_sha256 is required for pool freezing"))?;
        ensure!(
            crate::canon::is_hex64(ctx_sha) && crate::canon::is_hex64(&r.program_sha),
            "row {idx}: context_sha256 and program_sha must be lowercase 64-hex"
        );
        if let Some(&prev_fold) = fold_by_ctx.get(ctx_sha.as_str()) {
            ensure!(
                prev_fold == r.fold,
                "row {idx}: context_sha256 {ctx_sha} mixes multiple fold values ({prev_fold:?} vs {:?})",
                r.fold
            );
        } else {
            fold_by_ctx.insert(ctx_sha.as_str(), r.fold);
        }
        let role = match r.split_role {
            SplitRole::Train => "train",
            SplitRole::Val263 => "val",
            SplitRole::Sealed => "sealed",
            SplitRole::TestExposed => "test",
            SplitRole::None | SplitRole::Unknown => {
                bail!(
                    "row {idx}: split_role {} may not enter a measurement row",
                    r.split_role.as_str()
                )
            }
        };
        let (status, q) = match r.outcome {
            DistilledOutcome::Ok => {
                let raw_score = r
                    .score
                    .with_context(|| format!("row {idx}: complete row missing score"))?;
                let oriented_q = match direction {
                    ScoreDirection::LowerIsBetter => {
                        if raw_score == 0.0 {
                            0.0
                        } else {
                            -raw_score
                        }
                    }
                    ScoreDirection::HigherIsBetter => raw_score,
                };
                (crate::pools::pool::Status::Complete, Some(oriented_q))
            }
            DistilledOutcome::Failed | DistilledOutcome::Partial => {
                (crate::pools::pool::Status::ProgramFailed, None)
            }
            DistilledOutcome::Timeout => (crate::pools::pool::Status::Timeout, None),
        };
        let tokens = r.program.as_deref().map(program_expr_tokens);
        out.push(crate::pools::pool::MeasurementRow {
            context_id: ctx_sha.clone(),
            parent_id: r.dataset.clone(),
            role: role.into(),
            program_sha256: r.program_sha.clone(),
            status,
            q,
            incumbent_q: None,
            protocol_sha256: protocol_sha256.to_owned(),
            tokens,
        });
    }
    Ok(out)
}
