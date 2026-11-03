use super::*;

pub(super) struct DistilledRound {
    pub(super) reloaded: Vec<distilled::DistilledRow>,
    pub(super) raw_keys: BTreeSet<String>,
    pub(super) parquet_bytes: Vec<u8>,
    pub(super) parquet_sha256: String,
    pub(super) tokens_by_prog: BTreeMap<String, Option<Vec<u32>>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn distill_round(
    pools: &[Pool],
    folds: &[usize],
    active_acquired_by_fold: &[Vec<BTreeSet<usize>>],
    base_acquired: &[BTreeSet<usize>],
    cfg: &FlywheelConfig,
    direction: ScoreDirection,
    scores_set: &BTreeSet<String>,
    parquet_dir: Option<&Path>,
    r: usize,
) -> Result<DistilledRound> {
    // 1. Distill each outer fold's currently acquired training observations into gp_distilled/v1 Parquet and read back.
    let mut round_distilled = Vec::new();
    let mut raw_keys = BTreeSet::new();
    let mut tokens_by_prog: BTreeMap<String, Option<Vec<u32>>> = BTreeMap::new();

    for (k, fold_acquired) in active_acquired_by_fold.iter().enumerate() {
        let run_id = format!("flywheel-r{r:02}-fold{k}");
        for (j, pool) in pools.iter().enumerate() {
            if folds[j] == k {
                // Held-out fold `k` is strictly excluded from outer fold `k`'s training archive.
                continue;
            }
            let spec = SelfPlayDistillSpec {
                dataset: pool.parent_id.clone(),
                context_sha256: pool.context_id.clone(),
                fold: Some(folds[j] as i64),
                split_role: SplitRole::Train,
                score_name: cfg.score_name.clone(),
                policylm_ckpt: None,
                campaign: Some("clover-s3-distilled-flywheel".into()),
                run_id: run_id.clone(),
                family: "clover-selfplay".into(),
                source_bucket: "local-shadow-pools".into(),
            };
            let mut obs = Vec::with_capacity(fold_acquired[j].len());
            for &m_idx in &fold_acquired[j] {
                let m = &pool.members[m_idx];
                tokens_by_prog
                    .entry(m.program_sha256.clone())
                    .or_insert_with(|| m.tokens.clone());
                let (outcome, raw_score) = match (m.status, m.q) {
                    (Status::Complete, Some(q)) => {
                        let s = match direction {
                            ScoreDirection::LowerIsBetter => {
                                if q == 0.0 {
                                    0.0
                                } else {
                                    -q
                                }
                            }
                            ScoreDirection::HigherIsBetter => q,
                        };
                        (DistilledOutcome::Ok, Some(s))
                    }
                    (Status::Timeout, _) => (DistilledOutcome::Timeout, None),
                    _ => (DistilledOutcome::Failed, None),
                };
                let key = format!(
                    "shadow-pools/round-{r:02}/fold-{k}/{}/{}.json",
                    pool.context_id, m.program_sha256
                );
                let pool_sha256 = pool.sha256()?;
                let src_sha = canonical_sha256(&(
                    &pool_sha256,
                    r,
                    k,
                    &pool.context_id,
                    &m.program_sha256,
                    m.status,
                    m.q,
                ))?;
                raw_keys.insert(key.clone());
                obs.push(SelfPlayDistillObservation {
                    program_sha: m.program_sha256.clone(),
                    program: None,
                    outcome,
                    score: raw_score,
                    seed_packet: Some(if base_acquired[j].contains(&m_idx) {
                        "round0_cold_start".into()
                    } else {
                        format!("surrogate_funnel_up_to_r{r}")
                    }),
                    source_key: key,
                    source_version_id: Some(format!("r{r}-k{k}")),
                    source_sha256: src_sha,
                });
            }
            let batch = distill_selfplay_observations(&spec, &obs, &scores_set)?;
            round_distilled.extend(batch);
        }
    }

    let parquet_bytes = distilled::write(
        &round_distilled,
        &scores_set,
        &[("flywheel_round".into(), r.to_string())],
    )?;
    let parquet_sha256 = sha256_hex(&parquet_bytes);
    if let Some(dir) = parquet_dir {
        let shard_path = dir.join(format!("round-{r:02}.parquet"));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&shard_path)
            .with_context(|| {
                format!(
                    "create {} (outputs are never overwritten)",
                    shard_path.display()
                )
            })?;
        f.write_all(&parquet_bytes)?;
        f.sync_all()?;
    }
    let reloaded = distilled::read(&parquet_bytes, &scores_set)?;
    distilled::reconcile(&reloaded, &raw_keys)?;

    Ok(DistilledRound {
        reloaded,
        raw_keys,
        parquet_bytes,
        parquet_sha256,
        tokens_by_prog,
    })
}
