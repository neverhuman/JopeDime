use super::*;

pub fn sequence_nll(
    logits: &Tensor,
    targets: &Tensor,
    legal: &Tensor,
    pad: &Tensor,
    weights: &Tensor,
) -> Result<Tensor> {
    ensure!(
        legal.any_dim(-1, false).all().int64_value(&[]) != 0,
        "empty legal token set"
    );
    // A row without a supervised position would keep its weight in the
    // denominator with a zero numerator and dilute every other row (W6 G8).
    ensure!(
        pad.any_dim(-1, false).all().int64_value(&[]) != 0,
        "sequence row without supervised positions"
    );
    ensure!(
        legal
            .gather(-1, &targets.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .logical_or(&pad.logical_not())
            .all()
            .int64_value(&[])
            != 0,
        "illegal supervision token"
    );
    let logp = logits
        .to_kind(Kind::Float)
        .masked_fill(&legal.logical_not(), f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float);
    let nll = -logp
        .gather(-1, &targets.unsqueeze(-1), false)
        .squeeze_dim(-1);
    let smooth = -logp.masked_fill(&legal.logical_not(), 0.0).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / legal
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        .clamp_min(1.0);
    let nll: Tensor = nll * (1.0 - LABEL_SMOOTHING) + smooth * LABEL_SMOOTHING;
    let nll = nll.masked_fill(&pad.logical_not(), 0.0);
    let per_row = nll.sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        / pad
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .clamp_min(1.0);
    let weights = weights.clamp_min(0.0);
    Ok((per_row * &weights).sum(Kind::Float) / weights.sum(Kind::Float).clamp_min(1e-8))
}
/// Per-row log-probability of the supervised token sequence under the policy:
/// the pad-masked SUM of grammar-masked token log-probs, with no label
/// smoothing and no length normalisation. This is the quantity a preference
/// objective compares between the live policy and a frozen reference.
/// `sequence_nll` deliberately applies both smoothing and length normalisation
/// and must not be reused for it.
pub fn sequence_logprob(
    logits: &Tensor,
    targets: &Tensor,
    legal: &Tensor,
    pad: &Tensor,
) -> Result<Tensor> {
    ensure!(
        legal
            .gather(-1, &targets.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .logical_or(&pad.logical_not())
            .all()
            .int64_value(&[])
            != 0,
        "illegal supervision token"
    );
    let logp = logits
        .to_kind(Kind::Float)
        .masked_fill(&legal.logical_not(), f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float)
        .gather(-1, &targets.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .masked_fill(&pad.logical_not(), 0.0);
    Ok(logp.sum_dim_intlist(&[-1i64][..], false, Kind::Float))
}

/// Ordered preference pairs inside each comparison group: the group's best
/// program against every other member, plus each adjacent pair in value order
/// (where the champion-margin information lives). Pairs whose value gap is
/// zero, non-finite, or below `min_gap_fraction` of the group's median
/// non-zero gap are skipped and counted, so a preference objective does not
/// train on fold noise. Uniform pair weight: the |Δvalue| weighting used by the
/// critic's ranking losses is intentionally not reused here.
pub fn preference_pairs(
    values: &[f64],
    groups: &[String],
    min_gap_fraction: f64,
) -> (Vec<i64>, Vec<i64>, usize) {
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i);
    }
    let mut winners = Vec::new();
    let mut losers = Vec::new();
    let mut skipped = 0usize;
    for ids in grouped.values() {
        let mut ordered: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&i| values[i].is_finite())
            .collect();
        // Best first; ties break on the row index so the order is stable.
        ordered.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        if ordered.len() < 2 {
            continue;
        }
        let mut gaps: Vec<f64> = ordered
            .windows(2)
            .map(|w| values[w[0]] - values[w[1]])
            .filter(|g| *g > 0.0)
            .collect();
        gaps.sort_by(|a, b| a.total_cmp(b));
        let floor = if gaps.is_empty() {
            0.0
        } else {
            gaps[gaps.len() / 2] * min_gap_fraction
        };
        let mut push = |w: usize, l: usize| {
            let gap = values[w] - values[l];
            if gap > 0.0 && gap >= floor {
                winners.push(w as i64);
                losers.push(l as i64);
            } else {
                skipped += 1;
            }
        };
        let best = ordered[0];
        for &l in &ordered[1..] {
            push(best, l);
        }
        // Adjacent pairs; (best, second) is already covered above.
        for pair in ordered.windows(2).skip(1) {
            push(pair[0], pair[1]);
        }
    }
    (winners, losers, skipped)
}

/// Direct preference optimisation (Rafailov et al. 2023) over ordered pairs:
/// `-log sigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l)))`, mean over pairs.
/// The reference log-probabilities are constants, so the gradient lands only
/// on the live policy's own sequence probabilities — the decoder, not the
/// critic. Returns a connected zero when there are no pairs.
pub fn preference_dpo(
    logp: &Tensor,
    logp_ref: &Tensor,
    winners: &[i64],
    losers: &[i64],
    beta: f64,
) -> Result<Tensor> {
    ensure!(winners.len() == losers.len(), "preference pair mismatch");
    ensure!(
        logp.size() == logp_ref.size(),
        "preference log-probability shape mismatch"
    );
    if winners.is_empty() {
        return Ok(logp.sum(Kind::Float) * 0.0);
    }
    let device = logp.device();
    let w = Tensor::from_slice(winners).to_device(device);
    let l = Tensor::from_slice(losers).to_device(device);
    let margin = logp - logp_ref;
    let diff = margin.index_select(0, &w) - margin.index_select(0, &l);
    Ok(-(diff * beta).log_sigmoid().mean(Kind::Float))
}

pub fn ranking(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
) -> Result<(Tensor, Tensor)> {
    ranking_losses(reward, values, percentiles, groups, true)
}

/// The ablation objective uses only Bradley–Terry loss. Avoid constructing
/// per-group listwise graphs and transferring labels for a discarded result.
pub fn ranking_pairwise(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
) -> Result<Tensor> {
    Ok(ranking_losses(reward, values, percentiles, groups, false)?.0)
}

fn ranking_losses(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
    include_listwise: bool,
) -> Result<(Tensor, Tensor)> {
    ensure!(
        values.len() == groups.len() && percentiles.len() == groups.len(),
        "ranking labels mismatch"
    );
    let mut grouped: BTreeMap<&str, Vec<i64>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i as i64);
    }
    let mut winners = Vec::new();
    let mut losers = Vec::new();
    let mut weights = Vec::new();
    let mut listwise = Vec::new();
    for ids in grouped.values() {
        for (pos, &a) in ids.iter().enumerate() {
            for &b in &ids[pos + 1..] {
                let va = values[a as usize];
                let vb = values[b as usize];
                if !va.is_finite() || !vb.is_finite() || va == vb {
                    continue;
                }
                let (w, l) = if va > vb { (a, b) } else { (b, a) };
                winners.push(w);
                losers.push(l);
                weights.push(
                    (va - vb).abs()
                        * if percentiles[a as usize].max(percentiles[b as usize]) >= BOOST_THRESHOLD
                        {
                            BOOST_FACTOR
                        } else {
                            1.0
                        },
                );
            }
        }
        if include_listwise
            && ids.len() >= LISTWISE_MIN_MEMBERS
            && ids.iter().all(|&i| percentiles[i as usize].is_finite())
        {
            let ps = Tensor::from_slice(
                &ids.iter()
                    .map(|&i| percentiles[i as usize])
                    .collect::<Vec<_>>(),
            )
            .to_device(reward.device());
            let target = (&ps / LISTWISE_TEMPERATURE).softmax(0, Kind::Float);
            let pred = reward
                .index_select(0, &Tensor::from_slice(ids).to_device(reward.device()))
                .to_kind(Kind::Float)
                .log_softmax(0, Kind::Float);
            listwise.push(-(target * pred).sum(Kind::Float));
        }
    }
    let zero = reward.sum(Kind::Float) * 0.0;
    let bt = if winners.is_empty() {
        zero.shallow_clone()
    } else {
        let scale = weights.iter().sum::<f64>() / weights.len() as f64;
        // Repeated index_select entries backpropagate through CUDA atomic adds.
        // Tiny, unordered cancellation errors then become visible AdamW updates
        // to the otherwise translation-invariant reward biases. Broadcast each
        // reward once and reduce a dense comparison matrix in a fixed layout.
        let n = groups.len();
        let mut dense = vec![0f32; n * n];
        for ((&w, &l), &weight) in winners.iter().zip(&losers).zip(&weights) {
            dense[w as usize * n + l as usize] = (weight / scale) as f32;
        }
        let weights = Tensor::from_slice(&dense)
            .view([n as i64, n as i64])
            .to_device(reward.device());
        let scores = reward.to_kind(Kind::Float);
        (((scores.unsqueeze(0) - scores.unsqueeze(1)).softplus()) * weights).sum(Kind::Float)
            / winners.len() as f64
    };
    Ok((
        bt,
        if listwise.is_empty() {
            zero
        } else {
            Tensor::stack(&listwise, 0).mean(Kind::Float)
        },
    ))
}
