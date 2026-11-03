use anyhow::{ensure, Result};

use super::MAX_COLUMN_NAME_BYTES;

pub(super) fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
pub(super) fn probability(x: f64) -> bool {
    x.is_finite() && x > 0.0 && x < 1.0
}
pub(super) fn finite(x: f64) -> Result<f64> {
    ensure!(x.is_finite(), "non-finite fitness value");
    Ok(x)
}
pub(super) fn mean(v: &[f64]) -> f64 {
    v.iter().map(|x| x / v.len() as f64).sum()
}
/// Weighted mean; weights are validated nonnegative with a positive finite total.
pub(super) fn wmean(v: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => mean(v),
        Some(w) => {
            let total: f64 = w.iter().sum();
            v.iter().zip(w).map(|(x, w)| x * (w / total)).sum()
        }
    }
}
pub(super) fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let p = q * (v.len() - 1) as f64;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}
/// Linear interpolation at `q·(n−1)` over the sorted values. Weighted, each value sits at the
/// share of weight below it, scaled so the largest value sits at 1: equal weights give exactly the
/// unweighted quantile. Zero-weight rows are ignored.
pub(super) fn wquantile(v: &[f64], w: Option<&[f64]>, q: f64) -> Result<f64> {
    let Some(w) = w else {
        return Ok(quantile(v, q));
    };
    let mut order: Vec<usize> = (0..v.len()).filter(|&i| w[i] > 0.0).collect();
    ensure!(!order.is_empty(), "every row has zero weight");
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]).then(a.cmp(&b)));
    let last = order.len() - 1;
    let span: f64 = order[..last].iter().map(|&i| w[i]).sum();
    if last == 0 || q <= 0.0 {
        return Ok(v[order[0]]);
    }
    let mut below = 0.0;
    let positions: Vec<f64> = order
        .iter()
        .map(|&i| {
            let p = below / span;
            below += w[i];
            p
        })
        .collect();
    let k = positions.partition_point(|&p| p < q).clamp(1, last);
    let (p0, p1) = (positions[k - 1], positions[k]);
    let (v0, v1) = (v[order[k - 1]], v[order[k]]);
    Ok(v0 + (v1 - v0) * ((q - p0) / (p1 - p0)).min(1.0))
}
/// Mean of the largest values that hold `fraction` of the total weight (each row weighs 1 when
/// unweighted); the boundary row contributes the part of its weight that is needed.
pub(super) fn tail_mean(v: &[f64], w: Option<&[f64]>, fraction: f64) -> Result<f64> {
    let weight = |i: usize| w.map_or(1.0, |w| w[i]);
    let total: f64 = (0..v.len()).map(weight).sum();
    let mass = fraction * total;
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    let (mut taken, mut sum) = (0.0, 0.0);
    for i in order {
        let take = weight(i).min(mass - taken);
        if take <= 0.0 {
            if weight(i) > 0.0 {
                break;
            }
            continue;
        }
        sum += take * v[i];
        taken += take;
    }
    ensure!(taken > 0.0, "tail holds no weight");
    finite(sum / taken)
}
pub(super) fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}
pub(super) fn wvariance(v: &[f64], w: Option<&[f64]>) -> f64 {
    let m = wmean(v, w);
    wmean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>(), w)
}
pub(super) fn correlation(a: &[f64], b: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let am = wmean(a, w);
    let bm = wmean(b, w);
    let av = wvariance(a, w);
    let bv = wvariance(b, w);
    ensure!(
        av.is_finite() && bv.is_finite(),
        "correlation overflows: the variance of a vector is not finite"
    );
    ensure!(
        av > 0.0 && bv > 0.0,
        "correlation undefined for a constant vector"
    );
    finite(
        wmean(
            &a.iter()
                .zip(b)
                .map(|(x, y)| (x - am) * (y - bm))
                .collect::<Vec<_>>(),
            w,
        ) / av.sqrt()
            / bv.sqrt(),
    )
}
pub(super) fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite inputs"));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[i]] == v[order[j]] {
            j += 1;
        }
        for k in i..j {
            out[order[k]] = (i + j - 1) as f64 / 2.0;
        }
        i = j;
    }
    out
}
/// `(weight, target)` in descending order of `key`. A group of tied keys shares its weighted mean
/// target, so a tie is one straight segment of a Lorenz or capture curve (unweighted, the expected
/// curve over every order of the tied rows): the rows' file order never counts as ranking skill.
pub(super) fn ranked(y: &[f64], key: &[f64], w: Option<&[f64]>) -> Vec<(f64, f64)> {
    let weight = |i: usize| w.map_or(1.0, |w| w[i]);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    let mut out = Vec::with_capacity(order.len());
    for group in order.chunk_by(|&a, &b| key[a] == key[b]) {
        let total: f64 = group.iter().map(|&i| weight(i)).sum();
        let target = if total > 0.0 {
            group.iter().map(|&i| weight(i) * y[i]).sum::<f64>() / total
        } else {
            group.iter().map(|&i| y[i]).sum::<f64>() / group.len() as f64
        };
        out.extend(group.iter().map(|&i| (weight(i), target)));
    }
    out
}
/// Twice the trapezoid area under the cumulative-weight vs cumulative-(w·y) curve, minus one.
pub(super) fn lorenz_gini(ranked: &[(f64, f64)]) -> Result<f64> {
    let total_w: f64 = ranked.iter().map(|(w, _)| w).sum();
    let total_y: f64 = ranked.iter().map(|(w, y)| w * y).sum();
    ensure!(
        total_w > 0.0 && total_y.is_finite() && total_y != 0.0,
        "Gini needs a nonzero weighted target total"
    );
    let (mut x0, mut y0, mut area) = (0.0, 0.0, 0.0);
    for &(w, y) in ranked {
        let x1 = x0 + w / total_w;
        let y1 = y0 + w * y / total_y;
        area += (x1 - x0) * (y0 + y1) / 2.0;
        (x0, y0) = (x1, y1);
    }
    finite(2.0 * area - 1.0)
}
/// Net per-interval returns of the documented position convention.
pub(super) fn strategy_returns(y: &[f64], p: &[f64], max_position: f64, cost: f64) -> Result<Vec<f64>> {
    let mut last = 0.0;
    let returns: Vec<_> = y
        .iter()
        .zip(p)
        .map(|(y, p)| {
            let position = p.clamp(-max_position, max_position);
            let r = position * y - cost * (position - last).abs();
            last = position;
            r
        })
        .collect();
    ensure!(
        returns.iter().all(|x| x.is_finite()),
        "non-finite strategy return"
    );
    Ok(returns)
}
/// Compounded equity curve from 1; a period losing everything ends the curve at zero. An
/// overflowing curve is an error, never a finite score.
pub(super) fn equity_curve(returns: &[f64]) -> Result<Vec<f64>> {
    let mut equity = 1.0f64;
    let mut curve = Vec::with_capacity(returns.len());
    for r in returns {
        let next = equity * (1.0 + r);
        ensure!(next.is_finite(), "the equity curve overflowed");
        equity = next.max(0.0);
        curve.push(equity);
    }
    Ok(curve)
}
pub(super) fn max_drawdown(returns: &[f64]) -> Result<f64> {
    let mut peak = 1.0f64;
    let mut worst = 0.0f64;
    for e in equity_curve(returns)? {
        peak = peak.max(e);
        worst = worst.max(1.0 - e / peak);
    }
    Ok(worst)
}
pub(super) fn validate_column_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= MAX_COLUMN_NAME_BYTES
            && !name.chars().any(char::is_control),
        "column names need 1..{MAX_COLUMN_NAME_BYTES} bytes without control characters"
    );
    Ok(())
}

pub(super) fn prediction_turnover(p: &[f64]) -> f64 {
    if p.len() < 2 {
        0.0
    } else {
        p.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / ((p.len() - 1) as f64)
    }
}
