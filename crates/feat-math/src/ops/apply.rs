use super::*;

pub fn apply_op_train(op: &FittedOp, x: &Rows) -> std::result::Result<Rows, WidthMismatch> {
    apply_op(op, x)
}

pub fn apply_op(op: &FittedOp, x: &Rows) -> std::result::Result<Rows, WidthMismatch> {
    if matches!(op, FittedOp::Roll { .. }) {
        reject_zero_width_rows(x)?;
    }
    if op_append(op) && !x.is_empty() && x.iter().all(|r| r.len() >= crate::MAX_WIDTH) {
        if x.iter()
            .all(|r| r.len() == crate::MAX_WIDTH && r.iter().all(|v| v.is_finite()))
        {
            return Ok(x.clone());
        }
        return Ok(finish(op, x, vec![Vec::new(); x.len()]));
    }
    let generated = match op {
        FittedOp::QuantileBins {
            edges,
            bins,
            append,
        } if x.iter().all(|r| r.len() == edges.len()) => map_rows(x, |r| {
            let cap = if *append {
                crate::MAX_WIDTH.saturating_sub(r.len())
            } else {
                crate::MAX_WIDTH
            };
            r.iter()
                .take(cap)
                .enumerate()
                .map(|(j, v)| lower_bound(&edges[j], *v) as f64 / (*bins as f64 - 1.0))
                .collect()
        }),
        FittedOp::Hinge { knots, append } if x.iter().all(|r| r.len() == knots.len()) => {
            let knot_width = knots.first().map_or(0, Vec::len);
            map_rows(x, |r| {
                let cap = if *append {
                    crate::MAX_WIDTH.saturating_sub(r.len())
                } else {
                    crate::MAX_WIDTH
                };
                let mut o = Vec::with_capacity((r.len() * knot_width).min(cap));
                'outer: for (j, v) in r.iter().enumerate() {
                    for t in &knots[j] {
                        if o.len() >= cap {
                            break 'outer;
                        }
                        o.push((v - t).max(0.0));
                    }
                }
                o
            })
        }
        FittedOp::Cheb { degree, append } => map_rows(x, |r| {
            let cap = if *append {
                crate::MAX_WIDTH.saturating_sub(r.len())
            } else {
                crate::MAX_WIDTH
            };
            let mut o = Vec::with_capacity((r.len() * degree).min(cap));
            'outer: for v in r {
                let u = (2.0 * v - 1.0).clamp(-1.0, 1.0);
                let mut t_prev = 1.0;
                let mut t_cur = u;
                for _ in 0..*degree {
                    if o.len() >= cap {
                        break 'outer;
                    }
                    o.push(t_cur);
                    let t_next = 2.0 * u * t_cur - t_prev;
                    t_prev = t_cur;
                    t_cur = t_next;
                }
            }
            o
        }),
        FittedOp::PolyInteract { pairs, append } | FittedOp::PairProd { pairs, append } => {
            map_rows(x, |r| {
                let cap = if *append {
                    crate::MAX_WIDTH.saturating_sub(r.len())
                } else {
                    crate::MAX_WIDTH
                };
                pairs.iter().take(cap).map(|(a, b)| r[*a] * r[*b]).collect()
            })
        }
        _ => apply_op_generated(op, x),
    };
    Ok(finish(op, x, generated))
}

/// The existing transform kernel before the legacy append/width/sanitize epilogue.
/// Advanced-v3 owns those semantics through its feature roles, masks and lineage;
/// it must validate generated width before any information can be truncated.
pub(crate) fn apply_op_generated(op: &FittedOp, x: &Rows) -> Rows {
    match op {
        FittedOp::Standardize { means, scales, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| ((v - means[j]) / scales[j]).clamp(-20.0, 20.0))
                .collect()
        }),
        FittedOp::RobustScale {
            medians, scales, ..
        } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| (v - medians[j]) / scales[j])
                .collect()
        }),
        FittedOp::Rank { sorted_cols, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| mid_rank(&sorted_cols[j], *v))
                .collect()
        }),
        FittedOp::Blom { sorted_cols, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| {
                    let n = sorted_cols[j].len() as f64;
                    let mid = mid_rank(&sorted_cols[j], *v) * n;
                    probit((mid + 0.125) / (n + 0.25))
                })
                .collect()
        }),
        FittedOp::QuantileBins { edges, bins, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| lower_bound(&edges[j], *v) as f64 / (*bins as f64 - 1.0))
                .collect()
        }),
        FittedOp::BinOnehot { edges, bins, .. } => map_rows(x, |r| {
            let mut o = vec![0.0; r.len() * bins];
            for (j, v) in r.iter().enumerate() {
                let b = lower_bound(&edges[j], *v).min(bins - 1);
                o[j * bins + b] = 1.0;
            }
            o
        }),
        FittedOp::Hinge { knots, .. } => map_rows(x, |r| {
            let mut o = Vec::with_capacity(r.len() * knots[0].len());
            for (j, v) in r.iter().enumerate() {
                for t in &knots[j] {
                    o.push((v - t).max(0.0));
                }
            }
            o
        }),
        FittedOp::WinsorClip { lo, hi, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| v.clamp(lo[j], hi[j]))
                .collect()
        }),
        FittedOp::Cheb { degree, .. } => map_rows(x, |r| {
            let mut o = Vec::with_capacity(r.len() * degree);
            for v in r {
                let u = (2.0 * v - 1.0).clamp(-1.0, 1.0);
                let mut t_prev = 1.0;
                let mut t_cur = u;
                for _ in 0..*degree {
                    o.push(t_cur);
                    let t_next = 2.0 * u * t_cur - t_prev;
                    t_prev = t_cur;
                    t_cur = t_next;
                }
            }
            o
        }),
        FittedOp::PolyInteract { pairs, .. } | FittedOp::PairProd { pairs, .. } => {
            map_rows(x, |r| pairs.iter().map(|(a, b)| r[*a] * r[*b]).collect())
        }
        FittedOp::RffCos { w, b, .. } => map_rows(x, |r| {
            let scale = (2.0 / w.len() as f64).sqrt();
            (0..w.len())
                .map(|i| scale * crate::portable_math::frozen_s1::cos(dot(&w[i], r) + b[i]))
                .collect()
        }),
        FittedOp::RffSin { w, b, .. } => map_rows(x, |r| {
            let scale = (2.0 / w.len() as f64).sqrt();
            (0..w.len())
                .map(|i| scale * crate::portable_math::frozen_s1::sin(dot(&w[i], r) + b[i]))
                .collect()
        }),
        FittedOp::RandProj { w, .. } => map_rows(x, |r| {
            let inv = 1.0 / (w.len() as f64).sqrt();
            (0..w.len()).map(|i| dot(&w[i], r) * inv).collect()
        }),
        FittedOp::LandmarkDist {
            landmarks, gamma, ..
        }
        | FittedOp::Nystrom {
            landmarks, gamma, ..
        } => map_rows(x, |r| {
            landmarks
                .iter()
                .map(|l| {
                    let mut d2 = 0.0;
                    for j in 0..l.len().min(r.len()) {
                        let d = r[j] - l[j];
                        d2 += d * d;
                    }
                    (-gamma * d2).exp()
                })
                .collect()
        }),
        FittedOp::ObliqueHinge { w, t, .. } => map_rows(x, |r| {
            (0..w.len())
                .map(|i| (dot(&w[i], r) - t[i]).max(0.0))
                .collect()
        }),
        FittedOp::RowStats { .. } => map_rows(x, |r| row_stats(r)),
        FittedOp::TargetBinEncode { edges, enc, .. } => map_rows(x, |r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| {
                    let bins = enc[j].len();
                    enc[j][lower_bound(&edges[j], *v).min(bins - 1)]
                })
                .collect()
        }),
        // Same clamp as MathExpr's sigmoid: keeps Python's math.exp overflow-free.
        FittedOp::Sigmoid { .. } => map_rows(x, |r| {
            r.iter()
                .map(|v| 1.0 / (1.0 + crate::portable_math::frozen_s1::exp(-v.clamp(-30.0, 30.0))))
                .collect()
        }),
        // Hard indicator per cell: 1.0 when v > t else 0.0 (NaN => 0.0). The
        // explicit conditional mirrors the emitted Python/Rust exactly.
        FittedOp::Step { t, .. } => map_rows(x, |r| {
            r.iter().map(|v| if *v > *t { 1.0 } else { 0.0 }).collect()
        }),
        FittedOp::NormRow { .. } => map_rows(x, |r| {
            let mut s = 0.0;
            for v in r {
                s += v * v;
            }
            let inv = 1.0 / s.sqrt().max(1e-9);
            r.iter().map(|v| v * inv).collect()
        }),
        // Per consecutive pair (0,1),(2,3),..: radius then angle atan2(b, a);
        // an odd trailing column passes through unchanged (width preserved).
        FittedOp::Polar { .. } => map_rows(x, |r| {
            let d = r.len();
            let mut o = Vec::with_capacity(d);
            let mut j = 0;
            while j + 1 < d {
                let (a, b) = (r[j], r[j + 1]);
                o.push((a * a + b * b).sqrt());
                o.push(b.atan2(a));
                j += 2;
            }
            if d % 2 == 1 {
                o.push(r[d - 1]);
            }
            o
        }),
        // Row-internal shift: out[j] = x[j-k] for j >= k else 0.0 (k >= d ⇒ zeros).
        FittedOp::Lag { k, .. } => map_rows(x, |r| {
            (0..r.len())
                .map(|j| if j >= *k { r[j - *k] } else { 0.0 })
                .collect()
        }),
        // Circular shift: out[j] = x[(j - k) mod d].
        FittedOp::Roll { k, .. } => map_rows(x, |r| {
            let d = r.len();
            if d == 0 {
                Vec::new()
            } else {
                let s = *k % d;
                (0..d).map(|j| r[(j + d - s) % d]).collect()
            }
        }),
        FittedOp::FftMag { k, .. } => map_rows(x, |r| dft_mags(r, *k)),
        FittedOp::RowSpectral { .. } => map_rows(x, |r| row_spectral(r)),
        FittedOp::RowDiffStats { .. } => map_rows(x, |r| row_diff_stats(r)),
        FittedOp::SvdProject {
            means, components, ..
        } => map_rows(x, |r| {
            components
                .iter()
                .map(|c| {
                    let mut s = 0.0;
                    for j in 0..means.len().min(r.len()) {
                        s += (r[j] - means[j]) * c[j];
                    }
                    s
                })
                .collect()
        }),
        // Elementwise with row context: out[i][j] = expr(x = r[j], row = r).
        FittedOp::MapExpr { expr, .. } => match expr.compile() {
            Ok(compiled) if compiled.uses_row_summaries() => map_rows(x, |r| compiled.map_row(r)),
            // The same-source benchmark finds no benefit for simple formulas
            // without row reductions. Retain their direct execution path.
            Ok(_) => map_rows(x, |r| {
                r.iter()
                    .map(|v| expr.eval_element(*v, r).unwrap_or(0.0))
                    .collect()
            }),
            Err(_) => x.iter().map(|r| vec![0.0; r.len()]).collect(),
        },
        // One derived column per row: out[i] = [expr(row = r)].
        FittedOp::DeriveExpr { expr, .. } => match expr.compile() {
            Ok(compiled) if compiled.uses_row_summaries() => {
                map_rows(x, |r| vec![compiled.eval_row(r).unwrap_or(0.0)])
            }
            Ok(_) => map_rows(x, |r| vec![expr.eval_row(r).unwrap_or(0.0)]),
            Err(_) => vec![vec![0.0]; x.len()],
        },
        FittedOp::Symlog { c, .. } => {
            map_rows(x, |r| r.iter().map(|v| asinh_stable(v / c)).collect())
        }
        FittedOp::SelectCorr { indices, .. } => {
            map_rows(x, |r| indices.iter().map(|&j| r[j]).collect())
        }
        FittedOp::Impute {
            fill,
            indicator_cols,
            ..
        } => map_rows(x, |r| {
            let mut vals: Vec<f64> = (0..r.len())
                .map(|j| if r[j] == 0.0 { fill[j] } else { r[j] })
                .collect();
            for &j in indicator_cols.iter() {
                vals.push(if r[j] == 0.0 { 1.0 } else { 0.0 });
            }
            vals
        }),
        FittedOp::PairRatio { pairs, eps, .. } => map_rows(x, |r| {
            pairs
                .iter()
                .zip(eps)
                .map(|(&(a, b), e)| r[a] * r[b] / (r[b] * r[b] + e))
                .collect()
        }),
        // Truncated-power natural cubic basis; like hinge, over-wide output
        // relies on the shared MAX_WIDTH truncation in `finish`.
        FittedOp::Spline { knots, .. } => map_rows(x, |r| {
            let mut o = Vec::new();
            for (j, v) in r.iter().enumerate() {
                let xi = &knots[j];
                let k = xi.len();
                if xi[k - 1] - xi[0] < 1e-9 {
                    o.push(*v);
                } else {
                    o.push(*v);
                    let pl = (v - xi[k - 1]).max(0.0);
                    let pk = (v - xi[k - 2]).max(0.0);
                    let dlast = (pk * pk * pk - pl * pl * pl) / (xi[k - 1] - xi[k - 2]).max(1e-9);
                    for t in 0..(k - 2) {
                        let pt = (v - xi[t]).max(0.0);
                        let d = (pt * pt * pt - pl * pl * pl) / (xi[k - 1] - xi[t]).max(1e-9);
                        o.push(d - dlast);
                    }
                }
            }
            o
        }),
    }
}

pub(crate) fn op_append(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}
