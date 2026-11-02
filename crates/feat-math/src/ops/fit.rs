use super::*;

pub(crate) fn fit_op_with_cache(
    stage: &Stage,
    x: &Rows,
    y: &[f64],
    cache: &mut SortedColumnsCache,
) -> Result<FittedOp> {
    if x.is_empty() || x[0].is_empty() {
        bail!("empty input to stage {}", stage.op);
    }
    let cols = x[0].len();
    let n = x.len();
    let append = get_append(stage);
    let op = match stage.op.as_str() {
        "standardize" | "norm_features" => {
            let mut means = vec![0.0; cols];
            let mut scales = vec![1.0; cols];
            for j in 0..cols {
                let mut s = 0.0;
                for r in x {
                    s += r[j];
                }
                let m = s / n as f64;
                let mut v = 0.0;
                for r in x {
                    let d = r[j] - m;
                    v += d * d;
                }
                means[j] = m;
                scales[j] = (v / n as f64).sqrt().max(1e-9);
            }
            FittedOp::Standardize {
                means,
                scales,
                append,
            }
        }
        "robust_scale" => {
            cache.ensure_cols(x, cols);
            let mut medians = vec![0.0; cols];
            let mut scales = vec![1.0; cols];
            for j in 0..cols {
                let s = cache.col_slice(j);
                medians[j] = quantile_sorted(s, 0.5);
                let iqr = quantile_sorted(s, 0.75) - quantile_sorted(s, 0.25);
                scales[j] = iqr.max(1e-9);
            }
            FittedOp::RobustScale {
                medians,
                scales,
                append,
            }
        }
        "rank" => {
            cache.ensure_cols(x, cols);
            FittedOp::Rank {
                sorted_cols: (0..cols).map(|j| cache.col_vec(j)).collect(),
                append,
            }
        }
        "blom" => {
            cache.ensure_cols(x, cols);
            FittedOp::Blom {
                sorted_cols: (0..cols).map(|j| cache.col_vec(j)).collect(),
                append,
            }
        }
        "quantile_bins" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 64);
            FittedOp::QuantileBins {
                edges: quantile_edges_with_cache(x, bins, cache),
                bins,
                append,
            }
        }
        "bin_onehot" => {
            let bins = stage.get_usize("bins", 6).clamp(2, 32);
            FittedOp::BinOnehot {
                edges: quantile_edges_with_cache(x, bins, cache),
                bins,
                append,
            }
        }
        "hinge" => {
            let k = stage.get_usize("knots", 4).clamp(1, 16);
            cache.ensure_cols(x, cols);
            let knots = (0..cols)
                .map(|j| {
                    let s = cache.col_slice(j);
                    (1..=k)
                        .map(|i| quantile_sorted(s, i as f64 / (k + 1) as f64))
                        .collect()
                })
                .collect();
            FittedOp::Hinge { knots, append }
        }
        "winsor_clip" | "winsor_moments" => {
            let q = stage.get_f64("q", 0.05).clamp(0.0, 0.25);
            cache.ensure_cols(x, cols);
            let mut lo = vec![0.0; cols];
            let mut hi = vec![0.0; cols];
            for j in 0..cols {
                let s = cache.col_slice(j);
                lo[j] = quantile_sorted(s, q);
                hi[j] = quantile_sorted(s, 1.0 - q).max(lo[j]);
            }
            FittedOp::WinsorClip { lo, hi, append }
        }
        "cheb" => FittedOp::Cheb {
            degree: stage.get_usize("degree", 4).clamp(1, 8),
            append,
        },
        "poly_interact" => {
            let mut pairs = Vec::new();
            for i in 0..cols {
                pairs.push((i, i));
            }
            'outer: for i in 0..cols {
                for j in (i + 1)..cols {
                    if pairs.len() >= crate::MAX_WIDTH {
                        break 'outer;
                    }
                    pairs.push((i, j));
                }
            }
            FittedOp::PolyInteract { pairs, append }
        }
        "rff_cos" | "rff_sin" => {
            let nf = stage.get_usize("n", 32).clamp(1, 256);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, nf);
            let scale = (2.0 * gamma).sqrt();
            let mut w = Vec::with_capacity(nf);
            for _ in 0..nf {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian() * scale);
                }
                w.push(row);
            }
            let b: Vec<f64> = (0..nf).map(|_| rng.range_f64(0.0, PI2)).collect();
            if stage.op == "rff_cos" {
                FittedOp::RffCos { w, b, append }
            } else {
                FittedOp::RffSin { w, b, append }
            }
        }
        "rand_proj" => {
            let k = stage.get_usize("k", 16).clamp(1, 256);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut w = Vec::with_capacity(k);
            for _ in 0..k {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian());
                }
                w.push(row);
            }
            FittedOp::RandProj { w, append }
        }
        "landmark_dist" => {
            let k = stage.get_usize("k", 16).clamp(1, 128);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut landmarks = Vec::with_capacity(k);
            for _ in 0..k {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.range_f64(0.0, 1.0));
                }
                landmarks.push(row);
            }
            FittedOp::LandmarkDist {
                landmarks,
                gamma,
                append,
            }
        }
        "oblique_hinge" => {
            let nf = stage.get_usize("n", 32).clamp(1, 256);
            let seed = stage.get_u64("seed", 7);
            let mut rng = Pcg32::for_stage(seed, cols, nf);
            let inv = 1.0 / (cols as f64).sqrt();
            let mut w = Vec::with_capacity(nf);
            for _ in 0..nf {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(rng.gaussian() * inv);
                }
                w.push(row);
            }
            // Fitted thresholds: median of each projection over train rows.
            let mut t = Vec::with_capacity(nf);
            for wi in &w {
                let mut proj: Vec<f64> = x.iter().map(|r| dot(wi, r)).collect();
                proj.sort_by(|a, b| a.total_cmp(b));
                t.push(quantile_sorted(&proj, 0.5));
            }
            FittedOp::ObliqueHinge { w, t, append }
        }
        "row_stats" => FittedOp::RowStats { append },
        "pair_prod" => {
            let topk = stage.get_usize("topk", 8).clamp(2, 24).min(cols);
            // Target-guided column selection: |pearson(x_j, y)| descending, index ascending.
            let mut scored: Vec<(usize, f64)> = (0..cols)
                .map(|j| {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    (j, crate::metrics::pearson(&col, y).abs())
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut sel: Vec<usize> = scored[..topk].iter().map(|(j, _)| *j).collect();
            sel.sort_unstable();
            let mut pairs = Vec::new();
            for a in 0..sel.len() {
                for b in (a + 1)..sel.len() {
                    pairs.push((sel[a], sel[b]));
                }
            }
            FittedOp::PairProd { pairs, append }
        }
        "target_bin_encode" => {
            let bins = stage.get_usize("bins", 8).clamp(2, 32);
            let prior = stage.get_f64("prior", 10.0).max(0.0);
            let edges = quantile_edges_with_cache(x, bins, cache);
            let y_mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            let mut enc = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut sums = vec![0.0; bins];
                let mut cnts = vec![0.0; bins];
                for (r, row) in x.iter().enumerate() {
                    let b = lower_bound(&edges[j], row[j]).min(bins - 1);
                    sums[b] += y[r];
                    cnts[b] += 1.0;
                }
                enc.push(
                    (0..bins)
                        .map(|b| (sums[b] + prior * y_mean) / (cnts[b] + prior).max(1e-9))
                        .collect(),
                );
            }
            FittedOp::TargetBinEncode { edges, enc, append }
        }
        // NOTE: no `concat_raw` op (and no separate `winsor_moments`):
        // `winsor_moments` already aliases `winsor_clip` above, and a pure
        // passthrough whose only effect under append=true is duplicating the
        // input adds width without information (the 512-column cap makes that
        // strictly harmful). Deliberately skipped.
        "sigmoid" => FittedOp::Sigmoid { append },
        // Stateless hard indicator at a fixed threshold; no fit statistics.
        "step" => FittedOp::Step {
            t: stage.get_f64("t", 0.5),
            append,
        },
        "norm_row" => FittedOp::NormRow { append },
        "polar" => FittedOp::Polar { append },
        "lag" => FittedOp::Lag {
            k: stage.get_usize("k", 1).max(1),
            append,
        },
        "roll" => FittedOp::Roll {
            k: stage.get_usize("k", 1).max(1),
            append,
        },
        "fft_mag" => {
            // Frequencies 1..=k of the plain per-row DFT; k is capped at d/2
            // (at least 1) using the fit-time width, which the pipeline keeps
            // constant at apply time.
            let k = stage.get_usize("k", 8).clamp(1, (cols / 2).max(1));
            FittedOp::FftMag { k, append }
        }
        "row_spectral" => FittedOp::RowSpectral { append },
        "row_diff_stats" => FittedOp::RowDiffStats { append },
        "svd_project" => {
            let k = stage.get_usize("k", 8).clamp(1, 64).min(cols);
            let iters = stage.get_usize("iters", 20).clamp(1, 200);
            let seed = stage.get_u64("seed", 7);
            let (means, components) = svd_power_fit(x, k, iters, seed);
            FittedOp::SvdProject {
                means,
                components,
                append,
            }
        }
        "nystrom" => {
            let k = stage.get_usize("k", 16).clamp(1, 128).min(n);
            let gamma = stage.get_f64("gamma", 1.0).max(1e-6);
            let seed = stage.get_u64("seed", 7);
            // k distinct TRAIN rows as RBF landmarks via partial Fisher–Yates
            // over idx = [0..n). Draw order: for i = 0..k exactly one
            // next_u32(); j = i + (u32 % (n − i)); swap idx[i], idx[j].
            // Landmarks are stored in selection order idx[0..k].
            let mut rng = Pcg32::for_stage(seed, cols, k);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + (rng.next_u32() as usize) % (n - i);
                idx.swap(i, j);
            }
            let landmarks: Rows = idx[..k].iter().map(|&i| x[i].clone()).collect();
            FittedOp::Nystrom {
                landmarks,
                gamma,
                append,
            }
        }
        "symlog" => FittedOp::Symlog {
            c: stage.get_f64("c", 1.0).clamp(1e-9, 1e9),
            append,
        },
        "select_corr" => {
            let k = stage.get_usize("k", 32).clamp(1, 512).min(cols);
            let by_var = matches!(
                stage.params.get("by"),
                Some(crate::dsl::ParamValue::Str(s)) if s == "var"
            );
            // Same screening idiom as pair_prod: score descending, ties broken
            // by the lower column index, kept indices re-sorted ascending.
            let mut scored: Vec<(usize, f64)> = Vec::new();
            for j in 0..cols {
                let score = if by_var {
                    let mut s = 0.0;
                    for r in x {
                        s += r[j];
                    }
                    let m = s / n as f64;
                    let mut v = 0.0;
                    for r in x {
                        let d = r[j] - m;
                        v += d * d;
                    }
                    v / n as f64
                } else {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    crate::metrics::pearson(&col, y).abs()
                };
                scored.push((j, score));
            }
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut indices: Vec<usize> = scored[..k].iter().map(|&(j, _)| j).collect();
            indices.sort_unstable();
            FittedOp::SelectCorr { indices, append }
        }
        "spline" => {
            // Same knot placement as hinge: interior quantiles i/(k+1).
            let k = stage.get_usize("knots", 5).clamp(3, 12);
            cache.ensure_cols(x, cols);
            let knots = (0..cols)
                .map(|j| {
                    let s = cache.col_slice(j);
                    (1..=k)
                        .map(|i| quantile_sorted(s, i as f64 / (k + 1) as f64))
                        .collect()
                })
                .collect();
            FittedOp::Spline { knots, append }
        }
        "pair_ratio" => {
            let topk = stage.get_usize("topk", 8).clamp(1, 64);
            // Same target-guided screen as pair_prod: |pearson| descending,
            // ties to the lower index, kept indices re-sorted ascending.
            let m = cols.min(32);
            let mut scored: Vec<(usize, f64)> = (0..cols)
                .map(|j| {
                    let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    (j, crate::metrics::pearson(&col, y).abs())
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut screened: Vec<usize> = scored[..m].iter().map(|&(j, _)| j).collect();
            screened.sort_unstable();
            // Per screened column: population std -> denominator guard
            // eps = max(1e-12, (0.01 * std)^2).
            let mut guards = Vec::with_capacity(screened.len());
            for &j in &screened {
                let mut s = 0.0;
                for r in x {
                    s += r[j];
                }
                let mean = s / n as f64;
                let mut v = 0.0;
                for r in x {
                    let d = r[j] - mean;
                    v += d * d;
                }
                let e = 0.01 * (v / n as f64).sqrt();
                guards.push((e * e).max(1e-12));
            }
            // Score every ordered pair by |pearson(ratio, y)|.
            let mut candidates: Vec<(usize, usize, f64, f64)> = Vec::new();
            for (ai, &a) in screened.iter().enumerate() {
                for (bi, &b) in screened.iter().enumerate() {
                    if ai == bi {
                        continue;
                    }
                    let eps = guards[bi];
                    let ratio: Vec<f64> = x
                        .iter()
                        .map(|r| r[a] * r[b] / (r[b] * r[b] + eps))
                        .collect();
                    candidates.push((a, b, eps, crate::metrics::pearson(&ratio, y).abs()));
                }
            }
            candidates.sort_by(|l, r| r.3.total_cmp(&l.3).then(l.0.cmp(&r.0)).then(l.1.cmp(&r.1)));
            candidates.truncate(topk.min(candidates.len()));
            FittedOp::PairRatio {
                pairs: candidates.iter().map(|&(a, b, _, _)| (a, b)).collect(),
                eps: candidates.iter().map(|&(_, _, e, _)| e).collect(),
                append,
            }
        }
        "impute" => {
            // strategy encoding shared with the emitted twins: 0=mean,
            // 1=median (default), 2=const. `value` is the const-strategy fill;
            // it is passed but unused for mean/median.
            let strategy = match stage.params.get("strategy") {
                Some(crate::dsl::ParamValue::Str(s)) if s == "mean" => 0usize,
                Some(crate::dsl::ParamValue::Str(s)) if s == "const" => 2,
                _ => 1,
            };
            let value = stage.get_f64("value", 0.0).clamp(-1e12, 1e12);
            let add_indicator = matches!(
                stage.params.get("add_indicator"),
                Some(crate::dsl::ParamValue::Bool(true))
            ) || stage.get_usize("add_indicator", 0) == 1;
            let mut fill = vec![0.0; cols];
            let mut indicator_cols = Vec::new();
            for j in 0..cols {
                let mut vals: Vec<f64> = Vec::new();
                let mut missing = false;
                for r in x {
                    if r[j] == 0.0 {
                        missing = true;
                    } else {
                        vals.push(r[j]);
                    }
                }
                fill[j] = if vals.is_empty() {
                    // An all-zero column has no observed values to summarize.
                    0.0
                } else if strategy == 0 {
                    let mut s = 0.0;
                    for v in &vals {
                        s += *v;
                    }
                    s / vals.len() as f64
                } else if strategy == 2 {
                    value
                } else {
                    vals.sort_by(|a, b| a.total_cmp(b));
                    quantile_sorted(&vals, 0.5)
                };
                if missing && add_indicator && cols + indicator_cols.len() < crate::MAX_WIDTH {
                    indicator_cols.push(j);
                }
            }
            FittedOp::Impute {
                fill,
                indicator_cols,
                append,
            }
        }
        "map" | "derive" => {
            // Expression is parsed at FIT time; a parse failure fails the fit.
            let source = match stage.params.get("expr") {
                Some(crate::dsl::ParamValue::Str(s)) => s.clone(),
                _ => bail!("{} requires a string parameter expr=\"...\"", stage.op),
            };
            let expr = MathExpr::parse(&source)?;
            if stage.op == "map" {
                FittedOp::MapExpr {
                    source,
                    expr,
                    append,
                }
            } else {
                FittedOp::DeriveExpr {
                    source,
                    expr,
                    append,
                }
            }
        }
        other => bail!("unsupported battle op: {other}"),
    };
    Ok(op)
}
