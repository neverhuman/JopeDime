use super::fit::{normalize_category, parse_numeric_token};
use super::{ColumnEncoding, Encoders, NumericScaler};
use crate::data::Matrix;
use crate::hashing::mix64;
use crate::ingest::{ColKind, Frame, RawColumn};
use std::collections::HashMap;

fn apply_numeric(tok: &str, sc: &NumericScaler) -> f32 {
    let Some(v) = parse_numeric_token(tok) else {
        return 0.0;
    };
    if sc.constant {
        0.0
    } else {
        ((v - sc.min) / (sc.max - sc.min)).clamp(0.0, 1.0)
    }
}

fn stable_bucket(tok: &str, buckets: usize) -> usize {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in tok.as_bytes() {
        h = mix64(h ^ (*b as u64));
    }
    (h % buckets.max(1) as u64) as usize
}

fn push_encoded(tok: &str, enc: &ColumnEncoding, out: &mut Vec<f32>) {
    match enc {
        ColumnEncoding::Numeric { scaler } => out.push(apply_numeric(tok, scaler)),
        ColumnEncoding::Ordinal { dict } => {
            let v = normalize_category(tok)
                .and_then(|t| dict.map.get(&t).copied())
                .map(|c| {
                    if dict.map.len() <= 1 {
                        0.0
                    } else {
                        c as f32 / (dict.map.len() as f32 - 1.0)
                    }
                })
                .unwrap_or(0.0);
            out.push(v.clamp(0.0, 1.0));
        }
        ColumnEncoding::Frequency { dict } => {
            let v = normalize_category(tok)
                .and_then(|t| dict.map.get(&t).copied())
                .and_then(|c| dict.freq.get(c as usize).copied())
                .unwrap_or(0.0);
            out.push(v.clamp(0.0, 1.0));
        }
        ColumnEncoding::OneHot { levels } => {
            let width = levels.len() + 2;
            let start = out.len();
            out.resize(start + width, 0.0);
            match normalize_category(tok) {
                None => out[start] = 1.0,
                Some(t) => {
                    if let Ok(pos) = levels.binary_search(&t) {
                        out[start + 1 + pos] = 1.0;
                    } else {
                        out[start + width - 1] = 1.0;
                    }
                }
            }
        }
        ColumnEncoding::Hash { buckets } => {
            let width = *buckets + 1;
            let start = out.len();
            out.resize(start + width, 0.0);
            match normalize_category(tok) {
                None => out[start] = 1.0,
                Some(t) => out[start + 1 + stable_bucket(&t, *buckets)] = 1.0,
            }
        }
    }
}

/// Encode a frame to a column-major numeric `Matrix` with a train-fit preprocessing plan. Returns
/// (matrix, feature names, output indices of categorical-derived columns).
pub fn transform(frame: &Frame, e: &Encoders) -> (Matrix, Vec<String>, Vec<usize>) {
    let n = frame.n;
    let p = e.output_width();
    let names = e.output_names();
    let mut rows: Vec<Vec<f32>> = vec![Vec::with_capacity(p); n];
    let mut cat_idx = Vec::new();

    let by_name: HashMap<&str, &RawColumn> =
        frame.feats.iter().map(|c| (c.name.as_str(), c)).collect();
    let mut offset = 0usize;
    for plan in &e.columns {
        if plan.kind == ColKind::Categorical {
            cat_idx.extend(offset..offset + plan.output_names.len());
        }
        let width = plan.output_names.len();
        let col = by_name.get(plan.input_name.as_str());
        for (i, row) in rows.iter_mut().enumerate() {
            let tok = col
                .and_then(|c| c.tokens.get(i))
                .map(|s| s.as_str())
                .unwrap_or("");
            push_encoded(tok, &plan.encoding, row);
        }
        offset += width;
    }
    let mut flat = Vec::with_capacity(n * p);
    for j in 0..p {
        for row in &rows {
            let v = row.get(j).copied().unwrap_or(0.0);
            flat.push(if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            });
        }
    }
    (Matrix::new(flat, n, p), names, cat_idx)
}
