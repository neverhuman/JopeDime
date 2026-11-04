//! Numeric `parameter.<operation>.<key>` feature family from `rank_prepare.py`.

use std::collections::BTreeMap;

use super::parse::{self, ParsedProgram};

const PARAMETER_KEYS: &[(&str, &str)] = &[
    ("bin_onehot", "append"),
    ("bin_onehot", "bins"),
    ("blom", "append"),
    ("cheb", "append"),
    ("cheb", "degree"),
    ("elasticnet", "alpha"),
    ("elasticnet", "l1_ratio"),
    ("hinge", "append"),
    ("hinge", "knots"),
    ("histgbm", "depth"),
    ("histgbm", "lr"),
    ("histgbm", "rounds"),
    ("ksmooth", "gamma"),
    ("ksmooth", "k"),
    ("oblique_hinge", "append"),
    ("oblique_hinge", "gamma"),
    ("oblique_hinge", "n"),
    ("pair_prod", "append"),
    ("pair_prod", "topk"),
    ("poly_interact", "append"),
    ("quantile_bins", "append"),
    ("quantile_bins", "bins"),
    ("rand_proj", "append"),
    ("rand_proj", "k"),
    ("rank", "append"),
    ("rff_cos", "append"),
    ("rff_cos", "gamma"),
    ("rff_cos", "n"),
    ("rff_sin", "append"),
    ("rff_sin", "gamma"),
    ("rff_sin", "n"),
    ("ridge", "lambda_frac"),
    ("robust_scale", "append"),
    ("row_stats", "append"),
    ("sigmoid", "append"),
    ("standardize", "append"),
    ("target_bin_encode", "append"),
    ("target_bin_encode", "bins"),
    ("winsor_clip", "append"),
    ("winsor_clip", "q"),
];

/// Extract exactly the 80 numeric-parameter aggregate coordinates owned here.
///
/// Raw step tokens are intentionally used so duplicate parameter tokens retain
/// their source order and multiplicity; a typed genome map would collapse them.
pub fn extract(program: &ParsedProgram<'_>) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::new();
    for &(operation, key) in PARAMETER_KEYS {
        features.insert(parameter_name(operation, key, "count"), 0.0);
        features.insert(parameter_name(operation, key, "sum"), 0.0);
    }

    for raw_step in program.raw_steps() {
        let mut tokens = parse::tokens(raw_step);
        let Some(operation) = tokens.next() else {
            continue;
        };
        for token in tokens {
            let Some((key, value)) = numeric_parameter_token(token) else {
                continue;
            };
            if !PARAMETER_KEYS.contains(&(operation, key)) {
                continue;
            }

            let sum_name = parameter_name(operation, key, "sum");
            let count_name = parameter_name(operation, key, "count");
            *features.get_mut(&sum_name).expect("owned parameter sum") += value;
            *features
                .get_mut(&count_name)
                .expect("owned parameter count") += 1.0;
        }
    }
    features
}

/// Parse one raw `key=value` token using Python-float and boolean fallback
/// semantics, then apply `sign(value) * log1p(abs(value))`. `seed` is ignored.
/// The helper is crate-visible for RNK-04-B's position-feature reuse.
pub(super) fn numeric_parameter_token(token: &str) -> Option<(&str, f64)> {
    let (key, raw_value) = token.split_once('=')?;
    if key == "seed" {
        return None;
    }
    let value = match python_float(raw_value) {
        Some(value) => value,
        None if raw_value == "true" => 1.0,
        None if raw_value == "false" => 0.0,
        None => return None,
    };
    let sign = if value < 0.0 {
        -1.0
    } else if value > 0.0 {
        1.0
    } else {
        0.0
    };
    Some((key, sign * value.abs().ln_1p()))
}

fn python_float(value: &str) -> Option<f64> {
    let unsigned = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let negative = value.starts_with('-');
    if unsigned.eq_ignore_ascii_case("inf") || unsigned.eq_ignore_ascii_case("infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if unsigned.eq_ignore_ascii_case("nan") {
        return Some(if negative { -f64::NAN } else { f64::NAN });
    }

    let bytes = value.as_bytes();
    if bytes.iter().enumerate().any(|(index, byte)| {
        *byte == b'_'
            && !(index > 0
                && index + 1 < bytes.len()
                && bytes[index - 1].is_ascii_digit()
                && bytes[index + 1].is_ascii_digit())
    }) {
        return None;
    }
    value.replace('_', "").parse().ok()
}

fn parameter_name(operation: &str, key: &str, statistic: &str) -> String {
    format!("parameter.{operation}.{key}.{statistic}")
}

// Sparse, row-free family projections of full-reference-goldens.jsonl.
// Generated from pinned rank_prepare.py (SHA-256
// 534169af0443122600853ef9e96d734d570cd9d77c2a3d21e7b6456b4b19ab40) and the
// frozen ordered feature names (SHA-256
// c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962).
#[cfg(test)]
mod tests;
