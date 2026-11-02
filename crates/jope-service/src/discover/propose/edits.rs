use super::*;

pub(super) fn index(rng: &mut ChaCha12Rng, length: usize) -> usize {
    debug_assert!(length > 0);
    let bound = length as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

pub(super) fn delete_stage(transforms: &mut Vec<Stage>, rng: &mut ChaCha12Rng) -> Option<usize> {
    if transforms.len() < 2 {
        return None;
    }
    let removed = index(rng, transforms.len());
    transforms.remove(removed);
    Some(removed)
}

// A cyclic successor would double its weight when the preceding head is
// known. A uniform permutation preserves the preferred pool's conditional
// weights. Give each typed slot a separate non-main stream; the shared RNG
// position and first draw stay unchanged, without reusing future main bits.
pub(super) fn head_retry_order(
    first: usize,
    length: usize,
    step: usize,
    rng: &ChaCha12Rng,
) -> Vec<usize> {
    let mut remaining: Vec<_> = (0..length).filter(|&i| i != first).collect();
    let mut fork = rng.clone();
    fork.set_stream(rng.get_stream().wrapping_add(step as u64 + 1));
    fork.set_word_pos(0);
    for i in (1..remaining.len()).rev() {
        let j = index(&mut fork, i + 1);
        remaining.swap(i, j);
    }
    let mut order = vec![first];
    order.extend(remaining);
    order
}

pub(super) fn uniform(rng: &mut ChaCha12Rng, low: f64, high: f64) -> f64 {
    let unit = (rng.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
    low + (high - low) * unit
}

pub(super) fn mutate_parameters(stage: &mut Stage, rng: &mut ChaCha12Rng) {
    for (key, value) in &mut stage.params {
        if key == "seed" {
            continue;
        }
        let forced_float = matches!(key.as_str(), "q" | "lr" | "gamma" | "lambda_frac");
        match value.clone() {
            ParamValue::Float(v) if v.is_finite() => {
                *value = ParamValue::Float(jitter_float(key, v, rng));
            }
            ParamValue::Int(v) if forced_float => {
                *value = ParamValue::Float(jitter_float(key, v as f64, rng));
            }
            ParamValue::Int(v) => {
                let next = match key.as_str() {
                    "depth" | "degree" | "knots" => {
                        let upper = match key.as_str() {
                            "depth" => 6,
                            "degree" => 7,
                            _ => 8,
                        };
                        v.saturating_add(if index(rng, 2) == 0 { -1 } else { 1 })
                            .clamp(2, upper)
                    }
                    "rounds" => ((v as f64 * uniform(rng, 0.75, 1.30)).round_ties_even() as i64)
                        .clamp(40, 280),
                    "bins" | "topk" | "k" | "n" => {
                        ((v as f64 * uniform(rng, 0.75, 1.35)).round_ties_even() as i64)
                            .clamp(2, 128)
                    }
                    _ => v,
                };
                *value = ParamValue::Int(next);
            }
            _ => {}
        }
    }
}

pub(super) fn jitter_float(key: &str, value: f64, rng: &mut ChaCha12Rng) -> f64 {
    let next = value * uniform(rng, -0.30, 0.30).exp();
    let bounded = match key {
        "q" => next.clamp(0.01, 0.25),
        "lr" => next.clamp(0.015, 0.30),
        "gamma" => next.clamp(0.05, 8.0),
        "lambda_frac" => next.clamp(0.0001, 1.0),
        _ => next,
    };
    (bounded * 100_000_000.0).round_ties_even() / 100_000_000.0
}

pub(super) fn assemble(transforms: Vec<Stage>, head: Stage) -> String {
    let mut stages = transforms;
    stages.push(head);
    GenomeIr::Pipeline(DslExpr { stages }).canonical()
}
