#![cfg(feature = "neural")]
//! W6-T02 objective fixtures (CPU only, tiny, deterministic).
//!
//! Every expected loss and gradient below is computed independently in plain
//! f64 arithmetic from the term definitions in
//! `docs/internal/core-ip/work-orders/evidence/W06-objective-specification.md`.
//! No code under test is used to form an expectation. The accepted lineage-B
//! coefficients are restated here as literals on purpose.
//!
//! These fixtures prove source arithmetic only. They are not training,
//! GPU/BF16, multi-rank or live-corpus qualification.

use jope_parallel::neural::{ranking, ranking_pairwise, sequence_nll, AdamW, Config, Policy};
use jope_parallel::training::objective::{
    self, composite, Outputs, RowLabels, COEFFICIENTS, INACTIVE_PARAMETER_PREFIXES,
};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tch::{nn, Device, Kind, Tensor};

/// Accepted coefficients: seq_nll, reward_bt, reward_listwise, runtime_huber, ablation_bt.
const ACCEPTED: [f64; 5] = [1.0, 0.25, 0.10, 0.10, 0.05];
const SMOOTHING: f64 = 0.05;
const LISTWISE_TEMPERATURE: f64 = 0.25;

// ---------------------------------------------------------------------------
// Independent f64 references
// ---------------------------------------------------------------------------

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// §3.1: grammar-masked, 0.05-smoothed token NLL; per-row mean over supervised
/// positions (denominator max(1, n)); rows combined by `w / max(1e-8, Σw)`.
/// Returns the loss and the gradient over `logits[b][t1][v]`.
fn reference_sequence(
    logits: &[f64],
    targets: &[usize],
    legal: &[Vec<usize>],
    supervised: &[bool],
    weights: &[f64],
    (b, t1, v): (usize, usize, usize),
) -> (f64, Vec<f64>) {
    let clamped = weights.iter().map(|w| w.max(0.0)).collect::<Vec<_>>();
    let total = clamped.iter().sum::<f64>().max(1e-8);
    let mut loss = 0.0;
    let mut grad = vec![0.0; b * t1 * v];
    for i in 0..b {
        let n = (0..t1).filter(|&j| supervised[i * t1 + j]).count() as f64;
        let denominator = n.max(1.0);
        let mut row = 0.0;
        for j in 0..t1 {
            if !supervised[i * t1 + j] {
                continue;
            }
            let set = &legal[i * t1 + j];
            let z = |k: usize| logits[(i * t1 + j) * v + k];
            let peak = set.iter().map(|&k| z(k)).fold(f64::NEG_INFINITY, f64::max);
            let lse = peak + set.iter().map(|&k| (z(k) - peak).exp()).sum::<f64>().ln();
            let x = targets[i * t1 + j];
            let nll = lse - z(x);
            let uniform = set.iter().map(|&k| lse - z(k)).sum::<f64>() / set.len() as f64;
            row += (1.0 - SMOOTHING) * nll + SMOOTHING * uniform;
            for &k in set {
                let pi = (z(k) - lse).exp();
                let target = if k == x { 1.0 - SMOOTHING } else { 0.0 };
                grad[(i * t1 + j) * v + k] +=
                    clamped[i] / total / denominator * (pi - target - SMOOTHING / set.len() as f64);
            }
        }
        loss += clamped[i] * row / denominator;
    }
    (loss / total, grad)
}

/// §3.2: Bradley–Terry over all unordered pairs sharing a group key; ties and
/// nonfinite values skipped; weight |Δv|·(2 if max p ≥ 0.9 else 1); reduction
/// Σω·softplus(s_l − s_w)/Σω over the whole microbatch; empty support → 0.
fn reference_bt(s: &[f64], values: &[f64], p: &[f64], keys: &[String]) -> (f64, Vec<f64>) {
    let n = s.len();
    let mut pairs = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if keys[a] != keys[b]
                || !values[a].is_finite()
                || !values[b].is_finite()
                || values[a] == values[b]
            {
                continue;
            }
            let (w, l) = if values[a] > values[b] {
                (a, b)
            } else {
                (b, a)
            };
            let boost = if p[a].max(p[b]) >= 0.9 { 2.0 } else { 1.0 };
            pairs.push((w, l, (values[a] - values[b]).abs() * boost));
        }
    }
    let mut grad = vec![0.0; n];
    if pairs.is_empty() {
        return (0.0, grad);
    }
    let total = pairs.iter().map(|x| x.2).sum::<f64>();
    let mut loss = 0.0;
    for &(w, l, omega) in &pairs {
        let d = s[l] - s[w];
        loss += omega * softplus(d);
        grad[l] += omega / total * sigmoid(d);
        grad[w] -= omega / total * sigmoid(d);
    }
    (loss / total, grad)
}

/// §3.3: per group with ≥3 members and all-finite p, cross-entropy of
/// softmax(s) against softmax(p/0.25); mean over eligible groups; empty → 0.
fn reference_listwise(s: &[f64], p: &[f64], keys: &[String]) -> (f64, Vec<f64>) {
    let mut groups = BTreeMap::<&str, Vec<usize>>::new();
    for (i, key) in keys.iter().enumerate() {
        groups.entry(key).or_default().push(i);
    }
    let eligible = groups
        .values()
        .filter(|ids| ids.len() >= 3 && ids.iter().all(|&i| p[i].is_finite()))
        .collect::<Vec<_>>();
    let mut grad = vec![0.0; s.len()];
    if eligible.is_empty() {
        return (0.0, grad);
    }
    let count = eligible.len() as f64;
    let mut loss = 0.0;
    for ids in eligible {
        let scaled = ids
            .iter()
            .map(|&i| p[i] / LISTWISE_TEMPERATURE)
            .collect::<Vec<_>>();
        let top = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z = scaled.iter().map(|x| (x - top).exp()).sum::<f64>();
        let q = scaled
            .iter()
            .map(|x| (x - top).exp() / z)
            .collect::<Vec<_>>();
        let peak = ids.iter().map(|&i| s[i]).fold(f64::NEG_INFINITY, f64::max);
        let lse = peak + ids.iter().map(|&i| (s[i] - peak).exp()).sum::<f64>().ln();
        for (k, &i) in ids.iter().enumerate() {
            loss -= q[k] * (s[i] - lse) / count;
            grad[i] += ((s[i] - lse).exp() - q[k]) / count;
        }
    }
    (loss, grad)
}

/// §3.4: SmoothL1(β=1) of log1p(prediction) − label, mean over every row.
fn reference_runtime(prediction: &[f64], label: &[f64]) -> (f64, Vec<f64>) {
    let n = prediction.len() as f64;
    let mut loss = 0.0;
    let mut grad = Vec::with_capacity(prediction.len());
    for (&yhat, &y) in prediction.iter().zip(label) {
        let d = yhat.ln_1p() - y;
        let (h, dh) = if d.abs() < 1.0 {
            (0.5 * d * d, d)
        } else {
            (d.abs() - 0.5, d.signum())
        };
        loss += h / n;
        grad.push(dh / (1.0 + yhat) / n);
    }
    (loss, grad)
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn leaf(values: &[f64], shape: &[i64]) -> Tensor {
    Tensor::from_slice(&values.iter().map(|&x| x as f32).collect::<Vec<_>>())
        .reshape(shape)
        .set_requires_grad(true)
}

fn host(t: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(t.to_kind(Kind::Double).flatten(0, -1)).unwrap()
}

fn grad_of(t: &Tensor) -> Vec<f64> {
    let g = t.grad();
    assert!(g.defined(), "gradient undefined");
    host(&g)
}

fn legal_tensor(sets: &[Vec<usize>], shape: [i64; 3]) -> Tensor {
    let v = shape[2] as usize;
    let mut mask = vec![0u8; sets.len() * v];
    for (i, set) in sets.iter().enumerate() {
        for &k in set {
            mask[i * v + k] = 1;
        }
    }
    Tensor::from_slice(&mask).reshape(shape).to_kind(Kind::Bool)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{what}: actual {actual:.9e} expected {expected:.9e}"
    );
}

fn assert_all_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{what}[{i}]: actual {a:.9e} expected {e:.9e}"
        );
    }
}

fn assert_exact_zero(values: &[f64], what: &str) {
    assert!(
        values.iter().all(|x| *x == 0.0),
        "{what}: expected exact zeros, max |x| = {:e}",
        values.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    );
}

/// Group keys per the specification: `(context, row / 8)`.
fn keys(contexts: &[usize]) -> Vec<String> {
    contexts
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{c}:{}", i / 8))
        .collect()
}

// ---------------------------------------------------------------------------
// A 16-row microbatch: two comparisons of eight rows from the same context
// ---------------------------------------------------------------------------

struct Batch {
    b: usize,
    t: usize,
    v: usize,
    tokens: Vec<i64>,
    /// Legal set for target `tokens[i][j + 1]`, `j` in `0..t-1`.
    legal: Vec<Vec<usize>>,
    /// Decoder logits `[b][t + 1][v]`: position 0 is the context slot.
    logits: Vec<f64>,
    rewards: Vec<f64>,
    runtime: Vec<f64>,
    stability: Vec<f64>,
    ablation: Vec<f64>,
    labels: Vec<RowLabels>,
}

fn batch16() -> Batch {
    let (b, t, v) = (16usize, 5usize, 6usize);
    let mut tokens = vec![0i64; b * t];
    for i in 0..b {
        let body = 1 + i % 3;
        tokens[i * t] = 1;
        for k in 0..body {
            tokens[i * t + 1 + k] = 3 + ((i + k) % 3) as i64;
        }
        tokens[i * t + 1 + body] = 2;
    }
    let mut legal = Vec::with_capacity(b * (t - 1));
    for i in 0..b {
        for j in 0..t - 1 {
            let x = tokens[i * t + j + 1] as usize;
            let mut set = match x {
                0 => vec![0],
                2 => vec![2],
                _ => vec![x, 3, 4],
            };
            if x > 2 && (i + j) % 2 == 0 {
                set.push(5);
            }
            set.sort();
            set.dedup();
            legal.push(set);
        }
    }
    let logits = (0..b * (t + 1) * v)
        .map(|n| {
            let (i, p, k) = (n / ((t + 1) * v), (n / v) % (t + 1), n % v);
            ((i * 7 + p * 3 + k * 5) % 11) as f64 * 0.3 - 1.2 + i as f64 * 0.01
        })
        .collect();
    let rewards = (0..b * 4)
        .map(|n| ((n / 4 * 5 + n % 4 * 3) % 7) as f64 * 0.21 - 0.6)
        .collect();
    let runtime = (0..b).map(|i| 0.05 + (i % 5) as f64 * 0.7).collect();
    let stability = (0..b).map(|i| i as f64 * 0.1 - 0.5).collect();
    let ablation = (0..b).map(|i| ((i * 3) % 5) as f64 * 0.17 - 0.3).collect();
    let values = [
        0.95f32, 0.8, 0.8, 0.62, 0.5, 0.41, 0.3, 0.2, // comparison 0, one tie
        0.7, 0.69, 0.55, 0.55, 0.4, 0.33, 0.21, 0.1, // comparison 1, one tie
    ];
    let percentiles = [
        1.0f32,
        6.0 / 7.0,
        5.0 / 7.0,
        4.0 / 7.0,
        3.0 / 7.0,
        2.0 / 7.0,
        1.0 / 7.0,
        0.0,
        // below the 0.9 boost threshold throughout
        0.85,
        0.8,
        0.7,
        0.6,
        0.45,
        0.3,
        0.15,
        0.05,
    ];
    let seconds = [0.4f64, 0.1, 2.0, 0.0, 5.0, 0.3, 1.2, 0.7];
    let labels = (0..b)
        .map(|i| RowLabels {
            context: 7,
            value: values[i],
            percentile: percentiles[i],
            runtime: seconds[i % 8].ln_1p() as f32,
        })
        .collect();
    Batch {
        b,
        t,
        v,
        tokens,
        legal,
        logits,
        rewards,
        runtime,
        stability,
        ablation,
        labels,
    }
}

struct Expected {
    loss: f64,
    terms: [f64; 5],
    logits: Vec<f64>,
    rewards: Vec<f64>,
    runtime: Vec<f64>,
    ablation: Vec<f64>,
}

/// Independent composite over rows `lo..hi` treated as one microbatch.
fn reference_composite(batch: &Batch, lo: usize, hi: usize) -> Expected {
    let (t, v) = (batch.t, batch.v);
    let b = hi - lo;
    let rows = &batch.labels[lo..hi];
    let mut targets = Vec::new();
    let mut supervised = Vec::new();
    let mut legal = Vec::new();
    let mut logits = Vec::new();
    for i in lo..hi {
        for j in 0..t - 1 {
            let x = batch.tokens[i * t + j + 1];
            targets.push(x as usize);
            supervised.push(x != 0);
            legal.push(batch.legal[i * (t - 1) + j].clone());
            // Decoder position j + 1 has seen the context and x_0..x_j.
            let at = (i * (t + 1) + j + 1) * v;
            logits.extend_from_slice(&batch.logits[at..at + v]);
        }
    }
    let weights = rows
        .iter()
        .map(|r| 0.05 + (r.percentile as f64).powi(4))
        .collect::<Vec<_>>();
    let (seq, g_seq) = reference_sequence(
        &logits,
        &targets,
        &legal,
        &supervised,
        &weights,
        (b, t - 1, v),
    );
    let s = (lo..hi)
        .map(|i| batch.rewards[i * 4..i * 4 + 4].iter().sum::<f64>() / 4.0)
        .collect::<Vec<_>>();
    let values = rows.iter().map(|r| r.value as f64).collect::<Vec<_>>();
    let p = rows.iter().map(|r| r.percentile as f64).collect::<Vec<_>>();
    let contexts = rows.iter().map(|r| r.context).collect::<Vec<_>>();
    let k = keys(&contexts);
    let (bt, g_bt) = reference_bt(&s, &values, &p, &k);
    let (list, g_list) = reference_listwise(&s, &p, &k);
    let labels = rows.iter().map(|r| r.runtime as f64).collect::<Vec<_>>();
    let (rt, g_rt) = reference_runtime(&batch.runtime[lo..hi], &labels);
    let (abt, g_abt) = reference_bt(&batch.ablation[lo..hi], &values, &p, &k);
    let terms = [seq, bt, list, rt, abt];
    let loss = terms.iter().zip(ACCEPTED).map(|(x, c)| x * c).sum();
    // Position 0 (the context slot, which would predict BOS) and the final
    // position are never supervised.
    let mut g_logits = vec![0.0; b * (t + 1) * v];
    for i in 0..b {
        for j in 0..t - 1 {
            for k in 0..v {
                g_logits[(i * (t + 1) + j + 1) * v + k] =
                    ACCEPTED[0] * g_seq[(i * (t - 1) + j) * v + k];
            }
        }
    }
    let g_rewards = (0..b * 4)
        .map(|n| (ACCEPTED[1] * g_bt[n / 4] + ACCEPTED[2] * g_list[n / 4]) / 4.0)
        .collect();
    Expected {
        loss,
        terms,
        logits: g_logits,
        rewards: g_rewards,
        runtime: g_rt.iter().map(|g| ACCEPTED[3] * g).collect(),
        ablation: g_abt.iter().map(|g| ACCEPTED[4] * g).collect(),
    }
}

struct Leaves {
    logits: Tensor,
    rewards: Tensor,
    runtime: Tensor,
    stability: Tensor,
    ablation: Tensor,
    log_vars: Vec<Tensor>,
    tokens: Tensor,
    pad: Tensor,
    legal: Tensor,
}

fn leaves(batch: &Batch) -> Leaves {
    let (b, t, v) = (batch.b as i64, batch.t as i64, batch.v as i64);
    let tokens = Tensor::from_slice(&batch.tokens).reshape([b, t]);
    Leaves {
        logits: leaf(&batch.logits, &[b, t + 1, v]),
        rewards: leaf(&batch.rewards, &[b, 4]),
        runtime: leaf(&batch.runtime, &[b]),
        stability: leaf(&batch.stability, &[b]),
        ablation: leaf(&batch.ablation, &[b]),
        log_vars: (0..7).map(|_| leaf(&[0.0], &[])).collect(),
        pad: tokens.ne(0),
        tokens,
        legal: legal_tensor(&batch.legal, [b, t - 1, v]),
    }
}

fn call_composite(
    outputs: [&Tensor; 5],
    l: &Leaves,
    labels: &[RowLabels],
    lo: i64,
    n: i64,
) -> anyhow::Result<(Tensor, [f64; 5])> {
    composite(
        &Outputs {
            logits: outputs[0],
            rewards: outputs[1],
            runtime: outputs[2],
            stability: outputs[3],
            ablation: outputs[4],
        },
        l.log_vars.iter(),
        &l.tokens.narrow(0, lo, n),
        &l.pad.narrow(0, lo, n),
        &l.legal.narrow(0, lo, n),
        labels,
    )
}

fn run_composite(
    l: &Leaves,
    labels: &[RowLabels],
    lo: i64,
    n: i64,
) -> anyhow::Result<(Tensor, [f64; 5])> {
    let o =
        [&l.logits, &l.rewards, &l.runtime, &l.stability, &l.ablation].map(|t| t.narrow(0, lo, n));
    call_composite(
        [&o[0], &o[1], &o[2], &o[3], &o[4]],
        l,
        &labels[lo as usize..(lo + n) as usize],
        lo,
        n,
    )
}

// ---------------------------------------------------------------------------
// Per-term fixtures
// ---------------------------------------------------------------------------

#[test]
fn accepted_coefficients_and_telemetry_weights_are_the_declared_literals() {
    let c = COEFFICIENTS;
    assert_eq!(
        [
            c.sequence_nll,
            c.ranking_bt,
            c.ranking_listwise,
            c.runtime_huber,
            c.contextfree_bt
        ],
        ACCEPTED
    );
    assert_eq!((c.stability, c.loss_log_vars), (0.0, 0.0));
    // Telemetry previously multiplied the f32 rank means by these f32 literals.
    let raw = [1.387_687_2f32, 0.507_536_4, 2.218_858, 0.069_544_2, 0.431_2];
    let w = objective::weighted_losses(&raw);
    for (name, literal) in [
        ("sequence_nll", raw[0]),
        ("ranking_bt", raw[1] * 0.25),
        ("ranking_listwise", raw[2] * 0.10),
        ("runtime_huber", raw[3] * 0.10),
        ("contextfree_bt", raw[4] * 0.05),
    ] {
        assert_eq!(w[name].as_f64().unwrap() as f32, literal, "{name}");
    }
}

#[test]
fn sequence_term_matches_independent_smoothed_masked_weighted_mean() {
    let (b, t1, v) = (3usize, 4usize, 6usize);
    let legal: Vec<Vec<usize>> = vec![
        vec![3, 4, 5],
        vec![2], // forced EOS: zero loss, still counted
        vec![0],
        vec![0],
        vec![1, 2, 3, 4, 5],
        vec![3, 4],
        vec![2, 3, 5],
        vec![2],
        vec![2, 3],
        vec![4, 5],
        vec![2],
        vec![0],
    ];
    let targets = [4usize, 2, 0, 0, 5, 3, 2, 2, 3, 5, 2, 0];
    let supervised = targets.iter().map(|&x| x != 0).collect::<Vec<_>>();
    let logits = (0..b * t1 * v)
        .map(|n| {
            let (i, j, k) = (n / (t1 * v), (n / v) % t1, n % v);
            ((i * 7 + j * 3 + k * 5) % 11) as f64 * 0.3 - 1.2
        })
        .collect::<Vec<_>>();
    let weights = [1.0f64, 0.5, 0.0]
        .iter()
        .map(|p| 0.05 + p.powi(4))
        .collect::<Vec<_>>();
    let (expected, expected_grad) =
        reference_sequence(&logits, &targets, &legal, &supervised, &weights, (b, t1, v));
    let z = leaf(&logits, &[b as i64, t1 as i64, v as i64]);
    let x = Tensor::from_slice(&targets.map(|x| x as i64)).reshape([b as i64, t1 as i64]);
    let loss = sequence_nll(
        &z,
        &x,
        &legal_tensor(&legal, [b as i64, t1 as i64, v as i64]),
        &x.ne(0),
        &Tensor::from_slice(&weights.iter().map(|&w| w as f32).collect::<Vec<_>>()),
    )
    .unwrap();
    assert_close(loss.double_value(&[]), expected, 2e-6, "seq_nll");
    loss.backward();
    let g = grad_of(&z);
    assert_all_close(&g, &expected_grad, 2e-5, "seq_nll gradient");
    // Illegal and PAD logits receive exactly zero gradient.
    for i in 0..b * t1 {
        for k in 0..v {
            if !supervised[i] || !legal[i].contains(&k) {
                assert_eq!(g[i * v + k], 0.0, "masked logit ({i},{k})");
            }
        }
    }
    // Declared G7 normalization: the forced singleton position (row 0, j=1)
    // adds zero loss and gradient but counts in row 0's denominator.
    for k in 0..v {
        assert!(g[v + k].abs() < 1e-7, "forced position gradient");
    }
    let without_forced = {
        let mut s = supervised.clone();
        s[1] = false;
        reference_sequence(&logits, &targets, &legal, &s, &weights, (b, t1, v)).0
    };
    assert!(
        without_forced > expected,
        "excluding a forced token would raise row 0's mean: {without_forced} vs {expected}"
    );
}

#[test]
fn sequence_row_without_supervised_positions_is_rejected() {
    // G8: a row with no supervised position kept its weight in the denominator
    // with a zero numerator, silently diluting every other row.
    let logits = leaf(
        &(0..2 * 3 * 4).map(|n| n as f64 * 0.1).collect::<Vec<_>>(),
        &[2, 3, 4],
    );
    let weights = Tensor::from_slice(&[1.0f32, 1.0]);
    let targets = Tensor::from_slice(&[3i64, 2, 0, 0, 0, 0]).reshape([2, 3]);
    let legal = legal_tensor(
        &[vec![3], vec![2], vec![0], vec![0], vec![0], vec![0]],
        [2, 3, 4],
    );
    let error = sequence_nll(&logits, &targets, &legal, &targets.ne(0), &weights)
        .expect_err("an all-PAD supervised row must be rejected, not diluted");
    assert!(error.to_string().contains("supervised"), "{error}");
    // Positive control: the same row with one supervised position is accepted.
    let targets = Tensor::from_slice(&[3i64, 2, 0, 3, 0, 0]).reshape([2, 3]);
    let legal = legal_tensor(
        &[vec![3], vec![2], vec![0], vec![3], vec![0], vec![0]],
        [2, 3, 4],
    );
    let accepted = sequence_nll(&logits, &targets, &legal, &targets.ne(0), &weights).unwrap();
    assert!(accepted.double_value(&[]).is_finite());
}

#[test]
fn sequence_supervision_masks_fail_explicitly() {
    let logits = leaf(&[0.0; 3 * 5], &[1, 3, 5]);
    let targets = Tensor::from_slice(&[3i64, 2, 0]).reshape([1, 3]);
    let pad = targets.ne(0);
    let weights = Tensor::from_slice(&[1.0f32]);
    // A supervised target outside its legal set is an error, not a skip.
    let illegal = legal_tensor(&[vec![4], vec![2], vec![0]], [1, 3, 5]);
    let e = sequence_nll(&logits, &targets, &illegal, &pad, &weights).unwrap_err();
    assert!(e.to_string().contains("illegal supervision token"), "{e}");
    // An empty legal set is an error at any position, including PAD.
    let empty = legal_tensor(&[vec![3], vec![2], vec![]], [1, 3, 5]);
    let e = sequence_nll(&logits, &targets, &empty, &pad, &weights).unwrap_err();
    assert!(e.to_string().contains("empty legal token set"), "{e}");
    // A PAD position may carry any non-empty legal set; it is not supervised.
    let pad_other = legal_tensor(&[vec![3], vec![2], vec![4]], [1, 3, 5]);
    let accepted = sequence_nll(&logits, &targets, &pad_other, &pad, &weights).unwrap();
    assert!(accepted.double_value(&[]).is_finite());
}

#[test]
fn bradley_terry_hand_example_ties_boost_and_direction() {
    // One group: champion (p=1.0) against two tied comparators.
    let groups = vec!["c:0".to_string(); 3];
    let reward = leaf(&[0.2, 0.4, -0.1], &[3]);
    let (bt, _) = ranking(&reward, &[0.9, 0.5, 0.5], &[1.0, 0.5, 0.0], &groups).unwrap();
    // Pairs (0,1),(0,2) carry equal boosted weight 0.8; (1,2) is a skipped tie.
    // (softplus(0.2) + softplus(-0.3)) / 2, computed by hand.
    assert_close(bt.double_value(&[]), 0.676_247_057, 1e-6, "hand BT");
    bt.backward();
    assert_all_close(
        &grad_of(&reward),
        &[-0.487_695_741, 0.274_916_999, 0.212_778_742],
        2e-6,
        "hand BT gradient",
    );
    // Pair reversal flips the intended gradient direction.
    let s = [0.1f64, 0.3];
    for (values, winner) in [([0.8f64, 0.4], 0usize), ([0.4, 0.8], 1usize)] {
        let reward = leaf(&s, &[2]);
        let (bt, _) = ranking(&reward, &values, &[0.5, 0.0], &groups[..2]).unwrap();
        let loser = 1 - winner;
        assert_close(
            bt.double_value(&[]),
            softplus(s[loser] - s[winner]),
            1e-6,
            "pair",
        );
        bt.backward();
        let g = grad_of(&reward);
        let magnitude = sigmoid(s[loser] - s[winner]);
        assert_close(g[winner], -magnitude, 1e-6, "winner pushed up");
        assert_close(g[loser], magnitude, 1e-6, "loser pushed down");
    }
}

#[test]
fn bradley_terry_and_listwise_match_independent_references() {
    // Four groups in one microbatch: boosted, unboosted, one with a nonfinite
    // value and a nonfinite percentile, and a two-member group.
    let keys = [
        "a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "d", "d",
    ]
    .map(String::from)
    .to_vec();
    let s = (0..keys.len())
        .map(|i| ((i * 13 % 23) as f64 - 11.0) * 0.17)
        .collect::<Vec<_>>();
    let values = [
        0.9,
        0.7,
        0.7,
        0.4,
        0.2,
        0.8,
        0.6,
        0.5,
        0.1,
        0.6,
        f64::NAN,
        0.3,
        0.2,
        0.5,
        0.4,
    ];
    let p = [
        1.0f32,
        0.75,
        0.5,
        0.25,
        0.0,
        0.85,
        0.6,
        0.3,
        0.0,
        0.95,
        0.5,
        f32::NAN,
        0.0,
        0.6,
        0.1,
    ];
    let p64 = p.map(|x| x as f64);
    let (expected_bt, g_bt) = reference_bt(&s, &values, &p64, &keys);
    let (expected_list, g_list) = reference_listwise(&s, &p64, &keys);
    let reward = leaf(&s, &[s.len() as i64]);
    let (bt, list) = ranking(&reward, &values, &p, &keys).unwrap();
    assert_close(bt.double_value(&[]), expected_bt, 2e-6, "BT");
    // Only groups "a" and "b" are listwise-eligible ("c" has a NaN
    // percentile, "d" has two members).
    assert_close(list.double_value(&[]), expected_list, 2e-6, "listwise");
    (&bt + &list).backward();
    let expected = g_bt
        .iter()
        .zip(&g_list)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    assert_all_close(&grad_of(&reward), &expected, 2e-5, "BT+listwise gradient");
    // Ablation BT: identical pairs, weights and reduction; no listwise term.
    let ablation = leaf(&s, &[s.len() as i64]);
    let abt = ranking_pairwise(&ablation, &values, &p, &keys).unwrap();
    assert_close(abt.double_value(&[]), expected_bt, 2e-6, "ablation BT");
    abt.backward();
    assert_all_close(&grad_of(&ablation), &g_bt, 2e-5, "ablation BT gradient");
}

fn ranking_with_gradient(
    s: &[f64],
    values: &[f64],
    p: &[f32],
    keys: &[String],
) -> ((f64, f64), Vec<f64>) {
    let reward = leaf(s, &[s.len() as i64]);
    let (bt, list) = ranking(&reward, values, p, keys).unwrap();
    let out = (bt.double_value(&[]), list.double_value(&[]));
    (&bt + &list).backward();
    (out, grad_of(&reward))
}

#[test]
fn ranking_permutation_duplication_and_empty_support_follow_the_declared_objective() {
    let s = [0.3f64, -0.2, 0.5, 0.1, -0.4, 0.25, 0.0, -0.1];
    let values = [0.9f64, 0.8, 0.8, 0.6, 0.5, 0.4, 0.3, 0.1];
    let p = [1.0f32, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0];
    let group = vec!["g:0".to_string(); 8];
    let (base, base_grad) = ranking_with_gradient(&s, &values, &p, &group);
    // List permutation: a set function; gradients follow their rows.
    let order = [5usize, 2, 7, 0, 3, 6, 1, 4];
    let (permuted, permuted_grad) = ranking_with_gradient(
        &order.map(|i| s[i]),
        &order.map(|i| values[i]),
        &order.map(|i| p[i]),
        &group,
    );
    assert_close(permuted.0, base.0, 1e-6, "permuted BT");
    assert_close(permuted.1, base.1, 1e-6, "permuted listwise");
    for (k, &i) in order.iter().enumerate() {
        assert_close(permuted_grad[k], base_grad[i], 1e-5, "permuted gradient");
    }
    // Duplication: a duplicated member adds a skipped tie with itself,
    // repeats its pairs, and counts twice in the listwise softmax.
    let mut ds = s.to_vec();
    ds.push(s[3]);
    let mut dv = values.to_vec();
    dv.push(values[3]);
    let mut dp = p.to_vec();
    dp.push(p[3]);
    let dkeys = vec!["g:0".to_string(); 9];
    let dp64 = dp.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let (dup, dup_grad) = ranking_with_gradient(&ds, &dv, &dp, &dkeys);
    let (rbt, gbt) = reference_bt(&ds, &dv, &dp64, &dkeys);
    let (rlist, glist) = reference_listwise(&ds, &dp64, &dkeys);
    assert_close(dup.0, rbt, 2e-6, "duplicated BT");
    assert_close(dup.1, rlist, 2e-6, "duplicated listwise");
    let expected = gbt
        .iter()
        .zip(&glist)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    assert_all_close(&dup_grad, &expected, 2e-5, "duplicated gradient");
    assert_close(
        dup_grad[8],
        dup_grad[3],
        1e-6,
        "duplicate and original symmetric",
    );
    assert!(
        (dup.0 - base.0).abs() > 1e-4,
        "duplication must change BT influence"
    );
    // Separate comparisons in one microbatch never pair across group keys.
    let two = [&group[..], &vec!["g:1".to_string(); 8][..]].concat();
    let s2 = [&s[..], &s.map(|x| -x)[..]].concat();
    let v2 = [&values[..], &values[..]].concat();
    let p2 = [&p[..], &p[..]].concat();
    let p2_64 = p2.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let (pair, _) = ranking_with_gradient(&s2, &v2, &p2, &two);
    assert_close(
        pair.0,
        reference_bt(&s2, &v2, &p2_64, &two).0,
        2e-6,
        "two groups BT",
    );
    // Empty support: an all-tied two-member group gives a connected exact zero.
    let tied = vec!["t:0".to_string(); 3];
    let ((bt, list), g) = ranking_with_gradient(&[0.4, -0.3], &[0.5, 0.5], &[1.0, 0.0], &tied[..2]);
    assert_eq!((bt, list), (0.0, 0.0));
    assert_exact_zero(&g, "empty-support gradient");
    // All ties with three members: BT empty, listwise still defined by percentiles.
    let ((bt, list), _) =
        ranking_with_gradient(&[0.4, -0.3, 0.1], &[0.5; 3], &[1.0, 0.5, 0.0], &tied);
    assert_eq!(bt, 0.0);
    let (rlist, _) = reference_listwise(&[0.4, -0.3, 0.1], &[1.0, 0.5, 0.0], &tied);
    assert_close(list, rlist, 2e-6, "tied listwise");
}

// ---------------------------------------------------------------------------
// Composite fixtures through the trainer's own objective function
// ---------------------------------------------------------------------------

#[test]
fn composite_matches_independent_terms_coefficients_and_gradients() {
    let batch = batch16();
    let expected = reference_composite(&batch, 0, batch.b);
    let l = leaves(&batch);
    let (loss, terms) = run_composite(&l, &batch.labels, 0, batch.b as i64).unwrap();
    for (k, name) in objective::TERM_NAMES.iter().enumerate() {
        assert_close(terms[k], expected.terms[k], 2e-6, name);
    }
    assert_close(
        loss.double_value(&[]),
        expected.loss,
        2e-6,
        "composite loss",
    );
    let recombined = terms.iter().zip(ACCEPTED).map(|(x, c)| x * c).sum::<f64>();
    assert_close(
        loss.double_value(&[]),
        recombined,
        1e-6,
        "coefficient recombination",
    );
    loss.backward();
    let g = grad_of(&l.logits);
    assert_all_close(&g, &expected.logits, 2e-5, "logits gradient");
    assert_all_close(
        &grad_of(&l.rewards),
        &expected.rewards,
        2e-5,
        "reward gradient",
    );
    assert_all_close(
        &grad_of(&l.runtime),
        &expected.runtime,
        2e-5,
        "runtime gradient",
    );
    assert_all_close(
        &grad_of(&l.ablation),
        &expected.ablation,
        2e-5,
        "ablation gradient",
    );
    // The context slot (which would predict BOS) and the final position are
    // never supervised.
    let (t, v) = (batch.t, batch.v);
    for i in 0..batch.b {
        for position in [0, t] {
            let at = (i * (t + 1) + position) * v;
            assert_exact_zero(&g[at..at + v], "unsupervised decoder position");
        }
    }
    // Inactive stability and log-variance terms: connected, exact-zero gradient.
    assert_exact_zero(&grad_of(&l.stability), "stability gradient");
    for x in &l.log_vars {
        assert_exact_zero(&grad_of(x), "loss_log_vars gradient");
    }
    // Two comparisons of one context stay separate groups: pairing them would
    // change BT, which the independent reference forms per (context, i/8).
    let merged = reference_bt(
        &(0..batch.b)
            .map(|i| batch.rewards[i * 4..i * 4 + 4].iter().sum::<f64>() / 4.0)
            .collect::<Vec<_>>(),
        &batch
            .labels
            .iter()
            .map(|r| r.value as f64)
            .collect::<Vec<_>>(),
        &batch
            .labels
            .iter()
            .map(|r| r.percentile as f64)
            .collect::<Vec<_>>(),
        &vec!["7".to_string(); batch.b],
    )
    .0;
    assert!((merged - terms[1]).abs() > 1e-4);
}

#[test]
fn composite_objective_depends_on_the_microbatch_partition() {
    // Declared normalization `microbatch_weighted_means_rank_mean_v1`: the same
    // 16 rows as one microbatch or as two accumulated halves are different
    // objectives with different gradients.
    let partition = jope_parallel::task_contract::training_partition(1, 8, 16).unwrap();
    assert_eq!(partition["accumulation"], 2);
    assert_eq!(
        partition["normalization"],
        "microbatch_weighted_means_rank_mean_v1"
    );
    let batch = batch16();
    let one = reference_composite(&batch, 0, 16);
    let (h0, h1) = (
        reference_composite(&batch, 0, 8),
        reference_composite(&batch, 8, 16),
    );
    let accumulated = (h0.loss + h1.loss) / 2.0;
    assert!(
        (one.loss - accumulated).abs() > 1e-3,
        "fixture must separate the partitions: {} vs {accumulated}",
        one.loss
    );
    let l = leaves(&batch);
    let mut total = 0.0;
    for lo in [0i64, 8] {
        let (loss, _) = run_composite(&l, &batch.labels, lo, 8).unwrap();
        let loss = loss / 2.0;
        total += loss.double_value(&[]);
        loss.backward();
    }
    assert_close(total, accumulated, 2e-6, "accumulated objective");
    // Rows 0..8 take their gradient only from the first microbatch and rows
    // 8..16 only from the second, each scaled by 1/accumulation.
    let halves = |a: &[f64], b: &[f64]| a.iter().chain(b).map(|g| g / 2.0).collect::<Vec<_>>();
    assert_all_close(
        &grad_of(&l.logits),
        &halves(&h0.logits, &h1.logits),
        2e-5,
        "accumulated logits gradient",
    );
    let rewards = halves(&h0.rewards, &h1.rewards);
    assert_all_close(
        &grad_of(&l.rewards),
        &rewards,
        2e-5,
        "accumulated reward gradient",
    );
    assert!(
        one.rewards
            .iter()
            .zip(&rewards)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "re-partitioning changes the reward gradient"
    );
}

#[test]
fn nonfinite_objective_is_rejected_before_backward() {
    // G12: an inactive head overflowing to inf, multiplied by 0.0, made the
    // scalar loss NaN while every gradient stayed finite; it was logged, not failed.
    let batch = batch16();
    let mut overflow = batch16();
    overflow.stability[3] = f64::INFINITY;
    let error = run_composite(&leaves(&overflow), &overflow.labels, 0, 16)
        .err()
        .expect("a nonfinite composite objective must be rejected");
    assert!(error.to_string().contains("nonfinite"), "{error}");
    // A nonfinite runtime label is equally rejected, not averaged in.
    let mut labels = batch.labels.clone();
    labels[5].runtime = f32::NAN;
    let l = leaves(&batch);
    assert!(run_composite(&l, &labels, 0, 16).is_err());
    // Positive control on the same fixture.
    let (loss, _) = run_composite(&l, &batch.labels, 0, 16).unwrap();
    assert!(loss.double_value(&[]).is_finite());
}

#[test]
fn composite_rejects_malformed_outputs_explicitly() {
    // Malformed shapes must be explicit objective errors, never a silently
    // different reduction (three reward heads) or a LibTorch panic.
    let batch = batch16();
    let l = leaves(&batch);
    let three_heads = l.rewards.narrow(1, 0, 3);
    let short_runtime = l.runtime.narrow(0, 0, 15);
    let no_final_position = l.logits.narrow(1, 0, batch.t as i64);
    let long_ablation = Tensor::cat(&[&l.ablation, &l.ablation.narrow(0, 0, 1)], 0);
    for (what, o) in [
        (
            "reward heads",
            [
                &l.logits,
                &three_heads,
                &l.runtime,
                &l.stability,
                &l.ablation,
            ],
        ),
        (
            "runtime rows",
            [
                &l.logits,
                &l.rewards,
                &short_runtime,
                &l.stability,
                &l.ablation,
            ],
        ),
        (
            "decoder positions",
            [
                &no_final_position,
                &l.rewards,
                &l.runtime,
                &l.stability,
                &l.ablation,
            ],
        ),
        (
            "ablation rows",
            [
                &l.logits,
                &l.rewards,
                &l.runtime,
                &l.stability,
                &long_ablation,
            ],
        ),
    ] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            call_composite(o, &l, &batch.labels, 0, 16).map(|_| ())
        }));
        assert!(
            matches!(result, Ok(Err(_))),
            "{what}: expected an explicit error"
        );
    }
    // Positive control.
    let (loss, _) = call_composite(
        [&l.logits, &l.rewards, &l.runtime, &l.stability, &l.ablation],
        &l,
        &batch.labels,
        0,
        16,
    )
    .unwrap();
    assert!(loss.double_value(&[]).is_finite());
}

// ---------------------------------------------------------------------------
// The real Policy graph (tiny CPU configuration, dropout off)
// ---------------------------------------------------------------------------

fn tiny_policy() -> (nn::VarStore, Policy) {
    tch::set_num_threads(1);
    tch::manual_seed(20260926);
    let mut c = Config::regression(8);
    c.dataset_dim = 12;
    c.target_dim = 4;
    c.target_keep_dim = 4;
    c.width = 16;
    c.heads = 2;
    c.ffn = 32;
    c.layers = 2;
    c.dropout = 0.0;
    let vs = nn::VarStore::new(Device::Cpu);
    let policy = Policy::new(vs.root(), c).unwrap();
    (vs, policy)
}

struct PolicyBatch {
    dataset: Tensor,
    target: Tensor,
    tokens: Tensor,
    legal: Tensor,
    labels: Vec<RowLabels>,
}

/// Eight rows (one comparison) of width `t >= 8`; positions past each
/// program are PAD with legal set {0}. Supervised legal sets exclude 0 and 1.
fn policy_batch(t: usize) -> PolicyBatch {
    let (b, v) = (8usize, 8usize);
    let mut tokens = vec![0i64; b * t];
    let mut sets = Vec::new();
    for i in 0..b {
        let body = 2 + i % 4;
        tokens[i * t] = 1;
        for k in 0..body {
            tokens[i * t + 1 + k] = 3 + ((i + k) % 5) as i64;
        }
        tokens[i * t + 1 + body] = 2;
        for j in 0..t - 1 {
            let x = tokens[i * t + j + 1] as usize;
            let mut set = match x {
                0 => vec![0],
                2 => vec![2],
                _ => vec![x, 3, 5, 7],
            };
            if x > 2 && (i + j) % 2 == 0 {
                set.push(4);
            }
            set.sort();
            set.dedup();
            sets.push(set);
        }
    }
    let values = [0.95f32, 0.8, 0.7, 0.62, 0.5, 0.41, 0.3, 0.2];
    let p = [1.0f32, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0];
    let seconds = [0.4f64, 0.1, 2.0, 0.0, 5.0, 0.3, 1.2, 0.7];
    PolicyBatch {
        dataset: (Tensor::arange(b as i64 * 12, (Kind::Float, Device::Cpu)) * 0.037)
            .sin()
            .reshape([b as i64, 12]),
        target: (Tensor::arange(b as i64 * 4, (Kind::Float, Device::Cpu)) * 0.11)
            .cos()
            .reshape([b as i64, 4]),
        tokens: Tensor::from_slice(&tokens).reshape([b as i64, t as i64]),
        legal: legal_tensor(&sets, [b as i64, t as i64 - 1, v as i64]),
        labels: (0..b)
            .map(|i| RowLabels {
                context: 3,
                value: values[i],
                percentile: p[i],
                runtime: seconds[i].ln_1p() as f32,
            })
            .collect(),
    }
}

/// Real forward (dropout off) plus the trainer's composite objective.
fn policy_objective(policy: &Policy, batch: &PolicyBatch) -> (Tensor, [f64; 5], Vec<Tensor>) {
    let pad = batch.tokens.ne(0);
    let (logits, rewards, runtime, stability, ablation) = policy
        .forward(&batch.dataset, &batch.target, &batch.tokens, &pad, false)
        .unwrap();
    let (loss, terms) = composite(
        &Outputs {
            logits: &logits,
            rewards: &rewards,
            runtime: &runtime,
            stability: &stability,
            ablation: &ablation,
        },
        policy.loss_log_vars.values(),
        &batch.tokens,
        &pad,
        &batch.legal,
        &batch.labels,
    )
    .unwrap();
    (
        loss,
        terms,
        vec![logits, rewards, runtime, stability, ablation],
    )
}

#[test]
fn pad_only_extension_leaves_the_objective_unchanged_with_dropout_off() {
    // K05/G14: T depends on the longest row in a microbatch. With dropout off,
    // widening every row by PAD positions must not change supervision.
    let (_vs, policy) = tiny_policy();
    let (loss8, terms8, out8) = policy_objective(&policy, &policy_batch(8));
    let (loss16, terms16, out16) = policy_objective(&policy, &policy_batch(16));
    assert_close(
        loss16.double_value(&[]),
        loss8.double_value(&[]),
        1e-5,
        "loss",
    );
    for k in 0..5 {
        assert_close(terms16[k], terms8[k], 1e-5, objective::TERM_NAMES[k]);
    }
    assert_all_close(
        &host(&out16[0].narrow(1, 0, 9)),
        &host(&out8[0]),
        1e-5,
        "logits on original positions",
    );
    for (k, name) in [(1, "rewards"), (2, "runtime"), (4, "ablation")] {
        assert_all_close(&host(&out16[k]), &host(&out8[k]), 1e-5, name);
    }
}

fn gradient_norms(vs: &nn::VarStore) -> BTreeMap<String, f64> {
    vs.variables()
        .into_iter()
        .map(|(name, p)| {
            let g = p.grad();
            let norm = if g.defined() {
                g.abs().max().double_value(&[])
            } else {
                0.0
            };
            (name, norm)
        })
        .collect()
}

fn zero_grads(vs: &nn::VarStore) {
    for (_, mut p) in vs.variables() {
        p.zero_grad();
    }
}

fn inactive(name: &str) -> bool {
    INACTIVE_PARAMETER_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// Output biases of the ranking heads get zero gradient in exact arithmetic
/// because BT and listwise are shift invariant (§4, `0*`).
fn shift_invariant(name: &str) -> bool {
    (name.starts_with("reward_heads.") || name.starts_with("reward_head_ablation."))
        && name.ends_with(".3.bias")
}

#[test]
fn each_term_reaches_exactly_its_declared_parameters() {
    // §4 of the objective specification, measured on the real Policy graph.
    let (vs, policy) = tiny_policy();
    let batch = policy_batch(8);
    let pad = batch.tokens.ne(0);
    let values = batch
        .labels
        .iter()
        .map(|r| r.value as f64)
        .collect::<Vec<_>>();
    let p = batch
        .labels
        .iter()
        .map(|r| r.percentile)
        .collect::<Vec<_>>();
    let groups = vec!["3:0".to_string(); 8];
    let weights = Tensor::from_slice(&p.iter().map(|p| 0.05 + p.powi(4)).collect::<Vec<_>>());
    let runtime_label =
        Tensor::from_slice(&batch.labels.iter().map(|r| r.runtime).collect::<Vec<_>>());
    let decoder: &[&str] = &[
        "encoder.",
        "ctx_proj.",
        "token_emb.",
        "pos_emb.",
        "decoder.",
        "out_norm.",
    ];
    let reward_side: &[&str] = &[
        "reward_encoder.",
        "z_gamma.",
        "z_proj.",
        "reward_heads.",
        "reward_norm.",
        "reward_token_emb.",
    ];
    let runtime_side: &[&str] = &[
        "runtime_head.",
        "reward_encoder.",
        "reward_norm.",
        "reward_token_emb.",
    ];
    let ablation_side: &[&str] = &["reward_head_ablation.", "reward_norm.", "reward_token_emb."];
    let mut bias_report = BTreeMap::new();
    for (term, reach) in [
        ("seq_nll", decoder),
        ("reward_bt", reward_side),
        ("reward_listwise", reward_side),
        ("runtime_huber", runtime_side),
        ("ablation_bt", ablation_side),
        ("stability_x0", &[][..]),
        ("loss_log_vars_x0", &[][..]),
    ] {
        zero_grads(&vs);
        let (_, _, o) = policy_objective(&policy, &batch);
        let reward = o[1].mean_dim(&[1i64][..], false, Kind::Float);
        let loss = match term {
            "seq_nll" => sequence_nll(
                &o[0].narrow(1, 1, 7),
                &batch.tokens.narrow(1, 1, 7),
                &batch.legal,
                &pad.narrow(1, 1, 7),
                &weights,
            )
            .unwrap(),
            "reward_bt" => ranking(&reward, &values, &p, &groups).unwrap().0,
            "reward_listwise" => ranking(&reward, &values, &p, &groups).unwrap().1,
            "runtime_huber" => {
                o[2].log1p()
                    .smooth_l1_loss(&runtime_label, tch::Reduction::Mean, 1.0)
            }
            "ablation_bt" => ranking_pairwise(&o[4], &values, &p, &groups).unwrap(),
            "stability_x0" => o[3].sum(Kind::Float) * 0.0,
            _ => Tensor::stack(
                &policy
                    .loss_log_vars
                    .values()
                    .map(|x| x * 0.0)
                    .collect::<Vec<_>>(),
                0,
            )
            .sum(Kind::Float),
        };
        loss.backward();
        for (name, norm) in gradient_norms(&vs) {
            let expected = reach.iter().any(|prefix| name.starts_with(prefix));
            if expected && shift_invariant(&name) {
                assert!(
                    norm < 1e-6,
                    "{term}: shift-invariant bias {name} = {norm:e}"
                );
                bias_report.insert(format!("{term}:{name}"), norm);
            } else if expected {
                assert!(norm > 0.0, "{term} must reach {name}");
            } else {
                assert_eq!(norm, 0.0, "{term} must not reach {name}");
            }
        }
    }
    // §4.1: unused embedding rows (>= vocabulary) and the PAD row receive no
    // gradient from the composite; the tied output projection is narrowed.
    zero_grads(&vs);
    let (loss, _, _) = policy_objective(&policy, &batch);
    loss.backward();
    let variables = vs.variables();
    for name in ["token_emb.weight", "reward_token_emb.weight"] {
        let g = variables[name].grad();
        assert_exact_zero(&host(&g.narrow(0, 8, g.size()[0] - 8)), name);
        assert_exact_zero(&host(&g.get(0)), name);
    }
    // The composite's inactive tensors are exactly the declared prefixes.
    for (name, norm) in gradient_norms(&vs) {
        if inactive(&name) {
            assert_eq!(norm, 0.0, "inactive {name} must get an exact-zero gradient");
        } else if shift_invariant(&name) {
            assert!(norm < 1e-6, "{name}: {norm:e}");
        } else {
            assert!(norm > 0.0, "active {name} must receive a gradient");
        }
    }
    eprintln!(
        "CPU f32 realized shift-invariant bias gradients (not a GPU/BF16 measurement): {bias_report:?}"
    );
}

#[test]
fn inactive_tensors_receive_decay_only_updates() {
    // G5: inactive tensors are decayed by AdamW, not frozen. With exact-zero
    // gradients and moments, one reference step multiplies them by (1 - lr·λ).
    let (vs, policy) = tiny_policy();
    let before = vs
        .variables()
        .into_iter()
        .map(|(name, p)| (name, host(&p)))
        .collect::<BTreeMap<_, _>>();
    let (loss, _, _) = policy_objective(&policy, &policy_batch(8));
    loss.backward();
    let mut optimizer = AdamW::new(1e-4, 0.01);
    optimizer.update_reference(&vs).unwrap();
    let mut checked = 0;
    for (name, p) in vs.variables() {
        if !inactive(&name) {
            continue;
        }
        let expected = before[&name]
            .iter()
            .map(|x| x * (1.0 - 1e-4 * 0.01))
            .collect::<Vec<_>>();
        assert_all_close(&host(&p), &expected, 1e-7, &name);
        let (m, v) = &optimizer.moments[&name];
        assert_exact_zero(&host(m), "first moment");
        assert_exact_zero(&host(v), "second moment");
        checked += 1;
    }
    // stability_head.{0,2}.{weight,bias} and seven loss_log_vars scalars.
    assert_eq!(checked, 11);
}
