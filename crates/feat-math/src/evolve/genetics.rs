use super::*;

pub(super) fn pick<'a>(rng: &mut Pcg32, items: &'a [&'a str]) -> &'a str {
    items[(rng.next_u32() as usize) % items.len()]
}

pub(super) fn rand_params(rng: &mut Pcg32, op: &str) -> BTreeMap<String, ParamValue> {
    let mut p = BTreeMap::new();
    let int = |rng: &mut Pcg32, lo: i64, hi: i64| -> ParamValue {
        ParamValue::Int(lo + (rng.next_u32() as i64) % (hi - lo + 1))
    };
    // Discrete lotteries over the exact value grids the engine's own search uses.
    let pick_i = |rng: &mut Pcg32, choices: &[i64]| -> ParamValue {
        ParamValue::Int(choices[(rng.next_u32() as usize) % choices.len()])
    };
    let pick_f = |rng: &mut Pcg32, choices: &[f64]| -> ParamValue {
        ParamValue::Float(choices[(rng.next_u32() as usize) % choices.len()])
    };
    match op {
        "quantile_bins" => {
            p.insert("bins".into(), int(rng, 4, 32));
        }
        "bin_onehot" | "target_bin_encode" => {
            p.insert("bins".into(), int(rng, 4, 16));
        }
        "hinge" => {
            p.insert("knots".into(), int(rng, 2, 8));
        }
        "winsor_clip" => {
            p.insert("q".into(), ParamValue::Float(rng.range_f64(0.01, 0.2)));
        }
        "cheb" => {
            p.insert("degree".into(), int(rng, 2, 8));
        }
        "rff_cos" | "rff_sin" | "oblique_hinge" => {
            p.insert("n".into(), int(rng, 16, 96));
            p.insert("gamma".into(), ParamValue::Float(rng.range_f64(0.1, 2.0)));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "rand_proj" => {
            p.insert("k".into(), int(rng, 8, 64));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "landmark_dist" | "nystrom" => {
            p.insert("k".into(), int(rng, 8, 48));
            p.insert("gamma".into(), ParamValue::Float(rng.range_f64(0.1, 2.0)));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "pair_prod" => {
            p.insert("topk".into(), int(rng, 4, 16));
        }
        "svd_project" => {
            p.insert("k".into(), int(rng, 4, 32));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        // Expansion vocabulary: drawn only when `expansion_vocab` is on, so
        // these arms are inert for every legacy stream.
        "impute" => {
            let strategies = ["median", "median", "mean"];
            let strategy = strategies[(rng.next_u32() as usize) % strategies.len()];
            p.insert("strategy".into(), ParamValue::Str(strategy.to_string()));
            if rng.next_f64() < 0.5 {
                p.insert("add_indicator".into(), ParamValue::Bool(true));
            }
        }
        "symlog" => {
            p.insert("c".into(), pick_f(rng, &[0.1, 1.0, 10.0]));
        }
        "select_corr" => {
            p.insert("k".into(), pick_i(rng, &[16, 32, 64, 128]));
            let by = ["corr", "corr", "var"];
            let by = by[(rng.next_u32() as usize) % by.len()];
            p.insert("by".into(), ParamValue::Str(by.to_string()));
        }
        "pair_ratio" => {
            p.insert("topk".into(), int(rng, 4, 16));
        }
        "spline" => {
            p.insert("knots".into(), int(rng, 3, 8));
        }
        // heads
        "histgbm" | "gbm" | "ensemble_ridge_gbm" => {
            p.insert("rounds".into(), int(rng, 40, 300));
            p.insert("depth".into(), int(rng, 2, 6));
            p.insert("lr".into(), ParamValue::Float(rng.range_f64(0.03, 0.25)));
        }
        // Deep bagged boosting: leaf-wise trees with row/column subsampling, a
        // min-leaf guard, and a held-out early-stopping split (v2.1 knobs).
        "histgbm2" => {
            p.insert("leaves".into(), pick_i(rng, &[15, 31, 63, 127, 255]));
            p.insert("lr".into(), pick_f(rng, &[0.01, 0.03, 0.05, 0.08]));
            p.insert("bags".into(), pick_i(rng, &[3, 5]));
            p.insert("rounds".into(), pick_i(rng, &[500, 1000]));
            p.insert("row_subsample".into(), pick_f(rng, &[0.6, 0.7, 0.85, 1.0]));
            p.insert("col_subsample".into(), pick_f(rng, &[0.6, 0.8, 1.0]));
            p.insert("min_leaf".into(), pick_i(rng, &[1, 3, 5, 10, 20]));
            p.insert("val_pct".into(), pick_i(rng, &[15, 20]));
            if rng.next_f64() < 0.5 {
                p.insert("val_shuffle".into(), ParamValue::Bool(true));
            }
        }
        // Composite-kernel Gaussian process: m inducing points, a few marginal-
        // likelihood steps, and a kernel algebra over {rbf, rq, periodic,
        // linear} joined by sum (+) and product (*). Plain ARD ("rbf") is the
        // modal draw and carries no explicit kernel= param.
        "gpr" => {
            p.insert("m".into(), pick_i(rng, &[300, 400, 500]));
            p.insert("steps".into(), pick_i(rng, &[30, 40, 60]));
            let kernels = [
                "rbf",
                "rbf",
                "linear",
                "rq",
                "periodic",
                "rbf+linear",
                "rq+linear",
                "rbf*periodic",
                "periodic+linear",
            ];
            let kernel = kernels[(rng.next_u32() as usize) % kernels.len()];
            if kernel != "rbf" {
                p.insert("kernel".into(), ParamValue::Str(kernel.to_string()));
            }
        }
        // Nadaraya-Watson kernel smoother: k nearest anchors, RBF bandwidth gamma.
        "ksmooth" => {
            p.insert("k".into(), pick_i(rng, &[8, 16, 32, 64, 128]));
            p.insert("gamma".into(), pick_f(rng, &[0.5, 1.0, 2.0, 4.0]));
        }
        "elasticnet" => {
            p.insert("alpha".into(), ParamValue::Float(rng.range_f64(1e-4, 0.05)));
            p.insert(
                "l1_ratio".into(),
                ParamValue::Float(rng.range_f64(0.1, 0.9)),
            );
        }
        _ => {}
    }
    // transforms may append (keep original cols) or replace
    if !HEAD_OPS.contains(&op) && rng.next_f64() < 0.5 {
        p.insert("append".into(), ParamValue::Bool(true));
    }
    p
}

pub(crate) fn rand_stage_from(rng: &mut Pcg32, vocab: &[&str]) -> Stage {
    let op = pick(rng, vocab);
    Stage {
        op: op.to_string(),
        params: rand_params(rng, op),
    }
}

pub(super) fn rand_stage(rng: &mut Pcg32) -> Stage {
    rand_stage_from(rng, &TRANSFORMS)
}

pub(crate) fn rand_head(rng: &mut Pcg32) -> Stage {
    let op = pick(rng, &HEADS);
    Stage {
        op: op.to_string(),
        params: rand_params(rng, op),
    }
}

pub(super) fn rand_genome_from(rng: &mut Pcg32, max_transforms: usize, vocab: &[&str]) -> DslExpr {
    let n = 1 + (rng.next_u32() as usize) % max_transforms.clamp(1, 6);
    let mut stages: Vec<Stage> = (0..n).map(|_| rand_stage_from(rng, vocab)).collect();
    stages.push(rand_head(rng));
    let program = DslExpr { stages };
    crate::capture_gp_source!("legacy-operator", "rand_genome_from", &program.to_string());
    program
}

pub(super) fn rand_genome(rng: &mut Pcg32, max_transforms: usize) -> DslExpr {
    rand_genome_from(rng, max_transforms, &TRANSFORMS)
}

/// The transform vocabulary a search run draws from: the frozen legacy array,
/// plus the expansion ops when the config opts in. With the flag off the
/// contents (and so every draw) are identical to the legacy array.
pub(super) fn generation_vocab(cfg: &EvolveConfig) -> Vec<&'static str> {
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if cfg.expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    vocab
}

/// Split a genome into (transforms, head). The head is always present in generated genomes.
pub(super) fn split(g: &DslExpr) -> (Vec<Stage>, Option<Stage>) {
    match g.stages.last() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            (g.stages[..g.stages.len() - 1].to_vec(), Some(last.clone()))
        }
        _ => (g.stages.clone(), None),
    }
}

pub(super) fn jitter_params(rng: &mut Pcg32, stage: &mut Stage) {
    let keys: Vec<String> = stage.params.keys().cloned().collect();
    if keys.is_empty() {
        // maybe flip append on
        stage.params = rand_params(rng, &stage.op);
        return;
    }
    let key = keys[(rng.next_u32() as usize) % keys.len()].clone();
    if let Some(v) = stage.params.get(&key) {
        let jittered = match v {
            ParamValue::Int(i) => {
                let delta = ((rng.next_u32() % 5) as i64) - 2;
                ParamValue::Int((*i + delta).max(1))
            }
            ParamValue::Float(f) => ParamValue::Float((f * rng.range_f64(0.6, 1.5)).max(1e-6)),
            ParamValue::Bool(b) => ParamValue::Bool(!b),
            ParamValue::Str(s) => ParamValue::Str(s.clone()),
        };
        stage.params.insert(key, jittered);
    }
}

/// v2 param jitter: log-scale for positive floats (`v * exp(u)`, u uniform in
/// [-ln 2.5, +ln 2.5] — symmetric multiplicative spread, right for
/// lr/lambda/gamma-style params where v1's ×[0.6, 1.5] is narrow and
/// upward-biased), logit-space for `l1_ratio` (stays inside (0, 1)), and the
/// original int/bool/str behavior. Exactly the same RNG-draw count per param
/// as v1, so the two are clean drop-in twins; the frozen `jitter_params`
/// stays byte-identical for every recorded v1 stream. Schema clamps re-apply
/// at plan/fit time exactly as they do for v1 jitter.
pub(crate) fn jitter_params_v2(rng: &mut Pcg32, stage: &mut Stage) {
    let keys: Vec<String> = stage.params.keys().cloned().collect();
    if keys.is_empty() {
        // maybe flip append on (mirrors v1)
        stage.params = rand_params(rng, &stage.op);
        return;
    }
    let key = keys[(rng.next_u32() as usize) % keys.len()].clone();
    if let Some(v) = stage.params.get(&key) {
        let span = 2.5f64.ln();
        let jittered = match v {
            ParamValue::Int(i) => {
                let delta = ((rng.next_u32() % 5) as i64) - 2;
                ParamValue::Int((*i + delta).max(1))
            }
            ParamValue::Float(f) if key == "l1_ratio" => {
                let p = f.clamp(0.01, 0.99);
                let logit = (p / (1.0 - p)).ln() + rng.range_f64(-span, span);
                ParamValue::Float(1.0 / (1.0 + (-logit).exp()))
            }
            ParamValue::Float(f) => {
                ParamValue::Float((f * rng.range_f64(-span, span).exp()).max(1e-6))
            }
            ParamValue::Bool(b) => ParamValue::Bool(!b),
            ParamValue::Str(s) => ParamValue::Str(s.clone()),
        };
        stage.params.insert(key, jittered);
    }
}

/// First index i where stages i and i+1 are canonical-equal (same op, same
/// params — Stage derives PartialEq over its canonical fields).
pub(super) fn adjacent_duplicate(tf: &[Stage]) -> Option<usize> {
    (0..tf.len().saturating_sub(1)).find(|&i| tf[i] == tf[i + 1])
}

/// v2 mutation over 7 equally-drawn outcomes (arm = draw % 7): {0, 1} param
/// jitter via `jitter_params_v2` (2/7 base mass, exactly v1's inner
/// stage-vs-head split), {2} insert drawn from `vocab` (legacy ++ expansion
/// when the caller opts in), {3} drop, {4} swap-adjacent, {5} head replace,
/// {6} simplify — delete the LATER stage of the first adjacent
/// canonical-duplicate pair (dissolves the observed 4x-standardize
/// pathology); with no such pair the draw falls through deterministically to
/// param jitter, so no draw is ever a no-op beyond v1's own degenerate arms.
/// The frozen `mutate` stays byte-identical for every recorded v1 stream.
pub(super) fn mutate_v2(
    rng: &mut Pcg32,
    g: &DslExpr,
    max_transforms: usize,
    vocab: &[&str],
) -> DslExpr {
    let (mut tf, head) = split(g);
    let mut head = head.unwrap_or_else(|| rand_head(rng));
    let mut arm = rng.next_u32() % 7;
    if arm == 6 {
        match adjacent_duplicate(&tf) {
            Some(i) => {
                tf.remove(i + 1);
                arm = 7; // handled; the match below is a no-op
            }
            None => arm = 0, // deterministic fall-through to param jitter
        }
    }
    match arm {
        0 | 1 => {
            if !tf.is_empty() && rng.next_f64() < 0.7 {
                let i = (rng.next_u32() as usize) % tf.len();
                jitter_params_v2(rng, &mut tf[i]);
            } else {
                jitter_params_v2(rng, &mut head);
            }
        }
        2 => {
            if tf.len() < max_transforms {
                let i = (rng.next_u32() as usize) % (tf.len() + 1);
                tf.insert(i, rand_stage_from(rng, vocab));
            }
        }
        3 => {
            if !tf.is_empty() {
                let i = (rng.next_u32() as usize) % tf.len();
                tf.remove(i);
            }
        }
        4 => {
            if tf.len() >= 2 {
                let i = (rng.next_u32() as usize) % (tf.len() - 1);
                tf.swap(i, i + 1);
            }
        }
        5 => {
            head = rand_head(rng);
        }
        _ => {}
    }
    let mut stages = tf;
    stages.push(head);
    DslExpr { stages }
}

pub(super) fn mutate(rng: &mut Pcg32, g: &DslExpr, max_transforms: usize) -> DslExpr {
    let (mut tf, head) = split(g);
    let mut head = head.unwrap_or_else(|| rand_head(rng));
    match rng.next_u32() % 6 {
        // param jitter on a random stage (transform or head)
        0 | 1 => {
            if !tf.is_empty() && rng.next_f64() < 0.7 {
                let i = (rng.next_u32() as usize) % tf.len();
                jitter_params(rng, &mut tf[i]);
            } else {
                jitter_params(rng, &mut head);
            }
        }
        // insert a random transform
        2 => {
            if tf.len() < max_transforms {
                let i = (rng.next_u32() as usize) % (tf.len() + 1);
                tf.insert(i, rand_stage(rng));
            }
        }
        // drop a transform
        3 => {
            if !tf.is_empty() {
                let i = (rng.next_u32() as usize) % tf.len();
                tf.remove(i);
            }
        }
        // swap two adjacent transforms
        4 => {
            if tf.len() >= 2 {
                let i = (rng.next_u32() as usize) % (tf.len() - 1);
                tf.swap(i, i + 1);
            }
        }
        // replace the head
        _ => {
            head = rand_head(rng);
        }
    }
    let mut stages = tf;
    stages.push(head);
    DslExpr { stages }
}

pub(super) fn crossover(
    rng: &mut Pcg32,
    a: &DslExpr,
    b: &DslExpr,
    max_transforms: usize,
) -> DslExpr {
    let (ta, ha) = split(a);
    let (tb, hb) = split(b);
    let cut_a = (rng.next_u32() as usize) % (ta.len() + 1);
    let cut_b = (rng.next_u32() as usize) % (tb.len() + 1);
    let mut tf: Vec<Stage> = ta[..cut_a].to_vec();
    tf.extend_from_slice(&tb[cut_b..]);
    tf.truncate(max_transforms);
    let head = if rng.next_f64() < 0.5 { ha } else { hb };
    let mut stages = tf;
    stages.push(head.unwrap_or_else(|| rand_head(rng)));
    DslExpr { stages }
}

// Keep the immutable legacy bodies above byte-for-byte intact. All production
// callers cross these observation seams before selection or population admission.
pub(super) fn mutate_with_source_capture(
    rng: &mut Pcg32,
    g: &DslExpr,
    max_transforms: usize,
) -> DslExpr {
    let program = mutate(rng, g, max_transforms);
    crate::capture_gp_source!("legacy-operator", "mutate", &program.to_string());
    program
}

pub(super) fn mutate_v2_with_source_capture(
    rng: &mut Pcg32,
    g: &DslExpr,
    max_transforms: usize,
    vocab: &[&str],
) -> DslExpr {
    let program = mutate_v2(rng, g, max_transforms, vocab);
    crate::capture_gp_source!("legacy-operator", "mutate_v2", &program.to_string());
    program
}

pub(super) fn crossover_with_source_capture(
    rng: &mut Pcg32,
    a: &DslExpr,
    b: &DslExpr,
    max_transforms: usize,
) -> DslExpr {
    let program = crossover(rng, a, b, max_transforms);
    crate::capture_gp_source!("legacy-operator", "crossover", &program.to_string());
    program
}

/// Build one deterministic grammar-random pipeline using the evolution engine's RNG stream.
///
/// The returned source is canonical pipeline DSL. This is an additive generation-0 seam; it does
/// not alter or share RNG state with any evolution entry point.
pub fn random_canonical_pipeline(seed: u64, max_transforms: usize) -> String {
    let mut rng = Pcg32::new(seed, 0x5EED);
    GenomeIr::Pipeline(rand_genome(&mut rng, max_transforms)).canonical()
}

/// Deterministically mutate one valid pipeline and return canonical pipeline DSL.
///
/// A fresh RNG stream is derived solely from `seed`, so callers can record and replay each
/// generation-0 offspring independently.
pub fn mutate_canonical_pipeline(
    source: &str,
    seed: u64,
    max_transforms: usize,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(mutate_with_source_capture(
        &mut rng,
        &parent,
        max_transforms,
    ))
    .canonical())
}

/// Deterministically cross two valid pipelines and return canonical pipeline DSL.
///
/// A fresh RNG stream is derived solely from `seed`, so parent order, seed, and
/// `max_transforms` completely determine the child.
pub fn crossover_canonical_pipelines(
    left: &str,
    right: &str,
    seed: u64,
    max_transforms: usize,
) -> Result<String, String> {
    let left = parse_canonical_pipeline(left)?;
    let right = parse_canonical_pipeline(right)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(crossover_with_source_capture(
        &mut rng,
        &left,
        &right,
        max_transforms,
    ))
    .canonical())
}

/// v2 twin of `mutate_canonical_pipeline`: expansion-aware insert vocabulary,
/// log-scale param jitter, and a simplify arm that dissolves adjacent
/// duplicate stages. Pure function of (source, seed, max_transforms,
/// expansion_vocab) on the same Pcg32 seeding pattern; the v1 entry point and
/// every recorded stream stay byte-identical.
pub fn mutate_canonical_pipeline_v2(
    source: &str,
    seed: u64,
    max_transforms: usize,
    expansion_vocab: bool,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(mutate_v2_with_source_capture(
        &mut rng,
        &parent,
        max_transforms,
        &vocab,
    ))
    .canonical())
}

/// Explicit mutation arms for train-only search protocols that seal an
/// operator distribution independently from the legacy v2 sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalMutationArm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
}

/// Apply exactly one requested v2-compatible mutation arm.
///
/// This is additive: the frozen v1/v2 entry points and their RNG streams are
/// unchanged. A structurally inapplicable arm may return the canonical parent;
/// callers that require semantic uniqueness must reject and retry it.
pub fn mutate_canonical_pipeline_with_arm(
    source: &str,
    seed: u64,
    max_transforms: usize,
    expansion_vocab: bool,
    arm: CanonicalMutationArm,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    let (mut transforms, head) = split(&parent);
    let mut head = head.unwrap_or_else(|| rand_head(&mut rng));
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    match arm {
        CanonicalMutationArm::ParameterJitter => {
            if !transforms.is_empty() && rng.next_f64() < 0.7 {
                let index = (rng.next_u32() as usize) % transforms.len();
                jitter_params_v2(&mut rng, &mut transforms[index]);
            } else {
                jitter_params_v2(&mut rng, &mut head);
            }
        }
        CanonicalMutationArm::Insert => {
            if transforms.len() < max_transforms {
                let index = (rng.next_u32() as usize) % (transforms.len() + 1);
                transforms.insert(index, rand_stage_from(&mut rng, &vocab));
            }
        }
        CanonicalMutationArm::Delete => {
            if !transforms.is_empty() {
                let index = (rng.next_u32() as usize) % transforms.len();
                transforms.remove(index);
            }
        }
        CanonicalMutationArm::AdjacentSwap => {
            if transforms.len() >= 2 {
                let index = (rng.next_u32() as usize) % (transforms.len() - 1);
                transforms.swap(index, index + 1);
            }
        }
        CanonicalMutationArm::HeadReplacement => head = rand_head(&mut rng),
        CanonicalMutationArm::Simplify => {
            if let Some(index) = adjacent_duplicate(&transforms) {
                transforms.remove(index + 1);
            } else if !transforms.is_empty() {
                let index = (rng.next_u32() as usize) % transforms.len();
                transforms.remove(index);
            }
        }
    }
    transforms.push(head);
    let program = GenomeIr::Pipeline(DslExpr { stages: transforms }).canonical();
    crate::capture_gp_source!("legacy-operator", "explicit-mutation-arm", &program);
    Ok(program)
}

pub(super) fn parse_canonical_pipeline(source: &str) -> Result<DslExpr, String> {
    match GenomeIr::parse(source).map_err(|error| error.to_string())? {
        GenomeIr::Pipeline(genome) => Ok(genome),
        _ => Err("expected a pipeline genome".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Fitness
// ---------------------------------------------------------------------------

/// Deterministic row->fold assignment (FNV-1a over (key, row), matching folds.rs behavior).
pub(super) fn fold_assign(n: usize, k: usize, key: &str) -> Vec<usize> {
    (0..n)
        .map(|row| crate::folds::fold_of(key, row, k))
        .collect()
}
