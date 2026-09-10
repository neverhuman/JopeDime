//! Verified, arm-local stage reuse. The owned input is also the fitting input;
//! legacy caller-provided context tags are never accepted here.
use crate::{
    dsl::{DslExpr, Stage},
    genome::GenomeIr,
    kernel::Program,
    ops::FittedOp,
    Rows,
};
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    mem::size_of,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

pub const BYTE_CAP: usize = 64 * 1024 * 1024;
type Key = [u8; 32];

/// No mutable access or caller-supplied identity. Includes raw (pre-sanitize)
/// float bits, ordered source row IDs and the complete fold assignment.
#[derive(Debug)]
pub struct FoldInput {
    x: Rows,
    y: Vec<f64>,
    identity: Key,
}
impl FoldInput {
    pub fn new(
        x: Rows,
        y: Vec<f64>,
        row_ids: Vec<usize>,
        assignment: Vec<usize>,
        folds: usize,
        fold: usize,
        evaluator: Key,
    ) -> Result<Self> {
        if x.is_empty()
            || y.len() != x.len()
            || row_ids.len() != x.len()
            || folds < 2
            || fold >= folds
            || assignment.is_empty()
            || assignment.iter().any(|&f| f >= folds)
            || assignment.iter().filter(|&&f| f != fold).count() != x.len()
            || row_ids.iter().copied().collect::<HashSet<_>>().len() != row_ids.len()
            || x.iter().any(|row| row.len() != x[0].len())
        {
            bail!("invalid verified fold input");
        }
        let mut h = Sha256::new();
        h.update(b"feat-math.fold-input.v2\0");
        h.update(evaluator);
        for v in [x.len(), x[0].len(), y.len(), folds, fold, assignment.len()] {
            h.update((v as u64).to_le_bytes());
        }
        for &v in &assignment {
            h.update((v as u64).to_le_bytes());
        }
        for &v in &row_ids {
            h.update((v as u64).to_le_bytes());
        }
        for row in &x {
            for v in row {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        for v in &y {
            h.update(v.to_bits().to_le_bytes());
        }
        Ok(Self {
            x,
            y,
            identity: h.finalize().into(),
        })
    }
    pub fn features(&self) -> &Rows {
        &self.x
    }
    pub fn labels(&self) -> &[f64] {
        &self.y
    }
    pub fn identity(&self) -> Key {
        self.identity
    }
    pub(crate) fn root(&self, target: Option<&Stage>) -> Result<Key> {
        let mut h = Sha256::new();
        h.update(b"feat-math.stage-prefix.v2\0");
        h.update(self.identity);
        h.update([u8::from(target.is_some())]);
        if let Some(stage) = target {
            h.update(stage_bytes(stage)?);
        }
        Ok(h.finalize().into())
    }
}
fn stage_bytes(stage: &Stage) -> Result<Vec<u8>> {
    // Use the existing kernel codec; do not maintain another parameter codec.
    Ok(Program::import_legacy(&GenomeIr::Pipeline(DslExpr {
        stages: vec![stage.clone()],
    }))
    .to_bytes()?)
}
pub(crate) fn extend(key: Key, stage: &Stage) -> Result<Key> {
    let mut h = Sha256::new();
    h.update(key);
    h.update(stage_bytes(stage)?);
    Ok(h.finalize().into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Profile,
    Overhead,
    Reuse,
}

#[derive(Clone, Default, Debug, serde::Serialize)]
pub struct Stats {
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub skipped: u64,
    pub duplicates: u64,
    pub unsupported: u64,
    pub accounted_payload: usize,
    pub peak_accounted_payload: usize,
    pub retained_allocations: usize,
    pub fit_ns: u64,
    pub train_apply_ns: u64,
    pub repeated_ns: u64,
    pub repeated_stages: u64,
    pub head_ns: u64,
    pub target_ns: u64,
    pub key_ns: u64,
    pub cache_ns: u64,
    pub cleanup_ns: u64,
}
#[derive(Default)]
struct Account {
    bytes: AtomicUsize,
    allocations: AtomicUsize,
}
struct Charge {
    account: Arc<Account>,
    bytes: usize,
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.account.bytes.fetch_sub(self.bytes, Ordering::SeqCst);
        self.account.allocations.fetch_sub(1, Ordering::SeqCst);
    }
}
// The charge follows the same Arc as BOTH retained values, including eviction.
// Model-owned deep clones of fitted operators are ordinary fitting memory.
pub(crate) struct Payload {
    pub fitted: FittedOp,
    pub block: Rows,
    _charge: Option<Charge>,
}
struct Entry {
    payload: Arc<Payload>,
    generation: u64,
}
#[derive(Default)]
struct State {
    entries: HashMap<Key, Entry>,
    seen: HashSet<Key>,
    generation: u64,
    stats: Stats,
}
pub struct VerifiedStageCache {
    mode: Mode,
    cap: usize,
    account: Arc<Account>,
    state: Mutex<State>,
}
impl VerifiedStageCache {
    pub fn new(mode: Mode, byte_cap: usize) -> Self {
        Self {
            mode,
            cap: byte_cap,
            account: Arc::default(),
            state: Mutex::default(),
        }
    }
    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn stats(&self) -> Stats {
        let mut stats = self.state.lock().unwrap().stats.clone();
        stats.accounted_payload = self.account.bytes.load(Ordering::SeqCst);
        stats.retained_allocations = self.account.allocations.load(Ordering::SeqCst);
        stats
    }
    pub(crate) fn measure(&self, f: impl FnOnce(&mut Stats)) {
        f(&mut self.state.lock().unwrap().stats);
    }
    pub(crate) fn get(&self, key: Key) -> Option<Arc<Payload>> {
        let start = Instant::now();
        let mut state = self.state.lock().unwrap();
        let generation = state.generation;
        let hit = state.entries.get_mut(&key).map(|entry| {
            entry.generation = generation;
            Arc::clone(&entry.payload)
        });
        if hit.is_some() {
            state.stats.hits += 1;
        } else {
            state.stats.misses += 1;
        }
        state.stats.cache_ns += nanos(start);
        hit
    }
    pub(crate) fn record_stage(&self, key: Key, fit_ns: u64, apply_ns: u64) {
        let mut state = self.state.lock().unwrap();
        state.stats.fit_ns += fit_ns;
        state.stats.train_apply_ns += apply_ns;
        // Timings cover each individual stage, never cumulative prefix work.
        if self.mode == Mode::Profile && !state.seen.insert(key) {
            state.stats.repeated_ns += fit_ns + apply_ns;
            state.stats.repeated_stages += 1;
        }
    }
    pub(crate) fn insert(&self, key: Key, fitted: FittedOp, block: Rows) -> Arc<Payload> {
        let start = Instant::now();
        let mut state = self.state.lock().unwrap();
        if let Some(entry) = state.entries.get(&key) {
            let value = Arc::clone(&entry.payload);
            state.stats.duplicates += 1;
            state.stats.cache_ns += nanos(start);
            return value;
        }
        let bytes = fitted_heap(&fitted)
            .and_then(|v| v.checked_add(rows_heap(&block)))
            .and_then(|v| v.checked_add(size_of::<Payload>()));
        let mut charge = None;
        if self.mode == Mode::Reuse {
            if let Some(bytes) = bytes {
                if let Ok(previous) =
                    self.account
                        .bytes
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                            n.checked_add(bytes).filter(|&n| n <= self.cap)
                        })
                {
                    self.account.allocations.fetch_add(1, Ordering::SeqCst);
                    charge = Some(Charge {
                        account: Arc::clone(&self.account),
                        bytes,
                    });
                    state.stats.peak_accounted_payload =
                        state.stats.peak_accounted_payload.max(previous + bytes);
                }
            }
        }
        let admitted = charge.is_some();
        let payload = Arc::new(Payload {
            fitted,
            block,
            _charge: charge,
        });
        if admitted {
            let generation = state.generation;
            state.entries.insert(
                key,
                Entry {
                    payload: Arc::clone(&payload),
                    generation,
                },
            );
            state.stats.admissions += 1;
        } else {
            state.stats.skipped += 1;
        }
        state.stats.cache_ns += nanos(start);
        payload
    }
    /// Deterministic generation-boundary cleanup. No admission-time eviction.
    pub fn begin_generation(&self, generation: u64) {
        let start = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.generation = generation;
        let mut expired: Vec<_> = state
            .entries
            .iter()
            .filter(|(_, e)| e.generation < generation.saturating_sub(1))
            .map(|(k, e)| (e.generation, *k))
            .collect();
        expired.sort();
        for (_, key) in expired {
            state.entries.remove(&key);
            state.stats.evictions += 1;
        }
        state.stats.cleanup_ns += nanos(start);
    }
}
pub(crate) fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
fn vec_heap<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}
fn rows_heap(v: &Rows) -> usize {
    vec_heap(v) + v.iter().map(vec_heap).sum::<usize>()
}
fn fitted_heap(op: &FittedOp) -> Option<usize> {
    use FittedOp::*;
    Some(match op {
        Standardize { means, scales, .. } => vec_heap(means) + vec_heap(scales),
        RobustScale {
            medians, scales, ..
        } => vec_heap(medians) + vec_heap(scales),
        Rank { sorted_cols, .. } | Blom { sorted_cols, .. } => rows_heap(sorted_cols),
        QuantileBins { edges, .. } | BinOnehot { edges, .. } => rows_heap(edges),
        Hinge { knots, .. } | Spline { knots, .. } => rows_heap(knots),
        WinsorClip { lo, hi, .. } => vec_heap(lo) + vec_heap(hi),
        PolyInteract { pairs, .. } | PairProd { pairs, .. } => vec_heap(pairs),
        RffCos { w, b, .. } | RffSin { w, b, .. } => rows_heap(w) + vec_heap(b),
        RandProj { w, .. } => rows_heap(w),
        LandmarkDist { landmarks, .. } | Nystrom { landmarks, .. } => rows_heap(landmarks),
        ObliqueHinge { w, t, .. } => rows_heap(w) + vec_heap(t),
        TargetBinEncode { edges, enc, .. } => rows_heap(edges) + rows_heap(enc),
        SvdProject {
            means, components, ..
        } => vec_heap(means) + rows_heap(components),
        SelectCorr { indices, .. } => vec_heap(indices),
        Impute {
            fill,
            indicator_cols,
            ..
        } => vec_heap(fill) + vec_heap(indicator_cols),
        PairRatio { pairs, eps, .. } => vec_heap(pairs) + vec_heap(eps),
        Cheb { .. }
        | RowStats { .. }
        | Sigmoid { .. }
        | Step { .. }
        | NormRow { .. }
        | Polar { .. }
        | Lag { .. }
        | Roll { .. }
        | FftMag { .. }
        | RowSpectral { .. }
        | RowDiffStats { .. }
        | Symlog { .. } => 0,
        _ => return None,
    })
}
/// Explicit native contract allowlist. Unknown/private stages fall back for the
/// whole pipeline, including its prefix; later stages cannot cache their state.
pub(crate) fn supported(stage: &Stage) -> bool {
    matches!(
        stage.op.as_str(),
        "standardize"
            | "robust_scale"
            | "rank"
            | "blom"
            | "quantile_bins"
            | "bin_onehot"
            | "hinge"
            | "winsor_clip"
            | "cheb"
            | "poly_interact"
            | "rff_cos"
            | "rff_sin"
            | "rand_proj"
            | "landmark_dist"
            | "oblique_hinge"
            | "row_stats"
            | "pair_prod"
            | "target_bin_encode"
            | "sigmoid"
            | "step"
            | "norm_row"
            | "polar"
            | "lag"
            | "roll"
            | "fft_mag"
            | "row_spectral"
            | "row_diff_stats"
            | "svd_project"
            | "nystrom"
            | "symlog"
            | "select_corr"
            | "impute"
            | "pair_ratio"
            | "spline"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dsl::ParamValue, model};
    fn input() -> FoldInput {
        FoldInput::new(
            vec![vec![0.0, 2.0], vec![3.0, 4.0]],
            vec![1.0, 2.0],
            vec![7, 3],
            vec![0, 1, 0, 1],
            2,
            0,
            [1; 32],
        )
        .unwrap()
    }
    #[test]
    fn exact_identity_and_typed_keys() {
        let base = input();
        let key = base.identity();
        for variant in 0..7 {
            let mut x = base.x.clone();
            let mut y = base.y.clone();
            let mut ids = vec![7, 3];
            let mut assignment = vec![0, 1, 0, 1];
            let mut evaluator = [1; 32];
            let mut fold = 0;
            match variant {
                0 => x[0][0] = -0.0,
                1 => x[0][1] = f64::from_bits(2.0f64.to_bits() + 1),
                2 => y[0] = -1.0,
                3 => ids.reverse(),
                4 => assignment.reverse(),
                5 => evaluator[0] = 2,
                _ => fold = 1,
            }
            assert_ne!(
                key,
                FoldInput::new(x, y, ids, assignment, 2, fold, evaluator)
                    .unwrap()
                    .identity()
            );
        }
        let mut s = Stage::new("step");
        s.params.insert("t".into(), ParamValue::Float(1.0));
        let mut near = s.clone();
        near.params.insert(
            "t".into(),
            ParamValue::Float(f64::from_bits(1.0f64.to_bits() + 1)),
        );
        assert_eq!(s.to_string(), near.to_string());
        assert_ne!(extend(key, &s).unwrap(), extend(key, &near).unwrap());
        let mut typed = s.clone();
        typed.params.insert("t".into(), ParamValue::Int(1));
        assert_ne!(extend(key, &s).unwrap(), extend(key, &typed).unwrap());
        for v in [0.0, -0.0] {
            s.params.insert("t".into(), ParamValue::Float(v));
            near.params.insert("t".into(), ParamValue::Float(-v));
            assert_ne!(extend(key, &s).unwrap(), extend(key, &near).unwrap());
        }
        assert_ne!(base.root(None).unwrap(), base.root(Some(&s)).unwrap());
        assert_ne!(
            extend(extend(key, &s).unwrap(), &typed).unwrap(),
            extend(extend(key, &typed).unwrap(), &s).unwrap()
        );
    }
    #[test]
    fn reservation_and_shared_retention() {
        let cache = VerifiedStageCache::new(Mode::Reuse, 4096);
        let p = cache.insert(
            [0; 32],
            FittedOp::Sigmoid { append: false },
            vec![vec![1.0; 16]],
        );
        let bytes = cache.stats().accounted_payload;
        assert!(bytes > 128);
        let duplicate = cache.insert(
            [0; 32],
            FittedOp::Sigmoid { append: false },
            vec![vec![1.0; 16]],
        );
        assert!(Arc::ptr_eq(&p, &duplicate));
        assert_eq!(cache.stats().accounted_payload, bytes);
        cache.begin_generation(3);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get([0; 32]).is_none());
        assert_eq!(cache.stats().accounted_payload, bytes);
        drop(duplicate);
        drop(p);
        assert_eq!(cache.stats().accounted_payload, 0);
        for cap in [0, 1] {
            let c = VerifiedStageCache::new(Mode::Reuse, cap);
            c.insert(
                [0; 32],
                FittedOp::Sigmoid { append: false },
                vec![vec![1.0]],
            );
            assert_eq!(c.stats().admissions, 0);
            assert_eq!(c.stats().accounted_payload, 0);
        }
    }
    #[test]
    fn concurrent_duplicate_builders_never_over_admit() {
        let cache = Arc::new(VerifiedStageCache::new(Mode::Reuse, 1024));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let c = Arc::clone(&cache);
                scope.spawn(move || {
                    for i in 0..32 {
                        let p = c.insert(
                            [i; 32],
                            FittedOp::Sigmoid { append: false },
                            vec![vec![1.0; 16]],
                        );
                        assert!(c.stats().accounted_payload <= 1024);
                        drop(p);
                    }
                });
            }
        });
        assert!(cache.stats().peak_accounted_payload <= 1024);
        assert!(cache.stats().skipped > 0);
        assert!(cache.stats().duplicates > 0);
        cache.begin_generation(3);
        assert_eq!(cache.stats().retained_allocations, 0);
    }
    #[test]
    fn supervised_target_rank_and_training_pass_parity() {
        let x: Rows = (0..32)
            .map(|i| vec![(i % 7) as f64, (i % 5) as f64])
            .collect();
        let y: Vec<_> = (0..32).map(|i| (i % 9) as f64 + 1.0).collect();
        let input = FoldInput::new(
            x.clone(),
            y.clone(),
            (0..32).collect(),
            vec![1; 32],
            2,
            0,
            [3; 32],
        )
        .unwrap();
        for mode in [Mode::Profile, Mode::Overhead, Mode::Reuse] {
            for cap in [0, BYTE_CAP] {
                let cache = VerifiedStageCache::new(mode, cap);
                for transform in [
                    "(standardize)",
                    "(rank)",
                    "(blom)",
                    "(target_bin_encode)",
                    "(select_corr k=1)",
                    "(lag k=2)",
                    "(roll k=2)",
                    r#"(map expr="col(0)")"#,
                ] {
                    let genome = GenomeIr::parse(&format!(
                        "(pipe X (target kind=log1p) {transform} (ridge))"
                    ))
                    .unwrap();
                    let plain = model::fit(&genome, &x, &y);
                    for _ in 0..2 {
                        let cached = model::fit_with_verified_stage_cache(&genome, &input, &cache);
                        assert_eq!(plain.is_ok(), cached.is_ok(), "{transform}");
                        if let (Ok(a), Ok(b)) = (&plain, cached) {
                            assert_eq!(
                                a.predict(&x)
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                b.predict(&x)
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                "{transform}"
                            );
                        }
                    }
                }
                let bad = GenomeIr::parse("(pipe X (private_stage_v10) (ridge))").unwrap();
                assert!(model::fit_with_verified_stage_cache(&bad, &input, &cache).is_err());
                assert!(cache.stats().unsupported > 0);
            }
        }
    }
}
