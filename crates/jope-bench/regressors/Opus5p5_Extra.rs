#![forbid(unsafe_code)]

// =============================================================================
// CBRS: Cross-fitted Boosted-Ridge Stack -- a general-purpose regressor.
//
// Training table: row-major, n_cols columns; the LAST column is always the
// target, columns 0..n_cols-2 are features. Prediction input: features only.
//
// 1. Validate; map target affinely to a standardized internal scale.
// 2. Per-feature statistics from a deterministic row sample: quantile bins
//    (trees) and robust clipped standardization (ridge).
// 3. Stratified K-fold cross-fitting. Per fold:
//    a. ridge on transformed features + missing indicators (lambda chosen on
//       pooled out-of-fold error),
//    b. leaf-wise histogram gradient boosting, early-stopped on the fold,
//    c. oblivious (symmetric) gradient boosting initialised from the fold's
//       ridge predictions, early-stopped on the fold.
// 4. Simplex blend of {leaf-wise GBDT, ridge+oblivious GBDT, ridge, mean}
//    chosen by out-of-fold squared error.
// 5. Predict: average the K fold models of each component, blend, map back,
//    clip to a margin around the training target range.
// =============================================================================

pub trait RegressorApi: Sized {
    fn new(seed: u64) -> Self;

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String>;

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String>;
}

// ----------------------------------------------------------------------------
// Fixed algorithmic constants (general settings, not data-derived).
// ----------------------------------------------------------------------------

const MAX_BINS: usize = 255;
const WORK_ROWS_CAP: usize = 400_000;
const TREE_FEAT_CAP: usize = 2_000;
const TREE_BYTES_CAP: f64 = 6.0e8;
const STAT_CELLS_CAP: f64 = 1.0e8;
const GBDT_WORK_BUDGET: f64 = 4.0e10;
const PRED_VISIT_BUDGET: f64 = 5.0e9;
const RIDGE_FEAT_CAP: usize = 300;
const RIDGE_GRAM_BUDGET: f64 = 4.0e9;
const LW_MAX_LEAVES: usize = 31;
const LW_MAX_DEPTH: usize = 12;
const LW_LAMBDA: f64 = 1.0;
const OBL_MAX_DEPTH: usize = 6;
const OBL_LAMBDA: f64 = 3.0;
const MAX_ROUNDS: usize = 4_000;
const RIDGE_GRID: [f64; 14] = [
    1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0,
];
const BLEND_STEPS: usize = 20;
const U_MARGIN: f64 = 0.2;
const PRED_BLOCK: usize = 64;
const POOL_CAP: usize = 160;

// ----------------------------------------------------------------------------
// Deterministic pseudorandomness (SplitMix64).
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let r = self.next_u64() as u128;
        ((r * (n as u128)) >> 64) as usize
    }
}

fn derive_seed(seed: u64, tag: u64) -> u64 {
    let mut r = Rng::new(seed ^ tag.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let a = r.next_u64();
    let b = r.next_u64();
    a ^ b.rotate_left(17)
}

fn partial_shuffle(v: &mut [usize], k: usize, rng: &mut Rng) {
    let n = v.len();
    let k = k.min(n);
    for i in 0..k {
        let j = i + rng.below(n - i);
        v.swap(i, j);
    }
}

// ----------------------------------------------------------------------------
// Small numeric helpers.
// ----------------------------------------------------------------------------

fn ceil_f(x: f64) -> f64 {
    let f = x.floor();
    if f < x {
        f + 1.0
    } else {
        f
    }
}

// Threshold strictly between two consecutive distinct values (a < b), or a.
fn midpoint(a: f64, b: f64) -> f64 {
    let m = 0.5 * a + 0.5 * b;
    if m >= a && m < b {
        m
    } else {
        a
    }
}

// First index i with x <= uppers[i]; uppers is increasing and ends with +inf.
fn bin_of(uppers: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = uppers.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x <= uppers[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn mse_rows(s: &[f64], pred: &[f64], rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for &r in rows.iter() {
        let d = s[r] - pred[r];
        acc += d * d;
    }
    acc / rows.len() as f64
}

// ----------------------------------------------------------------------------
// Target mapping: y -> u = (y - center)/half in [-1,1] -> s = (u - mu)/sd.
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TargetMap {
    center: f64,
    half: f64,
    mu: f64,
    sd: f64,
    ymin: f64,
    ymax: f64,
}

impl TargetMap {
    fn to_output(&self, s: f64) -> f64 {
        let mut u = self.mu + self.sd * s;
        if !u.is_finite() {
            u = self.mu;
        }
        let lim = 1.0 + U_MARGIN;
        u = u.max(-lim).min(lim);
        let y = self.center + self.half * u;
        if y.is_finite() {
            y
        } else if u > 0.0 {
            self.ymax
        } else {
            self.ymin
        }
    }
}

// ----------------------------------------------------------------------------
// Per-feature statistics.
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ColStats {
    uppers: Vec<f64>,
    n_distinct: usize,
    has_missing: bool,
    has_value: bool,
    lin_ok: bool,
    lo: f64,
    hi: f64,
    c: f64,
    sc: f64,
    mean: f64,
    sd: f64,
    miss_frac: f64,
}

fn col_stats(vals: &mut Vec<f64>, n_miss: usize) -> ColStats {
    let n_val = vals.len();
    let total = n_val + n_miss;
    let miss_frac = if total > 0 {
        n_miss as f64 / total as f64
    } else {
        1.0
    };
    let mut st = ColStats {
        uppers: Vec::new(),
        n_distinct: 0,
        has_missing: n_miss > 0,
        has_value: n_val > 0,
        lin_ok: false,
        lo: 0.0,
        hi: 0.0,
        c: 0.0,
        sc: 1.0,
        mean: 0.0,
        sd: 1.0,
        miss_frac,
    };
    if n_val == 0 {
        return st;
    }
    vals.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut dv: Vec<f64> = Vec::new();
    let mut dc: Vec<usize> = Vec::new();
    for &v in vals.iter() {
        let same = match dv.last() {
            Some(&last) => v == last,
            None => false,
        };
        if same {
            let li = dc.len() - 1;
            dc[li] += 1;
        } else {
            dv.push(v);
            dc.push(1);
        }
    }
    let nd = dv.len();
    st.n_distinct = nd;
    let mut up: Vec<f64> = Vec::new();
    if nd <= MAX_BINS {
        for i in 0..nd - 1 {
            up.push(midpoint(dv[i], dv[i + 1]));
        }
    } else {
        let per = n_val as f64 / MAX_BINS as f64;
        let mut cum = 0.0;
        for i in 0..nd - 1 {
            cum += dc[i] as f64;
            if up.len() + 1 < MAX_BINS && cum >= per * (up.len() + 1) as f64 {
                up.push(midpoint(dv[i], dv[i + 1]));
            }
        }
    }
    up.push(f64::INFINITY);
    st.uppers = up;
    if nd >= 2 {
        let last = n_val - 1;
        let ilo = ((last as f64) * 0.001).floor() as usize;
        let ihi = ceil_f((last as f64) * 0.999) as usize;
        let mut lo = vals[ilo.min(last)];
        let mut hi = vals[ihi.min(last)];
        if !(hi > lo) {
            lo = vals[0];
            hi = vals[last];
        }
        if hi > lo {
            let c = 0.5 * lo + 0.5 * hi;
            let sc = 0.5 * hi - 0.5 * lo;
            if sc > 0.0 && sc.is_finite() && c.is_finite() {
                let mut s1 = 0.0;
                for &v in vals.iter() {
                    s1 += (v.max(lo).min(hi) - c) / sc;
                }
                let mean = s1 / n_val as f64;
                let mut s2 = 0.0;
                for &v in vals.iter() {
                    let d = (v.max(lo).min(hi) - c) / sc - mean;
                    s2 += d * d;
                }
                let sd = (s2 / n_val as f64).sqrt();
                if sd > 1e-9 && sd.is_finite() {
                    st.lin_ok = true;
                    st.lo = lo;
                    st.hi = hi;
                    st.c = c;
                    st.sc = sc;
                    st.mean = mean;
                    st.sd = sd;
                }
            }
        }
    }
    st
}

// Adjusted correlation ratio of the target over 16 coarse quantile groups
// plus a missing group; used only to screen features when there are too many.
fn eta_score(
    train: &[f64],
    n_cols: usize,
    col: usize,
    rows: &[usize],
    ys: &[f64],
    uppers: &[f64],
) -> f64 {
    let nb = uppers.len().max(1);
    let mut sum = [0.0f64; 17];
    let mut cnt = [0.0f64; 17];
    let mut tot = 0.0;
    let mut tot2 = 0.0;
    for (t, &i) in rows.iter().enumerate() {
        let x = train[i * n_cols + col];
        let g = if x.is_nan() {
            16
        } else {
            (bin_of(uppers, x) * 16) / nb
        };
        let y = ys[t];
        sum[g] += y;
        cnt[g] += 1.0;
        tot += y;
        tot2 += y * y;
    }
    let n = rows.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sst = tot2 - tot * tot / n;
    if !(sst > 0.0) {
        return 0.0;
    }
    let mut ssb = 0.0;
    let mut groups = 0.0;
    for g in 0..17 {
        if cnt[g] > 0.0 {
            ssb += sum[g] * sum[g] / cnt[g];
            groups += 1.0;
        }
    }
    ssb -= tot * tot / n;
    ssb / sst - (groups - 1.0) / (n - 1.0)
}

// ----------------------------------------------------------------------------
// Linear (ridge) feature candidates.
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LinCand {
    col: usize,
    ind: bool,
    lo: f64,
    hi: f64,
    ctr: f64,
    sc: f64,
    mean: f64,
    sd: f64,
}

fn lin_value(c: &LinCand, x: f64) -> f64 {
    if c.ind {
        let m = if x.is_nan() { 1.0 } else { 0.0 };
        (m - c.mean) / c.sd
    } else if x.is_nan() {
        0.0
    } else {
        let v = (x.max(c.lo).min(c.hi) - c.ctr) / c.sc;
        (v - c.mean) / c.sd
    }
}

// ----------------------------------------------------------------------------
// Tree data structures.
// ----------------------------------------------------------------------------

struct TreeData {
    p: usize,
    bins: Vec<u8>,
    nb: Vec<usize>,
    cols: Vec<usize>,
    uppers: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct TNode {
    feat: usize,
    col: usize,
    bin: usize,
    thr: f64,
    miss_left: bool,
    left: usize,
    right: usize,
    value: f64,
    is_leaf: bool,
}

impl TNode {
    fn leaf(value: f64) -> TNode {
        TNode {
            feat: 0,
            col: 0,
            bin: 0,
            thr: 0.0,
            miss_left: false,
            left: 0,
            right: 0,
            value,
            is_leaf: true,
        }
    }
}

#[derive(Clone, Debug)]
struct LwTree {
    nodes: Vec<TNode>,
}

impl LwTree {
    fn eval_binned(&self, td: &TreeData, r: usize) -> f64 {
        let base = r * td.p;
        let mut i = 0usize;
        loop {
            let nd = &self.nodes[i];
            if nd.is_leaf {
                return nd.value;
            }
            let v = td.bins[base + nd.feat] as usize;
            let left = if v == td.nb[nd.feat] {
                nd.miss_left
            } else {
                v <= nd.bin
            };
            i = if left { nd.left } else { nd.right };
        }
    }

    fn eval_raw(&self, x: &[f64]) -> f64 {
        let mut i = 0usize;
        loop {
            let nd = &self.nodes[i];
            if nd.is_leaf {
                return nd.value;
            }
            let v = x[nd.col];
            let left = if v.is_nan() { nd.miss_left } else { v <= nd.thr };
            i = if left { nd.left } else { nd.right };
        }
    }
}

#[derive(Clone, Debug)]
struct OblTree {
    feat: Vec<usize>,
    col: Vec<usize>,
    bin: Vec<usize>,
    thr: Vec<f64>,
    miss_left: Vec<bool>,
    values: Vec<f64>,
}

impl OblTree {
    fn eval_binned(&self, td: &TreeData, r: usize) -> f64 {
        let base = r * td.p;
        let mut node = 0usize;
        for d in 0..self.feat.len() {
            let f = self.feat[d];
            let v = td.bins[base + f] as usize;
            let left = if v == td.nb[f] {
                self.miss_left[d]
            } else {
                v <= self.bin[d]
            };
            let bit = if left { 0 } else { 1 };
            node = node * 2 + bit;
        }
        self.values[node]
    }

    fn eval_raw(&self, x: &[f64]) -> f64 {
        let mut node = 0usize;
        for d in 0..self.col.len() {
            let v = x[self.col[d]];
            let left = if v.is_nan() {
                self.miss_left[d]
            } else {
                v <= self.thr[d]
            };
            let bit = if left { 0 } else { 1 };
            node = node * 2 + bit;
        }
        self.values[node]
    }
}

struct Pool {
    bufs: Vec<Vec<f64>>,
}

impl Pool {
    fn get(&mut self, len: usize) -> Vec<f64> {
        let mut b = match self.bufs.pop() {
            Some(v) => v,
            None => Vec::new(),
        };
        b.clear();
        b.resize(len, 0.0);
        b
    }

    fn put(&mut self, b: Vec<f64>) {
        if self.bufs.len() < POOL_CAP {
            self.bufs.push(b);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Split {
    gain: f64,
    fi: usize,
    bin: usize,
    miss_left: bool,
    cl: f64,
}

impl Split {
    fn none() -> Split {
        Split {
            gain: f64::NEG_INFINITY,
            fi: 0,
            bin: 0,
            miss_left: false,
            cl: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BoostPlan {
    rounds: usize,
    lr: f64,
    bag: usize,
    feats: usize,
    patience: usize,
    min_child: f64,
    lambda: f64,
    depth: usize,
}

struct LeafState {
    start: usize,
    end: usize,
    g: f64,
    c: f64,
    depth: usize,
    node: usize,
    hg: Vec<f64>,
    hc: Vec<f64>,
    best: Split,
}

// ----------------------------------------------------------------------------
// Histogram construction, split search, partitioning.
// ----------------------------------------------------------------------------

fn build_hist(
    idx: &[usize],
    td: &TreeData,
    feats: &[usize],
    offs: &[usize],
    res: &[f64],
    hg: &mut [f64],
    hc: &mut [f64],
) {
    let p = td.p;
    for &r in idx.iter() {
        let g = res[r];
        let base = r * p;
        let row = &td.bins[base..base + p];
        for (fi, &f) in feats.iter().enumerate() {
            let t = offs[fi] + row[f] as usize;
            hg[t] += g;
            hc[t] += 1.0;
        }
    }
}

fn best_split_leaf(
    hg: &[f64],
    hc: &[f64],
    g_tot: f64,
    c_tot: f64,
    feats: &[usize],
    offs: &[usize],
    nb: &[usize],
    lambda: f64,
    min_child: f64,
) -> Split {
    let mut best = Split::none();
    if c_tot < 2.0 * min_child {
        return best;
    }
    let parent = g_tot * g_tot / (c_tot + lambda);
    for (fi, &f) in feats.iter().enumerate() {
        let o = offs[fi];
        let k = nb[f];
        let gm = hg[o + k];
        let cm = hc[o + k];
        let mut gl = 0.0;
        let mut cl = 0.0;
        for b in 0..k {
            gl += hg[o + b];
            cl += hc[o + b];
            let cr = c_tot - cl;
            if cl >= min_child && cr >= min_child {
                let gr = g_tot - gl;
                let gain = gl * gl / (cl + lambda) + gr * gr / (cr + lambda) - parent;
                if gain > best.gain {
                    best = Split {
                        gain,
                        fi,
                        bin: b,
                        miss_left: false,
                        cl,
                    };
                }
            }
            if cm > 0.0 {
                let gl2 = gl + gm;
                let cl2 = cl + cm;
                let cr2 = c_tot - cl2;
                if cl2 >= min_child && cr2 >= min_child {
                    let gr2 = g_tot - gl2;
                    let gain = gl2 * gl2 / (cl2 + lambda) + gr2 * gr2 / (cr2 + lambda) - parent;
                    if gain > best.gain {
                        best = Split {
                            gain,
                            fi,
                            bin: b,
                            miss_left: true,
                            cl: cl2,
                        };
                    }
                }
            }
        }
    }
    if best.gain > f64::NEG_INFINITY {
        let f = feats[best.fi];
        let cm = hc[offs[best.fi] + nb[f]];
        if cm <= 0.0 {
            // No missing values seen here: unseen missing goes to the larger side.
            best.miss_left = best.cl >= c_tot - best.cl;
        }
    }
    best
}

fn partition(idx: &mut [usize], td: &TreeData, f: usize, bin: usize, miss_left: bool) -> usize {
    let p = td.p;
    let k = td.nb[f];
    let mut i = 0usize;
    let mut j = idx.len();
    while i < j {
        let v = td.bins[idx[i] * p + f] as usize;
        let left = if v == k { miss_left } else { v <= bin };
        if left {
            i += 1;
        } else {
            j -= 1;
            idx.swap(i, j);
        }
    }
    i
}

// ----------------------------------------------------------------------------
// Leaf-wise (best-first) tree growth.
// ----------------------------------------------------------------------------

fn grow_leafwise(
    td: &TreeData,
    idx: &mut [usize],
    res: &[f64],
    feats: &[usize],
    offs: &[usize],
    btot: usize,
    plan: &BoostPlan,
    pool: &mut Pool,
) -> Option<(LwTree, Vec<(usize, usize, f64)>)> {
    let m = idx.len();
    let mut hg = pool.get(btot);
    let mut hc = pool.get(btot);
    build_hist(&idx[..], td, feats, offs, res, &mut hg, &mut hc);
    let mut g0 = 0.0;
    for &r in idx.iter() {
        g0 += res[r];
    }
    let c0 = m as f64;
    let thr_gain = 1e-10 * (1.0 + c0);
    let best0 = best_split_leaf(
        &hg,
        &hc,
        g0,
        c0,
        feats,
        offs,
        &td.nb,
        plan.lambda,
        plan.min_child,
    );
    if !(best0.gain > thr_gain) {
        pool.put(hg);
        pool.put(hc);
        return None;
    }
    let mut nodes: Vec<TNode> = vec![TNode::leaf(0.0)];
    let mut leaves: Vec<LeafState> = vec![LeafState {
        start: 0,
        end: m,
        g: g0,
        c: c0,
        depth: 0,
        node: 0,
        hg,
        hc,
        best: best0,
    }];
    while leaves.len() < LW_MAX_LEAVES {
        let mut bi = usize::MAX;
        let mut bg = thr_gain;
        for (i, lf) in leaves.iter().enumerate() {
            if lf.best.gain > bg {
                bg = lf.best.gain;
                bi = i;
            }
        }
        if bi == usize::MAX {
            break;
        }
        let lf = leaves.swap_remove(bi);
        let sp = lf.best;
        let f = feats[sp.fi];
        let mid = lf.start + partition(&mut idx[lf.start..lf.end], td, f, sp.bin, sp.miss_left);
        let nl = mid - lf.start;
        let nr = lf.end - mid;
        let left_id = nodes.len();
        nodes.push(TNode::leaf(0.0));
        let right_id = nodes.len();
        nodes.push(TNode::leaf(0.0));
        {
            let nd = &mut nodes[lf.node];
            nd.is_leaf = false;
            nd.feat = f;
            nd.col = td.cols[f];
            nd.bin = sp.bin;
            nd.thr = td.uppers[f][sp.bin];
            nd.miss_left = sp.miss_left;
            nd.left = left_id;
            nd.right = right_id;
        }
        let mut shg = pool.get(btot);
        let mut shc = pool.get(btot);
        let left_small = nl <= nr;
        if left_small {
            build_hist(&idx[lf.start..mid], td, feats, offs, res, &mut shg, &mut shc);
        } else {
            build_hist(&idx[mid..lf.end], td, feats, offs, res, &mut shg, &mut shc);
        }
        let mut bhg = lf.hg;
        let mut bhc = lf.hc;
        for t in 0..btot {
            bhg[t] -= shg[t];
            bhc[t] -= shc[t];
        }
        let (lhg, lhc, rhg, rhc) = if left_small {
            (shg, shc, bhg, bhc)
        } else {
            (bhg, bhc, shg, shc)
        };
        let mut gl = 0.0;
        for t in lf.start..mid {
            gl += res[idx[t]];
        }
        let mut gr = 0.0;
        for t in mid..lf.end {
            gr += res[idx[t]];
        }
        let depth = lf.depth + 1;
        let (lbest, rbest) = if depth < plan.depth {
            (
                best_split_leaf(
                    &lhg,
                    &lhc,
                    gl,
                    nl as f64,
                    feats,
                    offs,
                    &td.nb,
                    plan.lambda,
                    plan.min_child,
                ),
                best_split_leaf(
                    &rhg,
                    &rhc,
                    gr,
                    nr as f64,
                    feats,
                    offs,
                    &td.nb,
                    plan.lambda,
                    plan.min_child,
                ),
            )
        } else {
            (Split::none(), Split::none())
        };
        leaves.push(LeafState {
            start: lf.start,
            end: mid,
            g: gl,
            c: nl as f64,
            depth,
            node: left_id,
            hg: lhg,
            hc: lhc,
            best: lbest,
        });
        leaves.push(LeafState {
            start: mid,
            end: lf.end,
            g: gr,
            c: nr as f64,
            depth,
            node: right_id,
            hg: rhg,
            hc: rhc,
            best: rbest,
        });
    }
    let mut ranges: Vec<(usize, usize, f64)> = Vec::with_capacity(leaves.len());
    for lf in leaves.into_iter() {
        let v = plan.lr * lf.g / (lf.c + plan.lambda);
        nodes[lf.node].value = v;
        ranges.push((lf.start, lf.end, v));
        pool.put(lf.hg);
        pool.put(lf.hc);
    }
    Some((LwTree { nodes }, ranges))
}

// ----------------------------------------------------------------------------
// Oblivious (symmetric) tree growth: one split per level shared by all nodes.
// ----------------------------------------------------------------------------

fn grow_oblivious(
    td: &TreeData,
    idx: &mut [usize],
    res: &[f64],
    feats: &[usize],
    offs: &[usize],
    btot: usize,
    plan: &BoostPlan,
    pool: &mut Pool,
) -> Option<(OblTree, Vec<(usize, usize, f64)>)> {
    let m = idx.len();
    let lam = plan.lambda;
    let thr_gain = 1e-10 * (1.0 + m as f64);
    let mut ranges: Vec<(usize, usize)> = vec![(0, m)];
    let mut hgs: Vec<Vec<f64>> = Vec::new();
    let mut hcs: Vec<Vec<f64>> = Vec::new();
    {
        let mut hg = pool.get(btot);
        let mut hc = pool.get(btot);
        build_hist(&idx[..], td, feats, offs, res, &mut hg, &mut hc);
        hgs.push(hg);
        hcs.push(hc);
    }
    let mut tree = OblTree {
        feat: Vec::new(),
        col: Vec::new(),
        bin: Vec::new(),
        thr: Vec::new(),
        miss_left: Vec::new(),
        values: Vec::new(),
    };
    let mut acc_r: Vec<f64> = vec![0.0; MAX_BINS + 1];
    let mut acc_l: Vec<f64> = vec![0.0; MAX_BINS + 1];
    let mut node_g: Vec<f64> = Vec::new();
    let mut node_c: Vec<f64> = Vec::new();
    for d in 0..plan.depth {
        let n_nodes = ranges.len();
        node_g.clear();
        node_c.clear();
        for &(st, en) in ranges.iter() {
            let mut g = 0.0;
            for t in st..en {
                g += res[idx[t]];
            }
            node_g.push(g);
            node_c.push((en - st) as f64);
        }
        let mut best_gain = thr_gain;
        let mut best: Option<(usize, usize, bool)> = None;
        for (fi, &f) in feats.iter().enumerate() {
            let o = offs[fi];
            let k = td.nb[f];
            for b in 0..k {
                acc_r[b] = 0.0;
                acc_l[b] = 0.0;
            }
            for nd in 0..n_nodes {
                let ct = node_c[nd];
                if ct <= 0.0 {
                    continue;
                }
                let gt = node_g[nd];
                let hg = &hgs[nd];
                let hc = &hcs[nd];
                let parent = gt * gt / (ct + lam);
                let gm = hg[o + k];
                let cm = hc[o + k];
                let mut gl = 0.0;
                let mut cl = 0.0;
                for b in 0..k {
                    gl += hg[o + b];
                    cl += hc[o + b];
                    let gr = gt - gl;
                    let cr = ct - cl;
                    let ga = gl * gl / (cl + lam) + gr * gr / (cr + lam) - parent;
                    acc_r[b] += ga;
                    if cm > 0.0 {
                        let gl2 = gl + gm;
                        let cl2 = cl + cm;
                        let gr2 = gt - gl2;
                        let cr2 = ct - cl2;
                        acc_l[b] += gl2 * gl2 / (cl2 + lam) + gr2 * gr2 / (cr2 + lam) - parent;
                    } else {
                        acc_l[b] += ga;
                    }
                }
            }
            for b in 0..k {
                if acc_r[b] > best_gain {
                    best_gain = acc_r[b];
                    best = Some((fi, b, false));
                }
                if acc_l[b] > best_gain {
                    best_gain = acc_l[b];
                    best = Some((fi, b, true));
                }
            }
        }
        let (bfi, bb, bml0) = match best {
            Some(x) => x,
            None => break,
        };
        let bf = feats[bfi];
        let o = offs[bfi];
        let k = td.nb[bf];
        let mut tm = 0.0;
        let mut tl = 0.0;
        for nd in 0..n_nodes {
            tm += hcs[nd][o + k];
            for b in 0..=bb {
                tl += hcs[nd][o + b];
            }
        }
        let mut bml = bml0;
        if tm <= 0.0 {
            let mut tot = 0.0;
            for nd in 0..n_nodes {
                tot += node_c[nd];
            }
            bml = tl >= tot - tl;
        }
        tree.feat.push(bf);
        tree.col.push(td.cols[bf]);
        tree.bin.push(bb);
        tree.thr.push(td.uppers[bf][bb]);
        tree.miss_left.push(bml);
        let mut new_ranges: Vec<(usize, usize)> = Vec::with_capacity(2 * n_nodes);
        let mut mids: Vec<usize> = Vec::with_capacity(n_nodes);
        for nd in 0..n_nodes {
            let (st, en) = ranges[nd];
            let mid = st + partition(&mut idx[st..en], td, bf, bb, bml);
            mids.push(mid);
            new_ranges.push((st, mid));
            new_ranges.push((mid, en));
        }
        let old_g = std::mem::take(&mut hgs);
        let old_c = std::mem::take(&mut hcs);
        if d + 1 < plan.depth {
            for (nd, (mut phg, mut phc)) in old_g.into_iter().zip(old_c.into_iter()).enumerate() {
                let (st, en) = ranges[nd];
                let mid = mids[nd];
                let mut shg = pool.get(btot);
                let mut shc = pool.get(btot);
                let left_small = mid - st <= en - mid;
                if left_small {
                    build_hist(&idx[st..mid], td, feats, offs, res, &mut shg, &mut shc);
                } else {
                    build_hist(&idx[mid..en], td, feats, offs, res, &mut shg, &mut shc);
                }
                for t in 0..btot {
                    phg[t] -= shg[t];
                    phc[t] -= shc[t];
                }
                if left_small {
                    hgs.push(shg);
                    hcs.push(shc);
                    hgs.push(phg);
                    hcs.push(phc);
                } else {
                    hgs.push(phg);
                    hcs.push(phc);
                    hgs.push(shg);
                    hcs.push(shc);
                }
            }
        } else {
            for v in old_g.into_iter() {
                pool.put(v);
            }
            for v in old_c.into_iter() {
                pool.put(v);
            }
        }
        ranges = new_ranges;
    }
    for v in hgs.into_iter() {
        pool.put(v);
    }
    for v in hcs.into_iter() {
        pool.put(v);
    }
    if tree.feat.is_empty() {
        return None;
    }
    let mut out: Vec<(usize, usize, f64)> = Vec::with_capacity(ranges.len());
    for &(st, en) in ranges.iter() {
        let mut g = 0.0;
        for t in st..en {
            g += res[idx[t]];
        }
        let v = plan.lr * g / ((en - st) as f64 + lam);
        tree.values.push(v);
        out.push((st, en, v));
    }
    Some((tree, out))
}

// ----------------------------------------------------------------------------
// Boosting loops with early stopping on the held-out fold.
// ----------------------------------------------------------------------------

fn feature_subset(td: &TreeData, pc: usize, fperm: &mut Vec<usize>, rng: &mut Rng) -> Vec<usize> {
    if pc < td.p {
        partial_shuffle(fperm, pc, rng);
        let mut f = fperm[..pc].to_vec();
        f.sort_unstable();
        f
    } else {
        (0..td.p).collect()
    }
}

fn hist_offsets(td: &TreeData, feats: &[usize]) -> (Vec<usize>, usize) {
    let mut offs: Vec<usize> = Vec::with_capacity(feats.len());
    let mut btot = 0usize;
    for &f in feats.iter() {
        offs.push(btot);
        btot += td.nb[f] + 1;
    }
    (offs, btot)
}

fn train_leafwise(
    td: &TreeData,
    s: &[f64],
    init: &[f64],
    train_rows: &[usize],
    val_rows: &[usize],
    plan: &BoostPlan,
    rng: &mut Rng,
    pool: &mut Pool,
) -> (Vec<LwTree>, Vec<f64>) {
    let n_w = s.len();
    let mut pred: Vec<f64> = init.to_vec();
    let mut best_val: Vec<f64> = val_rows.iter().map(|&r| pred[r]).collect();
    let mut trees: Vec<LwTree> = Vec::new();
    let n_fit = train_rows.len();
    if td.p == 0 || n_fit < 2 || plan.rounds == 0 {
        return (trees, best_val);
    }
    let mut best_mse = mse_rows(s, &pred, val_rows);
    let mut best_iter = 0usize;
    let mut res = vec![0.0f64; n_w];
    let mut perm: Vec<usize> = train_rows.to_vec();
    let mut fperm: Vec<usize> = (0..td.p).collect();
    let m = plan.bag.max(1).min(n_fit);
    let pc = plan.feats.max(1).min(td.p);
    let mut idx: Vec<usize> = vec![0usize; m];
    for _ in 0..plan.rounds {
        if m < n_fit {
            partial_shuffle(&mut perm, m, rng);
        }
        let feats = feature_subset(td, pc, &mut fperm, rng);
        let (offs, btot) = hist_offsets(td, &feats);
        for t in 0..m {
            let r = perm[t];
            idx[t] = r;
            res[r] = s[r] - pred[r];
        }
        let grown = grow_leafwise(td, &mut idx, &res, &feats, &offs, btot, plan, pool);
        let (tree, ranges) = match grown {
            Some(x) => x,
            None => break,
        };
        for &(st, en, v) in ranges.iter() {
            for t in st..en {
                pred[idx[t]] += v;
            }
        }
        for t in m..n_fit {
            let r = perm[t];
            pred[r] += tree.eval_binned(td, r);
        }
        for &r in val_rows.iter() {
            pred[r] += tree.eval_binned(td, r);
        }
        trees.push(tree);
        let mse = mse_rows(s, &pred, val_rows);
        if mse < best_mse {
            best_mse = mse;
            best_iter = trees.len();
            for (t, &r) in val_rows.iter().enumerate() {
                best_val[t] = pred[r];
            }
        } else if trees.len() - best_iter >= plan.patience {
            break;
        }
    }
    trees.truncate(best_iter);
    (trees, best_val)
}

fn train_oblivious(
    td: &TreeData,
    s: &[f64],
    init: &[f64],
    train_rows: &[usize],
    val_rows: &[usize],
    plan: &BoostPlan,
    rng: &mut Rng,
    pool: &mut Pool,
) -> (Vec<OblTree>, Vec<f64>) {
    let n_w = s.len();
    let mut pred: Vec<f64> = init.to_vec();
    let mut best_val: Vec<f64> = val_rows.iter().map(|&r| pred[r]).collect();
    let mut trees: Vec<OblTree> = Vec::new();
    let n_fit = train_rows.len();
    if td.p == 0 || n_fit < 2 || plan.rounds == 0 {
        return (trees, best_val);
    }
    let mut best_mse = mse_rows(s, &pred, val_rows);
    let mut best_iter = 0usize;
    let mut res = vec![0.0f64; n_w];
    let mut perm: Vec<usize> = train_rows.to_vec();
    let mut fperm: Vec<usize> = (0..td.p).collect();
    let m = plan.bag.max(1).min(n_fit);
    let pc = plan.feats.max(1).min(td.p);
    let mut idx: Vec<usize> = vec![0usize; m];
    for _ in 0..plan.rounds {
        if m < n_fit {
            partial_shuffle(&mut perm, m, rng);
        }
        let feats = feature_subset(td, pc, &mut fperm, rng);
        let (offs, btot) = hist_offsets(td, &feats);
        for t in 0..m {
            let r = perm[t];
            idx[t] = r;
            res[r] = s[r] - pred[r];
        }
        let grown = grow_oblivious(td, &mut idx, &res, &feats, &offs, btot, plan, pool);
        let (tree, ranges) = match grown {
            Some(x) => x,
            None => break,
        };
        for &(st, en, v) in ranges.iter() {
            for t in st..en {
                pred[idx[t]] += v;
            }
        }
        for t in m..n_fit {
            let r = perm[t];
            pred[r] += tree.eval_binned(td, r);
        }
        for &r in val_rows.iter() {
            pred[r] += tree.eval_binned(td, r);
        }
        trees.push(tree);
        let mse = mse_rows(s, &pred, val_rows);
        if mse < best_mse {
            best_mse = mse;
            best_iter = trees.len();
            for (t, &r) in val_rows.iter().enumerate() {
                best_val[t] = pred[r];
            }
        } else if trees.len() - best_iter >= plan.patience {
            break;
        }
    }
    trees.truncate(best_iter);
    (trees, best_val)
}

// ----------------------------------------------------------------------------
// Deterministic work planning (no clock access).
// ----------------------------------------------------------------------------

fn rounds_from(x: f64) -> usize {
    if !(x >= 1.0) {
        1
    } else if x >= MAX_ROUNDS as f64 {
        MAX_ROUNDS
    } else {
        x.floor() as usize
    }
}

fn learning_rate_for(rounds: usize) -> f64 {
    let r = rounds.max(1) as f64;
    (0.05 * (1000.0 / r).sqrt()).max(0.05).min(0.3)
}

fn patience_for(lr: f64) -> usize {
    let p = (5.0 / lr).round();
    if p < 20.0 {
        20
    } else if p > 150.0 {
        150
    } else {
        p as usize
    }
}

fn plan_leafwise(
    n_fit: usize,
    n_val: usize,
    pt: usize,
    bavg: f64,
    budget: f64,
    pred_cap: usize,
) -> BoostPlan {
    let mut m = if n_fit >= 10 {
        ceil_f(n_fit as f64 * 0.8) as usize
    } else {
        n_fit
    };
    m = m.max(1).min(n_fit.max(1));
    let mut pc = if pt <= 4 {
        pt
    } else {
        (ceil_f(pt as f64 * 0.8) as usize).min(512)
    };
    pc = pc.max(1);
    let m_min = m.min(20_000);
    let pc_min = pc.min(32);
    let mut rounds_f: f64 = 0.0;
    loop {
        let oob = n_fit.saturating_sub(m) + n_val;
        let cost = 3.5 * m as f64 * pc as f64
            + 25.0 * pc as f64 * bavg
            + 24.0 * oob as f64
            + 4.0 * n_fit as f64
            + 100.0;
        rounds_f = budget / cost;
        if rounds_f >= 400.0 || (m <= m_min && pc <= pc_min) {
            break;
        }
        if m > m_min {
            m = ((m as f64 * 0.75) as usize).max(m_min);
        }
        if pc > pc_min {
            pc = ((pc as f64 * 0.85) as usize).max(pc_min);
        }
    }
    let rounds = rounds_from(rounds_f).min(pred_cap.max(1));
    let lr = learning_rate_for(rounds);
    let min_child = (n_fit as f64 / 40.0).round().max(2.0).min(20.0);
    BoostPlan {
        rounds,
        lr,
        bag: m,
        feats: pc,
        patience: patience_for(lr),
        min_child,
        lambda: LW_LAMBDA,
        depth: LW_MAX_DEPTH,
    }
}

fn plan_oblivious(
    n_fit: usize,
    n_val: usize,
    pt: usize,
    bavg: f64,
    budget: f64,
    pred_cap: usize,
) -> BoostPlan {
    let mut m = n_fit.max(1);
    let mut pc = pt.min(256).max(1);
    let m_min = m.min(20_000);
    let pc_min = pc.min(16);
    let mut rounds_f: f64 = 0.0;
    loop {
        let oob = n_fit.saturating_sub(m) + n_val;
        let cost = 3.5 * m as f64 * pc as f64
            + 75.0 * pc as f64 * bavg
            + 14.0 * oob as f64
            + 4.0 * n_fit as f64
            + 100.0;
        rounds_f = budget / cost;
        if rounds_f >= 400.0 || (m <= m_min && pc <= pc_min) {
            break;
        }
        if m > m_min {
            m = ((m as f64 * 0.75) as usize).max(m_min);
        }
        if pc > pc_min {
            pc = ((pc as f64 * 0.85) as usize).max(pc_min);
        }
    }
    let rounds = rounds_from(rounds_f).min(pred_cap.max(1));
    let lr = learning_rate_for(rounds);
    let mut depth = 1usize;
    while depth < OBL_MAX_DEPTH && (4usize << (depth + 1)) <= n_fit {
        depth += 1;
    }
    BoostPlan {
        rounds,
        lr,
        bag: m,
        feats: pc,
        patience: patience_for(lr),
        min_child: 1.0,
        lambda: OBL_LAMBDA,
        depth,
    }
}

// ----------------------------------------------------------------------------
// Ridge regression with per-fold screening and pooled out-of-fold lambda.
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RidgeFold {
    sel: Vec<usize>,
    beta: Vec<f64>,
    icpt: f64,
}

fn ridge_eval_row(rf: &RidgeFold, cands: &[LinCand], row: &[f64]) -> f64 {
    let mut v = rf.icpt;
    for (a, &ci) in rf.sel.iter().enumerate() {
        let c = &cands[ci];
        v += rf.beta[a] * lin_value(c, row[c.col]);
    }
    v
}

// Solves (A + lam I) x = b for symmetric positive semidefinite A (d x d).
fn chol_solve(a: &[f64], d: usize, lam: f64, b: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            if i == j {
                sum += lam;
            }
            for t in 0..j {
                sum -= l[i * d + t] * l[j * d + t];
            }
            if i == j {
                if !(sum > 0.0) || !sum.is_finite() {
                    return None;
                }
                l[i * d + i] = sum.sqrt();
            } else {
                l[i * d + j] = sum / l[j * d + j];
            }
        }
    }
    let mut y = vec![0.0f64; d];
    for i in 0..d {
        let mut sum = b[i];
        for t in 0..i {
            sum -= l[i * d + t] * y[t];
        }
        y[i] = sum / l[i * d + i];
    }
    let mut x = vec![0.0f64; d];
    for ii in 0..d {
        let i = d - 1 - ii;
        let mut sum = y[i];
        for t in (i + 1)..d {
            sum -= l[t * d + i] * x[t];
        }
        x[i] = sum / l[i * d + i];
    }
    for v in x.iter() {
        if !v.is_finite() {
            return None;
        }
    }
    Some(x)
}

fn ridge_stage(
    train: &[f64],
    n_cols: usize,
    work: &[usize],
    s: &[f64],
    cands: &[LinCand],
    fold: &[usize],
    kf: usize,
) -> Vec<RidgeFold> {
    let nw = work.len();
    let nc = cands.len();
    let p = n_cols - 1;
    let mut fcount = vec![0usize; kf];
    let mut fsum = vec![0.0f64; kf];
    let mut fyy = vec![0.0f64; kf];
    for q in 0..nw {
        let k = fold[q];
        fcount[k] += 1;
        fsum[k] += s[q];
        fyy[k] += s[q] * s[q];
    }
    let mut tot_s = 0.0;
    let mut min_count = nw;
    for k in 0..kf {
        tot_s += fsum[k];
        if fcount[k] < min_count {
            min_count = fcount[k];
        }
    }
    if nc == 0 {
        let mut out = Vec::with_capacity(kf);
        for k in 0..kf {
            let nt = (nw - fcount[k]).max(1) as f64;
            out.push(RidgeFold {
                sel: Vec::new(),
                beta: Vec::new(),
                icpt: (tot_s - fsum[k]) / nt,
            });
        }
        return out;
    }
    let nfit_max = (nw - min_count).max(1);
    let gcap_f = (2.0 * RIDGE_GRAM_BUDGET / (kf as f64 * nfit_max as f64)).sqrt().floor();
    let gram_cap = if gcap_f < 8.0 {
        8
    } else if gcap_f > 1.0e6 {
        1_000_000
    } else {
        gcap_f as usize
    };
    let pr = nc.min(RIDGE_FEAT_CAP).min(gram_cap);
    let need_screen = nc > pr;
    let mut fz: Vec<f64> = Vec::new();
    let mut fzz: Vec<f64> = Vec::new();
    let mut fzy: Vec<f64> = Vec::new();
    if need_screen {
        fz = vec![0.0f64; kf * nc];
        fzz = vec![0.0f64; kf * nc];
        fzy = vec![0.0f64; kf * nc];
        for q in 0..nw {
            let k = fold[q];
            let i = work[q];
            let row = &train[i * n_cols..i * n_cols + p];
            let y = s[q];
            let base = k * nc;
            for (ci, c) in cands.iter().enumerate() {
                let z = lin_value(c, row[c.col]);
                fz[base + ci] += z;
                fzz[base + ci] += z * z;
                fzy[base + ci] += z * y;
            }
        }
    }
    let ng = RIDGE_GRID.len();
    let mut sse = vec![0.0f64; ng];
    let mut fold_sel: Vec<Vec<usize>> = Vec::with_capacity(kf);
    let mut fold_beta: Vec<Vec<Vec<f64>>> = Vec::with_capacity(kf);
    let mut fold_icpt: Vec<Vec<f64>> = Vec::with_capacity(kf);
    for k in 0..kf {
        let nt = (nw - fcount[k]).max(1) as f64;
        let sel: Vec<usize> = if need_screen {
            let mut ty = 0.0;
            let mut tyy = 0.0;
            for k2 in 0..kf {
                if k2 != k {
                    ty += fsum[k2];
                    tyy += fyy[k2];
                }
            }
            let vy = tyy - ty * ty / nt;
            let mut scored: Vec<(f64, usize)> = Vec::with_capacity(nc);
            for ci in 0..nc {
                let mut a = 0.0;
                let mut aa = 0.0;
                let mut ay = 0.0;
                for k2 in 0..kf {
                    if k2 != k {
                        let o = k2 * nc + ci;
                        a += fz[o];
                        aa += fzz[o];
                        ay += fzy[o];
                    }
                }
                let vz = aa - a * a / nt;
                let cov = ay - a * ty / nt;
                let score = if vz > 1e-12 * nt && vy > 1e-12 * nt {
                    cov * cov / (vz * vy)
                } else {
                    -1.0
                };
                scored.push((score, ci));
            }
            scored.sort_by(|u, v| v.0.total_cmp(&u.0).then(u.1.cmp(&v.1)));
            let mut chosen: Vec<usize> = scored[..pr].iter().map(|x| x.1).collect();
            chosen.sort_unstable();
            chosen
        } else {
            (0..nc).collect()
        };
        let d = sel.len();
        let mut g = vec![0.0f64; d * d];
        let mut sz = vec![0.0f64; d];
        let mut bz = vec![0.0f64; d];
        let mut zrow = vec![0.0f64; d];
        let mut sy = 0.0;
        for q in 0..nw {
            if fold[q] == k {
                continue;
            }
            let i = work[q];
            let row = &train[i * n_cols..i * n_cols + p];
            for (a, &ci) in sel.iter().enumerate() {
                let c = &cands[ci];
                zrow[a] = lin_value(c, row[c.col]);
            }
            let y = s[q];
            sy += y;
            for a in 0..d {
                let za = zrow[a];
                if za == 0.0 {
                    continue;
                }
                sz[a] += za;
                bz[a] += za * y;
                let off = a * d;
                for b in a..d {
                    g[off + b] += za * zrow[b];
                }
            }
        }
        let ybar = sy / nt;
        let mut amat = vec![0.0f64; d * d];
        let mut trace = 0.0;
        for a in 0..d {
            for b in a..d {
                let v = g[a * d + b] - sz[a] * sz[b] / nt;
                amat[a * d + b] = v;
                amat[b * d + a] = v;
            }
            trace += amat[a * d + a];
        }
        let rhs: Vec<f64> = (0..d).map(|a| bz[a] - sz[a] * ybar).collect();
        let zbar: Vec<f64> = (0..d).map(|a| sz[a] / nt).collect();
        let scale = if d > 0 { trace / d as f64 } else { 0.0 };
        let mut betas: Vec<Vec<f64>> = Vec::with_capacity(ng);
        let mut icpts: Vec<f64> = Vec::with_capacity(ng);
        for gi in 0..ng {
            let beta = if d > 0 && scale > 1e-12 {
                match chol_solve(&amat, d, RIDGE_GRID[gi] * scale, &rhs) {
                    Some(b) => b,
                    None => vec![0.0f64; d],
                }
            } else {
                vec![0.0f64; d]
            };
            let mut ic = ybar;
            for a in 0..d {
                ic -= beta[a] * zbar[a];
            }
            betas.push(beta);
            icpts.push(ic);
        }
        for q in 0..nw {
            if fold[q] != k {
                continue;
            }
            let i = work[q];
            let row = &train[i * n_cols..i * n_cols + p];
            for (a, &ci) in sel.iter().enumerate() {
                let c = &cands[ci];
                zrow[a] = lin_value(c, row[c.col]);
            }
            let y = s[q];
            for gi in 0..ng {
                let bt = &betas[gi];
                let mut pv = icpts[gi];
                for a in 0..d {
                    pv += bt[a] * zrow[a];
                }
                let e = y - pv;
                sse[gi] += e * e;
            }
        }
        fold_sel.push(sel);
        fold_beta.push(betas);
        fold_icpt.push(icpts);
    }
    let mut best = 0usize;
    for gi in 1..ng {
        if sse[gi] < sse[best] {
            best = gi;
        }
    }
    let mut out = Vec::with_capacity(kf);
    for k in 0..kf {
        let sel = std::mem::take(&mut fold_sel[k]);
        let beta = std::mem::take(&mut fold_beta[k][best]);
        let icpt = fold_icpt[k][best];
        out.push(RidgeFold { sel, beta, icpt });
    }
    out
}

// ----------------------------------------------------------------------------
// Folds and blending.
// ----------------------------------------------------------------------------

// Target-stratified K-fold assignment: sort by target, then give each
// consecutive block of K rows a random permutation of fold labels.
fn make_folds(s: &[f64], kf: usize, seed: u64) -> Vec<usize> {
    let n = s.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| s[a].total_cmp(&s[b]).then(a.cmp(&b)));
    let mut rng = Rng::new(seed);
    let mut fold = vec![0usize; n];
    let mut perm: Vec<usize> = (0..kf).collect();
    let mut i = 0usize;
    while i < n {
        let blk = kf.min(n - i);
        let len = perm.len();
        partial_shuffle(&mut perm, len, &mut rng);
        for t in 0..blk {
            fold[order[i + t]] = perm[t];
        }
        i += blk;
    }
    fold
}

// Simplex-constrained least squares over 4 out-of-fold prediction columns,
// solved exactly on a grid of step 1/BLEND_STEPS via the 4x4 Gram matrix.
fn blend_weights(oof: &[f64], s: &[f64], nw: usize) -> [f64; 4] {
    let mut a = [[0.0f64; 4]; 4];
    let mut c = [0.0f64; 4];
    for q in 0..nw {
        let v = [oof[q], oof[nw + q], oof[2 * nw + q], oof[3 * nw + q]];
        let y = s[q];
        for i in 0..4 {
            c[i] += v[i] * y;
            for j in 0..4 {
                a[i][j] += v[i] * v[j];
            }
        }
    }
    let steps = BLEND_STEPS;
    let inv = 1.0 / steps as f64;
    let mut best = [0.0f64, 0.0, 0.0, 1.0];
    let mut best_val = f64::INFINITY;
    for i0 in 0..=steps {
        for i1 in 0..=(steps - i0) {
            for i2 in 0..=(steps - i0 - i1) {
                let i3 = steps - i0 - i1 - i2;
                let w = [
                    i0 as f64 * inv,
                    i1 as f64 * inv,
                    i2 as f64 * inv,
                    i3 as f64 * inv,
                ];
                let mut val = 0.0;
                for i in 0..4 {
                    val -= 2.0 * w[i] * c[i];
                    for j in 0..4 {
                        val += w[i] * w[j] * a[i][j];
                    }
                }
                if val < best_val {
                    best_val = val;
                    best = w;
                }
            }
        }
    }
    best
}

// ----------------------------------------------------------------------------
// Fitted state.
// ----------------------------------------------------------------------------

struct FoldModel {
    lw_base: f64,
    lw_trees: Vec<LwTree>,
    obl_trees: Vec<OblTree>,
    lin_icpt: f64,
    lin_coef: Vec<f64>,
}

struct Fitted {
    n_features: usize,
    tmap: TargetMap,
    constant: Option<f64>,
    lin_cands: Vec<LinCand>,
    folds: Vec<FoldModel>,
    weights: [f64; 4],
    mean_s: f64,
}

fn constant_fitted(p: usize, value: f64, tmap: TargetMap) -> Fitted {
    Fitted {
        n_features: p,
        tmap,
        constant: Some(value),
        lin_cands: Vec::new(),
        folds: Vec::new(),
        weights: [0.0, 0.0, 0.0, 1.0],
        mean_s: 0.0,
    }
}

fn fit_impl(train: &[f64], n_rows: usize, n_cols: usize, seed: u64) -> Result<Fitted, String> {
    // ---- validation -------------------------------------------------------
    if n_rows == 0 {
        return Err(String::from("fit: n_rows must be at least 1"));
    }
    if n_cols == 0 {
        return Err(String::from("fit: n_cols must be at least 1"));
    }
    let total = match n_rows.checked_mul(n_cols) {
        Some(t) => t,
        None => return Err(String::from("fit: n_rows * n_cols overflows usize")),
    };
    if train.len() != total {
        return Err(format!(
            "fit: train has {} values but n_rows * n_cols = {}",
            train.len(),
            total
        ));
    }
    let p = n_cols - 1;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..n_rows {
        let row = &train[i * n_cols..(i + 1) * n_cols];
        let y = row[p];
        if !y.is_finite() {
            return Err(format!("fit: non-finite target in row {}", i));
        }
        if y < ymin {
            ymin = y;
        }
        if y > ymax {
            ymax = y;
        }
        for j in 0..p {
            if row[j].is_infinite() {
                return Err(format!(
                    "fit: infinite feature value in row {}, column {}",
                    i, j
                ));
            }
        }
    }

    // ---- target mapping ---------------------------------------------------
    let center = 0.5 * ymin + 0.5 * ymax;
    let half = 0.5 * ymax - 0.5 * ymin;
    let mut tmap = TargetMap {
        center,
        half,
        mu: 0.0,
        sd: 0.0,
        ymin,
        ymax,
    };
    if !(half > 0.0) {
        return Ok(constant_fitted(p, ymin, tmap));
    }
    let mut u: Vec<f64> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let y = train[i * n_cols + p];
        let mut v = (y - center) / half;
        if !v.is_finite() {
            v = 0.0;
        }
        u.push(v.max(-1.0).min(1.0));
    }
    let mut mu = 0.0;
    for &v in u.iter() {
        mu += v;
    }
    mu /= n_rows as f64;
    let mut var = 0.0;
    for &v in u.iter() {
        let d = v - mu;
        var += d * d;
    }
    let sdu = (var / n_rows as f64).sqrt();
    tmap.mu = mu;
    tmap.sd = sdu;
    if !(sdu > 1e-12) || p == 0 || n_rows < 2 {
        let c = tmap.to_output(0.0);
        return Ok(constant_fitted(p, c, tmap));
    }

    // ---- working rows -----------------------------------------------------
    let mut work: Vec<usize> = (0..n_rows).collect();
    if n_rows > WORK_ROWS_CAP {
        let mut rng = Rng::new(derive_seed(seed, 11));
        partial_shuffle(&mut work, WORK_ROWS_CAP, &mut rng);
        work.truncate(WORK_ROWS_CAP);
        work.sort_unstable();
    }
    let nw = work.len();
    let s_w: Vec<f64> = work.iter().map(|&i| (u[i] - mu) / sdu).collect();
    let mut mean_s = 0.0;
    for &v in s_w.iter() {
        mean_s += v;
    }
    mean_s /= nw as f64;

    // ---- per-feature statistics from a deterministic sample ---------------
    let samp_cap = (STAT_CELLS_CAP / p as f64).max(2000.0).min(200_000.0) as usize;
    let n_samp = nw.min(samp_cap).max(1);
    let mut samp_pos: Vec<usize> = Vec::with_capacity(n_samp);
    for t in 0..n_samp {
        let q = ((t as u128 * nw as u128) / n_samp as u128) as usize;
        samp_pos.push(q.min(nw - 1));
    }
    let samp_rows: Vec<usize> = samp_pos.iter().map(|&q| work[q]).collect();
    let samp_s: Vec<f64> = samp_pos.iter().map(|&q| s_w[q]).collect();
    let mut stats: Vec<ColStats> = Vec::with_capacity(p);
    let mut vals: Vec<f64> = Vec::with_capacity(n_samp);
    for j in 0..p {
        vals.clear();
        let mut n_miss = 0usize;
        for &i in samp_rows.iter() {
            let x = train[i * n_cols + j];
            if x.is_nan() {
                n_miss += 1;
            } else {
                vals.push(x);
            }
        }
        stats.push(col_stats(&mut vals, n_miss));
    }

    // ---- tree features and binned matrix ----------------------------------
    let mut tf: Vec<usize> = Vec::new();
    for j in 0..p {
        let st = &stats[j];
        if st.has_value && (st.n_distinct >= 2 || st.has_missing) {
            tf.push(j);
        }
    }
    let tcap_f = (TREE_BYTES_CAP / nw as f64).floor();
    let tcap = if tcap_f < 1.0 {
        1
    } else {
        tcap_f.min(TREE_FEAT_CAP as f64) as usize
    };
    if tf.len() > tcap {
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(tf.len());
        for &j in tf.iter() {
            let sc = eta_score(train, n_cols, j, &samp_rows, &samp_s, &stats[j].uppers);
            scored.push((sc, j));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut keep: Vec<usize> = scored[..tcap].iter().map(|x| x.1).collect();
        keep.sort_unstable();
        tf = keep;
    }
    let pt = tf.len();
    let mut nbv: Vec<usize> = Vec::with_capacity(pt);
    let mut upv: Vec<Vec<f64>> = Vec::with_capacity(pt);
    for &j in tf.iter() {
        nbv.push(stats[j].uppers.len());
        upv.push(stats[j].uppers.clone());
    }
    let mut bins: Vec<u8> = vec![0u8; nw * pt];
    if pt > 0 {
        for (q, &i) in work.iter().enumerate() {
            let row = &train[i * n_cols..i * n_cols + p];
            let dst = &mut bins[q * pt..(q + 1) * pt];
            for (fi, &j) in tf.iter().enumerate() {
                let x = row[j];
                let b = if x.is_nan() {
                    nbv[fi]
                } else {
                    bin_of(&upv[fi], x)
                };
                dst[fi] = b as u8;
            }
        }
    }
    let mut bsum = 0.0;
    for &k in nbv.iter() {
        bsum += (k + 1) as f64;
    }
    let bavg = if pt > 0 { bsum / pt as f64 } else { 1.0 };
    let td = TreeData {
        p: pt,
        bins,
        nb: nbv,
        cols: tf.clone(),
        uppers: upv,
    };

    // ---- linear candidates -------------------------------------------------
    let mut cands: Vec<LinCand> = Vec::new();
    for j in 0..p {
        let st = &stats[j];
        if st.lin_ok {
            cands.push(LinCand {
                col: j,
                ind: false,
                lo: st.lo,
                hi: st.hi,
                ctr: st.c,
                sc: st.sc,
                mean: st.mean,
                sd: st.sd,
            });
        }
        if st.has_missing && st.has_value {
            let pm = st.miss_frac;
            let sdm = (pm * (1.0 - pm)).sqrt();
            if sdm > 1e-6 {
                cands.push(LinCand {
                    col: j,
                    ind: true,
                    lo: 0.0,
                    hi: 0.0,
                    ctr: 0.0,
                    sc: 1.0,
                    mean: pm,
                    sd: sdm,
                });
            }
        }
    }

    // ---- folds ---------------------------------------------------------------
    let mut kf = if nw <= 20_000 {
        5
    } else if nw <= 60_000 {
        4
    } else if nw <= 150_000 {
        3
    } else {
        2
    };
    if kf > nw {
        kf = nw;
    }
    let fold = make_folds(&s_w, kf, derive_seed(seed, 21));

    // ---- ridge ---------------------------------------------------------------
    let ridge = ridge_stage(train, n_cols, &work, &s_w, &cands, &fold, kf);

    // ---- boosting per fold -------------------------------------------------
    let budget_family = GBDT_WORK_BUDGET / (2.0 * kf as f64);
    let tvis = PRED_VISIT_BUDGET / n_rows as f64;
    let cap_lw = rounds_from(tvis / (2.0 * kf as f64 * 8.0));
    let cap_obl = rounds_from(tvis / (2.0 * kf as f64 * 6.0));
    let mut pool = Pool { bufs: Vec::new() };
    let mut oof = vec![0.0f64; 4 * nw];
    let mut lw_models: Vec<(f64, Vec<LwTree>)> = Vec::with_capacity(kf);
    let mut obl_models: Vec<Vec<OblTree>> = Vec::with_capacity(kf);
    let mut rbase = vec![0.0f64; nw];
    let mut lw_init = vec![0.0f64; nw];
    for k in 0..kf {
        let train_rows: Vec<usize> = (0..nw).filter(|&q| fold[q] != k).collect();
        let val_rows: Vec<usize> = (0..nw).filter(|&q| fold[q] == k).collect();
        let n_fit = train_rows.len();
        let n_val = val_rows.len();
        let mut tmean = 0.0;
        for &q in train_rows.iter() {
            tmean += s_w[q];
        }
        tmean /= n_fit.max(1) as f64;
        for q in 0..nw {
            lw_init[q] = tmean;
        }
        let rf = &ridge[k];
        for q in 0..nw {
            let i = work[q];
            rbase[q] = ridge_eval_row(rf, &cands, &train[i * n_cols..i * n_cols + p]);
        }
        for &q in val_rows.iter() {
            oof[2 * nw + q] = rbase[q];
            oof[3 * nw + q] = tmean;
        }
        let plan_lw = plan_leafwise(n_fit, n_val, pt, bavg, budget_family, cap_lw);
        let mut rng_lw = Rng::new(derive_seed(seed, 1_000 + k as u64));
        let (lw_trees, lw_val) = train_leafwise(
            &td,
            &s_w,
            &lw_init,
            &train_rows,
            &val_rows,
            &plan_lw,
            &mut rng_lw,
            &mut pool,
        );
        for (t, &q) in val_rows.iter().enumerate() {
            oof[q] = lw_val[t];
        }
        lw_models.push((tmean, lw_trees));
        let plan_ob = plan_oblivious(n_fit, n_val, pt, bavg, budget_family, cap_obl);
        let mut rng_ob = Rng::new(derive_seed(seed, 2_000 + k as u64));
        let (ob_trees, ob_val) = train_oblivious(
            &td,
            &s_w,
            &rbase,
            &train_rows,
            &val_rows,
            &plan_ob,
            &mut rng_ob,
            &mut pool,
        );
        for (t, &q) in val_rows.iter().enumerate() {
            oof[nw + q] = ob_val[t];
        }
        obl_models.push(ob_trees);
    }

    // ---- blend and assemble ------------------------------------------------
    let weights = blend_weights(&oof, &s_w, nw);
    let need_lin = weights[1] > 0.0 || weights[2] > 0.0;
    let mut used = vec![usize::MAX; cands.len()];
    let mut lin_final: Vec<LinCand> = Vec::new();
    if need_lin {
        for rf in ridge.iter() {
            for &ci in rf.sel.iter() {
                if used[ci] == usize::MAX {
                    used[ci] = lin_final.len();
                    lin_final.push(cands[ci].clone());
                }
            }
        }
    }
    let mut folds: Vec<FoldModel> = Vec::with_capacity(kf);
    for (k, ((base, lw_trees), ob_trees)) in lw_models
        .into_iter()
        .zip(obl_models.into_iter())
        .enumerate()
    {
        let rf = &ridge[k];
        let mut coef = vec![0.0f64; lin_final.len()];
        if need_lin {
            for (a, &ci) in rf.sel.iter().enumerate() {
                coef[used[ci]] = rf.beta[a];
            }
        }
        folds.push(FoldModel {
            lw_base: base,
            lw_trees: if weights[0] > 0.0 { lw_trees } else { Vec::new() },
            obl_trees: if weights[1] > 0.0 { ob_trees } else { Vec::new() },
            lin_icpt: rf.icpt,
            lin_coef: coef,
        });
    }
    Ok(Fitted {
        n_features: p,
        tmap,
        constant: None,
        lin_cands: lin_final,
        folds,
        weights,
        mean_s,
    })
}

// ----------------------------------------------------------------------------
// Prediction (block-wise for cache reuse; per-row arithmetic order is fixed,
// so outputs do not depend on batch composition or order).
// ----------------------------------------------------------------------------

fn predict_impl(
    f: &Fitted,
    features: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Result<Vec<f64>, String> {
    if n_features != f.n_features {
        return Err(format!(
            "predict: n_features = {} but the model was fitted with {} features",
            n_features, f.n_features
        ));
    }
    let total = match n_rows.checked_mul(n_features) {
        Some(t) => t,
        None => return Err(String::from("predict: n_rows * n_features overflows usize")),
    };
    if features.len() != total {
        return Err(format!(
            "predict: features has {} values but n_rows * n_features = {}",
            features.len(),
            total
        ));
    }
    for (pos, &x) in features.iter().enumerate() {
        if x.is_infinite() {
            let (r, c) = if n_features > 0 {
                (pos / n_features, pos % n_features)
            } else {
                (0, 0)
            };
            return Err(format!(
                "predict: infinite feature value in row {}, column {}",
                r, c
            ));
        }
    }
    let mut out: Vec<f64> = Vec::with_capacity(n_rows);
    if let Some(c) = f.constant {
        for _ in 0..n_rows {
            out.push(c);
        }
        return Ok(out);
    }
    let nf = n_features;
    let nl = f.lin_cands.len();
    let kf = f.folds.len().max(1) as f64;
    let use_lw = f.weights[0] > 0.0;
    let use_ob = f.weights[1] > 0.0;
    let mut z = vec![0.0f64; PRED_BLOCK * nl];
    let mut acc_lw = vec![0.0f64; PRED_BLOCK];
    let mut acc_ob = vec![0.0f64; PRED_BLOCK];
    let mut acc_r = vec![0.0f64; PRED_BLOCK];
    let mut rr = vec![0.0f64; PRED_BLOCK];
    let mut tmp = vec![0.0f64; PRED_BLOCK];
    let mut start = 0usize;
    while start < n_rows {
        let end = (start + PRED_BLOCK).min(n_rows);
        let nb = end - start;
        for b in 0..nb {
            let row = &features[(start + b) * nf..(start + b + 1) * nf];
            for (a, c) in f.lin_cands.iter().enumerate() {
                z[b * nl + a] = lin_value(c, row[c.col]);
            }
            acc_lw[b] = 0.0;
            acc_ob[b] = 0.0;
            acc_r[b] = 0.0;
        }
        for fm in f.folds.iter() {
            let ncoef = fm.lin_coef.len().min(nl);
            for b in 0..nb {
                let zb = &z[b * nl..(b + 1) * nl];
                let mut r = fm.lin_icpt;
                for a in 0..ncoef {
                    r += fm.lin_coef[a] * zb[a];
                }
                rr[b] = r;
                acc_r[b] += r;
            }
            if use_lw {
                for b in 0..nb {
                    tmp[b] = fm.lw_base;
                }
                for tr in fm.lw_trees.iter() {
                    for b in 0..nb {
                        let row = &features[(start + b) * nf..(start + b + 1) * nf];
                        tmp[b] += tr.eval_raw(row);
                    }
                }
                for b in 0..nb {
                    acc_lw[b] += tmp[b];
                }
            }
            if use_ob {
                for b in 0..nb {
                    tmp[b] = rr[b];
                }
                for tr in fm.obl_trees.iter() {
                    for b in 0..nb {
                        let row = &features[(start + b) * nf..(start + b + 1) * nf];
                        tmp[b] += tr.eval_raw(row);
                    }
                }
                for b in 0..nb {
                    acc_ob[b] += tmp[b];
                }
            }
        }
        for b in 0..nb {
            let sh = f.weights[0] * (acc_lw[b] / kf)
                + f.weights[1] * (acc_ob[b] / kf)
                + f.weights[2] * (acc_r[b] / kf)
                + f.weights[3] * f.mean_s;
            out.push(f.tmap.to_output(sh));
        }
        start = end;
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Public regressor.
// ----------------------------------------------------------------------------

pub struct Regressor {
    seed: u64,
    fitted: Option<Fitted>,
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Regressor { seed, fitted: None }
    }

    fn fit(&mut self, train: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String> {
        self.fitted = None;
        let model = fit_impl(train, n_rows, n_cols, self.seed)?;
        self.fitted = Some(model);
        Ok(())
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        match &self.fitted {
            Some(m) => predict_impl(m, features, n_rows, n_features),
            None => Err(String::from(
                "predict: the model has not been successfully fitted",
            )),
        }
    }
}
