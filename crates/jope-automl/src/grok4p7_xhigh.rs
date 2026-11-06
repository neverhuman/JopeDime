#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeSet;

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

const FIT_ROWS: usize = 32768;
const BITSET_LIMIT: usize = 20_000_000;
const SCREEN_BUDGET: usize = 40_000_000;
const FEATURE_CAP: usize = 24;
const MISS_CAP: usize = 8;
const MIN_OBS: u64 = 8;
const LIN_GATE: f64 = 6.0;
const QUAD_GATE: f64 = 8.0;
const MISS_RATE_MIN: f64 = 0.02;
const MISS_RATE_MAX: f64 = 0.98;
const SCALE_EPS: f64 = 1e-12;
const MAD_SCALE: f64 = 1.4826;
const RIDGE_CLIP: f64 = 6.0;
const RIDGE_EPOCHS: usize = 16;
const RIDGE_SHRINK: f64 = 4.0;
const RIDGE_TOL: f64 = 1e-10;
const ENERGY_EPS: f64 = 1e-18;
const UNI_ROUNDS: usize = 2;
const BIN_MAX: usize = 16;
const PRODUCT_MIN_RATIO: usize = 25;
const PRODUCT_SRC: usize = 4;
const TREE_ROW_MIN: usize = 48;
const TREE_SUBSAMPLE_FROM: usize = 200;
const SPLIT_GAIN_FLOOR: f64 = 1e-6;

pub struct Regressor {
    seed: u64,
    model: Option<Model>,
}

struct Model {
    n_features: usize,
    mean_y: f64,
    y_scale: f64,
    bias: f64,
    numeric: Vec<NumCol>,
    num_beta: Vec<f64>,
    missing: Vec<MissCol>,
    miss_beta: Vec<f64>,
    products: Vec<Product>,
    uni: Vec<UniCol>,
    trees: Vec<Tree>,
}

struct NumCol {
    index: usize,
    median: f64,
    scale: f64,
    center: f64,
}

struct MissCol {
    index: usize,
    center: f64,
}

struct Product {
    left: usize,
    right: usize,
    center: f64,
    beta: f64,
}

struct UniCol {
    index: usize,
    edges: Vec<f64>,
    values: Vec<f64>,
}

struct Tree {
    nodes: Vec<Node>,
    root: usize,
}

struct Node {
    leaf: bool,
    value: f64,
    orig: usize,
    sel: usize,
    threshold: f64,
    missing_left: bool,
    left: usize,
    right: usize,
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z == 0 {
            z = 0xA5A5_A5A5_A5A5_A5A5;
        }
        Self { state: z }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn gen_index(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let n_u = n as u64;
        let limit = u64::MAX - (u64::MAX % n_u);
        loop {
            let r = self.next_u64();
            if r < limit {
                return (r % n_u) as usize;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ColStat {
    obs: u64,
    miss: u64,
    su: f64,
    su2: f64,
    su3: f64,
    su4: f64,
    sv: f64,
    sv2: f64,
    suv: f64,
    su2v: f64,
    sv_miss: f64,
    sv2_miss: f64,
}

impl ColStat {
    fn zero() -> Self {
        Self {
            obs: 0,
            miss: 0,
            su: 0.0,
            su2: 0.0,
            su3: 0.0,
            su4: 0.0,
            sv: 0.0,
            sv2: 0.0,
            suv: 0.0,
            su2v: 0.0,
            sv_miss: 0.0,
            sv2_miss: 0.0,
        }
    }
}

struct Score {
    index: usize,
    score: f64,
}

struct Kept {
    index: usize,
    median: f64,
    scale: f64,
    numeric: bool,
    edges: Vec<f64>,
    splittable: bool,
}

struct UniSpec {
    kept_pos: usize,
    edges: Vec<f64>,
    values: Vec<f64>,
}

struct Choice {
    gain: f64,
    feat: usize,
    orig: usize,
    threshold: f64,
    missing_left: bool,
}

struct TreeCtx<'a> {
    raw: &'a [f64],
    n_kept: usize,
    resid: &'a [f64],
    bins: &'a [u8],
    n_uni: usize,
    uni: &'a [UniSpec],
    kept: &'a [Kept],
    bag: &'a [usize],
    max_depth: usize,
    subset_len: usize,
    nu: f64,
    lambda: f64,
}

fn sample_sorted(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    if k == 0 || n == 0 {
        return Vec::new();
    }
    if k >= n {
        return (0..n).collect();
    }
    if k > n / 2 {
        let dropped = sample_sorted(rng, n, n - k);
        let mut drop_i = 0;
        let mut out = Vec::with_capacity(k);
        for i in 0..n {
            if drop_i < dropped.len() && dropped[drop_i] == i {
                drop_i += 1;
            } else {
                out.push(i);
            }
        }
        return out;
    }
    let mut set = BTreeSet::new();
    while set.len() < k {
        set.insert(rng.gen_index(n));
    }
    set.into_iter().collect()
}

fn sample_prefix(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    let k = k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_index(n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

fn feature_cap(n_l: usize) -> usize {
    let by_size = if n_l < 16 {
        2
    } else if n_l < 64 {
        6
    } else {
        FEATURE_CAP
    };
    let by_ratio = (n_l / 4).max(1);
    by_size.min(by_ratio).min(FEATURE_CAP)
}

fn constant_model(n_features: usize, mean_y: f64) -> Model {
    Model {
        n_features,
        mean_y,
        y_scale: 1.0,
        bias: 0.0,
        numeric: Vec::new(),
        num_beta: Vec::new(),
        missing: Vec::new(),
        miss_beta: Vec::new(),
        products: Vec::new(),
        uni: Vec::new(),
        trees: Vec::new(),
    }
}

fn welford(count: &mut u64, mean: &mut f64, m2: &mut f64, x: f64) {
    *count += 1;
    let c = *count as f64;
    let d = x - *mean;
    *mean += d / c;
    let d2 = x - *mean;
    *m2 += d * d2;
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

fn r2_linear(st: &ColStat) -> f64 {
    let c = st.obs as f64;
    if st.obs < MIN_OBS || !parts_finite(&[st.su, st.su2, st.sv, st.sv2, st.suv]) {
        return 0.0;
    }
    let mu = st.su / c;
    let mv = st.sv / c;
    let var_u = st.su2 / c - mu * mu;
    let var_v = st.sv2 / c - mv * mv;
    let cov = st.suv / c - mu * mv;
    if var_u <= ENERGY_EPS || var_v <= ENERGY_EPS {
        return 0.0;
    }
    let r = cov / var_u.sqrt() / var_v.sqrt();
    if r.is_finite() {
        (r * r).min(1.0)
    } else {
        0.0
    }
}

fn r2_quad(st: &ColStat) -> f64 {
    let c = st.obs as f64;
    if st.obs < MIN_OBS
        || !parts_finite(&[
            st.su, st.su2, st.su3, st.su4, st.sv, st.sv2, st.suv, st.su2v,
        ])
    {
        return 0.0;
    }
    let mu = st.su / c;
    let mv = st.sv / c;
    let ex2 = st.su2 / c;
    let ex3 = st.su3 / c;
    let ex4 = st.su4 / c;
    let m2 = ex2 - mu * mu;
    if m2 <= ENERGY_EPS {
        return 0.0;
    }
    let mu4 = ex4 - 4.0 * mu * ex3 + 6.0 * mu * mu * ex2 - 3.0 * mu * mu * mu * mu;
    let var_z = mu4 - m2 * m2;
    let ezy = st.su2v / c - 2.0 * mu * (st.suv / c) + mu * mu * (st.sv / c);
    let cov = ezy - m2 * mv;
    let var_v = st.sv2 / c - mv * mv;
    if var_z <= ENERGY_EPS || var_v <= ENERGY_EPS {
        return 0.0;
    }
    let r = cov / var_z.sqrt() / var_v.sqrt();
    if r.is_finite() {
        (r * r).min(1.0)
    } else {
        0.0
    }
}

fn r2_missing(st: &ColStat) -> f64 {
    if st.obs < MIN_OBS || st.miss < MIN_OBS {
        return 0.0;
    }
    if !parts_finite(&[st.sv, st.sv2, st.sv_miss, st.sv2_miss]) {
        return 0.0;
    }
    let n0 = st.obs as f64;
    let n1 = st.miss as f64;
    let n = n0 + n1;
    if n < MIN_OBS as f64 {
        return 0.0;
    }
    let rate = n1 / n;
    if rate < MISS_RATE_MIN || rate > MISS_RATE_MAX {
        return 0.0;
    }
    let sum_v = st.sv + st.sv_miss;
    let sum_v2 = st.sv2 + st.sv2_miss;
    let mean_v = sum_v / n;
    let var_v = sum_v2 / n - mean_v * mean_v;
    let var_i = rate * (1.0 - rate);
    let cov = st.sv_miss / n - rate * mean_v;
    if var_v <= ENERGY_EPS || var_i <= ENERGY_EPS {
        return 0.0;
    }
    let r = cov / var_v.sqrt() / var_i.sqrt();
    if r.is_finite() {
        (r * r).min(1.0)
    } else {
        0.0
    }
}

fn parts_finite(xs: &[f64]) -> bool {
    xs.iter().all(|v| v.is_finite())
}

fn take_top(mut items: Vec<Score>, cap: usize) -> Vec<Score> {
    items.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index)));
    if items.len() > cap {
        items.truncate(cap);
    }
    items
}

fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn population_std(values: &[f64]) -> f64 {
    let mut count = 0u64;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for &v in values {
        welford(&mut count, &mut mean, &mut m2, v);
    }
    if count == 0 {
        return 0.0;
    }
    let var = m2 / count as f64;
    if var > 0.0 && var.is_finite() {
        var.sqrt()
    } else {
        0.0
    }
}

fn count_unique(sorted: &[f64]) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let mut n = 1usize;
    let mut prev = sorted[0];
    for &v in &sorted[1..] {
        if v != prev {
            n += 1;
            prev = v;
        }
    }
    n
}

fn push_cut(edges: &mut Vec<f64>, lo: f64, hi: f64) {
    if !(lo < hi) {
        return;
    }
    let mid = 0.5 * (lo + hi);
    if mid.is_finite() && lo < mid && mid < hi {
        edges.push(mid);
    } else {
        edges.push(lo);
    }
}

fn strictly_increasing(mut edges: Vec<f64>) -> Vec<f64> {
    edges.retain(|e| e.is_finite());
    edges.sort_by(|a, b| a.total_cmp(b));
    let mut out = Vec::new();
    for e in edges {
        if out.last().map(|p| e > *p).unwrap_or(true) {
            out.push(e);
        }
    }
    out
}

fn build_edges(sorted: &[f64], bin_target: usize) -> Vec<f64> {
    if sorted.len() < 2 {
        return Vec::new();
    }
    let mut unique = Vec::new();
    for &v in sorted {
        if unique.last().map(|u| *u != v).unwrap_or(true) {
            unique.push(v);
        }
    }
    if unique.len() < 2 {
        return Vec::new();
    }
    let mut edges = Vec::new();
    if unique.len() <= bin_target {
        for i in 0..unique.len() - 1 {
            push_cut(&mut edges, unique[i], unique[i + 1]);
        }
    } else {
        let n = sorted.len();
        for k in 1..bin_target {
            let mut pos = k.saturating_mul(n) / bin_target;
            if pos >= n {
                pos = n - 1;
            }
            edges.push(sorted[pos]);
        }
    }
    let mut edges = strictly_increasing(edges);
    if edges.is_empty() {
        push_cut(&mut edges, unique[0], unique[unique.len() - 1]);
        edges = strictly_increasing(edges);
    }
    if edges.len() > bin_target.saturating_sub(1) {
        edges.truncate(bin_target.saturating_sub(1));
    }
    edges
}

/// Finite bin in `1..=edges.len()+1`. Bin 0 is reserved for missing.
/// A finite value goes left of a threshold when `x <= threshold`.
fn finite_bin(x: f64, edges: &[f64]) -> usize {
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x <= edges[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo + 1
}

fn go_left(x: f64, threshold: f64, missing_left: bool) -> bool {
    if x.is_nan() {
        missing_left
    } else {
        x <= threshold
    }
}

fn screen_hit(screen_all: bool, bits: Option<&[u64]>, screen_rows: &[usize], i: usize) -> bool {
    if screen_all {
        return true;
    }
    if let Some(bits) = bits {
        let word = bits[i / 64];
        return ((word >> (i % 64)) & 1) == 1;
    }
    screen_rows.binary_search(&i).is_ok()
}

fn profile_column(
    train: &[f64],
    n_cols: usize,
    rows: &[usize],
    index: usize,
    bin_target: usize,
) -> Option<Kept> {
    let mut finite = Vec::new();
    let mut missing = 0usize;
    for &r in rows {
        let x = train[r * n_cols + index];
        if x.is_nan() {
            missing += 1;
        } else {
            finite.push(x);
        }
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let unique = count_unique(&finite);
    if finite.is_empty() || (unique <= 1 && missing == 0) {
        return None;
    }
    let median = median_sorted(&finite);
    let mut dev = Vec::with_capacity(finite.len());
    for &v in &finite {
        dev.push((v - median).abs());
    }
    dev.sort_by(|a, b| a.total_cmp(b));
    let mad = median_sorted(&dev);
    let std = population_std(&finite);
    let scale = if mad > SCALE_EPS {
        MAD_SCALE * mad
    } else {
        std
    };
    let numeric = finite.len() >= 2 && scale > SCALE_EPS && scale.is_finite();
    let edges = build_edges(&finite, bin_target);
    let splittable = !edges.is_empty() || (missing > 0 && !finite.is_empty());
    if !numeric && !splittable {
        return None;
    }
    Some(Kept {
        index,
        median: if median.is_finite() { median } else { 0.0 },
        scale: if numeric { scale } else { 0.0 },
        numeric,
        edges,
        splittable,
    })
}

fn center_column(z: &mut [f64], n: usize, p: usize, j: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        let v = z[i * p + j];
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    let mean = if count > 0 {
        sum / count as f64
    } else {
        0.0
    };
    let mean = if mean.is_finite() { mean } else { 0.0 };
    for i in 0..n {
        let v = z[i * p + j];
        let centered = if v.is_finite() { v - mean } else { 0.0 };
        z[i * p + j] = finite_or_zero(centered);
    }
    mean
}

fn sq_gain(sum: f64, cnt: u32, lambda: f64) -> f64 {
    let c = cnt as f64 + lambda;
    if !(c > 0.0) || !sum.is_finite() {
        return 0.0;
    }
    let g = sum * sum / c;
    if g.is_finite() {
        g
    } else {
        0.0
    }
}

fn better(a: &Choice, b: &Choice) -> bool {
    match a.gain.total_cmp(&b.gain) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match a.orig.cmp(&b.orig) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => match a.threshold.total_cmp(&b.threshold) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => !a.missing_left && b.missing_left,
            },
        },
    }
}

fn accept_split(
    best: &mut Option<Choice>,
    c_l: u32,
    s_l: f64,
    c_r: u32,
    s_r: f64,
    threshold: f64,
    missing_left: bool,
    feat: usize,
    orig: usize,
    min_count: usize,
    lambda: f64,
    parent: f64,
    min_reduction: f64,
) {
    if (c_l as usize) < min_count || (c_r as usize) < min_count {
        return;
    }
    let red = sq_gain(s_l, c_l, lambda) + sq_gain(s_r, c_r, lambda) - parent;
    if !red.is_finite() || red <= min_reduction {
        return;
    }
    let choice = Choice {
        gain: red,
        feat,
        orig,
        threshold,
        missing_left,
    };
    let replace = match best {
        None => true,
        Some(cur) => better(&choice, cur),
    };
    if replace {
        *best = Some(choice);
    }
}

fn leaf_base(subset_len: usize) -> usize {
    (subset_len / 64).clamp(1, 200)
}

fn leaf_node(value: f64) -> Node {
    Node {
        leaf: true,
        value: finite_or_zero(value),
        orig: 0,
        sel: 0,
        threshold: 0.0,
        missing_left: false,
        left: 0,
        right: 0,
    }
}

fn leaf_value(ids: &[usize], resid: &[f64], nu: f64, lambda: f64) -> f64 {
    let mut sum = 0.0;
    for &i in ids {
        sum += resid[i];
    }
    let c = ids.len() as f64;
    finite_or_zero(nu * sum / (c + lambda))
}

impl<'a> TreeCtx<'a> {
    fn find_split(&self, ids: &[usize], min_count: usize) -> Option<Choice> {
        if ids.len() < 2 {
            return None;
        }
        let min_reduction = SPLIT_GAIN_FLOOR * ids.len() as f64;
        let mut best: Option<Choice> = None;
        for &feat in self.bag {
            if feat >= self.n_uni {
                continue;
            }
            let edges = &self.uni[feat].edges;
            let orig = self.kept[self.uni[feat].kept_pos].index;
            let k_finite = edges.len() + 1;
            let n_slots = k_finite + 1;
            let mut cnt = vec![0u32; n_slots];
            let mut sum = vec![0.0; n_slots];
            for &i in ids {
                let b = self.bins[i * self.n_uni + feat] as usize;
                if b < n_slots {
                    cnt[b] = cnt[b].saturating_add(1);
                    sum[b] += self.resid[i];
                }
            }
            let mut c_all = 0u32;
            let mut s_all = 0.0;
            for b in 0..n_slots {
                c_all = c_all.saturating_add(cnt[b]);
                s_all += sum[b];
            }
            if c_all == 0 {
                continue;
            }
            let parent = sq_gain(s_all, c_all, self.lambda);
            let c_miss = cnt[0];
            let s_miss = sum[0];
            let mut c_fin = 0u32;
            let mut s_fin = 0.0;
            for b in 1..n_slots {
                c_fin = c_fin.saturating_add(cnt[b]);
                s_fin += sum[b];
            }
            accept_split(
                &mut best,
                c_fin,
                s_fin,
                c_miss,
                s_miss,
                f64::INFINITY,
                false,
                feat,
                orig,
                min_count,
                self.lambda,
                parent,
                min_reduction,
            );
            accept_split(
                &mut best,
                c_miss,
                s_miss,
                c_fin,
                s_fin,
                f64::NEG_INFINITY,
                true,
                feat,
                orig,
                min_count,
                self.lambda,
                parent,
                min_reduction,
            );
            let mut c_left = 0u32;
            let mut s_left = 0.0;
            for t in 1..k_finite {
                c_left = c_left.saturating_add(cnt[t]);
                s_left += sum[t];
                let c_right = c_fin.saturating_sub(c_left);
                let s_right = s_fin - s_left;
                let threshold = edges[t - 1];
                accept_split(
                    &mut best,
                    c_left.saturating_add(c_miss),
                    s_left + s_miss,
                    c_right,
                    s_right,
                    threshold,
                    true,
                    feat,
                    orig,
                    min_count,
                    self.lambda,
                    parent,
                    min_reduction,
                );
                accept_split(
                    &mut best,
                    c_left,
                    s_left,
                    c_right.saturating_add(c_miss),
                    s_right + s_miss,
                    threshold,
                    false,
                    feat,
                    orig,
                    min_count,
                    self.lambda,
                    parent,
                    min_reduction,
                );
            }
        }
        best
    }

    fn build(&self, nodes: &mut Vec<Node>, ids: &[usize], depth: usize) -> usize {
        let id = nodes.len();
        nodes.push(leaf_node(0.0));
        let base = leaf_base(self.subset_len);
        let min_count = base.min(ids.len() / 2).max(1);
        let make_leaf = depth >= self.max_depth
            || ids.len() < min_count.saturating_mul(2)
            || ids.is_empty();
        if make_leaf {
            nodes[id] = leaf_node(leaf_value(ids, self.resid, self.nu, self.lambda));
            return id;
        }
        let choice = match self.find_split(ids, min_count) {
            Some(choice) => choice,
            None => {
                nodes[id] = leaf_node(leaf_value(ids, self.resid, self.nu, self.lambda));
                return id;
            }
        };
        let sel = self.uni[choice.feat].kept_pos;
        let mut left_ids = Vec::new();
        let mut right_ids = Vec::new();
        for &i in ids {
            let x = if self.n_kept > 0 && sel < self.n_kept {
                self.raw[i * self.n_kept + sel]
            } else {
                f64::NAN
            };
            if go_left(x, choice.threshold, choice.missing_left) {
                left_ids.push(i);
            } else {
                right_ids.push(i);
            }
        }
        if left_ids.is_empty() || right_ids.is_empty() {
            nodes[id] = leaf_node(leaf_value(ids, self.resid, self.nu, self.lambda));
            return id;
        }
        let left = self.build(nodes, &left_ids, depth + 1);
        let right = self.build(nodes, &right_ids, depth + 1);
        nodes[id] = Node {
            leaf: false,
            value: 0.0,
            orig: choice.orig,
            sel,
            threshold: choice.threshold,
            missing_left: choice.missing_left,
            left,
            right,
        };
        id
    }
}

fn walk_predict(nodes: &[Node], root: usize, row: &[f64]) -> f64 {
    let mut i = root;
    for _ in 0..64 {
        if i >= nodes.len() {
            return 0.0;
        }
        let n = &nodes[i];
        if n.leaf {
            return finite_or_zero(n.value);
        }
        let x = if n.orig < row.len() { row[n.orig] } else { f64::NAN };
        let left = go_left(x, n.threshold, n.missing_left);
        i = if left { n.left } else { n.right };
    }
    0.0
}

fn apply_tree(tree: &Tree, raw: &[f64], n_kept: usize, resid: &mut [f64]) {
    let n = resid.len();
    for i in 0..n {
        let row_at = i * n_kept;
        let mut node = tree.root;
        let mut value = 0.0;
        for _ in 0..64 {
            if node >= tree.nodes.len() {
                value = 0.0;
                break;
            }
            let nref = &tree.nodes[node];
            if nref.leaf {
                value = finite_or_zero(nref.value);
                break;
            }
            let x = if n_kept > 0 && nref.sel < n_kept {
                raw[row_at + nref.sel]
            } else {
                f64::NAN
            };
            let left = go_left(x, nref.threshold, nref.missing_left);
            node = if left { nref.left } else { nref.right };
        }
        resid[i] -= value;
        if !resid[i].is_finite() {
            resid[i] = 0.0;
        }
    }
}

fn tree_schedule(n_l: usize) -> Option<(usize, usize, f64)> {
    if n_l < TREE_ROW_MIN {
        None
    } else if n_l < 300 {
        Some((2, 12, 0.12))
    } else if n_l < 3000 {
        Some((3, 24, 0.08))
    } else {
        Some((4, 36, 0.07))
    }
}

fn scrub(resid: &mut [f64]) {
    for r in resid.iter_mut() {
        if !r.is_finite() {
            *r = 0.0;
        }
    }
}

fn fit_ridge(
    raw: &[f64],
    n_l: usize,
    kept: &[Kept],
    indicators: &[Score],
    train: &[f64],
    n_cols: usize,
    learn_rows: &[usize],
    resid: &mut [f64],
) -> (Vec<NumCol>, Vec<f64>, Vec<MissCol>, Vec<f64>, Vec<Product>) {
    let n_num = kept.iter().filter(|k| k.numeric).count();
    let n_miss = indicators.len();
    let n_src = n_num.min(PRODUCT_SRC);
    let n_pairs_full = n_src.saturating_sub(1).saturating_mul(n_src) / 2;
    let width_with = n_num + n_miss + n_pairs_full;
    let use_pairs = n_src >= 2 && n_l >= PRODUCT_MIN_RATIO.saturating_mul(width_with);
    let n_pairs = if use_pairs { n_pairs_full } else { 0 };
    let p = n_num + n_miss + n_pairs;
    let mut numeric = Vec::with_capacity(n_num);
    let mut missing = Vec::with_capacity(n_miss);
    let mut products = Vec::with_capacity(n_pairs);
    if p == 0 || n_l == 0 {
        return (numeric, Vec::new(), missing, Vec::new(), products);
    }
    let cells = match n_l.checked_mul(p) {
        Some(cells) => cells,
        None => return (numeric, Vec::new(), missing, Vec::new(), products),
    };
    let mut z = vec![0.0; cells];
    let n_kept = kept.len();
    let mut col = 0usize;
    for (j, k) in kept.iter().enumerate() {
        if !k.numeric {
            continue;
        }
        for i in 0..n_l {
            let x = raw[i * n_kept + j];
            let filled = if x.is_nan() { k.median } else { x };
            let scaled = (filled - k.median) / k.scale;
            z[i * p + col] = if scaled.is_finite() {
                scaled.clamp(-RIDGE_CLIP, RIDGE_CLIP)
            } else {
                0.0
            };
        }
        let center = center_column(&mut z, n_l, p, col);
        numeric.push(NumCol {
            index: k.index,
            median: k.median,
            scale: k.scale,
            center,
        });
        col += 1;
    }
    for ind in indicators {
        for i in 0..n_l {
            let x = train[learn_rows[i] * n_cols + ind.index];
            z[i * p + col] = if x.is_nan() { 1.0 } else { 0.0 };
        }
        let center = center_column(&mut z, n_l, p, col);
        missing.push(MissCol {
            index: ind.index,
            center,
        });
        col += 1;
    }
    if use_pairs {
        for a in 0..n_src {
            for b in (a + 1)..n_src {
                for i in 0..n_l {
                    let prod = z[i * p + a] * z[i * p + b];
                    z[i * p + col] = finite_or_zero(prod);
                }
                let center = center_column(&mut z, n_l, p, col);
                products.push(Product {
                    left: a,
                    right: b,
                    center,
                    beta: 0.0,
                });
                col += 1;
            }
        }
    }
    let y_std = resid.to_vec();
    let mut beta = vec![0.0; p];
    let mut energy = vec![0.0; p];
    for j in 0..p {
        let mut dz = 0.0;
        for i in 0..n_l {
            let zij = z[i * p + j];
            dz += zij * zij;
        }
        energy[j] = if dz.is_finite() { dz } else { 0.0 };
    }
    let shrink = 1.0 + RIDGE_SHRINK * (p as f64) / (n_l as f64);
    for _epoch in 0..RIDGE_EPOCHS {
        let mut max_delta = 0.0;
        for j in 0..p {
            let dz = energy[j];
            if dz <= ENERGY_EPS {
                continue;
            }
            let mut dot = 0.0;
            for i in 0..n_l {
                dot += z[i * p + j] * resid[i];
            }
            if !dot.is_finite() {
                continue;
            }
            let beta_new = (dot + beta[j] * dz) / (dz * shrink);
            if !beta_new.is_finite() {
                continue;
            }
            let delta = beta_new - beta[j];
            let ad = delta.abs();
            if ad > max_delta {
                max_delta = ad;
            }
            if ad == 0.0 {
                continue;
            }
            for i in 0..n_l {
                resid[i] -= delta * z[i * p + j];
            }
            beta[j] = beta_new;
        }
        scrub(resid);
        if max_delta < RIDGE_TOL {
            break;
        }
    }
    for b in &mut beta {
        if !b.is_finite() {
            *b = 0.0;
        }
    }
    for i in 0..n_l {
        let mut acc = 0.0;
        let row = i * p;
        for j in 0..p {
            acc += z[row + j] * beta[j];
        }
        resid[i] = finite_or_zero(y_std[i] - acc);
    }
    let mut num_beta = Vec::with_capacity(n_num);
    for j in 0..n_num {
        num_beta.push(beta[j]);
    }
    let mut miss_beta = Vec::with_capacity(n_miss);
    for j in 0..n_miss {
        miss_beta.push(beta[n_num + j]);
    }
    for (t, prod) in products.iter_mut().enumerate() {
        prod.beta = beta[n_num + n_miss + t];
    }
    (numeric, num_beta, missing, miss_beta, products)
}

fn fit_univariate(raw: &[f64], n_l: usize, kept: &[Kept], resid: &mut [f64]) -> (Vec<UniSpec>, Vec<u8>) {
    let mut uni = Vec::new();
    for (pos, k) in kept.iter().enumerate() {
        if k.splittable {
            let slots = k.edges.len() + 2;
            uni.push(UniSpec {
                kept_pos: pos,
                edges: k.edges.clone(),
                values: vec![0.0; slots],
            });
        }
    }
    let n_uni = uni.len();
    let n_kept = kept.len();
    if n_uni == 0 || n_l == 0 {
        return (uni, Vec::new());
    }
    let bin_cells = match n_l.checked_mul(n_uni) {
        Some(cells) => cells,
        None => return (Vec::new(), Vec::new()),
    };
    let mut bins = vec![0u8; bin_cells];
    for i in 0..n_l {
        for (j, u) in uni.iter().enumerate() {
            let x = raw[i * n_kept + u.kept_pos];
            let b = if x.is_nan() {
                0
            } else {
                finite_bin(x, &u.edges)
            };
            bins[i * n_uni + j] = b as u8;
        }
    }
    for _round in 0..UNI_ROUNDS {
        for (j, u) in uni.iter_mut().enumerate() {
            let slots = u.values.len();
            let k_finite = u.edges.len() + 1;
            let lambda = (40.0 * k_finite as f64 / n_l as f64).max(1.0);
            let mut sum = vec![0.0; slots];
            let mut cnt = vec![0u64; slots];
            for i in 0..n_l {
                let b = bins[i * n_uni + j] as usize;
                if b < slots {
                    sum[b] += resid[i];
                    cnt[b] += 1;
                }
            }
            let mut step = vec![0.0; slots];
            let mut wsum = 0.0;
            let mut wcnt = 0.0;
            for b in 0..slots {
                let c = cnt[b] as f64;
                if c > 0.0 {
                    let v = sum[b] / (c + lambda);
                    step[b] = finite_or_zero(v);
                }
                wsum += c * step[b];
                wcnt += c;
            }
            if wcnt > 0.0 {
                let mean = wsum / wcnt;
                if mean.is_finite() {
                    for b in 0..slots {
                        step[b] = finite_or_zero(step[b] - mean);
                    }
                }
            }
            for i in 0..n_l {
                let b = bins[i * n_uni + j] as usize;
                if b < slots {
                    resid[i] -= step[b];
                    if !resid[i].is_finite() {
                        resid[i] = 0.0;
                    }
                }
            }
            for b in 0..slots {
                u.values[b] = finite_or_zero(u.values[b] + step[b]);
            }
        }
    }
    (uni, bins)
}

fn fit_trees(
    seed: u64,
    raw: &[f64],
    n_kept: usize,
    kept: &[Kept],
    uni: &[UniSpec],
    bins: &[u8],
    resid: &mut [f64],
    n_l: usize,
) -> Vec<Tree> {
    let (max_depth, n_trees, nu) = match tree_schedule(n_l) {
        Some(schedule) => schedule,
        None => return Vec::new(),
    };
    let n_uni = uni.len();
    if n_uni == 0 || bins.is_empty() {
        return Vec::new();
    }
    let mut trees = Vec::new();
    for t in 0..n_trees {
        let mut rng = Rng::new(
            seed.wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add(0xD1B5_4A32_D192_ED03),
        );
        let n_sub = if n_l >= TREE_SUBSAMPLE_FROM {
            (n_l * 3) / 4
        } else {
            n_l
        };
        if n_sub < 2 {
            continue;
        }
        let subset = sample_prefix(&mut rng, n_l, n_sub);
        let n_try = if n_uni > 2 {
            ((n_uni * 3) / 4).max(2)
        } else {
            n_uni
        };
        let mut order: Vec<usize> = (0..n_uni).collect();
        for i in 0..n_try {
            let j = i + rng.gen_index(n_uni - i);
            order.swap(i, j);
        }
        order.truncate(n_try);
        order.sort_unstable();
        let lambda = leaf_base(subset.len()) as f64;
        let mut nodes = Vec::new();
        let root = {
            let ctx = TreeCtx {
                raw,
                n_kept,
                resid,
                bins,
                n_uni,
                uni,
                kept,
                bag: &order,
                max_depth,
                subset_len: subset.len(),
                nu,
                lambda,
            };
            ctx.build(&mut nodes, &subset, 0)
        };
        let root_is_leaf = nodes.get(root).map(|n| n.leaf).unwrap_or(true);
        if nodes.is_empty() || root_is_leaf {
            continue;
        }
        let tree = Tree { nodes, root };
        apply_tree(&tree, raw, n_kept, resid);
        trees.push(tree);
    }
    trees
}

fn mean_bias(resid: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0u64;
    for &r in resid {
        if r.is_finite() {
            sum += r;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        finite_or_zero(sum / count as f64)
    }
}

fn predict_row(model: &Model, row: &[f64]) -> f64 {
    let mut f = model.bias;
    let mut zc = [0.0; FEATURE_CAP];
    for (j, col) in model.numeric.iter().enumerate() {
        if j >= FEATURE_CAP || col.index >= row.len() {
            break;
        }
        let raw = row[col.index];
        let filled = if raw.is_nan() { col.median } else { raw };
        let scaled = (filled - col.median) / col.scale;
        let clipped = if scaled.is_finite() {
            scaled.clamp(-RIDGE_CLIP, RIDGE_CLIP)
        } else {
            0.0
        };
        let z = finite_or_zero(clipped - col.center);
        zc[j] = z;
        let beta = model.num_beta.get(j).copied().unwrap_or(0.0);
        f += beta * z;
    }
    for (j, col) in model.missing.iter().enumerate() {
        if col.index >= row.len() {
            break;
        }
        let ind = if row[col.index].is_nan() { 1.0 } else { 0.0 };
        let z = ind - col.center;
        let beta = model.miss_beta.get(j).copied().unwrap_or(0.0);
        f += beta * finite_or_zero(z);
    }
    for prod in &model.products {
        if prod.left >= FEATURE_CAP || prod.right >= FEATURE_CAP {
            continue;
        }
        let value = zc[prod.left] * zc[prod.right] - prod.center;
        f += prod.beta * finite_or_zero(value);
    }
    for u in &model.uni {
        if u.index >= row.len() {
            continue;
        }
        let raw = row[u.index];
        let b = if raw.is_nan() {
            0
        } else {
            finite_bin(raw, &u.edges)
        };
        if let Some(v) = u.values.get(b) {
            f += *v;
        }
    }
    for tree in &model.trees {
        f += walk_predict(&tree.nodes, tree.root, row);
    }
    let y = model.mean_y + model.y_scale * f;
    if y.is_finite() {
        y
    } else {
        model.mean_y
    }
}

fn validate_shape(n_rows: usize, n_cols: usize, len: usize, what: &str) -> Result<usize, String> {
    if n_rows == 0 || n_cols == 0 {
        return Err(format!("{what} dimensions must be positive"));
    }
    match n_rows.checked_mul(n_cols) {
        Some(n) if n == len => Ok(n),
        _ => Err(format!("{what} length mismatch or dimension overflow")),
    }
}

fn build_model(seed: u64, train: &[f64], n_rows: usize, n_cols: usize) -> Result<Model, String> {
    validate_shape(n_rows, n_cols, train.len(), "train")?;
    let p = n_cols - 1;
    let mut y_count = 0u64;
    let mut y_mean = 0.0;
    let mut y_m2 = 0.0;
    if p == 0 {
        for i in 0..n_rows {
            let y = train[i * n_cols];
            if !y.is_finite() {
                return Err("nonfinite target".to_string());
            }
            welford(&mut y_count, &mut y_mean, &mut y_m2, y);
        }
        return Ok(constant_model(0, finite_or_zero(y_mean)));
    }

    let n_l = n_rows.min(FIT_ROWS);
    let mut rng = Rng::new(seed);
    let learn_rows = if n_l == n_rows {
        (0..n_rows).collect::<Vec<usize>>()
    } else {
        sample_sorted(&mut rng, n_rows, n_l)
    };
    let n_screen = n_l.min(400.max(SCREEN_BUDGET / p.max(1)));
    let screen_rows = if n_screen >= learn_rows.len() {
        learn_rows.clone()
    } else {
        let picked = sample_sorted(&mut rng, learn_rows.len(), n_screen);
        let mut rows: Vec<usize> = picked.iter().map(|&i| learn_rows[i]).collect();
        rows.sort_unstable();
        rows
    };
    let screen_all = screen_rows.len() == n_rows;
    let bits = if !screen_all && n_rows <= BITSET_LIMIT {
        let mut bits = vec![0u64; (n_rows + 63) / 64];
        for &r in &screen_rows {
            bits[r / 64] |= 1u64 << (r % 64);
        }
        Some(bits)
    } else {
        None
    };

    let mut stats = vec![ColStat::zero(); p];
    for i in 0..n_rows {
        let base = i * n_cols;
        let y = train[base + p];
        if !y.is_finite() {
            return Err("nonfinite target".to_string());
        }
        welford(&mut y_count, &mut y_mean, &mut y_m2, y);
        let screened = screen_hit(screen_all, bits.as_deref(), &screen_rows, i);
        let v = if screened { y.asinh() } else { 0.0 };
        for j in 0..p {
            let x = train[base + j];
            if x.is_infinite() {
                return Err("infinite feature".to_string());
            }
            if !screened || !v.is_finite() {
                continue;
            }
            let st = &mut stats[j];
            if x.is_nan() {
                st.miss += 1;
                st.sv_miss += v;
                st.sv2_miss += v * v;
            } else {
                let u = x.asinh();
                if u.is_finite() {
                    let u2 = u * u;
                    st.obs += 1;
                    st.su += u;
                    st.su2 += u2;
                    st.su3 += u2 * u;
                    st.su4 += u2 * u2;
                    st.sv += v;
                    st.sv2 += v * v;
                    st.suv += u * v;
                    st.su2v += u2 * v;
                }
            }
        }
    }
    let y_var = if y_count > 0 {
        y_m2 / y_count as f64
    } else {
        0.0
    };
    let y_scale = if y_var > 0.0 && y_var.is_finite() {
        y_var.sqrt()
    } else {
        0.0
    };
    let mean_y = finite_or_zero(y_mean);
    let scale_floor = SCALE_EPS * mean_y.abs().max(1.0);
    if !(y_scale > scale_floor) {
        return Ok(constant_model(p, mean_y));
    }

    let mut numeric_scores = Vec::new();
    let mut missing_scores = Vec::new();
    for (j, st) in stats.iter().enumerate() {
        let lin = r2_linear(st);
        let quad = r2_quad(st);
        let n_obs = st.obs as f64;
        if n_obs * lin > LIN_GATE || n_obs * quad > QUAD_GATE {
            let score = lin.max(quad);
            if score.is_finite() && score > 0.0 {
                numeric_scores.push(Score { index: j, score });
            }
        }
        let miss = r2_missing(st);
        let n_tot = (st.obs + st.miss) as f64;
        if n_tot * miss > LIN_GATE && miss.is_finite() && miss > 0.0 {
            missing_scores.push(Score {
                index: j,
                score: miss,
            });
        }
    }
    drop(stats);
    let selected = take_top(numeric_scores, feature_cap(n_l));
    let indicators = take_top(missing_scores, MISS_CAP);
    if selected.is_empty() && indicators.is_empty() {
        return Ok(constant_model(p, mean_y));
    }

    let bin_target = (n_l / 20).clamp(2, BIN_MAX);
    let mut kept = Vec::new();
    for s in &selected {
        if let Some(col) = profile_column(train, n_cols, &learn_rows, s.index, bin_target) {
            kept.push(col);
        }
    }
    if kept.is_empty() && indicators.is_empty() {
        return Ok(constant_model(p, mean_y));
    }

    let n_kept = kept.len();
    let raw_cells = match n_l.checked_mul(n_kept.max(1)) {
        Some(c) => c,
        None => return Ok(constant_model(p, mean_y)),
    };
    let mut raw = if n_kept == 0 {
        Vec::new()
    } else {
        vec![0.0; raw_cells]
    };
    let mut resid = vec![0.0; n_l];
    for (i, &row) in learn_rows.iter().enumerate() {
        let base = row * n_cols;
        for (j, k) in kept.iter().enumerate() {
            raw[i * n_kept + j] = train[base + k.index];
        }
        let ys = (train[base + p] - mean_y) / y_scale;
        resid[i] = finite_or_zero(ys);
    }

    let (numeric, num_beta, missing, miss_beta, products) = fit_ridge(
        &raw,
        n_l,
        &kept,
        &indicators,
        train,
        n_cols,
        &learn_rows,
        &mut resid,
    );
    let (uni_spec, bins) = fit_univariate(&raw, n_l, &kept, &mut resid);
    let trees = fit_trees(seed, &raw, n_kept, &kept, &uni_spec, &bins, &mut resid, n_l);
    let bias = mean_bias(&resid);
    let mut uni = Vec::with_capacity(uni_spec.len());
    for spec in uni_spec {
        let index = kept[spec.kept_pos].index;
        uni.push(UniCol {
            index,
            edges: spec.edges,
            values: spec.values,
        });
    }
    Ok(Model {
        n_features: p,
        mean_y,
        y_scale,
        bias,
        numeric,
        num_beta,
        missing,
        miss_beta,
        products,
        uni,
        trees,
    })
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self { seed, model: None }
    }

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String> {
        self.model = None;
        match build_model(self.seed, train, n_rows, n_cols) {
            Ok(model) => {
                self.model = Some(model);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let model = match &self.model {
            Some(model) => model,
            None => return Err("not fitted".to_string()),
        };
        if n_features != model.n_features {
            return Err("feature count mismatch".to_string());
        }
        let need = match n_rows.checked_mul(n_features) {
            Some(n) => n,
            None => {
                return Err("prediction length mismatch or dimension overflow".to_string());
            }
        };
        if features.len() != need {
            return Err("prediction length mismatch or dimension overflow".to_string());
        }
        let mut out = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let start = i * n_features;
            let row = &features[start..start + n_features];
            for &v in row {
                if v.is_infinite() {
                    return Err("infinite feature".to_string());
                }
            }
            out.push(predict_row(model, row));
        }
        Ok(out)
    }
}
