//! Histogram regression trees grown best-first on squared-loss gradients.
//!
//! One learner serves both forest and boosting models. With gradient `g = -y` and no L2 term,
//! the Newton leaf `-ΣG / (n + λ)` is the mean target and the split gain is the variance
//! reduction of a CART regression tree; with `g = prediction - y` it is a boosting step.

use crate::binning::BinnedMatrix;
use crate::rng::SplitMix64;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Gains at or below this are treated as no improvement.
const MIN_GAIN: f64 = 1e-12;
/// Nodes with fewer than `bins / SPARSE_NODE_FACTOR` rows are scanned by sorting instead of
/// by histogram, because clearing a full histogram per feature dominates deep, small nodes.
const SPARSE_NODE_FACTOR: usize = 2;

/// `SplitRule` is an enum for the split search contract in this module: exhaustive best-bin
/// search (random forests, boosting) or one random cut per feature (extra trees).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SplitRule {
    Best,
    Random,
}

/// `TreeParams` is a struct for the tree growth limits contract in this module.
#[derive(Clone, Debug)]
pub(crate) struct TreeParams {
    pub(crate) max_depth: Option<usize>,
    pub(crate) max_leaves: Option<usize>,
    pub(crate) min_samples_leaf: usize,
    pub(crate) l2: f64,
    /// Features examined per split, already resolved to `1..=cols`.
    pub(crate) features_per_split: usize,
    pub(crate) split_rule: SplitRule,
}

/// `TreeNode` is an enum for the fitted tree node contract in this module.
#[derive(Clone, Debug, PartialEq)]
enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

/// `RegressionTree` is a struct for the fitted regression tree contract in this module: nodes
/// that route an imputed row by raw-value thresholds to a leaf value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegressionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Clone, Copy, Debug)]
struct SplitChoice {
    feature: usize,
    bin: usize,
    gain: f64,
}

struct Frontier {
    gain_bits: u64,
    node: usize,
    depth: usize,
    rows: Vec<u32>,
    split: SplitChoice,
}

impl Frontier {
    fn key(&self) -> (u64, Reverse<usize>) {
        // Gains are positive, so their IEEE bit patterns order like the values; ties expand the
        // earliest-created node first, which keeps growth deterministic.
        (self.gain_bits, Reverse(self.node))
    }
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

struct Grower<'a> {
    data: &'a BinnedMatrix,
    gradients: &'a [f64],
    params: &'a TreeParams,
    candidates: Vec<usize>,
    hist_sum: Vec<f64>,
    hist_count: Vec<usize>,
    pairs: Vec<(u8, f64)>,
}

impl RegressionTree {
    /// Grow one tree over `rows` (training row indices, repeats allowed for bootstrap samples).
    pub(crate) fn grow(
        data: &BinnedMatrix,
        rows: Vec<u32>,
        gradients: &[f64],
        params: &TreeParams,
        rng: &mut SplitMix64,
    ) -> Self {
        let mut grower = Grower {
            data,
            gradients,
            params,
            candidates: (0..data.cols()).collect(),
            hist_sum: Vec::new(),
            hist_count: Vec::new(),
            pairs: Vec::new(),
        };
        let mut nodes = vec![TreeNode::Leaf {
            value: grower.leaf_value(&rows),
        }];
        let mut frontier = BinaryHeap::new();
        grower.enqueue(&mut frontier, 0, 0, rows, rng);
        let mut leaves = 1_usize;
        while params.max_leaves.is_none_or(|limit| leaves < limit) {
            let Some(next) = frontier.pop() else { break };
            let (left_rows, right_rows) = grower.partition(&next.rows, next.split);
            let left = nodes.len();
            nodes.push(TreeNode::Leaf {
                value: grower.leaf_value(&left_rows),
            });
            nodes.push(TreeNode::Leaf {
                value: grower.leaf_value(&right_rows),
            });
            nodes[next.node] = TreeNode::Split {
                feature: next.split.feature,
                threshold: data.threshold(next.split.feature, next.split.bin),
                left,
                right: left + 1,
            };
            leaves += 1;
            grower.enqueue(&mut frontier, left, next.depth + 1, left_rows, rng);
            grower.enqueue(&mut frontier, left + 1, next.depth + 1, right_rows, rng);
        }
        Self { nodes }
    }

    /// Route an imputed row to its leaf value.
    pub(crate) fn predict(&self, row: &[f64]) -> f64 {
        let mut index = 0;
        loop {
            match &self.nodes[index] {
                TreeNode::Leaf { value } => return *value,
                TreeNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    index = if row[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    }
                }
            }
        }
    }

    /// Multiply every leaf value by `factor` (boosting shrinkage).
    pub(crate) fn scale(&mut self, factor: f64) {
        for node in &mut self.nodes {
            if let TreeNode::Leaf { value } = node {
                *value *= factor;
            }
        }
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Grower<'_> {
    fn leaf_value(&self, rows: &[u32]) -> f64 {
        let sum: f64 = rows.iter().map(|row| self.gradients[*row as usize]).sum();
        -sum / (rows.len() as f64 + self.params.l2)
    }

    fn score(&self, sum: f64, count: usize) -> f64 {
        sum * sum / (count as f64 + self.params.l2)
    }

    fn enqueue(
        &mut self,
        frontier: &mut BinaryHeap<Frontier>,
        node: usize,
        depth: usize,
        rows: Vec<u32>,
        rng: &mut SplitMix64,
    ) {
        let depth_allows = self.params.max_depth.is_none_or(|limit| depth < limit);
        if !depth_allows || rows.len() < 2 * self.params.min_samples_leaf {
            return;
        }
        if let Some(split) = self.best_split(&rows, rng) {
            frontier.push(Frontier {
                gain_bits: split.gain.to_bits(),
                node,
                depth,
                rows,
                split,
            });
        }
    }

    fn partition(&self, rows: &[u32], split: SplitChoice) -> (Vec<u32>, Vec<u32>) {
        let codes = self.data.column(split.feature);
        rows.iter()
            .partition(|row| usize::from(codes[**row as usize]) <= split.bin)
    }

    fn best_split(&mut self, rows: &[u32], rng: &mut SplitMix64) -> Option<SplitChoice> {
        let total: f64 = rows.iter().map(|row| self.gradients[*row as usize]).sum();
        let parent = self.score(total, rows.len());
        let examined = self.params.features_per_split.min(self.candidates.len());
        if examined < self.candidates.len() {
            // Partial Fisher–Yates: the first `examined` entries become a uniform sample.
            for index in 0..examined {
                let other = index + rng.below(self.candidates.len() - index);
                self.candidates.swap(index, other);
            }
        }
        let mut best: Option<SplitChoice> = None;
        for slot in 0..examined {
            let feature = self.candidates[slot];
            let found = match self.params.split_rule {
                SplitRule::Best => self.best_bin(feature, rows, total, parent),
                SplitRule::Random => self.random_bin(feature, rows, total, parent, rng),
            };
            if let Some(choice) = found {
                if best.is_none_or(|current| choice.gain > current.gain) {
                    best = Some(choice);
                }
            }
        }
        best
    }

    fn admissible(&self, left: usize, right: usize) -> bool {
        left >= self.params.min_samples_leaf && right >= self.params.min_samples_leaf
    }

    fn best_bin(
        &mut self,
        feature: usize,
        rows: &[u32],
        total: f64,
        parent: f64,
    ) -> Option<SplitChoice> {
        let bins = self.data.bin_count(feature);
        if bins < 2 {
            return None;
        }
        if rows.len() * SPARSE_NODE_FACTOR < bins {
            self.best_bin_sorted(feature, rows, total, parent)
        } else {
            self.best_bin_histogram(feature, rows, total, parent)
        }
    }

    fn best_bin_histogram(
        &mut self,
        feature: usize,
        rows: &[u32],
        total: f64,
        parent: f64,
    ) -> Option<SplitChoice> {
        let bins = self.data.bin_count(feature);
        self.hist_sum.clear();
        self.hist_sum.resize(bins, 0.0);
        self.hist_count.clear();
        self.hist_count.resize(bins, 0);
        let codes = self.data.column(feature);
        for row in rows {
            let bin = usize::from(codes[*row as usize]);
            self.hist_sum[bin] += self.gradients[*row as usize];
            self.hist_count[bin] += 1;
        }
        let mut best: Option<SplitChoice> = None;
        let (mut left_sum, mut left_count) = (0.0, 0_usize);
        for bin in 0..bins - 1 {
            left_sum += self.hist_sum[bin];
            left_count += self.hist_count[bin];
            let right_count = rows.len() - left_count;
            if !self.admissible(left_count, right_count) {
                continue;
            }
            let gain = self.score(left_sum, left_count) + self.score(total - left_sum, right_count)
                - parent;
            if gain > MIN_GAIN && best.is_none_or(|current| gain > current.gain) {
                best = Some(SplitChoice { feature, bin, gain });
            }
        }
        best
    }

    /// Exact equivalent of the histogram scan for nodes with far fewer rows than bins: sort the
    /// node's (bin, gradient) pairs and evaluate only boundaries between occupied bins. The
    /// chosen bin is the lower occupied bin, exactly as the histogram scan picks it.
    fn best_bin_sorted(
        &mut self,
        feature: usize,
        rows: &[u32],
        total: f64,
        parent: f64,
    ) -> Option<SplitChoice> {
        let codes = self.data.column(feature);
        let gradients = self.gradients;
        self.pairs.clear();
        self.pairs.extend(
            rows.iter()
                .map(|row| (codes[*row as usize], gradients[*row as usize])),
        );
        self.pairs.sort_unstable_by_key(|(code, _)| *code);
        let mut best: Option<SplitChoice> = None;
        let (mut left_sum, mut left_count) = (0.0, 0_usize);
        for (index, (code, gradient)) in self.pairs.iter().enumerate() {
            left_sum += gradient;
            left_count += 1;
            let Some((next_code, _)) = self.pairs.get(index + 1) else {
                break;
            };
            if next_code == code {
                continue;
            }
            let right_count = rows.len() - left_count;
            if !self.admissible(left_count, right_count) {
                continue;
            }
            let gain = self.score(left_sum, left_count) + self.score(total - left_sum, right_count)
                - parent;
            if gain > MIN_GAIN && best.is_none_or(|current| gain > current.gain) {
                best = Some(SplitChoice {
                    feature,
                    bin: usize::from(*code),
                    gain,
                });
            }
        }
        best
    }

    fn random_bin(
        &self,
        feature: usize,
        rows: &[u32],
        total: f64,
        parent: f64,
        rng: &mut SplitMix64,
    ) -> Option<SplitChoice> {
        let codes = self.data.column(feature);
        let (low, high) = rows.iter().fold((u8::MAX, u8::MIN), |(low, high), row| {
            let code = codes[*row as usize];
            (low.min(code), high.max(code))
        });
        if low >= high {
            return None;
        }
        let bin = usize::from(low) + rng.below(usize::from(high - low));
        let (left_sum, left_count) = rows
            .iter()
            .filter(|row| usize::from(codes[**row as usize]) <= bin)
            .fold((0.0, 0_usize), |(sum, count), row| {
                (sum + self.gradients[*row as usize], count + 1)
            });
        let right_count = rows.len() - left_count;
        if !self.admissible(left_count, right_count) {
            return None;
        }
        let gain =
            self.score(left_sum, left_count) + self.score(total - left_sum, right_count) - parent;
        (gain > MIN_GAIN).then_some(SplitChoice { feature, bin, gain })
    }
}

#[cfg(test)]
mod tests;
