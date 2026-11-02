use super::*;

pub(super) struct Tree {
    nodes: Vec<Node>,
}

struct Node {
    feature: usize,
    threshold: u8,
    missing_left: bool,
    left: usize,
    right: usize,
    value: f64,
}

pub(super) struct Growth {
    depth: usize,
    min_leaf: usize,
    penalty: f64,
    rate: f64,
}

impl Tree {
    pub(super) fn predict_prepared(&self, data: &Prepared, row: usize) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            if node.left == usize::MAX {
                return node.value;
            }
            let bin = data.bins[node.feature * data.n + row];
            let left = if bin == 0 {
                node.missing_left
            } else {
                bin <= node.threshold
            };
            index = if left { node.left } else { node.right };
        }
    }

    pub(super) fn predict_bins(&self, bins: &[u8]) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            if node.left == usize::MAX {
                return node.value;
            }
            let bin = bins[node.feature];
            let left = if bin == 0 {
                node.missing_left
            } else {
                bin <= node.threshold
            };
            index = if left { node.left } else { node.right };
        }
    }

    pub(super) fn learn(
        data: &Prepared,
        residual: &[f64],
        rng: &mut Random,
        growth: &Growth,
    ) -> Self {
        let mut rows = Vec::new();
        for i in 0..data.fit_n {
            if rng.index(100) < 85 {
                rows.push(i);
            }
        }
        if rows.len() < 2 * growth.min_leaf {
            rows = (0..data.fit_n).collect();
        }
        let mut columns: Vec<usize> = (0..data.features.len()).collect();
        rng.shuffle(&mut columns);
        columns.truncate(tree_width(data.features.len()));
        let mut tree = Self { nodes: Vec::new() };
        tree.grow(data, residual, rows, &columns, 0, growth);
        tree
    }

    fn grow(
        &mut self,
        data: &Prepared,
        residual: &[f64],
        rows: Vec<usize>,
        columns: &[usize],
        depth: usize,
        growth: &Growth,
    ) -> usize {
        let total: f64 = rows.iter().map(|&i| residual[i]).sum();
        let index = self.nodes.len();
        self.nodes.push(Node {
            feature: 0,
            threshold: 0,
            missing_left: false,
            left: usize::MAX,
            right: usize::MAX,
            value: growth.rate * total / (rows.len() as f64 + growth.penalty),
        });
        if depth >= growth.depth || rows.len() < 2 * growth.min_leaf {
            return index;
        }
        let parent = total * total / (rows.len() as f64 + growth.penalty);
        let mut best_gain = 1e-10;
        let mut best = None;
        for &feature in columns {
            let last_bin = data.features[feature].cuts.len() + 1;
            let mut counts = vec![0usize; last_bin + 1];
            let mut sums = vec![0.0; last_bin + 1];
            for &row in &rows {
                let bin = data.bins[feature * data.n + row] as usize;
                counts[bin] += 1;
                sums[bin] += residual[row];
            }
            let mut left_count = 0;
            let mut left_sum = 0.0;
            // The two endpoint thresholds allow a pure missing-versus-observed split.
            for threshold in 0..=last_bin {
                if threshold > 0 {
                    left_count += counts[threshold];
                    left_sum += sums[threshold];
                }
                for missing_left in [false, true] {
                    let count = left_count + if missing_left { counts[0] } else { 0 };
                    if count < growth.min_leaf || rows.len() - count < growth.min_leaf {
                        continue;
                    }
                    let sum = left_sum + if missing_left { sums[0] } else { 0.0 };
                    let other = total - sum;
                    let gain = sum * sum / (count as f64 + growth.penalty)
                        + other * other / ((rows.len() - count) as f64 + growth.penalty)
                        - parent;
                    if gain > best_gain {
                        best_gain = gain;
                        best = Some((feature, threshold as u8, missing_left));
                    }
                }
            }
        }
        if let Some((feature, threshold, missing_left)) = best {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for row in rows {
                let bin = data.bins[feature * data.n + row];
                let goes_left = if bin == 0 {
                    missing_left
                } else {
                    bin <= threshold
                };
                if goes_left {
                    left.push(row);
                } else {
                    right.push(row);
                }
            }
            let left_index = self.grow(data, residual, left, columns, depth + 1, growth);
            let right_index = self.grow(data, residual, right, columns, depth + 1, growth);
            self.nodes[index].feature = feature;
            self.nodes[index].threshold = threshold;
            self.nodes[index].missing_left = missing_left;
            self.nodes[index].left = left_index;
            self.nodes[index].right = right_index;
        }
        index
    }
}

fn tree_width(p: usize) -> usize {
    p.min(32 + p / 4)
}

pub(super) fn growth(n: usize) -> Growth {
    Growth {
        depth: if n >= 2048 { 6 } else { 4 },
        min_leaf: 3usize.max(n / 2048),
        penalty: 5.0,
        rate: 0.06,
    }
}

pub(super) fn tree_limit(data: &Prepared) -> usize {
    if data.features.is_empty() || data.scale.sd <= 1e-150 {
        return 0;
    }
    let config = growth(data.fit_n);
    let work = data
        .fit_n
        .saturating_mul(tree_width(data.features.len()))
        .saturating_mul(config.depth)
        .max(1);
    (750_000_000usize / work).clamp(32, 640)
}
