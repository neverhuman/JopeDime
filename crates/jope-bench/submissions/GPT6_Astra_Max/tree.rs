use super::*;

struct Node {
    feature: usize,
    cut: u8,
    missing_left: bool,
    left: usize,
    right: usize,
    value: f64,
}

pub(crate) struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    pub(crate) fn prediction(&self, bins: &[u8], stride: usize, row: usize) -> f64 {
        let mut at = 0;
        loop {
            let node = &self.nodes[at];
            if node.feature == usize::MAX {
                return node.value;
            }
            let bin = bins[node.feature * stride + row];
            let left = if bin == 0 {
                node.missing_left
            } else {
                bin <= node.cut
            };
            at = if left { node.left } else { node.right };
        }
    }
}

struct Builder<'a> {
    matrix: &'a Matrix,
    residual: &'a [f64],
    features: &'a [usize],
    bin_counts: &'a [usize],
    min_leaf: usize,
    penalty: f64,
    min_gain: f64,
    max_depth: usize,
    nodes: Vec<Node>,
}

impl<'a> Builder<'a> {
    fn node(&mut self, rows: &mut [usize], depth: usize) -> usize {
        let mut total = 0.0;
        for &i in rows.iter() {
            total += self.residual[i];
        }
        let position = self.nodes.len();
        self.nodes.push(Node {
            feature: usize::MAX,
            cut: 0,
            missing_left: true,
            left: 0,
            right: 0,
            value: (total / (rows.len() as f64 + self.penalty)).clamp(-4.0, 4.0),
        });
        if depth >= self.max_depth || rows.len() < 2 * self.min_leaf {
            return position;
        }
        let parent = total * total / (rows.len() as f64 + self.penalty);
        let mut best_gain = self.min_gain;
        let mut best_feature = usize::MAX;
        let mut best_cut = 0u8;
        let mut best_missing_left = true;
        for &j in self.features {
            let mut sums = [0.0; BINS + 1];
            let mut counts = [0usize; BINS + 1];
            let offset = j * self.matrix.n;
            for &i in rows.iter() {
                let bin = self.matrix.bins[offset + i] as usize;
                sums[bin] += self.residual[i];
                counts[bin] += 1;
            }
            let mut finite_sum = 0.0;
            let mut finite_count = 0usize;
            for cut in 0..self.bin_counts[j] - 1 {
                if cut > 0 {
                    finite_sum += sums[cut];
                    finite_count += counts[cut];
                }
                for missing_left in [true, false] {
                    let count = finite_count + if missing_left { counts[0] } else { 0 };
                    if count < self.min_leaf || rows.len() - count < self.min_leaf {
                        continue;
                    }
                    let sum = finite_sum + if missing_left { sums[0] } else { 0.0 };
                    let other = total - sum;
                    let gain = sum * sum / (count as f64 + self.penalty)
                        + other * other / ((rows.len() - count) as f64 + self.penalty)
                        - parent;
                    if gain > best_gain {
                        best_gain = gain;
                        best_feature = j;
                        best_cut = cut as u8;
                        best_missing_left = missing_left;
                    }
                }
            }
        }
        if best_feature == usize::MAX {
            return position;
        }
        let mut split = 0;
        for i in 0..rows.len() {
            let bin = self.matrix.bins[best_feature * self.matrix.n + rows[i]];
            let left = if bin == 0 {
                best_missing_left
            } else {
                bin <= best_cut
            };
            if left {
                rows.swap(i, split);
                split += 1;
            }
        }
        let (left_rows, right_rows) = rows.split_at_mut(split);
        let left = self.node(left_rows, depth + 1);
        let right = self.node(right_rows, depth + 1);
        self.nodes[position].feature = best_feature;
        self.nodes[position].cut = best_cut;
        self.nodes[position].missing_left = best_missing_left;
        self.nodes[position].left = left;
        self.nodes[position].right = right;
        position
    }
}

fn depth_for(n: usize) -> usize {
    if n < 256 {
        4
    } else {
        6
    }
}

pub(super) fn round_limit(n: usize, p: usize) -> usize {
    let depth = depth_for(n);
    let columns = p.min(48);
    let sample = n.min(8192);
    let estimate = sample * columns * depth + (1usize << depth) * columns * (BINS + 1) + n * depth;
    (480_000_000usize / estimate.max(1)).clamp(16, 320)
}

pub(crate) struct Booster {
    rng: Random,
    rows: Vec<usize>,
    columns: Vec<usize>,
    prediction: Vec<f64>,
    bin_counts: Vec<usize>,
}

impl Booster {
    pub(crate) fn new(n: usize, prep: &Preparation, seed: u64) -> Self {
        Self {
            rng: Random::new(seed),
            rows: (0..n).collect(),
            columns: (0..prep.features.len()).collect(),
            prediction: vec![0.0; n],
            bin_counts: prep.features.iter().map(|f| f.cuts.len() + 2).collect(),
        }
    }

    pub(crate) fn step(&mut self, matrix: &Matrix, y: &[f64]) -> Tree {
        let n = matrix.n;
        let residual: Vec<f64> = y
            .iter()
            .zip(self.prediction.iter())
            .map(|(a, b)| a - b)
            .collect();
        let sample = n.min(8192);
        for i in 0..sample {
            let j = i + self.rng.below(n - i);
            self.rows.swap(i, j);
        }
        self.rng.shuffle(&mut self.columns);
        let features = &self.columns[..self.columns.len().min(48)];
        let mut energy = 0.0;
        for &i in &self.rows[..sample] {
            energy += residual[i] * residual[i] / sample as f64;
        }
        let mut builder = Builder {
            matrix,
            residual: &residual,
            features,
            bin_counts: &self.bin_counts,
            min_leaf: (n / 2048).clamp(3, 16),
            penalty: (sample as f64 * 0.001).max(1.0),
            min_gain: energy * 1.0e-10,
            max_depth: depth_for(n),
            nodes: Vec::new(),
        };
        builder.node(&mut self.rows[..sample], 0);
        let tree = Tree {
            nodes: builder.nodes,
        };
        for i in 0..n {
            self.prediction[i] += RATE * tree.prediction(&matrix.bins, n, i);
        }
        tree
    }
}
