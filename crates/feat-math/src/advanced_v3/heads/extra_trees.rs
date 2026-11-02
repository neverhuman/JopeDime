use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum ExtraNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        missing_left: bool,
        left: Box<ExtraNode>,
        right: Box<ExtraNode>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ExtraTreesModel {
    pub(super) binary: bool,
    pub(super) trees: Vec<ExtraNode>,
}

impl ExtraTreesModel {
    pub(super) fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let trees = stage.get_usize("trees", 128).clamp(1, 512);
        let depth = stage.get_usize("depth", 8).clamp(1, 20);
        let min_leaf = stage.get_usize("min_leaf", 2).clamp(1, 100);
        let max_features = stage
            .get_usize("max_features", (x[0].len() as f64).sqrt().ceil() as usize)
            .clamp(1, x[0].len());
        let seed = stage.get_u64("seed", 7);
        let mut rng = Pcg32::for_stage(seed, x[0].len(), trees);
        let rows: Vec<usize> = (0..x.len()).collect();
        let mut fitted = Vec::with_capacity(trees);
        for _ in 0..trees {
            fitted.push(fit_extra_node(
                x,
                y,
                &rows,
                depth,
                min_leaf,
                max_features,
                binary,
                &mut rng,
            ));
        }
        Ok(Self {
            binary,
            trees: fitted,
        })
    }

    pub(super) fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let value = self
                    .trees
                    .iter()
                    .map(|tree| predict_extra(tree, row))
                    .sum::<f64>()
                    / self.trees.len().max(1) as f64;
                if self.binary {
                    value.clamp(0.0, 1.0)
                } else {
                    finite(value)
                }
            })
            .collect()
    }
}

/// A candidate split held while searching: score, feature index, threshold, the branch
/// orientation flag, and the two row partitions it produces.
type ExtraSplit = (f64, usize, f64, bool, Vec<usize>, Vec<usize>);

// Recursive tree fitting: every argument is threaded to the child calls, so bundling them
// into a parameter struct would add a type and a construction per node without removing a
// single value from the recursion.
#[allow(clippy::too_many_arguments)]
fn fit_extra_node(
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    depth: usize,
    min_leaf: usize,
    max_features: usize,
    binary: bool,
    rng: &mut Pcg32,
) -> ExtraNode {
    if depth == 0 || rows.len() < 2 * min_leaf {
        return ExtraNode::Leaf(extra_leaf(y, rows, binary));
    }
    let cols = x[0].len();
    let start = rng.next_u32() as usize % cols;
    let step = (rng.next_u32() as usize % cols).max(1);
    let mut best: Option<ExtraSplit> = None;
    let mut visited = BTreeSet::new();
    let mut feature = start;
    while visited.len() < max_features {
        if !visited.insert(feature) {
            feature = (feature + 1) % cols;
            continue;
        }
        let finite_values: Vec<f64> = rows
            .iter()
            .map(|&r| x[r][feature])
            .filter(|v| v.is_finite())
            .collect();
        if let (Some(lo), Some(hi)) = (
            finite_values.iter().copied().min_by(f64::total_cmp),
            finite_values.iter().copied().max_by(f64::total_cmp),
        ) {
            if hi > lo {
                let threshold = rng.range_f64(lo, hi);
                for missing_left in [false, true] {
                    let (left, right): (Vec<usize>, Vec<usize>) =
                        rows.iter().copied().partition(|&r| {
                            if x[r][feature].is_finite() {
                                x[r][feature] <= threshold
                            } else {
                                missing_left
                            }
                        });
                    if left.len() < min_leaf || right.len() < min_leaf {
                        continue;
                    }
                    let score = split_variance(y, &left) + split_variance(y, &right);
                    if best.as_ref().is_none_or(|entry| score < entry.0) {
                        best = Some((score, feature, threshold, missing_left, left, right));
                    }
                }
            }
        }
        feature = (feature + step) % cols;
    }
    let Some((_, feature, threshold, missing_left, left_rows, right_rows)) = best else {
        return ExtraNode::Leaf(extra_leaf(y, rows, binary));
    };
    ExtraNode::Split {
        feature,
        threshold,
        missing_left,
        left: Box::new(fit_extra_node(
            x,
            y,
            &left_rows,
            depth - 1,
            min_leaf,
            max_features,
            binary,
            rng,
        )),
        right: Box::new(fit_extra_node(
            x,
            y,
            &right_rows,
            depth - 1,
            min_leaf,
            max_features,
            binary,
            rng,
        )),
    }
}

fn split_variance(y: &[f64], rows: &[usize]) -> f64 {
    let mean = rows.iter().map(|&r| y[r]).sum::<f64>() / rows.len().max(1) as f64;
    rows.iter().map(|&r| (y[r] - mean).powi(2)).sum()
}

fn extra_leaf(y: &[f64], rows: &[usize], binary: bool) -> f64 {
    let sum = rows.iter().map(|&r| y[r]).sum::<f64>();
    if binary {
        (sum + 1.0) / (rows.len() as f64 + 2.0)
    } else {
        sum / rows.len().max(1) as f64
    }
}

fn predict_extra(node: &ExtraNode, row: &[f64]) -> f64 {
    match node {
        ExtraNode::Leaf(value) => *value,
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => {
            let go_left = if row[*feature].is_finite() {
                row[*feature] <= *threshold
            } else {
                *missing_left
            };
            predict_extra(if go_left { left } else { right }, row)
        }
    }
}
