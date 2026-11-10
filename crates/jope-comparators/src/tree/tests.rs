use super::*;

fn params(split_rule: SplitRule) -> TreeParams {
    TreeParams {
        max_depth: None,
        max_leaves: None,
        min_samples_leaf: 1,
        l2: 0.0,
        features_per_split: 2,
        split_rule,
    }
}

#[test]
fn a_step_function_is_recovered_exactly() {
    // Feature 0 is a step at 0.5; feature 1 is constant and uninformative.
    let dense = [0.1, 7.0, 0.2, 7.0, 0.8, 7.0, 0.9, 7.0];
    let targets = [1.0, 1.0, 5.0, 5.0];
    let gradients: Vec<f64> = targets.iter().map(|target| -target).collect();
    let data = BinnedMatrix::build(&dense, 4, 2, 255);
    let mut rng = SplitMix64::new(3);
    let tree = RegressionTree::grow(
        &data,
        (0..4).collect(),
        &gradients,
        &params(SplitRule::Best),
        &mut rng,
    );
    assert_eq!(tree.node_count(), 3);
    assert_eq!(tree.predict(&[0.0, 7.0]), 1.0);
    assert_eq!(tree.predict(&[1.0, 7.0]), 5.0);
}

#[test]
fn leaf_limit_and_shrinkage_are_respected() {
    let dense: Vec<f64> = (0..32).map(f64::from).collect();
    let gradients: Vec<f64> = (0..32).map(|value| -f64::from(value % 7)).collect();
    let data = BinnedMatrix::build(&dense, 32, 1, 255);
    let mut limited = params(SplitRule::Random);
    limited.max_leaves = Some(4);
    let mut rng = SplitMix64::new(9);
    let mut tree = RegressionTree::grow(&data, (0..32).collect(), &gradients, &limited, &mut rng);
    assert!(tree.node_count() <= 7);
    let before = tree.predict(&[3.0]);
    tree.scale(0.5);
    assert_eq!(tree.predict(&[3.0]), before * 0.5);
}

#[test]
fn sorted_and_histogram_scans_choose_the_same_split() {
    // 300 distinct values give 255 bins. A 40-row node is far below the bin count, so
    // `best_bin` takes the sorted path; `best_bin_histogram` is the dense scan forced directly.
    let dense: Vec<f64> = (0..300)
        .map(|value| f64::from((value * 37) % 300))
        .collect();
    let gradients: Vec<f64> = dense.iter().map(|value| -(value / 40.0).floor()).collect();
    let data = BinnedMatrix::build(&dense, 300, 1, 255);
    let tree_params = params(SplitRule::Best);
    let mut grower = Grower {
        data: &data,
        gradients: &gradients,
        params: &tree_params,
        candidates: vec![0],
        hist_sum: Vec::new(),
        hist_count: Vec::new(),
        pairs: Vec::new(),
    };
    for count in [40_u32, 300] {
        let rows: Vec<u32> = (0..count).collect();
        let total: f64 = rows.iter().map(|row| gradients[*row as usize]).sum();
        let parent = grower.score(total, rows.len());
        let sorted = grower
            .best_bin_sorted(0, &rows, total, parent)
            .expect("sorted split");
        let dense_scan = grower
            .best_bin_histogram(0, &rows, total, parent)
            .expect("histogram split");
        assert_eq!(sorted.bin, dense_scan.bin);
        assert!((sorted.gain - dense_scan.gain).abs() <= 1e-9 * dense_scan.gain.abs().max(1.0));
    }
}
