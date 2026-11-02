use super::*;

#[test]
fn supervision_summary_matches_an_independent_count() {
    // Two comparisons; one tie in the first, none in the second.
    let values = [
        0.9f32, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.8, 0.75, 0.6, 0.55, 0.5, 0.3, 0.2, 0.1,
    ];
    let rows = (0..16)
        .map(|i| RowLabels {
            record: 100 + i,
            context: 3 + i / 8,
            value: values[i],
            percentile: 1.0 - (i % 8) as f32 / 7.0,
            runtime: 0.1,
        })
        .collect::<Vec<_>>();
    let lengths = (0..16).map(|i| 5 + (i * 7) % 20).collect::<Vec<_>>();
    let s = objective::supervision(&rows, &lengths).unwrap();
    let longest = *lengths.iter().max().unwrap();
    assert_eq!(s.rows, 16);
    assert_eq!(s.width, (longest + 7) / 8 * 8);
    assert_eq!(
        s.supervised_positions,
        lengths.iter().map(|l| l - 1).sum::<usize>()
    );
    let weights = rows
        .iter()
        .map(|r| (0.05f32 + r.percentile.powi(4)) as f64)
        .sum::<f64>();
    assert_eq!(s.row_weight_sum, weights);
    assert_eq!((s.ranking_pairs, s.tied_pairs), (55, 1));
    assert_eq!(s.listwise_groups, 2);
    // A one-token row supervises nothing and is refused.
    let mut short = lengths.clone();
    short[4] = 1;
    assert!(objective::supervision(&rows, &short).is_err());
}
