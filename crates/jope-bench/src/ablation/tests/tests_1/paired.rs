use super::*;

#[test]
pub(crate) fn ablation_paired_summary_arithmetic() {
    let runs: Vec<RunRecord> = [
        // (dataset, seed, A RMSE, B RMSE, A program, B program)
        ("d0", 1, 1.0, 0.75, "p", "q"), // win, -0.25
        ("d0", 2, 1.0, 1.0, "p", "p"),  // tie, same program
        ("d0", 3, 1.0, 1.5, "p", "q"),  // loss, +0.5
        ("d1", 1, 2.0, 1.5, "p", "q"),  // win, -0.5
        ("d1", 2, 2.0, 1.75, "p", "q"), // win, -0.25
        ("d1", 3, 2.0, 2.0, "p", "p"),  // tie, same program
    ]
    .iter()
    .flat_map(|&(dataset, seed, a, b, program_a, program_b)| {
        [
            paired_run("A", dataset, seed, a, program_a),
            paired_run("B", dataset, seed, b, program_b),
        ]
    })
    .collect();
    let datasets = vec!["d0".to_owned(), "d1".to_owned()];
    let summary = paired(&datasets, &runs).unwrap();
    assert_eq!(
        (summary.definitions, summary.rule),
        (PAIRED_DEFINITIONS, PAIRED_RULE)
    );
    let all = &summary.overall;
    assert_eq!((all.pairs, all.wins, all.ties, all.losses), (6, 3, 2, 1));
    assert_eq!((all.median_a, all.median_b), (1.5, 1.5));
    assert_eq!(all.median_difference, -0.125);
    assert_eq!(all.mean_difference, -0.5 / 6.0);
    assert_eq!(
        all.largest_gain,
        PairDifference {
            dataset: "d1".to_owned(),
            seed: 1,
            difference: -0.5
        }
    );
    assert_eq!(all.mean_difference_without_largest_gain, Some(0.0));
    // 2 * (C(4,0) + C(4,1)) / 2^4.
    assert_eq!(all.sign_test_p, 0.625);
    assert_eq!(all.same_winner, 2);

    let [d0, d1] = summary.per_dataset.as_slice() else {
        panic!("two datasets: {summary:?}");
    };
    assert_eq!((d0.dataset.as_str(), d1.dataset.as_str()), ("d0", "d1"));
    let c = &d0.counts;
    assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 1, 1, 1));
    assert_eq!(
        (c.median_a, c.median_b, c.median_difference),
        (1.0, 1.0, 0.0)
    );
    assert_eq!(c.mean_difference, 0.25 / 3.0);
    assert_eq!((c.largest_gain.seed, c.largest_gain.difference), (1, -0.25));
    assert_eq!(c.mean_difference_without_largest_gain, Some(0.25));
    assert_eq!((c.sign_test_p, c.same_winner), (1.0, 1));
    let c = &d1.counts;
    assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 2, 1, 0));
    assert_eq!(
        (c.median_a, c.median_b, c.median_difference),
        (2.0, 1.75, -0.25)
    );
    assert_eq!(c.mean_difference, -0.25);
    assert_eq!(c.mean_difference_without_largest_gain, Some(-0.125));
    // 2 * C(2,0) / 2^2.
    assert_eq!((c.sign_test_p, c.same_winner), (0.5, 1));

    // Nothing but ties: no untied pair, so p is 1, and every difference is zero.
    let ties: Vec<RunRecord> = [1, 2, 3]
        .iter()
        .flat_map(|&seed| {
            [
                paired_run("A", "d0", seed, 0.5, "p"),
                paired_run("B", "d0", seed, 0.5, "p"),
            ]
        })
        .collect();
    let tied = paired(&datasets[..1], &ties).unwrap().overall;
    assert_eq!(
        (tied.wins, tied.ties, tied.losses, tied.same_winner),
        (0, 3, 0, 3)
    );
    assert_eq!(tied.sign_test_p, 1.0);
    assert_eq!((tied.median_difference, tied.mean_difference), (0.0, 0.0));
    assert_eq!(tied.largest_gain.seed, 1);
    assert_eq!(tied.mean_difference_without_largest_gain, Some(0.0));

    // A tie is bit-equality: the next RMSE above A's is a loss.
    let near = [
        paired_run("A", "d0", 1, 1.0, "p"),
        paired_run("B", "d0", 1, f64::from_bits(1.0_f64.to_bits() + 1), "q"),
    ];
    let near = paired(&datasets[..1], &near).unwrap().overall;
    assert_eq!((near.wins, near.ties, near.losses), (0, 0, 1));
    assert_eq!(near.mean_difference_without_largest_gain, None);

    // Every A run needs exactly one B run, and every B run an A run.
    assert!(paired(&datasets, &runs[..runs.len() - 1]).is_err());
    for extra in [
        paired_run("B", "d0", 9, 1.0, "p"),
        paired_run("B", "d0", 1, 1.0, "p"),
    ] {
        let mut runs = runs.clone();
        runs.push(extra);
        assert!(paired(&datasets, &runs).is_err());
    }
}
