use super::*;

#[test]
fn frozen_probability_has_uniform_floor_and_explicit_missing_denominator() {
    let values = probabilities(&[known(0, 0), known(0, 1), known(0, 9)]).unwrap();
    let sum = 1.0 + 0.25 + 0.01;
    for (actual, weight) in values.iter().zip([1.0, 0.25, 0.01]) {
        assert!((*actual - (0.1 / 3.0 + 0.9 * weight / sum)).abs() < 1e-15);
    }
    assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    assert!(probabilities(&[Counts::default()]).is_err());
    let mut flight = known(0, 0);
    flight.in_flight = 1;
    assert_eq!(
        probabilities(&[flight, known(0, 1)]).unwrap(),
        vec![0.5, 0.5]
    );
}

#[test]
fn balances_family_then_dataset_and_sweeps_all_existing_cuts() {
    let mut rows = vec![
        (context("a", "many", 0), known(10, 100)),
        (context("b", "single", 4), known(1, 0)),
        (context("a", "few", 4), known(0, 0)),
    ];
    assert_eq!(select(&rows, 0, 100).unwrap(), Some(1));
    rows[1].1.attempted = 11;
    rows[1].1.selection_exposure = 11;
    assert_eq!(select(&rows, 0, 100).unwrap(), Some(2));
    let mut rows = (0..5)
        .map(|i| (context("family", "dataset", i), known(0, 0)))
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for sequence in [0, 10, 20, 30, 40] {
        let i = select(&rows, sequence, 100).unwrap().unwrap();
        selected.push(rows[i].0.cut);
        rows[i].1.attempted += 1;
        rows[i].1.selection_exposure += 1;
    }
    assert_eq!(selected, vec![0, 1, 2, 3, 4]);
    let selected = select(&rows, 19, 100).unwrap().map(|i| rows[i].0.clone());
    rows.reverse();
    assert_eq!(
        select(&rows, 19, 100).unwrap().map(|i| rows[i].0.clone()),
        selected
    );
}
