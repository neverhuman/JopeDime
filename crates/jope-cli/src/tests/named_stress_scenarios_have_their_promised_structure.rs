use super::*;

#[test]
fn named_stress_scenarios_have_their_promised_structure() {
    let imbalance = named(Task::Binary, "imbalanced");
    assert_eq!(
        imbalance.rows.iter().filter(|row| row.target == 1).count(),
        8
    );
    let constant = named(Task::Binary, "constant_features");
    assert!(constant.rows.iter().all(|row| row.features[1] == 42));
    let duplicated = named(Task::Binary, "duplicated_rows");
    assert!(duplicated
        .rows
        .chunks_exact(2)
        .all(|pair| pair[0].features == pair[1].features
            && pair[0].target == pair[1].target
            && pair[0].split == pair[1].split));
    let outliers = named(Task::Binary, "outliers");
    assert!(outliers
        .rows
        .iter()
        .any(|row| row.features[0].as_i64().unwrap().abs() > 10_000));
    let constant_target = named(Task::Regression, "constant_target");
    assert!(constant_target.rows.iter().all(|row| row.target == 7));
    let extrapolation = named(Task::Regression, "extrapolation");
    assert!(extrapolation
        .rows
        .iter()
        .filter(|row| row.split == Split::Train)
        .all(|row| row.features[0].as_i64().unwrap() <= 0));
    assert!(extrapolation
        .rows
        .iter()
        .filter(|row| row.split == Split::Test)
        .all(|row| row.features[0].as_i64().unwrap() >= 1000));
    for (task, scenario) in [
        (Task::Regression, "sparse_signal"),
        (Task::Multiclass, "high_dimensional_noise"),
    ] {
        assert_eq!(named(task, scenario).columns.len(), 12, "{scenario}");
    }
    assert_eq!(named(Task::Multiclass, "many_classes").classes.len(), 7);
    let rare = named(Task::Multiclass, "rare_class");
    assert_eq!(rare.rows.iter().filter(|row| row.target == 4).count(), 4);
    let lagged = named(Task::TimeSeries, "lag_dependence");
    for (index, row) in lagged.rows.iter().enumerate().skip(2) {
        assert_eq!(row.features[0], lagged.rows[index - 1].target);
        assert_eq!(row.features[1], lagged.rows[index - 2].target);
    }
}
