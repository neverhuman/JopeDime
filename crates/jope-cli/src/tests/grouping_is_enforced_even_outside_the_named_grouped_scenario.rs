use super::*;

#[test]
fn grouping_is_enforced_even_outside_the_named_grouped_scenario() {
    for task in [Task::Binary, Task::Regression, Task::Multiclass] {
        // Groups added to an ungrouped scenario are held to the same partition rules.
        let data = with_groups(named(task, task.scenarios()[0]));
        assert_ne!(data.scenario, "grouped_observations");
        validate_dataset(&data).unwrap();
        let mut leaked = data.clone();
        leaked.rows[64].group = leaked.rows[0].group.clone();
        assert!(validate_dataset(&leaked)
            .unwrap_err()
            .to_string()
            .contains("group leaks"));
        let mut missing = data;
        missing.rows[0].group = None;
        assert!(validate_dataset(&missing)
            .unwrap_err()
            .to_string()
            .contains("missing group identity"));
    }
}
