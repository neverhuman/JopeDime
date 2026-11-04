use super::*;

#[test]
fn groups_cannot_cross_partitions_or_lose_identity() {
    for task in [Task::Binary, Task::Multiclass] {
        let mut data = named(task, "grouped_observations");
        validate_dataset(&data).unwrap();
        assert!(data.rows.iter().all(|row| row.group.is_some()));
        data.rows[64].group = data.rows[0].group.clone();
        assert!(validate_dataset(&data)
            .unwrap_err()
            .to_string()
            .contains("group leaks"));
        data = named(task, "grouped_observations");
        data.rows[0].group = None;
        assert!(validate_dataset(&data).is_err());
    }
}
