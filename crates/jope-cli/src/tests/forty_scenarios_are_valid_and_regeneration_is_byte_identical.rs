use super::*;

#[test]
fn forty_scenarios_are_valid_and_regeneration_is_byte_identical() {
    let (first_catalog, first_files) = generated().unwrap();
    let (second_catalog, second_files) = generated().unwrap();
    assert_eq!(first_catalog, second_catalog);
    assert_eq!(first_files, second_files);
    assert_eq!(first_files.len(), 40);
    for bytes in first_files {
        let data: Dataset = serde_json::from_slice(&bytes).unwrap();
        validate_dataset(&data).unwrap_or_else(|error| panic!("{}: {error}", data.id));
    }
    let (_temporary, root) = corpus();
    let report = verify_inputs(&root).unwrap();
    assert_eq!(report.dataset_count, 40);
    assert_eq!(report.task_counts.len(), 4);
    assert!(report.task_counts.values().all(|count| *count == 10));
    assert_eq!(report.bytes, 828_029);
    assert_eq!(
        report.catalog_sha256,
        "651d28c8456bc160ed33960110142cdb8e9941d2a727348416db999f99d22e93"
    );
    assert_eq!(report.generator, "jope.synthetic.splitmix64.integer.v1");
}
