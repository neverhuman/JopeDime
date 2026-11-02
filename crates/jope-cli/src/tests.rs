use crate::{
    admission::validate_dataset,
    generate_fixtures,
    generator::{dataset, generated},
    schema::{digest, serialize, Catalog, Dataset, Kind, Split, Task, MAX_FILE_BYTES},
    storage::{components, Directory},
    verify_inputs,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};
use tempfile::{tempdir, TempDir};

fn corpus() -> (TempDir, PathBuf) {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("fixtures");
    generate_fixtures(&root).unwrap();
    (temporary, root)
}

fn named(task: Task, name: &str) -> Dataset {
    let scenario_index = task
        .scenarios()
        .iter()
        .position(|scenario| *scenario == name)
        .unwrap_or_else(|| panic!("unknown scenario {name}"));
    dataset(task, scenario_index)
}

fn with_groups(mut data: Dataset) -> Dataset {
    for (index, row) in data.rows.iter_mut().enumerate() {
        row.group = Some(format!("group-{:02}", index / 4));
    }
    data
}

fn rewrite_catalog(root: &std::path::Path, mutate: impl FnOnce(&mut Catalog)) {
    let path = root.join("catalog.json");
    let mut catalog: Catalog = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    mutate(&mut catalog);
    fs::write(path, serialize(&catalog).unwrap()).unwrap();
}

/// The ten fixtures committed before D5, with the per-file SHA-256 recorded in that catalog.
const PRE_D5_FIXTURES: [(&str, &str); 10] = [
    (
        "binary/linear.json",
        "7298e16774d2146de24c6d952dc806c87e12fa916a55e542663fa030c5e261a6",
    ),
    (
        "binary/imbalanced.json",
        "b3e5b86cf837dce3ddc08bc68b73dd8f4ed002cdf8c7b566fafcedb92dd15342",
    ),
    (
        "binary/outliers.json",
        "840626f6ac6696421822b41e478d9153974ab590bd56cd32cda6ab631ffd5231",
    ),
    (
        "regression/linear.json",
        "ec956c76409d16952fdb8e806ed16ec370965de72abfa1ae9a3fc54adbc445a9",
    ),
    (
        "regression/sparse_signal.json",
        "40fa77a82a53351f34786514354cdc6284299b7f269c36e3ccf95d40b9eb9e39",
    ),
    (
        "regression/extrapolation.json",
        "6bb74d1f324ef862f2c2c711a4518525533328b07f45d0b2aae2b9dc1e2053b2",
    ),
    (
        "time_series/trend.json",
        "0fae7d962f5853b8662dfb5e6ab6381dff6f1e2323cd95d3ccf841a0a0cc9919",
    ),
    (
        "time_series/lag_dependence.json",
        "45d55b17df4ec16022da48e9b3d2399862c2d646c8683374ec64304faae71c7a",
    ),
    (
        "multiclass/three_classes.json",
        "21c6d7777093fa145abda826f586d0e7c32783efcb33d5fbb236f88ded804888",
    ),
    (
        "multiclass/many_classes.json",
        "f4eef589bfacd3b1e6a59e0240429a8759a4c6b6cc80f15b7e81b25be1c5c2d1",
    ),
];

#[path = "tests/a_fifo_is_rejected_without_waiting_for_a_writer.rs"]
mod a_fifo_is_rejected_without_waiting_for_a_writer;
#[path = "tests/altered_file_and_replaced_checksum_cannot_redefine_trust.rs"]
mod altered_file_and_replaced_checksum_cannot_redefine_trust;
#[path = "tests/forty_scenarios_are_valid_and_regeneration_is_byte_identical.rs"]
mod forty_scenarios_are_valid_and_regeneration_is_byte_identical;
#[path = "tests/generation_never_overwrites_existing_or_incomplete_output.rs"]
mod generation_never_overwrites_existing_or_incomplete_output;
#[path = "tests/grouping_is_enforced_even_outside_the_named_grouped_scenario.rs"]
mod grouping_is_enforced_even_outside_the_named_grouped_scenario;
#[path = "tests/groups_cannot_cross_partitions_or_lose_identity.rs"]
mod groups_cannot_cross_partitions_or_lose_identity;
#[path = "tests/individual_dataset_reads_reject_unsafe_objects_and_oversized_payloads.rs"]
mod individual_dataset_reads_reject_unsafe_objects_and_oversized_payloads;
#[path = "tests/malformed_features_targets_ids_and_splits_fail_schema_validation.rs"]
mod malformed_features_targets_ids_and_splits_fail_schema_validation;
#[path = "tests/missing_extra_duplicate_and_untrusted_catalog_entries_fail.rs"]
mod missing_extra_duplicate_and_untrusted_catalog_entries_fail;
#[path = "tests/missing_unmanifested_and_oversized_files_fail.rs"]
mod missing_unmanifested_and_oversized_files_fail;
#[path = "tests/named_stress_scenarios_have_their_promised_structure.rs"]
mod named_stress_scenarios_have_their_promised_structure;
#[path = "tests/pre_d5_fixtures_keep_their_bytes.rs"]
mod pre_d5_fixtures_keep_their_bytes;
#[path = "tests/pre_d5_ten_member_catalog_is_rejected.rs"]
mod pre_d5_ten_member_catalog_is_rejected;
#[path = "tests/swapped_root_and_file_paths_cannot_change_bytes_already_opened.rs"]
mod swapped_root_and_file_paths_cannot_change_bytes_already_opened;
#[path = "tests/symlink_files_directories_and_roots_are_rejected.rs"]
mod symlink_files_directories_and_roots_are_rejected;
#[path = "tests/time_reversal_duplicate_times_and_future_partitions_fail.rs"]
mod time_reversal_duplicate_times_and_future_partitions_fail;
#[path = "tests/traversal_and_nonportable_paths_are_rejected.rs"]
mod traversal_and_nonportable_paths_are_rejected;
#[path = "tests/unknown_fields_and_unsupported_target_shapes_are_rejected.rs"]
mod unknown_fields_and_unsupported_target_shapes_are_rejected;
#[path = "tests/zero_width_rows_are_refused_at_cli_admission.rs"]
mod zero_width_rows_are_refused_at_cli_admission;
