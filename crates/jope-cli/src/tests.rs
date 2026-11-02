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

#[path = "tests/mod.rs"]
mod cases;
