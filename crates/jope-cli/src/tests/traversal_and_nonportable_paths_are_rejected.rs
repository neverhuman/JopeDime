use super::*;

#[test]
fn traversal_and_nonportable_paths_are_rejected() {
    for path in [
        "",
        "../outside.json",
        "/tmp/outside.json",
        "binary/../catalog.json",
        "binary//linear.json",
        "./catalog.json",
        "binary\\linear.json",
        "C:/outside.json",
    ] {
        assert!(components(path).is_err(), "path {path}");
    }
    let (_temporary, root) = corpus();
    rewrite_catalog(&root, |catalog| {
        catalog.datasets[0].path = "../outside.json".into()
    });
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("relative fixture path"));
}
