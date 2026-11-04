use super::*;

#[test]
fn generation_never_overwrites_existing_or_incomplete_output() {
    let (_temporary, root) = corpus();
    assert!(generate_fixtures(&root).is_err());
    fs::remove_file(root.join("catalog.json")).unwrap();
    assert!(verify_inputs(&root).is_err());
    assert!(generate_fixtures(&root).is_err());
}
