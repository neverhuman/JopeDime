use super::*;

#[test]
fn pre_d5_fixtures_keep_their_bytes() {
    let (catalog, files) = generated().unwrap();
    for (path, sha256) in PRE_D5_FIXTURES {
        let (entry, bytes) = catalog
            .datasets
            .iter()
            .zip(&files)
            .find(|(entry, _)| entry.path == path)
            .unwrap_or_else(|| panic!("{path} is no longer generated"));
        assert_eq!(entry.sha256, sha256, "{path}");
        assert_eq!(digest(bytes), sha256, "{path}");
    }
}
