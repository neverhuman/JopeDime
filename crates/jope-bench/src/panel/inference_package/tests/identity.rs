use super::*;

/// W9-T05: a load takes its identity from a record only when the record names the
/// package the authenticated qualification evidence seals. `load` alone accepts any
/// consistent installed package under whatever identity it is handed (the recorded trust
/// limit), so a record naming another fitted state's package would serve that state.
#[test]
fn a_recorded_identity_must_be_the_authenticated_seal() {
    let root = scratch();
    let (a, _) = fixture(Task::Regression, "ridge");
    let (b, _) = fixture(Task::Binary, "logit_ridge");
    let q = qualify(root.path(), &a, Backend::Python);
    let other = qualify(root.path(), &b, Backend::Python);
    let (manifest, files) = seal_default(&q, true);
    let (other_manifest, other_files) = seal_default(&other, true);
    let installed = root.path().join("installed");
    let recorded = install(&installed, &manifest, &files).unwrap();
    assert_eq!(recorded, identity(&manifest).unwrap());
    let loaded = load_authenticated(&installed, &recorded, &manifest).unwrap();
    assert_eq!(loaded.manifest.fitted_sha256, q.plan.fitted_sha256);
    // Another state's package, installed intact and loadable under its own identity.
    let substituted = root.path().join("substituted");
    let substituted_identity = install(&substituted, &other_manifest, &other_files).unwrap();
    load(&substituted, &substituted_identity).unwrap();
    let error = format!(
        "{:#}",
        load_authenticated(&substituted, &substituted_identity, &manifest).unwrap_err()
    );
    assert!(
        error.contains("authenticated qualification derives"),
        "{error}"
    );
    // The right record over another installation still fails the installed-byte checks.
    assert!(load_authenticated(&substituted, &recorded, &manifest).is_err());
    // A package sealed optional is another identity than the required one.
    let (optional, _) = seal_default(&q, false);
    assert!(load_authenticated(&installed, &recorded, &optional).is_err());
    load_authenticated(&installed, &recorded, &manifest).unwrap();
}
