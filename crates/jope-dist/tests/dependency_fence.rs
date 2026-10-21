//! The shipped package's closure obeys `deploy/appliance/dependencies-deny.toml` for every
//! audited feature set, and the policy is not vacuous: a research crate fails it.
use jope_dist::audit::{audit, closure, Policy};

#[test]
fn the_appliance_closure_obeys_the_policy() {
    let repo = jope_dist::repo_root();
    let policy = Policy::load(&repo.join("deploy/appliance/dependencies-deny.toml")).unwrap();
    let violations = audit(&repo.join("Cargo.toml"), &policy).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn the_policy_catches_a_research_crate() {
    let repo = jope_dist::repo_root();
    let mut policy = Policy::load(&repo.join("deploy/appliance/dependencies-deny.toml")).unwrap();
    policy.package = "jope-bench".into();
    policy.feature_sets = vec![Vec::new()];
    let violations = audit(&repo.join("Cargo.toml"), &policy).unwrap();
    assert!(
        violations.iter().any(|v| v.contains("jope-pipeline")),
        "{violations:#?}"
    );
    // The packer itself carries the zstd C library; the appliance must not.
    let packer = closure(&repo.join("Cargo.toml"), "jope-dist", &policy.target, &[]).unwrap();
    assert!(packer.contains("zstd-sys"));
}
