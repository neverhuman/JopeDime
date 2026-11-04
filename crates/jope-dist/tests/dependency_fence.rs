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
    // jope-bench takes feat-math with default features, so the C++ runtime is on there.
    assert!(
        violations
            .iter()
            .any(|v| v.contains("feat-math/fitted-cpp is enabled")),
        "{violations:#?}"
    );
    // The packer itself carries the zstd C library; the appliance must not.
    let packer = closure(&repo.join("Cargo.toml"), "jope-dist", &policy.target, &[]).unwrap();
    assert!(packer.contains("zstd-sys"));
}

#[test]
fn the_dope_gpu_appliance_closure_obeys_its_separate_policy() {
    let repo = jope_dist::repo_root();
    let policy = Policy::load(&repo.join("deploy/appliance/dependencies-dope-gpu.toml")).unwrap();
    let violations = audit(&repo.join("Cargo.toml"), &policy).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn full_cpu_matrix_covers_production_defaults_without_cpp_or_tls() {
    use std::collections::BTreeSet;
    let repo = jope_dist::repo_root();
    let manifest: toml::Value = toml::from_str(
        &std::fs::read_to_string(repo.join("crates/jope-appliance/Cargo.toml")).unwrap(),
    )
    .unwrap();
    let defaults: BTreeSet<String> = manifest["features"]["default"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        defaults,
        BTreeSet::from(["engine".into(), "mcp".into(), "client".into()])
    );
    let policy = Policy::load(&repo.join("deploy/appliance/dependencies-deny.toml")).unwrap();
    let selections: Vec<BTreeSet<String>> = policy
        .feature_sets
        .iter()
        .map(|s| s.iter().cloned().collect())
        .collect();
    assert!(selections.contains(&defaults));
    let mut full = defaults;
    full.insert("embedded".into());
    assert!(selections.contains(&full));
    let features: Vec<_> = full.into_iter().collect();
    let resolved = jope_dist::audit::closure_features(
        &repo.join("Cargo.toml"),
        &policy.package,
        &policy.target,
        &features,
    )
    .unwrap();
    for name in [
        "jope-service",
        "jope-client",
        "rmcp",
        "candle-core",
        "gemm",
        "hyper",
        "ureq",
    ] {
        assert!(resolved.contains_key(name), "full selection omitted {name}");
    }
    assert!(!resolved["feat-math"].contains("fitted-cpp"));
    assert!(resolved["ureq"].is_empty());
    assert!(!resolved.contains_key("rustls"));
    assert!(!resolved.contains_key("cudarc"));
}

/// A22: the released connected client selects `cloud-tls`, the policy audits exactly that
/// selection, and HTTPS comes from rustls + AWS-LC with no system TLS library or second provider.
#[test]
fn released_client_selection_is_audited_and_uses_one_tls_stack() {
    use std::collections::BTreeSet;
    let repo = jope_dist::repo_root();
    let release = std::fs::read_to_string(repo.join("deploy/release/release.sh")).unwrap();
    let command = release
        .lines()
        .find_map(|l| l.strip_prefix("readonly APPLIANCE_BUILD_COMMAND="))
        .expect("release.sh defines APPLIANCE_BUILD_COMMAND");
    assert!(
        command.contains("-p jope-appliance") && command.contains("--features cloud-tls"),
        "{command}"
    );
    assert!(!command.contains("--no-default-features"), "{command}");
    let policy = Policy::load(&repo.join("deploy/appliance/dependencies-deny.toml")).unwrap();
    let released: BTreeSet<String> = ["engine", "mcp", "client", "cloud-tls"]
        .map(str::to_owned)
        .into();
    assert!(policy
        .feature_sets
        .iter()
        .any(|s| s.iter().cloned().collect::<BTreeSet<_>>() == released));
    let features: Vec<_> = released.into_iter().collect();
    let resolved = jope_dist::audit::closure_features(
        &repo.join("Cargo.toml"),
        &policy.package,
        &policy.target,
        &features,
    )
    .unwrap();
    assert!(resolved["rustls"].contains("aws_lc_rs"));
    assert!(resolved["ureq"].contains("rustls-webpki-roots"));
    for absent in [
        "ring",
        "openssl",
        "openssl-sys",
        "native-tls",
        "rustls-platform-verifier",
    ] {
        assert!(
            !resolved.contains_key(absent),
            "{absent} reached the released client"
        );
    }
}
