//! Source/binary skew guard: sanitized copies of the `scqd.toml` files
//! deployed for the July fabric-v2 daemon must keep parsing on this build.

use std::path::PathBuf;

use super::*;

const NO_PRODUCT_WORKER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/deployed-scqd/standby-no-product-worker.toml"
));
const PRODUCT_WORKER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/deployed-scqd/standby-product-worker.toml"
));

fn parse(source: &str) -> Result<DaemonConfig, ConfigError> {
    let config: DaemonConfig = toml::from_str(source).map_err(ConfigError::Parse)?;
    config.validate()?;
    Ok(config)
}

#[test]
fn deployed_standby_without_product_worker_parses() {
    let config = parse(NO_PRODUCT_WORKER).expect("deployed layout parses");
    assert!(!config.fabric.enabled);
    assert!(config.fabric.local_executor);
    assert_eq!(config.fabric.cluster_id, "node-a-standby");
    assert_eq!(config.fabric.max_gpu_idle_slots, 1);
    assert_eq!(config.auth.edge_uid, Some(999));
    assert_eq!(config.auth.broker_uid, Some(994));
    assert_eq!(config.workers.product_worker, None);
    assert!(config.workers.product_worker_templates.is_empty());
}

#[test]
fn deployed_standby_with_product_worker_parses() {
    let config = parse(PRODUCT_WORKER).expect("deployed layout parses");
    assert_eq!(
        config.workers.product_worker,
        Some(PathBuf::from("/usr/libexec/jain/product-worker"))
    );
    assert_eq!(
        config
            .workers
            .product_worker_templates
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![
            ProductWorkerTemplate::Training,
            ProductWorkerTemplate::Prediction,
            ProductWorkerTemplate::PrimaryInference,
        ]
    );
}

#[test]
fn unknown_keys_are_still_rejected() {
    for (table, key) in [
        ("[fabric]", "listen_address = \"0.0.0.0:1\""),
        ("[workers]", "product_worker_digest = \"x\""),
        ("[auth]", "node_uid = 1"),
    ] {
        let source = PRODUCT_WORKER.replacen(table, &format!("{table}\n{key}"), 1);
        assert!(
            matches!(parse(&source), Err(ConfigError::Parse(_))),
            "{table} accepted {key}"
        );
    }
    let source = format!("{PRODUCT_WORKER}\n[fabric_v3]\nenabled = false\n");
    assert!(matches!(parse(&source), Err(ConfigError::Parse(_))));
}

#[test]
fn fabric_enabled_fails_closed() {
    let source = NO_PRODUCT_WORKER.replacen("enabled = false", "enabled = true", 1);
    let error = parse(&source).expect_err("fabric must not start");
    assert!(matches!(error, ConfigError::FabricNotImplemented));
    assert!(error
        .to_string()
        .contains("fabric networking is not implemented in this build"));
}

#[test]
fn fabric_modes_that_need_networking_fail_closed() {
    for (from, to) in [
        ("edge_enabled = false", "edge_enabled = true"),
        ("local_executor = true", "local_executor = false"),
    ] {
        let source = NO_PRODUCT_WORKER.replacen(from, to, 1);
        assert!(
            matches!(parse(&source), Err(ConfigError::InvalidFabric(_))),
            "{to} accepted"
        );
    }
    let source = NO_PRODUCT_WORKER.replacen(
        "artifact_root = \"/var/lib/jain-smartcluster/artifacts\"",
        "artifact_root = \"artifacts\"",
        1,
    );
    assert!(matches!(
        parse(&source),
        Err(ConfigError::RelativePath {
            name: "fabric.artifact_root",
            ..
        })
    ));
}

#[test]
fn product_worker_must_be_absolute() {
    let source = PRODUCT_WORKER.replacen(
        "product_worker = \"/usr/libexec/jain/product-worker\"",
        "product_worker = \"usr/libexec/jain/product-worker\"",
        1,
    );
    assert!(matches!(
        parse(&source),
        Err(ConfigError::RelativePath {
            name: "workers.product_worker",
            ..
        })
    ));
}

#[test]
fn unknown_product_worker_template_is_rejected() {
    let source = PRODUCT_WORKER.replacen("\"primary_inference\"", "\"shell\"", 1);
    let error = parse(&source).expect_err("unknown template must fail");
    assert!(matches!(error, ConfigError::Parse(_)));
    assert!(error.to_string().contains("shell"), "{error}");
    // Names are exact snake_case: manifest-style kebab-case is not accepted.
    let source = PRODUCT_WORKER.replacen("\"training\"", "\"dataset-profile\"", 1);
    let error = parse(&source).expect_err("kebab-case template must fail");
    assert!(matches!(error, ConfigError::Parse(_)));
    assert!(error.to_string().contains("dataset-profile"), "{error}");
    // Casing is exact too.
    let source = PRODUCT_WORKER.replacen("\"training\"", "\"Training\"", 1);
    assert!(matches!(parse(&source), Err(ConfigError::Parse(_))));
    // July names other than primary_inference are deliberately not accepted.
    for july_only in ["ci", "agent_session", "evaluation", "production_service"] {
        let source = PRODUCT_WORKER.replacen("\"training\"", &format!("\"{july_only}\""), 1);
        assert!(
            matches!(parse(&source), Err(ConfigError::Parse(_))),
            "{july_only} accepted"
        );
    }
}

#[test]
fn product_worker_keys_are_atomic_and_digest_is_lowercase_hex() {
    let without_digest = PRODUCT_WORKER.replacen(
        "product_worker_sha256 = \"0000000000000000000000000000000000000000000000000000000000000000\"\n",
        "",
        1,
    );
    assert!(matches!(
        parse(&without_digest),
        Err(ConfigError::IncompleteProductWorkerConfiguration)
    ));
    let without_templates = PRODUCT_WORKER.replacen(
        "[\"training\", \"prediction\", \"primary_inference\"]",
        "[]",
        1,
    );
    assert!(matches!(
        parse(&without_templates),
        Err(ConfigError::IncompleteProductWorkerConfiguration)
    ));
    let upper = PRODUCT_WORKER.replacen(&"0".repeat(64), &"A".repeat(64), 1);
    assert!(matches!(
        parse(&upper),
        Err(ConfigError::InvalidProductWorkerDigest)
    ));
    let short = PRODUCT_WORKER.replacen(&"0".repeat(64), &"0".repeat(63), 1);
    assert!(matches!(
        parse(&short),
        Err(ConfigError::InvalidProductWorkerDigest)
    ));
}

#[test]
fn reserved_fabric_uids_cannot_alias_broader_roles() {
    let mut config = DaemonConfig::default();
    config.auth.edge_uid = Some(config.auth.web_uid);
    assert!(matches!(
        config.validate(),
        Err(ConfigError::OverlappingEdgeUid(_))
    ));

    let mut config = DaemonConfig::default();
    config.auth.edge_uid = Some(993);
    config.auth.broker_uid = Some(993);
    assert!(matches!(
        config.validate(),
        Err(ConfigError::BrokerIdentityOverlap(_))
    ));

    let mut config = DaemonConfig::default();
    config.auth.broker_uid = Some(992);
    config.auth.admin_uids.push(992);
    assert!(matches!(
        config.validate(),
        Err(ConfigError::BrokerIdentityOverlap(_))
    ));
}

#[test]
fn files_without_new_tables_keep_defaults() {
    let config = parse(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../smartcluster.example.toml"
    )))
    .expect("example parses");
    assert!(!config.fabric.enabled);
    assert!(config.workers.product_worker.is_none());
    assert!(config.workers.product_worker_sha256.is_none());
}

/// Files written before the reserved fabric UIDs existed must keep loading,
/// whatever values their existing identities use.
#[test]
fn omitted_reserved_uids_never_collide() {
    let example = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../smartcluster.example.toml"
    ));
    for (from, to) in [
        ("web_uid = 994", "web_uid = 993"),
        ("system_uids = [995]", "system_uids = [995, 992]"),
        ("system_uids = [995]", "system_uids = [995, 993]"),
        ("admin_uids = [0]", "admin_uids = [0, 992, 993]"),
    ] {
        let source = example.replacen(from, to, 1);
        assert_ne!(source, example, "{from} not found");
        let config = parse(&source).unwrap_or_else(|error| panic!("{to}: {error}"));
        assert_eq!(config.auth.edge_uid, None);
        assert_eq!(config.auth.broker_uid, None);
    }
}
