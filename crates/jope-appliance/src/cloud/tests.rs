use super::*;
#[test]
fn workspace_origin_rejects_credential_leaks_and_plain_remote_http() {
    for url in [
        "http://evil.example",
        "https://u:p@example.test",
        "https://example.test/path",
        "https://example.test?token=secret",
        "https://example.test/#x",
    ] {
        assert!(origin(url).is_err(), "{url}");
    }
    assert_eq!(
        origin("https://APP.veox.ai/").unwrap(),
        "https://app.veox.ai"
    );
}
/// Non-secret test fixtures: these values only exercise header construction and never reach a server.
const FIXTURE_ACCOUNT_TOKEN: &str = "fixture-account-token";
const FIXTURE_GATEWAY_USER: &str = "dev";
const FIXTURE_GATEWAY_PASS: &str = "fixture-gateway-pass";

#[test]
fn basic_gateway_and_account_credentials_are_distinct_sensitive_headers() {
    let c = Client::new(
        "http://127.0.0.1:9",
        Some(FIXTURE_ACCOUNT_TOKEN.into()),
        Some(Gateway {
            origin: "http://127.0.0.1:9".into(),
            username: FIXTURE_GATEWAY_USER.into(),
            password: FIXTURE_GATEWAY_PASS.into(),
        }),
    )
    .unwrap();
    let request = c
        .wire("GET", "/api/cli/account", Vec::new(), "application/json")
        .unwrap();
    assert!(request.headers()["authorization"].is_sensitive());
    assert!(request.headers()["x-veox-api-token"].is_sensitive());
    assert_ne!(
        request.headers()["authorization"],
        request.headers()["x-veox-api-token"]
    );
    assert_eq!(basic_auth("u", "p"), "Basic dTpw");
}

#[cfg(not(feature = "cloud-tls"))]
#[test]
fn https_workspace_needs_a_tls_build() {
    let error = Client::new("https://example.test", None, None)
        .err()
        .unwrap();
    assert!(error.to_string().contains("cloud-tls"), "{error}");
    assert!(Client::new("http://127.0.0.1:9", None, None).is_ok());
}

#[test]
fn private_files_refuse_symlinks_and_group_readable_credentials() {
    let root = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    }
    let path = root.path().join("credential.json");
    write_private(&path, b"synthetic credential").unwrap();
    assert_eq!(read_private(&path).unwrap(), b"synthetic credential");
    #[cfg(unix)]
    {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let link = root.path().join("link");
        symlink(&path, &link).unwrap();
        assert!(read_private(&link).is_err());
        assert!(write_private(&link, b"replacement").is_err());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_private(&path).is_err());
    }
}
#[test]
fn model_transfer_preserves_exact_identity_and_refuses_changed_parts() {
    use jope_runtime::model_object::{Manifest, ModelObject, Part, Producer, SCHEMA};
    let data = b"opaque fixture".to_vec();
    let manifest = Manifest {
        schema: SCHEMA.into(),
        minimum_reader_version: 1,
        format: "org.example/opaque/v1".into(),
        producer: Producer::default(),
        primary: "state.bin".into(),
        required_capabilities: vec!["org.example/opaque/v1".into()],
        files: vec![Part::new("state.bin", "application/octet-stream", &data)],
        metadata: json!({}),
    };
    let object = ModelObject::new(
        manifest,
        std::collections::BTreeMap::from([("state.bin".into(), data)]),
        MAX_MODEL,
    )
    .unwrap();
    let bytes = pack_object(&object).unwrap();
    let decoded = unpack_object(&bytes).unwrap();
    assert_eq!(object.id(), decoded.id());
    assert_eq!(object.manifest_bytes(), decoded.manifest_bytes());
    let mut zip = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(object.manifest_bytes()).unwrap();
    zip.start_file("state.bin", options).unwrap();
    zip.write_all(b"changed fixture").unwrap();
    assert!(unpack_object(&zip.finish().unwrap().into_inner()).is_err());
}
#[test]
fn browser_command_uses_the_platform_opener_and_never_xdg_open_on_windows() {
    let url = "https://app.veox.ai/cli/connect?user_code=AB-12";
    assert_eq!(
        browser_command("macos", url),
        ("open", vec![url.to_owned()])
    );
    assert_eq!(
        browser_command("linux", url),
        ("xdg-open", vec![url.to_owned()])
    );
    let (program, args) = browser_command("windows", url);
    assert_eq!(program, "explorer.exe");
    assert_eq!(args, [url]);
    assert!(open_browser("https://app.veox.ai/x\"y").is_err());
    assert!(open_browser("http://evil.example").is_err());
}

#[test]
fn device_login_identifies_the_jain_cli() {
    assert_eq!(CLIENT_ID, "jain-cli");
    assert_eq!(CLIENT_NAME, "jAIn CLI");
}

#[test]
fn account_tier_reads_tier_or_plan_and_is_shown_with_the_email() {
    assert_eq!(
        account_tier(&json!({"tier": "free"})).as_deref(),
        Some("free")
    );
    assert_eq!(
        account_tier(&json!({"plan": "pro"})).as_deref(),
        Some("pro")
    );
    assert_eq!(
        account_tier(&json!({"plan": {"name": "enterprise"}})).as_deref(),
        Some("enterprise")
    );
    assert_eq!(account_tier(&json!({"tier": "\u{1b}[31m"})), None);
    assert_eq!(account_tier(&json!({})), None);
    let mut c = Connection {
        origin: "https://app.veox.ai".into(),
        account_id: "00000000-0000-0000-0000-000000000000".into(),
        email: "a@example.test".into(),
        token: String::new(),
        gateway: None,
        may_publish: false,
        tier: Some("free".into()),
        terms_current: None,
    };
    assert_eq!(c.signed_in_as(), "a@example.test (Free)");
    c.tier = None;
    assert_eq!(c.signed_in_as(), "a@example.test");
    // Connections saved before tiers existed still load.
    let old: Connection = serde_json::from_value(json!({
        "origin": "https://app.veox.ai", "account_id": "x", "email": "e", "token": "t",
        "may_publish": true
    }))
    .unwrap();
    assert_eq!(old.tier, None);
    assert_eq!(
        terms_version_field(&json!("2026-11-03")).as_deref(),
        Some("2026-11-03")
    );
    assert_eq!(terms_version_field(&json!("x\u{1b}")), None);
    assert_eq!(terms_version_field(&json!(5)), None);
}

#[test]
fn gateway_credentials_cannot_be_forwarded_to_another_origin() {
    assert!(Client::new(
        "https://app.veox.ai",
        None,
        Some(Gateway {
            origin: "https://dev.example.test".into(),
            username: "dev".into(),
            password: "synthetic".into()
        })
    )
    .is_err());
}
