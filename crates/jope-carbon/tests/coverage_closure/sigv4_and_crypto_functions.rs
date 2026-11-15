//! Function and closure coverage tests for `security/sigv4.rs` and `security/user_envelope.rs`.

use std::fs;
use zeroize::Zeroizing;

use jope_carbon::security::sigv4::{
    compute_backoff_with_jitter_ms, format_amz_timestamp_utc, resolve_s3_request_target,
    resolve_secretsmanager_endpoint, sign_aws_v4_headers, sign_aws_v4_headers_with_extra,
    split_lambda_name_and_qualifier, url_encode_component, url_encode_s3_key,
};
use jope_carbon::{
    decode_p256_pubkey_b64, decrypt_cuen_payload, decrypt_cuen_with_slot_subkey,
    derive_cuen_slot_subkey, encrypt_cuen_payload, generate_random_u_dek, unwrap_u_dek_with_kek,
    verify_ecdsa_p256_spki_pem_signature, wrap_u_dek_to_pubkey, CarbonAwsCredentials,
    CarbonEncryptedPredictPayload, CarbonEngineKek, CarbonPresigner,
};
use jope_prep::cuen::CuenPayloadKind;

use super::{write_executable_script, ScopedProcessEnv};

fn extract_sig_param(url: &str) -> String {
    url.split("sig=")
        .nth(1)
        .unwrap_or("")
        .split('&')
        .next()
        .unwrap_or("")
        .to_owned()
}

#[test]
fn sigv4_credentials_from_env_profile_cache_and_redacted_debug() {
    let creds_with_tok = CarbonAwsCredentials {
        access_key_id: "TESTKEYID0001".to_owned(),
        secret_access_key: Zeroizing::new("super-secret-value-never-printed".to_owned()),
        session_token: Some(Zeroizing::new("ephemeral-sess-tok".to_owned())),
    };
    let dbg_str = format!("{:?}", creds_with_tok.clone());
    assert!(dbg_str.contains("TESTKEYID0001") && dbg_str.contains("[REDACTED]"));
    assert!(!dbg_str.contains("super-secret-value-never-printed"));
    assert!(dbg_str.contains("has_session_token: true"));

    // 1. Direct env credentials (with & without session token) + missing env error
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", Some("TESTKEYIDENV1")),
            ("AWS_SECRET_ACCESS_KEY", Some("test-secret-env-1")),
            ("AWS_SESSION_TOKEN", Some("test-session-env-1")),
        ]);
        let loaded = CarbonAwsCredentials::from_env().expect("from_env with session token");
        assert_eq!(loaded.access_key_id, "TESTKEYIDENV1");
        assert!(loaded.session_token.is_some());
    }
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", Some("TESTKEYIDENV2")),
            ("AWS_SECRET_ACCESS_KEY", Some("test-secret-env-2")),
            ("AWS_SESSION_TOKEN", Some("   ")),
        ]);
        let loaded = CarbonAwsCredentials::from_env().expect("from_env blank session token");
        assert!(loaded.session_token.is_none());
    }
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("JOPE_AWS_PROFILE", Some("  ")),
            ("AWS_PROFILE", None),
        ]);
        let err = CarbonAwsCredentials::from_env().expect_err("missing env & profile");
        assert!(err.contains("missing AWS_ACCESS_KEY_ID"));
    }

    // 2. Profile resolution error branches via CARBON_AWS_CLI_BIN
    let tmp = tempfile::tempdir().expect("tempdir");
    {
        let missing_bin = tmp.path().join("does-not-exist-aws");
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", Some("")),
            ("AWS_SECRET_ACCESS_KEY", Some("")),
            ("JOPE_AWS_PROFILE", Some("prof-missing-bin")),
            (
                "CARBON_AWS_CLI_BIN",
                Some(missing_bin.to_str().expect("utf8")),
            ),
        ]);
        let err = CarbonAwsCredentials::from_env().expect_err("missing aws bin");
        assert!(err.contains("failed to run aws configure export-credentials"));
    }

    let mock_aws = tmp.path().join("mock-aws-prof.sh");
    write_executable_script(
        &mock_aws,
        r#"#!/bin/sh
case "$AWS_PROFILE" in
  prof-fail-rc) exit 3 ;;
  prof-bad-json) printf 'not-json\n' ;;
  prof-no-ak) printf '{"SecretAccessKey":"sec"}\n' ;;
  prof-no-sk) printf '{"AccessKeyId":"TESTKEY01"}\n' ;;
  *) printf '{"AccessKeyId":"TESTKEYPROF","SecretAccessKey":"sec-prof","SessionToken":"tok-prof"}\n' ;;
esac
"#,
    );

    for (prof, needle) in [
        ("prof-fail-rc", "export-credentials failed"),
        ("prof-bad-json", "invalid export-credentials JSON"),
        ("prof-no-ak", "missing AccessKeyId"),
        ("prof-no-sk", "missing SecretAccessKey"),
    ] {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("JOPE_AWS_PROFILE", None),
            ("AWS_PROFILE", Some(prof)),
            ("CARBON_AWS_CLI_BIN", Some(mock_aws.to_str().expect("utf8"))),
        ]);
        let err = CarbonAwsCredentials::from_env().expect_err(prof);
        assert!(err.contains(needle), "unexpected error for {prof}: {err}");
    }

    // 3. $HOME/.local/bin/aws discovery + PROFILE_CACHE populate & hit + PATH fallback
    let home_with_aws = tmp.path().join("home_aws");
    let local_bin = home_with_aws.join(".local/bin");
    fs::create_dir_all(&local_bin).expect("mkdir .local/bin");
    let home_aws_bin = local_bin.join("aws");
    fs::copy(&mock_aws, &home_aws_bin).expect("copy mock aws");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mut p = fs::metadata(&home_aws_bin).expect("stat").permissions();
        p.set_mode(0o755);
        fs::set_permissions(&home_aws_bin, p).expect("chmod");
    }

    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("JOPE_AWS_PROFILE", Some("prof-home-cached-01")),
            ("CARBON_AWS_CLI_BIN", Some("")),
            ("HOME", Some(home_with_aws.to_str().expect("utf8"))),
        ]);
        let first = CarbonAwsCredentials::from_env().expect("profile via HOME/.local/bin/aws");
        assert_eq!(first.access_key_id, "TESTKEYPROF");
        assert!(first.session_token.is_some());
        // Remove binary to prove second call hits PROFILE_CACHE!
        fs::remove_file(&home_aws_bin).expect("rm home aws");
        let cached = CarbonAwsCredentials::from_env().expect("profile cache hit");
        assert_eq!(cached.access_key_id, "TESTKEYPROF");
    }

    // 4. Fallback to "aws" on PATH when HOME has no .local/bin/aws
    let path_bin = tmp.path().join("path_bin");
    fs::create_dir_all(&path_bin).expect("mkdir path_bin");
    let path_aws = path_bin.join("aws");
    write_executable_script(
        &path_aws,
        "#!/bin/sh\nprintf '{\"AccessKeyId\":\"TESTKEYPATH\",\"SecretAccessKey\":\"sec-path\"}\\n'\n",
    );
    let empty_home = tmp.path().join("empty_home");
    fs::create_dir_all(&empty_home).expect("mkdir empty_home");
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{orig_path}", path_bin.display());
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("JOPE_AWS_PROFILE", Some("prof-path-fallback-01")),
            ("CARBON_AWS_CLI_BIN", None),
            ("HOME", Some(empty_home.to_str().expect("utf8"))),
            ("PATH", Some(&new_path)),
        ]);
        let resolved = CarbonAwsCredentials::from_env().expect("profile via PATH aws");
        assert_eq!(resolved.access_key_id, "TESTKEYPATH");
        assert!(resolved.session_token.is_none());
    }
}

#[test]
fn sigv4_gateway_presigner_and_s3_presign_all_functions_and_errors() {
    assert_eq!(
        split_lambda_name_and_qualifier("jopedime-carbon-dev-engine:live"),
        ("jopedime-carbon-dev-engine", Some("live"))
    );
    assert_eq!(
        split_lambda_name_and_qualifier("jopedime-carbon-dev-engine:"),
        ("jopedime-carbon-dev-engine:", None)
    );
    assert_eq!(
        split_lambda_name_and_qualifier("arn:aws:lambda:us-east-2:123:function:f:live"),
        ("arn:aws:lambda:us-east-2:123:function:f:live", None)
    );
    assert_eq!(
        split_lambda_name_and_qualifier("plain-fn"),
        ("plain-fn", None)
    );

    for attempt in 0..10 {
        let ms = compute_backoff_with_jitter_ms(attempt, 50, 800);
        assert!((50..=800).contains(&ms));
    }
    let (d0, ts0) = format_amz_timestamp_utc(0);
    assert_eq!(
        (d0.as_str(), ts0.as_str()),
        ("19700101", "19700101T000000Z")
    );
    assert_eq!(url_encode_s3_key(""), "/");
    assert_eq!(url_encode_s3_key("/"), "/");
    assert_eq!(url_encode_s3_key("/a b/c~d"), "/a%20b/c~d");
    assert_eq!(url_encode_component("a/b+c"), "a%2Fb%2Bc");

    let creds_sess = CarbonAwsCredentials {
        access_key_id: "TESTKEYS301".to_owned(),
        secret_access_key: Zeroizing::new("test-secret-s3-01".to_owned()),
        session_token: Some(Zeroizing::new("sess-tok-s3-01".to_owned())),
    };
    let hdrs = sign_aws_v4_headers(
        &creds_sess,
        "us-east-2",
        "lambda",
        "POST",
        "lambda.us-east-2.amazonaws.com",
        "/2015-03-31/functions/fn/invocations",
        "Qualifier=live",
        b"{}",
        1_760_000_000,
    );
    assert!(hdrs.iter().any(|(k, _)| k == "x-amz-security-token"));
    let hdrs_extra = sign_aws_v4_headers_with_extra(
        &creds_sess,
        "us-east-2",
        "s3",
        "PUT",
        "bkt.s3.us-east-2.amazonaws.com",
        "/obj",
        "",
        b"abc",
        1_760_000_000,
        &[("x-amz-Invocation-Type", "Event")],
    );
    assert!(hdrs_extra.iter().any(|(k, _)| k == "authorization"));

    // Gateway presigner: unscoped, scoped, dual-secret rotation, redacted Debug, and all failure modes
    let old_sec = [0x21u8; 32];
    let new_sec = [0x42u8; 32];
    let old_presigner = CarbonPresigner::from_secret(old_sec);
    let rot_presigner = CarbonPresigner::from_secret(new_sec).with_previous_secret(Some(old_sec));
    let presign_dbg = format!("{:?}", rot_presigner.clone());
    assert!(
        presign_dbg.contains("[REDACTED]") && presign_dbg.contains("has_previous_secret: true")
    );
    let pseudo = rot_presigner.daily_ip_pseudonym_hex(1_760_000_000, "203.0.113.9");
    assert_eq!(pseudo.len(), 16);

    let sha = "ab".repeat(32);
    let unscoped_put =
        old_presigner.sign_gateway_upload_url("https://gw.example", &sha, 128, 2_000);
    let unscoped_put_sig = extract_sig_param(&unscoped_put);
    rot_presigner
        .verify_gateway_upload(&sha, 128, 2_000, &unscoped_put_sig, 1_500)
        .expect("verify_gateway_upload via previous_secret");
    assert!(rot_presigner
        .verify_gateway_upload(&sha, 128, 2_000, &unscoped_put_sig, 2_001)
        .is_err());
    assert!(rot_presigner
        .verify_gateway_upload(&sha, 128, 2_000, "not-hex!", 1_500)
        .is_err());
    assert!(rot_presigner
        .verify_gateway_upload(&sha, 999, 2_000, &unscoped_put_sig, 1_500)
        .is_err());

    let scoped_put = rot_presigner.sign_gateway_scoped_upload_url(
        "https://gw.example/",
        Some("blobs/t1/u1"),
        &sha,
        64,
        2_000,
    );
    rot_presigner
        .verify_gateway_scoped_upload(
            Some("blobs/t1/u1"),
            &sha,
            64,
            2_000,
            &extract_sig_param(&scoped_put),
            1_900,
        )
        .expect("scoped upload verify");

    let unscoped_get =
        old_presigner.sign_gateway_scoped_download_url("https://gw.example/", None, &sha, 2_000);
    let unscoped_get_sig = extract_sig_param(&unscoped_get);
    rot_presigner
        .verify_gateway_scoped_download(None, &sha, 2_000, &unscoped_get_sig, 1_500)
        .expect("unscoped download verify via previous_secret");
    assert!(rot_presigner
        .verify_gateway_scoped_download(None, &sha, 2_000, &unscoped_get_sig, 2_500)
        .is_err());
    assert!(rot_presigner
        .verify_gateway_scoped_download(None, &sha, 2_000, "zz", 1_500)
        .is_err());
    assert!(rot_presigner
        .verify_gateway_scoped_download(Some("wrong"), &sha, 2_000, &unscoped_get_sig, 1_500)
        .is_err());

    let scoped_get = rot_presigner.sign_gateway_scoped_download_url(
        "https://gw.example",
        Some("runs/t1/r1"),
        &sha,
        2_000,
    );
    rot_presigner
        .verify_gateway_scoped_download(
            Some("runs/t1/r1"),
            &sha,
            2_000,
            &extract_sig_param(&scoped_get),
            1_500,
        )
        .expect("scoped download verify");

    // S3 SigV4 presigners + endpoint resolvers (default AWS, IPv4/IPv6 loopback, TLS, & error guards)
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("CARBON_S3_ENDPOINT_URL", None),
            ("CARBON_SECRETSMANAGER_ENDPOINT_URL", None),
        ]);
        let put_pinned = CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "blobs/k1",
            "20261115T120000Z",
            300,
            &sha,
            512,
        )
        .expect("presign_s3_put_v4_pinned_with_creds");
        assert!(put_pinned.contains("X-Amz-Security-Token="));
        assert!(CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "short",
            300,
            &sha,
            512
        )
        .is_err());
        assert!(CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "20261115T120000Z",
            0,
            &sha,
            512
        )
        .is_err());
        assert!(CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "20261115T120000Z",
            300,
            "badsha",
            512
        )
        .is_err());
        assert!(CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "20261115T120000Z",
            300,
            &sha,
            0
        )
        .is_err());

        let put_unpinned = CarbonPresigner::presign_s3_put_v4_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "blobs/k1",
            "20261115T120000Z",
            300,
            "UNSIGNED-PAYLOAD",
        )
        .expect("presign_s3_put_v4_with_creds");
        assert!(put_unpinned.contains("X-Amz-Security-Token="));
        assert!(CarbonPresigner::presign_s3_put_v4_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "20261115T120000Z",
            700_000,
            "UNSIGNED-PAYLOAD"
        )
        .is_err());

        let get_url = CarbonPresigner::presign_s3_get_v4_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "blobs/k1",
            "20261115T120000Z",
            300,
        )
        .expect("presign_s3_get_v4_with_creds");
        assert!(get_url.contains("X-Amz-Security-Token="));
        assert!(CarbonPresigner::presign_s3_get_v4_with_creds(
            &creds_sess,
            "us-east-2",
            "my-bkt",
            "k",
            "bad",
            300
        )
        .is_err());

        let static_put = CarbonPresigner::presign_s3_put_v4(
            "TESTKEYSTATIC",
            "test-secret-static",
            "us-east-2",
            "my-bkt",
            "blobs/k2",
            "20261115T120000Z",
            300,
            &sha,
        )
        .expect("presign_s3_put_v4");
        assert!(static_put.contains("my-bkt.s3.us-east-2.amazonaws.com"));

        let (sm_h, sm_u) = resolve_secretsmanager_endpoint("us-west-2").expect("default sm");
        assert_eq!(sm_h, "secretsmanager.us-west-2.amazonaws.com");
        assert_eq!(sm_u, "https://secretsmanager.us-west-2.amazonaws.com/");
    }

    // Endpoint override edge cases (IPv6 loopback, empty key, invalid schemes/hosts)
    for valid_s3 in [
        "http://127.0.0.1",
        "http://[::1]",
        "http://[::1]:9000",
        "https://s3.custom.example:8443",
    ] {
        let _env = ScopedProcessEnv::set(&[("CARBON_S3_ENDPOINT_URL", valid_s3)]);
        assert!(resolve_s3_request_target("bkt", "us-east-2", "").is_ok());
        assert!(resolve_s3_request_target("bkt", "us-east-2", "k/1").is_ok());
    }
    for bad_s3 in [
        "http://example.com:9000",
        "http://127.0.0.1:0",
        "http://[::1]:0",
        "ftp://127.0.0.1:9000",
        "https://example.com/subpath",
    ] {
        let _env = ScopedProcessEnv::set(&[("CARBON_S3_ENDPOINT_URL", bad_s3)]);
        assert!(resolve_s3_request_target("bkt", "us-east-2", "k").is_err());
    }

    {
        let _env =
            ScopedProcessEnv::set(&[("CARBON_SECRETSMANAGER_ENDPOINT_URL", "https://[::1]:8443")]);
        let (h, u) = resolve_secretsmanager_endpoint("us-east-2").expect("ipv6 sm");
        assert_eq!(
            (h.as_str(), u.as_str()),
            ("[::1]:8443", "https://[::1]:8443/")
        );
    }
    for bad_sm in [
        "ftp://127.0.0.1:9000",
        "https://secretsmanager.us-east-2.amazonaws.com",
    ] {
        let _env = ScopedProcessEnv::set(&[("CARBON_SECRETSMANAGER_ENDPOINT_URL", bad_sm)]);
        assert!(resolve_secretsmanager_endpoint("us-east-2").is_err());
    }
}

#[test]
fn user_envelope_kek_rotation_spki_and_tamper_function_closure() {
    assert!(CarbonEngineKek::from_scalar_bytes(&[0u8; 32]).is_err());
    assert!(CarbonEngineKek::from_hex("not-hex").is_err());
    assert!(CarbonEngineKek::from_hex("1122").is_err());

    let prev_hex = "11".repeat(32);
    let act_hex = "22".repeat(32);
    let prev_kek = CarbonEngineKek::from_hex(&prev_hex).expect("prev kek");
    let single_kek =
        CarbonEngineKek::from_hex_with_previous(&act_hex, Some("  ")).expect("blank prev");
    let rot_kek =
        CarbonEngineKek::from_hex_with_previous(&act_hex, Some(&prev_hex)).expect("rot kek");
    let dbg_kek = format!("{rot_kek:?}");
    assert!(dbg_kek.contains("[REDACTED]") && dbg_kek.contains("has_previous_kek: true"));
    assert!(!format!("{single_kek:?}").contains(&act_hex));

    // Wrap U_DEK to retiring `prev_kek` and unwrap via `rot_kek.previous_kek` fallback!
    let u_dek = generate_random_u_dek().expect("u_dek");
    let cudk_prev =
        wrap_u_dek_to_pubkey(&u_dek, prev_kek.pubkey_bytes(), "tenant_a", "run_01").expect("wrap");
    let unwrapped =
        unwrap_u_dek_with_kek(&rot_kek, &cudk_prev, "tenant_a", "run_01").expect("unwrap via prev");
    assert_eq!(&*unwrapped, &*u_dek);
    assert!(unwrap_u_dek_with_kek(&rot_kek, &cudk_prev, "wrong_tenant", "run_01").is_err());
    assert!(unwrap_u_dek_with_kek(&rot_kek, b"short", "tenant_a", "run_01").is_err());

    // decode_p256_pubkey_b64 negative branches
    assert!(decode_p256_pubkey_b64("%%%bad-base64%%%").is_err());
    assert!(decode_p256_pubkey_b64("AQID").is_err());
    let mut bad_prefix = *prev_kek.pubkey_bytes();
    bad_prefix[0] = 0x02;
    let bad_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bad_prefix);
    assert!(decode_p256_pubkey_b64(&bad_b64).is_err());

    // CUEN encrypt/decrypt + tamper & kind mismatch + PredictPayload struct derives
    let pred = CarbonEncryptedPredictPayload {
        feature_names: vec!["f1".to_owned()],
        rows: vec![vec![Some(1.5), None]],
    };
    assert_eq!(pred.clone(), pred);
    let _ = format!("{pred:?}");
    let pt = serde_json::to_vec(&pred).expect("json");
    let wire = encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::PredictFeatures,
        "tenant_a",
        "run_01",
        &pt,
    )
    .expect("encrypt cuen");
    let dec = decrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::PredictFeatures,
        "tenant_a",
        "run_01",
        &wire,
    )
    .expect("decrypt cuen");
    assert_eq!(&*dec, &pt);
    assert!(decrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::FitJmat,
        "tenant_a",
        "run_01",
        &wire
    )
    .is_err());
    let sub = derive_cuen_slot_subkey(
        &u_dek,
        CuenPayloadKind::PredictFeatures,
        "tenant_a",
        "run_01",
    )
    .expect("subkey");
    let mut tampered = wire.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x55;
    assert!(decrypt_cuen_with_slot_subkey(
        &sub,
        CuenPayloadKind::PredictFeatures,
        "tenant_a",
        "run_01",
        &tampered
    )
    .is_err());

    // SPKI ECDSA-P256 verifier negative & positive cases
    assert!(verify_ecdsa_p256_spki_pem_signature("no header", b"msg", b"sig").is_err());
    assert!(verify_ecdsa_p256_spki_pem_signature(
        "-----BEGIN PUBLIC KEY-----\n%%%bad%%%\n-----END PUBLIC KEY-----",
        b"msg",
        b"sig"
    )
    .is_err());
    assert!(verify_ecdsa_p256_spki_pem_signature(
        "-----BEGIN PUBLIC KEY-----\nAQIDBA==\n-----END PUBLIC KEY-----",
        b"msg",
        b"sig"
    )
    .is_err());
}
