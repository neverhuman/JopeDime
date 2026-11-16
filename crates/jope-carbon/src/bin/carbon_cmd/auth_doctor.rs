//! Authentication (`register`, `login`, `whoami`, `logout`), `self-verify`, `doctor`, `upgrade`, and `uninstall`.

use super::*;

pub(crate) fn is_auth_doctor_cmd(cmd: &Command) -> bool {
    matches!(
        cmd,
        Command::SelfVerify { .. }
            | Command::Register { .. }
            | Command::Login { .. }
            | Command::Whoami { .. }
            | Command::Logout { .. }
            | Command::Doctor { .. }
            | Command::Upgrade { .. }
            | Command::Uninstall { .. }
    )
}

pub(crate) fn dispatch_auth_doctor(cmd: Command) -> ExitCode {
    match cmd {
        Command::SelfVerify {
            manifest,
            signature,
            clouds_manifest,
            allow_test_pubkey,
            require_manifest,
            binary_only,
        } => execute_self_verify_command(
            manifest,
            signature,
            clouds_manifest,
            allow_test_pubkey,
            require_manifest,
            binary_only,
        ),
        Command::Register {
            cloud,
            free: _free,
            label,
            clouds_manifest,
            endpoint,
        } => {
            let (candidates, pinned_keks) = if let Some(ep) = endpoint {
                (vec![ep], BTreeMap::new())
            } else {
                match resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref()) {
                    Ok(entry) => (entry.candidate_endpoints(), entry.pinned_keks_by_endpoint()),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            };
            let mut client =
                match CarbonThinClient::connect_fastest_pinned(candidates, &pinned_keks) {
                    Ok((c, _)) => c,
                    Err(e) => {
                        eprintln!("error: probe '{cloud}' failed: {e}");
                        return ExitCode::from(1);
                    }
                };
            let reg = match client.register_free(label.as_deref()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: carbon register failed: {e}");
                    return ExitCode::from(1);
                }
            };
            let active_ep = client.active_endpoint().to_owned();
            let now_secs = now_epoch_ms() / 1_000;
            match save_stored_cloud_token(&cloud, &active_ep, &reg.capability_token) {
                Ok(saved) => {
                    let _ = write_stdout_line(
                        &serde_json::to_string_pretty(&serde_json::json!({
                            "registered": true,
                            "cloud": cloud.trim().to_ascii_lowercase(),
                            "endpoint": active_ep,
                            "tenant_id": saved.tenant_id,
                            "tier": saved.tier,
                            "expires_epoch_secs": reg.expires_epoch_secs.0,
                            "ttl_remaining_secs": reg.expires_epoch_secs.0.saturating_sub(now_secs),
                            "credentials_saved": true,
                        }))
                        .unwrap_or_default(),
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Login {
            cloud,
            clouds_manifest,
            endpoint,
            token,
            token_file,
            token_stdin,
        } => {
            let raw_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(Some(t)) => t,
                Ok(None) => match env::var("CARBON_AUTH_TOKEN") {
                    Ok(v) if !v.trim().is_empty() => v.trim().to_owned(),
                    _ if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() => {
                        eprint!("Paste ctk1.* capability token: ");
                        let _ = std::io::stderr().flush();
                        let mut line = String::new();
                        if let Err(e) = std::io::stdin().read_line(&mut line) {
                            eprintln!("error: read token from terminal: {e}");
                            return ExitCode::from(2);
                        }
                        let trimmed = line.trim().to_owned();
                        if trimmed.is_empty() {
                            eprintln!("error: capability token must not be empty");
                            return ExitCode::from(2);
                        }
                        trimmed
                    }
                    _ => {
                        eprintln!(
                            "error: pass --token-stdin, --token-file <PATH>, or --token <ctk1...> to `carbon login`"
                        );
                        return ExitCode::from(2);
                    }
                },
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let claims = match CarbonCapabilityClaims::decode_unverified(&raw_tok) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let now_secs = now_epoch_ms() / 1_000;
            if now_secs > claims.exp_epoch_secs {
                eprintln!(
                    "error: capability token for tenant '{}' has already expired (exp={})",
                    claims.tenant_id, claims.exp_epoch_secs
                );
                return ExitCode::from(2);
            }
            let (candidates, pinned_keks) = if let Some(ep) = endpoint {
                (vec![ep], BTreeMap::new())
            } else {
                match resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref()) {
                    Ok(entry) => (entry.candidate_endpoints(), entry.pinned_keks_by_endpoint()),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            };
            let active_ep = match CarbonThinClient::connect_fastest_pinned(candidates, &pinned_keks)
            {
                Ok((client, _)) => client.active_endpoint().to_owned(),
                Err(e) => {
                    eprintln!("error: probe '{cloud}' failed: {e}");
                    return ExitCode::from(1);
                }
            };
            match save_stored_cloud_token(&cloud, &active_ep, &raw_tok) {
                Ok(saved) => {
                    let _ = write_stdout_line(
                        &serde_json::to_string_pretty(&serde_json::json!({
                            "logged_in": true,
                            "cloud": cloud.trim().to_ascii_lowercase(),
                            "endpoint": active_ep,
                            "tenant_id": saved.tenant_id,
                            "tier": saved.tier,
                            "exp_epoch_secs": saved.exp_epoch_secs,
                            "ttl_remaining_secs": saved.exp_epoch_secs.saturating_sub(now_secs),
                        }))
                        .unwrap_or_default(),
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Whoami {
            cloud,
            clouds_manifest,
            token,
            token_file,
            token_stdin,
            json: _,
        } => {
            let now_secs = now_epoch_ms() / 1_000;
            let ep = resolve_public_cloud_endpoint(&cloud, clouds_manifest.as_deref())
                .ok()
                .map(|e| e.public_endpoint);
            let inline = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let env_tok = env::var("CARBON_AUTH_TOKEN")
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
            let (tok_opt, source_label) = if let Some(t) = inline {
                (Some(t), "cli_flag")
            } else if let Some(t) = env_tok {
                (Some(t), "env_CARBON_AUTH_TOKEN")
            } else if let Ok(Some(stored)) = load_stored_cloud_token(&cloud) {
                (Some(stored.capability_token), "stored_credentials")
            } else {
                (None, "anonymous_default")
            };
            if let Some(tok) = tok_opt {
                match CarbonCapabilityClaims::decode_unverified(&tok) {
                    Ok(claims) => {
                        let expired = now_secs > claims.exp_epoch_secs;
                        let _ = write_stdout_line(
                            &serde_json::to_string_pretty(&serde_json::json!({
                                "cloud": cloud.trim().to_ascii_lowercase(),
                                "endpoint": ep,
                                "auth_mode": if expired { "expired_token" } else { "capability_token" },
                                "source": source_label,
                                "tenant_id": claims.tenant_id,
                                "tier": claims.tier,
                                "admin": claims.admin,
                                "expired": expired,
                                "exp_epoch_secs": claims.exp_epoch_secs,
                                "ttl_remaining_secs": claims.exp_epoch_secs.saturating_sub(now_secs),
                            }))
                            .unwrap_or_default(),
                        );
                        if expired {
                            ExitCode::from(EXIT_AUTH_EXPIRED)
                        } else {
                            ExitCode::SUCCESS
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::from(2)
                    }
                }
            } else {
                let _ = write_stdout_line(
                    &serde_json::to_string_pretty(&serde_json::json!({
                        "cloud": cloud.trim().to_ascii_lowercase(),
                        "endpoint": ep,
                        "auth_mode": "anonymous_free",
                        "source": source_label,
                        "tier": "free",
                    }))
                    .unwrap_or_default(),
                );
                ExitCode::SUCCESS
            }
        }
        Command::Logout { cloud, all } => {
            let target = if all {
                None
            } else {
                Some(cloud.as_deref().unwrap_or("prod"))
            };
            match remove_stored_cloud_token(target) {
                Ok(removed) => {
                    let _ = write_stdout_line(
                        &serde_json::to_string_pretty(&serde_json::json!({
                            "logged_out": true,
                            "scope": target.unwrap_or("all"),
                            "entries_removed": removed,
                        }))
                        .unwrap_or_default(),
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Doctor {
            cloud,
            clouds_manifest,
            offline,
            allow_test_pubkey,
        } => execute_doctor_command(
            &cloud,
            clouds_manifest.as_deref(),
            offline,
            allow_test_pubkey.as_deref(),
        ),
        Command::Upgrade {
            check,
            force,
            version,
            base_url,
            allow_test_pubkey,
        } => {
            let using_test_pubkey = allow_test_pubkey.is_some();
            let pub_pem = if let Some(p) = allow_test_pubkey {
                match fs::read_to_string(&p) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: read --allow-test-pubkey {}: {e}", p.display());
                        return ExitCode::from(2);
                    }
                }
            } else {
                let canonical_pub_pem = canonical_embedded_release_signing_pub_pem();
                let fp = sha256_lower_hex(canonical_pub_pem.as_bytes());
                if fp != EXPECTED_SIGNING_PUB_SHA256 {
                    eprintln!("error: embedded signing key fingerprint mismatch");
                    return ExitCode::from(1);
                }
                canonical_pub_pem
            };
            match execute_self_upgrade_with_options(
                base_url.as_deref(),
                version.as_deref(),
                &pub_pem,
                check,
                force || using_test_pubkey,
            ) {
                Ok(receipt) => {
                    if let Ok(s) = serde_json::to_string_pretty(&receipt) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon upgrade failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Uninstall {
            keep_data,
            purge_all,
            yes,
            bin_path,
        } => execute_uninstall_command(keep_data, purge_all, yes, bin_path),
        _ => unreachable!("is_auth_doctor_cmd checked"),
    }
}

fn parse_sha256sums_entries(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let Some(hex_part) = parts.next() else {
            continue;
        };
        let rest = parts.next().unwrap_or("").trim().trim_start_matches('*');
        if hex_part.len() != 64
            || !hex_part.chars().all(|c| c.is_ascii_hexdigit())
            || rest.is_empty()
        {
            return Err(format!(
                "malformed SHA256SUMS.txt line {}: '{line}'",
                idx + 1
            ));
        }
        out.push((hex_part.to_ascii_lowercase(), rest.to_owned()));
    }
    if out.is_empty() {
        return Err("SHA256SUMS.txt contains zero digest entries".to_owned());
    }
    Ok(out)
}

fn execute_self_verify_command(
    manifest_arg: Option<PathBuf>,
    signature_arg: Option<PathBuf>,
    clouds_arg: Option<PathBuf>,
    allow_test_pubkey: Option<PathBuf>,
    require_manifest: bool,
    binary_only: bool,
) -> ExitCode {
    let exe_path = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: locate current executable: {e}");
            return ExitCode::from(1);
        }
    };
    let exe_bytes = match fs::read(&exe_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: read executable {}: {e}", exe_path.display());
            return ExitCode::from(1);
        }
    };
    let exe_sha256 = sha256_lower_hex(&exe_bytes);

    let canonical_embedded_pub_pem = canonical_embedded_release_signing_pub_pem();
    let embedded_pub_sha = sha256_lower_hex(canonical_embedded_pub_pem.as_bytes());
    if embedded_pub_sha != EXPECTED_SIGNING_PUB_SHA256 {
        eprintln!(
            "error: embedded release public key SHA-256 mismatch (got {embedded_pub_sha}, expected {EXPECTED_SIGNING_PUB_SHA256})"
        );
        return ExitCode::from(1);
    }

    let (pubkey_pem, using_test_pubkey) = if let Some(test_pub) = &allow_test_pubkey {
        match fs::read_to_string(test_pub) {
            Ok(pem) => (pem.replace("\r\n", "\n"), true),
            Err(e) => {
                eprintln!(
                    "error: read --allow-test-pubkey {}: {e}",
                    test_pub.display()
                );
                return ExitCode::from(2);
            }
        }
    } else {
        (canonical_embedded_pub_pem, false)
    };

    let config_dir_opt = verified_absolute_jopedime_config_dir().ok();
    let effective_clouds_path = clouds_arg.or_else(|| {
        config_dir_opt
            .as_ref()
            .map(|d| d.join("carbon-clouds.json"))
            .filter(|p| p.is_file())
    });

    let mut clouds_verified = false;
    let mut clouds_sha256: Option<String> = None;
    let mut kek_pins_verified = 0usize;
    if let Some(cpath) = &effective_clouds_path {
        let raw_clouds = match fs::read(cpath) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read clouds manifest {}: {e}", cpath.display());
                return ExitCode::from(1);
            }
        };
        let parsed: CarbonPublicCloudsManifest = match serde_json::from_slice(&raw_clouds) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: invalid JSON in {}: {e}", cpath.display());
                return ExitCode::from(1);
            }
        };
        for (cname, cell) in &parsed.clouds {
            let pins = cell.pinned_keks_by_endpoint();
            if pins.is_empty() {
                eprintln!(
                    "error: cloud '{cname}' in {} has zero P-256 KEK pins",
                    cpath.display()
                );
                return ExitCode::from(1);
            }
            for (ep, pin_b64) in &pins {
                if let Err(e) = decode_p256_pubkey_b64(pin_b64) {
                    eprintln!("error: invalid P-256 KEK pin for '{cname}' endpoint '{ep}': {e}");
                    return ExitCode::from(1);
                }
                kek_pins_verified += 1;
            }
        }
        clouds_sha256 = Some(sha256_lower_hex(&raw_clouds));
        clouds_verified = true;
    }

    let explicit_manifest = manifest_arg.is_some();
    let config_sig_exists = config_dir_opt
        .as_ref()
        .is_some_and(|d| d.join("SHA256SUMS.txt.sig").is_file());
    let must_verify_manifest = !binary_only
        && (explicit_manifest
            || require_manifest
            || signature_arg.is_some()
            || effective_clouds_path.is_some()
            || config_sig_exists);
    let effective_manifest_path = if binary_only {
        None
    } else {
        manifest_arg.or_else(|| {
            config_dir_opt
                .as_ref()
                .map(|d| d.join("SHA256SUMS.txt"))
                .filter(|p| p.is_file())
        })
    };

    let mut manifest_verified = false;
    let mut matched_binary_asset: Option<String> = None;
    let mut manifest_sha256: Option<String> = None;
    if let Some(mpath) = &effective_manifest_path {
        let sig_path = signature_arg.unwrap_or_else(|| {
            let mut s = mpath.as_os_str().to_os_string();
            s.push(".sig");
            PathBuf::from(s)
        });
        let manifest_bytes = match fs::read(mpath) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read {}: {e}", mpath.display());
                return ExitCode::from(1);
            }
        };
        let sig_raw = match fs::read_to_string(&sig_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: read {}: {e}", sig_path.display());
                return ExitCode::from(1);
            }
        };
        let clean_b64: String = sig_raw.chars().filter(|c| !c.is_whitespace()).collect();
        let sig_der = match base64::engine::general_purpose::STANDARD.decode(clean_b64.as_bytes()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "error: malformed base64 signature {}: {e}",
                    sig_path.display()
                );
                return ExitCode::from(1);
            }
        };
        if let Err(err) =
            verify_ecdsa_p256_spki_pem_signature(&pubkey_pem, &manifest_bytes, &sig_der)
        {
            eprintln!(
                "error: {} failed ECDSA-P256 verification against {}: {err}",
                mpath.display(),
                if using_test_pubkey {
                    "test public key"
                } else {
                    "embedded owner release public key"
                }
            );
            return ExitCode::from(1);
        }
        let manifest_text = match std::str::from_utf8(&manifest_bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {} is not valid UTF-8: {e}", mpath.display());
                return ExitCode::from(1);
            }
        };
        let entries = match parse_sha256sums_entries(manifest_text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        };
        for (digest, name) in &entries {
            if digest == &exe_sha256 {
                matched_binary_asset = Some(name.clone());
                break;
            }
        }
        if matched_binary_asset.is_none() {
            eprintln!(
                "error: running executable SHA-256 ({exe_sha256}) does not match any entry in signed {}",
                mpath.display()
            );
            return ExitCode::from(1);
        }
        if let Some(csha) = &clouds_sha256 {
            if let Some((expected_clouds_sha, _)) = entries
                .iter()
                .find(|(_, name)| name == "carbon-clouds.json")
            {
                if expected_clouds_sha != csha {
                    eprintln!(
                        "error: carbon-clouds.json SHA-256 ({csha}) does not match signed manifest ({expected_clouds_sha})"
                    );
                    return ExitCode::from(1);
                }
            }
        }
        manifest_sha256 = Some(sha256_lower_hex(&manifest_bytes));
        manifest_verified = true;
    } else if must_verify_manifest {
        eprintln!(
            "error: signed release manifest SHA256SUMS.txt is missing (pass --manifest <PATH> or --binary-only for uninstalled dev builds)"
        );
        return ExitCode::from(1);
    }

    let _ = write_stdout_line(
        &serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
            "binary_only": binary_only,
            "version": CARBON_PKG_VERSION,
            "release_tag": CARBON_RELEASE_TAG,
            "git_sha": CARBON_GIT_SHA,
            "build_target": CARBON_BUILD_TARGET,
            "build_profile": CARBON_BUILD_PROFILE,
            "executable_path": exe_path,
            "executable_bytes": exe_bytes.len(),
            "executable_sha256": exe_sha256,
            "signing_pub_fingerprint_sha256": embedded_pub_sha,
            "test_pubkey_override": using_test_pubkey,
            "manifest_ecdsa_verified": manifest_verified,
            "manifest_sha256": manifest_sha256,
            "matched_binary_asset": matched_binary_asset,
            "clouds_manifest_verified": clouds_verified,
            "clouds_manifest_sha256": clouds_sha256,
            "kek_pins_verified": kek_pins_verified,
        }))
        .unwrap_or_default(),
    );
    ExitCode::SUCCESS
}

fn execute_doctor_command(
    cloud_arg: &str,
    clouds_manifest: Option<&str>,
    offline: bool,
    allow_test_pubkey: Option<&Path>,
) -> ExitCode {
    let mut healthy = true;
    let mut checks = Vec::new();
    let tty_stderr = std::io::stderr().is_terminal();

    let roles = compiled_binary_roles();
    let canonical_pub_pem = canonical_embedded_release_signing_pub_pem();
    let pub_fp = sha256_lower_hex(canonical_pub_pem.as_bytes());
    let pub_fp_ok = pub_fp == EXPECTED_SIGNING_PUB_SHA256;
    if !pub_fp_ok {
        healthy = false;
    }
    let verify_pub_pem = match allow_test_pubkey {
        Some(test_p) => match fs::read_to_string(test_p) {
            Ok(s) => s.replace("\r\n", "\n"),
            Err(e) => {
                eprintln!("error: read --allow-test-pubkey {}: {e}", test_p.display());
                return ExitCode::from(2);
            }
        },
        None => canonical_pub_pem.clone(),
    };
    let current_exe_opt = env::current_exe().ok();
    let first_path_bin = find_first_carbon_in_path();
    let shadowed_by = match (&current_exe_opt, &first_path_bin) {
        (Some(cur), Some(in_path)) => {
            let cur_canon = fs::canonicalize(cur).unwrap_or_else(|_| cur.clone());
            let path_canon = fs::canonicalize(in_path).unwrap_or_else(|_| in_path.clone());
            if cur_canon != path_canon {
                Some(in_path.display().to_string())
            } else {
                None
            }
        }
        _ => None,
    };
    checks.push(serde_json::json!({
        "check": "binary",
        "passed": pub_fp_ok,
        "version": CARBON_PKG_VERSION,
        "release_tag": CARBON_RELEASE_TAG,
        "git_sha": CARBON_GIT_SHA,
        "build_target": CARBON_BUILD_TARGET,
        "build_profile": CARBON_BUILD_PROFILE,
        "signing_pub_fingerprint_ok": pub_fp_ok,
        "executable_path": current_exe_opt.as_ref().map(|p| p.display().to_string()),
        "first_in_path": first_path_bin.as_ref().map(|p| p.display().to_string()),
        "path_shadowed_by": shadowed_by,
        "roles": roles,
    }));
    if tty_stderr {
        let sha12: String = CARBON_GIT_SHA.chars().take(12).collect();
        eprintln!(
            "[doctor] {} binary: carbon v{} ({}, git {}, target {}, roles: {})",
            if pub_fp_ok { "PASS" } else { "FAIL" },
            CARBON_PKG_VERSION,
            CARBON_RELEASE_TAG,
            sha12,
            CARBON_BUILD_TARGET,
            roles.join(", ")
        );
        if let Some(other_bin) = &shadowed_by {
            eprintln!(
                "[doctor] INFO path_shadow: PATH resolves `carbon` to '{other_bin}' before running binary"
            );
        }
    }

    let (config_dir_str, config_dir_ok, cred_mode_ok, cached_manifest_exists) =
        match verified_absolute_jopedime_config_dir() {
            Ok(dir) => {
                #[cfg(unix)]
                let (dir_mode_ok, dir_mode_octal) = {
                    use std::os::unix::fs::PermissionsExt;
                    let mut ok = true;
                    let mut octal = None;
                    if dir.exists() {
                        if let Ok(meta) = fs::metadata(&dir) {
                            let m = meta.permissions().mode() & 0o777;
                            octal = Some(format!("{m:04o}"));
                            if (m & 0o077) != 0 {
                                ok = false;
                            }
                        }
                    }
                    (ok, octal)
                };
                #[cfg(not(unix))]
                let (dir_mode_ok, dir_mode_octal): (bool, Option<String>) = (true, None);

                let cred_path = dir.join("carbon-credentials.json");
                #[cfg(unix)]
                let (cred_ok, cred_mode_octal) = {
                    use std::os::unix::fs::PermissionsExt;
                    let mut ok = true;
                    let mut octal = None;
                    if cred_path.is_file() {
                        if let Ok(meta) = fs::metadata(&cred_path) {
                            let m = meta.permissions().mode() & 0o777;
                            octal = Some(format!("{m:04o}"));
                            ok = m == 0o600;
                        }
                    }
                    (ok, octal)
                };
                #[cfg(not(unix))]
                let (cred_ok, cred_mode_octal): (bool, Option<String>) = {
                    let _ = &cred_path;
                    (true, None)
                };
                if !dir_mode_ok || !cred_ok {
                    healthy = false;
                }
                let cached_clouds = dir.join("carbon-clouds.json").is_file();
                let sums_path = dir.join("SHA256SUMS.txt");
                let sig_path = dir.join("SHA256SUMS.txt.sig");
                let manifest_sig_ok = if sums_path.is_file() && sig_path.is_file() {
                    let ok = match (fs::read(&sums_path), fs::read_to_string(&sig_path)) {
                        (Ok(mbytes), Ok(sraw)) => {
                            let clean: String =
                                sraw.chars().filter(|c| !c.is_whitespace()).collect();
                            base64::engine::general_purpose::STANDARD
                                .decode(clean.as_bytes())
                                .ok()
                                .and_then(|der| {
                                    verify_ecdsa_p256_spki_pem_signature(
                                        &verify_pub_pem,
                                        &mbytes,
                                        &der,
                                    )
                                    .ok()
                                })
                                .is_some()
                        }
                        _ => false,
                    };
                    if !ok {
                        healthy = false;
                    }
                    Some(ok)
                } else {
                    None
                };
                checks.push(serde_json::json!({
                    "check": "config_security",
                    "passed": dir_mode_ok && cred_ok && manifest_sig_ok.unwrap_or(true),
                    "config_dir": dir.display().to_string(),
                    "config_dir_exists": dir.exists(),
                    "config_dir_mode": dir_mode_octal,
                    "credentials_present": cred_path.is_file(),
                    "credentials_mode": cred_mode_octal,
                    "local_clouds_manifest_present": cached_clouds,
                    "local_release_manifest_ecdsa_verified": manifest_sig_ok,
                }));
                if tty_stderr {
                    let tag = if dir_mode_ok && cred_ok && manifest_sig_ok.unwrap_or(true) {
                        "PASS"
                    } else {
                        "FAIL"
                    };
                    eprintln!(
                        "[doctor] {tag} config_security: dir={} (mode={:?}, credentials_mode={:?}, local_pins={cached_clouds})",
                        dir.display(),
                        dir_mode_octal.as_deref().unwrap_or("absent"),
                        cred_mode_octal.as_deref().unwrap_or("absent"),
                    );
                }
                (
                    dir.display().to_string(),
                    dir_mode_ok,
                    cred_ok,
                    cached_clouds,
                )
            }
            Err(err) => {
                healthy = false;
                checks.push(serde_json::json!({
                    "check": "config_security",
                    "passed": false,
                    "error": err,
                }));
                if tty_stderr {
                    eprintln!("[doctor] FAIL config_security: {err}");
                }
                (String::new(), false, false, false)
            }
        };

    let now_secs = now_epoch_ms() / 1_000;
    let mut stored_summary = BTreeMap::new();
    match list_stored_cloud_tokens() {
        Ok(map) => {
            for (cname, cred) in map {
                let expired = now_secs > cred.exp_epoch_secs;
                let ttl_rem = cred.exp_epoch_secs.saturating_sub(now_secs);
                stored_summary.insert(
                    cname.clone(),
                    serde_json::json!({
                        "tenant_id": cred.tenant_id,
                        "tier": cred.tier,
                        "expired": expired,
                        "ttl_remaining_secs": ttl_rem,
                    }),
                );
                if tty_stderr {
                    eprintln!(
                        "[doctor] INFO stored_token[{cname}]: tenant={} tier={:?} expired={expired} ttl_remaining={}s",
                        cred.tenant_id, cred.tier, ttl_rem
                    );
                }
            }
        }
        Err(err) => {
            healthy = false;
            checks.push(serde_json::json!({
                "check": "credentials_parse",
                "passed": false,
                "error": err,
            }));
            if tty_stderr {
                eprintln!("[doctor] FAIL credentials_parse: {err}");
            }
        }
    }

    let clouds_to_check: Vec<&str> = match cloud_arg.trim().to_ascii_lowercase().as_str() {
        "all" => vec!["prod", "dev"],
        "prod" => vec!["prod"],
        "dev" => vec!["dev"],
        other => {
            eprintln!("error: unsupported --cloud '{other}' for `carbon doctor` (expected prod, dev, or all)");
            return ExitCode::from(2);
        }
    };

    let mut cloud_reports = BTreeMap::new();
    for cname in clouds_to_check {
        match resolve_public_cloud_endpoint(cname, clouds_manifest) {
            Ok(entry) => {
                let candidates = entry.candidate_endpoints();
                let pins = entry.pinned_keks_by_endpoint();
                let all_pinned = !candidates.is_empty() && pins.len() == candidates.len();
                if !all_pinned {
                    healthy = false;
                }
                if offline {
                    cloud_reports.insert(
                        cname.to_owned(),
                        serde_json::json!({
                            "passed": all_pinned,
                            "configured_regions": candidates.len(),
                            "kek_pins_count": pins.len(),
                            "offline": true,
                        }),
                    );
                    if tty_stderr {
                        let tag = if all_pinned { "PASS" } else { "FAIL" };
                        eprintln!(
                            "[doctor] {tag} cloud[{cname}]: {} regional endpoints, {} P-256 KEK pins verified (offline)",
                            candidates.len(),
                            pins.len()
                        );
                    }
                } else {
                    match CarbonThinClient::connect_fastest_pinned(candidates.clone(), &pins) {
                        Ok((probed_client, probed)) => {
                            let remote_ver = probed_client.fetch_server_version().ok().flatten();
                            let fastest = probed.first();
                            let probed_list: Vec<serde_json::Value> = probed
                                .iter()
                                .map(|p| {
                                    serde_json::json!({
                                        "region": p.region,
                                        "endpoint": p.base_url,
                                        "rtt_ms": p.rtt_ms,
                                    })
                                })
                                .collect();
                            let cloud_ok = all_pinned && !probed.is_empty();
                            if !cloud_ok {
                                healthy = false;
                            }
                            cloud_reports.insert(
                                cname.to_owned(),
                                serde_json::json!({
                                    "passed": cloud_ok,
                                    "configured_regions": candidates.len(),
                                    "healthy_regions": probed.len(),
                                    "kek_pins_verified": all_pinned,
                                    "fastest_region": fastest.map(|f| f.region.clone()),
                                    "fastest_rtt_ms": fastest.map(|f| f.rtt_ms),
                                    "server_git_sha": remote_ver.as_ref().map(|v| v.git_sha.clone()),
                                    "server_release_id": remote_ver.as_ref().map(|v| v.release_id.clone()),
                                    "regions": probed_list,
                                }),
                            );
                            if tty_stderr {
                                if let Some(f) = fastest {
                                    eprintln!(
                                        "[doctor] PASS cloud[{cname}]: {}/{} healthy regions | fastest={} ({:.1} ms RTT) | KEK pins verified={all_pinned}",
                                        probed.len(),
                                        candidates.len(),
                                        f.region,
                                        f.rtt_ms
                                    );
                                }
                            }
                        }
                        Err(err) => {
                            healthy = false;
                            cloud_reports.insert(
                                cname.to_owned(),
                                serde_json::json!({
                                    "passed": false,
                                    "configured_regions": candidates.len(),
                                    "error": err,
                                }),
                            );
                            if tty_stderr {
                                eprintln!("[doctor] FAIL cloud[{cname}]: {err}");
                            }
                        }
                    }
                }
            }
            Err(err) => {
                healthy = false;
                cloud_reports.insert(
                    cname.to_owned(),
                    serde_json::json!({
                        "passed": false,
                        "error": err,
                    }),
                );
                if tty_stderr {
                    eprintln!("[doctor] FAIL cloud[{cname}] manifest: {err}");
                }
            }
        }
    }

    if tty_stderr {
        if stored_summary.is_empty() {
            eprintln!("[doctor] HINT: Run `carbon register --free` to mint a 30-day Free capability token (or run immediately in anonymous Free mode).");
        }
        eprintln!(
            "[doctor] OVERALL: {}",
            if healthy {
                "HEALTHY"
            } else {
                "DEGRADED / ACTION REQUIRED"
            }
        );
    }

    let _ = write_stdout_line(
        &serde_json::to_string_pretty(&serde_json::json!({
            "healthy": healthy,
            "version": CARBON_PKG_VERSION,
            "release_tag": CARBON_RELEASE_TAG,
            "git_sha": CARBON_GIT_SHA,
            "build_target": CARBON_BUILD_TARGET,
            "build_profile": CARBON_BUILD_PROFILE,
            "config_dir": config_dir_str,
            "config_dir_permissions_ok": config_dir_ok,
            "credentials_permissions_ok": cred_mode_ok,
            "local_clouds_manifest_present": cached_manifest_exists,
            "default_clouds_url": DEFAULT_PUBLIC_CLOUDS_URL,
            "fallback_clouds_url": FALLBACK_PUBLIC_CLOUDS_URL,
            "stored_credentials": stored_summary,
            "clouds": cloud_reports,
            "checks": checks,
        }))
        .unwrap_or_default(),
    );

    if healthy {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn zero_overwrite_and_remove_file(path: &Path) -> Result<bool, String> {
    if !path.is_file() {
        return Ok(false);
    }
    if let Ok(meta) = fs::metadata(path) {
        let len = usize::try_from(meta.len().min(4 * 1024 * 1024)).unwrap_or(0);
        if len > 0 {
            let zeros = vec![0u8; len];
            if let Ok(mut f) = fs::OpenOptions::new().write(true).open(path) {
                let _ = f.write_all(&zeros);
                let _ = f.sync_all();
            }
        }
    }
    fs::remove_file(path).map_err(|e| format!("remove {}: {e}", path.display()))?;
    Ok(true)
}

fn execute_uninstall_command(
    keep_data: bool,
    purge_all: bool,
    yes: bool,
    bin_path_override: Option<PathBuf>,
) -> ExitCode {
    let explicit_override = bin_path_override.or_else(|| {
        env::var_os("CARBON_UNINSTALL_BIN_PATH")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    });
    let had_override = explicit_override.is_some();
    let target_bin = match explicit_override.map(Ok).unwrap_or_else(env::current_exe) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: locate executable to uninstall: {e}");
            return ExitCode::from(1);
        }
    };
    let bin_display = target_bin.to_string_lossy();
    if !had_override && (bin_display.contains("/target/") || bin_display.contains("\\target\\")) {
        eprintln!(
            "error: refusing to delete Cargo build artifact '{}'; pass --bin-path <INSTALLED_CARBON_PATH>",
            target_bin.display()
        );
        return ExitCode::from(2);
    }
    if purge_all && !yes {
        if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
            eprint!(
                "Permanently shred all saved runs and credentials before uninstalling? [y/N]: "
            );
            let _ = std::io::stderr().flush();
            let mut ans = String::new();
            if std::io::stdin().read_line(&mut ans).is_err() {
                return ExitCode::from(2);
            }
            if !matches!(ans.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                eprintln!("error: uninstall --purge-all cancelled");
                return ExitCode::from(2);
            }
        } else {
            eprintln!("error: --purge-all requires --yes in non-interactive mode");
            return ExitCode::from(2);
        }
    }

    let mut removed_config_files: Vec<String> = Vec::new();
    let mut shredded_handles = 0usize;
    let mut shredded_credentials = false;
    if !keep_data {
        if let Ok(cfg_dir) = verified_absolute_jopedime_config_dir() {
            for rel in ["carbon-clouds.json", "SHA256SUMS.txt", "SHA256SUMS.txt.sig"] {
                let p = cfg_dir.join(rel);
                if p.is_file() && fs::remove_file(&p).is_ok() {
                    removed_config_files.push(p.display().to_string());
                }
            }
            if purge_all {
                let cred_p = cfg_dir.join("carbon-credentials.json");
                let lock_p = cfg_dir.join("carbon-credentials.json.lock");
                if let Ok(true) = zero_overwrite_and_remove_file(&cred_p) {
                    shredded_credentials = true;
                }
                let _ = fs::remove_file(&lock_p);
                let runs_dir = cfg_dir.join("runs");
                if let Ok(entries) = fs::read_dir(&runs_dir) {
                    for ent in entries.flatten() {
                        let p = ent.path();
                        if p.is_file() && zero_overwrite_and_remove_file(&p).unwrap_or(false) {
                            shredded_handles += 1;
                        }
                    }
                    let _ = fs::remove_dir(&runs_dir);
                }
                let _ = fs::remove_dir(&cfg_dir);
            }
        }
    }

    let mut binary_removed = false;
    if target_bin.is_file() {
        match fs::remove_file(&target_bin) {
            Ok(()) => binary_removed = true,
            Err(e) => {
                #[cfg(windows)]
                {
                    let parked =
                        target_bin.with_extension(format!("exe.old.{}", std::process::id()));
                    if fs::rename(&target_bin, &parked).is_ok() {
                        binary_removed = true;
                    } else {
                        eprintln!("error: remove {}: {e}", target_bin.display());
                        return ExitCode::from(1);
                    }
                }
                #[cfg(not(windows))]
                {
                    eprintln!("error: remove {}: {e}", target_bin.display());
                    return ExitCode::from(1);
                }
            }
        }
    }

    let _ = write_stdout_line(
        &serde_json::to_string_pretty(&serde_json::json!({
            "uninstalled": true,
            "binary_path": target_bin,
            "binary_removed": binary_removed,
            "keep_data": keep_data,
            "purge_all": purge_all,
            "removed_config_files": removed_config_files,
            "shredded_credentials": shredded_credentials,
            "shredded_handles": shredded_handles,
        }))
        .unwrap_or_default(),
    );
    ExitCode::SUCCESS
}
