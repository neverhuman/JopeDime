//! Packaging, CloudFormation deployment, rollback, Enterprise `serve`, local Lambda `emu`, benchmarks, red-team, and spec reporting.

use super::*;

pub(crate) fn dispatch_ops_admin(cmd: Command, global_quiet: bool) -> ExitCode {
    match cmd {
        Command::Version => {
            let _ = write_stdout_line(
                &serde_json::json!({
                    "binary": "carbon",
                    "version": CARBON_PKG_VERSION,
                    "release_tag": CARBON_RELEASE_TAG,
                    "git_sha": CARBON_GIT_SHA,
                    "build_target": CARBON_BUILD_TARGET,
                    "build_profile": CARBON_BUILD_PROFILE,
                    "thin_cli_only": !cfg!(feature = "engine"),
                    "roles": compiled_binary_roles(),
                })
                .to_string(),
            );
            ExitCode::SUCCESS
        }
        #[cfg(feature = "engine")]
        Command::Package {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            jobs,
            include_cli_binaries,
        } => match execute_carbon_package(&CarbonPackageOptions {
            r882_dir,
            out_dir,
            edge_bin,
            worker_bin,
            build_jobs: jobs,
            include_cli_binaries,
        }) {
            Ok(manifest) => match serde_json::to_string_pretty(&manifest) {
                Ok(s) => {
                    let _ = write_stdout_line(&s);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: serialize manifest: {e}");
                    ExitCode::from(1)
                }
            },
            Err(err) => {
                eprintln!("error: carbon package failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::Deploy {
            manifest,
            check_drift,
            clouds_manifest,
            environment,
            regions,
            artifacts_bucket,
            bootstrap_secrets,
            decommission_unstacked_prototype,
            verify_local_lambda,
            dry_run,
            skip_redteam,
            app_auth_pubkeys_b64,
            json_out,
        } => {
            if verify_local_lambda {
                #[cfg(feature = "emu")]
                {
                    let want_real = resolve_real_worker_binaries(None, None).is_ok();
                    if let Err(err) =
                        run_emu_self_test_with_output(None, None, want_real, None, None, false)
                    {
                        eprintln!(
                            "error: carbon deploy --verify-local-lambda preflight failed: {err}"
                        );
                        return ExitCode::from(1);
                    }
                }
                #[cfg(not(feature = "emu"))]
                {
                    eprintln!(
                        "error: --verify-local-lambda requires a build with `--features emu`"
                    );
                    return ExitCode::from(2);
                }
            }
            if check_drift {
                return match execute_carbon_drift_check(
                    &environment,
                    &regions,
                    clouds_manifest.as_deref().and_then(|p| p.to_str()),
                ) {
                    Ok(report) => {
                        let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                            return ExitCode::from(1);
                        };
                        if let Err(e) = write_optional_outputs(json_out, None, &json_text, "") {
                            eprintln!("error: {e}");
                            return ExitCode::from(1);
                        }
                        let _ = write_stdout_line(&json_text);
                        if report.all_in_sync {
                            ExitCode::SUCCESS
                        } else {
                            ExitCode::from(1)
                        }
                    }
                    Err(err) => {
                        eprintln!("error: carbon deploy --check-drift failed: {err}");
                        ExitCode::from(1)
                    }
                };
            }
            let Some(manifest_path) = manifest else {
                eprintln!("error: --manifest <PATH> is required unless --check-drift is passed");
                return ExitCode::from(2);
            };
            match jope_carbon::cli::deploy::execute_carbon_deploy_with_app_pubkeys(
                &CarbonDeployOptions {
                    manifest_path,
                    environment,
                    regions,
                    artifacts_bucket_override: artifacts_bucket,
                    bootstrap_secrets,
                    decommission_unstacked_prototype,
                    dry_run,
                    skip_redteam,
                },
                app_auth_pubkeys_b64.as_deref(),
            ) {
                Ok(report) => {
                    let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                        return ExitCode::from(1);
                    };
                    if let Err(e) = write_optional_outputs(json_out, None, &json_text, "") {
                        eprintln!("error: {e}");
                        return ExitCode::from(1);
                    }
                    let _ = write_stdout_line(&json_text);
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon deploy failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        #[cfg(feature = "engine")]
        Command::Rollback {
            environment,
            regions,
            to_versions,
        } => {
            let target_versions = if to_versions.is_empty() {
                None
            } else if to_versions.len() == 3 {
                Some(CarbonLiveAliasVersions {
                    edge_version: to_versions[0].clone(),
                    engine_version: to_versions[1].clone(),
                    eval_version: to_versions[2].clone(),
                })
            } else {
                eprintln!("error: --to-versions expects exactly 3 comma-separated versions (<edge>,<engine>,<eval>)");
                return ExitCode::from(2);
            };
            match execute_carbon_rollback_multi(&environment, &regions, target_versions) {
                Ok(mut receipts) => {
                    let s = if receipts.len() == 1 {
                        serde_json::to_string_pretty(&receipts.remove(0))
                    } else {
                        serde_json::to_string_pretty(&receipts)
                    };
                    if let Ok(text) = s {
                        let _ = write_stdout_line(&text);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon rollback failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        #[cfg(feature = "engine")]
        Command::CloudsManifest {
            out,
            dev_regions,
            prod_regions,
        } => match generate_public_clouds_manifest(&dev_regions, &prod_regions) {
            Ok(manifest) => {
                let Ok(json_bytes) = serde_json::to_vec_pretty(&manifest) else {
                    eprintln!("error: serialize clouds manifest failed");
                    return ExitCode::from(1);
                };
                if let Some(parent) = out.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                if let Err(e) = fs::write(&out, &json_bytes) {
                    eprintln!("error: write {}: {e}", out.display());
                    return ExitCode::from(1);
                }
                let _ = write_stdout_line(&String::from_utf8_lossy(&json_bytes));
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon clouds-manifest failed: {err}");
                ExitCode::from(1)
            }
        },
        Command::Completions { shell } => match emit_shell_completions(&shell) {
            Ok(script) => {
                print!("{script}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(2)
            }
        },
        Command::Bench {
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
            suite,
            datasets_dir,
            csv_dir,
            effort,
            seed,
            free,
            pro,
            tier,
            token,
            token_file,
            token_stdin,
            tenant,
            json_out,
            markdown_out,
        } => {
            let explicit_tier = match resolve_explicit_tier_flag(free, pro, tier.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let resolved = match resolve_cli_target_and_auth(
                endpoint,
                cloud,
                clouds_manifest.as_deref(),
                environment.as_deref(),
                &region,
                explicit_tier,
                tenant,
                "bench_tenant",
                token,
                token_file,
                token_stdin,
                CarbonTierWire::Pro,
            ) {
                Ok(r) => r,
                Err((code, msg)) => {
                    eprintln!("error: {msg}");
                    return code;
                }
            };
            let suite_norm = suite
                .as_deref()
                .map(|s| s.trim().to_ascii_lowercase())
                .unwrap_or_else(|| {
                    if datasets_dir.is_some() || csv_dir.is_some() {
                        "standard".to_owned()
                    } else {
                        "ops".to_owned()
                    }
                });
            if suite_norm != "ops" {
                let Ok(effort_val) = parse_api_effort(&effort) else {
                    eprintln!("error: invalid --effort `{effort}`");
                    return ExitCode::from(2);
                };
                let sci_opts = CarbonScientificBenchmarkOptions {
                    endpoints: resolved.endpoints,
                    pinned_keks: resolved.pinned_keks,
                    tier: resolved.tier,
                    effort: effort_val,
                    tenant_id: resolved.tenant_id,
                    auth_token: resolved.auth_token,
                    suite: suite_norm,
                    datasets_dir,
                    csv_dir,
                    seed,
                };
                return match execute_scientific_benchmark_suite(
                    &sci_opts,
                    std::io::stderr().is_terminal(),
                ) {
                    Ok(report) => {
                        let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                            return ExitCode::from(1);
                        };
                        let md_text = render_scientific_benchmark_markdown(&report);
                        if let Err(e) =
                            write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                        {
                            eprintln!("error: {e}");
                            return ExitCode::from(1);
                        }
                        let _ = write_stdout_line(&json_text);
                        ExitCode::SUCCESS
                    }
                    Err(err) => {
                        eprintln!("error: carbon scientific bench failed: {err}");
                        ExitCode::from(1)
                    }
                };
            }
            match execute_carbon_cli_benchmark_with_token(
                resolved.endpoints,
                resolved.tier,
                &resolved.tenant_id,
                resolved.auth_token,
            ) {
                Ok(report) => {
                    let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                        return ExitCode::from(1);
                    };
                    let md_text = render_benchmark_markdown(&report);
                    if let Err(e) =
                        write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                    {
                        eprintln!("error: {e}");
                        return ExitCode::from(1);
                    }
                    let _ = write_stdout_line(&json_text);
                    if report.all_passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(err) => {
                    eprintln!("error: carbon bench failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Redteam {
            endpoint,
            cloud,
            clouds_manifest,
            environment,
            region,
            json_out,
            markdown_out,
        } => {
            let resolved_ep = match (endpoint, cloud, environment) {
                (Some(ep), _, _) => ep,
                (None, Some(cloud_name), _) => {
                    match resolve_public_cloud_endpoint(&cloud_name, clouds_manifest.as_deref()) {
                        Ok(entry) => entry.public_endpoint,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                (None, None, Some(env_name)) => {
                    match resolve_stack_endpoint_if_compiled(&env_name, &region) {
                        Ok(url) => url,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
                (None, None, None) => {
                    eprintln!(
                        "error: pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>"
                    );
                    return ExitCode::from(2);
                }
            };
            match execute_carbon_redteam_suite(&resolved_ep) {
                Ok(report) => {
                    let Ok(json_text) = serde_json::to_string_pretty(&report) else {
                        return ExitCode::from(1);
                    };
                    let md_text = render_redteam_markdown(&report);
                    if let Err(e) =
                        write_optional_outputs(json_out, markdown_out, &json_text, &md_text)
                    {
                        eprintln!("error: {e}");
                        return ExitCode::from(1);
                    }
                    let _ = write_stdout_line(&json_text);
                    if report.all_passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(err) => {
                    eprintln!("error: carbon redteam failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::TokenMint {
            tenant,
            tier,
            ttl_secs,
            admin,
            environment,
            region,
            signing_key_file,
            issuer,
            kid,
        } => {
            if ttl_secs == 0 || ttl_secs > MAX_MINT_TOKEN_TTL_SECS {
                eprintln!(
                    "error: --ttl-secs must be within 1..={MAX_MINT_TOKEN_TTL_SECS} seconds (got {ttl_secs})"
                );
                return ExitCode::from(2);
            }
            let Ok(tier_wire) = parse_tier_wire(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            if let Some(key_path) = signing_key_file {
                if admin {
                    eprintln!("error: asymmetric ctk2.* app-minted tokens cannot grant --admin");
                    return ExitCode::from(2);
                }
                let app_tier = match tier_wire {
                    CarbonTierWire::Free => jope_carbon::envelope::AppTokenTier::Free,
                    CarbonTierWire::Pro => jope_carbon::envelope::AppTokenTier::Pro,
                    CarbonTierWire::Enterprise => {
                        eprintln!("error: asymmetric ctk2.* app-minted tokens support only `free` or `pro` tiers");
                        return ExitCode::from(2);
                    }
                };
                let raw_key = match fs::read_to_string(&key_path) {
                    Ok(s) => zeroize::Zeroizing::new(s),
                    Err(e) => {
                        eprintln!("error: read signing key {}: {e}", key_path.display());
                        return ExitCode::from(2);
                    }
                };
                let trimmed_key = raw_key.trim();
                let signing_key = if trimmed_key.starts_with("-----BEGIN ") {
                    jope_carbon::envelope::AppSigningKey::from_pkcs8_pem(trimmed_key)
                } else {
                    jope_carbon::envelope::AppSigningKey::from_hex(trimmed_key)
                };
                let signing_key = match signing_key {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                };
                let exp = (now_epoch_ms() / 1_000).saturating_add(ttl_secs);
                let claims = jope_carbon::envelope::AppCapabilityClaims {
                    tenant_id: tenant,
                    tier: app_tier,
                    exp_epoch_secs: exp,
                    admin: false,
                    iss: Some(issuer.unwrap_or_else(|| "app.veox.ai".to_owned())),
                    kid,
                };
                return match signing_key.mint_token(&claims) {
                    Ok(tok) => {
                        let _ = write_stdout_line(&tok);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::from(2)
                    }
                };
            }
            if let Some(env_name) = environment {
                return match mint_vault_token_if_compiled(
                    &env_name, &region, &tenant, tier_wire, ttl_secs, admin,
                ) {
                    Ok(tok) => {
                        let _ = write_stdout_line(&tok);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::from(2)
                    }
                };
            }
            let Ok(secret_hex) = env::var("CARBON_AUTH_SECRET_HEX") else {
                eprintln!("error: pass --env <dev|staging|prod> or set CARBON_AUTH_SECRET_HEX");
                return ExitCode::from(2);
            };
            let verifier = match CarbonAuthVerifier::from_hex(&secret_hex) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let exp = (now_epoch_ms() / 1_000).saturating_add(ttl_secs);
            match verifier.mint_token(&CarbonCapabilityClaims {
                tenant_id: tenant,
                tier: tier_wire,
                exp_epoch_secs: exp,
                admin,
            }) {
                Ok(tok) => {
                    let _ = write_stdout_line(&tok);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Plan {
            tier,
            effort,
            total_rows,
            fit_rows,
            features,
            ms_per_candidate,
        } => {
            let Ok(product_tier) = parse_product_tier(&tier) else {
                eprintln!("error: invalid --tier `{tier}`");
                return ExitCode::from(2);
            };
            let effort_tier: CarbonEffortTier = match effort.parse() {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("error: {err} (hint: {})", err.guidance().repair_hint);
                    return ExitCode::from(2);
                }
            };
            let est_ms = ms_per_candidate
                .unwrap_or_else(|| estimate_ms_per_candidate_1vcpu(fit_rows, features));
            match plan_carbon_execution(&CarbonPlanInput {
                product_tier,
                effort: effort_tier,
                total_rows,
                fit_rows,
                features,
                est_ms_per_candidate_1vcpu: est_ms,
            }) {
                Ok(decision) => {
                    if let Ok(s) = serde_json::to_string_pretty(&decision) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err} (hint: {})", err.guidance().repair_hint);
                    ExitCode::from(2)
                }
            }
        }
        Command::SpotCheck {
            region,
            monthly_spend_usd,
            p20_vcpus,
            mean_vcpus,
            p95_vcpus,
            cold_start_rate,
            consecutive_months,
        } => {
            let sample = MonthlyWindowMetrics::for_region(
                &region,
                monthly_spend_usd,
                p20_vcpus,
                mean_vcpus,
                p95_vcpus,
                cold_start_rate,
            );
            let history = vec![sample; consecutive_months.max(1)];
            let verdict = evaluate_spot_hybrid_gates(&history, &SpotAdvisorThresholds::default());
            if let Ok(s) = serde_json::to_string_pretty(&verdict) {
                let _ = write_stdout_line(&s);
            }
            ExitCode::SUCCESS
        }
        #[cfg(feature = "serve")]
        Command::Serve {
            listen,
            data_dir,
            cas_url,
            allow_local_fs_dev,
            dev_ephemeral,
            region,
            enable_user_envelope,
        } => {
            let effective_cas_url = cas_url
                .or_else(|| env::var("CARBON_CAS_URL").ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
            let is_loopback = listen.ip().is_loopback();
            if !is_loopback && effective_cas_url.is_none() && !allow_local_fs_dev {
                eprintln!(
                    "error: non-loopback `carbon serve` requires --cas-url s3://... (or CARBON_CAS_URL), or explicit --allow-local-fs-dev"
                );
                return ExitCode::from(2);
            }
            let allow_ephemeral_secrets = is_loopback || dev_ephemeral;
            let mut cfg = CarbonEnterpriseServeConfig::enterprise_default(listen, data_dir);
            cfg.cas_url = effective_cas_url;
            cfg.allow_local_fs_dev = is_loopback || allow_local_fs_dev;
            cfg.region = region;
            cfg.enable_user_envelope = enable_user_envelope;
            cfg.seal_secret = match decode_or_ephemeral_32byte_secret(
                "CARBON_SEAL_SECRET_HEX",
                allow_ephemeral_secrets,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            cfg.presign_secret = match decode_or_ephemeral_32byte_secret(
                "CARBON_PRESIGN_SECRET_HEX",
                allow_ephemeral_secrets,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if let Ok(auth_hex) = env::var("CARBON_AUTH_SECRET_HEX") {
                if !auth_hex.trim().is_empty() {
                    match decode_required_32byte_hex("CARBON_AUTH_SECRET_HEX") {
                        Ok(s) => cfg.auth_secret = Some(s),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if let Err(err) = CarbonEnterpriseServer::run_until_ctrl_c(cfg) {
                eprintln!("error: carbon serve failed: {err}");
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        #[cfg(feature = "emu")]
        Command::Emu {
            work_dir,
            listen,
            self_test,
            real_workers,
            edge_bin,
            worker_bin,
        } => {
            if self_test {
                match run_emu_self_test_with_output(
                    work_dir,
                    listen,
                    real_workers,
                    edge_bin,
                    worker_bin,
                    true,
                ) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("error: carbon emu --self-test failed: {err}");
                        ExitCode::from(1)
                    }
                }
            } else if listen.is_none() {
                eprintln!(
                    "error: carbon emu requires `--self-test` (one-shot E2E check) or `--listen <HOST:PORT>` (foreground emulator)"
                );
                ExitCode::from(2)
            } else {
                match run_foreground_emu(work_dir, listen, real_workers, edge_bin, worker_bin) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("error: carbon emu failed: {err}");
                        ExitCode::from(1)
                    }
                }
            }
        }
        #[cfg(feature = "emu")]
        Command::CiLive4Tier {
            source_sha,
            out_dir,
            require_live_pro,
            repo_root,
        } => {
            match jope_carbon::cli::spec_live_ci_4tier::execute_ci_live_4tier_docker(
                &jope_carbon::cli::spec_live_ci_4tier::LiveCi4TierOptions {
                    source_sha,
                    out_dir,
                    require_live_pro,
                    repo_root,
                },
            ) {
                Ok(receipt) => {
                    if let Ok(s) = serde_json::to_string_pretty(&serde_json::json!({
                        "status": "passed",
                        "all_passed": receipt.all_passed,
                        "source_sha": receipt.source_sha,
                        "runs": receipt.runs.len(),
                        "total_estimated_usd": receipt.total_estimated_usd,
                        "total_wall_secs": receipt.total_wall_secs,
                    })) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon ci-live-4tier failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Quickstart {
            emit_csv,
            csv_only,
            cloud,
            clouds_manifest,
            endpoint,
            local,
            effort,
            no_shred,
        } => run_quickstart_command(
            emit_csv,
            csv_only,
            cloud,
            clouds_manifest,
            endpoint,
            local,
            &effort,
            no_shred,
            global_quiet,
        ),
        #[cfg(feature = "emu")]
        Command::SpecReport {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
            live_ci_receipt,
            code_coverage_export,
            source_sha,
            repo_root,
        } => match execute_carbon_spec_report(&CarbonSpecReportOptions {
            evidence_dir,
            out_dir,
            run_ci_qualification,
            ci_qualification_in,
            live_ci_receipt_in: live_ci_receipt,
            code_coverage_in: code_coverage_export,
            source_sha,
            repo_root,
        }) {
            Ok(html_path) => {
                let _ = write_stdout_line(
                    &serde_json::json!({
                        "status": "ok",
                        "html_path": html_path,
                    })
                    .to_string(),
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon spec-report failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "emu")]
        Command::SpecVerify {
            dir,
            expected_source_sha,
        } => match execute_carbon_spec_verify(&CarbonSpecVerifyOptions {
            dir,
            expected_source_sha,
        }) {
            Ok(receipt) => {
                let _ = write_stdout_line(
                    &serde_json::json!({
                        "status": "verified",
                        "pdf_file": receipt.pdf_file,
                        "pdf_bytes": receipt.pdf_bytes,
                        "pdf_sha256": receipt.pdf_sha256,
                        "pdf_page_count": receipt.pdf_page_count,
                        "screenshots_verified": receipt.screenshots.len(),
                        "source_sha": receipt.source_sha,
                    })
                    .to_string(),
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: carbon spec-verify failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::CoverageAudit {
            clouds_manifest,
            signing_pub,
            json_out,
        } => match jope_carbon::execute_live_aws_product_coverage_audit(
            &clouds_manifest,
            &signing_pub,
            json_out.as_deref(),
        ) {
            Ok(report) => {
                let _ = write_stdout_line(
                    &serde_json::json!({
                        "status": if report.all_passed { "passed" } else { "failed" },
                        "all_passed": report.all_passed,
                        "total_checks_per_cloud": report.total_checks_per_cloud,
                        "total_redteam_vectors_per_cloud": report.total_redteam_vectors_per_cloud,
                    })
                    .to_string(),
                );
                if report.all_passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(err) => {
                eprintln!("error: carbon coverage-audit failed: {err}");
                ExitCode::from(1)
            }
        },
        #[cfg(feature = "engine")]
        Command::Engine { work_dir } => {
            let (Ok(runtime_api), is_aws_prod) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon engine`");
                return ExitCode::from(2);
            };
            run_engine_runtime(&runtime_api, work_dir, is_aws_prod)
        }
        #[cfg(feature = "eval")]
        Command::Eval { scratch_dir } => {
            let (Ok(runtime_api), _) = resolve_lambda_runtime_api() else {
                eprintln!("error: AWS_LAMBDA_RUNTIME_API or LAMBDA_RUNTIME_API is required for `carbon eval`");
                return ExitCode::from(2);
            };
            run_eval_runtime(&runtime_api, scratch_dir)
        }
        Command::TokenKeygen { out, kid } => {
            let mut signing_key_opt = None;
            for _ in 0..8 {
                let seed = match jope_carbon::generate_random_u_dek() {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: CSPRNG failure: {e}");
                        return ExitCode::from(1);
                    }
                };
                if let Ok(sk) = jope_carbon::envelope::AppSigningKey::from_scalar_bytes(&seed) {
                    signing_key_opt = Some(sk);
                    break;
                }
            }
            let Some(signing_key) = signing_key_opt else {
                eprintln!("error: failed to generate valid NIST P-256 signing key");
                return ExitCode::from(1);
            };
            let pem = match signing_key.to_pkcs8_pem() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: encode PKCS#8 PEM: {e}");
                    return ExitCode::from(1);
                }
            };
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        eprintln!("error: create {}: {e}", parent.display());
                        return ExitCode::from(1);
                    }
                }
            }
            #[cfg(unix)]
            let write_res = {
                use std::os::unix::fs::OpenOptionsExt;
                fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .mode(0o600)
                    .open(&out)
                    .and_then(|mut f| f.write_all(pem.as_bytes()).and_then(|()| f.sync_all()))
            };
            #[cfg(not(unix))]
            let write_res = fs::write(&out, pem.as_bytes());
            if let Err(e) = write_res {
                eprintln!("error: write private key {}: {e}", out.display());
                return ExitCode::from(1);
            }
            let spki_pem = signing_key.pubkey_spki_pem().unwrap_or_default();
            let pub_b64 = signing_key.pubkey_b64();
            let receipt = serde_json::json!({
                "algorithm": "ES256 (NIST P-256)",
                "token_format": "ctk2.<claims_b64url>.<es256_sig_b64url>",
                "private_key_path": out,
                "public_key_sec1_b64": pub_b64,
                "public_key_spki_pem": spki_pem.trim_end(),
                "kid": kid,
            });
            if let Ok(text) = serde_json::to_string_pretty(&receipt) {
                let _ = write_stdout_line(&text);
            }
            ExitCode::SUCCESS
        }
        _ => unreachable!("handled by auth_doctor or model_ops"),
    }
}

#[cfg(feature = "emu")]
fn start_cli_emulator_instance(
    work_dir: Option<PathBuf>,
    prefix: &str,
    listen: Option<SocketAddr>,
    real_workers: bool,
    edge_bin: Option<PathBuf>,
    worker_bin: Option<PathBuf>,
) -> Result<(EphemeralScratchDir, CarbonLocalLambdaEmulator), String> {
    let scratch = EphemeralScratchDir::existing_or_create(work_dir, prefix)?;
    let mut cfg = CarbonEmuConfig::for_dir(scratch.path.clone());
    if let Some(addr) = listen {
        cfg.listen_addr = addr;
    }
    if real_workers {
        cfg = cfg.with_real_worker_binaries(resolve_real_worker_binaries(edge_bin, worker_bin)?);
    }
    let emu = CarbonLocalLambdaEmulator::start(cfg)?;
    Ok((scratch, emu))
}

#[cfg(feature = "emu")]
pub(crate) fn run_emu_self_test_with_output(
    work_dir: Option<PathBuf>,
    listen: Option<SocketAddr>,
    real_workers: bool,
    edge_bin: Option<PathBuf>,
    worker_bin: Option<PathBuf>,
    emit_stdout: bool,
) -> Result<(), String> {
    let (_scratch, mut emu) = start_cli_emulator_instance(
        work_dir,
        ".emu-selftest",
        listen,
        real_workers,
        edge_bin,
        worker_bin,
    )?;
    let pro_tok = emu.mint_capability_token("emu_tenant", CarbonTierWire::Pro, false)?;
    let (raw_client, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()])?;
    let client = raw_client.with_capability_token(pro_tok);
    let mut rows = Vec::with_capacity(60);
    for i in 0..60 {
        let x1 = (i as f64) * 0.25 - 7.0;
        let x2 = ((i % 7) as f64) * 0.5;
        let y = 2.5 * x1 - 1.2 * x2 + 0.75;
        rows.push(vec![x1, x2, y]);
    }
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &rows,
        "y",
        42,
        0.20,
    )?;
    client.upload_prepared("emu_tenant", CarbonTierWire::Pro, &prepared)?;
    let run_id = "run_01JCCARB0NEM00000000000001"
        .parse()
        .map_err(|_| "invalid self-test RunId".to_owned())?;
    let ds_id = "ds_111111111111111111111111"
        .parse()
        .map_err(|_| "invalid self-test DatasetId".to_owned())?;
    let progress = client.start_run(
        run_id,
        ds_id,
        "emu_tenant",
        CarbonTierWire::Pro,
        Effort::Fast,
        "rmse",
        42,
        &prepared,
    )?;
    let pred = client.predict(
        "emu_tenant",
        "run_01JCCARB0NEM00000000000001",
        vec!["x1".to_owned(), "x2".to_owned()],
        vec![vec![Some(1.0), Some(2.0)]],
    )?;
    let shred = client.shred_run("emu_tenant", "run_01JCCARB0NEM00000000000001")?;
    let telemetry = emu.telemetry();
    emu.shutdown();
    if emit_stdout {
        let _ = write_stdout_line(
            &serde_json::json!({
                "self_test": "ok",
                "real_workers": real_workers,
                "probed_region": probed[0].region,
                "encrypted_cuen": progress.encrypted_cuen,
                "state": progress.state,
                "predictions_count": pred.row_count,
                "shredded": shred.shredded,
                "edge_invocations": telemetry.edge_invocations,
                "engine_invocations": telemetry.engine_invocations,
                "eval_invocations": telemetry.eval_invocations,
            })
            .to_string(),
        );
    }
    Ok(())
}

#[cfg(feature = "emu")]
fn run_foreground_emu(
    work_dir: Option<PathBuf>,
    listen: Option<SocketAddr>,
    real_workers: bool,
    edge_bin: Option<PathBuf>,
    worker_bin: Option<PathBuf>,
) -> Result<(), String> {
    let (_scratch, mut emu) = start_cli_emulator_instance(
        work_dir,
        ".emu-fg",
        listen,
        real_workers,
        edge_bin,
        worker_bin,
    )?;
    let dev_tok = emu.mint_capability_token("emu_dev_tenant", CarbonTierWire::Pro, false)?;
    let _ = write_stdout_line(
        &serde_json::to_string_pretty(&serde_json::json!({
            "status": "listening",
            "endpoint": emu.base_url(),
            "s3_cas_endpoint": emu.s3_cas_endpoint_url(),
            "secrets_manager_endpoint": emu.secrets_manager_endpoint_url(),
            "lambda_invoke_endpoint": emu.lambda_invoke_endpoint_url(),
            "engine_kek_pubkey_b64": emu.engine_kek_pubkey_b64(),
            "real_workers": real_workers,
            "pro_capability_token": dev_tok,
        }))
        .unwrap_or_default(),
    );
    let max_ms = env::var("CARBON_EMU_MAX_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let wait_res = jope_carbon::modes::emu::wait_for_emu_signal_or_timeout(max_ms);
    emu.shutdown();
    wait_res
}

#[allow(clippy::too_many_arguments)]
fn run_quickstart_command(
    emit_csv: Option<PathBuf>,
    csv_only: bool,
    cloud: Option<String>,
    clouds_manifest: Option<String>,
    mut endpoint: Vec<String>,
    local: bool,
    effort_raw: &str,
    no_shred: bool,
    quiet: bool,
) -> ExitCode {
    let effort_val = match parse_api_effort(effort_raw) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::from(2);
        }
    };

    if let Some(csv_out) = &emit_csv {
        let (_, rows, _) = generate_nonlinear_benchmark_rows(240, 6, 42);
        let mut buf = String::with_capacity(16 * 1024);
        buf.push_str("feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,target\n");
        for r in &rows {
            for (idx, val) in r.iter().enumerate() {
                if idx > 0 {
                    buf.push(',');
                }
                buf.push_str(&format!("{val:.6}"));
            }
            buf.push('\n');
        }
        if let Some(parent) = csv_out.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(err) = fs::write(csv_out, buf.as_bytes()) {
            eprintln!("error: write sample CSV {}: {err}", csv_out.display());
            return ExitCode::from(1);
        }
        if !quiet {
            eprintln!(
                "[quickstart] Wrote 240-row x 6-feature sample CSV -> {} (target column: 'target')",
                csv_out.display()
            );
        }
        if csv_only {
            let _ = write_stdout_line(
                &serde_json::json!({
                    "written_csv": csv_out,
                    "rows": 240,
                    "features": 6,
                    "target_column": "target",
                })
                .to_string(),
            );
            return ExitCode::SUCCESS;
        }
    }

    let mut _local_emu_guard: Option<LocalEmuGuard> = None;
    let mut local_tok: Option<String> = None;
    if local {
        match start_local_emu_session("quickstart_user") {
            Ok((guard, ep, tok)) => {
                endpoint = vec![ep];
                local_tok = Some(tok);
                _local_emu_guard = Some(guard);
            }
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(2);
            }
        }
    }

    let cloud_opt = if endpoint.is_empty() {
        match cloud.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()) {
            Some(explicit_cloud) => Some(explicit_cloud),
            None => {
                let has_env_tok = env::var("CARBON_AUTH_TOKEN")
                    .or_else(|_| env::var("CARBON_TOKEN"))
                    .map(|s| !s.trim().is_empty())
                    .unwrap_or(false);
                let has_prod_cred = jope_carbon::load_stored_cloud_token("prod")
                    .ok()
                    .flatten()
                    .is_some();
                if has_env_tok || has_prod_cred {
                    Some("prod".to_owned())
                } else {
                    Some("dev".to_owned())
                }
            }
        }
    } else {
        None
    };
    let resolved = match resolve_cli_target_and_auth(
        endpoint,
        cloud_opt,
        clouds_manifest.as_deref(),
        None,
        "us-east-2",
        None,
        None,
        "quickstart_user",
        local_tok,
        None,
        false,
        CarbonTierWire::Free,
    ) {
        Ok(r) => r,
        Err((code, msg)) => {
            eprintln!("error: {msg}");
            return code;
        }
    };

    if !quiet {
        eprintln!(
            "[quickstart] Carbon {} — Zero-Knowledge E2E Walkthrough",
            CARBON_PKG_VERSION,
        );
        eprintln!(
            "[quickstart] Step 1/4: Connected to fastest regional edge '{}' ({:.1} ms RTT, P-256 KEK verified)",
            resolved.first_probe.region, resolved.first_probe.rtt_ms
        );
        eprintln!(
            "[quickstart] Step 2/4: Conditioning & encrypting 240x6 tabular dataset client-side (CUEN v1 + CUDK v1)..."
        );
    }

    let opts = CarbonCliRunOptions {
        endpoints: resolved.endpoints,
        csv_path: emit_csv.clone(),
        target_column: if emit_csv.is_some() {
            "target".to_owned()
        } else {
            "y".to_owned()
        },
        synthetic_rows: 240,
        synthetic_features: 6,
        tenant_id: resolved.tenant_id,
        tier: resolved.tier,
        effort: effort_val,
        objective: "rmse".to_owned(),
        seed: 42,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 5,
        auth_token: resolved.auth_token,
        shred_after_run: !no_shred,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks: resolved.pinned_keks,
        pre_probed: Some((resolved.probed_client, resolved.first_probe)),
        target_was_explicit: emit_csv.is_some(),
        show_stderr_progress: !quiet,
        ..CarbonCliRunUxConfig::default()
    };

    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => {
            if !quiet {
                eprintln!(
                    "[quickstart] Step 3/4: Holdout R^2 = {:.4} | 5 encrypted /predict rows decrypted locally",
                    receipt.holdout_r2.unwrap_or(0.0)
                );
                eprintln!(
                    "[quickstart] Step 4/4: Remote S3/CAS blobs cryptographically shredded = {} | Next: `carbon register --free`",
                    receipt.shredded_after_run
                );
            }
            match serde_json::to_string_pretty(&receipt) {
                Ok(s) => {
                    let _ = write_stdout_line(&s);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: serialize receipt: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Err(err) => {
            eprintln!("error: carbon quickstart failed: {err}");
            ExitCode::from(1)
        }
    }
}
