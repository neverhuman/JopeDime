//! Zero-knowledge dataset conditioning, training, prediction, explanation, export, and run lifecycle commands.

use super::*;

fn apply_handle_target_overrides(
    handle: &mut jope_carbon::CarbonSavedModelHandle,
    target_opts: &HandleTargetArgs,
) -> Result<(), String> {
    let explicit_tier = resolve_explicit_tier_flag(
        target_opts.free,
        target_opts.pro,
        target_opts.tier.as_deref(),
    )?;
    if let Some(t) = explicit_tier {
        handle.tier = t;
    }
    if let Some(ref tid) = target_opts.tenant {
        let trimmed = tid.trim();
        if !trimmed.is_empty() {
            handle.tenant_id = trimmed.to_owned();
        }
    }
    if let Some(ref cloud_name) = target_opts.cloud {
        let trimmed = cloud_name.trim();
        if !trimmed.is_empty() {
            handle.cloud = Some(trimmed.to_owned());
        }
    }
    if let Some(ref ep) = target_opts.endpoint {
        let trimmed = ep.trim();
        if !trimmed.is_empty()
            && trimmed.trim_end_matches('/') != handle.endpoint.trim().trim_end_matches('/')
        {
            handle.endpoint = trimmed.to_owned();
            handle.engine_kek_pubkey_b64 = None;
        }
    } else if target_opts.clouds_manifest.is_some() {
        if let Some(ref cloud_name) = handle.cloud {
            let entry =
                resolve_public_cloud_endpoint(cloud_name, target_opts.clouds_manifest.as_deref())?;
            let pins = entry.pinned_keks_by_endpoint();
            let norm_ep = handle.endpoint.trim().trim_end_matches('/');
            if let Some(pin) = pins.get(norm_ep) {
                handle.engine_kek_pubkey_b64 = Some(pin.clone());
            } else if !entry.public_endpoint.trim().is_empty() {
                handle.endpoint = entry.public_endpoint.clone();
                handle.engine_kek_pubkey_b64 = entry.engine_kek_pubkey_b64;
            }
        }
    }
    Ok(())
}

fn execute_stateless_remote_shred(
    run_id_raw: &str,
    target_opts: &HandleTargetArgs,
    override_tok: Option<String>,
) -> Result<jope_api::carbon::CarbonShredRunResponse, String> {
    let run_id = run_id_raw.trim();
    jope_carbon::validate_strict_run_id(run_id).map_err(|e| e.to_string())?;
    let explicit_tier = resolve_explicit_tier_flag(
        target_opts.free,
        target_opts.pro,
        target_opts.tier.as_deref(),
    )?;
    let endpoints = target_opts
        .endpoint
        .as_ref()
        .map(|s| vec![s.clone()])
        .unwrap_or_default();
    let resolved = resolve_cli_target_and_auth(
        endpoints,
        target_opts.cloud.clone(),
        target_opts.clouds_manifest.as_deref(),
        None,
        "us-east-2",
        explicit_tier,
        target_opts.tenant.clone(),
        "",
        override_tok,
        None,
        false,
        CarbonTierWire::Pro,
    )
    .map_err(|(_, msg)| msg)?;
    if resolved.tenant_id.trim().is_empty() {
        return Err(
            "stateless `carbon shred --run <RUN_ID>` without a local handle requires --tenant <TENANT_ID> or a capability token (--token / `carbon login`)".to_owned(),
        );
    }
    let mut client = resolved.probed_client;
    if let Some(tok) = resolved.auth_token {
        client = client.with_capability_token(tok);
    }
    client.shred_run(&resolved.tenant_id, run_id)
}

pub(crate) fn is_model_ops_cmd(cmd: &Command) -> bool {
    matches!(
        cmd,
        Command::Ping { .. }
            | Command::Run { .. }
            | Command::Fit { .. }
            | Command::Predict { .. }
            | Command::Inspect { .. }
            | Command::Runs { .. }
            | Command::Status { .. }
            | Command::Shred { .. }
            | Command::Fitness { .. }
            | Command::Explain { .. }
            | Command::Export { .. }
            | Command::Resume { .. }
    )
}

pub(crate) fn dispatch_model_ops(
    cmd: Command,
    global_quiet: bool,
    global_ephemeral: bool,
) -> ExitCode {
    match cmd {
        Command::Ping {
            mut endpoint,
            mut cloud,
            clouds_manifest,
            environment,
            region,
        } => {
            if endpoint.is_empty() && cloud.is_none() && environment.is_none() {
                if let Ok(env_ep) = env::var("CARBON_ENDPOINT") {
                    let trimmed = env_ep.trim();
                    if !trimmed.is_empty() {
                        endpoint.push(trimmed.to_owned());
                    }
                }
                if endpoint.is_empty() {
                    cloud = Some("prod".to_owned());
                }
            }
            let mut pinned_keks: BTreeMap<String, String> = BTreeMap::new();
            if let Some(cloud_name) = &cloud {
                match resolve_public_cloud_endpoint(cloud_name, clouds_manifest.as_deref()) {
                    Ok(entry) => {
                        pinned_keks = entry.pinned_keks_by_endpoint();
                        if endpoint.is_empty() {
                            endpoint = entry.candidate_endpoints();
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            if let Some(env_name) = &environment {
                if endpoint.is_empty() {
                    match resolve_stack_endpoint_if_compiled(env_name, &region) {
                        Ok(url) => endpoint.push(url),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            if endpoint.is_empty() {
                eprintln!(
                    "error: pass --cloud <dev|prod>, --endpoint <URL>, or --env <dev|staging|prod>"
                );
                return ExitCode::from(2);
            }
            match CarbonThinClient::connect_fastest_pinned(endpoint, &pinned_keks) {
                Ok((_, probed)) => {
                    let Some(first) = probed.first() else {
                        eprintln!("error: empty probe response");
                        return ExitCode::from(1);
                    };
                    let probed_regions: Vec<serde_json::Value> = probed
                        .iter()
                        .map(|p| {
                            serde_json::json!({
                                "region": p.region,
                                "endpoint": p.base_url,
                                "rtt_ms": p.rtt_ms,
                            })
                        })
                        .collect();
                    let _ = write_stdout_line(
                        &serde_json::to_string_pretty(&serde_json::json!({
                            "healthy": true,
                            "cloud": cloud,
                            "endpoint": first.base_url,
                            "region": first.region,
                            "rtt_ms": first.rtt_ms,
                            "enterprise_self_hosted": first.enterprise_self_hosted,
                            "engine_kek_pubkey_b64": first.engine_kek_pubkey_b64,
                            "kek_pin_verified": !pinned_keks.is_empty(),
                            "healthy_regions": probed_regions,
                        }))
                        .unwrap_or_default(),
                    );
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: carbon ping failed: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Command::Run {
            csv,
            target,
            synthetic_rows,
            features,
            stop_after_ms,
            resume,
            predict_rows,
            save,
            format,
            shred,
            progress,
            common,
        } => {
            if global_quiet && progress {
                return emit_cli_error(common.json, "--quiet and --progress cannot be combined");
            }
            let effective_shred = shred || global_ephemeral;
            if effective_shred && save.is_some() {
                return emit_cli_error(
                    common.json,
                    "--shred / --ephemeral and --save cannot be combined (a shredded run cannot be queried by `carbon predict`)",
                );
            }
            let pretty_stdout = should_emit_pretty_stdout(&format, common.json);
            let persist_handle = !common.dry_run
                && !effective_shred
                && (save.is_some() || verified_absolute_jopedime_config_dir().is_ok());
            execute_train_command(
                "run",
                common,
                csv,
                target,
                synthetic_rows,
                features,
                stop_after_ms,
                resume,
                predict_rows,
                save,
                effective_shred,
                persist_handle,
                pretty_stdout,
                progress,
                global_quiet,
            )
        }
        Command::Fit {
            csv,
            csv_flag,
            target,
            save,
            format,
            common,
        } => {
            if global_ephemeral && save.is_some() {
                return emit_cli_error(
                    common.json,
                    "--ephemeral and --save cannot be combined (an ephemeral run is shredded upon completion)",
                );
            }
            let pretty_stdout = should_emit_pretty_stdout(&format, common.json);
            let persist_handle = !common.dry_run && !global_ephemeral;
            execute_train_command(
                "fit",
                common,
                csv.or(csv_flag),
                target,
                240,
                6,
                None,
                false,
                5,
                save,
                global_ephemeral,
                persist_handle,
                pretty_stdout,
                false,
                global_quiet,
            )
        }
        Command::Predict {
            csv,
            csv_flag,
            model,
            run,
            out,
            id_column,
            include_inputs,
            no_header,
            delimiter,
            token,
            token_file,
            token_stdin,
            format,
            json,
            target_opts,
        } => {
            let Some(csv_path) = csv.or(csv_flag) else {
                return emit_cli_error(json, "missing input CSV path");
            };
            let delim_override = match parse_delimiter_arg(delimiter.as_deref()) {
                Ok(d) => d,
                Err(e) => return emit_cli_error(json, &e),
            };
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref())
            {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                return emit_cli_error(json, &e);
            }
            match execute_predict_csv_with_handle(
                &handle,
                &csv_path,
                out.as_deref(),
                delim_override,
                no_header,
                id_column.as_deref(),
                include_inputs,
                override_tok,
            ) {
                Ok(receipt) => {
                    let streamed_stdout_csv = out.as_deref().is_some_and(|p| p.as_os_str() == "-");
                    if streamed_stdout_csv {
                        ExitCode::SUCCESS
                    } else if should_emit_pretty_stdout(&format, json) {
                        let mut card = String::with_capacity(512);
                        card.push_str("┌─ Carbon Zero-Knowledge Batch Prediction ───────────────────────────────┐\n");
                        card.push_str(&format!(
                            "  Run & Region : {} | {} (CUEN\\x01={})\n",
                            receipt.run_id, receipt.region, receipt.encrypted_cuen
                        ));
                        card.push_str(&format!(
                            "  Scored Rows  : {} rows across {} encrypted batch(es) in {:.1} ms\n",
                            receipt.rows_scored, receipt.batches, receipt.wall_ms
                        ));
                        if let Some(out_p) = &receipt.output_csv {
                            card.push_str(&format!(
                                "  Output CSV   : {out_p} (sha256={})\n",
                                &receipt.predictions_sha256[..16]
                            ));
                        }
                        if let Some(ev) = &receipt.evaluation_if_labeled {
                            let r2s = ev
                                .r2
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "n/a".to_owned());
                            card.push_str(&format!(
                                "  Labeled Eval : R^2 = {r2s} | RMSE = {:.4} | MAE = {:.4} (p90={:.4})\n",
                                ev.rmse, ev.mae, ev.abs_err_p90
                            ));
                        }
                        let preview: Vec<String> = receipt
                            .sample_predictions
                            .iter()
                            .take(5)
                            .map(|v| format!("{v:.4}"))
                            .collect();
                        card.push_str(&format!("  Sample Preds : [{}]\n", preview.join(", ")));
                        card.push_str("└────────────────────────────────────────────────────────────────────────┘");
                        let _ = write_stdout_line(&card);
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&receipt) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize predict receipt: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon predict failed: {err}")),
            }
        }
        Command::Inspect {
            csv,
            csv_flag,
            target,
            no_header,
            delimiter,
            tier,
            format,
            json,
        } => {
            let Some(csv_path) = csv.or(csv_flag) else {
                return emit_cli_error(json, "missing input CSV path");
            };
            let delim_override = match parse_delimiter_arg(delimiter.as_deref()) {
                Ok(d) => d,
                Err(e) => return emit_cli_error(json, &e),
            };
            let Ok(ptier) = parse_product_tier(&tier) else {
                return emit_cli_error(json, &format!("invalid --tier `{tier}`"));
            };
            match inspect_tabular_file_with_tier(
                &csv_path,
                target.as_deref(),
                delim_override,
                !no_header,
                ptier,
            ) {
                Ok(report) => {
                    if should_emit_pretty_stdout(&format, json) {
                        let _ = write_stdout_line(&render_pretty_inspect_report(&report));
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&report) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize inspect report: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon inspect failed: {err}")),
            }
        }
        Command::Runs { json } => match list_saved_model_handles() {
            Ok(entries) => {
                if !json && std::io::stdout().is_terminal() {
                    if entries.is_empty() {
                        let _ = write_stdout_line(
                            "No saved Carbon runs found. Train one with: carbon fit <CSV> --target <COL>",
                        );
                    } else {
                        let mut tbl = format!(
                            "{:<30}  {:<12}  {:<14}  {:>8}  {:>10}  HANDLE",
                            "RUN_ID", "REGION", "TARGET", "FIT_ROWS", "HOLDOUT_R2"
                        );
                        for e in &entries {
                            let r2s = e
                                .holdout_r2
                                .map(|v| format!("{v:.5}"))
                                .unwrap_or_else(|| "-".to_owned());
                            tbl.push_str(&format!(
                                "\n{:<30}  {:<12}  {:<14}  {:>8}  {:>10}  {}",
                                e.run_id, e.region, e.target_column, e.fit_rows, r2s, e.handle_path
                            ));
                        }
                        let _ = write_stdout_line(&tbl);
                    }
                } else if let Ok(s) = serde_json::to_string_pretty(&entries) {
                    let _ = write_stdout_line(&s);
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: list saved runs failed: {err}");
                ExitCode::from(1)
            }
        },
        Command::Status {
            model,
            run,
            token,
            token_file,
            token_stdin,
            json,
            target_opts,
        } => {
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref())
            {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                return emit_cli_error(json, &e);
            }
            match query_handle_status(&handle, override_tok) {
                Ok(status) => {
                    if let Ok(s) = serde_json::to_string_pretty(&status) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => emit_cli_error(json, &format!("carbon status failed: {err}")),
            }
        }
        Command::Shred {
            model,
            run,
            token,
            token_file,
            token_stdin,
            json,
            target_opts,
        } => {
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                Ok((mut handle, resolved_path)) => {
                    if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                        return emit_cli_error(json, &e);
                    }
                    let explicit_opt = model.as_deref().or(Some(resolved_path.as_path()));
                    match shred_handle_and_local_keys(&handle, explicit_opt, override_tok) {
                        Ok((remote, local_deleted)) => {
                            let _ = write_stdout_line(
                                &serde_json::to_string_pretty(&serde_json::json!({
                                    "run_id": remote.run_id,
                                    "remote_shredded": remote.shredded,
                                    "remote_objects_deleted": remote.objects_deleted,
                                    "local_key_handle_shredded": local_deleted,
                                }))
                                .unwrap_or_default(),
                            );
                            ExitCode::SUCCESS
                        }
                        Err(err) => emit_cli_error(json, &format!("carbon shred failed: {err}")),
                    }
                }
                Err(load_err) => {
                    if let (None, Some(ref rid)) = (&model, &run) {
                        if !rid.trim().eq_ignore_ascii_case("latest") {
                            return match execute_stateless_remote_shred(
                                rid,
                                &target_opts,
                                override_tok,
                            ) {
                                Ok(remote) => {
                                    let _ = write_stdout_line(
                                        &serde_json::to_string_pretty(&serde_json::json!({
                                            "run_id": remote.run_id,
                                            "remote_shredded": remote.shredded,
                                            "remote_objects_deleted": remote.objects_deleted,
                                            "local_key_handle_shredded": false,
                                        }))
                                        .unwrap_or_default(),
                                    );
                                    ExitCode::SUCCESS
                                }
                                Err(err) => {
                                    emit_cli_error(json, &format!("carbon shred failed: {err}"))
                                }
                            };
                        }
                    }
                    emit_cli_error(json, &load_err)
                }
            }
        }
        Command::Fitness { action } => execute_fitness_command(action),
        Command::Explain {
            model,
            run,
            top_k,
            token,
            token_file,
            token_stdin,
            format,
            json,
            target_opts,
        } => {
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref())
            {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                return emit_cli_error(json, &e);
            }
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.explain_run(&handle.tenant_id, &handle.run_id, top_k) {
                Ok(resp) => {
                    if should_emit_pretty_stdout(&format, json) {
                        let _ = write_stdout_line(&render_pretty_explain_response(&resp));
                        ExitCode::SUCCESS
                    } else {
                        match serde_json::to_string_pretty(&resp) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(json, &format!("serialize explain response: {e}"))
                            }
                        }
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon explain failed: {err}")),
            }
        }
        Command::Export {
            model,
            run,
            lang,
            out,
            token,
            token_file,
            token_stdin,
            json,
            target_opts,
        } => {
            let lang_wire = match parse_export_lang(&lang) {
                Ok(l) => l,
                Err(e) => return emit_cli_error(json, &e),
            };
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, _) = match load_model_handle_by_spec(model.as_deref(), run.as_deref())
            {
                Ok(pair) => pair,
                Err(e) => return emit_cli_error(json, &e),
            };
            if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                return emit_cli_error(json, &e);
            }
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.export_run(&handle.tenant_id, &handle.run_id, lang_wire) {
                Ok(resp) => {
                    let src_code = resp.source_code.clone().unwrap_or_default();
                    if let Some(out_path) = &out {
                        if out_path.as_os_str() == "-" {
                            let _ = write_stdout_line(src_code.trim_end_matches('\n'));
                            return ExitCode::SUCCESS;
                        }
                        if let Some(parent) = out_path.parent() {
                            if !parent.as_os_str().is_empty() {
                                let _ = fs::create_dir_all(parent);
                            }
                        }
                        let out_bytes: Vec<u8> = if lang_wire == CarbonExportLangWire::ModelObject {
                            use base64::Engine as _;
                            match base64::engine::general_purpose::STANDARD.decode(src_code.trim())
                            {
                                Ok(raw) => raw,
                                Err(e) => {
                                    return emit_cli_error(
                                        json,
                                        &format!("decode model-object ZIP base64: {e}"),
                                    );
                                }
                            }
                        } else {
                            src_code.as_bytes().to_vec()
                        };
                        if let Err(e) = fs::write(out_path, &out_bytes) {
                            return emit_cli_error(
                                json,
                                &format!("write export source {}: {e}", out_path.display()),
                            );
                        }
                        if json {
                            if let Ok(s) = serde_json::to_string_pretty(&resp) {
                                let _ = write_stdout_line(&s);
                            }
                        } else {
                            let _ = write_stdout_line(
                                &serde_json::to_string_pretty(&serde_json::json!({
                                    "run_id": resp.run_id,
                                    "lang": resp.lang,
                                    "output_path": out_path,
                                    "source_sha256": resp.source_sha256,
                                    "bytes": out_bytes.len(),
                                    "encrypted_cuen": resp.encrypted_export_cuen_b64.is_some()
                                        || resp.encrypted_export_cas_url.is_some(),
                                }))
                                .unwrap_or_default(),
                            );
                        }
                        ExitCode::SUCCESS
                    } else if json {
                        match serde_json::to_string_pretty(&resp) {
                            Ok(s) => {
                                let _ = write_stdout_line(&s);
                                ExitCode::SUCCESS
                            }
                            Err(e) => {
                                emit_cli_error(true, &format!("serialize export response: {e}"))
                            }
                        }
                    } else {
                        let _ = write_stdout_line(src_code.trim_end_matches('\n'));
                        ExitCode::SUCCESS
                    }
                }
                Err(err) => emit_cli_error(json, &format!("carbon export failed: {err}")),
            }
        }
        Command::Resume {
            model,
            run,
            effort,
            additional_waves,
            token,
            token_file,
            token_stdin,
            json,
            target_opts,
        } => {
            let effort_opt = match effort.as_deref() {
                Some(raw) => match parse_api_effort(raw) {
                    Ok(e) => Some(e),
                    Err(msg) => return emit_cli_error(json, &msg),
                },
                None => None,
            };
            let override_tok = match read_cli_token_source(token, token_file, token_stdin) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let (mut handle, resolved_path) =
                match load_model_handle_by_spec(model.as_deref(), run.as_deref()) {
                    Ok(pair) => pair,
                    Err(e) => return emit_cli_error(json, &e),
                };
            if let Err(e) = apply_handle_target_overrides(&mut handle, &target_opts) {
                return emit_cli_error(json, &e);
            }
            let client = match build_client_for_handle(&handle, override_tok) {
                Ok(c) => c,
                Err(e) => return emit_cli_error(json, &e),
            };
            match client.resume_run(
                &handle.tenant_id,
                &handle.run_id,
                effort_opt,
                additional_waves,
            ) {
                Ok(progress) => {
                    if let Some(loss) = progress.outer_holdout_loss {
                        handle.holdout_rmse = Some(loss);
                        let explicit_save = model.as_deref().or(Some(resolved_path.as_path()));
                        let has_embedded = handle.capability_token.is_some();
                        let _ =
                            persist_model_handle_with_options(&handle, explicit_save, has_embedded);
                    }
                    if let Ok(s) = serde_json::to_string_pretty(&progress) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => emit_cli_error(json, &format!("carbon resume failed: {err}")),
            }
        }
        _ => unreachable!("is_model_ops_cmd checked"),
    }
}

pub(crate) fn execute_positional_csv_shorthand(
    csv_path: PathBuf,
    keep_handle: bool,
    global_quiet: bool,
) -> ExitCode {
    let is_stdin_dash = csv_path.as_os_str() == "-";
    let lower_str = csv_path.to_string_lossy().to_ascii_lowercase();
    let looks_like_tabular_path = is_stdin_dash
        || lower_str.ends_with(".csv")
        || lower_str.ends_with(".tsv")
        || lower_str.ends_with(".jsonl")
        || lower_str.ends_with(".ndjson")
        || lower_str.ends_with(".gz")
        || lower_str.ends_with(".zst")
        || csv_path.to_string_lossy().contains('/')
        || csv_path.to_string_lossy().contains('\\');

    if !is_stdin_dash && !csv_path.is_file() {
        if looks_like_tabular_path {
            eprintln!("error: CSV file '{}' does not exist", csv_path.display());
        } else {
            eprintln!(
                "error: unrecognized subcommand '{}' (run `carbon --help` for commands, or pass an existing `.csv` file path)",
                csv_path.display()
            );
        }
        return ExitCode::from(2);
    }

    let clouds_override = env::var("CARBON_CLOUDS_MANIFEST")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    let resolved = match resolve_cli_target_and_auth(
        Vec::new(),
        None,
        clouds_override.as_deref(),
        None,
        "us-east-2",
        None,
        None,
        "cli_tenant",
        None,
        None,
        false,
        CarbonTierWire::Pro,
    ) {
        Ok(r) => r,
        Err((code, msg)) => {
            eprintln!("error: {msg}");
            return code;
        }
    };
    let show_stderr_progress = !global_quiet && std::io::stderr().is_terminal();
    let opts = CarbonCliRunOptions {
        endpoints: resolved.endpoints,
        csv_path: Some(csv_path),
        target_column: "y".to_owned(),
        synthetic_rows: 240,
        synthetic_features: 6,
        tenant_id: resolved.tenant_id,
        tier: resolved.tier,
        effort: Effort::Fast,
        objective: "rmse".to_owned(),
        seed: 42,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 5,
        auth_token: resolved.auth_token,
        shred_after_run: !keep_handle,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks: resolved.pinned_keks,
        pre_probed: Some((resolved.probed_client, resolved.first_probe)),
        target_was_explicit: false,
        show_stderr_progress,
        persist_handle: keep_handle,
        ..CarbonCliRunUxConfig::default()
    };
    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => match serde_json::to_string_pretty(&receipt) {
            Ok(s) => {
                let _ = write_stdout_line(&s);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: serialize receipt: {e}");
                ExitCode::from(1)
            }
        },
        Err(err) => {
            eprintln!("error: carbon CSV shorthand failed: {err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_train_command(
    verb_label: &str,
    common: CommonTrainArgs,
    csv: Option<PathBuf>,
    target: Option<String>,
    synthetic_rows: usize,
    features: usize,
    stop_after_ms: Option<u64>,
    resume: bool,
    predict_rows: usize,
    save: Option<PathBuf>,
    shred: bool,
    persist_handle: bool,
    pretty_stdout: bool,
    force_progress: bool,
    global_quiet: bool,
) -> ExitCode {
    let json = common.json;
    let json_mode = !pretty_stdout;
    let delim_override = match parse_delimiter_arg(common.delimiter.as_deref()) {
        Ok(d) => d,
        Err(e) => return emit_cli_error(json, &e),
    };
    let explicit_tier =
        match resolve_explicit_tier_flag(common.free, common.pro, common.tier.as_deref()) {
            Ok(t) => t,
            Err(e) => return emit_cli_error(json, &e),
        };
    let Ok(effort_val) = parse_api_effort(&common.effort) else {
        return emit_cli_error(json, &format!("invalid --effort `{}`", common.effort));
    };
    let cloud_clone = common.cloud.clone();
    let mut _local_emu_guard: Option<LocalEmuGuard> = None;
    let (endpoints, pinned_keks, pre_probed, tenant_id, eff_tier, auth_token) = if common.dry_run {
        (
            Vec::new(),
            BTreeMap::new(),
            None,
            common.tenant.unwrap_or_else(|| "dry_run_tenant".to_owned()),
            explicit_tier.unwrap_or(CarbonTierWire::Free),
            None,
        )
    } else {
        let (resolved_endpoints, inline_tok) = if common.local {
            let local_tenant = common.tenant.as_deref().unwrap_or("cli_tenant");
            match start_local_emu_session(local_tenant) {
                Ok((guard, ep, minted_tok)) => {
                    _local_emu_guard = Some(guard);
                    let tok = if common.token.is_some()
                        || common.token_file.is_some()
                        || common.token_stdin
                    {
                        common.token
                    } else if matches!(explicit_tier, Some(CarbonTierWire::Free)) {
                        None
                    } else {
                        Some(minted_tok)
                    };
                    (vec![ep], tok)
                }
                Err(msg) => return emit_cli_error(json, &msg),
            }
        } else {
            (common.endpoint, common.token)
        };
        match resolve_cli_target_and_auth(
            resolved_endpoints,
            common.cloud,
            common.clouds_manifest.as_deref(),
            common.environment.as_deref(),
            &common.region,
            explicit_tier,
            common.tenant,
            "cli_tenant",
            inline_tok,
            common.token_file,
            common.token_stdin,
            CarbonTierWire::Pro,
        ) {
            Ok(r) => (
                r.endpoints,
                r.pinned_keks,
                Some((r.probed_client, r.first_probe)),
                r.tenant_id,
                r.tier,
                r.auth_token,
            ),
            Err((code, msg)) => {
                if json {
                    return emit_cli_error(true, &msg);
                }
                eprintln!("error: {msg}");
                return code;
            }
        }
    };
    let target_was_explicit = target.is_some();
    let target_column = target.unwrap_or_else(|| {
        if common.no_header {
            "-1".to_owned()
        } else {
            "y".to_owned()
        }
    });
    let show_stderr_progress =
        !global_quiet && !pretty_stdout && (force_progress || std::io::stderr().is_terminal());
    let opts = CarbonCliRunOptions {
        endpoints,
        csv_path: csv,
        target_column,
        synthetic_rows,
        synthetic_features: features,
        tenant_id,
        tier: eff_tier,
        effort: effort_val,
        objective: common.objective,
        seed: common.seed,
        holdout_fraction: 0.20,
        stop_after_ms,
        resume_after_stop: resume,
        predict_rows,
        auth_token,
        shred_after_run: shred,
    };
    let ux = CarbonCliRunUxConfig {
        pinned_keks,
        pre_probed,
        target_was_explicit,
        show_stderr_progress,
        delimiter_override: delim_override,
        no_header: common.no_header,
        persist_handle,
        save_model_path: if common.dry_run { None } else { save },
        cloud_name: cloud_clone,
        split_mode: common.split,
        calibration_mode: common.calibration,
        extra_na_values: common.na_values,
        inf_as_nan: common.inf_as_nan,
        embed_token: common.embed_token,
        dry_run: common.dry_run,
        allow_unencrypted_enterprise: common.allow_unencrypted_enterprise,
    };
    match execute_carbon_cli_run_with_ux(&opts, &ux) {
        Ok(receipt) => {
            if pretty_stdout {
                let _ = write_stdout_line(&render_pretty_run_scorecard(&receipt));
                ExitCode::SUCCESS
            } else {
                match serde_json::to_string_pretty(&receipt) {
                    Ok(s) => {
                        let _ = write_stdout_line(&s);
                        ExitCode::SUCCESS
                    }
                    Err(e) => emit_cli_error(json_mode, &format!("serialize receipt: {e}")),
                }
            }
        }
        Err(err) => emit_cli_error(json, &format!("carbon {verb_label} failed: {err}")),
    }
}

fn parse_delimiter_arg(raw: Option<&str>) -> Result<Option<char>, String> {
    let Some(s) = raw else {
        return Ok(None);
    };
    match s.trim() {
        "" | "auto" => Ok(None),
        "," | "comma" | "csv" => Ok(Some(',')),
        "\t" | "\\t" | "tab" | "tsv" => Ok(Some('\t')),
        ";" | "semicolon" | "semi" => Ok(Some(';')),
        "|" | "pipe" => Ok(Some('|')),
        other if other.chars().count() == 1 => Ok(other.chars().next()),
        other => Err(format!(
            "unsupported --delimiter `{other}` (expected `,`, `;`, `tab`, `|`, or `auto`)"
        )),
    }
}

fn should_emit_pretty_stdout(format_arg: &str, force_json: bool) -> bool {
    if force_json {
        return false;
    }
    match format_arg.trim().to_ascii_lowercase().as_str() {
        "pretty" | "human" | "table" => true,
        "json" => false,
        _ => std::io::stdout().is_terminal(),
    }
}

fn render_pretty_inspect_report(report: &CarbonTabularInspectReport) -> String {
    let mut out = String::with_capacity(1_536);
    let delim_name = match report.delimiter.as_str() {
        "\t" => "TAB (TSV)",
        "," => "',' (CSV)",
        ";" => "';' (Semicolon)",
        other => other,
    };
    let free_eligible = report.valid_target_rows <= 5_000 && report.encoded_feature_count <= 64;
    out.push_str("┌─ Carbon Local Dataset Preflight (`carbon inspect` — zero network) ────────┐\n");
    out.push_str(&format!("  File          : {}\n", report.file_path));
    out.push_str(&format!(
        "  Shape         : {} rows x {} raw columns ({} encoded features) | Delimiter = {}\n",
        report.raw_rows, report.raw_columns, report.encoded_feature_count, delim_name
    ));
    out.push_str(&format!(
        "  Target Column : '{}' (valid={} rows, min={:.4}, median={:.4}, max={:.4}, 3σ-outliers={})\n",
        report.target_column,
        report.valid_target_rows,
        report.target_min,
        report.target_median,
        report.target_max,
        report.target_outliers_3sigma
    ));
    out.push_str(&format!(
        "  Recommended   : --effort {} (Free eligible = {})\n\n",
        report.recommended_effort, free_eligible
    ));
    out.push_str(&format!(
        "  {:<22}  {:<18}  {:>9}  {:>8}  {}\n",
        "COLUMN", "DETECTED_KIND", "MISSING%", "UNIQUE", "ROLE"
    ));
    for c in report.columns.iter().take(24) {
        out.push_str(&format!(
            "  {:<22}  {:<18}  {:>8.1}%  {:>8}  {}\n",
            c.name, c.detected_type, c.missing_pct, c.distinct_count, c.role
        ));
    }
    if report.columns.len() > 24 {
        out.push_str(&format!(
            "  ... ({} additional columns; pass `--json` for full schema)\n",
            report.columns.len() - 24
        ));
    }
    if !report.effort_previews.is_empty() {
        out.push_str("\n  Estimated Serverless Execution & Cost Preview:\n");
        for p in &report.effort_previews {
            let mode_label = format!("{:?}", p.execution_mode);
            out.push_str(&format!(
                "    [{:<4}] --effort {:<6} : {:<16} | {:>2} wave(s), {:>2} shard(s) | ~{:>5.1}s | ~${:.6} USD\n",
                p.tier,
                p.effort,
                mode_label,
                p.planned_waves,
                p.eval_shards,
                p.estimated_wall_secs,
                p.estimated_usd
            ));
        }
    }
    if !report.warnings.is_empty() {
        out.push_str("\n  Notes:\n");
        for w in &report.warnings {
            out.push_str(&format!("    - {w}\n"));
        }
    }
    out.push_str(&format!(
        "\n  Ready: carbon fit \"{}\" --target \"{}\" --save model.carbon.json\n",
        report.file_path, report.target_column
    ));
    out.push_str("└───────────────────────────────────────────────────────────────────────────┘");
    out
}

fn parse_export_lang(raw: &str) -> Result<CarbonExportLangWire, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "python" | "py" => Ok(CarbonExportLangWire::Python),
        "rust" | "rs" => Ok(CarbonExportLangWire::Rust),
        "rust-refittable" | "rust_refittable" => Ok(CarbonExportLangWire::RustRefittable),
        "js" | "javascript" | "ts" => Ok(CarbonExportLangWire::Js),
        "c" => Ok(CarbonExportLangWire::C),
        "cpp" | "c++" | "cxx" => Ok(CarbonExportLangWire::Cpp),
        "model-object" | "model_object" => Ok(CarbonExportLangWire::ModelObject),
        other => Err(format!(
            "unsupported --lang `{other}` (expected `python`, `rust`, `rust-refittable`, `js`, `c`, `cpp`, or `model-object`)"
        )),
    }
}

fn render_pretty_explain_response(resp: &CarbonExplainResponse) -> String {
    let mut out = String::with_capacity(1_024);
    let cuen = resp.encrypted_explain_cuen_b64.is_some();
    out.push_str("┌─ Carbon Zero-Knowledge Champion Explanation (`carbon explain`) ───────────┐\n");
    out.push_str(&format!(
        "  Run & Method  : {} | {} (CUEN\\x01={cuen})\n",
        resp.run_id, resp.method
    ));
    out.push_str(&format!(
        "  Baseline Loss : {:.6}\n\n",
        resp.baseline_holdout_loss
    ));
    out.push_str(&format!(
        "  {:<26}  {:>12}  {:>14}\n",
        "FEATURE", "IMPORTANCE%", "DELTA_LOSS"
    ));
    for f in &resp.feature_importances {
        out.push_str(&format!(
            "  {:<26}  {:>11.2}%  {:>+14.6}\n",
            f.feature,
            f.importance * 100.0,
            f.delta_loss
        ));
    }
    out.push_str("└───────────────────────────────────────────────────────────────────────────┘");
    out
}

fn execute_fitness_command(action: FitnessCommand) -> ExitCode {
    match action {
        FitnessCommand::List { json } => {
            let catalog = ObjectiveSpec::catalog();
            if !json && std::io::stdout().is_terminal() {
                let mut out = String::with_capacity(3_072);
                out.push_str(&format!(
                    "{:<24}  {:<9}  {:<6}  {}\n",
                    "OBJECTIVE", "DIRECTION", "ORDER?", "DESCRIPTION"
                ));
                for entry in &catalog {
                    out.push_str(&format!(
                        "{:<24}  {:<9}  {:<6}  {}\n",
                        entry.name,
                        entry.direction.as_str(),
                        if entry.requires_order { "yes" } else { "no" },
                        entry.description
                    ));
                }
                let _ = write_stdout_line(out.trim_end_matches('\n'));
            } else if let Ok(s) = serde_json::to_string_pretty(&catalog) {
                let _ = write_stdout_line(&s);
            }
            ExitCode::SUCCESS
        }
        FitnessCommand::Validate { spec, json } => match resolve_objective_argument(&spec) {
            Ok((wire, parsed)) => {
                let aux: Vec<String> = parsed.columns().into_iter().collect();
                let res: Vec<String> = parsed.resources().into_iter().collect();
                let payload = serde_json::json!({
                    "valid": true,
                    "name": parsed.name(),
                    "direction": parsed.direction(),
                    "requires_order": parsed.requires_order(),
                    "uses_execution_telemetry": parsed.uses_execution_telemetry(),
                    "benefits_from_post_hoc_calibration": parsed.benefits_from_post_hoc_calibration(),
                    "aux_columns": aux,
                    "resources": res,
                    "fitness_sha256": parsed.identity_sha256(),
                    "wire_objective": wire,
                    "canonical_spec": parsed.as_value(),
                });
                if let Ok(s) = serde_json::to_string_pretty(&payload) {
                    let _ = write_stdout_line(&s);
                }
                ExitCode::SUCCESS
            }
            Err(err) => emit_cli_error(json, &err),
        },
        FitnessCommand::Schema { json: _ } => {
            if let Ok(s) = serde_json::to_string_pretty(&ObjectiveSpec::schema_json()) {
                let _ = write_stdout_line(&s);
            }
            ExitCode::SUCCESS
        }
        FitnessCommand::Evaluate {
            spec,
            csv,
            target,
            pred,
            json,
        } => {
            let (_, parsed_spec) = match resolve_objective_argument(&spec) {
                Ok(v) => v,
                Err(e) => return emit_cli_error(json, &e),
            };
            let table = match load_delimited_table(&csv, None, true) {
                Ok(t) => t,
                Err(e) => return emit_cli_error(json, &e),
            };
            let find_col = |name: &str| -> Result<usize, String> {
                table.headers.iter().position(|h| h == name).ok_or_else(|| {
                    format!(
                        "column '{name}' not found in {} (headers: [{}])",
                        csv.display(),
                        table.headers.join(", ")
                    )
                })
            };
            let t_idx = match find_col(&target) {
                Ok(i) => i,
                Err(e) => return emit_cli_error(json, &e),
            };
            let p_idx = match find_col(&pred) {
                Ok(i) => i,
                Err(e) => return emit_cli_error(json, &e),
            };
            let mut aux_indices = Vec::new();
            for aux_name in parsed_spec.columns() {
                match find_col(&aux_name) {
                    Ok(i) => aux_indices.push((aux_name, i)),
                    Err(e) => return emit_cli_error(json, &e),
                }
            }
            let mut y_vec = Vec::with_capacity(table.rows.len());
            let mut p_vec = Vec::with_capacity(table.rows.len());
            let mut aux_map: BTreeMap<String, Vec<f64>> = aux_indices
                .iter()
                .map(|(n, _)| (n.clone(), Vec::with_capacity(table.rows.len())))
                .collect();
            for (r_idx, row) in table.rows.iter().enumerate() {
                let parse_cell = |col_i: usize, label: &str| -> Result<f64, String> {
                    let raw = row.get(col_i).and_then(|c| c.as_deref()).unwrap_or("");
                    parse_flexible_numeric_f64(raw)
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| {
                            format!(
                                "row {} column '{label}' ('{raw}') is not a finite number",
                                r_idx + 2
                            )
                        })
                };
                let yv = match parse_cell(t_idx, &target) {
                    Ok(v) => v,
                    Err(e) => return emit_cli_error(json, &e),
                };
                let pv = match parse_cell(p_idx, &pred) {
                    Ok(v) => v,
                    Err(e) => return emit_cli_error(json, &e),
                };
                for (aux_name, aux_i) in &aux_indices {
                    let av = match parse_cell(*aux_i, aux_name) {
                        Ok(v) => v,
                        Err(e) => return emit_cli_error(json, &e),
                    };
                    if let Some(slot) = aux_map.get_mut(aux_name) {
                        slot.push(av);
                    }
                }
                y_vec.push(yv);
                p_vec.push(pv);
            }
            match parsed_spec.evaluate(&y_vec, &p_vec, &aux_map) {
                Ok(score) => {
                    let payload = serde_json::json!({
                        "rows_scored": y_vec.len(),
                        "objective": parsed_spec.name(),
                        "value": score.value,
                        "loss": score.loss,
                        "direction": score.direction,
                        "fitness_sha256": score.fitness_sha256,
                    });
                    if let Ok(s) = serde_json::to_string_pretty(&payload) {
                        let _ = write_stdout_line(&s);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => emit_cli_error(json, &e),
            }
        }
    }
}
