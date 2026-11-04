use std::path::PathBuf;

use super::*;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/flywheel_selfplay/h256_pre_condense_residual_flywheel/weights")
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn test_h256_safetensors_all_artifacts_load_and_verify_schema() {
    let dir = weights_dir();
    let files = [
        "h256_production_full_1310_regression_bundle.safetensors",
        "h256_production_1219_dl_train_bundle.safetensors",
        "stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors",
        "stage1_full_s3_reg_all15796_encoder.safetensors",
        "full_s3_reg_7split_pretrained_encoder.safetensors",
        "sos_hurdle_and_sp_critic_models.safetensors",
        "golden_inference_fixture_context0.safetensors",
    ];
    for fname in files {
        let p = dir.join(fname);
        let bytes =
            std::fs::read(&p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()));
        assert!(
            bytes.len() <= 5 * 1024 * 1024,
            "{fname} exceeds 5 MiB limit: {}",
            bytes.len()
        );
        let st = SafetensorsMap::deserialize(&bytes)
            .unwrap_or_else(|e| panic!("failed to deserialize {fname}: {e}"));
        assert!(!st.st.names().is_empty(), "{fname} has no tensors");
    }

    // Verify standalone Stage-1 encoders and Critic models deserialize cleanly
    let s1_all_bytes =
        std::fs::read(dir.join("stage1_full_s3_reg_all15796_encoder.safetensors")).unwrap();
    let s1_all_st = SafetensorsMap::deserialize(&s1_all_bytes).unwrap();
    let _s1_all_enc = H256Stage1ContextEncoder::from_safetensors(&s1_all_st, "encoder").unwrap();
    let _s1_all_sc = H256FoldScaler::from_safetensors(&s1_all_st, "scaler_all15796").unwrap();

    let s1_pol_bytes =
        std::fs::read(dir.join("full_s3_reg_7split_pretrained_encoder.safetensors")).unwrap();
    let s1_pol_st = SafetensorsMap::deserialize(&s1_pol_bytes).unwrap();
    let _s1_pol_enc = H256Stage1ContextEncoder::from_safetensors(&s1_pol_st, "encoder").unwrap();
    let _s1_pol_sc = H256FoldScaler::from_safetensors(&s1_pol_st, "scaler_policy_b").unwrap();

    let crit_bytes =
        std::fs::read(dir.join("sos_hurdle_and_sp_critic_models.safetensors")).unwrap();
    let crit_st = SafetensorsMap::deserialize(&crit_bytes).unwrap();
    let _crit = H256CriticModels::from_safetensors(&crit_st, "critic").unwrap();
    let w1_proj = crit_st.f32_vec("w1_proj_412x16", &[412, 16]).unwrap();
    assert_eq!(w1_proj.len(), 412 * 16);
}

#[test]
fn test_h256_stage2_soup_and_5fold_ensemble_rust_golden_parity() {
    let dir = weights_dir();
    let s2_bytes =
        std::fs::read(dir.join("stage2_full_s3_reg_flywheel_pretrained_bridge.safetensors"))
            .unwrap();
    let s2_bundle = H256Stage2EnsembleBundle::from_safetensors_bytes(&s2_bytes).unwrap();

    let golden_bytes =
        std::fs::read(dir.join("golden_inference_fixture_context0.safetensors")).unwrap();
    let g = SafetensorsMap::deserialize(&golden_bytes).unwrap();

    // 1. Verify H256FoldScaler::transform_context against golden standardized_context_tensors
    let raw_dope629 = g.f32_vec("raw_ctx.dope629", &[629]).unwrap();
    let raw_hidden256 = g.f32_vec("raw_ctx.hidden256", &[256]).unwrap();
    let raw_col_lm = g.f32_vec("raw_ctx.col_lm_64x17", &[64, 17]).unwrap();
    let raw_col_mask = g.f32_vec("raw_ctx.col_mask_64", &[64]).unwrap();
    let raw_lm48_z = g.f32_vec("raw_ctx.lm48_z", &[48]).unwrap();

    let std_ctx = s2_bundle
        .full_train_scaler
        .transform_context(&raw_dope629, &raw_hidden256, &raw_col_lm, &raw_col_mask)
        .unwrap();

    let exp_dope629_z = g.f32_vec("std_ctx.dope629_z", &[1, 629]).unwrap();
    let exp_norms_z = g.f32_vec("std_ctx.block_log_norms", &[1, 11]).unwrap();
    let exp_hidden256_z = g.f32_vec("std_ctx.hidden256_z", &[1, 256]).unwrap();
    let exp_col18_z = g.f32_vec("std_ctx.col18_z", &[1, 64, 18]).unwrap();
    let exp_col_mask = g.f32_vec("std_ctx.col_mask_64", &[1, 64]).unwrap();

    assert!(max_abs_diff_f32(&std_ctx.dope629_z, &exp_dope629_z) <= 1e-6);
    assert!(max_abs_diff_f32(&std_ctx.block_log_norms, &exp_norms_z) <= 1e-6);
    assert!(max_abs_diff_f32(&std_ctx.hidden256_z, &exp_hidden256_z) <= 1e-6);
    assert!(max_abs_diff_f32(&std_ctx.col18_z, &exp_col18_z) <= 1e-6);
    assert!(max_abs_diff_f32(&std_ctx.col_mask_64, &exp_col_mask) <= 1e-6);

    // 2. Verify H256CriticModels::build_candidate_features_64 and predict_priors_and_h_cost
    let n_cand = 1152usize;
    let program_412 = g.f32_vec("pool.program_412", &[n_cand, 412]).unwrap();
    let sem24 = g.f32_vec("pool.sem24", &[n_cand, 24]).unwrap();
    let cand_q_32 = g.f32_vec("pool.cand_q_32", &[n_cand, 32]).unwrap();
    let rcost = g.f64_vec("pool.r882_total_cost", &[n_cand]).unwrap();
    let aprior = g.f64_vec("pool.r882_anchor_prior", &[n_cand]).unwrap();
    let atotal = g.f64_vec("pool.r882_anchor_total", &[n_cand]).unwrap();
    let idx4_i64 = g.i64_vec("pool.initial_probe_idx4", &[4]).unwrap();
    let idx4: Vec<usize> = idx4_i64.iter().map(|&v| v as usize).collect();
    let fitness = g.f64_vec("pool.candidate_fitness", &[n_cand]).unwrap();
    let logratio = g.f64_vec("pool.candidate_logratio", &[n_cand]).unwrap();
    let gate = g.f32_vec("pool.candidate_gate", &[n_cand]).unwrap();

    let sem24_struct =
        H256CriticModels::compute_sem24_from_program_412(&program_412, n_cand).unwrap();
    let cand_feats_64 = H256CriticModels::build_candidate_features_64(
        &program_412,
        &sem24_struct,
        &rcost,
        &aprior,
        &atotal,
        &raw_lm48_z,
        n_cand,
    )
    .unwrap();
    let exp_cand_feats_64 = g
        .f32_vec("expected_critic.cand_feats_64", &[n_cand, 64])
        .unwrap();
    // Column 23 of sem24 has within-pool std ~ 1.19e-3, so a 1-ULP (1.19e-7) f32 SIMD reduction
    // difference in L2 normalization scales by 1 / 1.19e-3 ≈ 1.02e-4 on that single column;
    // all other 63 columns match within 1e-6.
    assert!(
        max_abs_diff_f32(&cand_feats_64, &exp_cand_feats_64) <= 1.1e-4,
        "cand_feats_64 max diff: {}",
        max_abs_diff_f32(&cand_feats_64, &exp_cand_feats_64)
    );

    // Verify exact critic linear algebra against expected_critic.cand_feats_64 (<= 1e-6)
    let (sos_exact, sp_exact, h_cost_exact) = s2_bundle
        .critic
        .predict_priors_and_h_cost(&exp_cand_feats_64, &sem24_struct, &rcost, n_cand)
        .unwrap();
    let exp_sos_prior_2d = g
        .f32_vec("expected_critic.sos_prior_2d", &[n_cand, 2])
        .unwrap();
    let exp_sp_prior_1d = g.f32_vec("expected_critic.sp_prior_1d", &[n_cand]).unwrap();
    let exp_h_cost = g.f64_vec("expected_critic.h_cost", &[n_cand]).unwrap();

    assert!(
        max_abs_diff_f32(&sos_exact, &exp_sos_prior_2d) <= 1e-5,
        "sos_exact diff: {}",
        max_abs_diff_f32(&sos_exact, &exp_sos_prior_2d)
    );
    assert!(
        max_abs_diff_f32(&sp_exact, &exp_sp_prior_1d) <= 1e-5,
        "sp_exact diff: {}",
        max_abs_diff_f32(&sp_exact, &exp_sp_prior_1d)
    );
    assert!(
        max_abs_diff_f64(&h_cost_exact, &exp_h_cost) <= 1e-5,
        "h_cost_exact diff: {}",
        max_abs_diff_f64(&h_cost_exact, &exp_h_cost)
    );

    // Also verify end-to-end from Rust-computed cand_feats_64
    let (sos_prior_2d, sp_prior_1d, h_cost) = s2_bundle
        .critic
        .predict_priors_and_h_cost(&cand_feats_64, &sem24_struct, &rcost, n_cand)
        .unwrap();
    assert!(max_abs_diff_f32(&sos_prior_2d, &exp_sos_prior_2d) <= 3e-5);
    assert!(max_abs_diff_f32(&sp_prior_1d, &exp_sp_prior_1d) <= 1e-5);
    assert!(max_abs_diff_f64(&h_cost, &exp_h_cost) <= 3e-5);

    // 3. Verify Single-Soup PreCondenseResidualBridge Stage-2 forward pass
    let gate_risk: Vec<f32> = (0..n_cand).map(|i| sos_exact[i * 2 + 1]).collect();
    let (ctx_tokens, ctx_mask, scale_8) = s2_bundle.soup_bridge.encode_context_tokens(&std_ctx);
    let exp_ctx_tokens = g
        .f32_vec("expected_soup.ctx_tokens_92x32", &[1, 92, 32])
        .unwrap();
    let exp_ctx_mask = g.f32_vec("expected_soup.ctx_mask_92", &[1, 92]).unwrap();
    let exp_scale_8 = g.f32_vec("expected_soup.scale_feat_8", &[1, 8]).unwrap();

    assert!(max_abs_diff_f32(&ctx_tokens, &exp_ctx_tokens) <= 5e-6);
    assert!(max_abs_diff_f32(&ctx_mask, &exp_ctx_mask) <= 1e-6);
    assert!(max_abs_diff_f32(&scale_8, &exp_scale_8) <= 1e-6);

    let prior_out = s2_bundle
        .soup_bridge
        .forward_prior_and_context(&std_ctx, &cand_q_32, &sem24, &gate_risk, &sp_exact, n_cand)
        .unwrap();
    let exp_attn_ctx = g
        .f32_vec("expected_soup.attn_ctx_32", &[1, n_cand, 32])
        .unwrap();
    let exp_bilinear = g
        .f32_vec("expected_soup.bilinear_24", &[1, n_cand, 24])
        .unwrap();
    let exp_dp_soup = g
        .f64_vec("expected_soup.delta_prior_vec", &[n_cand])
        .unwrap();

    assert!(
        max_abs_diff_f32(&prior_out.attn_ctx_32, &exp_attn_ctx) <= 5e-6,
        "attn_ctx_32 diff: {}",
        max_abs_diff_f32(&prior_out.attn_ctx_32, &exp_attn_ctx)
    );
    assert!(
        max_abs_diff_f32(&prior_out.bilinear_24, &exp_bilinear) <= 5e-6,
        "bilinear_24 diff: {}",
        max_abs_diff_f32(&prior_out.bilinear_24, &exp_bilinear)
    );
    assert!(
        max_abs_diff_f64(&prior_out.delta_prior, &exp_dp_soup) <= 5e-6,
        "delta_prior_vec diff: {}",
        max_abs_diff_f64(&prior_out.delta_prior, &exp_dp_soup)
    );

    let (rev4_soup, cost_act_soup, y4_comb_soup, y4_f_soup) =
        H256CriticModels::run_2plus2_active_reveal_step(
            &rcost,
            &h_cost_exact,
            &sem24_struct,
            &fitness,
            &logratio,
            &gate,
            &idx4,
            Some(&prior_out.delta_prior),
            s2_bundle.alpha_prior,
            n_cand,
        )
        .unwrap();
    let exp_rev4_soup: Vec<usize> = g
        .i64_vec("expected_soup.rev4_indices", &[4])
        .unwrap()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let exp_cost_act_soup = g.f64_vec("expected_soup.cost_act", &[n_cand]).unwrap();
    assert_eq!(rev4_soup, exp_rev4_soup);
    assert!(max_abs_diff_f64(&cost_act_soup, &exp_cost_act_soup) <= 1e-6);

    let rev_gate_soup: Vec<f32> = rev4_soup.iter().map(|&idx| gate[idx]).collect();
    let base_cost_soup: Vec<f32> = (0..n_cand)
        .map(|i| (cost_act_soup[i] + s2_bundle.alpha_prior * prior_out.delta_prior[i]) as f32)
        .collect();
    let dtraj_soup = s2_bundle
        .soup_bridge
        .forward_causal_trajectory_residual(
            &prior_out.attn_ctx_32,
            &prior_out.bilinear_24,
            &cand_q_32,
            &sem24,
            &base_cost_soup,
            &rev4_soup,
            &y4_comb_soup,
            &y4_f_soup,
            &rev_gate_soup,
            n_cand,
        )
        .unwrap();
    let exp_dtraj_soup = g
        .f64_vec("expected_soup.delta_traj_vec", &[n_cand])
        .unwrap();
    assert!(
        max_abs_diff_f64(&dtraj_soup, &exp_dtraj_soup) <= 5e-6,
        "dtraj_soup diff: {}",
        max_abs_diff_f64(&dtraj_soup, &exp_dtraj_soup)
    );

    let cost_final_soup: Vec<f64> = (0..n_cand)
        .map(|i| {
            cost_act_soup[i]
                + s2_bundle.alpha_prior * prior_out.delta_prior[i]
                + s2_bundle.alpha_traj * dtraj_soup[i]
                + 0.005 * h_cost_exact[i]
        })
        .collect();
    let exp_cost_final_soup = g.f64_vec("expected_soup.cost_final", &[n_cand]).unwrap();
    assert!(max_abs_diff_f64(&cost_final_soup, &exp_cost_final_soup) <= 5e-6);

    let mut order_soup: Vec<usize> = (0..n_cand).collect();
    order_soup.sort_by(|&a, &b| {
        cost_final_soup[a]
            .total_cmp(&cost_final_soup[b])
            .then_with(|| a.cmp(&b))
    });
    let exp_pick1_soup = g.i64_vec("expected_soup.pick1_index", &[1]).unwrap()[0] as usize;
    let exp_top4_soup: Vec<usize> = g
        .i64_vec("expected_soup.top4_indices", &[4])
        .unwrap()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    assert_eq!(order_soup[0], exp_pick1_soup);
    assert_eq!(&order_soup[..4], exp_top4_soup.as_slice());

    // 4. Verify 5-Fold Ensemble PreCondenseResidualBridge Stage-2 forward pass
    let mut fold_priors = Vec::with_capacity(5);
    let mut dp_ens = vec![0.0_f64; n_cand];
    for fb in &s2_bundle.fold_bridges {
        let po = fb
            .forward_prior_and_context(&std_ctx, &cand_q_32, &sem24, &gate_risk, &sp_exact, n_cand)
            .unwrap();
        for i in 0..n_cand {
            dp_ens[i] += po.delta_prior[i] / 5.0;
        }
        fold_priors.push(po);
    }
    let exp_dp_ens = g
        .f64_vec("expected_ens.delta_prior_vec", &[n_cand])
        .unwrap();
    assert!(
        max_abs_diff_f64(&dp_ens, &exp_dp_ens) <= 1e-5,
        "dp_ens diff: {}",
        max_abs_diff_f64(&dp_ens, &exp_dp_ens)
    );

    let (rev4_ens, cost_act_ens, y4_comb_ens, y4_f_ens) =
        H256CriticModels::run_2plus2_active_reveal_step(
            &rcost,
            &h_cost_exact,
            &sem24_struct,
            &fitness,
            &logratio,
            &gate,
            &idx4,
            Some(&dp_ens),
            s2_bundle.alpha_prior,
            n_cand,
        )
        .unwrap();
    let exp_rev4_ens: Vec<usize> = g
        .i64_vec("expected_ens.rev4_indices", &[4])
        .unwrap()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    assert_eq!(rev4_ens, exp_rev4_ens);

    let rev_gate_ens: Vec<f32> = rev4_ens.iter().map(|&idx| gate[idx]).collect();
    let base_cost_ens: Vec<f32> = (0..n_cand)
        .map(|i| (cost_act_ens[i] + s2_bundle.alpha_prior * dp_ens[i]) as f32)
        .collect();
    let mut dt_ens = vec![0.0_f64; n_cand];
    for (fb, po) in s2_bundle.fold_bridges.iter().zip(&fold_priors) {
        let dt = fb
            .forward_causal_trajectory_residual(
                &po.attn_ctx_32,
                &po.bilinear_24,
                &cand_q_32,
                &sem24,
                &base_cost_ens,
                &rev4_ens,
                &y4_comb_ens,
                &y4_f_ens,
                &rev_gate_ens,
                n_cand,
            )
            .unwrap();
        for i in 0..n_cand {
            dt_ens[i] += dt[i] / 5.0;
        }
    }
    let exp_dt_ens = g.f64_vec("expected_ens.delta_traj_vec", &[n_cand]).unwrap();
    assert!(max_abs_diff_f64(&dt_ens, &exp_dt_ens) <= 5e-6);

    let cost_final_ens: Vec<f64> = (0..n_cand)
        .map(|i| {
            cost_act_ens[i]
                + s2_bundle.alpha_prior * dp_ens[i]
                + s2_bundle.alpha_traj * dt_ens[i]
                + 0.005 * h_cost_exact[i]
        })
        .collect();
    let exp_cost_final_ens = g.f64_vec("expected_ens.cost_final", &[n_cand]).unwrap();
    assert!(
        max_abs_diff_f64(&cost_final_ens, &exp_cost_final_ens) <= 1e-5,
        "cost_final_ens diff: {}",
        max_abs_diff_f64(&cost_final_ens, &exp_cost_final_ens)
    );

    let mut order_ens: Vec<usize> = (0..n_cand).collect();
    order_ens.sort_by(|&a, &b| {
        cost_final_ens[a]
            .total_cmp(&cost_final_ens[b])
            .then_with(|| a.cmp(&b))
    });
    let exp_pick1_ens = g.i64_vec("expected_ens.pick1_index", &[1]).unwrap()[0] as usize;
    let exp_top4_ens: Vec<usize> = g
        .i64_vec("expected_ens.top4_indices", &[4])
        .unwrap()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    assert_eq!(order_ens[0], exp_pick1_ens);
    assert_eq!(&order_ens[..4], exp_top4_ens.as_slice());
}

#[test]
fn test_h256_production_1310_and_1219_bundles_rust_golden_parity() {
    let dir = weights_dir();
    let p1310_bytes =
        std::fs::read(dir.join("h256_production_full_1310_regression_bundle.safetensors")).unwrap();
    let p1310 = H256ProductionBundle::from_safetensors_bytes(&p1310_bytes).unwrap();

    let p1219_bytes =
        std::fs::read(dir.join("h256_production_1219_dl_train_bundle.safetensors")).unwrap();
    let p1219 = H256ProductionBundle::from_safetensors_bytes(&p1219_bytes).unwrap();

    let golden_bytes =
        std::fs::read(dir.join("golden_inference_fixture_context0.safetensors")).unwrap();
    let g = SafetensorsMap::deserialize(&golden_bytes).unwrap();

    let n_cand = 1152usize;
    let raw_dope629 = g.f32_vec("raw_ctx.dope629", &[629]).unwrap();
    let raw_hidden256 = g.f32_vec("raw_ctx.hidden256", &[256]).unwrap();
    let raw_col_lm = g.f32_vec("raw_ctx.col_lm_64x17", &[64, 17]).unwrap();
    let raw_col_mask = g.f32_vec("raw_ctx.col_mask_64", &[64]).unwrap();
    let raw_lm48_z = g.f32_vec("raw_ctx.lm48_z", &[48]).unwrap();

    let program_412 = g.f32_vec("pool.program_412", &[n_cand, 412]).unwrap();
    let proj16 = g.f32_vec("pool.proj16", &[n_cand, 16]).unwrap();
    let sem24 = g.f32_vec("pool.sem24", &[n_cand, 24]).unwrap();
    let cand_q_32 = g.f32_vec("pool.cand_q_32", &[n_cand, 32]).unwrap();
    let rcost = g.f64_vec("pool.r882_total_cost", &[n_cand]).unwrap();
    let aprior = g.f64_vec("pool.r882_anchor_prior", &[n_cand]).unwrap();
    let sos_prior_2d = g
        .f32_vec("expected_critic.sos_prior_2d", &[n_cand, 2])
        .unwrap();
    let sp_prior_1d = g.f32_vec("expected_critic.sp_prior_1d", &[n_cand]).unwrap();
    let h_cost = g.f64_vec("expected_critic.h_cost", &[n_cand]).unwrap();
    let gate_risk: Vec<f32> = (0..n_cand).map(|i| sos_prior_2d[i * 2 + 1]).collect();

    // 1. Verify S3 16D context embedding extraction (`extract_s3_ctx_vec16`)
    let ct_s3 = p1219
        .s3_scaler
        .transform_context(&raw_dope629, &raw_hidden256, &raw_col_lm, &raw_col_mask)
        .unwrap();
    let s3_ctx_vec16 = p1219.s3_encoder.extract_s3_ctx_vec16(&ct_s3);
    let exp_s3_vec16 = g.f32_vec("expected_top8.s3_ctx_vec16", &[16]).unwrap();
    assert!(
        max_abs_diff_f32(&s3_ctx_vec16, &exp_s3_vec16) <= 5e-6,
        "s3_ctx_vec16 diff: {}",
        max_abs_diff_f32(&s3_ctx_vec16, &exp_s3_vec16)
    );

    // 2. Verify Arm-F and Arm-G soup prior predictions (`dp_f_full`, `dp_g_full`)
    let std_ctx = p1219
        .stage2_scaler
        .transform_context(&raw_dope629, &raw_hidden256, &raw_col_lm, &raw_col_mask)
        .unwrap();
    let dp_f_out = p1310
        .stage2_arm_f_soup
        .forward_prior_and_context(
            &std_ctx,
            &cand_q_32,
            &sem24,
            &gate_risk,
            &sp_prior_1d,
            n_cand,
        )
        .unwrap();
    let dp_g_out = p1310
        .stage2_arm_g_soup
        .forward_prior_and_context(
            &std_ctx,
            &cand_q_32,
            &sem24,
            &gate_risk,
            &sp_prior_1d,
            n_cand,
        )
        .unwrap();

    let exp_dp_f = g.f64_vec("expected_top8.dp_f_full", &[n_cand]).unwrap();
    let exp_dp_g = g.f64_vec("expected_top8.dp_g_full", &[n_cand]).unwrap();
    assert!(max_abs_diff_f64(&dp_f_out.delta_prior, &exp_dp_f) <= 5e-6);
    assert!(max_abs_diff_f64(&dp_g_out.delta_prior, &exp_dp_g) <= 5e-6);

    // 3. Verify pure-Rust 616D Top-8 feature builder (`build_enriched_top8_features_616`)
    let pkt = build_enriched_top8_features_616(
        &rcost,
        &aprior,
        &program_412,
        &sem24,
        &proj16,
        &raw_dope629,
        &raw_hidden256,
        &raw_col_lm,
        &raw_col_mask,
        &raw_lm48_z,
        &s3_ctx_vec16,
        &dp_f_out.delta_prior,
        &dp_g_out.delta_prior,
        &h_cost,
        &gate_risk,
        &sp_prior_1d,
        n_cand,
    )
    .unwrap();

    let exp_top8_indices: Vec<usize> = g
        .i64_vec("expected_top8.top8_indices", &[8])
        .unwrap()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    assert_eq!(pkt.top8_indices.as_slice(), exp_top8_indices.as_slice());

    let exp_x568 = g.f32_vec("expected_top8.X_568", &[8, 568]).unwrap();
    let exp_x616 = g.f32_vec("expected_top8.X_616", &[8, 616]).unwrap();
    assert!(
        max_abs_diff_f32(&pkt.x_568, &exp_x568) <= 1e-5,
        "X_568 max diff: {}",
        max_abs_diff_f32(&pkt.x_568, &exp_x568)
    );
    assert!(
        max_abs_diff_f32(&pkt.x_616, &exp_x616) <= 1e-5,
        "X_616 max diff: {}",
        max_abs_diff_f32(&pkt.x_616, &exp_x616)
    );

    // 4. Verify Production 1310 Top-8 Neural Reranker + 6 GBDTs + Bagged Ensemble Readout
    let pred_1310 = p1310
        .predict_top8_from_matrices(&exp_x568, &exp_x616)
        .unwrap();

    let exp_nn_scores = g.f64_vec("expected_top8.p1310_nn_scores", &[8]).unwrap();
    let exp_nn_inv = g.f64_vec("expected_top8.p1310_nn_inv_probs", &[8]).unwrap();
    let exp_r568 = g
        .f64_vec("expected_top8.p1310_lgbm_ranker_568", &[8])
        .unwrap();
    let exp_r580 = g
        .f64_vec("expected_top8.p1310_lgbm_ranker_580", &[8])
        .unwrap();
    let exp_r616 = g
        .f64_vec("expected_top8.p1310_lgbm_ranker_616", &[8])
        .unwrap();
    let exp_c568 = g.f64_vec("expected_top8.p1310_lgbm_clf_568", &[8]).unwrap();
    let exp_c580 = g.f64_vec("expected_top8.p1310_lgbm_clf_580", &[8]).unwrap();
    let exp_c616 = g.f64_vec("expected_top8.p1310_lgbm_clf_616", &[8]).unwrap();
    let exp_sc_1310 = g
        .f64_vec("expected_top8.p1310_bagged_scores", &[8])
        .unwrap();
    let exp_pinv_1310 = g.f64_vec("expected_top8.p1310_bagged_pinv", &[8]).unwrap();
    let exp_pick_1310 = g
        .i32_vec("expected_top8.p1310_static_pick_j", &[1])
        .unwrap()[0] as usize;

    assert!(
        max_abs_diff_f64(&pred_1310.nn_scores, &exp_nn_scores) <= 5e-6,
        "nn_scores diff: {}",
        max_abs_diff_f64(&pred_1310.nn_scores, &exp_nn_scores)
    );
    assert!(
        max_abs_diff_f64(&pred_1310.nn_inv_probs, &exp_nn_inv) <= 5e-6,
        "nn_inv_probs diff: {}",
        max_abs_diff_f64(&pred_1310.nn_inv_probs, &exp_nn_inv)
    );
    assert!(max_abs_diff_f64(&pred_1310.lgbm_ranker_568, &exp_r568) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.lgbm_ranker_580, &exp_r580) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.lgbm_ranker_616, &exp_r616) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.lgbm_clf_568, &exp_c568) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.lgbm_clf_580, &exp_c580) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.lgbm_clf_616, &exp_c616) <= 1e-12);
    assert!(max_abs_diff_f64(&pred_1310.bagged_scores, &exp_sc_1310) <= 5e-6);
    assert!(max_abs_diff_f64(&pred_1310.bagged_pinv, &exp_pinv_1310) <= 5e-6);
    assert_eq!(pred_1310.static_pick_j, exp_pick_1310);

    // Also verify end-to-end from Rust-computed pkt.x_568 & pkt.x_616
    let pred_1310_from_rust_pkt = p1310
        .predict_top8_from_matrices(&pkt.x_568, &pkt.x_616)
        .unwrap();
    assert!(max_abs_diff_f64(&pred_1310_from_rust_pkt.bagged_scores, &exp_sc_1310) <= 1e-5);
    assert!(max_abs_diff_f64(&pred_1310_from_rust_pkt.bagged_pinv, &exp_pinv_1310) <= 1e-5);
    assert_eq!(pred_1310_from_rust_pkt.static_pick_j, exp_pick_1310);

    // 5. Verify Production 1219 Holdout-Clean Bundle Readout
    let pred_1219 = p1219
        .predict_top8_from_matrices(&pkt.x_568, &pkt.x_616)
        .unwrap();
    let exp_sc_1219 = g
        .f64_vec("expected_top8.p1219_bagged_scores", &[8])
        .unwrap();
    let exp_pinv_1219 = g.f64_vec("expected_top8.p1219_bagged_pinv", &[8]).unwrap();
    let exp_pick_1219 = g
        .i32_vec("expected_top8.p1219_static_pick_j", &[1])
        .unwrap()[0] as usize;

    assert!(max_abs_diff_f64(&pred_1219.bagged_scores, &exp_sc_1219) <= 1e-5);
    assert!(max_abs_diff_f64(&pred_1219.bagged_pinv, &exp_pinv_1219) <= 1e-5);
    assert_eq!(pred_1219.static_pick_j, exp_pick_1219);
}
