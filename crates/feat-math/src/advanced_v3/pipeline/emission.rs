use super::*;

/// Static preflight check for direct `fit_table` + `emit_bound` support on an
/// all-numeric IID schema. Returns `Err((true, msg))` if `fit_table` will fail
/// closed (e.g. `generated dense width ... exceeds 512`) or `Err((false, msg))`
/// if `emit_bound` will fail closed (e.g. unsupported legacy head or transform),
/// avoiding a doomed redundant model fit while preserving exact error strings.
pub fn check_direct_bound_emission_support(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
) -> std::result::Result<(), (bool, String)> {
    if let Err(e) = validate_v3_genome(genome, task, schema) {
        return Err((true, format!("{e:#}")));
    }
    let GenomeIr::Pipeline(dsl) = genome else {
        return Err((
            false,
            "bound emission currently requires a v3 pipeline model".to_string(),
        ));
    };
    let parsed = match split_v3_pipeline(dsl) {
        Ok(p) => p,
        Err(e) => return Err((true, format!("{e:#}"))),
    };
    let input_cfg = match InputConfig::from_stage(parsed.input) {
        Ok(c) => c,
        Err(e) => return Err((true, format!("{e:#}"))),
    };
    if schema
        .columns
        .iter()
        .all(|c| c.kind == super::schema::ColumnKind::Numeric)
    {
        let mut cols = schema.columns.len();
        if parsed.input.is_some() {
            // Check if missing indicator is enabled via InputConfig serialization or stage param
            let _ = &input_cfg;
            if let Some(stage) = parsed.input {
                if matches!(str_param(stage, "missing", "indicator"), Ok("indicator")) {
                    cols *= 2;
                }
            }
        }
        let mut generated_dense = 0usize;
        let max_dense = super::feature_block::MAX_GENERATED_DENSE_WIDTH;
        for stage in parsed.transforms {
            let append = matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
                || stage.get_usize("append", 0) == 1;
            match stage.op.as_str() {
                "maxabs_scale" | "power_transform" | "standardize" | "norm_features"
                | "robust_scale" | "target_bin_encode" | "sigmoid" | "step" | "norm_row"
                | "winsor_clip" | "winsor_moments" => {
                    if append {
                        generated_dense += cols;
                        cols += cols;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    }
                }
                "corr_screen" => {
                    let req = stage.get_usize("k", 32).min(cols);
                    if append {
                        generated_dense += req;
                        cols += req;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    } else {
                        cols = req;
                    }
                }
                "select_corr" => {
                    let k = stage.get_usize("k", 32).clamp(1, 512).min(cols);
                    if append {
                        generated_dense += k;
                        cols += k;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    } else {
                        cols = k;
                    }
                }
                "piecewise_linear" => {
                    let bins = stage.get_usize("bins", 8);
                    let gen_w = cols.saturating_mul(bins);
                    if gen_w > max_dense {
                        return Err((
                            true,
                            format!("piecewise_linear derived width {gen_w} exceeds {max_dense}"),
                        ));
                    }
                    generated_dense = (if append { generated_dense } else { 0 }) + gen_w;
                    if generated_dense > max_dense {
                        return Err((
                            true,
                            format!("generated dense width {generated_dense} exceeds {max_dense}"),
                        ));
                    }
                    cols = (if append { cols } else { 0 }) + gen_w;
                }
                "spline" | "impute" | "lag" | "roll" => {
                    // Data-dependent width or table row op; defer to fit_table if needed
                }
                other => {
                    let gen_w = match other {
                        "pls_project" => stage.get_usize("k", 8),
                        "cheb" => cols * stage.get_usize("degree", 4).clamp(1, 8),
                        "poly_interact" => {
                            (cols + cols * cols.saturating_sub(1) / 2).min(crate::MAX_WIDTH)
                        }
                        "rand_proj" => stage.get_usize("k", 16).clamp(1, 256),
                        "rff_cos" | "rff_sin" | "oblique_hinge" => {
                            stage.get_usize("n", 32).clamp(1, 256)
                        }
                        "landmark_dist" | "nystrom" => stage.get_usize("k", 16).clamp(1, 128),
                        "svd_project" => stage.get_usize("k", 8).clamp(1, 64).min(cols),
                        "pair_prod" => {
                            let m = stage.get_usize("topk", 8).clamp(2, 24).min(cols);
                            m * m.saturating_sub(1) / 2
                        }
                        "pair_ratio" => {
                            let m = cols.min(32);
                            let max_p = m * m.saturating_sub(1);
                            stage.get_usize("topk", 8).clamp(1, 64).min(max_p)
                        }
                        "row_stats" => 10,
                        "row_spectral" | "row_diff_stats" => 4,
                        "fft_mag" => stage.get_usize("k", 8).clamp(1, (cols / 2).max(1)),
                        "polar" => (cols / 2) * 2,
                        "hinge" => cols * stage.get_usize("knots", 4).clamp(1, 16),
                        "quantile_bins" | "rank" | "blom" | "symlog" | "map" => cols,
                        "bin_onehot" => cols * stage.get_usize("bins", 6).clamp(2, 32),
                        "derive" => 1,
                        _ => 0,
                    };
                    if gen_w > 0 {
                        generated_dense = (if append { generated_dense } else { 0 }) + gen_w;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                        cols = (if append { cols } else { 0 }) + gen_w;
                    }
                }
            }
        }
    }

    // Check head support in emit_bound_python order
    let head_op = parsed.head.map_or(
        match task.kind {
            TaskKind::Binary => "logit_ridge",
            TaskKind::Multiclass => "multiclass_ridge",
            TaskKind::Regression | TaskKind::Timeseries => "ridge",
        },
        |stage| stage.op.as_str(),
    );
    match head_op {
        "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls" | "portfolio"
        | "ridge" => {}
        "multiclass_ridge"
        | "multiclass_ovr_logistic"
        | "multiclass_diagonal_lda"
        | "multiclass_nearest_centroid" => {
            return Err((
                false,
                "multiclass probability matrices are not supported by scalar bound emission"
                    .to_string(),
            ));
        }
        _ => {
            return Err((
                false,
                "this fitted legacy head is not supported by bound emission".to_string(),
            ));
        }
    }

    // Check transform support in emit_bound_rust order
    for stage in parsed.transforms {
        match stage.op.as_str() {
            "maxabs_scale" | "power_transform" | "piecewise_linear" | "corr_screen"
            | "pls_project" | "lag" | "roll" | "standardize" | "norm_features" | "robust_scale"
            | "target_bin_encode" | "sigmoid" | "step" | "norm_row" | "select_corr"
            | "winsor_clip" | "winsor_moments" | "impute" => {}
            _ => {
                return Err((
                    false,
                    "this legacy transform is not supported by bound emission".to_string(),
                ));
            }
        }
    }

    Ok(())
}
