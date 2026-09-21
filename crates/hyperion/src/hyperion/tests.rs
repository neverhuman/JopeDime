#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Matrix;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn projection(n_bands: usize, output_dim: usize) -> WeightProjection {
        let mut band_value = vec![0.0; n_bands * output_dim];
        let mut band_abs = vec![0.0; n_bands * output_dim];
        let mut band_missing = vec![0.0; n_bands * output_dim];
        for b in 0..n_bands {
            for d in 0..output_dim {
                let off = b * output_dim + d;
                band_value[off] = ((b + d + 1) as f32 * 0.013).sin() * 0.3;
                band_abs[off] = ((b * 7 + d + 3) as f32 * 0.017).cos() * 0.2;
                band_missing[off] = ((b * 5 + d + 11) as f32 * 0.019).sin() * 0.1;
            }
        }
        WeightProjection {
            n_bands,
            output_dim,
            value_scale: (0..n_bands).map(|b| 0.5 + b as f32 * 0.03).collect(),
            value_bias: vec![0.0; n_bands],
            band_value,
            band_abs,
            band_missing,
            output_bias: vec![0.0; output_dim],
        }
    }

    fn settings() -> HyperionSweepSettings {
        HyperionSweepSettings {
            version: HYPERION_VERSION.into(),
            step: 2000,
            dim: 512,
            heads: 8,
            layers: 12,
            emb_dim: 512,
            min_patch: 2,
            max_patch: 24,
            n_bands: 3,
            q_low: 1.0,
            q_high: 99.0,
            quant_scale: 3.0,
            pad_u8: 128,
            ws_list: vec![4, 7, 11],
            seed: 42,
        }
    }

    fn restore_env(key: &str, value: Option<OsString>) {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    fn with_hyperion_sweeps_env(value: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let key = "JAIN_HYPERION_SWEEPS";
        let previous = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        f();
        restore_env(key, previous);
    }

    fn with_weights_dir_env(weights_dir: Option<&str>, jope_home: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let keys = [
            WEIGHTS_DIR_ENV,
            JOPE_HOME_ENV,
            "JAIN_HYPERION_WEIGHTS",
            "JAIN_HYPERION_MANIFEST",
        ];
        let previous = keys.map(std::env::var_os);
        for key in keys {
            std::env::remove_var(key);
        }
        if let Some(dir) = weights_dir {
            std::env::set_var(keys[0], dir);
        }
        if let Some(home) = jope_home {
            std::env::set_var(keys[1], home);
        }
        f();
        for (key, value) in keys.into_iter().zip(previous) {
            restore_env(key, value);
        }
    }

    #[test]
    fn default_sweeps_is_thirty_and_env_override_still_works() {
        assert_eq!(HyperionConfig::default().sweeps, 30);
        with_hyperion_sweeps_env(None, || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 30);
        });
        with_hyperion_sweeps_env(Some("4"), || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 4);
        });
        with_hyperion_sweeps_env(Some("0"), || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 1);
        });
    }

    #[test]
    fn hyperion_v7_weights_env_supplies_bundle_defaults() {
        with_weights_dir_env(Some("/opt/hyperion-v7"), Some("/srv/jope"), || {
            let config = HyperionConfig::from_env().unwrap();
            assert_eq!(
                config.weights_path,
                PathBuf::from("/opt/hyperion-v7/model.safetensors")
            );
            assert_eq!(
                config.manifest_path,
                PathBuf::from("/opt/hyperion-v7/manifest.json")
            );
        });
        with_weights_dir_env(None, Some("/srv/jope"), || {
            let config = HyperionConfig::from_env().unwrap();
            assert_eq!(
                config.weights_path,
                PathBuf::from("/srv/jope/models/bundles/hyperion-v7/model.safetensors")
            );
            assert_eq!(
                config.manifest_path,
                PathBuf::from("/srv/jope/models/bundles/hyperion-v7/manifest.json")
            );
        });
        with_weights_dir_env(None, None, || {
            let config = HyperionConfig::from_env().unwrap();
            assert_eq!(
                config.weights_path,
                PathBuf::from("models/bundles/hyperion-v7/model.safetensors")
            );
        });
    }

    #[test]
    fn output_names_are_stable() {
        let names = hyperion_output_names(128);
        assert_eq!(names.len(), 128);
        assert_eq!(names[0], "hyperion::v000");
        assert_eq!(names[127], "hyperion::v127");
    }

    #[test]
    fn left_prepend_places_hyperion_before_base_columns() {
        let base = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let hyp = Matrix::new(vec![9.0, 8.0], 2, 1);
        let placed = place_columns(&base, &hyp, HyperionFeaturePlacement::PrependLeft);
        assert_eq!(placed.c, 3);
        assert_eq!(placed.col(0), &[9.0, 8.0]);
        assert_eq!(placed.col(1), &[1.0, 2.0]);
        assert_eq!(placed.col(2), &[3.0, 4.0]);

        let names = place_names(
            &["age".to_string(), "spend".to_string()],
            &["hyperion::v000".to_string()],
            HyperionFeaturePlacement::PrependLeft,
        );
        assert_eq!(names, vec!["hyperion::v000", "age", "spend"]);
    }

    #[test]
    fn old_state_defaults_to_right_append_placement() {
        let json = serde_json::json!({
            "version": HYPERION_VERSION,
            "requested_mode": "auto",
            "output_names": ["hyperion::v000"],
            "input_names": ["x"],
            "input_indices": [0],
            "quantile_bounds": [{"low": 0.0, "high": 1.0}],
            "sweep": settings(),
            "output_dim": 1,
            "model": {
                "weights_path": "w",
                "manifest_path": "m",
                "weights_bytes": 0,
                "weights_hash": "h",
                "manifest_hash": "h",
                "tensor_count": 0
            }
        });
        let state: HyperionState = serde_json::from_value(json).unwrap();
        assert_eq!(
            state.feature_placement,
            HyperionFeaturePlacement::AppendRight
        );
    }

    #[test]
    fn train_bounds_replay_holdout_shape_and_order() {
        let train = Matrix::new(
            vec![
                0.0, 0.5, 1.0, 1.5, // col 0
                1.0, 0.0, 1.0, 0.0, // col 1
            ],
            4,
            2,
        );
        let hold = Matrix::new(vec![0.25, 0.75, 1.0, 0.0], 2, 2);
        let bounds = fit_quantile_bounds(&train, 1.0, 99.0);
        let settings = settings();
        let projection = projection(settings.n_bands, 128);
        let h1 = encode_with_projection(&hold, &bounds, &settings, &projection);
        let h2 = encode_with_projection(&hold, &bounds, &settings, &projection);
        assert_eq!(h1.n, 2);
        assert_eq!(h1.c, 128);
        assert_eq!(h1.data, h2.data);
        assert!(h1.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn degenerate_matrices_encode_to_valid_values() {
        let x = Matrix::new(Vec::new(), 3, 0);
        let settings = settings();
        let projection = projection(settings.n_bands, 128);
        let encoded = encode_with_projection(&x, &[], &settings, &projection);
        assert_eq!(encoded.n, 3);
        assert_eq!(encoded.c, 128);
        assert!(encoded.data.iter().all(|v| *v == 0.0));

        let x = Matrix::new(vec![f32::NAN, f32::NAN, f32::NAN], 3, 1);
        let bounds = fit_quantile_bounds(&x, 1.0, 99.0);
        let encoded = encode_with_projection(&x, &bounds, &settings, &projection);
        assert!(encoded.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preview_flags_robust_row_concerns_without_raw_values() {
        let mut data = Vec::new();
        for col in 0..128 {
            for row in 0..12 {
                let value = if (5..=6).contains(&row) {
                    0.9
                } else {
                    0.02 + col as f32 * 0.00001
                };
                data.push(value);
            }
        }
        let block = Matrix::new(data, 12, 128);
        let preview = build_hyperion_preview(&block);
        assert_eq!(preview.rows, 12);
        assert_eq!(preview.points.len(), 12);
        assert!(preview
            .top_rows
            .iter()
            .any(|row| row.row == 5 || row.row == 6));
        assert_eq!(preview.concern_ranges[0].start, 5);
        assert_eq!(preview.concern_ranges[0].end, 6);
    }

    #[test]
    fn missing_weights_modes_differ() {
        let cfg = HyperionConfig {
            mode: HyperionMode::Auto,
            weights_path: PathBuf::from("/definitely/missing/model.safetensors"),
            manifest_path: PathBuf::from("/definitely/missing/manifest.json"),
            ..HyperionConfig::default()
        };
        let x = Matrix::new(vec![0.0, 1.0], 2, 1);
        let names = vec!["x".to_string()];
        let auto = fit_transform_pair(
            &x,
            &x,
            &names,
            &cfg,
            &crate::progress::NullSink,
            Instant::now(),
        )
        .unwrap();
        assert!(auto.state.is_none());
        assert_eq!(auto.train.c, 1);

        let mut on = cfg;
        on.mode = HyperionMode::On;
        let err = match fit_transform_pair(
            &x,
            &x,
            &names,
            &on,
            &crate::progress::NullSink,
            Instant::now(),
        ) {
            Ok(_) => panic!("Hyperion on unexpectedly succeeded with missing weights"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("Hyperion manifest not found"));

        let mut off = on;
        off.mode = HyperionMode::Off;
        let skipped = fit_transform_pair(
            &x,
            &x,
            &names,
            &off,
            &crate::progress::NullSink,
            Instant::now(),
        )
        .unwrap();
        assert!(skipped.state.is_none());
        assert_eq!(skipped.train.c, 1);
    }
}
