use super::{
    build_hyperion_preview, dump_hyperion_block, emit_hyperion, encode_with_projection,
    fit_quantile_bounds, fit_v7, hyperion_device_label, load_runtime, mix64, place_columns,
    place_names, HyperionAugment, HyperionConfig, HyperionEngine, HyperionFeaturePlacement,
    HyperionMode, HyperionState, Matrix, Result, Sink, V7Fit, DEFAULT_OUTPUT_DIM,
};
use anyhow::bail;
use std::time::Instant;

pub fn hyperion_output_names(output_dim: usize) -> Vec<String> {
    (0..output_dim)
        .map(|j| format!("hyperion::v{j:03}"))
        .collect()
}

pub fn fit_transform_pair(
    train: &Matrix,
    hold: &Matrix,
    input_names: &[String],
    cfg: &HyperionConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> Result<HyperionAugment> {
    if cfg.require_gpu && cfg.mode == HyperionMode::Off {
        bail!("Hyperion require_gpu cannot be combined with HyperionMode::Off");
    }
    if cfg.require_gpu && cfg.engine != HyperionEngine::V7 {
        bail!("Hyperion require_gpu requires the CUDA-backed v7 cellmat engine");
    }
    let output_dim = cfg.output_dim.min(DEFAULT_OUTPUT_DIM);
    emit_hyperion(
        sink,
        t0,
        "start",
        crate::hyperion::HyperionEvent {
            rows: train.n,
            cols: train.c,
            output_dim,
            engine: cfg.engine,
            device: Some(hyperion_device_label()),
            status: Some(match cfg.mode {
                HyperionMode::Auto => "auto",
                HyperionMode::On => "on",
                HyperionMode::Off => "off",
            }),
            message: None,
            done_frac: Some(0.0),
            preview: None,
        },
    );

    if cfg.mode == HyperionMode::Off {
        emit_hyperion(
            sink,
            t0,
            "done",
            crate::hyperion::HyperionEvent {
                rows: train.n,
                cols: train.c,
                output_dim: 0,
                engine: cfg.engine,
                device: None,
                status: Some("skipped"),
                message: Some("Hyperion disabled"),
                done_frac: Some(1.0),
                preview: None,
            },
        );
        return Ok(HyperionAugment {
            train: train.clone(),
            hold: hold.clone(),
            names: input_names.to_vec(),
            state: None,
            confidence: None,
            hold_confidence: None,
            dataset_vector: None,
        });
    }

    let load = load_runtime(cfg, output_dim);
    let runtime = match load {
        Ok(runtime) => runtime,
        Err(err) => {
            if cfg.require_gpu || cfg.mode == HyperionMode::On {
                emit_hyperion(
                    sink,
                    t0,
                    "failed",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim,
                        engine: cfg.engine,
                        device: Some(hyperion_device_label()),
                        status: Some("failed"),
                        message: Some(&err.to_string()),
                        done_frac: Some(1.0),
                        preview: None,
                    },
                );
                return Err(err);
            }
            emit_hyperion(
                sink,
                t0,
                "warning",
                crate::hyperion::HyperionEvent {
                    rows: train.n,
                    cols: train.c,
                    output_dim: 0,
                    engine: cfg.engine,
                    device: Some(hyperion_device_label()),
                    status: Some("skipped"),
                    message: Some(&format!("Hyperion unavailable in auto mode: {err:#}")),
                    done_frac: Some(1.0),
                    preview: None,
                },
            );
            emit_hyperion(
                sink,
                t0,
                "done",
                crate::hyperion::HyperionEvent {
                    rows: train.n,
                    cols: train.c,
                    output_dim: 0,
                    engine: cfg.engine,
                    device: None,
                    status: Some("skipped"),
                    message: None,
                    done_frac: Some(1.0),
                    preview: None,
                },
            );
            return Ok(HyperionAugment {
                train: train.clone(),
                hold: hold.clone(),
                names: input_names.to_vec(),
                state: None,
                confidence: None,
                hold_confidence: None,
                dataset_vector: None,
            });
        }
    };

    if train.c != input_names.len() {
        bail!(
            "Hyperion input shape mismatch: matrix has {} columns but {} names were provided",
            train.c,
            input_names.len()
        );
    }
    if hold.c != train.c {
        bail!(
            "Hyperion holdout shape mismatch: train has {} columns but holdout has {}",
            train.c,
            hold.c
        );
    }

    let bounds = fit_quantile_bounds(train, runtime.settings.q_low, runtime.settings.q_high);

    // ── Engine dispatch: the TRUE v7 forward (default) vs the classic projection encoder. A v7
    // load failure downgrades exactly like a runtime load failure (On → fail, Auto → skip).
    let mut v7_result: Option<V7Fit> = None;
    if cfg.engine == HyperionEngine::V7 {
        match fit_v7(
            train,
            hold,
            &bounds,
            &runtime.settings,
            cfg,
            output_dim,
            sink,
            t0,
        ) {
            Ok(fit) => {
                if fit.cancelled {
                    // Cooperative cancel mid-encode: a partial encoding would poison the features,
                    // so continue UNAUGMENTED; the caller's cancel flag halts the run right after.
                    emit_hyperion(
                        sink,
                        t0,
                        "done",
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim: 0,
                            engine: HyperionEngine::V7,
                            device: fit.device,
                            status: Some("skipped"),
                            message: Some("Hyperion encode cancelled"),
                            done_frac: Some(1.0),
                            preview: None,
                        },
                    );
                    return Ok(HyperionAugment {
                        train: train.clone(),
                        hold: hold.clone(),
                        names: input_names.to_vec(),
                        state: None,
                        confidence: None,
                        hold_confidence: None,
                        dataset_vector: None,
                    });
                }
                if fit.advanced {
                    emit_hyperion(sink,
t0,
"advanced",
crate::hyperion::HyperionEvent { rows: train.n, cols: train.c, output_dim: 0, engine: HyperionEngine::V7, device: fit.device, status: Some("skipped"), message: Some(
                            "Hyperion encode advanced after the current chunk; continuing without partial embedding",
                        ), done_frac: Some(1.0), preview: None },);
                    emit_hyperion(
                        sink,
                        t0,
                        "done",
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim: 0,
                            engine: HyperionEngine::V7,
                            device: fit.device,
                            status: Some("skipped"),
                            message: None,
                            done_frac: Some(1.0),
                            preview: None,
                        },
                    );
                    return Ok(HyperionAugment {
                        train: train.clone(),
                        hold: hold.clone(),
                        names: input_names.to_vec(),
                        state: None,
                        confidence: None,
                        hold_confidence: None,
                        dataset_vector: None,
                    });
                }
                v7_result = Some(fit);
            }
            Err(err) => {
                if cfg.require_gpu || cfg.mode == HyperionMode::On {
                    emit_hyperion(
                        sink,
                        t0,
                        "failed",
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim,
                            engine: HyperionEngine::V7,
                            device: Some(hyperion_device_label()),
                            status: Some("failed"),
                            message: Some(&err.to_string()),
                            done_frac: Some(1.0),
                            preview: None,
                        },
                    );
                    return Err(err);
                }
                emit_hyperion(
                    sink,
                    t0,
                    "warning",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim: 0,
                        engine: HyperionEngine::V7,
                        device: Some(hyperion_device_label()),
                        status: Some("skipped"),
                        message: Some(&format!("Hyperion v7 unavailable in auto mode: {err:#}")),
                        done_frac: Some(1.0),
                        preview: None,
                    },
                );
                emit_hyperion(
                    sink,
                    t0,
                    "done",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim: 0,
                        engine: HyperionEngine::V7,
                        device: None,
                        status: Some("skipped"),
                        message: None,
                        done_frac: Some(1.0),
                        preview: None,
                    },
                );
                return Ok(HyperionAugment {
                    train: train.clone(),
                    hold: hold.clone(),
                    names: input_names.to_vec(),
                    state: None,
                    confidence: None,
                    hold_confidence: None,
                    dataset_vector: None,
                });
            }
        }
    }

    let mut used_engine = HyperionEngine::Projection;
    let mut used_device = Some("cpu");
    let (train_block, hold_block, confidence, hold_confidence, dataset_vector) =
        if let Some(fit) = v7_result {
            used_engine = HyperionEngine::V7;
            used_device = fit.device;
            (
                fit.train_block,
                fit.hold_block,
                Some(fit.confidence),
                Some(fit.hold_confidence),
                Some(fit.dataset_vector),
            )
        } else {
            let sweeps = cfg.sweeps.max(1);
            let (tb, hb) = if sweeps <= 1 {
                (
                    encode_with_projection(train, &bounds, &runtime.settings, &runtime.projection),
                    encode_with_projection(hold, &bounds, &runtime.settings, &runtime.projection),
                )
            } else {
                // Combine K sweeps (each with a fresh per-row start/stride via a bumped seed) by averaging
                // the output blocks -> a more stable row embedding for the record-outlier detector.
                let mut tb = vec![0f32; train.n * output_dim];
                let mut hb = vec![0f32; hold.n * output_dim];
                for k in 0..sweeps {
                    let mut s = runtime.settings.clone();
                    s.seed = runtime.settings.seed ^ mix64(k as u64);
                    let t = encode_with_projection(train, &bounds, &s, &runtime.projection);
                    let h = encode_with_projection(hold, &bounds, &s, &runtime.projection);
                    for (a, b) in tb.iter_mut().zip(t.data.iter()) {
                        *a += *b;
                    }
                    for (a, b) in hb.iter_mut().zip(h.data.iter()) {
                        *a += *b;
                    }
                }
                let inv = 1.0 / sweeps as f32;
                tb.iter_mut().for_each(|a| *a *= inv);
                hb.iter_mut().for_each(|a| *a *= inv);
                (
                    Matrix::new(tb, train.n, output_dim),
                    Matrix::new(hb, hold.n, output_dim),
                )
            };
            (tb, hb, None, None, None)
        };
    // Optional dump of the (averaged) sweep matrix for external record-outlier scoring.
    if let Ok(prefix) = std::env::var("JAIN_HYPERION_DUMP") {
        if !prefix.trim().is_empty() {
            dump_hyperion_block(&format!("{}.train.csv", prefix.trim()), &train_block);
        }
    }
    let preview = build_hyperion_preview(&train_block);
    let output_names = hyperion_output_names(output_dim);
    let names = place_names(input_names, &output_names, cfg.feature_placement);
    let is_v7 = confidence.is_some();
    let state = HyperionState {
        version: runtime.settings.version.clone(),
        requested_mode: cfg.mode,
        output_names,
        input_names: input_names.to_vec(),
        input_indices: (0..input_names.len()).collect(),
        quantile_bounds: bounds,
        sweep: runtime.settings,
        output_dim,
        model: runtime.identity,
        engine: if is_v7 {
            "v7".to_string()
        } else {
            "projection".to_string()
        },
        sweeps: cfg.sweeps.max(1),
        scales: if is_v7 {
            cfg.scales.clone()
        } else {
            Vec::new()
        },
        batch: if is_v7 { cfg.batch.max(1) } else { 0 },
        feature_placement: cfg.feature_placement,
        orig_v2: None,
    };
    let train = place_columns(train, &train_block, cfg.feature_placement);
    let hold = place_columns(hold, &hold_block, cfg.feature_placement);

    emit_hyperion(
        sink,
        t0,
        "done",
        crate::hyperion::HyperionEvent {
            rows: train.n,
            cols: input_names.len(),
            output_dim,
            engine: used_engine,
            device: used_device,
            status: Some("used"),
            message: None,
            done_frac: Some(1.0),
            preview: Some(preview),
        },
    );
    Ok(HyperionAugment {
        train,
        hold,
        names,
        state: Some(state),
        confidence,
        hold_confidence,
        dataset_vector,
    })
}
