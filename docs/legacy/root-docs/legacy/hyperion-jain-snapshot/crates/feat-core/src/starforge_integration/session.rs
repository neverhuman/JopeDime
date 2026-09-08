#[cfg(feature = "starforge-cpu")]
use super::{
    context_rows_for_task, cuda_available, labels_f32, labels_u8, matrix_rows, StarforgeContext,
    StarforgeContextTask, StarforgeDType, StarforgeDevice, StarforgeRegressionEngine,
};
use super::{StarforgeConfig, StarforgeMemberPlan, StarforgeMode};
#[cfg(feature = "starforge-cpu")]
use crate::data::Matrix;
use crate::task::TaskKind;
use anyhow::{bail, Result};

#[derive(Clone)]
pub struct StarforgeCandidate {
    pub name: String,
    pub oof: Vec<f32>,
    pub infer: Vec<f32>,
    pub fold_auc: Vec<f64>,
    pub mean_auc: f64,
    pub lcb_auc: f64,
    pub plan: StarforgeMemberPlan,
}

#[cfg(feature = "starforge-cpu")]
mod enabled {
    use super::*;
    use crate::metrics::task_score;
    use starforge::model::{ChimeraRegressor, StarforgeModel};

    enum RuntimeModel {
        Starforge(StarforgeModel),
        Chimera(Box<ChimeraRegressor>),
    }

    impl RuntimeModel {
        fn as_starforge(&self) -> Result<&StarforgeModel> {
            match self {
                Self::Starforge(model) => Ok(model),
                Self::Chimera(_) => bail!("Chimera regression engine cannot run classifier smoke"),
            }
        }
    }

    pub struct Session {
        model: RuntimeModel,
        dtype: StarforgeDType,
        device_label: String,
        regression_engine: Option<StarforgeRegressionEngine>,
    }

    impl Session {
        pub fn load(cfg: &StarforgeConfig, task: TaskKind) -> Result<Option<Self>> {
            if cfg.mode == StarforgeMode::Off {
                return Ok(None);
            }
            let regression_engine = (task == TaskKind::Regression)
                .then(|| resolve_regression_engine(cfg))
                .transpose()?;
            let weights_path = cfg.weights_path_for_task(task);
            if task != TaskKind::Regression && !weights_path.is_file() {
                if cfg.mode == StarforgeMode::Auto {
                    eprintln!(
                        "[starforge] weights missing at {}; skipping Starforge candidates",
                        weights_path.display()
                    );
                    return Ok(None);
                }
                bail!("Starforge weights missing at {}", weights_path.display());
            }
            if task == TaskKind::Regression {
                match regression_engine {
                    Some(StarforgeRegressionEngine::Chimera) => {
                        for path in [
                            &cfg.regressor_weights_path,
                            &cfg.tabicl_regressor_weights_path,
                            &cfg.chimera_weights_path,
                        ] {
                            if !path.is_file() {
                                if cfg.mode == StarforgeMode::Auto {
                                    eprintln!(
                                        "[starforge] Chimera weights missing at {}; skipping Starforge candidates",
                                        path.display()
                                    );
                                    return Ok(None);
                                }
                                bail!("Starforge Chimera weights missing at {}", path.display());
                            }
                        }
                    }
                    _ if !cfg.regressor_weights_path.is_file() => {
                        if cfg.mode == StarforgeMode::Auto {
                            eprintln!(
                                "[starforge] regression weights missing at {}; skipping Starforge candidates",
                                cfg.regressor_weights_path.display()
                            );
                            return Ok(None);
                        }
                        bail!(
                            "Starforge regression weights missing at {}",
                            cfg.regressor_weights_path.display()
                        );
                    }
                    _ => {}
                }
            }
            let cuda_ok = cuda_available();
            if cfg.require_gpu && !cuda_ok {
                bail!("Starforge GPU was required but CUDA is unavailable");
            }
            let use_cuda = match cfg.device {
                StarforgeDevice::Cpu => false,
                StarforgeDevice::Cuda0 => {
                    if !cuda_ok {
                        bail!("Starforge device cuda:0 requested but CUDA is unavailable");
                    }
                    true
                }
                StarforgeDevice::Auto => cuda_ok,
            };
            if !use_cuda && cfg.device != StarforgeDevice::Cpu {
                eprintln!(
                    "[starforge] WARNING: CUDA is unavailable; using CPU execution mode for Starforge"
                );
            }
            let (device, device_label) = if use_cuda {
                (cuda_device()?, "cuda:0".to_string())
            } else {
                (candle_core::Device::Cpu, "cpu".to_string())
            };
            // BF16 is a CUDA-only optimization (candle's CPU backend has no bf16 matmul); on CPU we
            // always run f32 so a bf16-configured/GPU-trained model stays servable on a CPU endpoint.
            if !use_cuda && cfg.dtype == StarforgeDType::BF16 {
                eprintln!("[starforge] NOTE: bf16 is unsupported on CPU; running Starforge in f32");
            }
            let dtype = cfg.dtype.effective(use_cuda);
            let candle_dtype = match dtype {
                StarforgeDType::BF16 => candle_core::DType::BF16,
                StarforgeDType::F32 | StarforgeDType::Auto => candle_core::DType::F32,
            };
            eprintln!(
                "[starforge] loading weights={} device={} dtype={} regression_engine={}",
                weights_path.display(),
                device_label,
                dtype.label(),
                regression_engine
                    .map(StarforgeRegressionEngine::label)
                    .unwrap_or("n/a")
            );
            let model = if task == TaskKind::Regression
                && regression_engine == Some(StarforgeRegressionEngine::Chimera)
            {
                RuntimeModel::Chimera(Box::new(ChimeraRegressor::load_dtype(
                    &cfg.regressor_weights_path,
                    &cfg.tabicl_regressor_weights_path,
                    &cfg.chimera_weights_path,
                    device,
                    candle_dtype,
                )?))
            } else {
                let model = StarforgeModel::load_dtype(weights_path, device, candle_dtype)?;
                match task {
                    TaskKind::Regression
                        if model.kind() != starforge::model::StarforgeModelKind::Regressor =>
                    {
                        bail!("Starforge regression task was given classifier weights")
                    }
                    TaskKind::Binary | TaskKind::MultiLabel
                        if model.kind() != starforge::model::StarforgeModelKind::Classifier =>
                    {
                        bail!("Starforge classification task was given regressor weights")
                    }
                    _ => {}
                }
                RuntimeModel::Starforge(model)
            };
            Ok(Some(Self {
                model,
                dtype,
                device_label,
                regression_engine,
            }))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn candidate(
            &self,
            name: &str,
            x: &Matrix,
            y: &[f32],
            xh: &Matrix,
            _yh: &[f32],
            folds: &[(Vec<usize>, Vec<usize>)],
            cols: &[u32],
            cfg: &StarforgeConfig,
            task: TaskKind,
            n_classes: usize,
        ) -> Result<StarforgeCandidate> {
            if !matches!(
                task,
                TaskKind::Binary | TaskKind::MultiLabel | TaskKind::Regression
            ) {
                bail!(
                    "Starforge accepts binary, multilabel, or regression tasks; received multiclass"
                );
            }
            if cols.is_empty() {
                bail!("Starforge candidate {name} has no selected columns");
            }
            let mut oof = vec![f32::NAN; y.len()];
            let mut fold_auc = Vec::with_capacity(folds.len());
            for (fold_idx, (tr, va)) in folds.iter().enumerate() {
                let ytr: Vec<f32> = tr.iter().map(|&i| y[i]).collect();
                let rel_ctx =
                    context_rows_for_task(&ytr, cfg.max_context, cfg_seed(cfg, fold_idx), task);
                let ctx_rows: Vec<usize> = rel_ctx.into_iter().map(|i| tr[i]).collect();
                let ctx_x = matrix_rows(x, &ctx_rows, cols);
                let qx = matrix_rows(x, va, cols);
                let pred = match task {
                    TaskKind::Regression => {
                        let ctx_y = labels_f32(y, &ctx_rows);
                        self.predict_regression(&ctx_x, &ctx_y, &qx, cols.len())?
                    }
                    _ => {
                        let ctx_y = labels_u8(y, &ctx_rows);
                        match &self.model {
                            RuntimeModel::Starforge(model) => {
                                model.predict_ensemble_auc(&ctx_x, &ctx_y, &qx, cols.len())?
                            }
                            RuntimeModel::Chimera(_) => {
                                bail!("Chimera regression engine cannot score classification")
                            }
                        }
                    }
                };
                let pred: Vec<f32> = pred.into_iter().map(|v| v as f32).collect();
                for (j, &row) in va.iter().enumerate() {
                    oof[row] = pred[j];
                }
                let yv: Vec<f32> = va.iter().map(|&i| y[i]).collect();
                fold_auc.push(task_score(task, &yv, &pred, n_classes));
            }
            if oof.iter().any(|v| !v.is_finite()) {
                bail!("Starforge candidate {name} has incomplete OOF coverage");
            }
            let ctx_rows = context_rows_for_task(y, cfg.max_context, cfg.seed_for_context(), task);
            let ctx_x = matrix_rows(x, &ctx_rows, cols);
            let all_hold: Vec<usize> = (0..xh.n).collect();
            let qx = matrix_rows(xh, &all_hold, cols);
            let infer: Vec<f32> = match task {
                TaskKind::Regression => {
                    let ctx_y = labels_f32(y, &ctx_rows);
                    self.predict_regression(&ctx_x, &ctx_y, &qx, cols.len())?
                }
                _ => {
                    let ctx_y = labels_u8(y, &ctx_rows);
                    match &self.model {
                        RuntimeModel::Starforge(model) => {
                            model.predict_ensemble_auc(&ctx_x, &ctx_y, &qx, cols.len())?
                        }
                        RuntimeModel::Chimera(_) => {
                            bail!("Chimera regression engine cannot score classification")
                        }
                    }
                }
            }
            .into_iter()
            .map(|v| v as f32)
            .collect();
            let k = fold_auc.len().max(1) as f64;
            let mean = fold_auc.iter().sum::<f64>() / k;
            let var = fold_auc
                .iter()
                .map(|a| (a - mean) * (a - mean))
                .sum::<f64>()
                / k;
            Ok(StarforgeCandidate {
                name: name.into(),
                oof,
                infer,
                fold_auc,
                mean_auc: mean,
                lcb_auc: mean - var.sqrt() / k.sqrt(),
                plan: StarforgeMemberPlan {
                    selected_cols: cols.to_vec(),
                    weights_path: cfg.weights_path_for_task(task).display().to_string(),
                    dtype: self.dtype.label().to_string(),
                    max_context: cfg.max_context,
                    context_seed: cfg.seed_for_context(),
                    task: StarforgeContextTask::from_task(task),
                    regression_engine: self
                        .regression_engine
                        .map(|engine| engine.label().to_string()),
                    chimera_weights_path: (self.regression_engine
                        == Some(StarforgeRegressionEngine::Chimera))
                    .then(|| cfg.chimera_weights_path.display().to_string()),
                    tabicl_regressor_weights_path: (self.regression_engine
                        == Some(StarforgeRegressionEngine::Chimera))
                    .then(|| cfg.tabicl_regressor_weights_path.display().to_string()),
                },
            })
        }

        pub fn predict_context(&self, ctx: &StarforgeContext, q: &Matrix) -> Result<Vec<f32>> {
            let rows: Vec<usize> = (0..q.n).collect();
            let cols: Vec<u32> = (0..q.c as u32).collect();
            let qx = matrix_rows(q, &rows, &cols);
            let pred = match ctx.task {
                StarforgeContextTask::Regressor => {
                    self.predict_regression(&ctx.x, &ctx.y, &qx, ctx.n_features)?
                }
                StarforgeContextTask::Classifier => {
                    let y: Vec<u8> = ctx.y.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
                    match &self.model {
                        RuntimeModel::Starforge(model) => {
                            model.predict_ensemble_auc(&ctx.x, &y, &qx, ctx.n_features)?
                        }
                        RuntimeModel::Chimera(_) => {
                            bail!("Chimera regression engine cannot score classification")
                        }
                    }
                }
            };
            Ok(pred.into_iter().map(|v| v as f32).collect())
        }

        fn predict_regression(
            &self,
            ctx_x: &[f32],
            ctx_y: &[f32],
            qx: &[f32],
            n_features: usize,
        ) -> Result<Vec<f64>> {
            match &self.model {
                RuntimeModel::Starforge(model) => {
                    model.predict_regression_ensemble(ctx_x, ctx_y, qx, n_features)
                }
                RuntimeModel::Chimera(model) => {
                    model.predict_regression_single(ctx_x, ctx_y, qx, n_features)
                }
            }
        }

        pub fn device_label(&self) -> &str {
            &self.device_label
        }
    }

    fn cfg_seed(cfg: &StarforgeConfig, fold_idx: usize) -> u64 {
        cfg.seed_for_context()
            .wrapping_add((fold_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn resolve_regression_engine(cfg: &StarforgeConfig) -> Result<StarforgeRegressionEngine> {
        match cfg.regression_engine {
            StarforgeRegressionEngine::Auto | StarforgeRegressionEngine::Chimera => {
                Ok(StarforgeRegressionEngine::Chimera)
            }
            StarforgeRegressionEngine::Starlight => Ok(StarforgeRegressionEngine::Starlight),
        }
    }

    #[cfg(feature = "starforge-cuda")]
    fn cuda_device() -> Result<candle_core::Device> {
        Ok(candle_core::Device::new_cuda(0)?)
    }

    #[cfg(not(feature = "starforge-cuda"))]
    fn cuda_device() -> Result<candle_core::Device> {
        bail!("feat-core was not compiled with starforge-cuda")
    }

    pub fn smoke(cfg: &StarforgeConfig) -> Result<Option<(String, Vec<f32>)>> {
        let Some(session) = Session::load(cfg, TaskKind::Binary)? else {
            return Ok(None);
        };
        let ctx_x = vec![0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0];
        let ctx_y = vec![0, 1, 0, 1];
        let qx = vec![0.5, 0.5, 1.5, 1.5];
        let pred: Vec<f32> = session
            .model
            .as_starforge()?
            .predict_single(&ctx_x, &ctx_y, &qx, 2)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        if pred.iter().any(|v| !v.is_finite()) {
            bail!("Starforge smoke prediction produced non-finite values");
        }
        Ok(Some((session.device_label().to_string(), pred)))
    }

    pub use Session as LoadedSession;
}

#[cfg(feature = "starforge-cpu")]
pub use enabled::LoadedSession;

#[cfg(feature = "starforge-cpu")]
pub fn load_session_for_task(
    cfg: &StarforgeConfig,
    task: TaskKind,
) -> Result<Option<LoadedSession>> {
    enabled::LoadedSession::load(cfg, task)
}

#[cfg(feature = "starforge-cpu")]
pub fn load_session(cfg: &StarforgeConfig) -> Result<Option<LoadedSession>> {
    enabled::LoadedSession::load(cfg, TaskKind::Binary)
}

#[cfg(not(feature = "starforge-cpu"))]
pub struct LoadedSession;

#[cfg(not(feature = "starforge-cpu"))]
pub fn load_session_for_task(
    cfg: &StarforgeConfig,
    _task: TaskKind,
) -> Result<Option<LoadedSession>> {
    if cfg.mode == StarforgeMode::On || cfg.require_gpu {
        bail!("feat-core was not compiled with Starforge support")
    }
    Ok(None)
}

#[cfg(not(feature = "starforge-cpu"))]
pub fn load_session(cfg: &StarforgeConfig) -> Result<Option<LoadedSession>> {
    if cfg.mode == StarforgeMode::On || cfg.require_gpu {
        bail!("feat-core was not compiled with Starforge support")
    }
    Ok(None)
}

#[cfg(feature = "starforge-cpu")]
pub fn smoke(cfg: &StarforgeConfig) -> Result<Option<(String, Vec<f32>)>> {
    enabled::smoke(cfg)
}

#[cfg(not(feature = "starforge-cpu"))]
pub fn smoke(cfg: &StarforgeConfig) -> Result<Option<(String, Vec<f32>)>> {
    if cfg.mode == StarforgeMode::On || cfg.require_gpu {
        bail!("feat-core was not compiled with Starforge support")
    }
    Ok(None)
}
