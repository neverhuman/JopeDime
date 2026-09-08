use super::Model;
#[cfg(feature = "ci-smoke")]
use super::CI_SMOKE_BACKEND;
use std::path::Path;

#[cfg(feature = "catboost")]
pub use cb::CatBoostBackend;
#[cfg(feature = "lightgbm")]
pub use lgb::LgbBackend;
#[cfg(feature = "xgboost")]
pub use xgb::XgbBackend;

#[cfg(feature = "catboost")]
mod cb {
    use super::super::{Backend, BackendKind, LossKind, Model, TrainSpec};
    use catboost::{CatBoostModel, Loss, TrainParams};

    /// Real CatBoost backend (native C++ via FFI).
    pub struct CatBoostBackend;

    struct CbModel(CatBoostModel);

    fn loss_from(l: LossKind) -> Loss {
        match l {
            LossKind::Logloss => Loss::Logloss,
            LossKind::CrossEntropy => Loss::CrossEntropy,
            LossKind::MultiLogloss => Loss::MultiLogloss,
            LossKind::MultiClass => Loss::MultiClass,
            LossKind::Rmse => Loss::Rmse,
            LossKind::Mae => Loss::Mae,
            LossKind::Quantile => Loss::Quantile,
        }
    }

    fn params(s: &TrainSpec, loss: Loss) -> TrainParams {
        TrainParams {
            iterations: s.iters,
            depth: s.depth,
            learning_rate: s.lr,
            loss,
            thread_count: s.thread_count,
            random_seed: s.seed,
        }
    }

    impl Backend for CatBoostBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::CatBoost
        }

        fn train_single(
            &self,
            x: &[f32],
            n: usize,
            p: usize,
            y: &[f32],
            s: &TrainSpec,
        ) -> Box<dyn Model> {
            let m =
                CatBoostModel::train_single(x, n, p, y, &params(s, loss_from(s.objective.loss)))
                    .expect("catboost train_single");
            Box::new(CbModel(m))
        }

        fn train_multi(
            &self,
            x: &[f32],
            n: usize,
            p: usize,
            y_multi: &[f32],
            k: usize,
            s: &TrainSpec,
        ) -> Box<dyn Model> {
            let m = CatBoostModel::train_multi(x, n, p, y_multi, k, &params(s, Loss::MultiLogloss))
                .expect("catboost train_multi");
            Box::new(CbModel(m))
        }
    }

    impl Model for CbModel {
        fn output_dim(&self) -> usize {
            self.0.output_dim()
        }

        fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32> {
            self.0.predict(x, rows, p, proba).expect("catboost predict")
        }

        fn importance(&self, p: usize) -> Vec<f32> {
            self.0.feature_importance(p).expect("catboost importance")
        }

        fn save_native(&self, path: &std::path::Path) -> Result<(), String> {
            self.0.save(path).map_err(|e| e.to_string())
        }
    }

    pub(super) fn load(path: &std::path::Path) -> anyhow::Result<Box<dyn Model>> {
        Ok(Box::new(CbModel(CatBoostModel::load(path)?)))
    }
}

#[cfg(feature = "xgboost")]
mod xgb {
    use super::super::{Backend, BackendKind, Model, TrainSpec};
    use xgboost::{XgbModel, XgbParams};

    /// XGBoost backend (native C++ via FFI). Single-target binary only.
    pub struct XgbBackend;
    struct Xm(XgbModel);

    impl Backend for XgbBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::XGBoost
        }

        fn train_single(
            &self,
            x: &[f32],
            n: usize,
            p: usize,
            y: &[f32],
            s: &TrainSpec,
        ) -> Box<dyn Model> {
            let pr = XgbParams {
                iterations: s.iters,
                depth: s.depth,
                eta: s.lr,
                thread_count: s.thread_count,
            };
            Box::new(Xm(XgbModel::train(x, n, p, y, &pr).expect("xgboost train")))
        }

        fn train_multi(
            &self,
            _x: &[f32],
            _n: usize,
            _p: usize,
            _y: &[f32],
            _k: usize,
            _s: &TrainSpec,
        ) -> Box<dyn Model> {
            panic!("xgboost backend only supports single-target training in the ensemble")
        }
    }

    impl Model for Xm {
        fn output_dim(&self) -> usize {
            1
        }

        fn predict(&self, x: &[f32], rows: usize, p: usize, _proba: bool) -> Vec<f32> {
            self.0.predict(x, rows, p).expect("xgboost predict")
        }

        fn importance(&self, p: usize) -> Vec<f32> {
            vec![0.0; p]
        }

        fn save_native(&self, path: &std::path::Path) -> Result<(), String> {
            self.0.save(path).map_err(|e| e.to_string())
        }
    }

    pub(super) fn load(path: &std::path::Path) -> anyhow::Result<Box<dyn Model>> {
        Ok(Box::new(Xm(XgbModel::load(path)?)))
    }
}

#[cfg(feature = "lightgbm")]
mod lgb {
    use super::super::{Backend, BackendKind, Model, TrainSpec};
    use lightgbm::{LgbModel, LgbParams};

    /// LightGBM backend (native C++ via FFI). Single-target binary only.
    pub struct LgbBackend;
    struct Lm(LgbModel);

    impl Backend for LgbBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::LightGbm
        }

        fn train_single(
            &self,
            x: &[f32],
            n: usize,
            p: usize,
            y: &[f32],
            s: &TrainSpec,
        ) -> Box<dyn Model> {
            let pr = LgbParams {
                iterations: s.iters,
                depth: s.depth,
                learning_rate: s.lr,
                num_leaves: 31,
                thread_count: s.thread_count,
            };
            Box::new(Lm(LgbModel::train(x, n, p, y, &pr).expect("lightgbm train")))
        }

        fn train_multi(
            &self,
            _x: &[f32],
            _n: usize,
            _p: usize,
            _y: &[f32],
            _k: usize,
            _s: &TrainSpec,
        ) -> Box<dyn Model> {
            panic!("lightgbm backend only supports single-target training in the ensemble")
        }
    }

    impl Model for Lm {
        fn output_dim(&self) -> usize {
            1
        }

        fn predict(&self, x: &[f32], rows: usize, p: usize, _proba: bool) -> Vec<f32> {
            self.0.predict(x, rows, p).expect("lightgbm predict")
        }

        fn importance(&self, p: usize) -> Vec<f32> {
            self.0.importance().unwrap_or_else(|_| vec![0.0; p])
        }

        fn save_native(&self, path: &std::path::Path) -> Result<(), String> {
            self.0.save(path).map_err(|e| e.to_string())
        }
    }

    pub(super) fn load(path: &std::path::Path) -> anyhow::Result<Box<dyn Model>> {
        Ok(Box::new(Lm(LgbModel::load(path)?)))
    }
}

pub fn load_native_model(backend: &str, path: &Path) -> anyhow::Result<Box<dyn Model>> {
    match backend {
        #[cfg(feature = "ci-smoke")]
        CI_SMOKE_BACKEND => load_ci_smoke_model(path),
        "catboost" => load_catboost_model(path),
        "xgboost" => load_xgb_model(path),
        "lightgbm" => load_lgb_model(path),
        "math" => crate::learners::math::load_math_model(path),
        other => anyhow::bail!("unsupported persisted backend '{other}'"),
    }
}

#[cfg(feature = "ci-smoke")]
fn load_ci_smoke_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    let text = std::fs::read_to_string(path)?;
    let model: super::mock::MockModel = serde_json::from_str(&text)?;
    Ok(Box::new(model))
}

#[cfg(feature = "catboost")]
fn load_catboost_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    cb::load(path)
}
#[cfg(not(feature = "catboost"))]
fn load_catboost_model(_path: &Path) -> anyhow::Result<Box<dyn Model>> {
    anyhow::bail!("catboost feature is not enabled")
}

#[cfg(feature = "xgboost")]
fn load_xgb_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    xgb::load(path)
}
#[cfg(not(feature = "xgboost"))]
fn load_xgb_model(_path: &Path) -> anyhow::Result<Box<dyn Model>> {
    anyhow::bail!("xgboost feature is not enabled")
}

#[cfg(feature = "lightgbm")]
fn load_lgb_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    lgb::load(path)
}
#[cfg(not(feature = "lightgbm"))]
fn load_lgb_model(_path: &Path) -> anyhow::Result<Box<dyn Model>> {
    anyhow::bail!("lightgbm feature is not enabled")
}
