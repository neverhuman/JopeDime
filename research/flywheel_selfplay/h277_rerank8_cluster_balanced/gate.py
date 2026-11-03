import math
import numpy as np
import lightgbm as lgb

PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 15,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": {0: 1.0, 1: 10.0},
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def select_from_topk_scores_with_veto(
    scores_k: np.ndarray,
    p_inv_k: np.ndarray,
    tau: float,
    margin: float,
) -> int:
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_k))
    if best_j == 0:
        return 0
    if float(scores_k[best_j] - scores_k[0]) > float(margin):
        if float(p_inv_k[best_j]) > float(tau):
            return 0
        return best_j
    return 0


def fit(X, y, seed):
    model = lgb.LGBMClassifier(**dict(PARAMS, random_state=int(seed)))
    model.fit(X, y.astype(int))
    return model
