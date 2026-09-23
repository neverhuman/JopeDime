#!/usr/bin/env python3
"""jope-bench adapter: LightGBM with library defaults, `deterministic=True`, seed 0, one thread."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def fit_predict(x, y, predict, task):
    import lightgbm as lgb

    common = dict(random_state=0, n_jobs=1, deterministic=True, verbose=-1)
    if task == "regression":
        model = lgb.LGBMRegressor(**common)
        model.fit(x, y)
        return model.predict(predict)
    if task == "binary":
        model = lgb.LGBMClassifier(**common)
        model.fit(x, y.astype(int))
        return model.predict_proba(predict)[:, 1]
    classes, codes = _common.encode_classes(y)
    model = lgb.LGBMClassifier(**common)
    model.fit(x, codes)
    return classes[model.predict(predict).astype(int)]


def version() -> str:
    import lightgbm

    return f"lightgbm {lightgbm.__version__}"


if __name__ == "__main__":
    sys.exit(_common.run(fit_predict, version()))
