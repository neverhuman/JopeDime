#!/usr/bin/env python3
"""jope-bench adapter: CatBoost with library defaults, seed 0, one thread, no logging."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def fit_predict(x, y, predict, task):
    import catboost

    common = dict(random_seed=0, thread_count=1, verbose=False, allow_writing_files=False)
    if task == "regression":
        model = catboost.CatBoostRegressor(**common)
        model.fit(x, y)
        return model.predict(predict)
    if task == "binary":
        model = catboost.CatBoostClassifier(**common)
        model.fit(x, y.astype(int))
        return model.predict_proba(predict)[:, 1]
    classes, codes = _common.encode_classes(y)
    model = catboost.CatBoostClassifier(**common)
    model.fit(x, codes)
    return classes[model.predict(predict).reshape(-1).astype(int)]


def version() -> str:
    import catboost

    return f"catboost {catboost.__version__}"


if __name__ == "__main__":
    sys.exit(_common.run(fit_predict, version()))
