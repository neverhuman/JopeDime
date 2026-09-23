#!/usr/bin/env python3
"""jope-bench adapter: XGBoost with library defaults, `tree_method=hist`, seed 0, one thread."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if not (len(sys.argv) > 1 and sys.argv[1] == "--panel-v2"):
    import _common  # noqa: E402


def fit_predict(x, y, predict, task):
    import xgboost as xgb

    common = dict(random_state=0, n_jobs=1, tree_method="hist")
    if task == "regression":
        model = xgb.XGBRegressor(**common)
        model.fit(x, y)
        return model.predict(predict)
    if task == "binary":
        model = xgb.XGBClassifier(**common)
        model.fit(x, y.astype(int))
        return model.predict_proba(predict)[:, 1]
    classes, codes = _common.encode_classes(y)
    model = xgb.XGBClassifier(**common)
    model.fit(x, codes)
    return classes[model.predict(predict).astype(int)]


def version() -> str:
    import xgboost

    return f"xgboost {xgboost.__version__}"


def create_panel_model(task):
    import xgboost as xgb

    common = dict(random_state=0, n_jobs=1, tree_method="hist")
    return (xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier)(**common)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--panel-v2":
        import _panel

        sys.exit(_panel.run(create_panel_model, version))
    sys.exit(_common.run(fit_predict, version()))
