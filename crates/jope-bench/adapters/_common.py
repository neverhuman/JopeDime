"""Shared plumbing for the reference jope-bench adapters (contract jope.bench-adapter/v1).

Every adapter here uses its library's default hyperparameters, a fixed seed and one thread: no
tuning, no early stopping, no access to any held-out target. Multiclass class codes are mapped to
0..K-1 for libraries that need contiguous labels and mapped back on output.
"""
import sys

import numpy as np

TASKS = ("regression", "binary", "multiclass")


def load(train_path: str, predict_path: str):
    train = np.loadtxt(train_path, delimiter=",", ndmin=2)
    predict = np.loadtxt(predict_path, delimiter=",", ndmin=2)
    x, y = train[:, :-1], train[:, -1]
    if predict.shape[1] != x.shape[1]:
        raise SystemExit(f"predict.csv has {predict.shape[1]} columns, train.csv {x.shape[1]}")
    return x, y, predict


def encode_classes(y):
    classes = np.unique(y.astype(np.int64))
    index = {int(c): i for i, c in enumerate(classes)}
    return classes, np.array([index[int(v)] for v in y], dtype=np.int64)


def write(out_path: str, predictions) -> None:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(predictions)):
        raise SystemExit("non-finite prediction")
    with open(out_path, "w") as handle:
        for value in predictions:
            handle.write(f"{value!r}\n")


def run(fit_predict, version: str) -> int:
    """`fit_predict(x, y, predict, task) -> predictions`; handles `--version` and the arguments."""
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(version)
        return 0
    if len(sys.argv) != 5:
        sys.stderr.write("usage: <adapter> <train.csv> <predict.csv> <out.csv> <task>\n")
        return 2
    train_path, predict_path, out_path, task = sys.argv[1:5]
    if task not in TASKS:
        sys.stderr.write(f"unknown task {task!r}\n")
        return 2
    x, y, predict = load(train_path, predict_path)
    write(out_path, fit_predict(x, y, predict, task))
    return 0
