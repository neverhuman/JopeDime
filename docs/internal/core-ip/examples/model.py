"""jope emit.

This program trains a regression model on train.csv, in which every row holds the feature
columns followed by the target as the last column, and then writes one prediction per line for
the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed evaluation
order, so repeated runs give identical bits.

The feature pipeline has 1 stage, applied in order; each stage is marked "replaced" when it
overwrites the columns it receives or "appended" when it keeps them and adds its outputs
alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
columns. Stage 1 (standard_scaler, replaced) centers each column by its training mean and
divides by the standard deviation (1/n convention, floored at 1e-9), clamping the score to [-20,
20].

The head (ridge_regression) is ridge regression (Hoerl & Kennard, 1970) on standardized columns
with penalty max(0.02*n, 1e-3); the symmetric positive-definite normal equations are solved by a
Cholesky factorization and predictions are clipped to [-60, 60]. Seeded stages draw weights from
a permuted congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed and
the input/output widths, so training is fully deterministic. Missing or malformed CSV cells
parse as 0.0 and short rows are padded with zeros to the training width.

Usage: python3 model.py train.csv test_features.csv predictions.csv
"""


import csv
import math
import sys


def _clamp(value, lo, hi):
    """Clamp value to [lo, hi]; NaN passes through (same branch order as the mirror)."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _sanitize(x):
    """Replace non-finite values with 0.0 in every row."""
    return [[v if math.isfinite(v) else 0.0 for v in row] for row in x]


def _finish_stage(x, out, append):
    """Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells."""
    if append:
        out = [x[i] + out[i] for i in range(len(out))]
    result = []
    for row in out:
        if len(row) > 512:
            row = row[:512]
        row = [v if math.isfinite(v) else 0.0 for v in row]
        if not row:
            row = [0.0]
        result.append(row)
    return result


def _std_fit(x):
    """Per-column mean and standard deviation (1/n convention), floored at 1e-9."""
    n = max(len(x), 1)
    p = len(x[0])
    means = [0.0] * p
    scales = [1.0] * p
    for j in range(p):
        s = 0.0
        for row in x:
            s += row[j]
        means[j] = s / n
        v = 0.0
        for row in x:
            d = row[j] - means[j]
            v += d * d
        scales[j] = max(math.sqrt(v / n), 1e-9)
    return means, scales


def _cholesky_solve(a, b, p):
    """Solve the symmetric positive-definite system A w = b.

    A (flattened row-major, overwritten) is factored in place as L L^T, then a
    forward and a backward triangular solve produce w. A tiny 1e-6 pivot stands
    in whenever a diagonal entry falls under 1e-12."""
    for i in range(p):
        for j in range(i + 1):
            s = a[i * p + j]
            for k in range(j):
                s -= a[i * p + k] * a[j * p + k]
            if i == j:
                a[i * p + j] = math.sqrt(s) if s > 1e-12 else 1e-6
            else:
                a[i * p + j] = s / a[j * p + j]
    z = [0.0] * p
    for i in range(p):
        s = b[i]
        for k in range(i):
            s -= a[i * p + k] * z[k]
        z[i] = s / a[i * p + i]
    w = [0.0] * p
    for i in range(p - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, p):
            s -= a[k * p + i] * w[k]
        w[i] = s / a[i * p + i]
    return w


def standard_scaler_fit(x):
    """This transform centers each column by its training mean and divides by the standard
    deviation (1/n convention, floored at 1e-9), clamping the score to [-20, 20].
    """
    return _std_fit(x)


def standard_scaler_apply(state, x, append):
    means, scales = state
    out = [[_clamp((v - means[j]) / scales[j], -20.0, 20.0) for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)


def ridge_regression_fit(x, y, lambda_frac):
    """This head is ridge regression (Hoerl & Kennard, 1970) on standardized columns with penalty
    max(0.02*n, 1e-3); the symmetric positive-definite normal equations are solved by a Cholesky
    factorization and predictions are clipped to [-60, 60].
    """
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    lam = max(lambda_frac * n, 1e-3)
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    ata = [0.0] * (p * p)
    atb = [0.0] * p
    xs_row = [0.0] * p
    for r in range(n):
        row = x[r]
        for j in range(p):
            xs_row[j] = _clamp((row[j] - means[j]) / scales[j], -20.0, 20.0)
        yr = y[r] - y_mean
        for i in range(p):
            xi = xs_row[i]
            atb[i] += xi * yr
            for j in range(i, p):
                ata[i * p + j] += xi * xs_row[j]
    for i in range(p):
        ata[i * p + i] += lam
        for j in range(i):
            ata[i * p + j] = ata[j * p + i]
    w = [y_mean] + _cholesky_solve(ata, atb, p)
    return (means, scales, w)


def ridge_regression_predict(state, x):
    means, scales, w = state
    p = len(means)
    predictions = []
    for row in x:
        z = w[0]
        for j in range(min(p, len(row))):
            xs = _clamp((row[j] - means[j]) / scales[j], -20.0, 20.0)
            z += w[j + 1] * xs
        predictions.append(_clamp(z, -60.0, 60.0))
    return predictions


class Model:
    """Feature stages applied in order, then the ridge_regression head."""

    def fit(self, x_rows: list, y: list) -> "Model":
        """Fit every stage in order on the training rows."""
        if not x_rows or len(y) < len(x_rows):
            raise SystemExit("training data is empty or target is short")
        x = _sanitize(x_rows)
        self.stage1 = standard_scaler_fit(x)
        x = standard_scaler_apply(self.stage1, x, False)
        self.head = ridge_regression_fit(x, y, 0.02)
        return self

    def predict(self, x_rows: list) -> list:
        """Apply the fitted stages, then the regression head."""
        x = _sanitize(x_rows)
        x = standard_scaler_apply(self.stage1, x, False)
        return ridge_regression_predict(self.head, x)

    def score(self, x_rows: list, y: list) -> float:
        """Coefficient of determination R^2 of predict(x_rows) against y."""
        predictions = self.predict(x_rows)
        n = min(len(y), len(predictions))
        if n == 0:
            return 0.0
        mean = 0.0
        for i in range(n):
            mean += y[i]
        mean /= n
        ss_res = 0.0
        ss_tot = 0.0
        for i in range(n):
            d = y[i] - predictions[i]
            ss_res += d * d
            t = y[i] - mean
            ss_tot += t * t
        if ss_tot <= 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot


def _parse_cell(cell):
    """Parse one CSV cell; blank, malformed, or non-finite cells become 0.0."""
    text = cell.strip()
    if not text or "_" in text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _looks_numeric(cell):
    text = cell.strip()
    if not text:
        return True
    if "_" in text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _read_csv(path):
    """Read CSV rows of floats, skipping blank lines and one header line if present."""
    records = []
    with open(path, newline="") as handle:
        for record in csv.reader(handle):
            if not record or all(not cell.strip() for cell in record):
                continue
            records.append(record)
    if records and not all(_looks_numeric(cell) for cell in records[0]):
        records = records[1:]
    return [[_parse_cell(cell) for cell in record] for record in records]


def _pad(row, width):
    """Pad short rows with 0.0 (and trim long ones) to the training feature width."""
    if len(row) < width:
        return row + [0.0] * (width - len(row))
    return row[:width]


def _write_predictions(path, predictions):
    """One repr() per line: the shortest decimal that parses back to the same float."""
    with open(path, "w") as handle:
        for value in predictions:
            handle.write(repr(value) + "\n")


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python3 model.py train.csv test_features.csv predictions.csv")
    train_rows = _read_csv(argv[0])
    if not train_rows:
        raise SystemExit("training data is empty")
    y_train = [row[-1] for row in train_rows]
    features = [row[:-1] for row in train_rows]
    width = 0
    for row in features:
        if len(row) > width:
            width = len(row)
    x_train = [_pad(row, width) for row in features]
    x_test = [_pad(row, width) for row in _read_csv(argv[1])]
    model = Model()
    model.fit(x_train, y_train)
    _write_predictions(argv[2], model.predict(x_test))


if __name__ == "__main__":
    main(sys.argv[1:])
