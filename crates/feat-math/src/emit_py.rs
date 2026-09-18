//! Python 3 emitter. Every snippet mirrors battle-runtime semantics exactly;
//! see the crate-level notes on cross-language seams (integer-masked u64
//! arithmetic for PCG32, guarded libm wrappers where CPython raises).

use crate::mathexpr::fmt_f64;
use crate::spec::{
    doc_paragraphs, head_fn_name, helper_set, op_fn_name, target_fn_name, wrap, Arena, Helper,
    Plan, PlannedHead, PlannedOp, PlannedStage, PlannedTarget, TargetKind,
};
use crate::transpile::EmitOptions;

pub fn emit_python(plan: &Plan, opts: &EmitOptions) -> String {
    // Sections are separated by two blank lines (PEP 8 top-level spacing); one
    // pre-reserved buffer avoids the per-section reallocation of the old path.
    let mut out = Arena::new("\n\n\n", 24 * 1024);
    out.push(&docstring(plan, opts));
    out.push("import csv\nimport math\nimport sys");
    let helpers = helper_set(plan);
    for h in &helpers {
        if *h == Helper::Pcg32 {
            out.push(&pcg32_class(&helpers));
            continue;
        }
        let src = helper_src(*h);
        if !src.is_empty() {
            out.push(src);
        }
    }
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            if let Some(target) = target {
                out.push(&target_src(target));
            }
            let mut seen: Vec<&'static str> = Vec::new();
            for (i, stage) in stages.iter().enumerate() {
                match &stage.op {
                    // Formula stages carry their own expression: one snippet
                    // per stage, named by position (never deduped).
                    PlannedOp::MapExpr { source, expr } => {
                        out.push(&map_src(i + 1, source, expr));
                    }
                    PlannedOp::DeriveExpr { source, expr } => {
                        out.push(&derive_src(i + 1, source, expr));
                    }
                    op => {
                        let name = op_fn_name(op);
                        if !seen.contains(&name) {
                            seen.push(name);
                            out.push(&op_src(op));
                        }
                    }
                }
            }
            if let PlannedHead::Portfolio(pp) = head {
                for src in portfolio_srcs(pp, &mut seen) {
                    out.push(&src);
                }
            } else {
                for src in head_srcs(head) {
                    out.push(&src);
                }
            }
            out.push(&model_class_pipeline(target.as_ref(), stages, head));
        }
        Plan::Symbolic {
            source,
            expr,
            affine,
        } => {
            out.push(&expression_fn(source, expr));
            out.push(&model_class_symbolic(*affine));
        }
    }
    out.push(IO_BLOCK);
    out.finish()
}

fn docstring(plan: &Plan, opts: &EmitOptions) -> String {
    let (title, paragraphs) = doc_paragraphs(plan, opts);
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("\"\"\"{title}."));
    for p in &paragraphs {
        lines.push(String::new());
        lines.extend(wrap(p, 96));
    }
    lines.push(String::new());
    lines.push("Usage: python3 model.py train.csv test_features.csv predictions.csv".to_string());
    if opts.fast_runtime {
        // Same interpreter semantics, same libm, same evaluation order: the
        // results are bit-identical under PyPy, just much faster.
        lines.push(
            "Tip: PyPy (pypy3 model.py ...) runs this file bit-identically, typically 5-20x faster."
                .to_string(),
        );
    }
    lines.push("\"\"\"".to_string());
    lines.join("\n")
}

// -------------------------------- helpers ---------------------------------

/// The PCG32 class with exactly the methods this genome draws through.
fn pcg32_class(helpers: &std::collections::BTreeSet<Helper>) -> String {
    let mut src = String::from(
        r#"class Pcg32:
    """Permuted congruential generator, XSH-RR 64/32 variant (O'Neill, 2014).

    All integer arithmetic is masked to 64 bits so the stream matches the
    reference implementation bit for bit."""

    def __init__(self, seed, stream):
        self.state = 0
        self.inc = ((stream << 1) | 1) & 0xFFFFFFFFFFFFFFFF
        self.next_u32()
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
        self.next_u32()

    def next_u32(self):
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF"#,
    );
    if helpers.contains(&Helper::RngUniform) {
        src.push_str(
            r#"

    def next_f64(self):
        """Uniform in [0, 1) from 53 random bits: ((hi<<32|lo) >> 11) * 2**-53."""
        hi = self.next_u32() << 32
        lo = self.next_u32()
        return ((hi | lo) >> 11) * (1.0 / 9007199254740992.0)"#,
        );
    }
    if helpers.contains(&Helper::RngRange) {
        src.push_str(
            r#"

    def range_f64(self, a, b):
        return a + (b - a) * self.next_f64()"#,
        );
    }
    if helpers.contains(&Helper::RngGaussian) {
        src.push_str(
            r#"

    def gaussian(self):
        """Standard normal via the Box-Muller cosine branch (u1 floored at 1e-12)."""
        u1 = self.next_f64()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)"#,
        );
    }
    src
}

fn helper_src(h: Helper) -> &'static str {
    match h {
        Helper::Clamp => {
            r#"def _clamp(value, lo, hi):
    """Clamp value to [lo, hi]; NaN passes through (same branch order as the mirror)."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value"#
        }
        Helper::Fmin => {
            r#"def _fmin(a, b):
    """IEEE-754 minNum: if one argument is NaN, return the other."""
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b"#
        }
        Helper::Fmax => {
            r#"def _fmax(a, b):
    """IEEE-754 maxNum: if one argument is NaN, return the other."""
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b"#
        }
        Helper::SafeSin => {
            r#"def _safe_sin(value):
    """sin() that returns NaN on an infinite argument instead of raising."""
    try:
        return math.sin(value)
    except ValueError:
        return float("nan")"#
        }
        Helper::SafeCos => {
            r#"def _safe_cos(value):
    """cos() that returns NaN on an infinite argument instead of raising."""
    try:
        return math.cos(value)
    except ValueError:
        return float("nan")"#
        }
        Helper::PowGuard => {
            r#"def _pow(base, exponent):
    """C-style pow(): NaN for a negative base with fractional exponent, +/-inf on overflow."""
    try:
        return math.pow(base, exponent)
    except OverflowError:
        if base < 0.0 and exponent == int(exponent) and int(exponent) % 2 != 0:
            return float("-inf")
        return float("inf")
    except ValueError:
        if base == 0.0 and exponent < 0.0:
            if math.copysign(1.0, base) < 0.0 and exponent == int(exponent) and int(exponent) % 2 != 0:
                return float("-inf")
            return float("inf")
        return float("nan")"#
        }
        Helper::MathDiv => {
            r#"def _div(a, b):
    """Guarded division: a denominator with magnitude under 1e-12 yields 0.0."""
    if abs(b) < 1e-12:
        return 0.0
    return a / b"#
        }
        Helper::MathLog1p => {
            r#"def _log1p(value):
    """log(1 + value), defined as 0.0 for value <= -1 where the logarithm diverges."""
    if value <= -1.0:
        return 0.0
    return math.log1p(value)"#
        }
        Helper::MathSigmoid => {
            r#"def _sigmoid(value):
    """Logistic function 1/(1+exp(-v)) with the argument clipped to [-30, 30]."""
    return 1.0 / (1.0 + math.exp(-_clamp(value, -30.0, 30.0)))"#
        }
        Helper::MathClip => {
            r#"def _clip(value, a, b):
    """Clip value to the interval spanned by a and b."""
    lo, hi = _fmin(a, b), _fmax(a, b)
    if math.isnan(lo) or math.isnan(hi):
        raise ArithmeticError("invalid clip bounds")
    return _clamp(value, lo, hi)"#
        }
        Helper::Asinh => {
            r#"def _asinh(v):
    """Inverse hyperbolic sine via the stable identity sign(v)*ln(|v| + sqrt(v*v + 1))."""
    a = abs(v)
    s = 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)
    return s * math.log(a + math.sqrt(v * v + 1.0))"#
        }
        Helper::MathExp => {
            r#"def _exp(value):
    """exp() with the argument clipped to [-30, 30], keeping the result finite."""
    return math.exp(_clamp(value, -30.0, 30.0))"#
        }
        Helper::MathLog => {
            r#"def _log(value):
    """Natural logarithm, defined as 0.0 for arguments no greater than 1e-300."""
    if value <= 1e-300:
        return 0.0
    return math.log(value)"#
        }
        Helper::Erf => {
            r#"def _erf(v):
    """Error function via the Abramowitz-Stegun 7.1.26 rational approximation."""
    a = abs(v)
    s = 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return s * (1.0 - poly * math.exp(-(a * a)))"#
        }
        Helper::Sign => {
            r#"def _sign(v):
    """Sign function: 1.0 for positive, -1.0 for negative, 0.0 at zero (and for NaN)."""
    return 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)"#
        }
        Helper::MathFloor => {
            r#"def _floor(v):
    """Floor as a float; non-finite values pass through (math.floor would raise)."""
    if not math.isfinite(v):
        return v
    return float(math.floor(v))"#
        }
        Helper::RoundHalfAway => {
            r#"def _round_half_away(v):
    """Round half away from zero (Python's builtin round is banker's); non-finite
    values pass through."""
    if not math.isfinite(v):
        return v
    if v >= 0.0:
        return float(math.floor(v + 0.5))
    return float(math.ceil(v - 0.5))"#
        }
        Helper::RowMean => {
            r#"def _row_mean(row):
    """Arithmetic mean of a row (0.0 for an empty row)."""
    total = 0.0
    for v in row:
        total += v
    return total / max(len(row), 1)"#
        }
        Helper::RowSum => {
            r#"def _row_sum(row):
    total = 0.0
    for v in row:
        total += v
    return total"#
        }
        Helper::RowL1 => {
            r#"def _row_l1(row):
    """Sum of absolute values (L1 norm) of a row."""
    total = 0.0
    for v in row:
        total += abs(v)
    return total"#
        }
        Helper::RowL2 => {
            r#"def _row_l2(row):
    """Euclidean (L2) norm of a row."""
    total = 0.0
    for v in row:
        total += v * v
    return math.sqrt(total)"#
        }
        Helper::RowMin => {
            r#"def _row_min(row):
    m = float("inf")
    for v in row:
        m = _fmin(m, v)
    return m"#
        }
        Helper::RowMax => {
            r#"def _row_max(row):
    m = float("-inf")
    for v in row:
        m = _fmax(m, v)
    return m"#
        }
        Helper::RowStd => {
            r#"def _row_std(row):
    """Standard deviation of a row with the 1/n convention."""
    mu = _row_mean(row)
    total = 0.0
    for v in row:
        d = v - mu
        total += d * d
    return math.sqrt(total / max(len(row), 1))"#
        }
        // Assembled per-genome by pcg32_class() from the helper set.
        Helper::Pcg32 | Helper::RngUniform | Helper::RngRange | Helper::RngGaussian => "",
        Helper::StageRng => {
            r#"def _stage_rng(seed, in_cols, n_out):
    """Independent random stream keyed by a stage's input and output widths."""
    return Pcg32(seed, (in_cols << 32) ^ n_out)"#
        }
        Helper::Sanitize => {
            r#"def _sanitize(x):
    """Replace non-finite values with 0.0 in every row."""
    return [[v if math.isfinite(v) else 0.0 for v in row] for row in x]"#
        }
        Helper::FinishStage => {
            r#"def _finish_stage(x, out, append):
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
    return result"#
        }
        Helper::SortedColumn => {
            r#"def _sorted_column(x, j):
    """Column j of x in ascending order."""
    return sorted(row[j] for row in x)"#
        }
        Helper::Quantile => {
            r#"def _quantile(sorted_values, q):
    """Linearly interpolated quantile of an ascending list (the numpy 'linear' rule)."""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac"#
        }
        Helper::LowerBound => {
            r#"def _lower_bound(sorted_values, v):
    """Count of elements strictly less than v (hand-rolled binary search)."""
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo"#
        }
        Helper::UpperBound => {
            r#"def _upper_bound(sorted_values, v):
    """Count of elements less than or equal to v (hand-rolled binary search)."""
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo"#
        }
        Helper::MidRank => {
            r#"def _mid_rank(sorted_values, v):
    """Average of the strict and non-strict ranks of v, scaled to [0, 1]."""
    lb = _lower_bound(sorted_values, v)
    ub = _upper_bound(sorted_values, v)
    return (lb + ub) / (2.0 * len(sorted_values))"#
        }
        Helper::Probit => {
            r#"def _probit(p):
    """Standard normal quantile function via Acklam's rational approximation."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = _clamp(p, 1e-12, 1.0 - 1e-12)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)"#
        }
        Helper::QuantileEdges => {
            r#"def _quantile_edges(x, bins):
    """Interior quantile cut points (1/bins .. (bins-1)/bins) for every column."""
    edges = []
    for j in range(len(x[0])):
        s = _sorted_column(x, j)
        edges.append([_quantile(s, b / bins) for b in range(1, bins)])
    return edges"#
        }
        Helper::Dot => {
            r#"def _dot(a, b):
    """Inner product over the shorter of the two vectors."""
    total = 0.0
    for i in range(min(len(a), len(b))):
        total += a[i] * b[i]
    return total"#
        }
        Helper::Normalize => {
            r#"def _normalize(v):
    """Scale a vector to unit length in place; a 1e-12 floor keeps zeros finite."""
    s = 0.0
    for e in v:
        s += e * e
    inv = 1.0 / max(math.sqrt(s), 1e-12)
    for i in range(len(v)):
        v[i] *= inv"#
        }
        Helper::RbfRow => {
            r#"def _rbf_row(landmarks, row, gamma):
    """Gaussian kernel similarities exp(-gamma * ||row - l||^2) to each landmark."""
    values = []
    for landmark in landmarks:
        d2 = 0.0
        for j in range(min(len(landmark), len(row))):
            d = row[j] - landmark[j]
            d2 += d * d
        values.append(math.exp(-gamma * d2))
    return values"#
        }
        Helper::DftMags => {
            r#"def _dft_magnitudes(row, k):
    """Magnitudes of the direct discrete Fourier transform at frequencies 1..k, scaled by 1/d."""
    d = len(row)
    out = []
    for f in range(1, k + 1):
        re = 0.0
        im = 0.0
        for j, v in enumerate(row):
            ang = 2.0 * math.pi * f * j / d
            re += v * math.cos(ang)
            im += v * math.sin(ang)
        out.append(math.sqrt(re * re + im * im) / d)
    return out"#
        }
        Helper::MeanOf => {
            r#"def _mean(values):
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += v
    return total / len(values)"#
        }
        Helper::Pearson => {
            r#"def _pearson(a, b):
    """Pearson correlation, 0.0 for degenerate inputs."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma = 0.0
    for i in range(n):
        ma += a[i]
    ma /= n
    mb = 0.0
    for i in range(n):
        mb += b[i]
    mb /= n
    sab = 0.0
    saa = 0.0
    sbb = 0.0
    for i in range(n):
        da = a[i] - ma
        db = b[i] - mb
        sab += da * db
        saa += da * da
        sbb += db * db
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    return sab / math.sqrt(saa * sbb)"#
        }
        Helper::Rmse => {
            r#"def _rmse(y, p):
    """Root mean squared error over the common prefix."""
    n = min(len(y), len(p))
    if n == 0:
        return 0.0
    ss = 0.0
    for i in range(n):
        d = y[i] - p[i]
        ss += d * d
    return math.sqrt(ss / n)"#
        }
        Helper::RSquared => {
            r#"def _r2_score(y, p):
    """Coefficient of determination R^2 = 1 - SS_res/SS_tot (0.0 when SS_tot <= 0)."""
    n = min(len(y), len(p))
    if n == 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += y[i]
    mean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d = y[i] - p[i]
        ss_res += d * d
        t = y[i] - mean
        ss_tot += t * t
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot"#
        }
        Helper::StdFit => {
            r#"def _std_fit(x):
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
    return means, scales"#
        }
        Helper::CholeskySolve => {
            r#"def _cholesky_solve(a, b, p):
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
    return w"#
        }
        Helper::JacobiEigen => {
            r#"def _jacobi_eigen(a, p):
    """Eigendecomposition of the symmetric p x p matrix a (flattened row-major,
    overwritten) by cyclic Jacobi rotations (Jacobi, 1846): returns
    (eigenvalues, eigenvectors) with the eigenvectors flattened row-major so
    column i holds the unit vector of eigenvalue i. Deterministic: fixed sweep
    order, at most 20 sweeps, stopping as soon as a whole-matrix check finds
    the off-diagonal mass under a 1e-12-of-trace threshold."""
    v = [0.0] * (p * p)
    for i in range(p):
        v[i * p + i] = 1.0
    scale = 0.0
    for i in range(p):
        scale += a[i * p + i]
    limit = (scale * 1e-12) * (scale * 1e-12)
    for _ in range(20):
        off = 0.0
        for i in range(p):
            for j in range(i + 1, p):
                off += a[i * p + j] * a[i * p + j]
        if off <= limit:
            break
        for i in range(p):
            for j in range(i + 1, p):
                apq = a[i * p + j]
                if apq == 0.0:
                    continue
                theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq)
                if theta >= 0.0:
                    t = 1.0 / (theta + math.sqrt(theta * theta + 1.0))
                else:
                    t = -1.0 / (math.sqrt(theta * theta + 1.0) - theta)
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(p):
                    if k != i and k != j:
                        aki = a[k * p + i]
                        akj = a[k * p + j]
                        a[k * p + i] = c * aki - s * akj
                        a[i * p + k] = a[k * p + i]
                        a[k * p + j] = s * aki + c * akj
                        a[j * p + k] = a[k * p + j]
                aii = a[i * p + i]
                ajj = a[j * p + j]
                a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj
                a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj
                a[i * p + j] = 0.0
                a[j * p + i] = 0.0
                for k in range(p):
                    vki = v[k * p + i]
                    vkj = v[k * p + j]
                    v[k * p + i] = c * vki - s * vkj
                    v[k * p + j] = s * vki + c * vkj
    e = [0.0] * p
    for i in range(p):
        e[i] = a[i * p + i]
    return e, v"#
        }
        Helper::SoftThreshold => {
            r#"def _soft_threshold(v, t):
    """Shrink v toward zero by t (the lasso proximal step)."""
    if v > t:
        return v - t
    if v < -t:
        return v + t
    return 0.0"#
        }
        Helper::TreeCore => TREE_CORE,
    }
}

const TREE_CORE: &str = r#"def _tree_bin_edges(x, f, n, bins):
    """Quantile cut points for one feature, consecutive duplicates collapsed."""
    vals = sorted(x[r][f] for r in range(n))
    edges = []
    for q in range(1, bins):
        idx = int(len(vals) * (q / bins))
        edges.append(vals[min(idx, len(vals) - 1)])
    deduped = []
    for e in edges:
        if not deduped or e != deduped[-1]:
            deduped.append(e)
    return deduped


def _tree_bin_rows(x, rows, n_feat, edges):
    """Flat row-major matrix of per-feature bin indices."""
    out = [0] * (rows * n_feat)
    for r in range(rows):
        row = x[r]
        for f in range(n_feat):
            v = row[f] if f < len(row) else 0.0
            out[r * n_feat + f] = _lower_bound(edges[f], v)
    return out


def _grow_tree(bins, grad, hess, rows, n_feat, nbins, max_depth, min_leaf, lam, depth, tree, leaf_of):
    """Greedy histogram split search; returns the index of the new node.

    Nodes are tuples (feature, threshold, left, right, value); feature -1 marks
    a leaf. Leaf values are the second-order step -g/(h + lam)."""
    g = 0.0
    for i in rows:
        g += grad[i]
    h = 0.0
    for i in rows:
        h += hess[i]
    leaf_value = -g / (h + lam)
    if depth >= max_depth or len(rows) < 2 * min_leaf:
        idx = len(tree)
        tree.append((-1, 0, 0, 0, leaf_value))
        for i in rows:
            leaf_of[i] = idx
        return idx
    best_gain = 0.0
    best_feat = -1
    best_thr = 0
    parent = g * g / (h + lam)
    for f in range(n_feat):
        gh = [0.0] * nbins
        hh = [0.0] * nbins
        for i in rows:
            b = bins[i * n_feat + f]
            gh[b] += grad[i]
            hh[b] += hess[i]
        gl = 0.0
        hl = 0.0
        for b in range(nbins - 1):
            gl += gh[b]
            hl += hh[b]
            gr = g - gl
            hr = h - hl
            if hl < 1.0 or hr < 1.0:
                continue
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thr = b
    if best_feat < 0 or best_gain <= 1e-9:
        idx = len(tree)
        tree.append((-1, 0, 0, 0, leaf_value))
        for i in rows:
            leaf_of[i] = idx
        return idx
    left_rows = []
    right_rows = []
    for i in rows:
        if bins[i * n_feat + best_feat] <= best_thr:
            left_rows.append(i)
        else:
            right_rows.append(i)
    if len(left_rows) < min_leaf or len(right_rows) < min_leaf:
        idx = len(tree)
        tree.append((-1, 0, 0, 0, leaf_value))
        for i in rows:
            leaf_of[i] = idx
        return idx
    idx = len(tree)
    tree.append((-1, 0, 0, 0, 0.0))
    left = _grow_tree(bins, grad, hess, left_rows, n_feat, nbins, max_depth, min_leaf, lam, depth + 1, tree, leaf_of)
    right = _grow_tree(bins, grad, hess, right_rows, n_feat, nbins, max_depth, min_leaf, lam, depth + 1, tree, leaf_of)
    tree[idx] = (best_feat, best_thr, left, right, 0.0)
    return idx


def _route_tree(tree, bins, r, n_feat):
    """Follow split nodes to a leaf and return its value."""
    idx = 0
    while True:
        feature, threshold, left, right, value = tree[idx]
        if feature < 0:
            return value
        idx = left if bins[r * n_feat + feature] <= threshold else right"#;

// ----------------------------- target transform ----------------------------

/// The target-transform snippet for one kind. Every arithmetic step mirrors
/// the reference target transform exactly: total-order sort, the shared
/// mid-rank + Acklam probit helpers, binary search, linear interpolation.
fn target_src(target: &PlannedTarget) -> String {
    match target.kind {
        TargetKind::RankGauss => r#"def normal_scores_target_fit(y):
    """Fit the normal-scores target transform: sort the training target and
    tabulate one (value, score) pair per distinct value, where the score is
    the standard normal quantile of the value's mid-rank."""
    sorted_values = sorted(y)
    table_values = []
    table_scores = []
    for i in range(len(sorted_values)):
        if i == 0 or sorted_values[i] > sorted_values[i - 1]:
            table_values.append(sorted_values[i])
            table_scores.append(_probit(_mid_rank(sorted_values, sorted_values[i])))
    return (sorted_values, table_values, table_scores)


def normal_scores_target_apply(state, y):
    """Map every target value to the standard normal quantile of its mid-rank."""
    sorted_values, _, _ = state
    return [_probit(_mid_rank(sorted_values, v)) for v in y]


def normal_scores_target_invert(state, z):
    """Map one prediction back to the original target scale by piecewise-linear
    interpolation through the fitted (score, value) table, clamped to the
    extreme training values."""
    _, values, scores = state
    m = len(values)
    if not (z > scores[0]):
        return values[0]
    if z >= scores[m - 1]:
        return values[m - 1]
    hi = _lower_bound(scores, z)
    lo = hi - 1
    ds = scores[hi] - scores[lo]
    if ds > 0.0:
        return values[lo] + (z - scores[lo]) / ds * (values[hi] - values[lo])
    return values[lo]"#
            .to_string(),
        TargetKind::Log1p => r#"def shifted_log_target_apply(y):
    """Map every target value through the shifted logarithm ln(1 + y), with the
    argument floored at -0.999999 so the logarithm stays finite."""
    return [math.log(1.0 + _fmax(v, -0.999999)) for v in y]


def shifted_log_target_invert(z):
    """Exact inverse exp(z) - 1, the exponent clipped to [-745, 700] so exp()
    stays within double range."""
    return math.exp(_clamp(z, -745.0, 700.0)) - 1.0"#
            .to_string(),
        TargetKind::Winsor => r#"def winsorized_target_fit(y, q):
    """Fit the winsorized target transform: the [q, 1-q] quantiles of the
    training target."""
    s = sorted(y)
    lo = _quantile(s, q)
    hi = _quantile(s, 1.0 - q)
    # interpolated quantiles on a near-constant target can cross by one ulp
    if hi < lo:
        hi = lo
    return (lo, hi)


def winsorized_target_apply(state, y):
    """Clip every target value to the fitted quantiles; the inverse is the
    identity, so winsorized predictions stay winsorized."""
    lo, hi = state
    return [_clamp(v, lo, hi) for v in y]"#
            .to_string(),
        TargetKind::ZScore => r#"def standardized_target_fit(y):
    """Fit the standardized target transform: training mean and standard
    deviation (floored at 1e-9), plus the training-range prediction clamp
    bounds (min/max widened by 5% of the range)."""
    n = float(len(y))
    mu = 0.0
    for v in y:
        mu += v
    mu /= n
    var = 0.0
    for v in y:
        d = v - mu
        var += d * d
    var /= n
    sigma = _fmax(math.sqrt(var), 1e-9)
    y_min = math.inf
    y_max = -math.inf
    for v in y:
        y_min = _fmin(y_min, v)
        y_max = _fmax(y_max, v)
    spread = _fmax(y_max - y_min, 1e-9)
    return (mu, sigma, _fmin(y_min - 0.05 * spread, -60.0), _fmax(y_max + 0.05 * spread, 60.0))


def standardized_target_apply(state, y):
    """Standardize every target value to (v - mu) / sigma."""
    mu, sigma, _, _ = state
    return [(v - mu) / sigma for v in y]


def standardized_target_invert(state, z):
    """Map one prediction back to the original target scale (z * sigma + mu),
    then clamp to the training-target range widened by 5%."""
    mu, sigma, lo, hi = state
    return _clamp(z * sigma + mu, lo, hi)"#
            .to_string(),
    }
}

/// (fit-call expression or None for the stateless kind, forward-apply call on
/// the training target, inverse expression over one prediction `z`).
fn target_calls(target: &PlannedTarget) -> (Option<String>, String, String) {
    let name = target_fn_name(target.kind);
    match target.kind {
        TargetKind::RankGauss => (
            Some(format!("{name}_fit(y)")),
            format!("{name}_apply(self.target, y)"),
            format!("{name}_invert(self.target, z)"),
        ),
        TargetKind::Log1p => (
            None,
            format!("{name}_apply(y)"),
            format!("{name}_invert(z)"),
        ),
        TargetKind::Winsor => (
            Some(format!("{name}_fit(y, {})", fmt_f64(target.q))),
            format!("{name}_apply(self.target, y)"),
            "z".to_string(),
        ),
        TargetKind::ZScore => (
            Some(format!("{name}_fit(y)")),
            format!("{name}_apply(self.target, y)"),
            format!("{name}_invert(self.target, z)"),
        ),
    }
}

// ------------------------------- transforms --------------------------------

fn op_src(op: &PlannedOp) -> String {
    let doc = crate::spec::op_doc(op);
    let doc_first = wrap(&format!("This transform {doc}"), 92);
    let mut docstring = String::new();
    for (i, line) in doc_first.iter().enumerate() {
        if i == 0 {
            docstring.push_str(&format!("    \"\"\"{line}\n"));
        } else {
            docstring.push_str(&format!("    {line}\n"));
        }
    }
    docstring.push_str("    \"\"\"");
    let body: &str = match op {
        PlannedOp::Standardize => {
            r#"def standard_scaler_fit(x):
{DOC}
    return _std_fit(x)


def standard_scaler_apply(state, x, append):
    means, scales = state
    out = [[_clamp((v - means[j]) / scales[j], -20.0, 20.0) for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RobustScale => {
            r#"def robust_scaler_fit(x):
{DOC}
    medians = []
    scales = []
    for j in range(len(x[0])):
        s = _sorted_column(x, j)
        medians.append(_quantile(s, 0.5))
        iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
        scales.append(max(iqr, 1e-9))
    return (medians, scales)


def robust_scaler_apply(state, x, append):
    medians, scales = state
    out = [[(v - medians[j]) / scales[j] for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Rank => {
            r#"def mid_rank_transform_fit(x):
{DOC}
    return [_sorted_column(x, j) for j in range(len(x[0]))]


def mid_rank_transform_apply(state, x, append):
    out = [[_mid_rank(state[j], v) for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Blom => {
            r#"def blom_normal_scores_fit(x):
{DOC}
    return [_sorted_column(x, j) for j in range(len(x[0]))]


def blom_normal_scores_apply(state, x, append):
    out = []
    for row in x:
        values = []
        for j, v in enumerate(row):
            n = len(state[j])
            mid = _mid_rank(state[j], v) * n
            values.append(_probit((mid + 0.125) / (n + 0.25)))
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::QuantileBins { .. } => {
            r#"def quantile_bin_indexer_fit(x, bins):
{DOC}
    return _quantile_edges(x, bins)


def quantile_bin_indexer_apply(edges, x, bins, append):
    out = [[_lower_bound(edges[j], v) / (bins - 1.0) for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::BinOnehot { .. } => {
            r#"def quantile_onehot_encoder_fit(x, bins):
{DOC}
    return _quantile_edges(x, bins)


def quantile_onehot_encoder_apply(edges, x, bins, append):
    out = []
    for row in x:
        encoded = [0.0] * (len(row) * bins)
        for j, v in enumerate(row):
            b = min(_lower_bound(edges[j], v), bins - 1)
            encoded[j * bins + b] = 1.0
        out.append(encoded)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Hinge { .. } => {
            r#"def hinge_basis_fit(x, knots):
{DOC}
    fitted = []
    for j in range(len(x[0])):
        s = _sorted_column(x, j)
        fitted.append([_quantile(s, i / (knots + 1)) for i in range(1, knots + 1)])
    return fitted


def hinge_basis_apply(state, x, append):
    out = []
    for row in x:
        values = []
        for j, v in enumerate(row):
            for t in state[j]:
                values.append(max(v - t, 0.0))
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::WinsorClip { .. } => {
            r#"def winsor_limiter_fit(x, q):
{DOC}
    lo = []
    hi = []
    for j in range(len(x[0])):
        s = _sorted_column(x, j)
        low = _quantile(s, q)
        high = _quantile(s, 1.0 - q)
        # interpolated quantiles on a near-constant column can cross by one ulp
        if high < low:
            high = low
        lo.append(low)
        hi.append(high)
    return (lo, hi)


def winsor_limiter_apply(state, x, append):
    lo, hi = state
    out = [[_clamp(v, lo[j], hi[j]) for j, v in enumerate(row)] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Cheb { .. } => {
            r#"def chebyshev_basis_apply(x, degree, append):
{DOC}
    out = []
    for row in x:
        values = []
        for v in row:
            u = _clamp(2.0 * v - 1.0, -1.0, 1.0)
            t_prev = 1.0
            t_cur = u
            for _ in range(degree):
                values.append(t_cur)
                t_next = 2.0 * u * t_cur - t_prev
                t_prev = t_cur
                t_cur = t_next
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::PolyInteract => {
            r#"def pairwise_products_fit(x):
{DOC}
    cols = len(x[0])
    pairs = [(i, i) for i in range(cols)]
    full = False
    for i in range(cols):
        if full:
            break
        for j in range(i + 1, cols):
            if len(pairs) >= 512:
                full = True
                break
            pairs.append((i, j))
    return pairs


def pairwise_products_apply(pairs, x, append):
    out = [[row[a] * row[b] for a, b in pairs] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RffCos { .. } => {
            r#"def random_fourier_cosine_features_fit(x, n_features, gamma, seed):
{DOC}
    cols = len(x[0])
    rng = _stage_rng(seed, cols, n_features)
    scale = math.sqrt(2.0 * gamma)
    w = [[rng.gaussian() * scale for _ in range(cols)] for _ in range(n_features)]
    b = [rng.range_f64(0.0, 2.0 * math.pi) for _ in range(n_features)]
    return (w, b)


def random_fourier_cosine_features_apply(state, x, append):
    w, b = state
    scale = math.sqrt(2.0 / len(w))
    out = [[scale * _safe_cos(_dot(w[i], row) + b[i]) for i in range(len(w))] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RffSin { .. } => {
            r#"def random_fourier_sine_features_fit(x, n_features, gamma, seed):
{DOC}
    cols = len(x[0])
    rng = _stage_rng(seed, cols, n_features)
    scale = math.sqrt(2.0 * gamma)
    w = [[rng.gaussian() * scale for _ in range(cols)] for _ in range(n_features)]
    b = [rng.range_f64(0.0, 2.0 * math.pi) for _ in range(n_features)]
    return (w, b)


def random_fourier_sine_features_apply(state, x, append):
    w, b = state
    scale = math.sqrt(2.0 / len(w))
    out = [[scale * _safe_sin(_dot(w[i], row) + b[i]) for i in range(len(w))] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RandProj { .. } => {
            r#"def gaussian_random_projection_fit(x, k, seed):
{DOC}
    cols = len(x[0])
    rng = _stage_rng(seed, cols, k)
    return [[rng.gaussian() for _ in range(cols)] for _ in range(k)]


def gaussian_random_projection_apply(w, x, append):
    inv = 1.0 / math.sqrt(len(w))
    out = [[_dot(w[i], row) * inv for i in range(len(w))] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::LandmarkDist { .. } => {
            r#"def gaussian_landmark_features_fit(x, k, seed):
{DOC}
    cols = len(x[0])
    rng = _stage_rng(seed, cols, k)
    return [[rng.range_f64(0.0, 1.0) for _ in range(cols)] for _ in range(k)]


def gaussian_landmark_features_apply(landmarks, x, gamma, append):
    out = [_rbf_row(landmarks, row, gamma) for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::ObliqueHinge { .. } => {
            r#"def oblique_hinge_features_fit(x, n_features, seed):
{DOC}
    cols = len(x[0])
    rng = _stage_rng(seed, cols, n_features)
    inv = 1.0 / math.sqrt(cols)
    w = [[rng.gaussian() * inv for _ in range(cols)] for _ in range(n_features)]
    t = []
    for wi in w:
        proj = sorted(_dot(wi, row) for row in x)
        t.append(_quantile(proj, 0.5))
    return (w, t)


def oblique_hinge_features_apply(state, x, append):
    w, t = state
    out = [[max(_dot(w[i], row) - t[i], 0.0) for i in range(len(w))] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RowStats => {
            r#"def row_summary_statistics_apply(x, append):
{DOC}
    out = []
    for row in x:
        n = len(row)
        mean = 0.0
        for v in row:
            mean += v
        mean /= n
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        energy = 0.0
        mn = float("inf")
        mx = float("-inf")
        for v in row:
            d = v - mean
            m2 += d * d
            m3 += d * d * d
            m4 += d * d * d * d
            energy += v * v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        m2 /= n
        m3 /= n
        m4 /= n
        energy /= n
        std = math.sqrt(m2)
        skew = m3 / (m2 * std) if m2 > 1e-24 else 0.0
        kurt = m4 / (m2 * m2) - 3.0 if m2 > 1e-24 else 0.0
        s = sorted(row)
        out.append([mean, std, mn, mx, _quantile(s, 0.5), _quantile(s, 0.25),
                    _quantile(s, 0.75), skew, kurt, energy])
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::PairProd { .. } => {
            r#"def correlation_screened_products_fit(x, y, top_k):
{DOC}
    cols = len(x[0])
    top_k = min(top_k, cols)
    scored = []
    for j in range(cols):
        col = [row[j] for row in x]
        scored.append((j, abs(_pearson(col, y))))
    scored.sort(key=lambda item: (0 if item[1] != item[1] else 1, -item[1], item[0]))
    selected = sorted(item[0] for item in scored[:top_k])
    pairs = []
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            pairs.append((selected[a], selected[b]))
    return pairs


def correlation_screened_products_apply(pairs, x, append):
    out = [[row[a] * row[b] for a, b in pairs] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Sigmoid => {
            r#"def logistic_squash_apply(x, append):
{DOC}
    out = [[_sigmoid(v) for v in row] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Symlog { .. } => {
            r#"def signed_log_apply(x, c, append):
{DOC}
    out = [[_asinh(v / c) for v in row] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Impute { .. } => {
            r#"def zero_imputer_fit(x, strategy, value, add_indicator):
{DOC}
    cols = len(x[0])
    fill = [0.0] * cols
    indicator_cols = []
    for j in range(cols):
        vals = []
        missing = False
        for row in x:
            if row[j] == 0.0:
                missing = True
            else:
                vals.append(row[j])
        if not vals:
            fill[j] = 0.0
        elif strategy == 0:
            s = 0.0
            for v in vals:
                s += v
            fill[j] = s / len(vals)
        elif strategy == 2:
            fill[j] = value
        else:
            vals.sort()
            fill[j] = _quantile(vals, 0.5)
        if missing and add_indicator and cols + len(indicator_cols) < 512:
            indicator_cols.append(j)
    return (fill, indicator_cols)


def zero_imputer_apply(state, x, append):
    fill, indicator_cols = state
    out = []
    for row in x:
        vals = [fill[j] if row[j] == 0.0 else row[j] for j in range(len(row))]
        for j in indicator_cols:
            vals.append(1.0 if row[j] == 0.0 else 0.0)
        out.append(vals)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Spline { .. } => {
            r#"def natural_cubic_spline_fit(x, knots):
{DOC}
    fitted = []
    for j in range(len(x[0])):
        s = _sorted_column(x, j)
        fitted.append([_quantile(s, i / (knots + 1)) for i in range(1, knots + 1)])
    return fitted


def natural_cubic_spline_apply(state, x, append):
    out = []
    for row in x:
        values = []
        for j, v in enumerate(row):
            xi = state[j]
            k = len(xi)
            if xi[k - 1] - xi[0] < 1e-9:
                values.append(v)
            else:
                values.append(v)
                pl = max(v - xi[k - 1], 0.0)
                pk = max(v - xi[k - 2], 0.0)
                dlast = (pk * pk * pk - pl * pl * pl) / max(xi[k - 1] - xi[k - 2], 1e-9)
                for t in range(k - 2):
                    pt = max(v - xi[t], 0.0)
                    d = (pt * pt * pt - pl * pl * pl) / max(xi[k - 1] - xi[t], 1e-9)
                    values.append(d - dlast)
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::PairRatio { .. } => {
            r#"def ratio_screened_pairs_fit(x, y, top_k):
{DOC}
    cols = len(x[0])
    m = min(cols, 32)
    scored = []
    for j in range(cols):
        col = [row[j] for row in x]
        scored.append((j, abs(_pearson(col, y))))
    scored.sort(key=lambda item: (0 if item[1] != item[1] else 1, -item[1], item[0]))
    screened = sorted(item[0] for item in scored[:m])
    guards = []
    for j in screened:
        s = 0.0
        for row in x:
            s += row[j]
        mean = s / len(x)
        v = 0.0
        for row in x:
            d = row[j] - mean
            v += d * d
        e = 0.01 * math.sqrt(v / len(x))
        guards.append(_fmax(e * e, 1e-12))
    ranked_pairs = []
    for ai in range(len(screened)):
        for bi in range(len(screened)):
            if ai == bi:
                continue
            a = screened[ai]
            b = screened[bi]
            eps = guards[bi]
            ratio = [row[a] * row[b] / (row[b] * row[b] + eps) for row in x]
            ranked_pairs.append((a, b, eps, abs(_pearson(ratio, y))))
    ranked_pairs.sort(key=lambda item: (0 if item[3] != item[3] else 1, -item[3], item[0], item[1]))
    kept = ranked_pairs[:min(top_k, len(ranked_pairs))]
    return ([(item[0], item[1]) for item in kept], [item[2] for item in kept])


def ratio_screened_pairs_apply(state, x, append):
    pairs, eps = state
    out = []
    for row in x:
        vals = []
        for i in range(len(pairs)):
            a, b = pairs[i]
            vals.append(row[a] * row[b] / (row[b] * row[b] + eps[i]))
        out.append(vals)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::SelectCorr { .. } => {
            r#"def column_select_fit(x, y, k, by_var):
{DOC}
    cols = len(x[0])
    k = min(k, cols)
    scored = []
    for j in range(cols):
        if by_var:
            s = 0.0
            for row in x:
                s += row[j]
            m = s / len(x)
            v = 0.0
            for row in x:
                d = row[j] - m
                v += d * d
            score = v / len(x)
        else:
            col = [row[j] for row in x]
            score = abs(_pearson(col, y))
        scored.append((j, score))
    scored.sort(key=lambda item: (0 if item[1] != item[1] else 1, -item[1], item[0]))
    return sorted(item[0] for item in scored[:k])


def column_select_apply(indices, x, append):
    out = [[row[j] for j in indices] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Step { .. } => {
            r#"def step_indicator_apply(x, t, append):
{DOC}
    out = [[1.0 if v > t else 0.0 for v in row] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::NormRow => {
            r#"def unit_norm_rows_apply(x, append):
{DOC}
    out = []
    for row in x:
        s = 0.0
        for v in row:
            s += v * v
        inv = 1.0 / max(math.sqrt(s), 1e-9)
        out.append([v * inv for v in row])
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Polar => {
            r#"def polar_coordinates_apply(x, append):
{DOC}
    out = []
    for row in x:
        d = len(row)
        values = []
        j = 0
        while j + 1 < d:
            a = row[j]
            b = row[j + 1]
            values.append(math.sqrt(a * a + b * b))
            values.append(math.atan2(b, a))
            j += 2
        if d % 2 == 1:
            values.append(row[d - 1])
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Lag { .. } => {
            r#"def shift_columns_apply(x, k, append):
{DOC}
    out = [[row[j - k] if j >= k else 0.0 for j in range(len(row))] for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Roll { .. } => {
            r#"def rotate_columns_apply(x, k, append):
{DOC}
    out = []
    for row in x:
        d = len(row)
        s = k % d
        out.append([row[(j + d - s) % d] for j in range(d)])
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::FftMag { .. } => {
            r#"def fourier_magnitudes_fit(x, k):
{DOC}
    cols = len(x[0])
    return max(1, min(k, max(cols // 2, 1)))


def fourier_magnitudes_apply(k, x, append):
    out = [_dft_magnitudes(row, k) for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RowSpectral => {
            r#"def row_spectral_summary_apply(x, append):
{DOC}
    out = []
    for row in x:
        nf = len(row) // 2
        if nf == 0:
            out.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        mags = _dft_magnitudes(row, nf)
        total = 0.0
        for m in mags:
            total += m
        denom = max(total, 1e-12)
        ent = 0.0
        m1 = 0.0
        m2 = 0.0
        peak = 0
        for i, m in enumerate(mags):
            p = m / denom
            if p > 1e-12:
                ent -= p * math.log(p)
            f = i + 1
            m1 += f * p
            m2 += f * f * p
            if m > mags[peak]:
                peak = i
        out.append([ent, m1, m2, float(peak + 1), mags[peak]])
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::RowDiffStats => {
            r#"def row_difference_statistics_apply(x, append):
{DOC}
    out = []
    for row in x:
        d = len(row)
        if d < 2:
            out.append([0.0] * 8)
            continue
        nd = d - 1
        diffs = [row[j + 1] - row[j] for j in range(nd)]
        mean = 0.0
        for v in diffs:
            mean += v
        mean /= nd
        m2 = 0.0
        mean_abs = 0.0
        energy = 0.0
        mn = float("inf")
        mx = float("-inf")
        signs = 0
        for j, v in enumerate(diffs):
            dd = v - mean
            m2 += dd * dd
            mean_abs += abs(v)
            energy += v * v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
            if j > 0 and diffs[j - 1] * v < 0.0:
                signs += 1
        std = math.sqrt(m2 / nd)
        auto = _pearson(row[:nd], row[1:])
        out.append([mean, std, mn, mx, mean_abs / nd, signs / nd, auto, energy / nd])
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::SvdProject { .. } => {
            r#"def principal_component_projection_fit(x, k, iters, seed):
{DOC}
    cols = len(x[0])
    n = len(x)
    k = min(k, cols)
    means = [0.0] * cols
    for j in range(cols):
        s = 0.0
        for row in x:
            s += row[j]
        means[j] = s / n
    rng = _stage_rng(seed, cols, k)
    components = []
    for _ in range(k):
        v = [rng.gaussian() for _ in range(cols)]
        _normalize(v)
        for _ in range(iters):
            w = [0.0] * cols
            for row in x:
                s = 0.0
                for j in range(cols):
                    s += (row[j] - means[j]) * v[j]
                for j in range(cols):
                    w[j] += s * (row[j] - means[j])
            for u in components:
                dd = _dot(w, u)
                for j in range(cols):
                    w[j] -= dd * u[j]
            _normalize(w)
            v = w
        best = 0
        for j in range(1, cols):
            if abs(v[j]) > abs(v[best]):
                best = j
        if v[best] < 0.0:
            v = [-e for e in v]
        components.append(v)
    return (means, components)


def principal_component_projection_apply(state, x, append):
    means, components = state
    out = []
    for row in x:
        values = []
        for c in components:
            s = 0.0
            for j in range(min(len(means), len(row))):
                s += (row[j] - means[j]) * c[j]
            values.append(s)
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::Nystrom { .. } => {
            r#"def nystrom_landmark_features_fit(x, k, seed):
{DOC}
    cols = len(x[0])
    n = len(x)
    k = min(k, n)
    rng = _stage_rng(seed, cols, k)
    idx = list(range(n))
    for i in range(k):
        j = i + rng.next_u32() % (n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return [list(x[idx[i]]) for i in range(k)]


def nystrom_landmark_features_apply(landmarks, x, gamma, append):
    out = [_rbf_row(landmarks, row, gamma) for row in x]
    return _finish_stage(x, out, append)"#
        }
        PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. } => {
            unreachable!("formula stages are emitted per stage index")
        }
        PlannedOp::TargetBinEncode { .. } => {
            r#"def smoothed_target_bin_encoder_fit(x, y, bins, prior):
{DOC}
    edges = _quantile_edges(x, bins)
    y_mean = 0.0
    for v in y:
        y_mean += v
    y_mean /= max(len(y), 1)
    enc = []
    for j in range(len(x[0])):
        sums = [0.0] * bins
        counts = [0.0] * bins
        for r in range(len(x)):
            b = min(_lower_bound(edges[j], x[r][j]), bins - 1)
            sums[b] += y[r]
            counts[b] += 1.0
        enc.append([(sums[b] + prior * y_mean) / max(counts[b] + prior, 1e-9) for b in range(bins)])
    return (edges, enc)


def smoothed_target_bin_encoder_apply(state, x, append):
    edges, enc = state
    out = []
    for row in x:
        values = []
        for j, v in enumerate(row):
            bins = len(enc[j])
            values.append(enc[j][min(_lower_bound(edges[j], v), bins - 1)])
        out.append(values)
    return _finish_stage(x, out, append)"#
        }
    };
    body.replace("{DOC}", &docstring)
}

/// Elementwise `map` stage: one pair of functions per stage index (the
/// expression is baked in, so stages cannot share code).
fn map_src(n: usize, source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("def elementwise_formula{n}(x, row):"));
    lines.push(format!(
        "    \"\"\"f(x, row) = {source}, with guarded arithmetic; x is one cell of row.\"\"\""
    ));
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if len(row) <= {maxc}:"));
        lines.push("        return 0.0".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    try:".to_string());
        lines.push(format!("        value = {}", expr.to_python()));
        lines.push("    except ArithmeticError:".to_string());
        lines.push("        return 0.0".to_string());
    } else {
        lines.push(format!("    value = {}", expr.to_python()));
    }
    lines.push("    if math.isfinite(value):".to_string());
    lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
    lines.push("    return 0.0".to_string());
    lines.push(String::new());
    lines.push(String::new());
    lines.push(format!("def elementwise_formula{n}_apply(x, append):"));
    lines.push(format!(
        "    \"\"\"Stage {n}: map every cell through elementwise_formula{n}.\"\"\""
    ));
    lines.push(format!(
        "    out = [[elementwise_formula{n}(v, row) for v in row] for row in x]"
    ));
    lines.push("    return _finish_stage(x, out, append)".to_string());
    lines.join("\n")
}

/// `derive` stage: reduce each row to one derived column.
fn derive_src(n: usize, source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("def derived_column{n}(row):"));
    lines.push(format!(
        "    \"\"\"g(row) = {source}, with guarded arithmetic.\"\"\""
    ));
    if features.uses_x {
        lines.push(
            "    # The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    # row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    return 0.0".to_string());
    } else {
        if let Some(maxc) = features.max_col {
            lines.push(format!("    if len(row) <= {maxc}:"));
            lines.push("        return 0.0".to_string());
        }
        if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
            lines.push("    try:".to_string());
            lines.push(format!("        value = {}", expr.to_python()));
            lines.push("    except ArithmeticError:".to_string());
            lines.push("        return 0.0".to_string());
        } else {
            lines.push(format!("    value = {}", expr.to_python()));
        }
        lines.push("    if math.isfinite(value):".to_string());
        lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
        lines.push("    return 0.0".to_string());
    }
    lines.push(String::new());
    lines.push(String::new());
    lines.push(format!("def derived_column{n}_apply(x, append):"));
    lines.push(format!(
        "    \"\"\"Stage {n}: reduce every row to the single column derived_column{n}(row).\"\"\""
    ));
    lines.push(format!("    out = [[derived_column{n}(row)] for row in x]"));
    lines.push("    return _finish_stage(x, out, append)".to_string());
    lines.join("\n")
}

// --------------------------------- heads -----------------------------------

fn head_srcs(head: &PlannedHead) -> Vec<String> {
    let doc = crate::spec::head_doc(head);
    let doc_lines = wrap(&format!("This head is {doc}"), 92);
    let mut docstring = String::new();
    for (i, line) in doc_lines.iter().enumerate() {
        if i == 0 {
            docstring.push_str(&format!("    \"\"\"{line}\n"));
        } else {
            docstring.push_str(&format!("    {line}\n"));
        }
    }
    docstring.push_str("    \"\"\"");
    head_srcs_with_doc(head, &docstring)
}

fn head_srcs_with_doc(head: &PlannedHead, docstring: &str) -> Vec<String> {
    match head {
        PlannedHead::Ridge { .. } => vec![RIDGE_SRC.replace("{DOC}", docstring)],
        PlannedHead::RidgeGcv => vec![GCV_RIDGE_SRC.replace("{DOC}", docstring)],
        PlannedHead::ElasticNet { .. } => vec![ELASTIC_SRC.replace("{DOC}", docstring)],
        PlannedHead::Gbm(_) => vec![GBM_SRC.replace("{DOC}", docstring)],
        PlannedHead::Blend { .. } => vec![
            RIDGE_SRC.replace("{DOC}", "    \"\"\"Ridge regression member of the blend (see blended_regression_fit).\"\"\""),
            GBM_SRC.replace("{DOC}", "    \"\"\"Boosted-trees member of the blend (see blended_regression_fit).\"\"\""),
            BLEND_SRC.replace("{DOC}", docstring),
        ],
        PlannedHead::HistGbm2 { .. } => vec![HISTGBM2_SRC.replace("{DOC}", docstring)],
        PlannedHead::Krr { .. } => vec![KRR_SRC.replace("{DOC}", docstring)],
        PlannedHead::Gpr { .. } => vec![GPR_SRC.replace("{DOC}", docstring)],
        PlannedHead::GprKernel { .. } => vec![GPRK_SRC.replace("{DOC}", docstring)],
        PlannedHead::Ksmooth { .. } => vec![KSMOOTH_SRC.replace("{DOC}", docstring)],
        PlannedHead::Stack => vec![
            RIDGE_SRC.replace("{DOC}", "    \"\"\"Ridge regression member of the library (see greedy_ensemble_selection_fit).\"\"\""),
            ELASTIC_SRC.replace("{DOC}", "    \"\"\"Elastic net member of the library (see greedy_ensemble_selection_fit).\"\"\""),
            HISTGBM2_SRC.replace("{DOC}", "    \"\"\"Boosted-trees member of the library (see greedy_ensemble_selection_fit).\"\"\""),
            KRR_SRC.replace("{DOC}", "    \"\"\"Kernel ridge member of the library (see greedy_ensemble_selection_fit).\"\"\""),
            stack_src().replace("{DOC}", docstring),
        ],
        // Portfolio sections depend on the resolved component roster and are
        // assembled by portfolio_srcs.
        PlannedHead::Portfolio(_) => unreachable!("portfolio sections come from portfolio_srcs"),
    }
}

// ---------------------- internally validated ensemble ----------------------

/// Every section the compact portfolio head needs, in deterministic order:
/// component transform ops (deduplicated against the outer pipeline through
/// `seen`), the projection-view op, component model heads, shared machinery
/// (row signatures, blend solver, duplicate-row average), per-component
/// fit/predict wrappers, the dispatchers, and the driver. Mirrors
/// portfolio.rs statement by statement — parity is bit-for-bit.
fn portfolio_srcs(pp: &crate::spec::PortfolioPlan, seen: &mut Vec<&'static str>) -> Vec<String> {
    use crate::spec::PlannedMemberKind;
    let mut out: Vec<String> = Vec::new();
    for member in &pp.members {
        if let PlannedMemberKind::Pipeline { stages, .. } = &member.kind {
            for stage in stages {
                let name = op_fn_name(&stage.op);
                if !seen.contains(&name) {
                    seen.push(name);
                    out.push(op_src(&stage.op));
                }
            }
        }
    }
    // The wide-data projection view (an svd_project twin, k=256 at fit time).
    let proj_op = PlannedOp::SvdProject {
        k: 256,
        iters: 20,
        seed: 7,
    };
    let proj_name = op_fn_name(&proj_op);
    if !seen.contains(&proj_name) {
        seen.push(proj_name);
        out.push(op_src(&proj_op));
    }
    let mut seen_heads: Vec<&'static str> = Vec::new();
    let mut seen_aux: Vec<&'static str> = Vec::new();
    for member in &pp.members {
        if let PlannedMemberKind::Pipeline { head, .. } = &member.kind {
            let hname = head_fn_name(head);
            if !seen_heads.contains(&hname) {
                seen_heads.push(hname);
                let doc = "    \"\"\"Component model of the internally validated ensemble (see \
                           internally_validated_ensemble_fit).\"\"\""
                    .to_string();
                for src in head_srcs_with_doc(head, &doc) {
                    out.push(py_strip_dup_aux(src, &mut seen_aux));
                }
            }
        }
    }
    out.push(PF_ROW_SIGNATURE_SRC.to_string());
    out.push(PF_BLEND_SOLVER_SRC.to_string());
    if pp
        .members
        .iter()
        .any(|m| matches!(m.kind, PlannedMemberKind::GroupMean))
    {
        out.push(PF_DUP_AVERAGE_SRC.to_string());
    }
    for (i, member) in pp.members.iter().enumerate() {
        out.push(py_component_src(i + 1, member));
    }
    out.push(py_component_dispatch(pp));
    out.push(py_portfolio_driver(pp));
    out
}

/// Head snippets self-carry two small shared helpers (_seeded_rows,
/// _chol_inverse) because a single-pipe artifact only ever holds one of
/// these heads. A portfolio can hold several, so snippets after the first
/// drop the duplicate definitions (their bodies are byte-identical across
/// snippets — asserted by the golden emission tests).
fn py_strip_dup_aux(src: String, seen_aux: &mut Vec<&'static str>) -> String {
    let mut out = src;
    for marker in ["def _seeded_rows(", "def _chol_inverse("] {
        if let Some(start) = out.find(marker) {
            if seen_aux.contains(&marker) {
                // Cut from the def line through the next top-level def.
                let next = out[start + 1..]
                    .find("\ndef ")
                    .map(|i| start + 1 + i + 1)
                    .unwrap_or(out.len());
                out.replace_range(start..next, "");
            } else {
                seen_aux.push(marker);
            }
        }
    }
    out
}

/// Short natural-language title of one component (op/head shorts joined).
fn member_title(member: &crate::spec::PlannedMember) -> String {
    match &member.kind {
        crate::spec::PlannedMemberKind::GroupMean => {
            "shrunk per-duplicate-row averages".to_string()
        }
        crate::spec::PlannedMemberKind::Pipeline { stages, head } => {
            let mut parts: Vec<&str> = Vec::new();
            for s in stages {
                parts.push(crate::spec::op_short(&s.op));
            }
            parts.push(crate::spec::head_short(head));
            parts.join(", then ")
        }
    }
}

fn py_component_src(cid: usize, member: &crate::spec::PlannedMember) -> String {
    use crate::spec::PlannedMemberKind;
    let title = member_title(member);
    let mut lines: Vec<String> = Vec::new();
    match &member.kind {
        PlannedMemberKind::GroupMean => {
            lines.push(format!("def component{cid}_fit(x, y):"));
            lines.push(format!("    \"\"\"Fit component {cid}: {title}.\"\"\""));
            lines.push("    return duplicate_row_average_fit(x, y)".to_string());
            lines.push(String::new());
            lines.push(String::new());
            lines.push(format!("def component{cid}_predict(state, x):"));
            lines.push("    return duplicate_row_average_predict(state, x)".to_string());
        }
        PlannedMemberKind::Pipeline { stages, head } => {
            lines.push(format!("def component{cid}_fit(x, y):"));
            lines.push(format!("    \"\"\"Fit component {cid}: {title}.\"\"\""));
            lines.push("    x = _sanitize(x)".to_string());
            let mut state_names: Vec<String> = Vec::new();
            for (i, stage) in stages.iter().enumerate() {
                let n = i + 1;
                let (fit_call, apply_call) = op_calls(&stage.op, n);
                let state = format!("stage{n}");
                if let Some(fit_call) = fit_call {
                    lines.push(format!("    {state} = {fit_call}"));
                    state_names.push(state.clone());
                }
                let apply = apply_call
                    .replace("{state}", &state)
                    .replace("{append}", py_bool(stage.append));
                lines.push(format!("    x = {apply}"));
            }
            let (head_fit, head_predict) = head_calls(head);
            lines.push(format!("    head = {head_fit}"));
            state_names.push("head".to_string());
            lines.push(format!("    return ({},)", state_names.join(", ")));
            lines.push(String::new());
            lines.push(String::new());
            lines.push(format!("def component{cid}_predict(state, x):"));
            lines.push(format!("    ({},) = state", state_names.join(", ")));
            lines.push("    x = _sanitize(x)".to_string());
            for (i, stage) in stages.iter().enumerate() {
                let n = i + 1;
                let (_, apply_call) = op_calls(&stage.op, n);
                let state = format!("stage{n}");
                let apply = apply_call
                    .replace("{state}", &state)
                    .replace("{append}", py_bool(stage.append));
                lines.push(format!("    x = {apply}"));
            }
            let predict = head_predict.replace("self.head", "head");
            lines.push(format!("    return {predict}"));
        }
    }
    lines.join("\n")
}

fn py_component_dispatch(pp: &crate::spec::PortfolioPlan) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("def component_fit(index, x, y):".to_string());
    lines.push("    \"\"\"Fit one component pipeline by its roster index.\"\"\"".to_string());
    for i in 1..pp.members.len() {
        lines.push(format!("    if index == {i}:"));
        lines.push(format!("        return component{i}_fit(x, y)"));
    }
    lines.push(format!(
        "    return component{}_fit(x, y)",
        pp.members.len()
    ));
    lines.push(String::new());
    lines.push(String::new());
    lines.push("def component_predict(index, state, x):".to_string());
    lines.push("    \"\"\"Predict with a fitted component by its roster index.\"\"\"".to_string());
    for i in 1..pp.members.len() {
        lines.push(format!("    if index == {i}:"));
        lines.push(format!("        return component{i}_predict(state, x)"));
    }
    lines.push(format!(
        "    return component{}_predict(state, x)",
        pp.members.len()
    ));
    lines.join("\n")
}

/// One roster line (with its activation condition) per visible component.
fn py_roster_lines(pp: &crate::spec::PortfolioPlan) -> String {
    use crate::members::Gate;
    let mut lines: Vec<String> = Vec::new();
    for (i, member) in pp.members.iter().enumerate() {
        if member.hidden {
            continue;
        }
        let cid = i + 1;
        let append = format!("roster.append(({cid}, {}))", py_bool(member.proj_view));
        match member.gate {
            Gate::Always => lines.push(format!("        {append}")),
            Gate::NAtLeast(t) => {
                lines.push(format!("        if n >= {t}:"));
                lines.push(format!("            {append}"));
            }
            Gate::NBelow(t) => {
                lines.push(format!("        if n < {t}:"));
                lines.push(format!("            {append}"));
            }
            Gate::PAtMost(t) => {
                lines.push(format!("        if p <= {t}:"));
                lines.push(format!("            {append}"));
            }
            Gate::DupFracAbove(t) => {
                lines.push(format!("        if dup > {}:", fmt_f64(t)));
                lines.push(format!("            {append}"));
            }
        }
    }
    lines.join("\n")
}

fn py_portfolio_driver(pp: &crate::spec::PortfolioPlan) -> String {
    let doc = crate::spec::head_doc(&PlannedHead::Portfolio(pp.clone()));
    let doc_lines = wrap(&format!("This head is {doc}"), 92);
    let mut docstring = String::new();
    for (i, line) in doc_lines.iter().enumerate() {
        if i == 0 {
            docstring.push_str(&format!("    \"\"\"{line}\n"));
        } else {
            docstring.push_str(&format!("    {line}\n"));
        }
    }
    docstring.push_str("    \"\"\"");
    let repeats_expr = if pp.repeats_auto {
        "3 if n < 120 else (2 if n < 300 else 1)"
    } else {
        "1"
    };
    PF_DRIVER_SRC
        .replace("{DOC}", &docstring)
        .replace("{ROSTER}", &py_roster_lines(pp))
        .replace("{FB_CID}", &(pp.fallback + 1).to_string())
        .replace("{FB_PROJ}", py_bool(pp.members[pp.fallback].proj_view))
        .replace("{REPEATS}", repeats_expr)
        .replace("{FOLDS}", &pp.folds.to_string())
}

const PF_ROW_SIGNATURE_SRC: &str = r#"def row_signature(row):
    """Row key for duplicate detection: every value rounded to nine decimals
    (half away from zero; negative zero normalized)."""
    return tuple(math.floor(v * 1e9 + 0.5) / 1e9 + 0.0 for v in row)


def duplicate_row_share(x):
    """Share of rows that duplicate an earlier row under row_signature."""
    if not x:
        return 0.0
    distinct = set()
    for row in x:
        distinct.add(row_signature(row))
    return 1.0 - len(distinct) / len(x)"#;

const PF_BLEND_SOLVER_SRC: &str = r#"def non_negative_blend_weights(g, c, k):
    """min 0.5 w'Gw - c'w subject to w >= 0 by exact cyclic coordinate descent
    (at most 500 sweeps; stops when the largest coordinate move is under 1e-10)."""
    w = [0.0] * k
    for _ in range(500):
        max_d = 0.0
        for j in range(k):
            gjj = g[j * k + j]
            if gjj <= 1e-12:
                continue
            acc = 0.0
            for l in range(k):
                acc += g[j * k + l] * w[l]
            nj = w[j] + (c[j] - acc) / gjj
            if nj < 0.0:
                nj = 0.0
            d = abs(nj - w[j])
            if d > 0.0:
                w[j] = nj
                if d > max_d:
                    max_d = d
        if max_d < 1e-10:
            break
    return w"#;

const PF_DUP_AVERAGE_SRC: &str = r#"def duplicate_row_average_fit(x, y):
    """Per-duplicate-row target averages with m-estimate shrinkage (strength 5)
    toward the global mean; unknown rows fall back to the global mean."""
    n = min(len(x), len(y))
    g_mean = 0.0
    for i in range(n):
        g_mean += y[i]
    g_mean /= max(n, 1)
    sums = {}
    counts = {}
    for i in range(n):
        key = row_signature(x[i])
        sums[key] = sums.get(key, 0.0) + y[i]
        counts[key] = counts.get(key, 0) + 1
    table = {}
    for key in sums:
        table[key] = (sums[key] + 5.0 * g_mean) / (counts[key] + 5.0)
    return (table, g_mean)


def duplicate_row_average_predict(state, x):
    table, g_mean = state
    return [table.get(row_signature(row), g_mean) for row in x]"#;

const PF_DRIVER_SRC: &str = r#"def internally_validated_ensemble_fit(x, y):
{DOC}
    n = min(len(x), len(y))
    if n == 0:
        return (True, 0.0, 0, [], None, [], 0.0, 0.0)
    width = len(x[0])
    rows = []
    for row in x[:n]:
        r = list(row[:width])
        while len(r) < width:
            r.append(0.0)
        rows.append(r)
    x = _sanitize(rows)
    y = [v if math.isfinite(v) else 0.0 for v in y[:n]]
    y_mean = 0.0
    for v in y:
        y_mean += v
    y_mean /= n
    y_var = 0.0
    for v in y:
        d = v - y_mean
        y_var += d * d
    y_var /= n
    y_std = math.sqrt(y_var)
    if n < 5 or y_std < 1e-12:
        return (True, y_mean, width, [], None, [], y_mean, y_mean)
    keep = []
    for j in range(width):
        m = 0.0
        for row in x:
            m += row[j]
        m /= n
        v = 0.0
        for row in x:
            d = row[j] - m
            v += d * d
        if math.sqrt(v / n) > 1e-12:
            keep.append(j)
    if not keep:
        return (True, y_mean, width, [], None, [], y_mean, y_mean)
    x_raw = [[row[j] for j in keep] for row in x]
    p = len(keep)
    dup = duplicate_row_share(x_raw)
    if n < 20:
        roster = [({FB_CID}, {FB_PROJ})]
    else:
        roster = []
{ROSTER}
    if not roster:
        roster = [({FB_CID}, {FB_PROJ})]
    m_count = len(roster)
    if m_count > 12:
        raise SystemExit("too many active components")
    use_projection = False
    if p > 300:
        for entry in roster:
            if entry[1]:
                use_projection = True
    proj = None
    x_proj = []
    if use_projection:
        proj = principal_component_projection_fit(x_raw, 256, 20, 7)
        x_proj = principal_component_projection_apply(proj, x_raw, False)
    if n >= 50:
        k_folds = {FOLDS}
    else:
        k_folds = max(2, min(5, n // 8))
    repeats = {REPEATS}
    rows_total = repeats * n
    held_out = [[0.0] * m_count for _ in range(rows_total)]
    for rep in range(repeats):
        idx = list(range(n))
        rng = Pcg32(17, 1000 + rep)
        for i in range(n - 1, 0, -1):
            j = rng.next_u32() % (i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        for f in range(k_folds):
            lo = f * n // k_folds
            hi = (f + 1) * n // k_folds
            if lo >= hi:
                continue
            va = idx[lo:hi]
            in_va = [False] * n
            for r in va:
                in_va[r] = True
            tr = [r for r in range(n) if not in_va[r]]
            ytr = [y[r] for r in tr]
            tr_mean = 0.0
            for v in ytr:
                tr_mean += v
            tr_mean /= len(ytr)
            f_lo = ytr[0]
            f_hi = ytr[0]
            for v in ytr:
                if v < f_lo:
                    f_lo = v
                if v > f_hi:
                    f_hi = v
            f_pad = 0.05 * max(f_hi - f_lo, 1e-12)
            xtr_raw = [x_raw[r] for r in tr]
            xva_raw = [x_raw[r] for r in va]
            xtr_proj = []
            xva_proj = []
            if use_projection:
                xtr_proj = [x_proj[r] for r in tr]
                xva_proj = [x_proj[r] for r in va]
            for mi in range(m_count):
                cid, projected = roster[mi]
                if use_projection and projected:
                    xtr_v = xtr_proj
                    xva_v = xva_proj
                else:
                    xtr_v = xtr_raw
                    xva_v = xva_raw
                state = component_fit(cid, xtr_v, ytr)
                pm = component_predict(cid, state, xva_v)
                for t, r in enumerate(va):
                    v = pm[t]
                    if not math.isfinite(v):
                        v = tr_mean
                    if v < f_lo - f_pad:
                        v = f_lo - f_pad
                    if v > f_hi + f_pad:
                        v = f_hi + f_pad
                    held_out[rep * n + r][mi] = v
    lam = 1e-3 + 4.0 * y_var * m_count / max(rows_total, 1)
    g = [0.0] * (m_count * m_count)
    c = [0.0] * m_count
    for i in range(rows_total):
        ti = y[i % n] - y_mean
        for a in range(m_count):
            pa = held_out[i][a] - y_mean
            c[a] += pa * ti
            for b in range(a, m_count):
                g[a * m_count + b] += pa * (held_out[i][b] - y_mean)
    for a in range(m_count):
        g[a * m_count + a] += lam
        for b in range(a):
            g[a * m_count + b] = g[b * m_count + a]
    w = non_negative_blend_weights(g, c, m_count)
    wsum = 0.0
    for v in w:
        wsum += v
    if wsum <= 1e-12:
        w = [1.0 / m_count] * m_count
    blend_sse = 0.0
    part_sse = [0.0] * m_count
    for i in range(rows_total):
        ti = y[i % n]
        z = y_mean
        for a in range(m_count):
            z += (held_out[i][a] - y_mean) * w[a]
        d = z - ti
        blend_sse += d * d
        for a in range(m_count):
            d = held_out[i][a] - ti
            part_sse[a] += d * d
    blend_rmse = math.sqrt(blend_sse / rows_total)
    part_rmse = [math.sqrt(s / rows_total) for s in part_sse]
    best = 0
    for a in range(1, m_count):
        if part_rmse[a] < part_rmse[best]:
            best = a
    if part_rmse[best] < blend_rmse - 1e-9:
        w = [0.0] * m_count
        w[best] = 1.0
    t_lo = y[0]
    t_hi = y[0]
    for v in y:
        if v < t_lo:
            t_lo = v
        if v > t_hi:
            t_hi = v
    pad = 0.05 * max(t_hi - t_lo, 1e-12)
    clip_lo = t_lo - pad
    clip_hi = t_hi + pad
    entries = []
    for mi in range(m_count):
        cid, projected = roster[mi]
        if w[mi] <= 1e-4:
            entries.append((cid, projected, 0.0, None))
            continue
        if use_projection and projected:
            xv = x_proj
        else:
            xv = x_raw
        state = component_fit(cid, xv, y)
        ptr = component_predict(cid, state, xv)
        sse = 0.0
        for i in range(n):
            d = ptr[i] - y[i]
            sse += d * d
        tr_rmse = math.sqrt(sse / n)
        if not math.isfinite(tr_rmse) or tr_rmse > 1.05 * y_std:
            entries.append((cid, projected, w[mi], None))
        else:
            entries.append((cid, projected, w[mi], state))
    return (False, y_mean, width, keep, proj, entries, clip_lo, clip_hi)


def internally_validated_ensemble_predict(state, x):
    """Weighted blend of the fitted components' predictions around the training
    mean; every component output and the blend are clipped to the padded
    training target range and non-finite values fall back to the mean."""
    mean_only, y_mean, width, keep, proj, entries, clip_lo, clip_hi = state
    if mean_only:
        return [y_mean] * len(x)
    rows = []
    for row in x:
        r = list(row[:width])
        while len(r) < width:
            r.append(0.0)
        rows.append(r)
    x = _sanitize(rows)
    x_raw = [[row[j] for j in keep] for row in x]
    x_proj = []
    if proj is not None:
        x_proj = principal_component_projection_apply(proj, x_raw, False)
    acc = [y_mean] * len(x)
    for entry in entries:
        cid, projected, weight, fitted = entry
        if weight <= 0.0 or fitted is None:
            continue
        if projected and proj is not None:
            xv = x_proj
        else:
            xv = x_raw
        pm = component_predict(cid, fitted, xv)
        for i in range(len(acc)):
            v = pm[i]
            if not math.isfinite(v):
                v = y_mean
            if v < clip_lo:
                v = clip_lo
            if v > clip_hi:
                v = clip_hi
            acc[i] += weight * (v - y_mean)
    out = []
    for z in acc:
        v = z
        if not math.isfinite(v):
            v = y_mean
        if v < clip_lo:
            v = clip_lo
        if v > clip_hi:
            v = clip_hi
        out.append(v)
    return out"#;

/// The stack head bakes the library defaults in as literals; they are read from
/// battle-runtime so the three implementations cannot drift apart.
fn stack_src() -> String {
    STACK_SRC
        .replace("{LEAVES}", &crate::boost2::DEFAULT_LEAVES.to_string())
        .replace("{LR}", &fmt_f64(crate::boost2::DEFAULT_LR))
        .replace("{BAGS}", &crate::boost2::DEFAULT_BAGS.to_string())
        .replace("{M}", &crate::krr::DEFAULT_M.to_string())
}

const RIDGE_SRC: &str = r#"def ridge_regression_fit(x, y, lambda_frac):
{DOC}
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
    return predictions"#;

const GCV_RIDGE_SRC: &str = r#"def gcv_ridge_regression_fit(x, y):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    for j in range(p):
        if scales[j] <= 1e-9:
            scales[j] = 1.0
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    y_var = 0.0
    for i in range(n):
        d = y[i] - y_mean
        y_var += d * d
    cols = list(range(p))
    if n < 2 or p == 0 or math.sqrt(y_var / max(n, 1)) < 1e-12:
        w = [0.0] * (p + 1)
        w[0] = y_mean
        return (cols, means, scales, w)
    if min(n, p) > 384:
        score = [0.0] * p
        for r in range(n):
            row = x[r]
            yr = y[r] - y_mean
            for j in range(p):
                score[j] += ((row[j] - means[j]) / scales[j]) * yr
        for j in range(p):
            if score[j] < 0.0:
                score[j] = -score[j]
        used = [False] * p
        sel = []
        for _ in range(384):
            best_j = 0
            best_s = -1.0
            for j in range(p):
                if not used[j] and score[j] > best_s:
                    best_s = score[j]
                    best_j = j
            used[best_j] = True
            sel.append(best_j)
        sel.sort()
        cols = sel
        p = len(cols)
        means = [means[j] for j in cols]
        scales = [scales[j] for j in cols]
    lambdas = (
        1e-05, 2.1544346900318823e-05, 4.641588833612782e-05, 0.0001,
        0.00021544346900318823, 0.00046415888336127773, 0.001,
        0.002154434690031882, 0.004641588833612777, 0.01,
        0.021544346900318822, 0.046415888336127774, 0.1,
        0.21544346900318823, 0.46415888336127725, 1.0,
        2.154434690031882, 4.641588833612772, 10.0,
        21.54434690031882, 46.41588833612773, 100.0,
        215.44346900318823, 464.1588833612773, 1000.0,
    )
    if n < p:
        xs = []
        for r in range(n):
            row = x[r]
            xs.append([(row[cols[k]] - means[k]) / scales[k] for k in range(p)])
        yc = [0.0] * n
        yty = 0.0
        for r in range(n):
            yc[r] = y[r] - y_mean
            yty += yc[r] * yc[r]
        km = [0.0] * (n * n)
        for r in range(n):
            xr = xs[r]
            for s in range(r, n):
                xo = xs[s]
                acc = 0.0
                for j in range(p):
                    acc += xr[j] * xo[j]
                km[r * n + s] = acc
        for i in range(n):
            for j in range(i):
                km[i * n + j] = km[j * n + i]
        e, q = _jacobi_eigen(km, n)
        e_max = 0.0
        for i in range(n):
            if e[i] > e_max:
                e_max = e[i]
        for i in range(n):
            if e[i] < e_max * 1e-12:
                e[i] = 0.0
        u = [0.0] * n
        for i in range(n):
            s = 0.0
            for r in range(n):
                s += q[r * n + i] * yc[r]
            u[i] = s
        best_lambda = lambdas[0]
        best_gcv = math.inf
        for lam in lambdas:
            rss = yty
            tr = 0.0
            for i in range(n):
                if e[i] > 0.0:
                    c = u[i] / (e[i] + lam)
                    f = e[i] * c
                    rss += f * f - 2.0 * f * u[i]
                    tr += e[i] / (e[i] + lam)
            if rss < 0.0:
                rss = 0.0
            denom = 1.0 - tr / n
            if denom < 1e-3:
                denom = 1e-3
            gcv = (rss / n) / (denom * denom)
            if gcv < best_gcv:
                best_gcv = gcv
                best_lambda = lam
        t = [0.0] * n
        for r in range(n):
            s = 0.0
            for i in range(n):
                if e[i] > 0.0:
                    s += q[r * n + i] * (u[i] / (e[i] + best_lambda))
            t[r] = s
        w = [y_mean]
        for j in range(p):
            s = 0.0
            for r in range(n):
                s += xs[r][j] * t[r]
            w.append(s)
        return (cols, means, scales, w)
    g = [0.0] * (p * p)
    b = [0.0] * p
    yty = 0.0
    xs_row = [0.0] * p
    for r in range(n):
        row = x[r]
        for j in range(p):
            xs_row[j] = (row[cols[j]] - means[j]) / scales[j]
        yr = y[r] - y_mean
        yty += yr * yr
        for i in range(p):
            xi = xs_row[i]
            b[i] += xi * yr
            for j in range(i, p):
                g[i * p + j] += xi * xs_row[j]
    for i in range(p):
        for j in range(i):
            g[i * p + j] = g[j * p + i]
    e, v = _jacobi_eigen(g, p)
    e_max = 0.0
    for i in range(p):
        if e[i] > e_max:
            e_max = e[i]
    for i in range(p):
        if e[i] < e_max * 1e-12:
            e[i] = 0.0
    bt = [0.0] * p
    for i in range(p):
        s = 0.0
        for j in range(p):
            s += v[j * p + i] * b[j]
        bt[i] = s
    best_lambda = lambdas[0]
    best_gcv = math.inf
    for lam in lambdas:
        rss = yty
        tr = 0.0
        for i in range(p):
            if e[i] > 0.0:
                c = bt[i] / (e[i] + lam)
                rss += c * c * e[i] - 2.0 * c * bt[i]
                tr += e[i] / (e[i] + lam)
        if rss < 0.0:
            rss = 0.0
        denom = 1.0 - tr / n
        if denom < 1e-3:
            denom = 1e-3
        gcv = (rss / n) / (denom * denom)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam
    w = [y_mean]
    for j in range(p):
        s = 0.0
        for i in range(p):
            if e[i] > 0.0:
                s += v[j * p + i] * (bt[i] / (e[i] + best_lambda))
        w.append(s)
    return (cols, means, scales, w)


def gcv_ridge_regression_predict(state, x):
    cols, means, scales, w = state
    predictions = []
    for row in x:
        z = w[0]
        for k in range(len(cols)):
            j = cols[k]
            if j < len(row):
                z += w[k + 1] * ((row[j] - means[k]) / scales[k])
        predictions.append(_clamp(z, -60.0, 60.0))
    return predictions"#;

const ELASTIC_SRC: &str = r#"def elastic_net_fit(x, y, alpha, l1_ratio, sweeps):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    xs = []
    for r in range(n):
        row = x[r]
        xs.append([_clamp((row[j] - means[j]) / scales[j], -20.0, 20.0) for j in range(p)])
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    col_sq = [0.0] * p
    for row in xs:
        for j in range(p):
            col_sq[j] += row[j] * row[j]
    w = [0.0] * p
    resid = [y[r] - y_mean for r in range(n)]
    for _ in range(max(sweeps, 1)):
        max_delta = 0.0
        for j in range(p):
            if col_sq[j] <= 1e-12:
                continue
            rho = 0.0
            for r in range(n):
                rho += xs[r][j] * resid[r]
            rho = rho / n + w[j] * col_sq[j] / n
            denom = col_sq[j] / n + l2
            new_w = _soft_threshold(rho, l1) / denom
            delta = new_w - w[j]
            if delta != 0.0:
                for r in range(n):
                    resid[r] -= delta * xs[r][j]
                w[j] = new_w
                ad = abs(delta)
                if ad > max_delta:
                    max_delta = ad
        if max_delta < 1e-9:
            break
    return (means, scales, y_mean, w)


def elastic_net_predict(state, x):
    means, scales, w0, w = state
    p = len(means)
    predictions = []
    for row in x:
        z = w0
        for j in range(min(p, len(row))):
            xs = _clamp((row[j] - means[j]) / scales[j], -20.0, 20.0)
            z += w[j] * xs
        predictions.append(_clamp(z, -60.0, 60.0))
    return predictions"#;

const GBM_SRC: &str = r#"def gradient_boosted_trees_fit(x, y, rounds, max_depth, lr, lam, bins, min_leaf, max_features):
{DOC}
    n = min(len(x), len(y))
    cols = len(x[0])
    n_feat = min(cols, max_features)
    if n < 4 or n_feat == 0:
        m = _mean(y[:min(max(n, 1), len(y))])
        return ([], [], m, lr, 0, m)
    edges = [_tree_bin_edges(x, f, n, bins) for f in range(n_feat)]
    train_bins = _tree_bin_rows(x, n, n_feat, edges)
    init = _mean(y[:n])
    f_train = [init] * n
    trees = []
    for _ in range(rounds):
        grad = [f_train[i] - y[i] for i in range(n)]
        hess = [1.0] * n
        tree = []
        train_leaf = [0] * n
        _grow_tree(train_bins, grad, hess, list(range(n)), n_feat, bins, max_depth, min_leaf, lam, 0, tree, train_leaf)
        for i in range(n):
            f_train[i] += lr * tree[train_leaf[i]][4]
        trees.append(tree)
    return (edges, trees, init, lr, n_feat, init)


def gradient_boosted_trees_predict(state, x):
    edges, trees, init, lr, n_feat, fallback = state
    if n_feat == 0:
        return [fallback] * len(x)
    bins = _tree_bin_rows(x, len(x), n_feat, edges)
    predictions = []
    for r in range(len(x)):
        value = init
        for tree in trees:
            value += lr * _route_tree(tree, bins, r, n_feat)
        predictions.append(value)
    return predictions"#;

const BLEND_SRC: &str = r#"def blended_regression_fit(x, y, lambda_frac, rounds, max_depth, lr, lam, bins, min_leaf, max_features):
{DOC}
    n = len(x)
    if n < 40:
        boosted = gradient_boosted_trees_fit(x, y, rounds, max_depth, lr, lam, bins, min_leaf, max_features)
        linear = ridge_regression_fit(x, y, lambda_frac)
        return (linear, boosted, 0.0)
    split = (n * 4) // 5
    linear_hold = ridge_regression_predict(ridge_regression_fit(x[:split], y[:split], lambda_frac), x[split:])
    boosted_hold = gradient_boosted_trees_predict(
        gradient_boosted_trees_fit(x[:split], y[:split], rounds, max_depth, lr, lam, bins, min_leaf, max_features),
        x[split:])
    best_w = 0.5
    best = float("-inf")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        blend = [w * linear_hold[i] + (1.0 - w) * boosted_hold[i] for i in range(len(linear_hold))]
        score = -_rmse(y[split:], blend)
        if score > best:
            best = score
            best_w = w
    linear = ridge_regression_fit(x, y, lambda_frac)
    boosted = gradient_boosted_trees_fit(x, y, rounds, max_depth, lr, lam, bins, min_leaf, max_features)
    return (linear, boosted, best_w)


def blended_regression_predict(state, x):
    linear, boosted, weight = state
    a = ridge_regression_predict(linear, x)
    b = gradient_boosted_trees_predict(boosted, x)
    return [weight * a[i] + (1.0 - weight) * b[i] for i in range(len(a))]"#;

const HISTGBM2_SRC: &str = r#"def _holdout_error(y, p, absolute):
    """Early-stopping criterion: mean absolute error when `absolute`, else
    root mean squared error."""
    if absolute:
        n = min(len(y), len(p))
        if n == 0:
            return 0.0
        s = 0.0
        for i in range(n):
            s += abs(y[i] - p[i])
        return s / n
    return _rmse(y, p)


def _leafwise_best_split(xcols, grad, lists, g, h, col_ok, n_feat, min_leaf, lam):
    """Exact split search over the pre-sorted rows of one leaf.

    Only boundaries between distinct values qualify; the strict > comparison
    with an ascending (feature, position) scan keeps the lowest feature and
    threshold on ties. Features with col_ok[f] False are skipped (per-tree
    column subsampling). Returns (gain, feature, threshold); gain 0 means no
    split clears the 1e-9 floor."""
    m = len(lists[0])
    if m < 2 * min_leaf:
        return (0.0, 0, 0.0)
    parent = g * g / (h + lam)
    best_gain = 1e-9
    out = (0.0, 0, 0.0)
    for f in range(n_feat):
        if not col_ok[f]:
            continue
        lst = lists[f]
        xf = xcols[f]
        gl = 0.0
        for t in range(m - 1):
            i = lst[t]
            gl += grad[i]
            cnt = t + 1
            if cnt < min_leaf:
                continue
            if m - cnt < min_leaf:
                break
            v = xf[i]
            if v < xf[lst[t + 1]]:
                hl = float(cnt)
                hr = float(m - cnt)
                gr = g - gl
                gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
                if gain > best_gain:
                    best_gain = gain
                    out = (gain, f, v)
    return out


def _leafwise_tree(xcols, grad, order, in_sample, col_ok, n_feat, max_leaves, min_leaf, lam):
    """Grow one regression tree leaf-wise (best-first): the leaf with the
    highest gain splits next. Nodes are tuples (feature, threshold, left,
    right, value); feature -1 marks a leaf holding the step -g/(h + lam)."""
    lists = []
    for f in range(n_feat):
        lists.append([i for i in order[f] if in_sample[i]])
    g = 0.0
    for i in lists[0]:
        g += grad[i]
    h = float(len(lists[0]))
    tree = [(-1, 0.0, 0, 0, -g / (h + lam))]
    gain, feat, thr = _leafwise_best_split(xcols, grad, lists, g, h, col_ok, n_feat, min_leaf, lam)
    pool = [(0, lists, gain, feat, thr)]
    leaf_count = 1
    while leaf_count < max_leaves:
        pick = -1
        pick_gain = 0.0
        for k in range(len(pool)):
            if pool[k][2] > pick_gain:
                pick_gain = pool[k][2]
                pick = k
        if pick < 0:
            break
        node, lists, gain, feat, thr = pool.pop(pick)
        xf = xcols[feat]
        left_lists = []
        right_lists = []
        for f in range(n_feat):
            left = []
            right = []
            for i in lists[f]:
                if xf[i] <= thr:
                    left.append(i)
                else:
                    right.append(i)
            left_lists.append(left)
            right_lists.append(right)
        gl = 0.0
        for i in left_lists[0]:
            gl += grad[i]
        hl = float(len(left_lists[0]))
        gr = 0.0
        for i in right_lists[0]:
            gr += grad[i]
        hr = float(len(right_lists[0]))
        li = len(tree)
        tree.append((-1, 0.0, 0, 0, -gl / (hl + lam)))
        ri = len(tree)
        tree.append((-1, 0.0, 0, 0, -gr / (hr + lam)))
        tree[node] = (feat, thr, li, ri, 0.0)
        bg, bf, bt = _leafwise_best_split(xcols, grad, left_lists, gl, hl, col_ok, n_feat, min_leaf, lam)
        pool.append((li, left_lists, bg, bf, bt))
        bg, bf, bt = _leafwise_best_split(xcols, grad, right_lists, gr, hr, col_ok, n_feat, min_leaf, lam)
        pool.append((ri, right_lists, bg, bf, bt))
        leaf_count += 1
    return tree


def _leafwise_route(tree, row):
    """Follow split nodes (go left when value <= threshold) to a leaf value."""
    idx = 0
    while True:
        feature, threshold, left, right, value = tree[idx]
        if feature < 0:
            return value
        v = row[feature] if feature < len(row) else 0.0
        idx = left if v <= threshold else right


def leafwise_boosted_trees_fit(x, y, max_leaves, lr, n_bags, rounds, row_share, col_share,
                               min_leaf, holdout_pct, shuffled_split, robust_clip, absolute_error):
{DOC}
    n = min(len(x), len(y))
    cols = len(x[0])
    n_feat = min(cols, 256)
    if n < 12 or n_feat == 0:
        m = _mean(y[:min(max(n, 1), len(y))])
        return (m, lr, [])
    n_tr = (n * (100 - holdout_pct)) // 100
    n_hold = n - n_tr
    split_idx = list(range(n))
    if shuffled_split:
        rng = Pcg32(4127, n)
        for i in range(n - 1, 0, -1):
            j = rng.next_u32() % (i + 1)
            split_idx[i], split_idx[j] = split_idx[j], split_idx[i]
    tr_idx = sorted(split_idx[:n_tr])
    hold_idx = sorted(split_idx[n_tr:])
    xtr = [x[i] for i in tr_idx]
    ytr = [y[i] for i in tr_idx]
    xh = [x[i] for i in hold_idx]
    y_hold = [y[i] for i in hold_idx]
    init = _mean(ytr)
    xcols = [[xtr[i][f] for i in range(n_tr)] for f in range(n_feat)]
    order = []
    for f in range(n_feat):
        xf = xcols[f]
        order.append(sorted(range(n_tr), key=lambda i: (xf[i], i)))
    bags = []
    for bag in range(1, n_bags + 1):
        rng = Pcg32(bag, 1442)
        f_tr = [init] * n_tr
        f_hold = [init] * n_hold
        best_err = _holdout_error(y_hold, f_hold, absolute_error)
        best_round = -1
        trees = []
        for round_index in range(rounds):
            in_sample = [rng.next_f64() < row_share for _ in range(n_tr)]
            if col_share < 1.0:
                col_ok = [rng.next_f64() < col_share for _ in range(n_feat)]
                if True not in col_ok:
                    col_ok = [True] * n_feat
            else:
                col_ok = [True] * n_feat
            if robust_clip > 0.0:
                grad = [_clamp(f_tr[i] - ytr[i], -robust_clip, robust_clip) for i in range(n_tr)]
            else:
                grad = [f_tr[i] - ytr[i] for i in range(n_tr)]
            tree = _leafwise_tree(xcols, grad, order, in_sample, col_ok, n_feat, max_leaves,
                                  min_leaf, 1.0)
            for i in range(n_tr):
                f_tr[i] += lr * _leafwise_route(tree, xtr[i])
            for hh in range(n_hold):
                f_hold[hh] += lr * _leafwise_route(tree, xh[hh])
            trees.append(tree)
            r = _holdout_error(y_hold, f_hold, absolute_error)
            if r < best_err:
                best_err = r
                best_round = round_index
            if round_index - best_round >= 50:
                break
        bags.append(trees[:best_round + 1])
    return (init, lr, bags)


def leafwise_boosted_trees_predict(state, x):
    init, lr, bags = state
    predictions = []
    for row in x:
        if not bags:
            predictions.append(init)
            continue
        acc = 0.0
        for trees in bags:
            s = 0.0
            for tree in trees:
                s += _leafwise_route(tree, row)
            acc += s
        predictions.append(init + lr * acc / len(bags))
    return predictions"#;

const KRR_SRC: &str = r#"def _squared_distance(a, b):
    """Squared Euclidean distance over the shorter of the two vectors."""
    d = 0.0
    for j in range(min(len(a), len(b))):
        t = a[j] - b[j]
        d += t * t
    return d


def _seeded_rows(n, k, seed):
    """k distinct row indices out of n via a seeded partial Fisher-Yates
    shuffle, returned in ascending order."""
    rng = Pcg32(seed, 54)
    idx = list(range(n))
    for i in range(k):
        j = i + rng.next_u32() % (n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


def kernel_ridge_regression_fit(x, y, m):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    if n < 12 or p == 0:
        return (means, scales, [], [], 0.0, y_mean)
    z = []
    for i in range(n):
        row = x[i]
        z.append([(row[j] - means[j]) / scales[j] for j in range(p)])
    n_tr = (n * 4) // 5
    n_hold = n - n_tr
    k_sel = min(n_tr, 256)
    if n_tr > 256:
        sel = _seeded_rows(n_tr, 256, 401)
    else:
        sel = list(range(n_tr))
    ys = [y[i] - y_mean for i in sel]
    d2ss = [0.0] * (k_sel * k_sel)
    for a in range(k_sel):
        za = z[sel[a]]
        for b in range(a + 1, k_sel):
            d = _squared_distance(za, z[sel[b]])
            d2ss[a * k_sel + b] = d
            d2ss[b * k_sel + a] = d
    d2hs = [0.0] * (n_hold * k_sel)
    for hh in range(n_hold):
        zh = z[n_tr + hh]
        for b in range(k_sel):
            d2hs[hh * k_sel + b] = _squared_distance(zh, z[sel[b]])
    y_hold = y[n_tr:n]
    best_score = float("inf")
    best_gamma = 0.1 / p
    best_lambda = 0.001
    for coef in (0.1, 0.3, 1.0, 3.0, 10.0):
        gamma = coef / p
        k0 = [math.exp(-gamma * d) for d in d2ss]
        kh = [math.exp(-gamma * d) for d in d2hs]
        for lam in (0.001, 0.01, 0.1, 1.0, 10.0):
            a = list(k0)
            for q in range(k_sel):
                a[q * k_sel + q] += lam
            alpha = _cholesky_solve(a, ys, k_sel)
            preds = []
            for hh in range(n_hold):
                s = y_mean
                base = hh * k_sel
                for q in range(k_sel):
                    s += kh[base + q] * alpha[q]
                preds.append(s)
            score = _rmse(y_hold, preds)
            if score < best_score:
                best_score = score
                best_gamma = gamma
                best_lambda = lam
    if n > m:
        fin = _seeded_rows(n, m, 402)
    else:
        fin = list(range(n))
    k_fin = len(fin)
    basis = [z[i] for i in fin]
    yf = [y[i] - y_mean for i in fin]
    a = [0.0] * (k_fin * k_fin)
    for i2 in range(k_fin):
        zi = basis[i2]
        for j2 in range(i2 + 1, k_fin):
            v = math.exp(-best_gamma * _squared_distance(zi, basis[j2]))
            a[i2 * k_fin + j2] = v
            a[j2 * k_fin + i2] = v
    for i2 in range(k_fin):
        a[i2 * k_fin + i2] = 1.0 + best_lambda
    alpha = _cholesky_solve(a, yf, k_fin)
    return (means, scales, basis, alpha, best_gamma, y_mean)


def kernel_ridge_regression_predict(state, x):
    means, scales, basis, alpha, gamma, y_mean = state
    p = len(means)
    predictions = []
    for row in x:
        if not basis:
            predictions.append(_clamp(y_mean, -60.0, 60.0))
            continue
        zr = [(row[j] - means[j]) / scales[j] for j in range(min(p, len(row)))]
        s = y_mean
        for b in range(len(basis)):
            s += alpha[b] * math.exp(-gamma * _squared_distance(zr, basis[b]))
        predictions.append(_clamp(s, -60.0, 60.0))
    return predictions"#;

const GPR_SRC: &str = r#"def _seeded_rows(n, k, seed):
    """k distinct row indices out of n via a seeded partial Fisher-Yates
    shuffle, returned in ascending order."""
    rng = Pcg32(seed, 54)
    idx = list(range(n))
    for i in range(k):
        j = i + rng.next_u32() % (n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


def _chol_inverse(a, m):
    """Symmetric-positive-definite inverse of a (flattened m*m, overwritten):
    factor A = L L^T in place, invert the lower-triangular L, then form
    A^-1 = L^-T L^-1. A tiny 1e-6 pivot stands in under 1e-12."""
    for i in range(m):
        for j in range(i + 1):
            s = a[i * m + j]
            for k in range(j):
                s -= a[i * m + k] * a[j * m + k]
            if i == j:
                a[i * m + j] = math.sqrt(s) if s > 1e-12 else 1e-6
            else:
                a[i * m + j] = s / a[j * m + j]
    linv = [0.0] * (m * m)
    for i in range(m):
        linv[i * m + i] = 1.0 / a[i * m + i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= a[i * m + k] * linv[k * m + j]
            linv[i * m + j] = s / a[i * m + i]
    inv = [0.0] * (m * m)
    for i in range(m):
        for j in range(i + 1):
            s = 0.0
            for k in range(i, m):
                s += linv[k * m + i] * linv[k * m + j]
            inv[i * m + j] = s
            inv[j * m + i] = s
    return inv


def gaussian_process_ard_regression_fit(x, y, m, steps):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    if n < 12 or p == 0:
        return (means, scales, [], [], [], 0.0, y_mean)
    basis_size = min(n, m)
    if n > m:
        idx = _seeded_rows(n, m, 909)
    else:
        idx = list(range(n))
    z = [[(x[i][j] - means[j]) / scales[j] for j in range(p)] for i in idx]
    yc = [y[i] - y_mean for i in idx]
    opt_size = min(basis_size, 256)
    # Seed each lengthscale from the median pairwise per-feature distance, scaled
    # by sqrt(p) so the summed kernel exponent stays O(1) as p grows.
    n_init = min(opt_size, 96)
    dim_scale = math.sqrt(p)
    log_ls = [0.0] * p
    for d in range(p):
        dists = []
        for i in range(n_init):
            for j in range(i + 1, n_init):
                dists.append(abs(z[i][d] - z[j][d]))
        dists.sort()
        med = _quantile(dists, 0.5) if dists else 1.0
        log_ls[d] = math.log(max(med, 1e-3) * dim_scale)
    var = 0.0
    for i in range(opt_size):
        var += yc[i] * yc[i]
    sy = math.sqrt(var / max(opt_size, 1))
    log_sf = math.log(max(sy, 1e-6))
    log_sn = math.log(max(0.1 * sy, 1e-6))
    # Gradient ascent on the log marginal likelihood over the first opt_size rows.
    for _ in range(steps):
        mo = opt_size
        ls_cur = [math.exp(u) for u in log_ls]
        inv_ls2 = [1.0 / (v * v) for v in ls_cur]
        sf2 = math.exp(2.0 * log_sf)
        sn2 = math.exp(2.0 * log_sn)
        kmat = [0.0] * (mo * mo)
        for i in range(mo):
            kmat[i * mo + i] = sf2
            zi = z[i]
            for j in range(i + 1, mo):
                zj = z[j]
                r2 = 0.0
                for d in range(p):
                    diff = zi[d] - zj[d]
                    r2 += diff * diff * inv_ls2[d]
                v = sf2 * math.exp(-0.5 * r2)
                kmat[i * mo + j] = v
                kmat[j * mo + i] = v
        a = list(kmat)
        for i in range(mo):
            a[i * mo + i] += sn2
        kinv = _chol_inverse(a, mo)
        alpha = [0.0] * mo
        for i in range(mo):
            s = 0.0
            base = i * mo
            for j in range(mo):
                s += kinv[base + j] * yc[j]
            alpha[i] = s
        g_sf = 0.0
        g_sn = 0.0
        g_ls = [0.0] * p
        for i in range(mo):
            wii = alpha[i] * alpha[i] - kinv[i * mo + i]
            g_sf += wii * kmat[i * mo + i]
            g_sn += wii
            zi = z[i]
            ai = alpha[i]
            for j in range(i + 1, mo):
                wij = ai * alpha[j] - kinv[i * mo + j]
                kij = kmat[i * mo + j]
                g_sf += 2.0 * wij * kij
                c = wij * kij
                zj = z[j]
                for d in range(p):
                    diff = zi[d] - zj[d]
                    g_ls[d] += c * diff * diff * inv_ls2[d]
        g_sn *= sn2
        scale = 1.0 / mo
        for d in range(p):
            step = _clamp(scale * g_ls[d], -0.5, 0.5)
            log_ls[d] = _clamp(log_ls[d] + step, -6.0, 6.0)
        step_sf = _clamp(scale * g_sf, -0.5, 0.5)
        log_sf = _clamp(log_sf + step_sf, -15.0, 15.0)
        step_sn = _clamp(scale * g_sn, -0.5, 0.5)
        log_sn = _clamp(log_sn + step_sn, log_sf - 6.0, log_sf)
    ls_final = [math.exp(u) for u in log_ls]
    inv_ls2 = [1.0 / (v * v) for v in ls_final]
    sf2 = math.exp(2.0 * log_sf)
    sn2 = math.exp(2.0 * log_sn)
    mb = basis_size
    a = [0.0] * (mb * mb)
    for i in range(mb):
        a[i * mb + i] = sf2 + sn2
        zi = z[i]
        for j in range(i + 1, mb):
            zj = z[j]
            r2 = 0.0
            for d in range(p):
                diff = zi[d] - zj[d]
                r2 += diff * diff * inv_ls2[d]
            v = sf2 * math.exp(-0.5 * r2)
            a[i * mb + j] = v
            a[j * mb + i] = v
    alpha = _cholesky_solve(a, yc, mb)
    return (means, scales, z, alpha, ls_final, sf2, y_mean)


def gaussian_process_ard_regression_predict(state, x):
    means, scales, basis, alpha, ls, sf2, y_mean = state
    p = len(means)
    if not basis:
        return [_clamp(y_mean, -60.0, 60.0) for _ in x]
    inv_ls2 = [1.0 / (v * v) for v in ls]
    predictions = []
    for row in x:
        zr = [(row[j] - means[j]) / scales[j] for j in range(min(p, len(row)))]
        s = y_mean
        for b in range(len(basis)):
            zb = basis[b]
            r2 = 0.0
            for d in range(min(len(zr), len(zb))):
                diff = zr[d] - zb[d]
                r2 += diff * diff * inv_ls2[d]
            s += alpha[b] * sf2 * math.exp(-0.5 * r2)
        predictions.append(_clamp(s, -60.0, 60.0))
    return predictions"#;

const GPRK_SRC: &str = r#"def _seeded_rows(n, k, seed):
    """k distinct row indices out of n via a seeded partial Fisher-Yates
    shuffle, returned in ascending order."""
    rng = Pcg32(seed, 54)
    idx = list(range(n))
    for i in range(k):
        j = i + rng.next_u32() % (n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


def _chol_inverse(a, m):
    """Symmetric-positive-definite inverse of a (flattened m*m, overwritten):
    factor A = L L^T in place, invert the lower-triangular L, then form
    A^-1 = L^-T L^-1. A tiny 1e-6 pivot stands in under 1e-12."""
    for i in range(m):
        for j in range(i + 1):
            s = a[i * m + j]
            for k in range(j):
                s -= a[i * m + k] * a[j * m + k]
            if i == j:
                a[i * m + j] = math.sqrt(s) if s > 1e-12 else 1e-6
            else:
                a[i * m + j] = s / a[j * m + j]
    linv = [0.0] * (m * m)
    for i in range(m):
        linv[i * m + i] = 1.0 / a[i * m + i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= a[i * m + k] * linv[k * m + j]
            linv[i * m + j] = s / a[i * m + i]
    inv = [0.0] * (m * m)
    for i in range(m):
        for j in range(i + 1):
            s = 0.0
            for k in range(i, m):
                s += linv[k * m + i] * linv[k * m + j]
            inv[i * m + j] = s
            inv[j * m + i] = s
    return inv


def _kernel_decode(kernel_terms, theta, p):
    """Decode the flat hyperparameter vector: per sum term [log amplitude,
    factor shape parameters...], then the global log noise std last. Returns
    (per-term variances, per-factor inverse squared lengthscales, per-factor
    rational-quadratic alphas, per-factor periods, noise variance)."""
    sf2s = []
    inv2 = []
    alphas = []
    periods = []
    off = 0
    for term in kernel_terms:
        sf2s.append(math.exp(2.0 * theta[off]))
        off += 1
        inv_t = []
        alpha_t = []
        period_t = []
        for kind in term:
            if kind == "squared_exponential":
                inv_f = []
                for d in range(p):
                    length = math.exp(theta[off + d])
                    inv_f.append(1.0 / (length * length))
                inv_t.append(inv_f)
                alpha_t.append(0.0)
                period_t.append(0.0)
                off += p
            elif kind == "rational_quadratic":
                inv_f = []
                for d in range(p):
                    length = math.exp(theta[off + d])
                    inv_f.append(1.0 / (length * length))
                inv_t.append(inv_f)
                alpha_t.append(math.exp(theta[off + p]))
                period_t.append(0.0)
                off += p + 1
            elif kind == "periodic":
                period_t.append(math.exp(theta[off]))
                length = math.exp(theta[off + 1])
                inv_t.append([1.0 / (length * length)])
                alpha_t.append(0.0)
                off += 2
            else:
                inv_t.append([])
                alpha_t.append(0.0)
                period_t.append(0.0)
        inv2.append(inv_t)
        alphas.append(alpha_t)
        periods.append(period_t)
    sn2 = math.exp(2.0 * theta[off])
    return (sf2s, inv2, alphas, periods, sn2)


def _kernel_value(kernel_terms, dec, pf, za, zb):
    """Covariance k(za, zb): the sum over terms of the term amplitude times
    the product of its factor values, in declaration order."""
    sf2s, inv2, alphas, periods, _ = dec
    dm = min(len(za), len(zb))
    total = 0.0
    for t in range(len(kernel_terms)):
        term = kernel_terms[t]
        val = sf2s[t]
        for f in range(len(term)):
            kind = term[f]
            if kind == "squared_exponential":
                inv = inv2[t][f]
                r2 = 0.0
                for d in range(dm):
                    diff = za[d] - zb[d]
                    r2 += diff * diff * inv[d]
                val *= math.exp(-0.5 * r2)
            elif kind == "rational_quadratic":
                inv = inv2[t][f]
                r2 = 0.0
                for d in range(dm):
                    diff = za[d] - zb[d]
                    r2 += diff * diff * inv[d]
                al = alphas[t][f]
                s = r2 / (2.0 * al)
                val *= math.exp(-al * math.log(1.0 + s))
            elif kind == "periodic":
                per = periods[t][f]
                inv0 = inv2[t][f][0]
                s2 = 0.0
                for d in range(dm):
                    ang = math.pi * (za[d] - zb[d]) / per
                    sa = math.sin(ang)
                    s2 += sa * sa
                val *= math.exp(-2.0 * s2 * inv0)
            else:
                dot = 0.0
                for d in range(dm):
                    dot += za[d] * zb[d]
                val *= dot / pf
        total += val
    return total


def gaussian_process_composite_kernel_regression_fit(x, y, m, steps, kernel_terms):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    if n < 12 or p == 0:
        return (means, scales, [], [], [], y_mean)
    pf = float(p)
    basis_size = min(n, m)
    if n > m:
        idx = _seeded_rows(n, m, 909)
    else:
        idx = list(range(n))
    z = [[(x[i][j] - means[j]) / scales[j] for j in range(p)] for i in idx]
    yc = [y[i] - y_mean for i in idx]
    opt_size = min(basis_size, 256)
    # Lengthscale seed for every distance factor: the median pairwise
    # per-feature distance scaled by sqrt(p), so the summed kernel exponent
    # stays O(1) as the feature count grows.
    n_init = min(opt_size, 96)
    dim_scale = math.sqrt(p)
    ard_init = [0.0] * p
    for d in range(p):
        dists = []
        for i in range(n_init):
            for j in range(i + 1, n_init):
                dists.append(abs(z[i][d] - z[j][d]))
        dists.sort()
        med = _quantile(dists, 0.5) if dists else 1.0
        ard_init[d] = math.log(max(med, 1e-3) * dim_scale)
    var = 0.0
    for i in range(opt_size):
        var += yc[i] * yc[i]
    sy = math.sqrt(var / max(opt_size, 1))
    # Flat hyperparameter vector with per-slot box clamps; term amplitudes
    # split the target variance so the summed prior variance stays sy^2.
    amp_init = math.log(max(sy, 1e-6)) - 0.5 * math.log(len(kernel_terms))
    theta = []
    lo = []
    hi = []
    amp_idx = []
    for term in kernel_terms:
        amp_idx.append(len(theta))
        theta.append(amp_init)
        lo.append(-15.0)
        hi.append(15.0)
        for kind in term:
            if kind == "squared_exponential":
                for d in range(p):
                    theta.append(ard_init[d])
                    lo.append(-6.0)
                    hi.append(6.0)
            elif kind == "rational_quadratic":
                for d in range(p):
                    theta.append(ard_init[d])
                    lo.append(-6.0)
                    hi.append(6.0)
                theta.append(0.0)
                lo.append(-3.0)
                hi.append(5.0)
            elif kind == "periodic":
                theta.append(math.log(2.0))
                lo.append(-3.0)
                hi.append(3.0)
                theta.append(0.5 * math.log(2.0 * pf))
                lo.append(-6.0)
                hi.append(6.0)
    noise_idx = len(theta)
    theta.append(math.log(max(0.1 * sy, 1e-6)))
    # Gradient ascent on the log marginal likelihood over the first opt_size
    # rows; dK/dtheta uses per-term / per-factor chain rules.
    for _ in range(steps):
        mo = opt_size
        dec = _kernel_decode(kernel_terms, theta, p)
        sf2s, inv2, alphas, periods, sn2 = dec
        a = [0.0] * (mo * mo)
        for i in range(mo):
            a[i * mo + i] = _kernel_value(kernel_terms, dec, pf, z[i], z[i]) + sn2
            for j in range(i + 1, mo):
                v = _kernel_value(kernel_terms, dec, pf, z[i], z[j])
                a[i * mo + j] = v
                a[j * mo + i] = v
        kinv = _chol_inverse(a, mo)
        alpha = [0.0] * mo
        for i in range(mo):
            s = 0.0
            base = i * mo
            for j in range(mo):
                s += kinv[base + j] * yc[j]
            alpha[i] = s
        g = [0.0] * len(theta)
        for i in range(mo):
            wii = alpha[i] * alpha[i] - kinv[i * mo + i]
            g[noise_idx] += wii
            # Diagonal: distance-based factor values are exactly 1 there, so
            # only the term amplitudes (and linear factors) contribute.
            off = 0
            for t in range(len(kernel_terms)):
                term = kernel_terms[t]
                amp = off
                off += 1
                val = sf2s[t]
                for kind in term:
                    if kind == "squared_exponential":
                        off += p
                    elif kind == "rational_quadratic":
                        off += p + 1
                    elif kind == "periodic":
                        off += 2
                    else:
                        dot = 0.0
                        for d in range(p):
                            dot += z[i][d] * z[i][d]
                        val *= dot / pf
                g[amp] += wii * val
            zi = z[i]
            for j in range(i + 1, mo):
                wij = alpha[i] * alpha[j] - kinv[i * mo + j]
                zj = z[j]
                off = 0
                for t in range(len(kernel_terms)):
                    term = kernel_terms[t]
                    amp = off
                    nf = len(term)
                    r2s = [0.0] * nf
                    s2s = [0.0] * nf
                    psums = [0.0] * nf
                    val = sf2s[t]
                    for f in range(nf):
                        kind = term[f]
                        if kind == "squared_exponential":
                            inv = inv2[t][f]
                            r2 = 0.0
                            for d in range(p):
                                diff = zi[d] - zj[d]
                                r2 += diff * diff * inv[d]
                            r2s[f] = r2
                            val *= math.exp(-0.5 * r2)
                        elif kind == "rational_quadratic":
                            inv = inv2[t][f]
                            r2 = 0.0
                            for d in range(p):
                                diff = zi[d] - zj[d]
                                r2 += diff * diff * inv[d]
                            r2s[f] = r2
                            al = alphas[t][f]
                            s = r2 / (2.0 * al)
                            val *= math.exp(-al * math.log(1.0 + s))
                        elif kind == "periodic":
                            per = periods[t][f]
                            inv0 = inv2[t][f][0]
                            s2 = 0.0
                            psum = 0.0
                            for d in range(p):
                                ang = math.pi * (zi[d] - zj[d]) / per
                                sa = math.sin(ang)
                                ca = math.cos(ang)
                                s2 += sa * sa
                                psum += ang * sa * ca
                            s2s[f] = s2
                            psums[f] = psum
                            val *= math.exp(-2.0 * s2 * inv0)
                        else:
                            dot = 0.0
                            for d in range(p):
                                dot += zi[d] * zj[d]
                            val *= dot / pf
                    g[amp] += 2.0 * wij * val
                    c = wij * val
                    o2 = amp + 1
                    for f in range(nf):
                        kind = term[f]
                        if kind == "squared_exponential":
                            inv = inv2[t][f]
                            for d in range(p):
                                diff = zi[d] - zj[d]
                                g[o2 + d] += c * diff * diff * inv[d]
                            o2 += p
                        elif kind == "rational_quadratic":
                            inv = inv2[t][f]
                            al = alphas[t][f]
                            s = r2s[f] / (2.0 * al)
                            denom = 1.0 + s
                            for d in range(p):
                                diff = zi[d] - zj[d]
                                g[o2 + d] += c * diff * diff * inv[d] / denom
                            g[o2 + p] += c * al * (s / denom - math.log(denom))
                            o2 += p + 1
                        elif kind == "periodic":
                            inv0 = inv2[t][f][0]
                            g[o2] += c * 4.0 * inv0 * psums[f]
                            g[o2 + 1] += c * 4.0 * s2s[f] * inv0
                            o2 += 2
                    off = o2
        scale = 1.0 / mo
        for q in range(noise_idx):
            step = _clamp(scale * g[q], -0.5, 0.5)
            theta[q] = _clamp(theta[q] + step, lo[q], hi[q])
        # Noise floored at exp(-6) of the LARGEST term amplitude (the same
        # conditioning jitter as a single-kernel Gaussian process).
        g_sn = g[noise_idx] * sn2
        max_amp = float("-inf")
        for ai in amp_idx:
            if theta[ai] > max_amp:
                max_amp = theta[ai]
        step_sn = _clamp(scale * g_sn, -0.5, 0.5)
        theta[noise_idx] = _clamp(theta[noise_idx] + step_sn, max_amp - 6.0, max_amp)
    dec = _kernel_decode(kernel_terms, theta, p)
    sn2 = dec[4]
    mb = basis_size
    a = [0.0] * (mb * mb)
    for i in range(mb):
        a[i * mb + i] = _kernel_value(kernel_terms, dec, pf, z[i], z[i]) + sn2
        for j in range(i + 1, mb):
            v = _kernel_value(kernel_terms, dec, pf, z[i], z[j])
            a[i * mb + j] = v
            a[j * mb + i] = v
    alpha = _cholesky_solve(a, yc, mb)
    return (means, scales, z, alpha, theta, y_mean)


def gaussian_process_composite_kernel_regression_predict(state, x, kernel_terms):
    means, scales, basis, alpha, theta, y_mean = state
    p = len(means)
    if not basis:
        return [_clamp(y_mean, -60.0, 60.0) for _ in x]
    pf = float(p)
    dec = _kernel_decode(kernel_terms, theta, p)
    predictions = []
    for row in x:
        zr = [(row[j] - means[j]) / scales[j] for j in range(min(p, len(row)))]
        s = y_mean
        for b in range(len(basis)):
            s += alpha[b] * _kernel_value(kernel_terms, dec, pf, zr, basis[b])
        predictions.append(_clamp(s, -60.0, 60.0))
    return predictions"#;

const KSMOOTH_SRC: &str = r#"def nadaraya_watson_regression_fit(x, y):
{DOC}
    n = min(len(x), len(y))
    p = len(x[0])
    means, scales = _std_fit(x)
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= max(n, 1)
    if n < 8 or p == 0:
        return (means, scales, [], [], y_mean)
    z = [[(x[i][j] - means[j]) / scales[j] for j in range(p)] for i in range(n)]
    ys = [y[i] for i in range(n)]
    return (means, scales, z, ys, y_mean)


def nadaraya_watson_regression_predict(state, x, k, gamma):
    means, scales, z, ys, y_mean = state
    p = len(means)
    if not z:
        return [_clamp(y_mean, -60.0, 60.0) for _ in x]
    pf = float(p)
    kk = min(k, len(z))
    predictions = []
    for row in x:
        zr = [(row[j] - means[j]) / scales[j] for j in range(min(p, len(row)))]
        d2s = []
        for zb in z:
            d2 = 0.0
            for d in range(min(len(zr), len(zb))):
                t = zr[d] - zb[d]
                d2 += t * t
            d2s.append(d2)
        order = sorted(range(len(z)), key=lambda i: (d2s[i], i))
        base = d2s[order[0]]
        num = 0.0
        den = 0.0
        for q in range(kk):
            i = order[q]
            w = math.exp(-gamma * (d2s[i] - base) / pf)
            num += w * ys[i]
            den += w
        predictions.append(_clamp(num / den, -60.0, 60.0))
    return predictions"#;

const STACK_SRC: &str = r#"def _ensemble_member_holdout(b, xtr, ytr, xva):
    """Fit library member b on the fold-train rows and predict the held fold."""
    if b == 0:
        return ridge_regression_predict(ridge_regression_fit(xtr, ytr, 0.02), xva)
    if b == 1:
        return ridge_regression_predict(ridge_regression_fit(xtr, ytr, 0.002), xva)
    if b == 2:
        return elastic_net_predict(elastic_net_fit(xtr, ytr, 0.001, 0.5, 60), xva)
    if b == 3:
        return leafwise_boosted_trees_predict(
            leafwise_boosted_trees_fit(xtr, ytr, {LEAVES}, {LR}, {BAGS}, 500, 0.7, 1.0, 3, 20, False, 0.0, False), xva)
    return kernel_ridge_regression_predict(kernel_ridge_regression_fit(xtr, ytr, {M}), xva)


def greedy_ensemble_selection_fit(x, y):
{DOC}
    n = min(len(x), len(y))
    if n < 20:
        return ([1.0, 0.0, 0.0, 0.0, 0.0], ridge_regression_fit(x, y, 0.02), None, None, None, None)
    oof = [[0.0] * n for _ in range(5)]
    for f in range(5):
        lo = f * n // 5
        hi = (f + 1) * n // 5
        if lo >= hi:
            continue
        xtr = [x[r] for r in range(lo)] + [x[r] for r in range(hi, n)]
        ytr = [y[r] for r in range(lo)] + [y[r] for r in range(hi, n)]
        xva = [x[r] for r in range(lo, hi)]
        for b in range(5):
            preds = _ensemble_member_holdout(b, xtr, ytr, xva)
            for k in range(hi - lo):
                oof[b][lo + k] = preds[k]
    yn = y[:n]
    counts = [0, 0, 0, 0, 0]
    total = 0
    sum_vec = [0.0] * n
    current = float("-inf")
    for _ in range(20):
        best_b = -1
        best_score = current
        for b in range(5):
            denom = float(total + 1)
            ob = oof[b]
            blend = [(sum_vec[i] + ob[i]) / denom for i in range(n)]
            s = _r2_score(yn, blend)
            if s > best_score:
                best_score = s
                best_b = b
        if best_b < 0:
            break
        ob = oof[best_b]
        for i in range(n):
            sum_vec[i] += ob[i]
        counts[best_b] += 1
        total += 1
        current = best_score
    if total == 0:
        counts[0] = 1
        total = 1
    weights = [c / total for c in counts]
    ridge_a = ridge_regression_fit(x, y, 0.02) if counts[0] > 0 else None
    ridge_b = ridge_regression_fit(x, y, 0.002) if counts[1] > 0 else None
    net = elastic_net_fit(x, y, 0.001, 0.5, 60) if counts[2] > 0 else None
    trees = leafwise_boosted_trees_fit(
        x, y, {LEAVES}, {LR}, {BAGS}, 500, 0.7, 1.0, 3, 20, False, 0.0, False) if counts[3] > 0 else None
    kernel = kernel_ridge_regression_fit(x, y, {M}) if counts[4] > 0 else None
    return (weights, ridge_a, ridge_b, net, trees, kernel)


def greedy_ensemble_selection_predict(state, x):
    weights, ridge_a, ridge_b, net, trees, kernel = state
    acc = [0.0] * len(x)
    if ridge_a is not None:
        p = ridge_regression_predict(ridge_a, x)
        for i in range(len(x)):
            acc[i] += weights[0] * p[i]
    if ridge_b is not None:
        p = ridge_regression_predict(ridge_b, x)
        for i in range(len(x)):
            acc[i] += weights[1] * p[i]
    if net is not None:
        p = elastic_net_predict(net, x)
        for i in range(len(x)):
            acc[i] += weights[2] * p[i]
    if trees is not None:
        p = leafwise_boosted_trees_predict(trees, x)
        for i in range(len(x)):
            acc[i] += weights[3] * p[i]
    if kernel is not None:
        p = kernel_ridge_regression_predict(kernel, x)
        for i in range(len(x)):
            acc[i] += weights[4] * p[i]
    return acc"#;

// ------------------------------- model class -------------------------------

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Python literal for the composite-GP kernel structure, e.g.
/// `[["squared_exponential", "periodic"], ["linear"]]`.
fn py_kernel_terms(spec: &crate::gpr::KernelSpec) -> String {
    let terms: Vec<String> = spec
        .iter()
        .map(|term| {
            let factors: Vec<String> = term
                .iter()
                .map(|p| format!("\"{}\"", crate::spec::kernel_prim_tag(*p)))
                .collect();
            format!("[{}]", factors.join(", "))
        })
        .collect();
    format!("[{}]", terms.join(", "))
}

/// (fit-call expression or None for stateless, apply-call with `{state}` and
/// `{append}` placeholders); `n` is the 1-based stage index (used by the
/// per-stage formula functions).
fn op_calls(op: &PlannedOp, n: usize) -> (Option<String>, String) {
    match op {
        PlannedOp::Standardize => (
            Some("standard_scaler_fit(x)".into()),
            "standard_scaler_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RobustScale => (
            Some("robust_scaler_fit(x)".into()),
            "robust_scaler_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Rank => (
            Some("mid_rank_transform_fit(x)".into()),
            "mid_rank_transform_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Blom => (
            Some("blom_normal_scores_fit(x)".into()),
            "blom_normal_scores_apply({state}, x, {append})".into(),
        ),
        PlannedOp::QuantileBins { bins } => (
            Some(format!("quantile_bin_indexer_fit(x, {bins})")),
            format!("quantile_bin_indexer_apply({{state}}, x, {bins}, {{append}})"),
        ),
        PlannedOp::BinOnehot { bins } => (
            Some(format!("quantile_onehot_encoder_fit(x, {bins})")),
            format!("quantile_onehot_encoder_apply({{state}}, x, {bins}, {{append}})"),
        ),
        PlannedOp::Hinge { knots } => (
            Some(format!("hinge_basis_fit(x, {knots})")),
            "hinge_basis_apply({state}, x, {append})".into(),
        ),
        PlannedOp::WinsorClip { q } => (
            Some(format!("winsor_limiter_fit(x, {})", fmt_f64(*q))),
            "winsor_limiter_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Cheb { degree } => (
            None,
            format!("chebyshev_basis_apply(x, {degree}, {{append}})"),
        ),
        PlannedOp::PolyInteract => (
            Some("pairwise_products_fit(x)".into()),
            "pairwise_products_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RffCos { n, gamma, seed } => (
            Some(format!(
                "random_fourier_cosine_features_fit(x, {n}, {}, {seed})",
                fmt_f64(*gamma)
            )),
            "random_fourier_cosine_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RffSin { n, gamma, seed } => (
            Some(format!(
                "random_fourier_sine_features_fit(x, {n}, {}, {seed})",
                fmt_f64(*gamma)
            )),
            "random_fourier_sine_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RandProj { k, seed } => (
            Some(format!("gaussian_random_projection_fit(x, {k}, {seed})")),
            "gaussian_random_projection_apply({state}, x, {append})".into(),
        ),
        PlannedOp::LandmarkDist { k, gamma, seed } => (
            Some(format!("gaussian_landmark_features_fit(x, {k}, {seed})")),
            format!(
                "gaussian_landmark_features_apply({{state}}, x, {}, {{append}})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedOp::ObliqueHinge { n, seed } => (
            Some(format!("oblique_hinge_features_fit(x, {n}, {seed})")),
            "oblique_hinge_features_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RowStats => (None, "row_summary_statistics_apply(x, {append})".into()),
        PlannedOp::PairProd { topk } => (
            Some(format!("correlation_screened_products_fit(x, y, {topk})")),
            "correlation_screened_products_apply({state}, x, {append})".into(),
        ),
        PlannedOp::TargetBinEncode { bins, prior } => (
            Some(format!(
                "smoothed_target_bin_encoder_fit(x, y, {bins}, {})",
                fmt_f64(*prior)
            )),
            "smoothed_target_bin_encoder_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Sigmoid => (None, "logistic_squash_apply(x, {append})".into()),
        PlannedOp::Symlog { c } => (
            None,
            format!("signed_log_apply(x, {}, {{append}})", fmt_f64(*c)),
        ),
        PlannedOp::SelectCorr { k, by_var } => (
            Some(format!(
                "column_select_fit(x, y, {k}, {})",
                py_bool(*by_var)
            )),
            "column_select_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Impute {
            strategy,
            value,
            add_indicator,
        } => (
            Some(format!(
                "zero_imputer_fit(x, {strategy}, {}, {})",
                fmt_f64(*value),
                py_bool(*add_indicator)
            )),
            "zero_imputer_apply({state}, x, {append})".into(),
        ),
        PlannedOp::PairRatio { topk } => (
            Some(format!("ratio_screened_pairs_fit(x, y, {topk})")),
            "ratio_screened_pairs_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Spline { knots } => (
            Some(format!("natural_cubic_spline_fit(x, {knots})")),
            "natural_cubic_spline_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Step { t } => (
            None,
            format!("step_indicator_apply(x, {}, {{append}})", fmt_f64(*t)),
        ),
        PlannedOp::NormRow => (None, "unit_norm_rows_apply(x, {append})".into()),
        PlannedOp::Polar => (None, "polar_coordinates_apply(x, {append})".into()),
        PlannedOp::Lag { k } => (None, format!("shift_columns_apply(x, {k}, {{append}})")),
        PlannedOp::Roll { k } => (None, format!("rotate_columns_apply(x, {k}, {{append}})")),
        PlannedOp::FftMag { k } => (
            Some(format!("fourier_magnitudes_fit(x, {k})")),
            "fourier_magnitudes_apply({state}, x, {append})".into(),
        ),
        PlannedOp::RowSpectral => (None, "row_spectral_summary_apply(x, {append})".into()),
        PlannedOp::RowDiffStats => (None, "row_difference_statistics_apply(x, {append})".into()),
        PlannedOp::SvdProject { k, iters, seed } => (
            Some(format!(
                "principal_component_projection_fit(x, {k}, {iters}, {seed})"
            )),
            "principal_component_projection_apply({state}, x, {append})".into(),
        ),
        PlannedOp::Nystrom { k, gamma, seed } => (
            Some(format!("nystrom_landmark_features_fit(x, {k}, {seed})")),
            format!(
                "nystrom_landmark_features_apply({{state}}, x, {}, {{append}})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedOp::MapExpr { .. } => (None, format!("elementwise_formula{n}_apply(x, {{append}})")),
        PlannedOp::DeriveExpr { .. } => (None, format!("derived_column{n}_apply(x, {{append}})")),
    }
}

fn head_calls(head: &PlannedHead) -> (String, String) {
    match head {
        PlannedHead::Ridge { lambda_frac } => (
            format!("ridge_regression_fit(x, y, {})", fmt_f64(*lambda_frac)),
            "ridge_regression_predict(self.head, x)".into(),
        ),
        PlannedHead::RidgeGcv => (
            "gcv_ridge_regression_fit(x, y)".into(),
            "gcv_ridge_regression_predict(self.head, x)".into(),
        ),
        PlannedHead::ElasticNet {
            alpha,
            l1_ratio,
            sweeps,
        } => (
            format!(
                "elastic_net_fit(x, y, {}, {}, {sweeps})",
                fmt_f64(*alpha),
                fmt_f64(*l1_ratio)
            ),
            "elastic_net_predict(self.head, x)".into(),
        ),
        PlannedHead::Gbm(g) => (
            format!(
                "gradient_boosted_trees_fit(x, y, {}, {}, {}, {}, {}, {}, {})",
                g.rounds,
                g.depth,
                fmt_f64(g.lr),
                fmt_f64(g.lambda),
                g.bins,
                g.min_leaf,
                g.max_features
            ),
            "gradient_boosted_trees_predict(self.head, x)".into(),
        ),
        PlannedHead::Blend { lambda_frac, gbm } => (
            format!(
                "blended_regression_fit(x, y, {}, {}, {}, {}, {}, {}, {}, {})",
                fmt_f64(*lambda_frac),
                gbm.rounds,
                gbm.depth,
                fmt_f64(gbm.lr),
                fmt_f64(gbm.lambda),
                gbm.bins,
                gbm.min_leaf,
                gbm.max_features
            ),
            "blended_regression_predict(self.head, x)".into(),
        ),
        PlannedHead::HistGbm2 {
            leaves,
            lr,
            bags,
            rounds,
            row_subsample,
            col_subsample,
            min_leaf,
            val_pct,
            val_shuffle,
            huber_delta,
            stop_mae,
        } => (
            format!(
                "leafwise_boosted_trees_fit(x, y, {leaves}, {}, {bags}, {rounds}, {}, {}, \
                 {min_leaf}, {val_pct}, {}, {}, {})",
                fmt_f64(*lr),
                fmt_f64(*row_subsample),
                fmt_f64(*col_subsample),
                py_bool(*val_shuffle),
                fmt_f64(*huber_delta),
                py_bool(*stop_mae)
            ),
            "leafwise_boosted_trees_predict(self.head, x)".into(),
        ),
        PlannedHead::Krr { m } => (
            format!("kernel_ridge_regression_fit(x, y, {m})"),
            "kernel_ridge_regression_predict(self.head, x)".into(),
        ),
        PlannedHead::Gpr { m, steps } => (
            format!("gaussian_process_ard_regression_fit(x, y, {m}, {steps})"),
            "gaussian_process_ard_regression_predict(self.head, x)".into(),
        ),
        PlannedHead::GprKernel { m, steps, spec, .. } => {
            let terms = py_kernel_terms(spec);
            (
                format!(
                    "gaussian_process_composite_kernel_regression_fit(x, y, {m}, {steps}, {terms})"
                ),
                format!(
                    "gaussian_process_composite_kernel_regression_predict(self.head, x, {terms})"
                ),
            )
        }
        PlannedHead::Ksmooth { k, gamma } => (
            "nadaraya_watson_regression_fit(x, y)".into(),
            format!(
                "nadaraya_watson_regression_predict(self.head, x, {k}, {})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedHead::Stack => (
            "greedy_ensemble_selection_fit(x, y)".into(),
            "greedy_ensemble_selection_predict(self.head, x)".into(),
        ),
        PlannedHead::Portfolio(_) => (
            "internally_validated_ensemble_fit(x, y)".into(),
            "internally_validated_ensemble_predict(self.head, x)".into(),
        ),
    }
}

fn model_class_pipeline(
    target: Option<&PlannedTarget>,
    stages: &[PlannedStage],
    head: &PlannedHead,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("class Model:".to_string());
    if target.is_some() {
        lines.push(format!(
            "    \"\"\"A fitted target transform, feature stages applied in order, then the {} \
             head; predictions are mapped back through the transform's inverse.\"\"\"",
            head_fn_name(head)
        ));
    } else {
        lines.push(format!(
            "    \"\"\"Feature stages applied in order, then the {} head.\"\"\"",
            head_fn_name(head)
        ));
    }
    lines.push(String::new());
    lines.push("    def fit(self, x_rows: list, y: list) -> \"Model\":".to_string());
    lines.push("        \"\"\"Fit every stage in order on the training rows.\"\"\"".to_string());
    lines.push("        if not x_rows or len(y) < len(x_rows):".to_string());
    lines.push(
        "            raise SystemExit(\"training data is empty or target is short\")".to_string(),
    );
    lines.push("        x = _sanitize(x_rows)".to_string());
    if let Some(target) = target {
        // The transform is fitted on the raw train target FIRST; every stage
        // and the head see only the transformed target.
        let (fit_call, apply_call, _) = target_calls(target);
        if let Some(fit_call) = fit_call {
            lines.push(format!("        self.target = {fit_call}"));
        }
        lines.push(format!("        y = {apply_call}"));
    }
    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let (fit_call, apply_call) = op_calls(&stage.op, n);
        let state = format!("self.stage{n}");
        if let Some(fit_call) = fit_call {
            lines.push(format!("        {state} = {fit_call}"));
        }
        let apply = apply_call
            .replace("{state}", &state)
            .replace("{append}", py_bool(stage.append));
        lines.push(format!("        x = {apply}"));
    }
    let (head_fit, _) = head_calls(head);
    lines.push(format!("        self.head = {head_fit}"));
    lines.push("        return self".to_string());
    lines.push(String::new());
    lines.push("    def predict(self, x_rows: list) -> list:".to_string());
    lines
        .push("        \"\"\"Apply the fitted stages, then the regression head.\"\"\"".to_string());
    lines.push("        x = _sanitize(x_rows)".to_string());
    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let (_, apply_call) = op_calls(&stage.op, n);
        let state = format!("self.stage{n}");
        let apply = apply_call
            .replace("{state}", &state)
            .replace("{append}", py_bool(stage.append));
        lines.push(format!("        x = {apply}"));
    }
    let (_, head_predict) = head_calls(head);
    match target {
        None => lines.push(format!("        return {head_predict}")),
        // Exact inverse back to the original target scale, then the kind's
        // final prediction clamp — the same order as the reference model.
        // Range-aware kinds carry their train-range clamp inside the inverse.
        Some(target) => {
            let (_, _, invert) = target_calls(target);
            lines.push(format!("        predictions = {head_predict}"));
            if crate::spec::target_range_aware(target.kind) {
                lines.push(format!("        return [{invert} for z in predictions]"));
            } else {
                lines.push(format!(
                    "        return [_clamp({invert}, -60.0, 60.0) for z in predictions]"
                ));
            }
        }
    }
    lines.push(String::new());
    lines.extend(py_score_method());
    lines.join("\n")
}

/// The `score` method (coefficient of determination) shared by both Model
/// classes; self-contained so it needs no helper.
fn py_score_method() -> Vec<String> {
    [
        "    def score(self, x_rows: list, y: list) -> float:",
        "        \"\"\"Coefficient of determination R^2 of predict(x_rows) against y.\"\"\"",
        "        predictions = self.predict(x_rows)",
        "        n = min(len(y), len(predictions))",
        "        if n == 0:",
        "            return 0.0",
        "        mean = 0.0",
        "        for i in range(n):",
        "            mean += y[i]",
        "        mean /= n",
        "        ss_res = 0.0",
        "        ss_tot = 0.0",
        "        for i in range(n):",
        "            d = y[i] - predictions[i]",
        "            ss_res += d * d",
        "            t = y[i] - mean",
        "            ss_tot += t * t",
        "        if ss_tot <= 0.0:",
        "            return 0.0",
        "        return 1.0 - ss_res / ss_tot",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// -------------------------------- symbolic ---------------------------------

fn expression_fn(source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push("def expression_value(row):".to_string());
    lines.push(format!(
        "    \"\"\"Evaluate f(row) = {source} with guarded arithmetic.\"\"\""
    ));
    if features.uses_x {
        lines.push(
            "    # The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    # row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    return 0.0".to_string());
        return lines.join("\n");
    }
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if len(row) <= {maxc}:"));
        lines.push("        return 0.0".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    try:".to_string());
        lines.push(format!("        value = {}", expr.to_python()));
        lines.push("    except ArithmeticError:".to_string());
        lines.push("        return 0.0".to_string());
    } else {
        lines.push(format!("    value = {}", expr.to_python()));
    }
    lines.push("    if math.isfinite(value):".to_string());
    lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
    lines.push("    return 0.0".to_string());
    lines.join("\n")
}

fn model_class_symbolic(affine: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("class Model:".to_string());
    if affine {
        lines.push(
            "    \"\"\"Closed-form expression with a least-squares affine calibration.\"\"\""
                .to_string(),
        );
    } else {
        lines.push("    \"\"\"Closed-form expression applied directly.\"\"\"".to_string());
    }
    lines.push(String::new());
    lines.push("    def fit(self, x_rows: list, y: list) -> \"Model\":".to_string());
    lines.push("        if not x_rows or len(y) < len(x_rows):".to_string());
    lines.push(
        "            raise SystemExit(\"training data is empty or target is short\")".to_string(),
    );
    if affine {
        lines.push("        f = [expression_value(row) for row in x_rows]".to_string());
        lines.push("        n = float(len(f))".to_string());
        lines.push("        fm = 0.0".to_string());
        lines.push("        for v in f:".to_string());
        lines.push("            fm += v".to_string());
        lines.push("        fm /= n".to_string());
        lines.push("        ym = 0.0".to_string());
        lines.push("        for i in range(len(f)):".to_string());
        lines.push("            ym += y[i]".to_string());
        lines.push("        ym /= n".to_string());
        lines.push("        cov = 0.0".to_string());
        lines.push("        var = 0.0".to_string());
        lines.push("        for i in range(len(f)):".to_string());
        lines.push("            d = f[i] - fm".to_string());
        lines.push("            cov += d * (y[i] - ym)".to_string());
        lines.push("            var += d * d".to_string());
        lines.push("        if var > 1e-24:".to_string());
        lines.push("            self.slope = cov / var".to_string());
        lines.push("            self.intercept = ym - self.slope * fm".to_string());
        lines.push("        else:".to_string());
        lines.push("            self.slope = 0.0".to_string());
        lines.push("            self.intercept = ym".to_string());
    } else {
        lines.push("        self.slope = 1.0".to_string());
        lines.push("        self.intercept = 0.0".to_string());
    }
    lines.push("        return self".to_string());
    lines.push(String::new());
    lines.push("    def predict(self, x_rows: list) -> list:".to_string());
    lines.push(
        "        \"\"\"slope * f(row) + intercept, clipped to [-60, 60]; non-finite maps to 0.\"\"\""
            .to_string(),
    );
    lines.push("        predictions = []".to_string());
    lines.push("        for row in x_rows:".to_string());
    lines.push("            value = expression_value(row)".to_string());
    lines.push("            z = self.slope * value + self.intercept".to_string());
    lines.push("            if math.isfinite(z):".to_string());
    lines.push("                predictions.append(_clamp(z, -60.0, 60.0))".to_string());
    lines.push("            else:".to_string());
    lines.push("                predictions.append(0.0)".to_string());
    lines.push("        return predictions".to_string());
    lines.push(String::new());
    lines.extend(py_score_method());
    lines.join("\n")
}

// ----------------------------------- I/O -----------------------------------

const IO_BLOCK: &str = r#"def _parse_cell(cell):
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
    main(sys.argv[1:])"#;
