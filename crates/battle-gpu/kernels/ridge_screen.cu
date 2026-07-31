// Rung-0 ridge screening kernels for the feat-math GPU seam.
//
// BIT-EXACTNESS CONTRACT: compiled with `-fmad=false` (no FMA contraction),
// all reductions are per-thread SEQUENTIAL loops in the same order as the CPU
// reference (feat-math heads.rs RidgeModel::fit / cholesky_solve and
// metrics.rs r2), and every arithmetic step is f64 after an explicit f32->f64
// widening of the stored inputs. Under IEEE-754 (both sides), each score is
// therefore bit-identical to the CPU exact path, not merely close. Determinism
// across runs is trivial: no atomics, no cross-thread reductions, fixed launch
// geometry.
//
// Problem layout (packed by the host, offsets in ELEMENTS):
//   train_x  concat of per-problem column-major f32 blocks (p * n)
//   train_y  concat of per-problem f32 targets (n)
//   valid_x  concat of per-problem row-major f32 blocks (vn * p)
//   valid_y  concat of per-problem f64 targets (vn)
//   stats    per-problem f64 scratch of width p (means/scales/atb/z share it)
//   gram     per-problem f64 p*p
//   w        per-problem f64 p+1 ([intercept, coeffs..])

extern "C" __global__ void col_stats(
    const float* __restrict__ train_x,
    const float* __restrict__ train_y,
    const unsigned long long* __restrict__ x_off,
    const unsigned long long* __restrict__ y_off,
    const unsigned long long* __restrict__ stat_off,
    const int* __restrict__ train_n,
    const int* __restrict__ p_dims,
    double* __restrict__ means,
    double* __restrict__ scales,
    double* __restrict__ y_means,
    int problems)
{
    int prob = blockIdx.x;
    if (prob >= problems) return;
    int n = train_n[prob];
    int p = p_dims[prob];
    const float* x = train_x + x_off[prob];
    // std_fit divides by x.len().max(1); y_mean divides by n.max(1) — same n.
    double n_div = (double)(n > 0 ? n : 1);
    for (int j = threadIdx.x; j < p; j += blockDim.x) {
        const float* col = x + (unsigned long long)j * (unsigned long long)n;
        // Two-pass mean/variance, sequential row order (mirrors std_fit).
        double s = 0.0;
        for (int r = 0; r < n; ++r) s += (double)col[r];
        double mean = s / n_div;
        double v = 0.0;
        for (int r = 0; r < n; ++r) { double d = (double)col[r] - mean; v += d * d; }
        double scale = sqrt(v / n_div);
        if (!(scale > 1e-9)) scale = 1e-9; // .max(1e-9)
        means[stat_off[prob] + j] = mean;
        scales[stat_off[prob] + j] = scale;
    }
    if (threadIdx.x == 0) {
        const float* y = train_y + y_off[prob];
        double s = 0.0;
        for (int r = 0; r < n; ++r) s += (double)y[r];
        y_means[prob] = s / n_div;
    }
}

extern "C" __global__ void gram_atb(
    const float* __restrict__ train_x,
    const float* __restrict__ train_y,
    const unsigned long long* __restrict__ x_off,
    const unsigned long long* __restrict__ y_off,
    const unsigned long long* __restrict__ stat_off,
    const unsigned long long* __restrict__ gram_off,
    const int* __restrict__ train_n,
    const int* __restrict__ p_dims,
    const double* __restrict__ means,
    const double* __restrict__ scales,
    const double* __restrict__ y_means,
    const double* __restrict__ lambdas,
    double* __restrict__ ata,
    double* __restrict__ atb,
    int problems)
{
    int prob = blockIdx.x;
    if (prob >= problems) return;
    int n = train_n[prob];
    int p = p_dims[prob];
    const float* x = train_x + x_off[prob];
    const double* mu = means + stat_off[prob];
    const double* sc = scales + stat_off[prob];
    double* A = ata + gram_off[prob];
    double* B = atb + stat_off[prob];
    int pairs = p * (p + 1) / 2;
    int total = pairs + p;
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        if (t < pairs) {
            // t -> (i, j), i <= j, row-major upper-triangle order.
            int i = 0;
            int rem = t;
            while (rem >= p - i) { rem -= p - i; ++i; }
            int j = i + rem;
            const float* ci = x + (unsigned long long)i * (unsigned long long)n;
            const float* cj = x + (unsigned long long)j * (unsigned long long)n;
            // Sequential row loop: identical accumulation order to the CPU
            // Gram loop (per row: xs_i * xs_j added once).
            double s = 0.0;
            for (int r = 0; r < n; ++r) {
                double xi = ((double)ci[r] - mu[i]) / sc[i];
                xi = xi < -20.0 ? -20.0 : (xi > 20.0 ? 20.0 : xi);
                double xj = ((double)cj[r] - mu[j]) / sc[j];
                xj = xj < -20.0 ? -20.0 : (xj > 20.0 ? 20.0 : xj);
                s += xi * xj;
            }
            if (i == j) s += lambdas[prob]; // lambda added after accumulation
            A[(unsigned long long)i * p + j] = s;
            A[(unsigned long long)j * p + i] = s;
        } else {
            int i = t - pairs;
            const float* ci = x + (unsigned long long)i * (unsigned long long)n;
            const float* y = train_y + y_off[prob];
            double ym = y_means[prob];
            double s = 0.0;
            for (int r = 0; r < n; ++r) {
                double xi = ((double)ci[r] - mu[i]) / sc[i];
                xi = xi < -20.0 ? -20.0 : (xi > 20.0 ? 20.0 : xi);
                s += xi * ((double)y[r] - ym);
            }
            B[i] = s;
        }
    }
}

extern "C" __global__ void chol_solve(
    const unsigned long long* __restrict__ gram_off,
    const unsigned long long* __restrict__ stat_off,
    const unsigned long long* __restrict__ w_off,
    const int* __restrict__ p_dims,
    const double* __restrict__ y_means,
    double* __restrict__ ata,
    double* __restrict__ atb,
    double* __restrict__ z_scratch,
    double* __restrict__ w_out,
    int problems)
{
    // One thread per problem: the whole factor/solve is sequential, exactly
    // heads.rs cholesky_solve (including the s > 1e-12 ? sqrt(s) : 1e-6 pivot
    // floor, so no failure status is possible for well-formed p).
    int prob = blockIdx.x * blockDim.x + threadIdx.x;
    if (prob >= problems) return;
    int p = p_dims[prob];
    double* a = ata + gram_off[prob];
    double* b = atb + stat_off[prob];
    double* z = z_scratch + stat_off[prob];
    double* w = w_out + w_off[prob];
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = a[(unsigned long long)i * p + j];
            for (int k = 0; k < j; ++k)
                s -= a[(unsigned long long)i * p + k] * a[(unsigned long long)j * p + k];
            if (i == j) {
                a[(unsigned long long)i * p + j] = s > 1e-12 ? sqrt(s) : 1e-6;
            } else {
                a[(unsigned long long)i * p + j] = s / a[(unsigned long long)j * p + j];
            }
        }
    }
    for (int i = 0; i < p; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= a[(unsigned long long)i * p + k] * z[k];
        z[i] = s / a[(unsigned long long)i * p + i];
    }
    for (int i = p - 1; i >= 0; --i) {
        double s = z[i];
        for (int k = i + 1; k < p; ++k)
            s -= a[(unsigned long long)k * p + i] * w[k + 1];
        w[i + 1] = s / a[(unsigned long long)i * p + i];
    }
    w[0] = y_means[prob];
}

extern "C" __global__ void predict_r2(
    const float* __restrict__ valid_x,
    const double* __restrict__ valid_y,
    const unsigned long long* __restrict__ vx_off,
    const unsigned long long* __restrict__ vy_off,
    const unsigned long long* __restrict__ stat_off,
    const unsigned long long* __restrict__ w_off,
    const int* __restrict__ valid_n,
    const int* __restrict__ p_dims,
    const double* __restrict__ means,
    const double* __restrict__ scales,
    const double* __restrict__ w_all,
    double* __restrict__ scores,
    int problems)
{
    // One thread per problem, fully sequential: mirrors RidgeModel::predict
    // (standardize + clamp(-20,20), dot, clamp(-60,60)) then metrics::r2
    // (test-mean centered, ss_tot <= 0 -> 0).
    int prob = blockIdx.x * blockDim.x + threadIdx.x;
    if (prob >= problems) return;
    int vn = valid_n[prob];
    int p = p_dims[prob];
    const float* vx = valid_x + vx_off[prob];
    const double* vy = valid_y + vy_off[prob];
    const double* mu = means + stat_off[prob];
    const double* sc = scales + stat_off[prob];
    const double* w = w_all + w_off[prob];
    if (vn == 0) { scores[prob] = 0.0; return; }
    double mean = 0.0;
    for (int r = 0; r < vn; ++r) mean += vy[r];
    mean /= (double)vn;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (int r = 0; r < vn; ++r) {
        const float* row = vx + (unsigned long long)r * (unsigned long long)p;
        double zp = w[0];
        for (int j = 0; j < p; ++j) {
            double xs = ((double)row[j] - mu[j]) / sc[j];
            xs = xs < -20.0 ? -20.0 : (xs > 20.0 ? 20.0 : xs);
            zp += w[j + 1] * xs;
        }
        zp = zp < -60.0 ? -60.0 : (zp > 60.0 ? 60.0 : zp);
        double d = vy[r] - zp;
        ss_res += d * d;
        double t = vy[r] - mean;
        ss_tot += t * t;
    }
    scores[prob] = ss_tot <= 0.0 ? 0.0 : 1.0 - ss_res / ss_tot;
}
