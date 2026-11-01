// Frozen inference arithmetic from cpu.rs, executed on the selected CUDA device.
// exp_u10 is the same SLEEF 3.6.1 adaptation: Copyright Naoki Shibata and
// contributors 2010-2024, Boost Software License 1.0; see BSL-1.0.txt.
__device__ float exp_u10(float x) {
    if (isnan(x)) return x;
    if (x < -104.f) return 0.f;
    if (x > 100.f) return __int_as_float(0x7f800000);
    int q = __float2int_rn(x * __int_as_float(0x3fb8aa3b));
    float s = __fmaf_rn((float)q, -__int_as_float(0x3f317200), x);
    s = __fmaf_rn((float)q, -__int_as_float(0x35bfbe8e), s);
    float u = __int_as_float(0x39502bda);
    u = __fmaf_rn(u, s, __int_as_float(0x3ab696c9));
    u = __fmaf_rn(u, s, __int_as_float(0x3c0888a6));
    u = __fmaf_rn(u, s, __int_as_float(0x3d2aaa7a));
    u = __fmaf_rn(u, s, __int_as_float(0x3e2aaaab));
    u = __fmaf_rn(u, s, 0.5f);
    u = 1.f + __fmaf_rn(s * s, u, s);
    int first = q >> 1;
    return u * __int_as_float(((unsigned int)(first + 127)) << 23)
             * __int_as_float(((unsigned int)(q - first + 127)) << 23);
}
extern "C" __global__ void frozen_silu(unsigned int n, unsigned int width,
    const float *x, unsigned long long offset, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[offset + i]; y[i] = v / (1.f + exp_u10(-v)); }
}
extern "C" __global__ void frozen_tanh(unsigned int n, unsigned int width,
    const float *x, unsigned long long offset, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)tanh((double)x[offset + i]);
}
extern "C" __global__ void frozen_softmax(unsigned int rows, unsigned int width,
    const float *x, unsigned long long offset, float *y) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    x += offset + r * 8; y += r * 8;
    float m = x[0]; for (int j = 1; j < 8; ++j) m = fmaxf(m, x[j]);
    float e[8]; for (int j = 0; j < 8; ++j) e[j] = exp_u10(x[j] - m);
    float sum = ((e[0] + e[4]) + (e[2] + e[6])) + ((e[1] + e[5]) + (e[3] + e[7]));
    float reciprocal = 1.f / sum;
    for (int j = 0; j < 8; ++j) y[j] = e[j] * reciprocal;
}
extern "C" __global__ void frozen_linear(unsigned int rows, unsigned int input,
    unsigned int output, const float *x, unsigned long long ox,
    const float *w, unsigned long long ow, const float *b, unsigned long long ob, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * output) return;
    unsigned int row = i / output, col = i % output;
    float sum = 0.f;
    for (unsigned int k = 0; k < input; ++k) sum += x[ox + row * input + k] * w[ow + col * input + k];
    y[i] = sum + b[ob + col];
}
extern "C" __global__ void frozen_norm(unsigned int rows, unsigned int width,
    unsigned int unused, const float *x, unsigned long long ox,
    const float *w, unsigned long long ow, const float *b, unsigned long long ob, float *y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    x += ox + row * width; y += row * width;
    unsigned int vectors = width / 8, count_vec = 0;
    float means[8] = {}, variances[8] = {};
    for (unsigned int start = 0; start < vectors; start += 16) {
        unsigned int take = min(vectors - start, 16u);
        float mean[8] = {}, variance[8] = {};
        for (unsigned int j = 0; j < take; ++j) {
            float reciprocal = 1.f / (float)(j + 1);
            for (int k = 0; k < 8; ++k) {
                float v = x[(start + j) * 8 + k], delta = v - mean[k];
                mean[k] = __fmaf_rn(delta, reciprocal, mean[k]);
                variance[k] = __fmaf_rn(delta, v - mean[k], variance[k]);
            }
        }
        unsigned int next = count_vec + take;
        float ratio = (float)take / (float)next;
        for (int k = 0; k < 8; ++k) {
            float delta = mean[k] - means[k];
            means[k] = __fmaf_rn(ratio, delta, means[k]);
            variances[k] += __fmaf_rn(delta * delta * ratio, (float)count_vec, variance[k]);
        }
        count_vec = next;
    }
    unsigned int count = 0; float mean = 0.f, variance = 0.f;
    for (unsigned int j = vectors * 8; j < width; ++j) {
        float delta = x[j] - mean; ++count;
        mean += delta / (float)count;
        variance = __fmaf_rn(delta, x[j] - mean, variance);
    }
    for (int k = 0; k < 8; ++k) {
        unsigned int next = count + vectors;
        float ratio = (float)vectors / (float)next, delta = means[k] - mean;
        mean = __fmaf_rn(ratio, delta, mean);
        variance += __fmaf_rn(delta * delta * ratio, (float)count, variances[k]);
        count = next;
    }
    float reciprocal = 1.f / sqrtf(variance / (float)width + 1e-5f);
    for (unsigned int j = 0; j < width; ++j) y[j] = __fmaf_rn((x[j] - mean) * reciprocal, w[ow + j], b[ob + j]);
}
extern "C" __global__ void frozen_matmul(unsigned int rows, unsigned int inner,
    unsigned int cols, const float *a, unsigned long long oa,
    const float *b, unsigned long long ob, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int row = i / cols, col = i % cols;
    float sum = 0.f;
    for (unsigned int k = 0; k < inner; ++k) sum += a[oa + row * inner + k] * b[ob + k * cols + col];
    y[i] = sum;
}
