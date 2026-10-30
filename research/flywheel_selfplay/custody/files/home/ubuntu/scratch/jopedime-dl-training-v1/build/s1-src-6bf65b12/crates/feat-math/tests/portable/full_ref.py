#!/usr/bin/env python3
# Python all-function port of portable_math.rs. Byte-identical to full_ref.rs.
# Only +-*/ sqrt + bit-manipulation; NEVER native math.exp/log/sin/... (the point).
import struct, math, sys

def to_bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]
def from_bits(b):
    return struct.unpack('<d', struct.pack('<Q', b & 0xffffffffffffffff))[0]
def high_word(x):
    return (to_bits(x) >> 32) & 0xffffffff
def low_word(x):
    return to_bits(x) & 0xffffffff
def as_i32(u):
    u &= 0xffffffff
    return u - 0x100000000 if u >= 0x80000000 else u
def with_high_word(x, hi):
    return from_bits(((hi & 0xffffffff) << 32) | (to_bits(x) & 0xffffffff))
def with_low_word(x, lo):
    return from_bits((to_bits(x) & 0xffffffff00000000) | (lo & 0xffffffff))
def fabs(x):
    return from_bits(to_bits(x) & 0x7fffffffffffffff)
def canonical_nan():
    return from_bits(0x7ff8000000000000)
def two_pow(n):
    return from_bits(((0x3ff + n) & 0xffffffffffffffff) << 52)
def round_half_away(v):
    return float(math.trunc(v + (0.5 if v >= 0 else -0.5)))
def is_nan(x):
    return x != x
def fdiv(a, b):  # IEEE f64 division that never raises (Python raises on /0)
    if b == 0.0:
        if a == 0.0 or a != a:
            return canonical_nan()
        neg = (math.copysign(1.0, a) * math.copysign(1.0, b)) < 0.0
        return float('-inf') if neg else float('inf')
    return a / b

def scalbn(x, n):
    p1023 = two_pow(1023); pm1022 = two_pow(-1022); p53 = two_pow(53); y = x
    if n > 1023:
        y *= p1023; n -= 1023
        if n > 1023:
            y *= p1023; n -= 1023
            if n > 1023: n = 1023
    elif n < -1022:
        y *= pm1022 * p53; n += 1022 - 53
        if n < -1022:
            y *= pm1022 * p53; n += 1022 - 53
            if n < -1022: n = -1022
    return y * two_pow(n)

def xexp(x):
    if is_nan(x): return canonical_nan()
    HALF = [0.5, -0.5]; LN2HI = [6.9314718036912381649e-1, -6.9314718036912381649e-1]
    LN2LO = [1.9082149292705877000e-10, -1.9082149292705877000e-10]
    INVLN2 = 1.4426950408889633870; P1 = 1.66666666666666019037e-1; P2 = -2.77777777770155933842e-3
    P3 = 6.61375632143793436117e-5; P4 = -1.65339022054652515390e-6; P5 = 4.13813679705723846039e-8
    HUGE = 1.0e300; TWOM1000 = 9.33263618503218878990e-302
    O_THRESHOLD = 7.09782712893383973096e2; U_THRESHOLD = -7.45133219101941108420e2
    hx = high_word(x); xsb = (hx >> 31) & 1; hx = hx & 0x7fffffff
    if hx >= 0x40862E42:
        if hx >= 0x7ff00000:
            if ((hx & 0xfffff) | low_word(x)) != 0: return x + x
            return x if xsb == 0 else 0.0
        if x > O_THRESHOLD: return HUGE * HUGE
        if x < U_THRESHOLD: return TWOM1000 * TWOM1000
    k = 0; hi = 0.0; lo = 0.0; xr = 0.0
    if hx > 0x3fd62e42:
        if hx < 0x3FF0A2B2:
            hi = x - LN2HI[xsb]; lo = LN2LO[xsb]; k = 1 - xsb - xsb
        else:
            k = int(INVLN2 * x + HALF[xsb]); t = float(k); hi = x - t * LN2HI[0]; lo = t * LN2LO[0]
        xr = hi - lo
    elif hx < 0x3e300000:
        return 1.0 + x
    else:
        xr = x
    t = xr * xr; c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    if k == 0: return 1.0 - ((xr * c) / (c - 2.0) - xr)
    y = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi); return scalbn(y, k)

def xln(x):
    if is_nan(x): return canonical_nan()
    LN2_HI = 6.9314718036912381649e-1; LN2_LO = 1.9082149292705877000e-10; TWO54 = 1.80143985094819840000e16
    LG1 = 6.666666666666735130e-1; LG2 = 3.999999999940941908e-1; LG3 = 2.857142874366239149e-1
    LG4 = 2.222219843214978396e-1; LG5 = 1.818357216161805012e-1; LG6 = 1.531383769920937332e-1
    LG7 = 1.479819860511658591e-1
    hx = as_i32(high_word(x)); lx = low_word(x); k = 0; xx = x
    if hx < 0x00100000:
        if (hx & 0x7fffffff) == 0 and lx == 0: return float('-inf')
        if hx < 0: return canonical_nan()
        k -= 54; xx *= TWO54; hx = as_i32(high_word(xx))
    if hx >= 0x7ff00000: return x + x
    k += (hx >> 20) - 1023; hx &= 0x000fffff; i = (hx + 0x95f64) & 0x100000
    xx = with_high_word(xx, (hx | (i ^ 0x3ff00000)) & 0xffffffff); k += i >> 20; f = xx - 1.0
    if (0x000fffff & (2 + hx)) < 3:
        if f == 0.0:
            if k == 0: return 0.0
            dk = float(k); return dk * LN2_HI + dk * LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0: return f - r
        dk = float(k); return dk * LN2_HI - ((r - dk * LN2_LO) - f)
    s = f / (2.0 + f); dk = float(k); z = s * s; w = z * z
    t1 = w * (LG2 + w * (LG4 + w * LG6)); t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7))); r = t2 + t1
    i2 = hx - 0x6147a; j2 = 0x6b851 - hx
    if (i2 | j2) > 0:
        hfsq = 0.5 * f * f
        if k == 0: return f - (hfsq - s * (hfsq + r))
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
    elif k == 0:
        return f - s * (f - r)
    return dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)

def xlog1p(x):
    if is_nan(x): return canonical_nan()
    LN2_HI = 6.9314718036912381649e-1; LN2_LO = 1.9082149292705877000e-10
    LG1 = 6.666666666666735130e-1; LG2 = 3.999999999940941908e-1; LG3 = 2.857142874366239149e-1
    LG4 = 2.222219843214978396e-1; LG5 = 1.818357216161805012e-1; LG6 = 1.531383769920937332e-1
    LG7 = 1.479819860511658591e-1
    hx = high_word(x); k = 1; f = 0.0; c = 0.0
    if hx < 0x3fda827a or (hx >> 31) != 0:
        if hx >= 0xbff00000:
            if x == -1.0: return float('-inf')
            return canonical_nan()
        if ((hx << 1) & 0xffffffff) < ((0x3ca00000 << 1) & 0xffffffff): return x
        if hx <= 0xbfd2bec4:
            k = 0; c = 0.0; f = x
    elif hx >= 0x7ff00000:
        return x
    if k != 0:
        u = 1.0 + x; hu = high_word(u); hu = (hu + (0x3ff00000 - 0x3fe6a09e)) & 0xffffffff
        k = (hu >> 20) - 0x3ff
        if k < 54:
            c = (1.0 - (u - x)) if k >= 2 else (x - (u - 1.0)); c /= u
        else:
            c = 0.0
        hu = ((hu & 0x000fffff) + 0x3fe6a09e) & 0xffffffff; f = with_high_word(u, hu) - 1.0
    hfsq = 0.5 * f * f; s = f / (2.0 + f); z = s * s; w = z * z
    t1 = w * (LG2 + w * (LG4 + w * LG6)); t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7))); r = t2 + t1; dk = float(k)
    return s * (hfsq + r) + (dk * LN2_LO + c) - hfsq + f + dk * LN2_HI

def xexpm1(x):
    if is_nan(x): return canonical_nan()
    LN2_HI = 6.9314718036912381649e-1; LN2_LO = 1.90821492927058770002e-10; INVLN2 = 1.4426950408889633870
    Q1 = -3.33333333333331315428e-2; Q2 = 1.58730158725481460165e-3; Q3 = -7.93650757867487942473e-5
    Q4 = 4.00821782732936239552e-6; Q5 = -2.01099218183624371326e-7
    hx = high_word(x) & 0x7fffffff; sign = (high_word(x) >> 31) != 0; c = 0.0; k = 0; xx = x
    if hx >= 0x4043687A:
        if sign: return -1.0
        if hx >= 0x40862E42: return x * two_pow(1023)
    if hx > 0x3fd62e42:
        if hx < 0x3FF0A2B2:
            if not sign:
                hi = x - LN2_HI; lo = LN2_LO; k = 1
            else:
                hi = x + LN2_HI; lo = -LN2_LO; k = -1
        else:
            k = int(INVLN2 * x + (-0.5 if sign else 0.5)); t = float(k); hi = x - t * LN2_HI; lo = t * LN2_LO
        xx = hi - lo; c = (hi - xx) - lo
    elif hx < 0x3c900000:
        return x
    else:
        k = 0
    hfx = 0.5 * xx; hxs = xx * hfx
    r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5)))); t = 3.0 - r1 * hfx
    e = hxs * ((r1 - t) / (6.0 - xx * t))
    if k == 0: return xx - (xx * e - hxs)
    e = xx * (e - c) - c; e -= hxs
    if k == -1: return 0.5 * (xx - e) - 0.5
    if k == 1:
        if xx < -0.25: return -2.0 * (e - (xx + 0.5))
        return 1.0 + 2.0 * (xx - e)
    twopk = two_pow(k)
    if not (0 <= k <= 56):
        y = (xx - e + 1.0) * twopk; return y - 1.0
    inv = two_pow(-k)
    if k < 20: return (xx - e + (1.0 - inv)) * twopk
    return (xx - e - inv + 1.0) * twopk

def xpow(x, y):
    BP = [1.0, 1.5]; DP_H = [0.0, 5.84962487220764160156e-1]; DP_L = [0.0, 1.35003920212974897128e-8]
    TWO53 = 9007199254740992.0; HUGE = 1.0e300; TINY = 1.0e-300
    L1 = 5.99999999999994648725e-1; L2 = 4.28571428578550184252e-1; L3 = 3.33333329818377432918e-1
    L4 = 2.72728123808534006489e-1; L5 = 2.30660745775561366331e-1; L6 = 2.06975017800338417784e-1
    P1 = 1.66666666666666019037e-1; P2 = -2.77777777770155933842e-3; P3 = 6.61375632143793436117e-5
    P4 = -1.65339022054652515390e-6; P5 = 4.13813679705723846039e-8
    LG2 = 6.93147180559945286227e-1; LG2_H = 6.93147182464599609375e-1; LG2_L = -1.90465429995776804525e-9
    OVT = 8.0085662595372944372e-17
    CP = 9.61796693925975554329e-1; CP_H = 9.61796700954437255859e-1; CP_L = -7.02846165095275826516e-9
    IVLN2 = 1.4426950408889633870; IVLN2_H = 1.44269502162933349609e0; IVLN2_L = 1.92596299112661746887e-8
    hx = as_i32(high_word(x)); lx = low_word(x); hy = as_i32(high_word(y)); ly = low_word(y)
    ix = hx & 0x7fffffff; iy = hy & 0x7fffffff
    if iy == 0 and ly == 0: return 1.0
    if is_nan(x) or is_nan(y): return canonical_nan()
    if ix > 0x7ff00000 or (ix == 0x7ff00000 and lx != 0) or iy > 0x7ff00000 or (iy == 0x7ff00000 and ly != 0):
        return canonical_nan()
    yisint = 0
    if hx < 0:
        if iy >= 0x43400000:
            yisint = 2
        elif iy >= 0x3ff00000:
            k = (iy >> 20) - 0x3ff
            if k > 20:
                j = ly >> (52 - k)
                if ((j << (52 - k)) & 0xffffffff) == ly: yisint = 2 - (j & 1)
            elif ly == 0:
                j = iy >> (20 - k)
                if (j << (20 - k)) == iy: yisint = 2 - (j & 1)
    if ly == 0:
        if iy == 0x7ff00000:
            if ix == 0x3ff00000 and lx == 0: return 1.0
            elif ix >= 0x3ff00000: return y if hy >= 0 else 0.0
            else: return -y if hy < 0 else 0.0
        if iy == 0x3ff00000: return fdiv(1.0, x) if hy < 0 else x
        if hy == 0x40000000: return x * x
        if hy == 0x3fe00000 and hx >= 0: return math.sqrt(x)
    ax = fabs(x)
    if lx == 0 and (ix == 0x7ff00000 or ix == 0 or ix == 0x3ff00000):
        z = ax
        if hy < 0: z = fdiv(1.0, z)
        if hx < 0:
            if ix == 0x3ff00000 and yisint == 0: z = canonical_nan()
            elif yisint == 1: z = -z
        return z
    n = (hx >> 31) + 1
    if n == 0 and yisint == 0: return canonical_nan()
    s = 1.0
    if n == 0 and yisint == 1: s = -1.0
    if iy > 0x41e00000:
        if iy > 0x43f00000:
            if ix <= 0x3fefffff: return HUGE * HUGE if hy < 0 else TINY * TINY
            if ix >= 0x3ff00000: return HUGE * HUGE if hy > 0 else TINY * TINY
        if ix < 0x3fefffff: return s * HUGE * HUGE if hy < 0 else s * TINY * TINY
        if ix > 0x3ff00000: return s * HUGE * HUGE if hy > 0 else s * TINY * TINY
        t = ax - 1.0; w = (t * t) * (0.5 - t * (0.333333333333333333333 - t * 0.25))
        u = IVLN2_H * t; v = t * IVLN2_L - w * IVLN2; t1 = with_low_word(u + v, 0); t2 = v - (t1 - u)
    else:
        n = 0
        if ix < 0x00100000:
            ax *= TWO53; n -= 53; ix = as_i32(high_word(ax))
        n += (ix >> 20) - 0x3ff; j0 = ix & 0x000fffff; ix = j0 | 0x3ff00000
        if j0 <= 0x3988E:
            k = 0
        elif j0 < 0xBB67A:
            k = 1
        else:
            k = 0; n += 1; ix -= 0x00100000
        ax = with_high_word(ax, ix & 0xffffffff)
        u = ax - BP[k]; v = 1.0 / (ax + BP[k]); ss = u * v; s_h = with_low_word(ss, 0)
        t_h = with_high_word(0.0, (((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18)) & 0xffffffff)
        t_l = ax - (t_h - BP[k]); s_l = v * ((u - s_h * t_h) - s_h * t_l)
        s2 = ss * ss; r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6))))); r += s_l * (s_h + ss)
        s2b = s_h * s_h; t_h2 = with_low_word(3.0 + s2b + r, 0); t_l2 = r - ((t_h2 - 3.0) - s2b)
        u2 = s_h * t_h2; v2 = s_l * t_h2 + t_l2 * ss; p_h = with_low_word(u2 + v2, 0); p_l = v2 - (p_h - u2)
        z_h = CP_H * p_h; z_l = CP_L * p_h + p_l * CP + DP_L[k]
        t = float(n); t1 = with_low_word(((z_h + z_l) + DP_H[k]) + t, 0); t2 = z_l - (((t1 - t) - DP_H[k]) - z_h)
    y1 = with_low_word(y, 0); p_l = (y - y1) * t1 + y * t2; p_h = y1 * t1; z = p_l + p_h
    j = as_i32(high_word(z)); i = low_word(z)
    if j >= 0x40900000:
        if (j - 0x40900000) != 0 or i != 0: return s * HUGE * HUGE
        if p_l + OVT > z - p_h: return s * HUGE * HUGE
    elif (j & 0x7fffffff) >= 0x4090cc00:
        if (((j & 0xffffffff) - 0xc090cc00) & 0xffffffff) != 0 or i != 0: return s * TINY * TINY
        if p_l <= z - p_h: return s * TINY * TINY
    iabs = j & 0x7fffffff; k = (iabs >> 20) - 0x3ff; nn = 0
    if iabs > 0x3fe00000:
        nn = as_i32(j + (0x00100000 >> (k + 1))); k = ((nn & 0x7fffffff) >> 20) - 0x3ff
        t = with_high_word(0.0, (nn & (0xffffffff ^ (0x000fffff >> k))) & 0xffffffff)
        nn = ((nn & 0x000fffff) | 0x00100000) >> (20 - k)
        if j < 0: nn = -nn
        p_h -= t
    t = with_low_word(p_l + p_h, 0); u = t * LG2_H; v = (p_l - (t - p_h)) * LG2 + t * LG2_L
    z2 = u + v; w = v - (z2 - u); tt = z2 * z2; t1b = z2 - tt * (P1 + tt * (P2 + tt * (P3 + tt * (P4 + tt * P5))))
    r = (z2 * t1b) / (t1b - 2.0) - (w + z2 * w); z2 = 1.0 - (r - z2)
    j = as_i32(high_word(z2)); j = as_i32(j + (nn << 20))
    if (j >> 20) <= 0:
        z2 = scalbn(z2, nn)
    else:
        z2 = with_high_word(z2, j & 0xffffffff)
    return s * z2

def xtanh(x):
    if is_nan(x): return canonical_nan()
    sign = (to_bits(x) >> 63) != 0; ax = from_bits(to_bits(x) & 0x7fffffffffffffff); w = high_word(ax)
    if w > 0x3fe193ea:
        if w > 0x40340000:
            t = 1.0 - 0.0 / ax
        else:
            e = xexpm1(2.0 * ax); t = 1.0 - 2.0 / (e + 2.0)
    elif w > 0x3fd058ae:
        e = xexpm1(2.0 * ax); t = e / (e + 2.0)
    elif w >= 0x00100000:
        e = xexpm1(-2.0 * ax); t = -e / (e + 2.0)
    else:
        t = ax
    return -t if sign else t

def kernel_sin(x, y, iy):
    S1 = -1.66666666666666324348e-1; S2 = 8.33333333332248946124e-3; S3 = -1.98412698298579493134e-4
    S4 = 2.75573137070700676789e-6; S5 = -2.50507602534068634195e-8; S6 = 1.58969099521155010221e-10
    z = x * x; w = z * z; r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6); v = z * x
    if iy == 0: return x + v * (S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * S1)

def kernel_cos(x, y):
    C1 = 4.16666666666666019037e-2; C2 = -1.38888888888741095749e-3; C3 = 2.48015872894767294178e-5
    C4 = -2.75573143513906633035e-7; C5 = 2.08757232129817482790e-9; C6 = -1.13596475577881948265e-11
    z = x * x; w = z * z; r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6))
    hz = 0.5 * z; w2 = 1.0 - hz
    return w2 + (((1.0 - w2) - hz) + (z * r - x * y))

def rem_pio2(x):
    INVPIO2 = 6.36619772367581382433e-1; PIO2_1 = 1.57079632673412561417e0; PIO2_1T = 6.07710050650619224932e-11
    PIO2_2 = 6.07710050630396597660e-11; PIO2_2T = 2.02226624879595063154e-21
    PIO2_3 = 2.02226624871116645580e-21; PIO2_3T = 8.47842766036889956997e-32
    INV_TWO_PI = 1.59154943091895335769e-1; TWO_PI_HI = 6.28318530717958623199e0; TWO_PI_LO = 2.44929359829470641471e-16
    ix = high_word(x) & 0x7fffffff
    if ix >= 0x7ff00000:
        z = canonical_nan(); return (0, z, z)
    xr = x
    while (high_word(xr) & 0x7fffffff) >= 0x413921fb:
        k = round_half_away(xr * INV_TWO_PI); xr = (xr - k * TWO_PI_HI) - k * TWO_PI_LO
    ixr = high_word(xr) & 0x7fffffff
    if ixr <= 0x3fe921fb:
        return (0, xr, 0.0)
    half = 0.5 if xr > 0.0 else -0.5; n = int(xr * INVPIO2 + half); fnn = float(n)
    r = xr - fnn * PIO2_1; w = fnn * PIO2_1T; j = ixr >> 20
    y0 = r - w; hi = (high_word(y0) >> 20) & 0x7ff; i = j - hi
    if i > 16:
        t = r; w = fnn * PIO2_2; r = t - w; w = fnn * PIO2_2T - ((t - r) - w); y0 = r - w
        hi = (high_word(y0) >> 20) & 0x7ff; i = j - hi
        if i > 49:
            t2 = r; w = fnn * PIO2_3; r = t2 - w; w = fnn * PIO2_3T - ((t2 - r) - w); y0 = r - w
    y1 = (r - y0) - w; return (n & 3, y0, y1)

def xsin(x):
    if is_nan(x): return canonical_nan()
    ix = high_word(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        if ix < 0x3e500000: return x
        return kernel_sin(x, 0.0, 0)
    if ix >= 0x7ff00000: return canonical_nan()
    n, y0, y1 = rem_pio2(x)
    if n == 0: return kernel_sin(y0, y1, 1)
    if n == 1: return kernel_cos(y0, y1)
    if n == 2: return -kernel_sin(y0, y1, 1)
    return -kernel_cos(y0, y1)

def xcos(x):
    if is_nan(x): return canonical_nan()
    ix = high_word(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        if ix < 0x3e46a09e: return 1.0
        return kernel_cos(x, 0.0)
    if ix >= 0x7ff00000: return canonical_nan()
    n, y0, y1 = rem_pio2(x)
    if n == 0: return kernel_cos(y0, y1)
    if n == 1: return -kernel_sin(y0, y1, 1)
    if n == 2: return -kernel_cos(y0, y1)
    return kernel_sin(y0, y1, 1)

def xatan(x):
    if is_nan(x): return canonical_nan()
    ATANHI = [4.63647609000806093515e-1, 7.85398163397448278999e-1, 9.82793723247329054082e-1, 1.57079632679489655800e0]
    ATANLO = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17]
    AT = [3.33333333333329318027e-1, -1.99999999998764832476e-1, 1.42857142725034663711e-1, -1.11111104054623557880e-1,
          9.09088713343650656196e-2, -7.69187620504482999495e-2, 6.66107313738753120669e-2, -5.83357013379057348645e-2,
          4.97687799461593236017e-2, -3.65315727442169155270e-2, 1.62858201153657823623e-2]
    hx = as_i32(high_word(x)); ix = hx & 0x7fffffff
    if ix >= 0x44100000:
        if ix > 0x7ff00000 or (ix == 0x7ff00000 and low_word(x) != 0): return canonical_nan()
        return ATANHI[3] + ATANLO[3] if hx > 0 else -ATANHI[3] - ATANLO[3]
    xr = x
    if ix < 0x3fdc0000:
        if ix < 0x3e400000: return x
        id = -1
    else:
        xr = fabs(x)
        if ix < 0x3ff30000:
            if ix < 0x3fe60000:
                id = 0; xr = (2.0 * xr - 1.0) / (2.0 + xr)
            else:
                id = 1; xr = (xr - 1.0) / (xr + 1.0)
        elif ix < 0x40038000:
            id = 2; xr = (xr - 1.5) / (1.0 + 1.5 * xr)
        else:
            id = 3; xr = -1.0 / xr
    z = xr * xr; w = z * z
    s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))))
    s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))))
    if id < 0: return xr - xr * (s1 + s2)
    z2 = ATANHI[id] - ((xr * (s1 + s2) - ATANLO[id]) - xr)
    return -z2 if hx < 0 else z2

def xatan2(y, x):
    if is_nan(x) or is_nan(y): return canonical_nan()
    TINY = 1.0e-300; PI_O_4 = 7.85398163397448278990e-1; PI_O_2 = 1.57079632679489655800e0
    PI = 3.14159265358979311600e0; PI_LO = 1.22464679914735317722e-16
    hx = as_i32(high_word(x)); ix = hx & 0x7fffffff; lx = low_word(x)
    hy = as_i32(high_word(y)); iy = hy & 0x7fffffff; ly = low_word(y)
    if (ix | (1 if lx != 0 else 0)) > 0x7ff00000 or (iy | (1 if ly != 0 else 0)) > 0x7ff00000:
        return canonical_nan()
    if hx == 0x3ff00000 and lx == 0: return xatan(y)
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2)
    if iy == 0 and ly == 0:
        if m == 0 or m == 1: return y
        if m == 2: return PI + TINY
        return -PI - TINY
    if ix == 0 and lx == 0: return -PI_O_2 - TINY if hy < 0 else PI_O_2 + TINY
    if ix == 0x7ff00000:
        if iy == 0x7ff00000:
            if m == 0: return PI_O_4 + TINY
            if m == 1: return -PI_O_4 - TINY
            if m == 2: return 3.0 * PI_O_4 + TINY
            return -3.0 * PI_O_4 - TINY
        if m == 0: return 0.0
        if m == 1: return -0.0
        if m == 2: return PI + TINY
        return -PI - TINY
    if iy == 0x7ff00000: return -PI_O_2 - TINY if hy < 0 else PI_O_2 + TINY
    k = (iy - ix) >> 20
    if k > 60:
        z = PI_O_2 + 0.5 * PI_LO
    elif hx < 0 and k < -60:
        z = 0.0
    else:
        z = xatan(fabs(y / x))
    if m == 0: return z
    if m == 1: return -z
    if m == 2: return PI - (z - PI_LO)
    return (z - PI_LO) - PI

def grid():
    g = []
    for i in range(-900, 901):
        g.append(i * 0.41)
    for e in range(-55, 55):
        g.append(two_pow(e)); g.append(-two_pow(e)); g.append(1.0 + two_pow(e))
    g.extend([0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 709.0, -745.0, 710.0,
              from_bits(1), from_bits(0x000fffffffffffff), from_bits(0x0010000000000000),
              from_bits(0x8010000000000000), 1e300, 1e-300, from_bits(0x7fefffffffffffff),
              from_bits(0xffefffffffffffff), float('inf'), float('-inf'),
              from_bits(0x7ff8000000000000), from_bits(0xfff8000000001234)])
    return g

def hexout(v):
    return format(to_bits(v), '016x')

def main():
    out = []
    g = grid()
    for x in g:
        xb = format(to_bits(x), '016x')
        out.append('exp %s %s' % (xb, hexout(xexp(x))))
        out.append('ln %s %s' % (xb, hexout(xln(x))))
        out.append('log1p %s %s' % (xb, hexout(xlog1p(x))))
        out.append('expm1 %s %s' % (xb, hexout(xexpm1(x))))
        out.append('tanh %s %s' % (xb, hexout(xtanh(x))))
        out.append('sin %s %s' % (xb, hexout(xsin(x))))
        out.append('cos %s %s' % (xb, hexout(xcos(x))))
        out.append('atan %s %s' % (xb, hexout(xatan(x))))
    pow_bases = [float('-inf'), from_bits(0xffefffffffffffff), -3.5, -1.0,
                 from_bits(0x8010000000000000), from_bits(0x8000000000000001), -0.0, 0.0,
                 from_bits(1), from_bits(0x0010000000000000), 0.5, 1.0, 2.0, 3.5,
                 from_bits(0x7fefffffffffffff), float('inf'), from_bits(0x7ff8000000000000)]
    exps = [float('-inf'), -8.0, -3.0, -2.5, -1.0, -0.5, -0.0, 0.0, 0.3333, 0.5,
            1.0, 2.0, 2.5, 3.0, 8.0, float('inf'), from_bits(0x7ff8000000000000)]
    for b in pow_bases:
        for e in exps:
            out.append('pow %s %s %s' % (format(to_bits(b), '016x'), format(to_bits(e), '016x'), hexout(xpow(b, e))))
    ys = [float('-inf'), from_bits(0xffefffffffffffff), -3.0, -1.0,
          from_bits(0x8010000000000000), from_bits(0x8000000000000001), -0.0, 0.0,
          from_bits(1), from_bits(0x0010000000000000), 1.0, from_bits(0x3ff0000080000000), 3.0,
          from_bits(0x7fefffffffffffff), float('inf'), from_bits(0x7ff8000000000000)]
    xs = ys
    for yy in ys:
        for xx in xs:
            out.append('atan2 %s %s %s' % (format(to_bits(yy), '016x'), format(to_bits(xx), '016x'), hexout(xatan2(yy, xx))))
    sys.stdout.write('\n'.join(out) + '\n')

main()
