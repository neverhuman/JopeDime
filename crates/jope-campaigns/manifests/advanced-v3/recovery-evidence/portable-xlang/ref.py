#!/usr/bin/env python3
# Python port of the portable exp/ln kernel. Must print byte-identical output
# to ref.rs. Python gotchas handled: division-by-zero RAISES (return explicit
# inf/nan), `as i32` needs signed conversion, u64 ops need explicit masking.
import struct

MASK32 = 0xffffffff
MASK64 = 0xffffffffffffffff

def to_bits(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def from_bits(b): return struct.unpack('<d', struct.pack('<Q', b & MASK64))[0]
def high_word(x): return (to_bits(x) >> 32) & MASK32          # u32
def low_word(x): return to_bits(x) & MASK32
def as_i32(u): return u - 0x100000000 if u >= 0x80000000 else u
def with_high_word(x, hi): return from_bits(((hi & MASK32) << 32) | (to_bits(x) & MASK32))
def two_pow(n): return from_bits((((0x3ff + n) & MASK64) << 52) & MASK64)

def scalbn(x, n):
    p1023 = two_pow(1023); pm1022 = two_pow(-1022); p53 = two_pow(53)
    y = x
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
    HALF = (0.5, -0.5)
    LN2HI = (6.9314718036912381649e-1, -6.9314718036912381649e-1)
    LN2LO = (1.9082149292705877000e-10, -1.9082149292705877000e-10)
    INVLN2 = 1.4426950408889633870
    P1 = 1.66666666666666019037e-1; P2 = -2.77777777770155933842e-3; P3 = 6.61375632143793436117e-5
    P4 = -1.65339022054652515390e-6; P5 = 4.13813679705723846039e-8
    HUGE = 1.0e300; TWOM1000 = 9.33263618503218878990e-302
    O_THRESHOLD = 7.09782712893383973096e2; U_THRESHOLD = -7.45133219101941108420e2
    hx = high_word(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7fffffff
    if hx >= 0x40862E42:
        if hx >= 0x7ff00000:
            if ((hx & 0xfffff) | low_word(x)) != 0: return x + x
            return x if xsb == 0 else 0.0
        if x > O_THRESHOLD: return HUGE * HUGE
        if x < U_THRESHOLD: return TWOM1000 * TWOM1000
    k = 0; hi = 0.0; lo = 0.0
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
    t = xr * xr
    c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    if k == 0: return 1.0 - ((xr * c) / (c - 2.0) - xr)
    y = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi)
    return scalbn(y, k)

def xln(x):
    LN2_HI = 6.9314718036912381649e-1; LN2_LO = 1.9082149292705877000e-10; TWO54 = 1.80143985094819840000e16
    LG1 = 6.666666666666735130e-1; LG2 = 3.999999999940941908e-1; LG3 = 2.857142874366239149e-1
    LG4 = 2.222219843214978396e-1; LG5 = 1.818357216161805012e-1; LG6 = 1.531383769920937332e-1; LG7 = 1.479819860511658591e-1
    hx = as_i32(high_word(x))
    lx = low_word(x)
    k = 0; xx = x
    if hx < 0x00100000:
        if ((hx & 0x7fffffff) | lx) == 0: return float('-inf')     # ln(+-0)
        if hx < 0: return float('nan')                             # ln(neg)
        k -= 54; xx *= TWO54; hx = as_i32(high_word(xx))
    if hx >= 0x7ff00000: return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000fffff
    i = (hx + 0x95f64) & 0x100000
    xx = with_high_word(xx, (hx | (i ^ 0x3ff00000)) & MASK32)
    k += i >> 20
    f = xx - 1.0
    if (0x000fffff & (2 + hx)) < 3:
        if f == 0.0:
            if k == 0: return 0.0
            dk = float(k); return dk * LN2_HI + dk * LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0: return f - r
        dk = float(k); return dk * LN2_HI - ((r - dk * LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s; w = z * z
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    r = t2 + t1
    i2 = hx - 0x6147a
    j2 = 0x6b851 - hx
    if (i2 | j2) > 0:
        hfsq = 0.5 * f * f
        if k == 0: return f - (hfsq - s * (hfsq + r))
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
    if k == 0: return f - s * (f - r)
    return dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)

def grid():
    g = []
    for i in range(-1900, 1901): g.append(i * 0.37)
    for e in range(-60, 60):
        g.append(two_pow(e)); g.append(-two_pow(e)); g.append(1.0 + two_pow(e))
    g += [0.0, -0.0, 1.0, -1.0, 709.0, -745.0, 710.0, -746.0,
          from_bits(1), from_bits(0x000fffffffffffff), 1e300, 1e-300,
          float('inf'), float('-inf'), float('nan')]
    return g

def hexout(v):
    if v != v: return 'NaN'
    return format(to_bits(v), '016x')

lines = []
for x in grid():
    xb = format(to_bits(x), '016x')
    lines.append(f"exp {xb} {hexout(xexp(x))}")
    lines.append(f"ln {xb} {hexout(xln(x))}")
print('\n'.join(lines))
