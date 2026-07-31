// JavaScript port of the portable exp/ln kernel. Must print byte-identical
// output to ref.rs / ref.cpp. The f64 arithmetic uses plain JS numbers; only
// the 64-bit bit manipulation uses BigInt (JS bitwise ops are 32-bit).
// Native Math.exp/Math.log are NEVER used — that is the whole point.
'use strict';

const _buf = new ArrayBuffer(8);
const _f64 = new Float64Array(_buf);
const _u64 = new BigUint64Array(_buf);
const MASK64 = 0xffffffffffffffffn;

function toBits(x) { _f64[0] = x; return _u64[0]; }            // -> BigInt
function fromBits(b) { _u64[0] = b & MASK64; return _f64[0]; } // b: BigInt
function highWord(x) { return Number((toBits(x) >> 32n) & 0xffffffffn); } // u32 as Number [0..2^32)
function lowWord(x) { return Number(toBits(x) & 0xffffffffn); }
function withHighWord(x, hi) { // hi: Number u32
  return fromBits((BigInt(hi >>> 0) << 32n) | (toBits(x) & 0xffffffffn));
}
function twoPow(n) { // n: Number i32 ; mirrors ((0x3ff+n) as i64 as u64) << 52
  let v = BigInt.asUintN(64, BigInt(0x3ff + n));
  return fromBits(BigInt.asUintN(64, v << 52n));
}

function scalbn(x, n) {
  const p1023 = twoPow(1023), pm1022 = twoPow(-1022), p53 = twoPow(53);
  let y = x;
  if (n > 1023) { y *= p1023; n -= 1023; if (n > 1023) { y *= p1023; n -= 1023; if (n > 1023) n = 1023; } }
  else if (n < -1022) { y *= pm1022 * p53; n += 1022 - 53; if (n < -1022) { y *= pm1022 * p53; n += 1022 - 53; if (n < -1022) n = -1022; } }
  return y * twoPow(n);
}

function xexp(x) {
  const HALF = [0.5, -0.5];
  const LN2HI = [6.9314718036912381649e-1, -6.9314718036912381649e-1];
  const LN2LO = [1.9082149292705877000e-10, -1.9082149292705877000e-10];
  const INVLN2 = 1.4426950408889633870;
  const P1 = 1.66666666666666019037e-1, P2 = -2.77777777770155933842e-3, P3 = 6.61375632143793436117e-5,
        P4 = -1.65339022054652515390e-6, P5 = 4.13813679705723846039e-8;
  const HUGE = 1.0e300, TWOM1000 = 9.33263618503218878990e-302,
        O_THRESHOLD = 7.09782712893383973096e2, U_THRESHOLD = -7.45133219101941108420e2;
  let hx = highWord(x);
  const xsb = (hx >>> 31) & 1;
  hx = hx & 0x7fffffff; // -> [0, 0x7fffffff]
  if (hx >= 0x40862E42) {
    if (hx >= 0x7ff00000) {
      if (((hx & 0xfffff) | lowWord(x)) !== 0) return x + x;
      return xsb === 0 ? x : 0.0;
    }
    if (x > O_THRESHOLD) return HUGE * HUGE;
    if (x < U_THRESHOLD) return TWOM1000 * TWOM1000;
  }
  let k = 0, hi = 0.0, lo = 0.0, xr;
  if (hx > 0x3fd62e42) {
    if (hx < 0x3FF0A2B2) { hi = x - LN2HI[xsb]; lo = LN2LO[xsb]; k = 1 - xsb - xsb; }
    else { k = Math.trunc(INVLN2 * x + HALF[xsb]); const t = k; hi = x - t * LN2HI[0]; lo = t * LN2LO[0]; }
    xr = hi - lo;
  } else if (hx < 0x3e300000) { return 1.0 + x; } else { xr = x; }
  const t = xr * xr;
  const c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k === 0) return 1.0 - ((xr * c) / (c - 2.0) - xr);
  const y = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi);
  return scalbn(y, k);
}

function xln(x) {
  const LN2_HI = 6.9314718036912381649e-1, LN2_LO = 1.9082149292705877000e-10, TWO54 = 1.80143985094819840000e16;
  const LG1 = 6.666666666666735130e-1, LG2 = 3.999999999940941908e-1, LG3 = 2.857142874366239149e-1,
        LG4 = 2.222219843214978396e-1, LG5 = 1.818357216161805012e-1, LG6 = 1.531383769920937332e-1, LG7 = 1.479819860511658591e-1;
  let hx = highWord(x) | 0; // as i32
  const lx = lowWord(x);
  let k = 0, xx = x;
  if (hx < 0x00100000) {
    if (((hx & 0x7fffffff) | (lx | 0)) === 0) return -TWO54 / 0.0;
    if (hx < 0) return (x - x) / 0.0;
    k -= 54; xx *= TWO54; hx = highWord(xx) | 0;
  }
  if (hx >= 0x7ff00000) return x + x;
  k += (hx >> 20) - 1023;
  hx = hx & 0x000fffff;
  const i = (hx + 0x95f64) & 0x100000;
  xx = withHighWord(xx, (hx | (i ^ 0x3ff00000)) >>> 0);
  k += i >> 20;
  const f = xx - 1.0;
  if ((0x000fffff & (2 + hx)) < 3) {
    if (f === 0.0) { if (k === 0) return 0.0; const dk = k; return dk * LN2_HI + dk * LN2_LO; }
    const r = f * f * (0.5 - 0.33333333333333333 * f);
    if (k === 0) return f - r;
    const dk = k; return dk * LN2_HI - ((r - dk * LN2_LO) - f);
  }
  const s = f / (2.0 + f);
  const dk = k;
  const z = s * s, w = z * z;
  const t1 = w * (LG2 + w * (LG4 + w * LG6));
  const t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
  const r = t2 + t1;
  const i2 = hx - 0x6147a;
  const j2 = 0x6b851 - hx;
  if ((i2 | j2) > 0) {
    const hfsq = 0.5 * f * f;
    if (k === 0) return f - (hfsq - s * (hfsq + r));
    return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f);
  } else if (k === 0) return f - s * (f - r);
  return dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f);
}

function grid() {
  const g = [];
  for (let i = -1900; i <= 1900; i++) g.push(i * 0.37);
  for (let e = -60; e < 60; e++) { g.push(twoPow(e)); g.push(-twoPow(e)); g.push(1.0 + twoPow(e)); }
  const specials = [0.0, -0.0, 1.0, -1.0, 709.0, -745.0, 710.0, -746.0,
    fromBits(1n), fromBits(0x000fffffffffffffn), 1e300, 1e-300, Infinity, -Infinity, NaN];
  for (const v of specials) g.push(v);
  return g;
}

function hexout(v) {
  if (Number.isNaN(v)) return 'NaN';
  return toBits(v).toString(16).padStart(16, '0');
}

let out = '';
for (const x of grid()) {
  const xb = toBits(x).toString(16).padStart(16, '0');
  out += `exp ${xb} ${hexout(xexp(x))}\n`;
  out += `ln ${xb} ${hexout(xln(x))}\n`;
}
process.stdout.write(out);
