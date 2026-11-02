use super::log_tables::*;

pub(crate) fn ln(x: f64) -> f64 {
    let mut ix = x.to_bits();
    if ix.wrapping_sub((1.0f64 - 0.0625).to_bits())
        < (1.0f64 + f64::from_bits(0x3fb0900000000000)).to_bits() - (1.0f64 - 0.0625).to_bits()
    {
        if x == 1.0 {
            return 0.0;
        }
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        let inner = r3.mul_add(B[10], r2.mul_add(B[9], r.mul_add(B[8], B[7])));
        let middle = r3.mul_add(inner, r2.mul_add(B[6], r.mul_add(B[5], B[4])));
        let polynomial = r3.mul_add(middle, r2.mul_add(B[3], r.mul_add(B[2], B[1])));
        let w = r * 134217728.0;
        let rhi = (r + w) - w;
        let rlo = r - rhi;
        let w = rhi * rhi * B[0];
        let hi = r + w;
        let lo = (B[0] * rlo).mul_add(rhi + r, (r - hi) + w);
        // The reference contracts this product with the low correction.
        // Rounding before adding lo changes rare inputs near one.
        return r3.mul_add(polynomial, lo) + hi;
    }
    if !x.is_finite() || x <= 0.0 {
        return x.ln();
    }
    if x < f64::MIN_POSITIVE {
        ix = (x * 4503599627370496.0).to_bits().wrapping_sub(52u64 << 52);
    }
    let tmp = ix.wrapping_sub(0x3fe6000000000000);
    let i = ((tmp >> 45) % 128) as usize;
    let k = (tmp as i64) >> 52;
    let z = f64::from_bits(ix.wrapping_sub(tmp & (0xfffu64 << 52)));
    let r = z.mul_add(INVC[i], -1.0);
    let kd = k as f64;
    let w = kd.mul_add(f64::from_bits(0x3fe62e42fefa3800), LOGC[i]);
    let hi = w + r;
    let lo = kd.mul_add(f64::from_bits(0x3d2ef35793c76730), (w - hi) + r);
    let r2 = r * r;
    let p = r2.mul_add(r.mul_add(A[4], A[3]), r.mul_add(A[2], A[1]));
    (r * r2).mul_add(p, r2.mul_add(A[0], lo)) + hi
}
