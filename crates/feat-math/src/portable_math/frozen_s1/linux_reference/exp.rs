use super::exp_tables::*;
use super::trig_tables::TOINT;

pub(super) fn exp(x: f64) -> f64 {
    if !x.is_finite() || x.abs() > 500.0 {
        return x.exp();
    }
    if x.abs() < f64::from_bits(0x3c90000000000000) {
        return 1.0 + x;
    }
    let z = x * f64::from_bits(0x40671547652b82fe);
    let shifted = z + TOINT;
    let ki = shifted.to_bits();
    let kd = shifted - TOINT;
    let r = kd.mul_add(
        f64::from_bits(0xbd0cf79abc9e3b3a),
        kd.mul_add(f64::from_bits(0xbf762e42fefa0000), x),
    );
    let idx = 2 * ((ki % 128) as usize);
    let tail = f64::from_bits(ET[idx]);
    let scale = f64::from_bits(ET[idx + 1].wrapping_add(ki << 45));
    let r2 = r * r;
    let tmp = (r2 * r2).mul_add(
        r.mul_add(C[3], C[2]),
        r2.mul_add(r.mul_add(C[1], C[0]), tail + r),
    );
    scale.mul_add(tmp, scale)
}
const SN3: f64 = f64::from_bits(0xbfc5555555555515);
const SN5: f64 = f64::from_bits(0x3f811110e829872f);
const CS2: f64 = f64::from_bits(0x3fe0000000000000);
const CS4: f64 = f64::from_bits(0xbfa5555555555535);
const CS6: f64 = f64::from_bits(0x3f56c16bedd9e239);
