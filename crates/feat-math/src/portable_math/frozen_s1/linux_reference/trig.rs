use super::trig_tables::*;

fn table(u: f64) -> (f64, f64, f64, f64) {
    let k = ((u.to_bits() as u32) as usize) * 4;
    (ST[k], ST[k + 1], ST[k + 2], ST[k + 3])
}
fn cos_reduced(x: f64, mut dx: f64) -> f64 {
    if x < 0.0 {
        dx = -dx;
    }
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG) + dx;
    let xx = x * x;
    let s = (x * xx).mul_add(xx.mul_add(SN5, SN3), x);
    let c = xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2);
    let (sn, ssn, cs, ccs) = table(u);
    let cor = (-sn).mul_add(s, (-cs).mul_add(c, (-s).mul_add(ssn, ccs)));
    cs + cor
}
fn sin_reduced(x: f64, mut dx: f64) -> f64 {
    let xx = x * x;
    if x.abs() < 0.126 {
        let p = S5
            .mul_add(xx, S4)
            .mul_add(xx, S3)
            .mul_add(xx, S2)
            .mul_add(xx, S1);
        return x + p.mul_add(x, -0.5 * dx).mul_add(xx, dx);
    }
    let old = x;
    if x <= 0.0 {
        dx = -dx;
    }
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG);
    let xx = x * x;
    let s = x + (x * xx).mul_add(xx.mul_add(SN5, SN3), dx);
    let c = x.mul_add(dx, xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2));
    let (sn, ssn, cs, ccs) = table(u);
    let cor = cs.mul_add(s, (-sn).mul_add(c, s.mul_add(ccs, ssn)));
    (sn + cor).copysign(old)
}
fn reduce(x: f64) -> (f64, f64, i32) {
    let t = x.mul_add(HPINV, TOINT);
    let xn = t - TOINT;
    let y = (-xn).mul_add(MP2, (-xn).mul_add(MP1, x));
    let t1 = xn * PP3;
    let t2 = y - t1;
    let db = (y - t2) - t1;
    let t1 = xn * PP4;
    let b = t2 - t1;
    let db = db + ((t2 - b) - t1);
    (b, db, t.to_bits() as i32 & 3)
}
fn quadrant(a: f64, da: f64, n: i32) -> f64 {
    let y = if n & 1 == 1 {
        cos_reduced(a, da)
    } else {
        sin_reduced(a, da)
    };
    if n & 2 == 2 {
        -y
    } else {
        y
    }
}
pub(crate) fn sin(x: f64) -> f64 {
    let k = (x.to_bits() >> 32) as u32 & 0x7fffffff;
    if k < 0x3e500000 {
        x
    } else if k < 0x3feb6000 {
        sin_reduced(x, 0.0)
    } else if k < 0x400368fd {
        cos_reduced(HP0 - x.abs(), HP1).copysign(x)
    } else if k < 0x419921fb {
        let (a, da, n) = reduce(x);
        quadrant(a, da, n)
    } else {
        x.sin()
    }
}
pub(crate) fn cos(x: f64) -> f64 {
    let k = (x.to_bits() >> 32) as u32 & 0x7fffffff;
    if k < 0x3e400000 {
        1.0
    } else if k < 0x3feb6000 {
        cos_reduced(x, 0.0)
    } else if k < 0x400368fd {
        let y = HP0 - x.abs();
        let a = y + HP1;
        let da = (y - a) + HP1;
        sin_reduced(a, da)
    } else if k < 0x419921fb {
        let (a, da, n) = reduce(x);
        quadrant(a, da, n + 1)
    } else {
        x.cos()
    }
}
