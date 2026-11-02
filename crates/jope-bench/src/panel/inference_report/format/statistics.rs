use super::{number, object, py_order, Json, JsonNumber, Ordering};
use anyhow::{bail, Context, Result};

// ---------------------------------------------------------------------------------------------
// `statistics` with exact rational rounding.

/// An unsigned fixed-point magnitude in units of 2^-1074, wide enough for any finite f64 or
/// JSON integer and sums of them.
#[derive(Clone)]
struct Wide([u64; 40]);

impl Wide {
    const SCALE: u32 = 1074;
    fn zero() -> Self {
        Wide([0; 40])
    }
    fn add_shifted(&mut self, value: u64, shift: u32) {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = value << offset;
        let high = if offset == 0 {
            0
        } else {
            value >> (64 - offset)
        };
        self.add_at(limb, low);
        self.add_at(limb + 1, high);
    }
    fn add_at(&mut self, mut limb: usize, value: u64) {
        let mut carry = value;
        while carry != 0 {
            let (sum, overflow) = self.0[limb].overflowing_add(carry);
            self.0[limb] = sum;
            carry = u64::from(overflow);
            limb += 1;
        }
    }
    fn add_float(&mut self, x: f64) {
        let bits = x.abs().to_bits();
        let exponent = (bits >> 52) as u32;
        let fraction = bits & ((1 << 52) - 1);
        if exponent == 0 {
            self.add_shifted(fraction, 0);
        } else {
            self.add_shifted(fraction | (1 << 52), exponent - 1);
        }
    }
    fn add_int(&mut self, magnitude: u128) {
        self.add_shifted(magnitude as u64, Self::SCALE);
        self.add_shifted((magnitude >> 64) as u64, Self::SCALE + 64);
    }
    fn compare(&self, other: &Wide) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
    /// `self - other`, for `self >= other`.
    fn minus(&self, other: &Wide) -> Wide {
        let mut result = Wide::zero();
        let mut borrow = false;
        for (limb, (a, b)) in self.0.iter().zip(&other.0).enumerate() {
            let (difference, first) = a.overflowing_sub(*b);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            result.0[limb] = difference;
            borrow = first || second;
        }
        result
    }
    /// Divide in place; returns the remainder.
    fn divide(&mut self, divisor: u64) -> u64 {
        let mut remainder = 0_u128;
        for limb in self.0.iter_mut().rev() {
            let current = (remainder << 64) | u128::from(*limb);
            *limb = (current / u128::from(divisor)) as u64;
            remainder = current % u128::from(divisor);
        }
        remainder as u64
    }
    fn bits(&self) -> u32 {
        self.0
            .iter()
            .rposition(|limb| *limb != 0)
            .map_or(0, |top| top as u32 * 64 + 64 - self.0[top].leading_zeros())
    }
    fn bit(&self, position: u32) -> bool {
        (self.0[(position / 64) as usize] >> (position % 64)) & 1 == 1
    }
    fn any_below(&self, position: u32) -> bool {
        let (limb, offset) = ((position / 64) as usize, position % 64);
        self.0[..limb].iter().any(|l| *l != 0) || self.0[limb] & ((1 << offset) - 1) != 0
    }
    /// The 64 bits starting at `shift`.
    fn window(&self, shift: u32) -> u64 {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = self.0[limb] >> offset;
        if offset == 0 || limb + 1 >= self.0.len() {
            low
        } else {
            low | (self.0[limb + 1] << (64 - offset))
        }
    }
    /// `self / divisor` × 2^-1074, rounded half to even.
    fn quotient(mut self, divisor: u64) -> f64 {
        let remainder = self.divide(divisor);
        let length = self.bits();
        let (mantissa, shift, up) = if length <= 53 {
            let mantissa = self.window(0);
            let twice = 2 * u128::from(remainder);
            let half = u128::from(divisor);
            (
                mantissa,
                0,
                twice > half || (twice == half && mantissa & 1 == 1),
            )
        } else {
            let shift = length - 53;
            let mantissa = self.window(shift) & ((1 << 53) - 1);
            let sticky = self.any_below(shift - 1) || remainder != 0;
            (
                mantissa,
                shift,
                self.bit(shift - 1) && (sticky || mantissa & 1 == 1),
            )
        };
        (mantissa + u64::from(up)) as f64 * power_of_two(shift as i32 - Self::SCALE as i32)
    }
}

pub(super) fn power_of_two(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    }
}

/// The correctly rounded quotient of an exact signed sum and a count.
pub(super) fn rational(values: &[JsonNumber], count: u64) -> f64 {
    let (mut positive, mut negative) = (Wide::zero(), Wide::zero());
    for value in values {
        match *value {
            JsonNumber::Int(i) if i < 0 => negative.add_int(i.unsigned_abs()),
            JsonNumber::Int(i) => positive.add_int(i.unsigned_abs()),
            JsonNumber::Float(f) if f.is_sign_negative() => negative.add_float(f),
            JsonNumber::Float(f) => positive.add_float(f),
        }
    }
    match positive.compare(&negative) {
        Ordering::Less => -negative.minus(&positive).quotient(count),
        _ => positive.minus(&negative).quotient(count),
    }
}

/// `statistics.mean`: an exact mean, an int when every value is an int and it divides.
pub(super) fn mean(values: &[Json]) -> Result<Json> {
    let numbers = values
        .iter()
        .map(|v| number(v).context("TypeError: mean of a non-number"))
        .collect::<Result<Vec<_>>>()?;
    let count = numbers.len() as i128;
    if numbers.iter().all(|n| matches!(n, JsonNumber::Int(_))) {
        let total: i128 = numbers
            .iter()
            .map(|n| match n {
                JsonNumber::Int(i) => *i,
                JsonNumber::Float(_) => unreachable!(),
            })
            .sum();
        if total % count == 0 {
            return Ok(Json::Int(total / count));
        }
    }
    Ok(Json::Float(rational(&numbers, count as u64)))
}

/// `statistics.median` of an even-length sorted pair: `(a + b) / 2`.
pub(super) fn midpoint(a: &Json, b: &Json) -> Result<Json> {
    let (Some(x), Some(y)) = (number(a), number(b)) else {
        bail!("TypeError: median of a non-number");
    };
    Ok(Json::Float(match (x, y) {
        (JsonNumber::Int(i), JsonNumber::Int(j)) => rational(&[JsonNumber::Int(i + j)], 2),
        _ => (x.to_f64() + y.to_f64()) / 2.0,
    }))
}

/// Sample count, median, 95th-percentile order statistic and mean, or nulls without samples.
pub(super) fn distribution(values: &[Json]) -> Result<Json> {
    let mut sorted = values.to_vec();
    let mut failure = None;
    sorted.sort_by(|a, b| {
        py_order(a, b).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            Ordering::Equal
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let n = sorted.len();
    if n == 0 {
        return Ok(object([
            ("n", 0_usize.into()),
            ("median_seconds", Json::Null),
            ("p95_seconds", Json::Null),
            ("mean_seconds", Json::Null),
        ]));
    }
    let median = if n % 2 == 1 {
        sorted[n / 2].clone()
    } else {
        midpoint(&sorted[n / 2 - 1], &sorted[n / 2])?
    };
    let p95 = sorted[(0.95 * n as f64).ceil() as usize - 1].clone();
    Ok(object([
        ("n", n.into()),
        ("median_seconds", median),
        ("p95_seconds", p95),
        ("mean_seconds", mean(&sorted)?),
    ]))
}
