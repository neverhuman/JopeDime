//! B13's sparse Goertzel bank, v2 wire 61, in FeatureWithinRow mode only: each configured
//! normalized frequency `f` (cycles per feature position, `0 <= f <= 0.5`) yields the squared DFT
//! magnitude `|sum_n x_n exp(-2 pi i f n)|^2` of the row. A SampleTime mode needs its own later
//! admission; this mode never reads another observation.
//!
//! Geometry-only preparation freezes, per bin, `cosine = cos(2 pi f)` and `sine = sin(2 pi f)` from
//! the portable fdlibm kernel (`angle = TAU * f`, one multiplication), so the bits are identical on
//! every host; the endpoints are exact (`f = 0`: cosine `+1`, sine `+0`; `f = 0.5`: cosine `-1`,
//! sine `+0`). Runtime and every artifact read those bits and call no trigonometric function.
//!
//! Frozen arithmetic order, for each bin and each row: every input is protected first (non-finite
//! to `+0` with NONFINITE_INPUT, `|x| > 1e12` clamped with MAGNITUDE_CLAMPED); `c = 2 * cosine`;
//! with `q1 = q2 = +0`, for each input in increasing column order `term = c * q1`,
//! `sum = x + term`, `q0 = sum - q2`, `q2 = q1`, `q1 = q0`; then `real = q1 - cosine * q2`,
//! `imaginary = sine * q2`, `power = real * real + imaginary * imaginary`. The sum of squares
//! cannot cancel to a negative value, unlike the traditional `q1^2 + q2^2 - c q1 q2`, and keeps
//! the power of near-DC bins (`f = 1e-9` on a width-512 alternating row is about `2.587e-12`,
//! where the traditional form returns 0). A non-finite power is `+0` with NONFINITE_SANITIZED and
//! a power above `1e12` is clamped with MAGNITUDE_CLAMPED. Every bin carries the faults of the
//! row's inputs and its own output.
use super::ordered_axis::Axis;
use super::savgol::{normalize, project, Output};
use super::{Error, Result};
use serde::{Deserialize, Serialize};

pub const FEATURE_ID: &str = "feat-math.feature-goertzel.v2";
/// The most frequencies one bank holds.
pub const MAX_BINS: usize = 32;
/// The widest row a bank accepts.
pub const MAX_INPUT_WIDTH: usize = 512;

/// A bank of normalized frequencies, kept in the supplied order with duplicates, as exact bit
/// patterns. A signed zero is stored as `+0`, so each frequency has one representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    frequency_bits: Vec<u64>,
}

impl Config {
    pub fn new(version: u8, wire: u16, axis: Axis, frequencies: &[f64]) -> Result<Self> {
        if version != 2 || wire != 61 || axis != Axis::FeatureWithinRow {
            return Err(Error::Unsupported(
                "goertzel_bank requires v2 / wire 61 / FeatureWithinRow".into(),
            ));
        }
        let config = Self {
            frequency_bits: frequencies
                .iter()
                .map(|&f| (if f == 0.0 { 0.0_f64 } else { f }).to_bits())
                .collect(),
        };
        config.validate()?;
        Ok(config)
    }

    /// The bank's frequencies, in bank order.
    pub fn frequencies(&self) -> Vec<f64> {
        self.frequency_bits
            .iter()
            .map(|&b| f64::from_bits(b))
            .collect()
    }

    /// The bank length, which is the stage's output width.
    pub fn bins(&self) -> usize {
        self.frequency_bits.len()
    }

    /// Restored banks are checked here as well as at construction: one to [`MAX_BINS`] finite
    /// frequencies in `[0, 0.5]`, none written as `-0`.
    fn validate(&self) -> Result<()> {
        if self.frequency_bits.is_empty() || self.frequency_bits.len() > MAX_BINS {
            return Err(Error::Domain(
                "goertzel_bank requires 1..=32 frequencies".into(),
            ));
        }
        for &bits in &self.frequency_bits {
            let f = f64::from_bits(bits);
            if !(0.0..=0.5).contains(&f) || bits == (-0.0_f64).to_bits() {
                return Err(Error::Domain(
                    "goertzel_bank frequencies must be canonical finite values in [0, 0.5]".into(),
                ));
            }
        }
        Ok(())
    }

    /// Geometry-only preparation for rows of `width` features: no observation, label or fitted
    /// head state is read.
    pub fn fit(&self, width: usize) -> Result<Fitted> {
        self.validate()?;
        if width > MAX_INPUT_WIDTH {
            return Err(Error::Shape("goertzel_bank input width exceeds 512".into()));
        }
        let bins = self
            .frequency_bits
            .iter()
            .map(|&bits| Bin::prepare(f64::from_bits(bits)))
            .collect();
        Ok(Fitted {
            config: self.clone(),
            width,
            bins,
        })
    }
}

/// One bin's frozen trigonometric constants, as integer bit patterns in all serialized state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Bin {
    pub(crate) cosine_bits: u64,
    pub(crate) sine_bits: u64,
}

impl Bin {
    fn prepare(frequency: f64) -> Self {
        let (cosine, sine) = if frequency == 0.0 {
            (1.0, 0.0)
        } else if frequency == 0.5 {
            (-1.0, 0.0)
        } else {
            let angle = std::f64::consts::TAU * frequency;
            (
                crate::portable_math::cos(angle),
                crate::portable_math::sin(angle),
            )
        };
        Self {
            cosine_bits: f64::to_bits(cosine),
            sine_bits: f64::to_bits(sine),
        }
    }

    pub(crate) fn cosine(self) -> f64 {
        f64::from_bits(self.cosine_bits)
    }

    pub(crate) fn sine(self) -> f64 {
        f64::from_bits(self.sine_bits)
    }

    /// The frozen recurrence and component finalization over already protected inputs.
    fn power(self, inputs: &[f64]) -> f64 {
        let cosine = self.cosine();
        let c = 2.0 * cosine;
        let (mut q1, mut q2) = (0.0_f64, 0.0_f64);
        for &x in inputs {
            let term = c * q1;
            let sum = x + term;
            let q0 = sum - q2;
            q2 = q1;
            q1 = q0;
        }
        let real = q1 - cosine * q2;
        let imaginary = self.sine() * q2;
        real * real + imaginary * imaginary
    }
}

/// A prepared bank for one input width. Restoring checks the constants against the canonical
/// preparation; an arbitrary set of constants is not a Goertzel bank.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fitted {
    config: Config,
    width: usize,
    bins: Vec<Bin>,
}

impl Fitted {
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The input width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The output width, one power per bin.
    pub fn output_width(&self) -> usize {
        self.bins.len()
    }

    pub(crate) fn bins(&self) -> &[Bin] {
        &self.bins
    }

    pub fn validate(&self) -> Result<()> {
        if *self != self.config.fit(self.width)? {
            return Err(Error::Domain("noncanonical goertzel_bank constants".into()));
        }
        Ok(())
    }

    pub fn apply(&self, row: &[f64]) -> Result<Output> {
        self.validate()?;
        self.apply_validated(row)
    }

    pub(crate) fn apply_validated(&self, row: &[f64]) -> Result<Output> {
        if row.len() != self.width {
            return Err(Error::Shape(
                "goertzel_bank row width differs from fitted width".into(),
            ));
        }
        let mut inputs = Vec::with_capacity(row.len());
        let mut input_faults = 0;
        for &raw in row {
            let (x, fault) = normalize(raw);
            input_faults |= fault;
            inputs.push(x);
        }
        let mut values = Vec::with_capacity(self.bins.len());
        let mut faults = Vec::with_capacity(self.bins.len());
        for bin in &self.bins {
            let (power, fault) = project(bin.power(&inputs));
            values.push(power);
            faults.push(input_faults | fault);
        }
        Ok(Output { values, faults })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_expr::FaultSet;

    fn bank(frequencies: &[f64], width: usize) -> Fitted {
        Config::new(2, 61, Axis::FeatureWithinRow, frequencies)
            .unwrap()
            .fit(width)
            .unwrap()
    }

    fn powers(fitted: &Fitted, row: &[f64]) -> Vec<f64> {
        fitted.apply(row).unwrap().values
    }

    /// The squared magnitude of the plain DFT sum at `f`, an independent reference: direct
    /// platform `cos`/`sin` of `TAU * f * n` and no recurrence.
    fn dft_power(row: &[f64], f: f64) -> f64 {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &x) in row.iter().enumerate() {
            let angle = std::f64::consts::TAU * f * n as f64;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        re * re + im * im
    }

    /// The recurrence's rounding error grows like `N^2 eps` near DC and Nyquist; with the
    /// reference's own `N eps` error, `|power - reference|` stays below
    /// `16 N^2 eps (sum |x|)^2`.
    fn assert_matches_dft(fitted: &Fitted, row: &[f64]) {
        let mass: f64 = row.iter().map(|x| x.abs()).sum();
        let n = row.len().max(1) as f64;
        let bound = 16.0 * n * n * f64::EPSILON * mass * mass;
        for (power, f) in powers(fitted, row)
            .iter()
            .zip(fitted.config().frequencies())
        {
            let reference = dft_power(row, f);
            assert!(
                (power - reference).abs() <= bound,
                "f={f}: power {power} vs reference {reference} (bound {bound})"
            );
        }
    }

    fn tone(width: usize, f: f64, amplitude: f64, phase: f64) -> Vec<f64> {
        (0..width)
            .map(|n| amplitude * (std::f64::consts::TAU * f * n as f64 + phase).cos())
            .collect()
    }

    #[test]
    fn empty_and_one_element_rows_have_analytic_powers() {
        let frequencies = [0.0, 0.1, 0.25, 0.37, 0.5];
        let empty = bank(&frequencies, 0);
        let output = empty.apply(&[]).unwrap();
        assert!(output.values.iter().all(|p| p.to_bits() == 0));
        assert!(output.faults.iter().all(|&f| f == 0));
        // One input: q1 = x and q2 = 0, so real = x, imaginary = 0 and power = x * x exactly.
        for x in [3.0, -0.5, 1e-200, 7e5, -1e6] {
            let one = bank(&frequencies, 1);
            for power in powers(&one, &[x]) {
                assert_eq!(power.to_bits(), (x * x).to_bits(), "x={x}");
            }
        }
    }

    #[test]
    fn endpoint_bins_are_exact_sums_on_integer_rows() {
        // DC: real = q1 - q2 = sum x; Nyquist: real = q1 + q2 = alternating sum. Integer inputs
        // keep every intermediate exact.
        let row = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0];
        let fitted = bank(&[0.0, 0.5], row.len());
        let sum: f64 = row.iter().sum();
        let alternating: f64 = row
            .iter()
            .enumerate()
            .map(|(n, x)| if n % 2 == 0 { *x } else { -*x })
            .sum();
        assert_eq!(
            powers(&fitted, &row),
            [sum * sum, alternating * alternating]
        );
        // Width 512 alternating +/-1 has Nyquist power 512^2 and DC power 0.
        let row: Vec<f64> = (0..512)
            .map(|n| if n % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let fitted = bank(&[0.5, 0.0], 512);
        assert_eq!(powers(&fitted, &row), [262_144.0, 0.0]);
        let endpoint = fitted.bins();
        assert_eq!(
            (endpoint[0].cosine(), endpoint[0].sine().to_bits()),
            (-1.0, 0)
        );
        assert_eq!(
            (endpoint[1].cosine(), endpoint[1].sine().to_bits()),
            (1.0, 0)
        );
    }

    #[test]
    fn tones_and_mixtures_match_an_independent_dft() {
        for width in [8, 31, 64, 129, 512] {
            for &(f, amplitude, phase) in &[
                (0.125, 1.0, 0.0),
                (0.25, 2.0, 0.3),
                (3.0 / width as f64, 0.5, -1.1),
                (0.0731, 4.0, 2.0),
                (0.4999, 1.0, 0.7),
            ] {
                let row = tone(width, f, amplitude, phase);
                assert_matches_dft(&bank(&[f, 0.0, 0.2, 0.5, 0.05], width), &row);
            }
            let mixture: Vec<f64> = tone(width, 0.1, 1.0, 0.2)
                .iter()
                .zip(tone(width, 0.33, 0.25, -0.4))
                .map(|(a, b)| a + b + 0.5)
                .collect();
            assert_matches_dft(&bank(&[0.1, 0.33, 0.0, 0.123, 0.5], width), &mixture);
            // An impulse of height 2 has power 4 in every bin.
            let mut impulse = vec![0.0; width];
            impulse[width / 3] = 2.0;
            assert_matches_dft(&bank(&[0.0, 0.17, 0.5], width), &impulse);
        }
        // An integer bin of a pure tone carries (A N / 2)^2 = 64^2 here.
        let row = tone(64, 5.0 / 64.0, 2.0, 0.0);
        let power = powers(&bank(&[5.0 / 64.0], 64), &row)[0];
        assert!((power - 4096.0).abs() <= 1e-7, "integer-bin power {power}");
    }

    #[test]
    fn near_dc_bins_keep_the_power_the_traditional_form_loses() {
        // References: 80-digit DFTs of the width-512 alternating row (B13_GOERTZEL probe v1).
        let row: Vec<f64> = (0..512)
            .map(|n| if n % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        for (f, reference) in [
            (1e-12, 2.587_257_576_119_168_7e-18),
            (1e-9, 2.587_257_576_116_938e-12),
            (1e-6, 2.587_255_344_844_885_1e-6),
        ] {
            let power = powers(&bank(&[f], 512), &row)[0];
            assert!(
                ((power - reference) / reference).abs() < 1e-9,
                "f={f}: {power} vs {reference}"
            );
        }
    }

    #[test]
    fn inputs_and_outputs_are_protected_with_fault_bits() {
        let fitted = bank(&[0.0, 0.2], 4);
        let output = fitted
            .apply(&[f64::NAN, 2e12, f64::NEG_INFINITY, -0.0])
            .unwrap();
        let inputs = FaultSet::NONFINITE_INPUT.bits() | FaultSet::MAGNITUDE_CLAMPED.bits();
        // DC power of {0, 1e12, 0, 0} is 1e24, clamped to 1e12.
        assert_eq!(output.values[0], 1e12);
        assert_eq!(
            output.faults[0],
            inputs | FaultSet::MAGNITUDE_CLAMPED.bits()
        );
        assert!(output.faults.iter().all(|f| f & inputs == inputs));
        // Signed-zero and subnormal inputs pass unchanged and fault-free.
        let quiet = bank(&[0.0, 0.3], 3).apply(&[-0.0, 5e-324, 0.0]).unwrap();
        assert_eq!(quiet.faults, [0, 0]);
        assert!(quiet.values.iter().all(|p| p.is_finite() && *p >= 0.0));
        // A power is never -0: every power is a sum of two squares.
        let zero = bank(&[0.0, 0.5, 0.21], 2).apply(&[-0.0, -0.0]).unwrap();
        assert!(zero.values.iter().all(|p| p.to_bits() == 0));
    }

    #[test]
    fn banks_and_widths_outside_the_contract_refuse() {
        let ok = |f: &[f64]| Config::new(2, 61, Axis::FeatureWithinRow, f);
        assert!(matches!(ok(&[]), Err(Error::Domain(_))));
        assert!(matches!(ok(&[0.1; 33]), Err(Error::Domain(_))));
        assert!(ok(&[0.1; 32]).is_ok());
        for bad in [f64::NAN, -0.1, 0.5000000000000001, f64::INFINITY] {
            assert!(matches!(ok(&[0.2, bad]), Err(Error::Domain(_))), "{bad}");
        }
        assert!(matches!(
            Config::new(2, 58, Axis::FeatureWithinRow, &[0.1]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            Config::new(1, 61, Axis::FeatureWithinRow, &[0.1]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            Config::new(2, 61, Axis::SampleTime, &[0.1]),
            Err(Error::Unsupported(_))
        ));
        // A signed zero is stored as +0, so equal endpoints have one representation.
        assert_eq!(ok(&[-0.0]).unwrap(), ok(&[0.0]).unwrap());
        assert!(matches!(ok(&[0.1]).unwrap().fit(513), Err(Error::Shape(_))));
        let fitted = bank(&[0.1], 3);
        assert!(matches!(fitted.apply(&[1.0, 2.0]), Err(Error::Shape(_))));
        assert_eq!(fitted.output_width(), 1);
        assert_eq!(bank(&[0.1, 0.1, 0.4], 9).output_width(), 3);
    }

    #[test]
    fn restored_state_must_be_the_canonical_preparation() {
        let fitted = bank(&[0.3, 0.0, 0.3], 16);
        let json = serde_json::to_string(&fitted).unwrap();
        let restored: Fitted = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, fitted);
        restored.validate().unwrap();
        // Tampered constants, a written -0 frequency or an extra field all refuse.
        let mut forged = fitted.clone();
        forged.bins[1].sine_bits ^= 1;
        assert!(matches!(forged.validate(), Err(Error::Domain(_))));
        let mut negative_zero: serde_json::Value = serde_json::from_str(&json).unwrap();
        negative_zero["config"]["frequency_bits"][1] = (-0.0_f64).to_bits().into();
        let restored: Fitted = serde_json::from_value(negative_zero).unwrap();
        assert!(matches!(restored.validate(), Err(Error::Domain(_))));
        let extra = json.replacen("\"width\"", "\"unknown\":1,\"width\"", 1);
        assert!(serde_json::from_str::<Fitted>(&extra).is_err());
    }

    #[test]
    fn preparation_is_portable_and_frozen() {
        // Exact constants: the endpoints, and fdlibm cos/sin of TAU * 0.125 (= pi / 4 rounded).
        let fitted = bank(&[0.0, 0.5, 0.125, 0.25], 1);
        let bins: Vec<(u64, u64)> = fitted
            .bins()
            .iter()
            .map(|b| (b.cosine_bits, b.sine_bits))
            .collect();
        assert_eq!(bins[0], (1.0_f64.to_bits(), 0));
        assert_eq!(bins[1], ((-1.0_f64).to_bits(), 0));
        let quarter = std::f64::consts::TAU * 0.125;
        assert_eq!(
            bins[2],
            (
                crate::portable_math::cos(quarter).to_bits(),
                crate::portable_math::sin(quarter).to_bits()
            )
        );
        // cos(pi/2 rounded) is not zero; the bank keeps the true residue rather than a guess.
        assert!(f64::from_bits(bins[3].0) > 0.0 && f64::from_bits(bins[3].0) < 1e-16);
        // Repeated application is bit-identical.
        let row = tone(1, 0.1, 1.0, 0.0);
        assert_eq!(fitted.apply(&row).unwrap(), fitted.apply(&row).unwrap());
    }
}
