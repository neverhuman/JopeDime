//! Fault classes of the scalar evaluator (kernel2 WP-1.1 to WP-1.3).
//!
//! A [`FaultSet`] records which guarded branches an evaluation took. It never changes a value:
//! every guard still returns what it returned before, and the set only says that it fired. The
//! guards, their fallbacks and their classes are listed in
//! `docs/internal/kernel2/math-domain-table.md`.
//!
//! Bit positions are append-only. A class keeps its bit, its name and its severity forever; a new
//! class takes the next free bit.
//!
//! Faults accumulate in evaluation order until the evaluation ends. A Hard class ends it: the
//! legacy entry point returns an error at that point. Every evaluation error therefore carries
//! exactly one Hard class, together with the faults raised before it, and a Hard class is raised
//! only together with an error.

use super::*;
use std::fmt;

/// How serious a fault class is. Ordered from the mildest to the most serious.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaultSeverity {
    /// A value was clamped to a bound; the result is still a number.
    Saturation,
    /// An operation met an input outside its mathematical domain, or a non-finite value, and a
    /// guard substituted its fallback value.
    Domain,
    /// The expression cannot be evaluated on this input; the legacy entry point returns an error.
    Hard,
}

impl FaultSeverity {
    /// The stable name of the severity.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Saturation => "saturation",
            Self::Domain => "domain",
            Self::Hard => "hard",
        }
    }

    /// Every class of this severity.
    pub fn classes(self) -> FaultSet {
        FaultClass::ALL
            .into_iter()
            .filter(|class| class.severity() == self)
            .collect()
    }
}

impl fmt::Display for FaultSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One fault class. Its discriminant is its bit position in a [`FaultSet`].
#[non_exhaustive]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaultClass {
    /// A divisor with `|b| < 1e-12`; the quotient is 0.0.
    DivNearZero = 0,
    /// A `log` argument `<= 1e-300`; the result is 0.0.
    LogDomain = 1,
    /// A `log1p` argument `<= -1`; the result is 0.0.
    Log1pDomain = 2,
    /// A `sqrt` argument below zero or NaN, projected to 0.0 first. `-0.0` is not a fault.
    SqrtProjected = 3,
    /// `^` returned a non-finite value for a finite base and a finite (clamped) exponent.
    PowNonfinite = 4,
    /// A `^` exponent outside `[-8, 8]`, clamped to it.
    PowExpClamped = 5,
    /// An `exp` argument outside `[-30, 30]`, clamped to it.
    ExpArgClamped = 6,
    /// A `sigmoid` argument outside `[-30, 30]`, clamped to it.
    SigmoidArgClamped = 7,
    /// The output filter replaced a non-finite result with 0.0.
    NonfiniteSanitized = 8,
    /// The output filter clamped a finite result to `±1e12`.
    MagnitudeClamped = 9,
    /// Both `clip` bounds evaluated to NaN.
    InvalidClipBounds = 10,
    /// A column index past the end of the row.
    ColumnOob = 11,
    /// `x` read outside an elementwise map.
    IllegalScope = 12,
    /// A non-finite input cell was read: `x`, a `col(i)`, or any cell of a row reduction.
    NonfiniteInput = 13,
    /// `row_mean`, `row_std`, `row_min` or `row_max` of an empty row.
    EmptyRowReduction = 14,
    /// The expression is structurally invalid: `compile()` refused it (reported by the stage),
    /// or the interpreter met a function with the wrong number of arguments, which only a
    /// deserialized AST can hold.
    CompileFailed = 15,
}

impl FaultClass {
    /// Every class, in bit order.
    pub const ALL: [FaultClass; 16] = [
        Self::DivNearZero,
        Self::LogDomain,
        Self::Log1pDomain,
        Self::SqrtProjected,
        Self::PowNonfinite,
        Self::PowExpClamped,
        Self::ExpArgClamped,
        Self::SigmoidArgClamped,
        Self::NonfiniteSanitized,
        Self::MagnitudeClamped,
        Self::InvalidClipBounds,
        Self::ColumnOob,
        Self::IllegalScope,
        Self::NonfiniteInput,
        Self::EmptyRowReduction,
        Self::CompileFailed,
    ];

    /// The class's bit position.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The set that holds only this class.
    pub const fn set(self) -> FaultSet {
        FaultSet(1 << self as u32)
    }

    /// The class's severity. Each class has exactly one.
    pub const fn severity(self) -> FaultSeverity {
        match self {
            Self::InvalidClipBounds
            | Self::ColumnOob
            | Self::IllegalScope
            | Self::CompileFailed => FaultSeverity::Hard,
            Self::PowExpClamped
            | Self::ExpArgClamped
            | Self::SigmoidArgClamped
            | Self::MagnitudeClamped => FaultSeverity::Saturation,
            Self::DivNearZero
            | Self::LogDomain
            | Self::Log1pDomain
            | Self::SqrtProjected
            | Self::PowNonfinite
            | Self::NonfiniteSanitized
            | Self::NonfiniteInput
            | Self::EmptyRowReduction => FaultSeverity::Domain,
        }
    }

    /// The class's stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DivNearZero => "DIV_NEAR_ZERO",
            Self::LogDomain => "LOG_DOMAIN",
            Self::Log1pDomain => "LOG1P_DOMAIN",
            Self::SqrtProjected => "SQRT_PROJECTED",
            Self::PowNonfinite => "POW_NONFINITE",
            Self::PowExpClamped => "POW_EXP_CLAMPED",
            Self::ExpArgClamped => "EXP_ARG_CLAMPED",
            Self::SigmoidArgClamped => "SIGMOID_ARG_CLAMPED",
            Self::NonfiniteSanitized => "NONFINITE_SANITIZED",
            Self::MagnitudeClamped => "MAGNITUDE_CLAMPED",
            Self::InvalidClipBounds => "INVALID_CLIP_BOUNDS",
            Self::ColumnOob => "COLUMN_OOB",
            Self::IllegalScope => "ILLEGAL_SCOPE",
            Self::NonfiniteInput => "NONFINITE_INPUT",
            Self::EmptyRowReduction => "EMPTY_ROW_REDUCTION",
            Self::CompileFailed => "COMPILE_FAILED",
        }
    }
}

impl fmt::Display for FaultClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`FaultClass`]es, one bit per class at the class's [`FaultClass::index`].
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaultSet(u32);

impl FaultSet {
    /// No fault.
    pub const EMPTY: FaultSet = FaultSet(0);
    /// Every class this build knows.
    pub const ALL: FaultSet = {
        let mut bits = 0;
        let mut i = 0;
        while i < FaultClass::ALL.len() {
            bits |= FaultClass::ALL[i].set().0;
            i += 1;
        }
        FaultSet(bits)
    };

    // Each class alone, under its stable name. See `FaultClass` for what each one means.
    pub const DIV_NEAR_ZERO: FaultSet = FaultClass::DivNearZero.set();
    pub const LOG_DOMAIN: FaultSet = FaultClass::LogDomain.set();
    pub const LOG1P_DOMAIN: FaultSet = FaultClass::Log1pDomain.set();
    pub const SQRT_PROJECTED: FaultSet = FaultClass::SqrtProjected.set();
    pub const POW_NONFINITE: FaultSet = FaultClass::PowNonfinite.set();
    pub const POW_EXP_CLAMPED: FaultSet = FaultClass::PowExpClamped.set();
    pub const EXP_ARG_CLAMPED: FaultSet = FaultClass::ExpArgClamped.set();
    pub const SIGMOID_ARG_CLAMPED: FaultSet = FaultClass::SigmoidArgClamped.set();
    pub const NONFINITE_SANITIZED: FaultSet = FaultClass::NonfiniteSanitized.set();
    pub const MAGNITUDE_CLAMPED: FaultSet = FaultClass::MagnitudeClamped.set();
    pub const INVALID_CLIP_BOUNDS: FaultSet = FaultClass::InvalidClipBounds.set();
    pub const COLUMN_OOB: FaultSet = FaultClass::ColumnOob.set();
    pub const ILLEGAL_SCOPE: FaultSet = FaultClass::IllegalScope.set();
    pub const NONFINITE_INPUT: FaultSet = FaultClass::NonfiniteInput.set();
    pub const EMPTY_ROW_REDUCTION: FaultSet = FaultClass::EmptyRowReduction.set();
    pub const COMPILE_FAILED: FaultSet = FaultClass::CompileFailed.set();

    /// The set's bits: bit `i` is the class whose [`FaultClass::index`] is `i`.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The set with these bits, or `None` when a bit names no class of this build.
    pub const fn from_bits(bits: u32) -> Option<FaultSet> {
        if bits & !Self::ALL.0 == 0 {
            Some(FaultSet(bits))
        } else {
            None
        }
    }

    /// Every class in either set.
    pub const fn union(self, other: FaultSet) -> FaultSet {
        FaultSet(self.0 | other.0)
    }

    /// Whether every class of `other` is in this set.
    pub const fn contains(self, other: FaultSet) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the two sets share a class.
    pub const fn intersects(self, other: FaultSet) -> bool {
        self.0 & other.0 != 0
    }

    /// Whether the set holds no class.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// How many classes the set holds.
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// The classes in the set, in bit order.
    pub fn iter(self) -> impl Iterator<Item = FaultClass> + Clone {
        FaultClass::ALL
            .into_iter()
            .filter(move |class| self.contains(class.set()))
    }

    /// The most serious severity in the set, or `None` when it is empty.
    pub fn severity(self) -> Option<FaultSeverity> {
        self.iter().map(FaultClass::severity).max()
    }
}

impl From<FaultClass> for FaultSet {
    fn from(class: FaultClass) -> Self {
        class.set()
    }
}

impl FromIterator<FaultClass> for FaultSet {
    fn from_iter<I: IntoIterator<Item = FaultClass>>(classes: I) -> Self {
        classes
            .into_iter()
            .fold(FaultSet::EMPTY, |set, class| set.union(class.set()))
    }
}

/// The class names in bit order, as `{DIV_NEAR_ZERO, NONFINITE_SANITIZED}`; `{}` when empty.
impl fmt::Display for FaultSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (i, class) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(class.name())?;
        }
        f.write_str("}")
    }
}

impl fmt::Debug for FaultSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FaultSet({self})")
    }
}

/// Where the evaluator reports the guards it takes.
pub(crate) trait FaultSink {
    /// `false` when raised faults are discarded. Detection work beyond a guard's own test, such
    /// as scanning a row for non-finite cells, is skipped then, so the legacy entry points pay
    /// nothing for it.
    const ENABLED: bool;

    fn raise(&mut self, faults: FaultSet);
}

/// The legacy entry points' sink: it discards every fault.
pub(crate) struct NoSink;

impl FaultSink for NoSink {
    const ENABLED: bool = false;

    #[inline(always)]
    fn raise(&mut self, _faults: FaultSet) {}
}

/// A sink that keeps every fault raised into it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Collect(pub(crate) FaultSet);

impl FaultSink for Collect {
    const ENABLED: bool = true;

    #[inline]
    fn raise(&mut self, faults: FaultSet) {
        self.0 = self.0.union(faults);
    }
}

impl MathExpr {
    /// [`MathExpr::eval_row`] with the faults its evaluation raised, in the compat shape: the
    /// value is `eval_row(row).unwrap_or(0.0)`, bit for bit, as the stage fallback computes it.
    /// When `eval_row` returns an error, the value is 0.0 and the faults hold exactly one Hard
    /// class, with every fault raised before it.
    pub fn eval_row_traced(&self, row: &[f64]) -> (f64, FaultSet) {
        compat_shape(self.eval_traced(None, row))
    }

    /// [`MathExpr::eval_element`] with the faults its evaluation raised, in the compat shape of
    /// [`MathExpr::eval_row_traced`]: the value is `eval_element(x, row).unwrap_or(0.0)`.
    pub fn eval_element_traced(&self, x: f64, row: &[f64]) -> (f64, FaultSet) {
        compat_shape(self.eval_traced(Some(x), row))
    }

    /// [`MathExpr::eval_row`] with the faults its evaluation raised, for callers that refuse Hard
    /// faults: exactly when `eval_row` returns an error, this returns that error, unchanged. The
    /// faults of a failed evaluation are those of [`MathExpr::eval_row_traced`].
    pub fn eval_row_traced_strict(&self, row: &[f64]) -> Result<(f64, FaultSet)> {
        let (value, faults) = self.eval_traced(None, row);
        Ok((value?, faults))
    }

    /// `eval_row` (`x` is `None`) or `eval_element` (`x` is the element) with a collecting sink.
    fn eval_traced(&self, x: Option<f64>, row: &[f64]) -> (Result<f64>, FaultSet) {
        let ctx = EvalContext { x, row };
        let mut sink = Collect::default();
        let value = compute_expr_with(&self.root, &ctx, &mut sink).map(|v| safe_with(v, &mut sink));
        (value, sink.0)
    }
}

/// The stage fallback's shape: an evaluation error is the value 0.0, and keeps its faults.
fn compat_shape((value, faults): (Result<f64>, FaultSet)) -> (f64, FaultSet) {
    (value.unwrap_or(0.0), faults)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_legacy_sink_is_disabled_and_the_collecting_sink_accumulates() {
        const { assert!(!NoSink::ENABLED && Collect::ENABLED) };
        let mut none = NoSink;
        none.raise(FaultSet::ALL);
        let mut collect = Collect::default();
        assert_eq!(collect.0, FaultSet::EMPTY);
        collect.raise(FaultSet::DIV_NEAR_ZERO);
        collect.raise(FaultSet::NONFINITE_SANITIZED);
        collect.raise(FaultSet::DIV_NEAR_ZERO);
        assert_eq!(
            collect.0,
            FaultSet::DIV_NEAR_ZERO.union(FaultSet::NONFINITE_SANITIZED)
        );
    }
}
