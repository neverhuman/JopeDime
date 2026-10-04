//! Explicit train-fold scalar calibration, separate from legacy search/scoring.
//! A winner stores its typed expression, input width and exact slope/intercept
//! bits. The affine epilogue is two binary64 operations, without a legacy model's
//! [-60, 60] prediction clip. Nonfinite fits/predictions are structured refusals.
//! No VarPro, implicit selection, new scalar IDs or Baseline promotion.
use super::codec::{identity, Reader, Writer};
use super::wire_v2::{VersionError, VersionResult, MAGIC_V2, READER_VERSION};
use super::{Error, Program, Result};
use crate::math_expr::MathExpr;
use crate::{GenomeIr, MAX_WIDTH};

mod emit;
mod fit;
pub use emit::ScaledArtifacts;

pub const PROFILE: &str = "feat-math.scalar-linear-scaling.v2";
const VARIANT: u8 = 2;
const FEATURE: u64 = 16;

#[derive(Clone, Debug)]
pub struct LinearScaledWinner {
    expression: MathExpr,
    input_width: usize,
    slope: f64,
    intercept: f64,
}

#[derive(Clone, Debug)]
pub struct LinearScalingFit {
    pub winner: LinearScaledWinner,
    /// Rescored on precisely the supplied training fold, with persisted coefficients.
    pub training_mse: f64,
}

impl LinearScaledWinner {
    /// Call explicitly after structural scoring. There is no outer-fold/cut
    /// object, callback, RNG or default legacy-scoring change in this API.
    pub fn fit_training(
        expression: &MathExpr,
        training_rows: &[Vec<f64>],
        training_targets: &[f64],
    ) -> Result<LinearScalingFit> {
        fit::fit_training(expression, training_rows, training_targets)
    }

    /// [slope a, intercept b], always finite; codec preserves their exact bits.
    pub fn coefficients(&self) -> [f64; 2] {
        [self.slope, self.intercept]
    }

    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>> {
        check_rows(rows, self.input_width)?;
        let compiled = self
            .expression
            .compile()
            .map_err(|e| Error::Domain(e.to_string()))?;
        rows.iter()
            .map(|row| {
                // Preserve the base expression's legacy protected value,
                // including its hard-error zero fallback, before calibration.
                let (value, _) = compiled.eval_row_traced(row);
                let scaled = self.slope * value + self.intercept;
                if scaled.is_finite() {
                    Ok(scaled)
                } else {
                    Err(Error::NonFinite("linear-scaled prediction".into()))
                }
            })
            .collect()
    }

    fn raw_program(&self) -> Result<Program> {
        Ok(Program::import_legacy(&GenomeIr::Symbolic {
            source: self.expression.to_source()?,
            expr: self.expression.clone(),
            affine: false,
        }))
    }

    /// FMKPRG v2 variant 2, minimum reader 2, required scaling feature 16.
    /// The embedded raw expression uses its unchanged typed v1 codec.
    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        validate(self)?;
        let body = self.raw_program()?.to_bytes()?;
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(VARIANT)?;
        writer.byte(READER_VERSION)?;
        writer.word(FEATURE)?;
        writer.word(self.input_width as u64)?;
        writer.word(self.slope.to_bits())?;
        writer.word(self.intercept.to_bits())?;
        writer.word(body.len() as u64)?;
        writer.raw(&body)?;
        Ok(writer.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(MAGIC_V2.len())? != MAGIC_V2
            || reader.byte()? != VARIANT
            || reader.byte()? != READER_VERSION
            || reader.word()? != FEATURE
        {
            return Err(VersionError::UnsupportedVersion(
                "linear-scaling envelope/features".into(),
            ));
        }
        let input_width = usize::try_from(reader.word()?)
            .map_err(|_| VersionError::Malformed("linear-scaling width".into()))?;
        let slope = f64::from_bits(reader.word()?);
        let intercept = f64::from_bits(reader.word()?);
        let len = reader.length()?;
        let body = reader.take(len)?;
        let Program::LegacyV1(GenomeIr::Symbolic {
            expr,
            affine: false,
            ..
        }) = Program::from_bytes(body)?
        else {
            return Err(VersionError::Malformed(
                "linear-scaling requires a raw scalar tree".into(),
            ));
        };
        reader.finish()?;
        let winner = Self {
            expression: expr,
            input_width,
            slope,
            intercept,
        };
        validate(&winner)?;
        if winner.raw_program()?.to_bytes()? != body {
            return Err(VersionError::Malformed(
                "noncanonical raw scalar tree".into(),
            ));
        }
        Ok(winner)
    }

    pub fn identity(&self) -> VersionResult<String> {
        Ok(identity(PROFILE, &self.to_bytes()?))
    }
}

fn validate(winner: &LinearScaledWinner) -> Result<()> {
    if winner.input_width == 0 || winner.input_width > MAX_WIDTH {
        return Err(Error::Shape("linear-scaling input width".into()));
    }
    if !winner.slope.is_finite() || !winner.intercept.is_finite() {
        return Err(Error::NonFinite("linear-scaling coefficients".into()));
    }
    winner.expression.to_source()?;
    let features = winner.expression.ast().features();
    if features.uses_x
        || features
            .max_col
            .is_some_and(|col| col >= winner.input_width)
    {
        return Err(Error::Shape("linear-scaling row expression scope".into()));
    }
    Ok(())
}

fn check_rows(rows: &[Vec<f64>], width: usize) -> Result<()> {
    if rows.iter().any(|row| row.len() != width) {
        return Err(Error::Shape("linear-scaling row width".into()));
    }
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::NonFinite("linear-scaling input".into()));
    }
    Ok(())
}
