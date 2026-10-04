//! A causal ordered pre-stage followed by the existing numeric fitter.
use super::*;
use crate::kernel::codec::{identity, Reader, Writer};
use crate::kernel::wire_v2::{VersionError, VersionResult, MAGIC_V2, READER_VERSION};
use crate::{kernel::Program, GenomeIr};

pub const PROGRAM_VARIANT: u8 = 4;
pub const REQUIRED_FEATURE: u64 = 64;
pub const PROFILE_ID: &str = "feat-math.causal-difference-program.v2";

#[derive(Clone, Debug)]
pub struct OrderedRow {
    pub group: GroupKey,
    pub timestamp: f64,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct DifferenceProgram {
    pub(super) config: DifferenceConfig,
    pub(super) width: usize,
    pub(super) input: OrderedInputIdentity,
    pub(super) downstream: GenomeIr,
}

impl DifferenceProgram {
    pub fn new(
        config: DifferenceConfig,
        width: usize,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
    ) -> Result<Self> {
        validate_width(width)?;
        input.validate()?;
        crate::kernel::scope::validate_scope(&downstream, width)
            .map_err(|e| Error::Shape(e.to_string()))?;
        Program::LegacyV1(downstream.clone()).to_bytes()?;
        Ok(Self {
            config,
            width,
            input,
            downstream,
        })
    }

    pub fn config(&self) -> DifferenceConfig {
        self.config
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn input_identity(&self) -> &OrderedInputIdentity {
        &self.input
    }
    pub fn downstream(&self) -> &GenomeIr {
        &self.downstream
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(PROGRAM_VARIANT)?;
        writer.byte(READER_VERSION)?;
        writer.word(REQUIRED_FEATURE)?;
        writer.word(59)?;
        writer.byte(self.config.order)?;
        writer.word(self.config.step.to_bits())?;
        writer.word(self.width as u64)?;
        writer.text(
            &serde_json::to_string(&self.input)
                .map_err(|e| VersionError::Malformed(e.to_string()))?,
        )?;
        let downstream = Program::LegacyV1(self.downstream.clone()).to_bytes()?;
        writer.word(downstream.len() as u64)?;
        writer.raw(&downstream)?;
        Ok(writer.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(8)? != MAGIC_V2
            || reader.byte()? != PROGRAM_VARIANT
            || reader.byte()? != READER_VERSION
            || reader.word()? != REQUIRED_FEATURE
            || reader.word()? != 59
        {
            return Err(VersionError::UnsupportedVersion(
                "difference program envelope/features/wire".into(),
            ));
        }
        let order = reader.byte()?;
        let config = DifferenceConfig::new(2, 59, order, f64::from_bits(reader.word()?))?;
        let width = usize::try_from(reader.word()?)
            .map_err(|_| VersionError::Malformed("difference width".into()))?;
        let input = serde_json::from_str(&reader.text()?)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let n = reader.length()?;
        let Program::LegacyV1(downstream) = Program::from_bytes(reader.take(n)?)?;
        reader.finish()?;
        let program = Self::new(config, width, input, downstream)?;
        if program.to_bytes()? != bytes {
            return Err(VersionError::Malformed(
                "noncanonical difference program".into(),
            ));
        }
        Ok(program)
    }

    pub fn identity(&self) -> VersionResult<String> {
        Ok(identity(PROFILE_ID, &self.to_bytes()?))
    }
}
