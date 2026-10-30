//! Lossless wire-60 program envelope and ordered input identity.
use super::{Config, Mode};
use crate::kernel::codec::{identity, Reader, Writer};
use crate::kernel::ordered_axis::OrderedInputIdentity;
use crate::kernel::wire_v2::{VersionError, VersionResult, MAGIC_V2, READER_VERSION};
use crate::{kernel::Program, GenomeIr};

pub const PROGRAM_VARIANT: u8 = 5;
pub const REQUIRED_FEATURE: u64 = 256;
pub const PROFILE_ID: &str = "feat-math.causal-integral-program.v2";

#[derive(Clone, Debug)]
pub struct IntegrationProgram {
    pub(super) config: Config,
    pub(super) input: OrderedInputIdentity,
    pub(super) downstream: GenomeIr,
}
impl IntegrationProgram {
    pub fn new(
        config: Config,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
    ) -> crate::kernel::Result<Self> {
        input.validate()?;
        crate::kernel::scope::validate_scope(&downstream, config.initial().len())
            .map_err(|e| crate::kernel::Error::Shape(e.to_string()))?;
        Program::LegacyV1(downstream.clone()).to_bytes()?;
        Ok(Self {
            config,
            input,
            downstream,
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn width(&self) -> usize {
        self.config.initial().len()
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
        writer.word(60)?;
        writer.byte(match self.config.mode() {
            Mode::Rect => 0,
            Mode::Trap => 1,
        })?;
        writer.word(self.width() as u64)?;
        for value in self.config.initial() {
            writer.word(value.to_bits())?;
        }
        writer.text(
            &serde_json::to_string(&self.input)
                .map_err(|e| VersionError::Malformed(e.to_string()))?,
        )?;
        let child = Program::LegacyV1(self.downstream.clone()).to_bytes()?;
        writer.word(child.len() as u64)?;
        writer.raw(&child)?;
        Ok(writer.finish())
    }
    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(8)? != MAGIC_V2
            || reader.byte()? != PROGRAM_VARIANT
            || reader.byte()? != READER_VERSION
            || reader.word()? != REQUIRED_FEATURE
            || reader.word()? != 60
        {
            return Err(VersionError::UnsupportedVersion(
                "integral program envelope/features/wire".into(),
            ));
        }
        let mode = match reader.byte()? {
            0 => Mode::Rect,
            1 => Mode::Trap,
            _ => return Err(VersionError::UnsupportedVersion("integral mode".into())),
        };
        let width = reader.length()?;
        if width == 0 || width > crate::MAX_WIDTH {
            return Err(VersionError::Malformed("integral width".into()));
        }
        let initial = (0..width)
            .map(|_| reader.word().map(f64::from_bits))
            .collect::<crate::kernel::Result<Vec<_>>>()?;
        let config = Config::new(2, 60, mode, initial)?;
        let input = serde_json::from_str(&reader.text()?)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let n = reader.length()?;
        let Program::LegacyV1(downstream) = Program::from_bytes(reader.take(n)?)?;
        reader.finish()?;
        let program = Self::new(config, input, downstream)?;
        if program.to_bytes()? != bytes {
            return Err(VersionError::Malformed(
                "noncanonical integral program".into(),
            ));
        }
        Ok(program)
    }
    pub fn identity(&self) -> VersionResult<String> {
        Ok(identity(PROFILE_ID, &self.to_bytes()?))
    }
}
