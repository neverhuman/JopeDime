//! Canonical wire-64 program envelope and reader-first ingress.
use super::Config;
use crate::kernel::codec::{identity, Reader, Writer};
use crate::kernel::ordered_axis::OrderedInputIdentity;
use crate::kernel::wire_v2::{VersionError, VersionResult, MAGIC_V2, READER_VERSION};
use crate::{kernel::Program, GenomeIr};

pub const PROGRAM_VARIANT: u8 = 7;
pub const REQUIRED_FEATURE: u64 = 1024;
pub const PROFILE_ID: &str = "feat-math.causal-path-stats-program.v2";

#[derive(Clone, Debug)]
pub struct PathStatsProgram {
    pub(super) config: Config,
    pub(super) input: OrderedInputIdentity,
    pub(super) downstream: GenomeIr,
}
impl PathStatsProgram {
    pub fn new(
        config: Config,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
    ) -> crate::kernel::Result<Self> {
        input.validate()?;
        crate::kernel::scope::validate_scope(&downstream, 5)
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
    /// Fixed output width; the stage consumes one scalar path observation.
    pub fn width(&self) -> usize {
        5
    }
    pub fn input_width(&self) -> usize {
        1
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
        writer.word(64)?;
        writer.word(self.config.window() as u64)?;
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
            || reader.word()? != 64
        {
            return Err(VersionError::UnsupportedVersion(
                "path-stats program envelope/features/wire".into(),
            ));
        }
        let window = usize::try_from(reader.word()?)
            .map_err(|_| VersionError::Malformed("path-stats window".into()))?;
        let config = Config::new(2, 64, window)?;
        let input = serde_json::from_str(&reader.text()?)
            .map_err(|e| VersionError::Malformed(e.to_string()))?;
        let n = reader.length()?;
        let Program::LegacyV1(downstream) = Program::from_bytes(reader.take(n)?)?;
        reader.finish()?;
        let program = Self::new(config, input, downstream)?;
        if program.to_bytes()? != bytes {
            return Err(VersionError::Malformed(
                "noncanonical path-stats program".into(),
            ));
        }
        Ok(program)
    }
    pub fn identity(&self) -> VersionResult<String> {
        Ok(identity(PROFILE_ID, &self.to_bytes()?))
    }
}
