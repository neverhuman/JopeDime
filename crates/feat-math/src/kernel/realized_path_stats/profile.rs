//! Bounded, removable wire-64 search admission for the ordered Experimental profile.
use super::{Config, PathStatsProgram};
use crate::kernel::ordered_axis::{
    Axis, Causality, FitScope, OrderedInputIdentity, StageAxis, TaskMode,
};
use crate::kernel::wire_v2::{VersionError, VersionResult, VersionedProgram};
use crate::kernel::{Error, Program, Result};
use crate::{GenomeIr, Pcg32};

pub const PROFILE: &str = "feat-math.causal-path_stats.experimental.v2";
const CONTRACT: &[u8] = b"wire=64;axis=SampleTime;causal;one-stage;input=1;output=5;window=2,4,8,16,32,64,129;increment-clamp=1e6;prior<=0.25;numeric-ridge";
const MAGIC: &[u8; 8] = b"FMKPPF\0\x02";

#[derive(Clone, Copy, Debug)]
pub struct ExperimentalProfile {
    pub path_stats: bool,
    pub prior: f64,
}

impl ExperimentalProfile {
    /// Bind the explicit switch/prior and the bounded search contract.
    pub fn identity(self) -> VersionResult<String> {
        let mut bytes = CONTRACT.to_vec();
        bytes.extend(self.to_bytes()?);
        Ok(crate::kernel::codec::identity(PROFILE, &bytes))
    }

    pub fn validate(self) -> Result<()> {
        if !self.prior.is_finite() || self.prior < 0.0 || self.prior > 0.25 {
            return Err(Error::Domain(
                "path_stats Experimental prior must be in [0,0.25]".into(),
            ));
        }
        Ok(())
    }

    pub fn to_bytes(self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = MAGIC.to_vec();
        bytes.push(u8::from(self.path_stats));
        bytes.extend_from_slice(&self.prior.to_bits().to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        if bytes.len() != 17 {
            return Err(VersionError::Malformed("path_stats profile length".into()));
        }
        if &bytes[..8] != MAGIC || bytes[8] > 1 {
            return Err(VersionError::UnsupportedVersion(
                "path_stats profile envelope/flags".into(),
            ));
        }
        let profile = Self {
            path_stats: bytes[8] == 1,
            prior: f64::from_bits(u64::from_le_bytes(bytes[9..].try_into().unwrap())),
        };
        profile.validate()?;
        Ok(profile)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Edit {
    Grow,
    Change,
    Drop,
}

/// One optional ordered pre-stage, with explicit metadata outside the AST.
pub struct PathStatsSearch {
    input: OrderedInputIdentity,
    downstream: GenomeIr,
}

impl PathStatsSearch {
    pub fn new(
        width: usize,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
        task: TaskMode,
    ) -> Result<Self> {
        if width != 1 {
            return Err(Error::Shape("path stats input width must be1".into()));
        }
        StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless)
            .admit(task, Some(&input))?;
        let candidate =
            PathStatsProgram::new(Config::new(2, 64, 4)?, input.clone(), downstream.clone())?;
        candidate
            .validate_recipe_closure()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(Self { input, downstream })
    }

    pub fn base(&self) -> VersionedProgram {
        VersionedProgram::LegacyV1(Program::LegacyV1(self.downstream.clone()))
    }

    /// A prior miss returns the plain base; disabling growth leaves decoding open.
    pub fn generate(&self, seed: u64, profile: ExperimentalProfile) -> Result<VersionedProgram> {
        profile.validate()?;
        let mut rng = Pcg32::new(seed, 0x646966667072);
        if !profile.path_stats || rng.next_f64() >= profile.prior {
            return Ok(self.base());
        }
        self.edit(&self.base(), rng.next_u32() as u64, Edit::Grow, profile)
    }

    /// Explicit arms create from the plain base, alter, or remove the wire-64 node.
    pub fn edit(
        &self,
        candidate: &VersionedProgram,
        seed: u64,
        arm: Edit,
        profile: ExperimentalProfile,
    ) -> Result<VersionedProgram> {
        profile.validate()?;
        if matches!(
            (candidate, arm),
            (VersionedProgram::PathStatsV2(_), Edit::Drop)
        ) {
            let VersionedProgram::PathStatsV2(program) = candidate else {
                unreachable!()
            };
            return Ok(VersionedProgram::LegacyV1(Program::LegacyV1(
                program.downstream().clone(),
            )));
        }
        if !profile.path_stats {
            return Err(Error::Unsupported("path_stats generation disabled".into()));
        }
        let mut rng = Pcg32::new(seed, 0x646966666564);
        let windows = [2, 4, 8, 16, 32, 64, 129];
        let (window, input, downstream) = match (candidate, arm) {
            (VersionedProgram::LegacyV1(Program::LegacyV1(genome)), Edit::Grow) => (
                windows[rng.next_u32() as usize % windows.len()],
                self.input.clone(),
                genome.clone(),
            ),
            (VersionedProgram::PathStatsV2(program), Edit::Change) => {
                let index = windows.iter().position(|v| *v == program.config().window());
                (
                    index.map_or(windows[0], |i| windows[(i + 1) % windows.len()]),
                    program.input_identity().clone(),
                    program.downstream().clone(),
                )
            }
            _ => return Err(Error::Unsupported("path-stats GP arm unavailable".into())),
        };
        let program = PathStatsProgram::new(Config::new(2, 64, window)?, input, downstream)?;
        program
            .validate_recipe_closure()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(VersionedProgram::PathStatsV2(program))
    }
}
