//! Bounded, removable wire-75 search admission for the ordered Experimental profile.
use super::{Config, TimeGradientProgram};
use crate::kernel::ordered_axis::{
    Axis, Causality, FitScope, OrderedInputIdentity, StageAxis, TaskMode,
};
use crate::kernel::wire_v2::{VersionError, VersionResult, VersionedProgram};
use crate::kernel::{Error, Program, Result};
use crate::{GenomeIr, Pcg32};

pub const PROFILE: &str = "feat-math.causal-time_gradient.experimental.v2";
const CONTRACT: &[u8] = b"wire=75;axis=SampleTime;causal;one-stage;min-gap=0.125,0.25,0.5,1,2;strict-gap;prior<=0.25;numeric-ridge";
const MAGIC: &[u8; 8] = b"FMKTPF\0\x02";

#[derive(Clone, Copy, Debug)]
pub struct ExperimentalProfile {
    pub time_gradient: bool,
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
                "time_gradient Experimental prior must be in [0,0.25]".into(),
            ));
        }
        Ok(())
    }

    pub fn to_bytes(self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = MAGIC.to_vec();
        bytes.push(u8::from(self.time_gradient));
        bytes.extend_from_slice(&self.prior.to_bits().to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        if bytes.len() != 17 {
            return Err(VersionError::Malformed(
                "time_gradient profile length".into(),
            ));
        }
        if &bytes[..8] != MAGIC || bytes[8] > 1 {
            return Err(VersionError::UnsupportedVersion(
                "time_gradient profile envelope/flags".into(),
            ));
        }
        let profile = Self {
            time_gradient: bytes[8] == 1,
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
pub struct TimeGradientSearch {
    width: usize,
    input: OrderedInputIdentity,
    downstream: GenomeIr,
}

impl TimeGradientSearch {
    pub fn new(
        width: usize,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
        task: TaskMode,
    ) -> Result<Self> {
        if width == 0 || width > crate::MAX_WIDTH {
            return Err(Error::Shape(
                "time_gradient width must be1..MAX_WIDTH".into(),
            ));
        }
        StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless)
            .admit(task, Some(&input))?;
        let candidate = TimeGradientProgram::new(
            Config::new(2, 75, width, 0.125)?,
            input.clone(),
            downstream.clone(),
        )?;
        candidate
            .validate_recipe_closure()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(Self {
            width,
            input,
            downstream,
        })
    }

    pub fn base(&self) -> VersionedProgram {
        VersionedProgram::LegacyV1(Program::LegacyV1(self.downstream.clone()))
    }

    /// A prior miss returns the plain base; disabling growth leaves decoding open.
    pub fn generate(&self, seed: u64, profile: ExperimentalProfile) -> Result<VersionedProgram> {
        profile.validate()?;
        let mut rng = Pcg32::new(seed, 0x646966667072);
        if !profile.time_gradient || rng.next_f64() >= profile.prior {
            return Ok(self.base());
        }
        self.edit(&self.base(), rng.next_u32() as u64, Edit::Grow, profile)
    }

    /// Explicit arms create from the plain base, alter, or remove the wire-75 node.
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
            (VersionedProgram::TimeGradientV2(_), Edit::Drop)
        ) {
            let VersionedProgram::TimeGradientV2(program) = candidate else {
                unreachable!()
            };
            return Ok(VersionedProgram::LegacyV1(Program::LegacyV1(
                program.downstream().clone(),
            )));
        }
        if !profile.time_gradient {
            return Err(Error::Unsupported(
                "time_gradient generation disabled".into(),
            ));
        }
        let mut rng = Pcg32::new(seed, 0x646966666564);
        let steps: [f64; 5] = [0.125, 0.25, 0.5, 1.0, 2.0];
        let (width, gap, input, downstream) = match (candidate, arm) {
            (VersionedProgram::LegacyV1(Program::LegacyV1(genome)), Edit::Grow) => (
                self.width,
                steps[rng.next_u32() as usize % steps.len()],
                self.input.clone(),
                genome.clone(),
            ),
            (VersionedProgram::TimeGradientV2(program), Edit::Change) => {
                let index = steps
                    .iter()
                    .position(|v| v.to_bits() == program.config().minimum_gap().to_bits());
                let gap = index.map_or(steps[0], |i| steps[(i + 1) % steps.len()]);
                (
                    program.width(),
                    gap,
                    program.input_identity().clone(),
                    program.downstream().clone(),
                )
            }
            _ => {
                return Err(Error::Unsupported(
                    "time-gradient GP arm unavailable".into(),
                ))
            }
        };
        let program = TimeGradientProgram::new(Config::new(2, 75, width, gap)?, input, downstream)?;
        program
            .validate_recipe_closure()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(VersionedProgram::TimeGradientV2(program))
    }
}
