//! Bounded, removable wire-60 search admission for the ordered Experimental profile.
use super::{Config, IntegrationProgram, Mode};
use crate::kernel::ordered_axis::{
    Axis, Causality, FitScope, OrderedInputIdentity, StageAxis, TaskMode,
};
use crate::kernel::wire_v2::{VersionError, VersionResult, VersionedProgram};
use crate::kernel::{Error, Program, Result};
use crate::{GenomeIr, Pcg32};

pub const PROFILE: &str = "feat-math.causal-integration.experimental.v2";
const CONTRACT: &[u8] = b"wire=60;axis=SampleTime;causal;one-stage;rect,trap;initial=zeros;positive-gap;prior<=0.25;numeric-ridge";
const MAGIC: &[u8; 8] = b"FMKIPF\0\x02";

#[derive(Clone, Copy, Debug)]
pub struct ExperimentalProfile {
    pub integration: bool,
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
                "integration Experimental prior must be in [0,0.25]".into(),
            ));
        }
        Ok(())
    }

    pub fn to_bytes(self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = MAGIC.to_vec();
        bytes.push(u8::from(self.integration));
        bytes.extend_from_slice(&self.prior.to_bits().to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        if bytes.len() != 17 {
            return Err(VersionError::Malformed("integration profile length".into()));
        }
        if &bytes[..8] != MAGIC || bytes[8] > 1 {
            return Err(VersionError::UnsupportedVersion(
                "integration profile envelope/flags".into(),
            ));
        }
        let profile = Self {
            integration: bytes[8] == 1,
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
pub struct IntegrationSearch {
    width: usize,
    input: OrderedInputIdentity,
    downstream: GenomeIr,
}

impl IntegrationSearch {
    pub fn new(
        width: usize,
        input: OrderedInputIdentity,
        downstream: GenomeIr,
        task: TaskMode,
    ) -> Result<Self> {
        if width == 0 || width > crate::MAX_WIDTH {
            return Err(Error::Shape("integration width must be1..MAX_WIDTH".into()));
        }
        StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless)
            .admit(task, Some(&input))?;
        let candidate = IntegrationProgram::new(
            Config::new(2, 60, Mode::Rect, vec![0.0; width])?,
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
        if !profile.integration || rng.next_f64() >= profile.prior {
            return Ok(self.base());
        }
        self.edit(&self.base(), rng.next_u32() as u64, Edit::Grow, profile)
    }

    /// Explicit arms create from the plain base, alter, or remove the wire-60 node.
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
            (VersionedProgram::IntegrationV2(_), Edit::Drop)
        ) {
            let VersionedProgram::IntegrationV2(program) = candidate else {
                unreachable!()
            };
            return Ok(VersionedProgram::LegacyV1(Program::LegacyV1(
                program.downstream().clone(),
            )));
        }
        if !profile.integration {
            return Err(Error::Unsupported("integration generation disabled".into()));
        }
        let mut rng = Pcg32::new(seed, 0x646966666564);
        let (mode, initial, input, downstream) = match (candidate, arm) {
            (VersionedProgram::LegacyV1(Program::LegacyV1(genome)), Edit::Grow) => (
                if rng.next_u32() & 1 == 0 {
                    Mode::Rect
                } else {
                    Mode::Trap
                },
                vec![0.0; self.width],
                self.input.clone(),
                genome.clone(),
            ),
            (VersionedProgram::IntegrationV2(program), Edit::Change) => {
                if program
                    .config()
                    .initial()
                    .iter()
                    .any(|value| value.to_bits() != 0)
                {
                    return Err(Error::Unsupported(
                        "integral Change initial state outside Experimental zero menu".into(),
                    ));
                }
                (
                    if program.config().mode() == Mode::Rect {
                        Mode::Trap
                    } else {
                        Mode::Rect
                    },
                    program.config().initial().to_vec(),
                    program.input_identity().clone(),
                    program.downstream().clone(),
                )
            }
            _ => return Err(Error::Unsupported("integration GP arm unavailable".into())),
        };
        let program =
            IntegrationProgram::new(Config::new(2, 60, mode, initial)?, input, downstream)?;
        program
            .validate_recipe_closure()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        Ok(VersionedProgram::IntegrationV2(program))
    }
}
