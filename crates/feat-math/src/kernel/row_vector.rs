//! Explicit v2 FeatureWithinRow pipelines. The legacy generator and grammar
//! are unchanged. This bounded Experimental profile admits at most eight
//! row stages followed by the existing ridge learner, with train-only fit.
use super::codec::{Reader, Writer, MAX_BYTES};
use super::goertzel;
use super::savgol;
use super::wire_v2::{VersionError, VersionResult, MAGIC_V2};
use super::{Error, Result};
use crate::heads::RidgeModel;
use crate::spec::{Plan, PlannedHead, PlannedOp, PlannedStage};
use crate::transpile::{lint_shipped_text, EmitOptions};
use crate::{Pcg32, Rows};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod emit;
pub const PROFILE: &str = "feat-math.feature-row.experimental.v2";
pub const REQUIRED_FEATURES: u64 = 32;
const PROFILE_CONTRACT: &str = "feat-math.feature-row.experimental.v2;axis=FeatureWithinRow;max_stages=8;prior_max=0.25;head=ridge-v1";
const MAX_STAGES: usize = 8;
// B13: a program that holds a goertzel_bank stage also declares this bit and
// the bank's grammar/emitter identities; savgol-only artifacts are unchanged.
pub const GOERTZEL_FEATURE: u64 = 128;
const GOERTZEL_PROFILE_CONTRACT: &str = "feat-math.feature-row.experimental.v2;axis=FeatureWithinRow;max_stages=8;prior_max=0.25;head=ridge-v1;wire61=goertzel_bank;bins=1..32;width<=512";
const EMITTER_IDENTITY: &str = "feat-math.feature-row.recipes.v2+wire58";
const GOERTZEL_EMITTER_IDENTITY: &str = "feat-math.feature-row.recipes.v2+wire58+wire61";

mod model;
mod program;

pub use model::Model;
use model::{
    goertzel_config, goertzel_stages, grid_frequency, savgol_stages, transform, validate_rows,
};
pub use program::{Edit, ExperimentalProfile, FittedStage, GoertzelProfile, Program, RowStage};
