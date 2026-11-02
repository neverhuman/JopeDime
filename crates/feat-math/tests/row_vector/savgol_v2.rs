use feat_math::kernel::ordered_axis::Axis;
use feat_math::kernel::row_vector::{
    emit::{self, RowRecipeLanguage},
    Edit, ExperimentalProfile, FittedStage, Model, Program, RowStage,
};
use feat_math::kernel::savgol::{Config, BOUNDARY_DEGREE};
use feat_math::kernel::wire_v2::{VersionedModel, VersionedProgram};
use feat_math::math_expr::FaultSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

include!("savgol_v2/fixtures.rs");
include!("savgol_v2/contracts.rs");
include!("savgol_v2/training.rs");
include!("savgol_v2/emission.rs");
include!("savgol_v2/recipes.rs");
