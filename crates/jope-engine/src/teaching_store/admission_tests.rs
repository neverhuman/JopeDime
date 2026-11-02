use crate as jope_parallel;
use crate::manifest::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

mod fixtures;
mod native_strict;
pub(super) use fixtures::{attach_completion, Fixture};
