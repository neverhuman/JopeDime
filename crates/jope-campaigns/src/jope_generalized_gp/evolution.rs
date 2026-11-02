use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use feat_math::dsl::{ParamValue, Stage};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};
use feat_math::genome::GenomeIr;

use super::contracts::*;

mod archive;
mod offspring;
mod ranking;

pub use archive::*;
pub use offspring::*;
pub use ranking::*;
