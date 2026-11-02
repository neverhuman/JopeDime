//! Typed grammar and strict compiler for schema-bound advanced-v3 genomes.
//!
//! The legacy-v2 parser deliberately remains permissive.  This module is the
//! advanced-v3 admission boundary: every accepted parameter is typed,
//! materialized, checked against the selected compiler mode, and included in
//! the semantic hash.

#![allow(dead_code)]

use super::feature_block::{
    FeatureBlock, FeatureOrigin, FeatureRole, FeatureWidths, MAX_GENERATED_DENSE_WIDTH,
};
use super::interfaces::CompiledV3Genome;
use super::schema::{ColumnKind, DataSchema, RowSemantics};
use super::types::{TaskKind, TaskSpec};
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::kernel::capabilities::WidthEffect;
pub(crate) use crate::kernel::pipeline::StageClass;
use crate::GenomeIr;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const V3_GRAMMAR_VERSION: &str = "advanced-v3-grammar-v1";

const BOTH_TASKS: TaskMask = TaskMask::new(true, true);
const REGRESSION_ONLY: TaskMask = TaskMask::new(true, false);
const BINARY_ONLY: TaskMask = TaskMask::new(false, true);
const NO_ALIASES: &[&str] = &[];

const TARGET_KINDS: &[&str] = &["rank_gauss", "log1p", "winsor", "zscore"];
const NUMERIC_MODES: &[&str] = &["median", "mean", "zero"];
const MISSING_MODES: &[&str] = &["indicator", "none"];
const CATEGORICAL_MODES: &[&str] = &["native", "frequency", "onehot", "ordered", "hybrid"];
const BOOLEAN_VALUES: &[&str] = &[];
const CLASS_WEIGHTS: &[&str] = &["none", "balanced"];
const CALIBRATIONS: &[&str] = &["none", "platt", "isotonic"];
const THRESHOLDS_PROTOTYPE: &[&str] = &["0.5", "fixed", "f1", "youden"];
const THRESHOLDS_PRODUCTION: &[&str] = &["0.5", "f1", "youden"];
const TAB_REGRESSION_LOSSES: &[&str] = &["squared", "huber"];
const TAB_BINARY_LOSSES: &[&str] = &["log_loss"];
const IMPUTE_STRATEGIES: &[&str] = &["median", "mean", "const"];
const SELECT_BY: &[&str] = &["corr", "var"];
const POWER_KINDS: &[&str] = &["yeo_johnson"];

const PIECEWISE_BINS: &[i64] = &[8, 16, 32];
const CORR_WIDTHS: &[i64] = &[32, 64, 128, 256, 512];
const PLS_COMPONENTS: &[i64] = &[2, 4, 8, 16, 32];
const LOGIT_L2: &[f64] = &[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
const LOGIT_ITERATIONS: &[i64] = &[25, 50, 100];
const BOOST_ROUNDS: &[i64] = &[32, 64, 128, 256];
const BOOST_DEPTHS: &[i64] = &[2, 3, 4, 6];
const BOOST_BINS: &[i64] = &[16, 32, 64, 128];
const BOOST_LEARNING_RATES: &[f64] = &[0.02, 0.05, 0.1];
const BOOST_MIN_LEAF: &[i64] = &[2, 5, 10, 20];
const BOOST_LAMBDAS: &[f64] = &[0.0, 1.0, 10.0];
const SAMPLE_FRACTIONS: &[f64] = &[0.7, 0.85, 1.0];
const EXTRA_TREES: &[i64] = &[32, 64, 128, 256, 512];
const EXTRA_DEPTHS: &[i64] = &[4, 8, 12, 16, 20];
const EXTRA_MIN_LEAF: &[i64] = &[1, 2, 5, 10];
const GAM_BINS: &[i64] = &[8, 16, 32, 64];
const GAM_INTERACTIONS: &[i64] = &[0, 4, 8, 16, 32];
const GAM_BAGS: &[i64] = &[1, 4, 8];
const RIDGE_PRODUCTION_LAMBDA: &[f64] = &[0.02];

macro_rules! param {
    ($name:literal, $kind:ident, $defaults:expr, $prototype:expr, $production:expr) => {
        ParamDescriptor {
            name: $name,
            aliases: NO_ALIASES,
            kind: ParamKind::$kind,
            defaults: $defaults,
            prototype_domain: $prototype,
            production_domain: $production,
            active_when: ActiveWhen::Always,
        }
    };
}

macro_rules! binary_param {
    ($name:literal, $kind:ident, $defaults:expr, $prototype:expr, $production:expr) => {
        ParamDescriptor {
            name: $name,
            aliases: NO_ALIASES,
            kind: ParamKind::$kind,
            defaults: $defaults,
            prototype_domain: $prototype,
            production_domain: $production,
            active_when: ActiveWhen::Task(TaskKind::Binary),
        }
    };
}

macro_rules! operation {
    (
        $name:literal, aliases=$aliases:expr, class=$class:ident, tasks=$tasks:expr,
        roles=$prototype_roles:ident/$production_roles:ident, rows=$rows:ident,
        params=$params:expr, append=$append:expr, width=$width:expr, masks=$masks:literal,
        target=$target:expr, support=$support:expr, family=$family:literal,
        island=$island:literal, surrogate=$surrogate:literal, admission=$admission:ident
    ) => {
        OperationDescriptor {
            name: $name,
            aliases: $aliases,
            stage_class: StageClass::$class,
            tasks: $tasks,
            prototype_roles: RolePolicy::$prototype_roles,
            production_roles: RolePolicy::$production_roles,
            row_requirement: RowRequirement::$rows,
            parameters: $params,
            accepts_append: $append,
            width_effect: $width,
            missing_mask_effect: $masks,
            target_aware: $target,
            backend_support: $support,
            family: $family,
            island: $island,
            surrogate: $surrogate,
            admission: ProductionAdmission::$admission,
        }
    };
}

mod compile_layout;
mod compile_shape;
mod compile_stages;
mod compile_types;
mod compile_values;
mod params_base;
mod params_models;
mod registry;
mod registry_entry;
#[cfg(test)]
mod tests;
mod types;

pub(super) use compile_layout::*;
pub(super) use compile_shape::*;
pub(crate) use compile_stages::compile_v3_genome;
pub(super) use compile_stages::*;
pub(crate) use compile_types::{CompileFeature, CompileInputLayout, CompileV3Options};
pub(super) use compile_values::*;
pub(super) use params_base::*;
pub(super) use params_models::*;
pub(super) use registry::*;
pub(crate) use registry_entry::v3_operation_registry;
pub(crate) use types::{
    ActiveWhen, BackendSupport, CompileMode, EmitterTarget, ModeDefaults, OperationDescriptor,
    ParamDescriptor, ParamDomain, ParamKind, ProductionAdmission, RolePolicy, RowRequirement,
    StaticValue, TargetSupport, TaskMask, V3ParamDomain,
};
