//! Shared internal contracts for the advanced-v3 compiler, fitter, CV service,
//! estimator heads, and portable artifact pipeline.
//!
//! These types are intentionally opaque outside the `advanced_v3` module.
//! Their responsibilities are stable; their private representations can evolve
//! as the corresponding gated implementation tasks land.

#![allow(dead_code)]

use super::csv::RawTable;
use super::feature_block::FeatureWidths;
use super::heads::V3Head;
use super::pipeline::BoundModel;
use super::schema::DataSchema;
use super::transforms::V3Op;
use super::types::TaskSpec;
use crate::dsl::Stage;

#[derive(Clone, Debug)]
pub(crate) struct CompiledV3Genome {
    canonical: String,
    semantic_hash: String,
    stages: Vec<Stage>,
    feature_widths: FeatureWidths,
    requires_ordered_rows: bool,
    target_aware: bool,
    families: Vec<String>,
}

impl CompiledV3Genome {
    pub(super) fn new(
        canonical: String,
        semantic_hash: String,
        stages: Vec<Stage>,
        feature_widths: FeatureWidths,
        requires_ordered_rows: bool,
        target_aware: bool,
        families: Vec<String>,
    ) -> Self {
        Self {
            canonical,
            semantic_hash,
            stages,
            feature_widths,
            requires_ordered_rows,
            target_aware,
            families,
        }
    }

    pub(crate) fn canonical(&self) -> &str {
        &self.canonical
    }

    pub(crate) fn semantic_hash(&self) -> &str {
        &self.semantic_hash
    }

    pub(super) fn stages(&self) -> &[Stage] {
        &self.stages
    }

    pub(crate) fn feature_widths(&self) -> FeatureWidths {
        self.feature_widths
    }

    pub(crate) fn requires_ordered_rows(&self) -> bool {
        self.requires_ordered_rows
    }

    pub(crate) fn target_aware(&self) -> bool {
        self.target_aware
    }

    pub(crate) fn families(&self) -> &[String] {
        &self.families
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FitContext<'a> {
    table: &'a RawTable,
    schema: &'a DataSchema,
    task: &'a TaskSpec,
    targets: &'a [f64],
    fold: Option<usize>,
    seed: u64,
}

impl<'a> FitContext<'a> {
    pub(super) fn new(
        table: &'a RawTable,
        schema: &'a DataSchema,
        task: &'a TaskSpec,
        targets: &'a [f64],
        fold: Option<usize>,
        seed: u64,
    ) -> Self {
        Self {
            table,
            schema,
            task,
            targets,
            fold,
            seed,
        }
    }

    pub(crate) fn table(&self) -> &'a RawTable {
        self.table
    }

    pub(crate) fn schema(&self) -> &'a DataSchema {
        self.schema
    }

    pub(crate) fn task(&self) -> &'a TaskSpec {
        self.task
    }

    pub(crate) fn targets(&self) -> &'a [f64] {
        self.targets
    }

    pub(crate) fn fold(&self) -> Option<usize> {
        self.fold
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadBudget {
    max_iterations: usize,
    max_rounds: usize,
    max_trees: usize,
    max_depth: usize,
}

impl HeadBudget {
    pub(super) fn new(
        max_iterations: usize,
        max_rounds: usize,
        max_trees: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            max_iterations,
            max_rounds,
            max_trees,
            max_depth,
        }
    }

    pub(crate) fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub(crate) fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    pub(crate) fn max_trees(&self) -> usize {
        self.max_trees
    }

    pub(crate) fn max_depth(&self) -> usize {
        self.max_depth
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FittedV3Transform {
    state: V3Op,
}

impl FittedV3Transform {
    pub(super) fn new(state: V3Op) -> Self {
        Self { state }
    }

    pub(super) fn state(&self) -> &V3Op {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FittedV3Head {
    state: V3Head,
}

impl FittedV3Head {
    pub(super) fn new(state: V3Head) -> Self {
        Self { state }
    }

    pub(super) fn state(&self) -> &V3Head {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FittedV3Pipeline {
    state: BoundModel,
}

impl FittedV3Pipeline {
    pub(super) fn new(state: BoundModel) -> Self {
        Self { state }
    }

    pub(super) fn state(&self) -> &BoundModel {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PortableModel {
    payload: Vec<u8>,
    sha256: String,
}

impl PortableModel {
    pub(super) fn new(payload: Vec<u8>, sha256: String) -> Self {
        Self { payload, sha256 }
    }

    pub(super) fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug)]
pub(crate) struct V3CvEvidence {
    fold_assignments: Vec<usize>,
    oof_predictions: Vec<f64>,
    semantic_hash: String,
}

impl V3CvEvidence {
    pub(super) fn new(
        fold_assignments: Vec<usize>,
        oof_predictions: Vec<f64>,
        semantic_hash: String,
    ) -> Self {
        Self {
            fold_assignments,
            oof_predictions,
            semantic_hash,
        }
    }

    pub(crate) fn fold_assignments(&self) -> &[usize] {
        &self.fold_assignments
    }

    pub(crate) fn oof_predictions(&self) -> &[f64] {
        &self.oof_predictions
    }

    pub(crate) fn semantic_hash(&self) -> &str {
        &self.semantic_hash
    }
}
