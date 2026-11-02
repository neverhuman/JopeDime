use super::*;
use crate::{
    encoder_receipt::{Encoder, EncoderCapability, Inputs, Member, Product, Scope},
    hwm::{self, Candidate},
    record::{ComparisonContext, Cost, Evaluator, Genome, Provenance, Score, HWM_SCHEMA},
};
use jope_pipeline::evaluation_result::{Consumption, RESULT_SCHEMA};
use jope_pipeline::execution::{Artifact, ObservedConsumption, WorkerCommand, EXECUTION_SCHEMA};
use jope_pipeline::experiment::REQUEST_SCHEMA;
use serde_json::json;
use std::path::PathBuf;

mod admission;
mod fitted;
mod fixture_builder;
mod fixtures;
mod handoff;
mod handoff_fixture;
mod inner_admission;
mod linkage;
mod matrix;
mod metrics;
mod producer;
mod provenance;
mod revocation;
mod transitions;

pub(super) use fixture_builder::*;
pub(super) use fixtures::*;
pub(super) use handoff_fixture::*;
