//! B13 goertzel_bank (wire 61) as a v2 FeatureWithinRow row stage beside B12's savgol.
use feat_math::kernel::goertzel::{self, Config};
use feat_math::kernel::ordered_axis::Axis;
use feat_math::kernel::row_vector::{
    emit::{self, RowRecipeLanguage},
    Edit, ExperimentalProfile, FittedStage, GoertzelProfile, Model, Program, RowStage,
    GOERTZEL_FEATURE, REQUIRED_FEATURES,
};
use feat_math::kernel::savgol;
use feat_math::kernel::wire_v2::{VersionError, VersionedModel, VersionedProgram};
use feat_math::math_expr::FaultSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn bank(frequencies: &[f64]) -> Config {
    Config::new(2, 61, Axis::FeatureWithinRow, frequencies).unwrap()
}
fn smoothing() -> savgol::Config {
    savgol::Config::new(2, 58, Axis::FeatureWithinRow, 5, 2, 0, "shrink").unwrap()
}
fn program(width: usize, stages: Vec<RowStage>) -> Program {
    Program::new(2, width, stages, 0.02).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn hex_words(values: &serde_json::Value) -> Vec<u64> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|b| u64::from_str_radix(b.as_str().unwrap(), 16).unwrap())
        .collect()
}
fn training(width: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = (0..24)
        .map(|i| {
            (0..width)
                .map(|j| ((i * 13 + j * 7) % 37) as f64 / 9.0 - 2.0)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| r[0] * 0.5 + r[width - 1] * 0.2 - 0.3)
        .collect();
    (x, y)
}

#[path = "goertzel_v2_artifacts.rs"]
mod artifacts;
#[path = "goertzel_v2_contracts.rs"]
mod contracts;
#[path = "goertzel_v2_interop.rs"]
mod interop;
#[path = "goertzel_v2_mutation.rs"]
mod mutation;
