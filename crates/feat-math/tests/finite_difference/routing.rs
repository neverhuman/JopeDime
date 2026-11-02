use feat_math::kernel::finite_difference::{
    DifferenceConfig, DifferenceModel, DifferenceProgram, DifferenceState, OrderedRow,
};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::wire_v2::{VersionError, VersionedModel, VersionedProgram};
use feat_math::{kernel::Program, GenomeIr};

fn input() -> OrderedInputIdentity {
    OrderedInputIdentity::new("group", "time")
}
fn config() -> DifferenceConfig {
    DifferenceConfig::new(2, 59, 2, 0.5).unwrap()
}
fn row(group: &str, time: f64, value: f64) -> OrderedRow {
    OrderedRow {
        group: GroupKey::categorical(group).unwrap(),
        timestamp: time,
        values: vec![value],
    }
}
fn program() -> DifferenceProgram {
    DifferenceProgram::new(
        config(),
        1,
        input(),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap()
}

#[path = "routing/boundary.rs"]
mod boundary;
#[path = "routing/state.rs"]
mod state;
