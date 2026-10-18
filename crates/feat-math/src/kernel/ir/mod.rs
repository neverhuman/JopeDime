//! Bounded, opt-in typed graph/edit ABI. No shipping sampler or old codec is changed.
pub mod canon;
pub mod edit;
pub mod effects;
pub mod graph;
pub mod legacy_bridge;

pub use canon::{ExactF64, ProgramIdV2, Value};
pub use edit::{
    apply, crossover_edit, legal_edits, AppliedEdit, Edit, EditKind, EditOutcome, EditRecord,
};
pub use effects::{
    Dataset, Effects, ExecutionStats, FitPlan, FittedGraph, InputIdentity, RowClosure,
};
pub use graph::{
    ColumnRef, Constraints, Expression, InputSchema, Node, OperatorRef, ProgramGraph, RowSemantics,
};

pub type Result<T> = std::result::Result<T, String>;
