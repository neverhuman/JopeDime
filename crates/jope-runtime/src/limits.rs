//! Input and resource defaults (`deploy/appliance/limits.toml` mirrors these; a test keeps the
//! two equal). Every limit can be raised explicitly on the command line; none is raised silently.
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Limits {
    /// Bytes of one input CSV file.
    pub max_input_bytes: u64,
    /// Data rows of one input table.
    pub max_rows: usize,
    /// Columns of one input table, target included.
    pub max_columns: usize,
    /// Classes of a classification target.
    pub max_classes: usize,
    /// Bytes of one fitted model file.
    pub max_model_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 30,
            max_rows: 2_000_000,
            max_columns: 10_000,
            max_classes: 256,
            max_model_bytes: 2 << 30,
        }
    }
}
