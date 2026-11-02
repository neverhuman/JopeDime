//! Train-only, fail-closed gate for optional Hyperion feature expansion.
//!
//! The gate deliberately has no sealed/test-table type.  It compares a frozen
//! AFPM predictor on paired, group-disjoint OOF folds and selects `raw` on any
//! incomplete or inconsistent evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::jope_dlgp_best5::FrozenPredictorV1;
use crate::jope_generalized_gp::{
    raw_sha256, CorpusEvalAdapter, DatasetRowV1, DatasetTableV1, FrozenArtifactV1, GpTaskV1,
    RowSplitV1, SplitRowV1,
};
use crate::search_v3::{canonical_json_bytes, canonical_sha256, validate_sha256};

pub const ENCODING_BUDGET: Duration = Duration::from_secs(8 * 60);
pub const SCORING_BUDGET: Duration = Duration::from_secs(2 * 60);
pub const WALL_CLOCK_BUDGET: Duration = Duration::from_secs(10 * 60);
pub const ROW_EMBEDDING_WIDTH: usize = 128;
pub const BOOTSTRAP_REPLICATES: usize = 4_096;
pub const MINIMUM_FOLD_FIT_ROWS: usize = 32;
pub const MINIMUM_FOLD_HOLD_ROWS: usize = 2;
pub const MINIMUM_FOLD_HOLD_GROUPS: usize = 2;
const CV5_SEED: u64 = 0x4a4f_5045_4859_5035;
const CV7_SEED: u64 = 0x4a4f_5045_4859_5037;

mod cache_registry;
mod cache_types;
mod encoder_process;
mod encoding;
mod feature_ops;
mod folds;
mod model;
mod precomputed;
mod scoring;
#[cfg(test)]
mod tests;

pub use cache_registry::*;
pub use cache_types::*;
pub use encoder_process::*;
pub use encoding::*;
pub use feature_ops::*;
pub use folds::*;
pub use model::*;
pub use precomputed::*;
use precomputed::{
    load_precomputed_train_features, read_authorized_table,
    require_frozen_precomputed_dataset_binding,
};
pub use scoring::*;

use cache_registry::{precomputed_fold_encoding, sidecars_for_view, StageExecution};
use cache_types::{FoldEncoding, PrecomputedTrainFeatureBlocks, RegressionTable};
use encoder_process::{
    csv_for_rows, elapsed_milliseconds, hash_f64, hash_matrix, hash_selected_strings, hash_strings,
    output_artifacts, parse_regression_csv, read_output_matrices, read_vs_matrix,
    require_cuda_device, run_encoder_with_timeout, select_rows, verify_runtime_identity,
};
use encoding::{encode_fold, FoldEncodingScope};
use feature_ops::{rmse, seed64};
use folds::{validate_plan_feature_feasibility, validate_stage_folds, write_immutable};
use scoring::{build_measured_decision, cache_key, encode_full_train, execute_stage};
