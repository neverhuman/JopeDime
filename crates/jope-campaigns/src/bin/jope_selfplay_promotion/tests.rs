use super::*;
use tempfile::TempDir;

#[path = "tests/basics.rs"]
mod basics;
#[path = "tests/gate.rs"]
mod gate;
use gate::{gate_reasons, synthetic_evidence};
#[path = "tests/lifecycle.rs"]
mod lifecycle;
