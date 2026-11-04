use super::*;
use crate::code_of;
use crate::fixture;
use crate::mine::mine;
use crate::sha256_hex;
use crate::synthetic::PROBE_ROWS;
use crate::table::{Lifecycle, MineParameters, Promotion};

const PINNED_CHAMPION_PY: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/8944cbdf.py.sha256");
const PINNED_CHAMPION_RS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/8944cbdf.rs.sha256");
const PINNED_PORTABLE_PY: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.py.sha256");
const PINNED_PORTABLE_RS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.rs.sha256");
const PINNED_PORTABLE_CPP: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.cpp.sha256");
const PINNED_PORTABLE_JS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.js.sha256");

fn validation_code(table: &MotifTable) -> Option<&'static str> {
    validate(table, &ValidateOptions::default())
        .err()
        .and_then(|error| code_of(&error))
}

fn codes(result: &CheckResult) -> Vec<&str> {
    result
        .failures
        .iter()
        .map(|failure| failure.code.as_str())
        .collect()
}

mod artifacts;
mod baseline;
mod contexts;
mod controls;
mod promotion;
