//! Reuse the merged ordered metadata codec, then emit the path-stats state twin.
use super::PathStatsLanguage;
use crate::math_expr::FaultSet;

pub(super) fn emit_state(language: PathStatsLanguage) -> String {
    let (common, marker, state, imports) = match language {
        PathStatsLanguage::Rust => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_rust.txt")),
            "#[derive(Clone)]\npub struct DifferenceState",
            include_str!("recipes/path_state_rust.txt"),
            "",
        ),
        PathStatsLanguage::Python => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_python.txt")),
            "class DifferenceState:",
            include_str!("recipes/path_state_python.txt"),
            "import math,struct,sys\n",
        ),
        PathStatsLanguage::Cpp17 => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_cpp.txt")),
            "struct DifferenceState {",
            include_str!("recipes/path_state_cpp.txt"),
            "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <stdexcept>\n#include <utility>\n#include <vector>\n",
        ),
    };
    let common = common
        .split_once(marker)
        .expect("ordered metadata codec prefix")
        .0;
    format!("{imports}\n{common}\n{state}")
        .replace(
            "@INPUTFAULT@",
            &FaultSet::NONFINITE_INPUT.bits().to_string(),
        )
        .replace(
            "@CLAMPFAULT@",
            &FaultSet::MAGNITUDE_CLAMPED.bits().to_string(),
        )
        .replace("@BOUNDARYFAULT@", &super::BOUNDARY_HISTORY.to_string())
}
