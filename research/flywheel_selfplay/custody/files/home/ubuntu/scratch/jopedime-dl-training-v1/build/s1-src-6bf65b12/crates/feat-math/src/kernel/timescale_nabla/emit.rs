//! Reuse the merged ordered metadata codec, then emit the time-gradient state twin.
use super::TimeGradientLanguage;
use crate::math_expr::FaultSet;

pub(super) fn emit_state(language: TimeGradientLanguage) -> String {
    let (common, marker, state, imports) = match language {
        TimeGradientLanguage::Rust => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_rust.txt")),
            "#[derive(Clone)]\npub struct DifferenceState",
            include_str!("recipes/time_gradient_state_rust.txt"),
            "",
        ),
        TimeGradientLanguage::Python => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_python.txt")),
            "class DifferenceState:",
            include_str!("recipes/time_gradient_state_python.txt"),
            "import math,struct,sys\n",
        ),
        TimeGradientLanguage::Cpp17 => (
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernel/finite_difference/recipes/ordered_state_cpp.txt")),
            "struct DifferenceState {",
            include_str!("recipes/time_gradient_state_cpp.txt"),
            "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <stdexcept>\n#include <utility>\n#include <vector>\n",
        ),
    };
    let common = common
        .split_once(marker)
        .expect("ordered metadata codec prefix")
        .0;
    let common = match language {
        TimeGradientLanguage::Python => common.replace("def __init__(self, kind):", "def __init__(self, kind, faults=0):").replace("self.kind = kind", "self.kind = kind\n        self.faults = faults"),
        TimeGradientLanguage::Cpp17 => common.replace("struct OrderedError : std::invalid_argument { explicit OrderedError(const char* kind):std::invalid_argument(kind){} };", "struct OrderedError : std::invalid_argument { std::uint32_t faults; explicit OrderedError(const char* kind,std::uint32_t bits=0):std::invalid_argument(kind),faults(bits){} };"),
        TimeGradientLanguage::Rust => common.to_string(),
    };
    format!("{imports}\n{common}\n{state}")
        .replace(
            "@INPUTFAULT@",
            &FaultSet::NONFINITE_INPUT.bits().to_string(),
        )
        .replace(
            "@CLAMPFAULT@",
            &FaultSet::MAGNITUDE_CLAMPED.bits().to_string(),
        )
        .replace(
            "@SANITIZEDFAULT@",
            &FaultSet::NONFINITE_SANITIZED.bits().to_string(),
        )
        .replace("@GAPFAULT@", &super::TIME_GAP_REJECTED.to_string())
        .replace("@BOUNDARYFAULT@", &super::BOUNDARY_HISTORY.to_string())
}
