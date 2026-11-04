//! The 30 structural coordinates from rank_prepare.py::features.

use super::{parse::ParsedProgram, FEATURE_NAMES};
use std::collections::BTreeMap;

pub fn is_feature(name: &str) -> bool {
    name == "steps"
        || name == "dsl_characters"
        || name.starts_with("count.")
        || name.starts_with("estimator.")
}

/// Include zeros for absent known operators, and preserve all reference stages
/// in the length/final-operator decisions even when their names are not in 412D.
pub fn extract(program: &ParsedProgram<'_>) -> BTreeMap<String, f64> {
    let mut counts = BTreeMap::new();
    for operation in program.operations() {
        *counts.entry(operation.as_str()).or_insert(0usize) += 1;
    }
    FEATURE_NAMES
        .iter()
        .filter(|name| is_feature(name))
        .map(|&name| {
            let value = if name == "steps" {
                program.operations().len() as f64
            } else if name == "dsl_characters" {
                program.source().chars().count() as f64
            } else if let Some(operation) = name.strip_prefix("count.") {
                counts.get(operation).copied().unwrap_or(0) as f64
            } else {
                let estimator = name.strip_prefix("estimator.").expect("structural name");
                f64::from(program.operations().last().is_some_and(|s| s == estimator))
            };
            (name.to_owned(), value)
        })
        .collect()
}

#[cfg(test)]
mod tests;
