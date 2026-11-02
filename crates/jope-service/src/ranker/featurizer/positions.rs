//! Position namespace features from `rank_prepare.py::features`.

use std::collections::BTreeMap;

use super::parse::{self, ParsedProgram};

/// Extract exactly the 302 frozen position coordinates. The reference writes
/// the operation flag first and then lets recognized numeric/boolean parameter
/// tokens overwrite the same position key in source order.
pub fn extract(program: &ParsedProgram<'_>) -> BTreeMap<String, f64> {
    let mut features: BTreeMap<String, f64> = POSITION_FEATURE_NAMES
        .iter()
        .map(|name| ((*name).to_owned(), 0.0))
        .collect();

    for (position, raw_step) in program.raw_steps().enumerate() {
        let mut tokens = parse::tokens(raw_step);
        let Some(operation) = tokens.next() else {
            continue;
        };
        set_position_value(&mut features, position, operation, 1.0);

        for token in tokens {
            let Some((key, value)) = super::parameters::numeric_parameter_token(token) else {
                continue;
            };
            set_position_value(&mut features, position, key, value);
        }
    }
    features
}

fn set_position_value(
    features: &mut BTreeMap<String, f64>,
    position: usize,
    key: &str,
    value: f64,
) {
    let name = format!("position{position}.{key}");
    if let Some(slot) = features.get_mut(&name) {
        *slot = value;
    }
}

mod schema;
use schema::POSITION_FEATURE_NAMES;

#[cfg(test)]
mod oracle_data;
#[cfg(test)]
use oracle_data::{EXPECTED_GLOBAL_INDICES, PYTHON_ORACLE_GOLDENS};
#[cfg(test)]
mod tests;
