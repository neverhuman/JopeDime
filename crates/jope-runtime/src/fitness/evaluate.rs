use super::*;

mod scoring;
mod validation;

/// Build [`Columns`] from `(name, values)` pairs.
#[cfg(test)]
pub(super) fn columns_from_pairs(pairs: &[(&str, Vec<f64>)]) -> Columns {
    pairs
        .iter()
        .map(|(name, values)| ((*name).to_owned(), values.clone()))
        .collect()
}

/// Bail with a uniform message when a specification reads columns the caller cannot supply.
pub fn require_no_columns(fitness: &Fitness, context: &str) -> Result<()> {
    let columns = fitness.columns();
    if !columns.is_empty() {
        bail!(
            "{context} cannot supply fitness columns {:?}",
            columns.into_iter().collect::<Vec<_>>()
        );
    }
    Ok(())
}
