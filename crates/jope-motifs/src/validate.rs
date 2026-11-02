//! Checks of a private fragment table (section 5 of the design).
//!
//! A reference is a macro, so the proof obligation for every recorded occurrence is the
//! byte-exact round trip. Everything after the canonical text is a function of that text and
//! the data, so comparing fits of the same text twice tests engine determinism and is labelled
//! that way. The checks that are not vacuous are context compatibility and the pinned-digest
//! control tests. Context compatibility counts only real fits of contexts whose text differs
//! from every recorded host, compared both as fitted and with the task's default head written
//! out; a context that repeats a host is recorded as not independent and never satisfies it.
//!
//! No check passes without being exercised. A check class that examined nothing records a
//! `<class>_not_exercised` failure (`context_not_exercised` for context compatibility), so
//! an empty artifact list, a fragment whose every fit was refused, or a missing context can
//! never be reported or tallied as a pass. A table with no fragments never passes, and the
//! source check ([`verify_source`]) never confirms a table that records no occurrence.

use crate::expand::{expand, rewrite, At};
use crate::fragment::{classification_schema, input_stage};
use crate::source::{QualifiedProgram, QualifiedSet, TaskLabel};
use crate::synthetic::{Shape, Synthetic};
use crate::table::{MotifDef, MotifTable};
use crate::{Code, NAME_PREFIX};
use anyhow::{ensure, Result};
use feat_math::advanced_v3::{
    emit_bound, emit_fitted_cpp17, fit_table, predict_multiclass_table, predict_table,
    supports_fitted_cpp17, validate_v3_genome, BoundModel, DataSchema,
};
use feat_math::kernel::pipeline::PipelineSlots;
use feat_math::transpile::{self, EmitOptions};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod analysis;
mod run;
mod types;
use analysis::*;
use run::check_motif;
pub use types::*;

pub fn verify_source(
    validation: &mut Validation,
    table: &MotifTable,
    set: &QualifiedSet,
) -> Option<&'static str> {
    let fresh: BTreeMap<&str, &QualifiedProgram> = set
        .programs()
        .iter()
        .map(|program| (program.record_sha256(), program))
        .collect();
    let mut examined = 0_usize;
    let mut current = true;
    for motif in &table.body.motifs {
        for occurrence in &motif.provenance {
            examined += 1;
            current &= fresh
                .get(occurrence.record_sha256.as_str())
                .is_some_and(|program| {
                    program.comparison_sha256() == occurrence.comparison_sha256
                        && program.recipe_sha256() == occurrence.recipe_sha256
                        && program.task() == occurrence.task
                        && program.dataset_sha256() == occurrence.dataset_sha256
                        && program.cut_sha256() == occurrence.cut_sha256
                        && *program.label() == occurrence.label
                        && !program.label().is_adaptive()
                        && table
                            .body
                            .hosts
                            .get(&occurrence.recipe_sha256)
                            .is_some_and(|host| host == program.program())
                });
        }
    }
    validation.source_digest_equal = Some(set.source_sha256() == table.body.source.sha256);
    validation.source_verified = examined > 0 && current;
    if examined == 0 {
        Some("source_not_exercised")
    } else if current {
        None
    } else {
        Some("source_not_confirmed")
    }
}

/// Check a table: V0 first (any failure is an error), then C1 to C4 for every fragment.
pub fn validate(table: &MotifTable, options: &ValidateOptions) -> Result<Validation> {
    ensure!(options.hosts_per_fragment >= 1, Code("invalid_parameters"));
    table.verify()?;
    let tables = Synthetic::all()?;
    let mut motifs = Vec::with_capacity(table.body.motifs.len());
    for (index, motif) in table.body.motifs.iter().enumerate() {
        let (result, _artifacts) = check_motif(table, index, motif, options, &tables)?;
        motifs.push(result);
    }
    Ok(Validation {
        schema: VALIDATION_SCHEMA.to_owned(),
        table_sha256: table.table_sha256.clone(),
        source_verified: false,
        source_digest_equal: None,
        motifs,
    })
}

/// One emitted artifact and its kind.
#[derive(Clone, Debug)]
#[cfg(test)]
mod tests;
