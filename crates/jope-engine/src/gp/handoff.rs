//! Sealed selection, final refit and export handoff (W5-T07).
//!
//! [`seal`] derives a completed search's selected subject only from the
//! independent [`super::selection_replay`] and the verified
//! [`super::lineage`] — never from `search.json`'s summary — and binds it to
//! its fold fit, fold model and the authorised final-fit scope: the selected
//! cut's training rows. The cut's validation rows (the selection labels) and
//! the original final test stay outside every fit.
//!
//! [`export_sealed`] fits exactly the sealed recipe on exactly that scope
//! through the fitted-view export, requires the result to carry the selected
//! fold fit's fitted identity ([`super::lineage::verify_final_refit`]), and
//! records a handoff receipt. A failed or mismatched export keeps the sealed
//! selection and its fold fit as the selected fitted predictor and reports
//! `export_failed`; it never substitutes another program or another fit.
use super::{lineage, relative_artifact, selection_replay};
use crate::manifest::{read_json, write_json};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub const SEAL_SCHEMA: &str = "parallel.gp_selection_seal.v1";
pub const HANDOFF_SCHEMA: &str = "parallel.gp_export_handoff.v1";
/// The only final-fit scope the fitted-view export implements.
pub const FINAL_FIT_SCOPE: &str = "selected_cut_training_rows";

/// The frozen inputs of one completed search.
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub root: &'a Path,
    pub output: &'a Path,
    pub seeds: Option<&'a Path>,
    pub vocabulary: Option<&'a Path>,
}

/// One completed search's predictor handoff, as the cycle pipeline requests it.
pub struct SelectedExport<'a> {
    pub root: &'a Path,
    pub search_output: &'a Path,
    pub prepared: &'a Path,
    pub cut: usize,
    pub seeds: Option<&'a Path>,
    pub vocabulary: Option<&'a Path>,
    pub executable: &'a Path,
    pub executable_sha: &'a str,
    pub predictor: &'a Path,
}

/// Export the selected subject of one completed search as a fitted predictor.
/// The selection is the sealed, independently replayed one; the summary's
/// `best_program` is never read. The seal (`*.seal.json`) and handoff
/// (`*.handoff.json`) receipts are written beside the predictor.
pub fn export_selected(export: &SelectedExport) -> Result<Value> {
    use crate::manifest::file_hash;
    ensure!(
        file_hash(export.executable)? == export.executable_sha,
        "pinned predictor evaluator changed"
    );
    let inputs = Inputs {
        root: export.root,
        output: export.search_output,
        seeds: export.seeds,
        vocabulary: export.vocabulary,
    };
    std::fs::create_dir_all(
        export
            .predictor
            .parent()
            .context("predictor output parent absent")?,
    )?;
    let seal_path = export.predictor.with_extension("seal.json");
    let seal = write_seal(&inputs, &seal_path)?;
    let scope = &seal["final_fit_scope"];
    ensure!(
        export.root.canonicalize()?.join(
            scope["prepared"]
                .as_str()
                .context("sealed preparation absent")?
        ) == export.prepared.canonicalize()?
            && scope["cut"] == export.cut,
        "the sealed selection belongs to another preparation or cut"
    );
    let handoff = export_sealed(&inputs, &seal_path, export.executable, export.predictor)?;
    ensure!(
        handoff["status"] == "exported",
        "sealed selection not exported ({}): {}",
        handoff["status"],
        handoff["failure"]
    );
    let receipt_path = export.predictor.with_extension("fit.json");
    let handoff_path = export.predictor.with_extension("handoff.json");
    Ok(json!({"passed":true,"predictor":export.predictor,
        "predictor_sha256":handoff["predictor"]["sha256"],
        "program_sha256":seal["selected"]["program_sha256"],
        "receipt":receipt_path,"receipt_sha256":file_hash(&receipt_path)?,
        "seal":seal_path,"seal_sha256":file_hash(&seal_path)?,
        "handoff":handoff_path,"handoff_sha256":file_hash(&handoff_path)?,
        "executable_sha256":export.executable_sha}))
}

/// The seal of a completed search, recomputed from its evidence.
pub fn seal(inputs: &Inputs) -> Result<(Value, lineage::SelectedLineage)> {
    let verified =
        lineage::verify_search(inputs.root, inputs.output, inputs.seeds, inputs.vocabulary)?;
    let root = inputs.root.canonicalize()?;
    let output = inputs.output.canonicalize()?;
    let request: Value = read_json(&output.join("search-request.json"))?;
    let replay = &verified.selection.replay;
    let (status, selected) = match (&verified.selected, &replay.winner) {
        (Some(chosen), Some(winner)) => (
            "selected",
            json!({"position":winner.position,"executable_program":winner.genome,
                "program_sha256":chosen.chain.executable_sha256,
                "original_sha256":chosen.chain.original_sha256,"source":chosen.chain.source,
                "adapted":chosen.chain.adapted,"ancestry":chosen.ancestry,
                "objective_rmse_original_units":winner.objective,
                "fold_fit_completion":chosen.fold_fit_completion,"fold_model":chosen.fold_model,
                "training_sha256":chosen.training_sha256}),
        ),
        // The incumbent's chain and fitted state belong to its direct search.
        (None, Some(_)) if verified.inherited_selected => ("inherited_incumbent", Value::Null),
        _ => ("no_measured_candidate", Value::Null),
    };
    let seal = json!({"schema":SEAL_SCHEMA,"status":status,
        "search":relative_artifact(&root, &output.join("search.json"))?,
        "search_request":relative_artifact(&root, &output.join("search-request.json"))?,
        "observations_sha256":verified.selection.observations_sha256,
        "selection_contract":selection_replay::CONTRACT,"stop_reason":replay.stop,
        "seeds_sha256":request["seeds_sha256"],"vocabulary_sha256":request["vocabulary_sha256"],
        "selected":selected,
        "final_fit_scope":{"kind":FINAL_FIT_SCOPE,"prepared":request["prepared"],
            "prepared_sha256":request["prepared_sha256"],"cut":request["cut"],
            "training_sha256":request["training_sha256"],
            "held_out_validation_sha256":request["validation_sha256"],
            "held_out_labels_in_fit":false,"final_tests_opened":false}});
    Ok((seal, verified))
}

/// Record the seal at `destination` (immutable: an existing seal must match).
pub fn write_seal(inputs: &Inputs, destination: &Path) -> Result<Value> {
    let (seal, _) = seal(inputs)?;
    if destination.exists() {
        ensure!(
            read_json::<Value>(destination)? == seal,
            "an existing seal differs from the replayed selection"
        );
    } else {
        write_json(destination, &seal)?;
    }
    Ok(seal)
}

/// A recorded seal is valid only while it equals the seal recomputed from the
/// search's evidence: no field of it is trusted on its own.
pub fn verify_seal(inputs: &Inputs, seal_path: &Path) -> Result<(Value, lineage::SelectedLineage)> {
    verify_seal_bytes(inputs, &std::fs::read(seal_path)?)
}

/// [`verify_seal`] of the exact seal bytes a caller already read and bound
/// (W1 route R8): the seal compared with the replay is the recorded one.
pub fn verify_seal_bytes(
    inputs: &Inputs,
    recorded: &[u8],
) -> Result<(Value, lineage::SelectedLineage)> {
    let recorded: Value = serde_json::from_slice(recorded)?;
    let (seal, verified) = seal(inputs)?;
    ensure!(
        recorded == seal,
        "recorded seal differs from the replayed selection"
    );
    Ok((seal, verified))
}

/// Export exactly the sealed selection, or record why it was not exported.
/// The handoff receipt is written beside the predictor as `*.handoff.json`.
pub fn export_sealed(
    inputs: &Inputs,
    seal_path: &Path,
    executable: &Path,
    predictor: &Path,
) -> Result<Value> {
    let (seal, verified) = verify_seal(inputs, seal_path)?;
    let root = inputs.root.canonicalize()?;
    let receipt_path = predictor.with_extension("handoff.json");
    if receipt_path.exists() {
        // A resumed consumer re-verifies the recorded outcome; a recorded
        // failure is terminal and is never retried with anything else.
        let recorded: Value = read_json(&receipt_path)?;
        ensure!(
            recorded["schema"] == HANDOFF_SCHEMA
                && recorded["seal"] == relative_artifact(&root, seal_path)?,
            "recorded handoff belongs to another seal"
        );
        if recorded["status"] == "exported" {
            ensure!(
                recorded["predictor"] == relative_artifact(&root, predictor)?,
                "exported predictor changed after its handoff"
            );
            lineage::verify_final_refit(&root, &verified, predictor)?;
        }
        return Ok(recorded);
    }
    let mut handoff = json!({"schema":HANDOFF_SCHEMA,"seal":relative_artifact(&root, seal_path)?,
        "status":seal["status"],"program_sha256":seal["selected"]["program_sha256"],
        "selected_fitted_state":seal["selected"]["fold_model"],
        "final_fit_scope":seal["final_fit_scope"],"predictor":null,"final_refit":null,
        "failure":null,"final_tests_opened":false});
    if seal["status"] == "selected" {
        let scope = &seal["final_fit_scope"];
        let prepared = root.join(scope["prepared"].as_str().context("sealed preparation")?);
        let cut = scope["cut"].as_u64().context("sealed cut")? as usize;
        let genome = seal["selected"]["executable_program"]
            .as_str()
            .context("sealed recipe")?;
        // W1 follow-up 2c: the export projects its fitting input under the
        // search's declared root; the worker receives only that projection.
        let exported =
            super::fit_view_isolated_in_root(executable, &root, &prepared, cut, genome, predictor)
                .and_then(|()| lineage::verify_final_refit(&root, &verified, predictor));
        match exported {
            Ok(refit) => {
                handoff["status"] = json!("exported");
                handoff["predictor"] = relative_artifact(&root, predictor)?;
                handoff["final_refit"] = serde_json::to_value(&refit)?;
            }
            Err(error) => {
                // The selected subject and its fold fit remain the selection;
                // nothing else is fitted or handed off in its place.
                handoff["status"] = json!("export_failed");
                handoff["failure"] = json!(format!("{error:#}"));
            }
        }
    }
    write_json(&receipt_path, &handoff)?;
    Ok(handoff)
}
