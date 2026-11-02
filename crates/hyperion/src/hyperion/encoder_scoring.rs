use super::*;

pub fn transform_scoring_matrix(
    encoded: &Matrix,
    encoded_names: &[String],
    state: &HyperionState,
) -> Result<Matrix> {
    if state.output_dim != state.output_names.len() {
        bail!("artifact Hyperion state has inconsistent output name count");
    }
    if state.input_names.len() != encoded_names.len() {
        bail!(
            "artifact Hyperion state expects {} encoded input columns but scoring produced {}",
            state.input_names.len(),
            encoded_names.len()
        );
    }
    for (idx, (expected, actual)) in state.input_names.iter().zip(encoded_names).enumerate() {
        if expected != actual {
            bail!(
                "artifact Hyperion input column {idx} expected '{expected}' but scoring produced '{actual}'"
            );
        }
    }
    if encoded.c != state.input_names.len() {
        bail!(
            "artifact Hyperion state expects {} input columns but scoring matrix has {}",
            state.input_names.len(),
            encoded.c
        );
    }
    if state.engine == "v7" {
        return transform_scoring_v7(encoded, state);
    }
    if state.engine == "orig_v2" {
        return transform_scoring_orig_v2(encoded, state);
    }
    let projection = load_projection_for_state(state)?;
    let block = encode_with_projection(encoded, &state.quantile_bounds, &state.sweep, &projection);
    Ok(place_columns(encoded, &block, state.feature_placement))
}

/// Scoring replay for v7-encoded artifacts: verify the recorded weights, then re-run the composite
/// sweep with the artifact's recorded plan (deterministic given the recorded seed).
#[cfg(feature = "hyperion-cpu")]
fn transform_scoring_v7(encoded: &Matrix, state: &HyperionState) -> Result<Matrix> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::config::SweepPlan;
    use crate::hyperion_v7::grid::quantize_u8_with_bounds;
    use crate::hyperion_v7::sweep::run_composite_sweep;
    use crate::hyperion_v7::EMB_KEEP;
    use crate::progress::NullSink;

    let path = PathBuf::from(&state.model.weights_path);
    if !path.is_file() {
        bail!(
            "artifact requires Hyperion v7 weights but {} is unavailable",
            path.display()
        );
    }
    let current_hash = hash_file(&path)?;
    if current_hash != state.model.weights_hash {
        bail!(
            "artifact requires Hyperion weights hash {} but {} has hash {}",
            state.model.weights_hash,
            path.display(),
            current_hash
        );
    }
    let model = V7Model::load(&path, &state.sweep)?;
    let plan = SweepPlan {
        sweeps: state.sweeps.max(1) as u32,
        scales: if state.scales.is_empty() {
            vec![3, 5, 7, 9]
        } else {
            state.scales.clone()
        },
        batch: state.batch.max(1),
        seed: state.sweep.seed,
        target_chunks_per_sweep: 8,
        fixed_perms: None,
    };
    let xu8 = quantize_u8_with_bounds(encoded, &state.quantile_bounds, state.sweep.pad_u8);
    let out = run_composite_sweep(
        &model,
        &xu8,
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: state.sweep.pad_u8,
            quant_scale: state.sweep.quant_scale,
            max_patch: state.sweep.max_patch,
        },
        &plan,
        &NullSink,
        |_| {},
    )?;
    let keep = state.output_dim.min(EMB_KEEP);
    let block = column_major_block(&out.row_emb, encoded.n, EMB_KEEP, keep);
    Ok(place_columns(encoded, &block, state.feature_placement))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn transform_scoring_v7(_encoded: &Matrix, _state: &HyperionState) -> Result<Matrix> {
    bail!(
        "artifact was encoded with the Hyperion v7 engine but feat-core was compiled without the 'hyperion-cpu' feature"
    )
}

#[cfg(feature = "hyperion-cpu")]
fn transform_scoring_orig_v2(encoded: &Matrix, state: &HyperionState) -> Result<Matrix> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::sweep::run_orig_v2_cell_sweep;

    let orig = state
        .orig_v2
        .as_ref()
        .context("artifact orig-v2 engine is missing its train-fitted rank state")?;
    if orig.sorted_fit_columns.len() != encoded.c {
        bail!(
            "artifact orig-v2 rank state has {} columns but scoring has {}",
            orig.sorted_fit_columns.len(),
            encoded.c
        );
    }
    let weights = PathBuf::from(&state.model.weights_path);
    let manifest = PathBuf::from(&state.model.manifest_path);
    orig_v2::validate_authority(&weights, orig_v2::ORIG_V2_WEIGHTS_SHA256, "weights")?;
    orig_v2::validate_authority(&manifest, orig_v2::ORIG_V2_MANIFEST_SHA256, "manifest")?;
    let weights_hash = hash_file(&weights)?;
    let manifest_hash = hash_file(&manifest)?;
    if weights_hash != state.model.weights_hash || manifest_hash != state.model.manifest_hash {
        bail!("artifact orig-v2 model authority hash mismatch");
    }
    let model = V7Model::load(&weights, &state.sweep)?;
    let grid = orig_v2::rank_replay_grid(
        encoded,
        &orig.sorted_fit_columns,
        state.sweep.q_low,
        state.sweep.q_high,
    )?;
    let output = run_orig_v2_cell_sweep(
        &model,
        &grid,
        state.sweep.quant_scale,
        state.sweep.max_patch,
        &orig.plan(),
    )?;
    let block = orig_v2::row_major_cells(&output.logits, 0, encoded.n, encoded.c);
    Ok(place_columns(encoded, &block, state.feature_placement))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn transform_scoring_orig_v2(_encoded: &Matrix, _state: &HyperionState) -> Result<Matrix> {
    bail!(
        "artifact was encoded with Hyperion orig-v2 but feat-core was compiled without the 'hyperion-cpu' feature"
    )
}
