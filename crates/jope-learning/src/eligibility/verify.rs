use super::*;

#[cfg(test)]
pub(super) fn combine(parts: &[Predictions]) -> Result<Predictions> {
    match parts.first().context("missing OOF predictions")? {
        Predictions::Values(_) => {
            let mut out = Vec::new();
            for part in parts {
                let Predictions::Values(values) = part else {
                    bail!("mixed OOF prediction kinds")
                };
                out.extend(values.iter().copied());
            }
            Ok(Predictions::Values(out))
        }
        Predictions::Probabilities { classes, .. } => {
            let mut out = Vec::new();
            for part in parts {
                let Predictions::Probabilities {
                    classes: order,
                    values,
                } = part
                else {
                    bail!("mixed OOF prediction kinds")
                };
                ensure!(order == classes, "OOF class order changed");
                out.extend(values.iter().cloned());
            }
            Ok(Predictions::Probabilities {
                classes: classes.clone(),
                values: out,
            })
        }
    }
}

/// Bind scored bytes to retained worker outputs, including native JSON artifact envelopes.
pub(super) fn verify_measured_outputs(
    state: &State,
    worker: &WorkerOutcome,
    candidate: &FoldPredictions,
    predictions: &Predictions,
    inner_fit: bool,
) -> Result<()> {
    let mut found_actual_fit = !inner_fit;
    let mut found_predictions = worker
        .outputs
        .iter()
        .any(|output| output.sha256 == candidate.predictions_sha256);
    let mut found_model = worker
        .outputs
        .iter()
        .any(|output| output.sha256 == candidate.fitted_artifact_sha256);
    for output in &worker.outputs {
        if found_predictions && found_model && found_actual_fit {
            break;
        }
        // Binary model files may be large; only bounded JSON envelopes can embed predictions/state.
        if output.path.extension().and_then(|value| value.to_str()) != Some("json")
            || state
                .read_blob(&output.sha256)?
                .context("retained output missing")?
                .len() as u64
                > 128 * 1024 * 1024
        {
            continue;
        }
        let Ok(envelope) = serde_json::from_slice::<serde_json::Value>(
            &state
                .read_blob(&output.sha256)?
                .context("retained output missing")?,
        ) else {
            continue;
        };
        if let Some(values) = envelope.get("predictions") {
            if serde_json::from_value::<Predictions>(values.clone())
                .ok()
                .as_ref()
                == Some(predictions)
            {
                found_predictions = true;
            }
        }
        if envelope
            .get("schema")
            .and_then(|value| value.as_str())
            .is_some_and(|schema| {
                matches!(
                    schema,
                    "jope.panel-rust-worker-output/v1" | "jope.panel-synthesized-worker-output/v1"
                )
            })
            && envelope
                .get("artifact_sha256")
                .and_then(|value| value.as_str())
                == Some(&candidate.fitted_artifact_sha256)
        {
            let stored: serde_json::Value = blob(state, &candidate.fitted_artifact_sha256)?;
            if envelope.get("artifact") == Some(&stored) {
                found_model = true;
                if inner_fit {
                    ensure!(
                        envelope["schema"] == "jope.panel-synthesized-worker-output/v1"
                            && envelope["physical_fit_starts"].as_u64() == Some(1)
                            && envelope["artifact"]["recipe"]["executable_sha256"]
                                == candidate.program_sha256,
                        "inner fit lacks actual one-fit synthesized output"
                    );
                    found_actual_fit = true;
                }
            }
        }
    }
    ensure!(
        found_predictions && found_model && found_actual_fit,
        "scored predictions or fitted state are absent from the bound worker outputs"
    );
    Ok(())
}
