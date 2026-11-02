use super::*;

pub fn artifact(path: &Path) -> Result<Artifact> {
    let path = fs::canonicalize(path)?;
    let value = Artifact {
        sha256: super::methods::digest(&fs::read(&path)?),
        path,
    };
    value.verify()?;
    Ok(value)
}
pub fn location(journal: &Path, observation: &Observation) -> Result<PathBuf> {
    Ok(journal
        .join("fitted-inference")
        .join(observation.request.identity()?))
}
/// The original worker output, deserialized once; standalone views borrow from it.
enum FittedOutput {
    Registered(Box<super::worker::RustWorkerOutput>),
    Synthesized(Box<super::synthesized::SynthesizedWorkerOutput>),
    External,
}
impl FittedOutput {
    fn parse(output: &Value) -> Result<Self> {
        Ok(if output["schema"] == "jope.panel-rust-worker-output/v1" {
            Self::Registered(Box::new(serde_json::from_value(output.clone())?))
        } else if output["schema"] == "jope.panel-synthesized-worker-output/v1" {
            Self::Synthesized(Box::new(serde_json::from_value(output.clone())?))
        } else {
            Self::External
        })
    }
    fn view(&self) -> Result<Option<FittedInferenceView<'_>>> {
        Ok(match self {
            Self::Registered(value) => Some(FittedInferenceView::registered(&value.artifact)?),
            Self::Synthesized(value) => Some(FittedInferenceView::synthesized(&value.artifact)?),
            Self::External => None,
        })
    }
}
/// Bind every backend entry to the exact authenticated fitted output and to the
/// public files materialized under `public`. Public source is re-emitted from the
/// fitted state and compared by identity, and each executed plan must run exactly the
/// shipped files; nothing is compiled, predicted or fitted here. An unsupported entry
/// must carry the reason the fitted state itself yields, so support is neither hidden
/// nor invented, and a same-recipe state swap cannot survive with rebound checksums.
pub(super) fn verify_entries(
    entries: &[Entry],
    output: &Value,
    fitted_sha256: &str,
    public: &Path,
) -> Result<()> {
    ensure!(entries.len() == ORDER.len(), "backend inventory incomplete");
    let fitted = FittedOutput::parse(output)?;
    let view = fitted.view()?;
    if let Some(view) = &view {
        ensure!(
            view.fitted_sha256() == fitted_sha256,
            "fitted state differs from the selected fitted checksum"
        );
    }
    for (entry, backend) in entries.iter().zip(ORDER) {
        ensure!(
            entry.backend() == backend,
            "backend qualification order changed"
        );
        let derived = match &view {
            Some(view) => Some(inference::export_inference(view, backend)?),
            None => None,
        };
        if let (Entry::Unsupported { reason, .. }, Some(derived)) = (entry, &derived) {
            let ExportOutcome::Unsupported { reason: actual, .. } = derived else {
                anyhow::bail!("fitted state supports a backend recorded as unsupported")
            };
            ensure!(
                serde_json::to_value(actual)? == *reason,
                "unsupported reason differs from the fitted state"
            );
        }
        if let Entry::Prepared {
            bundle,
            qualification,
            ..
        } = entry
        {
            bundle.validate()?;
            let Some(ExportOutcome::Emitted { bundle: rebuilt }) = derived else {
                anyhow::bail!("prepared backend is not emitted from the exact fitted state")
            };
            ensure!(
                rebuilt.identity()? == bundle.identity()?,
                "public bundle differs from the exact fitted state"
            );
            inference::verify_plan_binding(qualification, bundle)?;
            let public = public.join(name(backend));
            for (file, bytes) in &bundle.files {
                ensure!(
                    fs::read(public.join(file))? == bytes.as_bytes(),
                    "materialized public bundle changed"
                );
            }
            let manifest: Value = serde_json::from_slice(&fs::read(public.join("manifest.json"))?)?;
            ensure!(
                manifest == serde_json::to_value(&bundle.manifest)?,
                "materialized public manifest changed"
            );
            ensure!(
                serde_json::to_value(&qualification.expected)? == output["predictions"],
                "qualification reference differs from original fitted predictions"
            );
            ensure!(
                qualification.backend == backend
                    && qualification.bundle_sha256 == bundle.identity()?
                    && qualification.fitted_sha256 == fitted_sha256
                    && qualification.parity_policy == backend.parity_policy(),
                "exact fitted export identity changed"
            );
        }
    }
    Ok(())
}
