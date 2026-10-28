//! Hash actual prediction inputs without retaining or publishing their cell contents.
use crate::fitted::FittedModel;
use anyhow::Result;
use feat_math::advanced_v3::RawTable;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, Write};

struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn identity(domain: &[u8], value: &impl Serialize) -> Result<String> {
    let mut writer = HashWriter(Sha256::new());
    writer.write_all(domain)?;
    writer.write_all(&[0])?;
    serde_json::to_writer(&mut writer, value)?;
    Ok(format!("{:x}", writer.0.finalize()))
}
pub(crate) fn bindings(fitted: &FittedModel, input: &RawTable) -> Result<(String, String)> {
    Ok((
        identity(
            b"jope.prediction-model/v1",
            &(&fitted.header, &fitted.model),
        )?,
        identity(b"jope.prediction-input/v1", input)?,
    ))
}

pub(crate) fn from_env<T>(run: impl FnOnce(Option<&jope_narrator::Tap>) -> Result<T>) -> Result<T> {
    let session = jope_narrator::journal_from_env().unwrap_or_else(|error| {
        eprintln!("jopedime: narration disabled - {error:#}");
        None
    });
    let result = run(session.as_ref().map(|(tap, _)| tap));
    if let Some((tap, journal)) = session {
        tap.emit(jope_narrator::LineageFact::RunStopped {
            status: if result.is_ok() {
                "completed"
            } else {
                "failed"
            }
            .into(),
        });
        if let Err(error) = journal.finish() {
            eprintln!("jopedime: narration incomplete - {error:#}");
        }
    }
    result
}

/// Called inside the actual executor, so 'started' never means waiting for its pool.
pub(crate) fn observe<T>(
    tap: Option<&jope_narrator::Tap>,
    bindings: impl FnOnce() -> Result<(String, String)>,
    rows: usize,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(tap) = tap else { return run() };
    let (model_sha256, input_sha256) = match bindings() {
        Ok(ids) => ids,
        Err(_) => {
            tap.emit(jope_narrator::LineageFact::Prediction {
                status: "binding_unavailable".into(),
                program_sha256: None,
            });
            return run();
        }
    };
    let start = std::time::Instant::now();
    tap.emit(jope_narrator::LineageFact::PredictionBound {
        status: "started".into(),
        model_sha256: model_sha256.clone(),
        input_sha256: input_sha256.clone(),
        rows,
        seconds: None,
    });
    let result = run();
    tap.emit(jope_narrator::LineageFact::PredictionBound {
        status: if result.is_ok() {
            "completed"
        } else {
            "failed"
        }
        .into(),
        model_sha256,
        input_sha256,
        rows,
        seconds: Some(start.elapsed().as_secs_f64()),
    });
    result
}
