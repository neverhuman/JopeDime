//! Hash actual prediction inputs without retaining or publishing their cell contents.
use crate::fitted::FittedModel;
use anyhow::Result;
use feat_math::advanced_v3::RawTable;
use serde::Serialize;
use sha2::{Digest, Sha256};
mod typed_hash;

pub(crate) fn identity(domain: &[u8], value: &impl Serialize) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    value.serialize(&mut typed_hash::Encoder(&mut hash))?;
    Ok(format!("{:x}", hash.finalize()))
}
pub(crate) fn bindings(fitted: &FittedModel, input: &RawTable) -> Result<(String, String)> {
    Ok((
        identity(
            b"jope.prediction-model/v2",
            &(&fitted.header, &fitted.model),
        )?,
        identity(b"jope.prediction-input/v2", input)?,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prediction_binding_preserves_every_nonfinite_and_signed_zero_payload() {
        let values = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
            f64::from_bits(0x7ff8_0000_0000_0043),
        ];
        let hashes = values
            .iter()
            .map(|v| identity(b"typed-test", v).unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(hashes.len(), values.len());
        assert_ne!(
            identity(b"typed-test", &["a", "bc"]).unwrap(),
            identity(b"typed-test", &["ab", "c"]).unwrap()
        );
        assert_ne!(
            identity(b"typed-test", &1u64).unwrap(),
            identity(b"typed-test", &1i64).unwrap()
        );
    }
}
