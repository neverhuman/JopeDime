//! Native, original-source regressors available to AutoML callers and same-cut benchmarks.
//!
//! Sol is included byte-for-byte; Gemini has only four recorded type annotations. Adapters own transport and registration;
//! they do not replace its preprocessing, fitting, or prediction procedure.

#[rustfmt::skip]
#[allow(clippy::all)]
#[path = "GPT6p1_Sol_Max.rs"]
mod submission;

pub use submission::{Regressor as Gpt6p1SolMax, RegressorApi};

// Preserve the frozen entrant's bytes. Style and unused private fields are
// reviewed in the submission record; the adapters retain normal workspace lints.
#[rustfmt::skip]
#[allow(clippy::all, dead_code)]
#[path = "gemini_argon_4.rs"]
mod gemini;
pub use gemini::{Regressor as GeminiArgon4, RegressorApi as GeminiRegressorApi};

/// Select the annotation-only Gemini submission through either benchmark path.
pub const GEMINI_ARGON_4: &str = "gemini-argon-4";
/// SHA-256 of the corrected, otherwise unchanged submitted source.
pub const GEMINI_ARGON_4_SHA256: &str =
    "b0aa42d24d0b067e0597f6f538b14cab75dca8b976731668c894d80684d4992d";
/// Source bytes corresponding to the compiled Gemini module.
pub const GEMINI_ARGON_4_SOURCE: &str = include_str!("gemini_argon_4.rs");

/// Frozen candidate identifier. This is a regressor, not a GP grammar operator.
pub const GPT6P1_SOL_MAX: &str = "GPT6p1_Sol_Max";
/// Hash of the standalone source, frozen before repository-code inspection or execution.
pub const GPT6P1_SOL_MAX_SHA256: &str =
    "fe5bb21ed62edb3aebc1bd141d54834dd3f42b97cb9b0366f2e6b24bc99f75a8";
/// The exact file can also be built directly with `rustc --crate-type rlib`.
pub const GPT6P1_SOL_MAX_SOURCE: &str = include_str!("GPT6p1_Sol_Max.rs");

/// One immutable, selectable native algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Definition {
    pub id: &'static str,
    pub source_sha256: &'static str,
    pub source_file: &'static str,
}

/// Native AutoML candidate registry. Entries have no dataset-specific configuration.
pub const REGRESSORS: &[Definition] = &[
    Definition {
        id: GPT6P1_SOL_MAX,
        source_sha256: GPT6P1_SOL_MAX_SHA256,
        source_file: "GPT6p1_Sol_Max.rs",
    },
    Definition {
        id: GEMINI_ARGON_4,
        source_sha256: GEMINI_ARGON_4_SHA256,
        source_file: "gemini_argon_4.rs",
    },
];

/// Resolve a requested candidate; an unknown name never silently selects another model.
pub fn definition(id: &str) -> Result<&'static Definition, String> {
    REGRESSORS
        .iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| format!("unknown native regressor {id:?}"))
}

/// Independently fitted state for one AutoML candidate.
pub struct FittedRegressor {
    model: Model,
    width: usize,
}

enum Model {
    Sol(Box<Gpt6p1SolMax>),
    Gemini(Box<GeminiArgon4>),
}

fn validate_length(values: &[f64], rows: usize, columns: usize) -> Result<(), String> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| String::from("table dimensions overflow"))?;
    if values.len() != expected {
        return Err(String::from("flat table length does not match dimensions"));
    }
    Ok(())
}

/// Fit the challenge's row-major table with the target in its last column.
pub fn fit_flat(
    id: &str,
    table: &[f64],
    rows: usize,
    columns: usize,
    seed: u64,
) -> Result<FittedRegressor, String> {
    definition(id)?;
    if rows == 0 || columns == 0 {
        return Err(String::from(
            "nonempty training rows and target column required",
        ));
    }
    validate_length(table, rows, columns)?;
    for row in table.chunks_exact(columns) {
        if !row[columns - 1].is_finite() {
            return Err(String::from("training targets must be finite"));
        }
        if row[..columns - 1].iter().any(|value| value.is_infinite()) {
            return Err(String::from("infinite training feature"));
        }
    }
    let model = match id {
        GPT6P1_SOL_MAX => {
            let mut model = Gpt6p1SolMax::new(seed);
            model.fit(table, rows, columns)?;
            Model::Sol(Box::new(model))
        }
        GEMINI_ARGON_4 => {
            let mut model = GeminiArgon4::new(seed);
            model.fit(table, rows, columns)?;
            Model::Gemini(Box::new(model))
        }
        _ => unreachable!("definition checked above"),
    };
    Ok(FittedRegressor {
        model,
        width: columns - 1,
    })
}

/// Fit only the supplied training rows. The adapter appends each training target last.
pub fn fit_rows(
    id: &str,
    features: &[Vec<f64>],
    targets: &[f64],
    seed: u64,
) -> Result<FittedRegressor, String> {
    definition(id)?;
    if features.is_empty() || targets.len() != features.len() {
        return Err(String::from(
            "nonempty training rows and matching targets required",
        ));
    }
    let width = features[0].len();
    let columns = width
        .checked_add(1)
        .ok_or_else(|| String::from("training width overflow"))?;
    let length = features
        .len()
        .checked_mul(columns)
        .ok_or_else(|| String::from("training dimensions overflow"))?;
    let mut table = Vec::new();
    table
        .try_reserve(length)
        .map_err(|_| String::from("training transport allocation failed"))?;
    for (row, target) in features.iter().zip(targets) {
        if row.len() != width {
            return Err(String::from("ragged training features"));
        }
        table.extend_from_slice(row);
        table.push(*target);
    }
    fit_flat(id, &table, features.len(), columns, seed)
}

impl FittedRegressor {
    /// Predict feature rows only. No target argument or placeholder is accepted.
    pub fn predict_rows(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        let length = features
            .len()
            .checked_mul(self.width)
            .ok_or_else(|| String::from("prediction dimensions overflow"))?;
        let mut flat = Vec::new();
        flat.try_reserve(length)
            .map_err(|_| String::from("prediction transport allocation failed"))?;
        for row in features {
            if row.len() != self.width {
                return Err(String::from("prediction feature width changed"));
            }
            flat.extend_from_slice(row);
        }
        self.predict_flat(&flat, features.len(), self.width)
    }

    /// Predict feature-only flat rows, retaining the learner's shape validation.
    pub fn predict_flat(
        &self,
        features: &[f64],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f64>, String> {
        if columns != self.width {
            return Err(String::from("prediction feature width changed"));
        }
        if rows > isize::MAX as usize / std::mem::size_of::<f64>() {
            return Err(String::from("prediction output dimensions overflow"));
        }
        validate_length(features, rows, columns)?;
        if features.iter().any(|value| value.is_infinite()) {
            return Err(String::from("infinite prediction feature"));
        }
        match &self.model {
            Model::Sol(model) => model.predict(features, rows, columns),
            Model::Gemini(model) => model.predict(features, rows, columns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_shapes_and_failed_refit_are_explicit() {
        let mut model = Gpt6p1SolMax::new(7);
        assert!(model.predict(&[], 0, 0).is_err());
        assert!(model.fit(&[], 0, 1).is_err());
        assert!(model.fit(&[], 1, 0).is_err());
        assert!(model.fit(&[], usize::MAX, 2).is_err());
        assert!(model.fit(&[1.0, 2.0], 2, 2).is_err());
        model.fit(&[2.0, 4.0, 6.0], 3, 1).unwrap();
        assert_eq!(model.predict(&[], 3, 0).unwrap(), vec![4.0; 3]);
        assert_eq!(model.predict(&[], 0, 0).unwrap(), Vec::<f64>::new());
        assert!(model.predict(&[0.0], 1, 1).is_err());
        assert!(model.fit(&[f64::NAN], 1, 1).is_err());
        assert!(model.predict(&[], 1, 0).is_err());
        assert!(model.fit(&[f64::INFINITY, 1.0], 1, 2).is_err());
        assert!(model.fit(&[1.0, f64::INFINITY], 1, 2).is_err());
    }

    #[test]
    fn prediction_is_row_local_and_refit_matches_fresh_state() {
        // Contract checks only; no accuracy threshold, model comparison, or tuning.
        let table: Vec<f64> = (0..80)
            .flat_map(|i| {
                [
                    if i % 7 == 0 { f64::NAN } else { i as f64 },
                    (i % 5) as f64,
                    (i % 13) as f64,
                ]
            })
            .collect();
        let mut model = Gpt6p1SolMax::new(19);
        model.fit(&table, 80, 3).unwrap();
        let rows = [f64::NAN, 2.0, 4.0, 1.0, f64::MAX, 0.0];
        let all = model.predict(&rows, 3, 2).unwrap();
        assert!(all.iter().all(|value| value.is_finite()));
        for i in 0..3 {
            assert_eq!(
                model.predict(&rows[2 * i..2 * i + 2], 1, 2).unwrap()[0],
                all[i]
            );
        }
        let permuted = model
            .predict(&[f64::MAX, 0.0, f64::NAN, 2.0, 4.0, 1.0], 3, 2)
            .unwrap();
        assert_eq!(permuted, vec![all[2], all[0], all[1]]);
        assert_eq!(model.predict(&rows, 3, 2).unwrap(), all);
        assert!(model.predict(&[f64::INFINITY, 0.0], 1, 2).is_err());
        assert!(model.predict(&[], usize::MAX, 2).is_err());
        model.fit(&[f64::NAN, 8.0, 0.0, 8.0], 2, 2).unwrap();
        let mut fresh = Gpt6p1SolMax::new(19);
        fresh.fit(&[f64::NAN, 8.0, 0.0, 8.0], 2, 2).unwrap();
        assert_eq!(model.predict(&[3.0], 1, 1), fresh.predict(&[3.0], 1, 1));
    }

    #[test]
    fn registry_and_transport_preserve_target_placement() {
        assert_eq!(REGRESSORS.len(), 2);
        assert!(definition("missing").is_err());
        let model = fit_rows(GPT6P1_SOL_MAX, &[vec![10.0], vec![20.0]], &[7.0, 7.0], 1).unwrap();
        assert_eq!(
            model.predict_rows(&[vec![1234.0], vec![f64::NAN]]).unwrap(),
            vec![7.0; 2]
        );
        assert!(model.predict_rows(&[vec![1.0, 2.0]]).is_err());
        assert!(fit_rows(GPT6P1_SOL_MAX, &[vec![1.0], vec![]], &[1.0, 2.0], 1).is_err());
        assert!(fit_rows(GPT6P1_SOL_MAX, &[vec![1.0]], &[], 1).is_err());
        let model = fit_rows(GPT6P1_SOL_MAX, &[vec![], vec![]], &[2.0, 6.0], 1).unwrap();
        assert_eq!(
            model.predict_rows(&[vec![], vec![], vec![]]).unwrap(),
            vec![4.0; 3]
        );
    }

    #[test]
    fn wide_missing_and_extreme_finite_inputs_remain_defined() {
        let mut model = Gpt6p1SolMax::new(2);
        model
            .fit(&[f64::NAN, -f64::MAX, f64::NAN, f64::MAX], 2, 2)
            .unwrap();
        assert!(model
            .predict(&[f64::NAN, f64::MAX], 2, 1)
            .unwrap()
            .iter()
            .all(|p| p.is_finite()));
        let table: Vec<f64> = (0..8)
            .flat_map(|i| {
                let mut row = vec![f64::NAN; 110];
                row[i] = i as f64;
                row.push(i as f64);
                row
            })
            .collect();
        model.fit(&table, 8, 111).unwrap();
        assert_eq!(
            model.predict(&vec![f64::NAN; 110], 1, 110).unwrap().len(),
            1
        );
    }
}

#[cfg(test)]
#[path = "gemini_contract_tests.rs"]
mod gemini_contract_tests;
