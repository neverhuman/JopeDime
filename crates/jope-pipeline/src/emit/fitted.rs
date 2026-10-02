//! Fitted C++17 export from complete numeric training rows, separate from recipes.
use super::{Artifact, Emitted, Language};
use anyhow::{ensure, Context, Result};
use feat_math::{emit_fitted_cpp17, fit_legacy_numeric_bound, GenomeIr};
use std::path::Path;

pub(super) fn emit(
    genome: &str,
    languages: &[Language],
    directory: &Path,
    rows: &[Vec<f64>],
) -> Result<Emitted> {
    let mut x = Vec::with_capacity(rows.len());
    let mut y = Vec::with_capacity(rows.len());
    for row in rows {
        let (target, features) = row
            .split_last()
            .context("training row needs features and target")?;
        ensure!(
            !features.is_empty(),
            "training row needs at least one feature"
        );
        x.push(features.to_vec());
        y.push(*target);
    }
    // Validate and fit before creating any output. The existing bound numeric
    // ingress refuses empty, ragged and nonfinite training matrices.
    let parsed = GenomeIr::parse(genome.trim()).context("parse fitted export genome")?;
    let model =
        fit_legacy_numeric_bound(&parsed, &x, &y).context("fit C++17 export on training rows")?;
    let fitted = emit_fitted_cpp17(&model).context("emit fitted C++17")?;
    let recipes: Vec<_> = languages
        .iter()
        .copied()
        .filter(|language| *language != Language::Cpp17Fitted)
        .collect();
    let mut emitted = super::emit(genome, &recipes, directory)?;
    std::fs::create_dir_all(directory.join("cpp17-fitted"))?;
    for (language, name, source) in [
        (
            "cpp17-fitted",
            "cpp17-fitted/model.cpp",
            fitted.source.as_str(),
        ),
        (
            "cpp17-fitted-codec",
            "cpp17-fitted/model.codec.json",
            fitted.codec.as_str(),
        ),
        (
            "cpp17-fitted-standalone",
            "cpp17-fitted/standalone.cpp",
            STANDALONE,
        ),
    ] {
        let build = (language == "cpp17-fitted-standalone").then(|| {
            "g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -DJOPE_STANDALONE cpp17-fitted/standalone.cpp -o model-cpp17-fitted".to_owned()
        });
        emitted.artifacts.push(Artifact {
            language,
            build,
            ..super::write(directory, name, source)?
        });
    }
    let previous = std::fs::read_to_string(directory.join("targets.txt"))?;
    let prefix = previous.trim_end_matches('\n');
    let separator = if prefix.is_empty() { "" } else { " " };
    super::write(
        directory,
        "targets.txt",
        &format!("{prefix}{separator}cpp17-fitted\n"),
    )?;
    Ok(emitted)
}

/// Standalone inference CSV runner shared with `jope emit --lang cpp17-fitted`.
/// Compile with `-DJOPE_STANDALONE`; inputs have a header and finite numeric cells.
pub const STANDALONE: &str = r#"// Build: g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -DJOPE_STANDALONE standalone.cpp -o model
// Run: ./model test_features.csv predictions.csv (numeric CSV with a header, no missing cells).
#ifdef JOPE_STANDALONE
#include "model.cpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream input(argv[1]);
  if (!input) return 3;
  std::error_code alias_error;
  if (std::filesystem::equivalent(argv[1], argv[2], alias_error)) return 10;
  std::string line;
  if (!std::getline(input, line)) return 4;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  const std::size_t columns = jope_fitted_columns();
  std::size_t header_columns = 1;
  for (char c : line) if (c == ',') ++header_columns;
  if (header_columns != columns) return 5;
  V values;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) return 7;
    if (line.back() == ',') return 7;
    std::stringstream record(line);
    std::string cell;
    std::size_t count = 0;
    while (std::getline(record, cell, ',')) {
      std::size_t used = 0;
      double value;
      try { value = std::stod(cell, &used); } catch (...) { return 6; }
      if (used != cell.size() || !std::isfinite(value)) return 6;
      values.push_back(value);
      ++count;
    }
    if (count != columns) return 7;
  }
  if (input.bad()) return 4;
  const std::size_t rows = values.size() / columns;
  std::vector<std::uint8_t> missing(values.size(), 0);
  V predictions(rows);
  if (jope_fitted_predict(values.data(), values.size(), missing.data(),
                          missing.size(), rows, columns, predictions.data(),
                          predictions.size()) != 0) return 8;
  std::ofstream output(argv[2]);
  if (!output) return 3;
  output << "prediction\n" << std::setprecision(17);
  for (double value : predictions) output << value << '\n';
  return output ? 0 : 9;
}
#endif
"#;
