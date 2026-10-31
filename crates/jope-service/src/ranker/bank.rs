//! Champion-bank loading and nearest-landmark retrieval.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

const LANDMARK_DIM: usize = 12;
const FEAT412_DIM: usize = 412;
const PROJ16_DIM: usize = 16;

/// One row from the fused ranker's compressed champion bank.
#[derive(Clone, Debug, PartialEq)]
pub struct Champion {
    pub program_sha256: String,
    pub canonical_dsl: String,
    pub feat412: Vec<f32>,
    pub proj16: Vec<f32>,
    pub landmark_summary: Vec<f32>,
    pub family_hash: String,
}

/// Loaded champion rows, in their source order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bank {
    champions: Vec<Champion>,
}

impl Bank {
    /// Load a zstd-compressed JSONL bank from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).context("open champion bank")?;
        Self::from_zstd_reader(file)
    }

    /// Decode and load a compressed JSONL bank from a reader.
    pub fn from_zstd_reader<R: Read>(reader: R) -> Result<Self> {
        let decoder = ruzstd::decoding::StreamingDecoder::new(reader)
            .context("decode champion bank zstd frame")?;
        let mut champions = Vec::new();
        for (index, line) in BufReader::new(decoder).lines().enumerate() {
            let line_number = index + 1;
            let line = line.with_context(|| format!("read champion bank line {line_number}"))?;
            let value: Value = serde_json::from_str(&line)
                .with_context(|| format!("parse champion bank line {line_number}"))?;
            champions.push(
                parse_champion(value)
                    .with_context(|| format!("invalid champion bank line {line_number}"))?,
            );
        }
        Ok(Self { champions })
    }

    pub fn champions(&self) -> &[Champion] {
        &self.champions
    }

    /// Return up to `k` champions closest to the first 12 lm48 coordinates.
    /// Distance is squared Euclidean distance in f64; ties use program SHA-256.
    pub fn nearest(&self, lm48: &[f32], k: usize) -> Result<Vec<&Champion>> {
        ensure!(
            lm48.len() >= LANDMARK_DIM,
            "lm48 must contain at least {LANDMARK_DIM} values"
        );
        ensure!(
            lm48[..LANDMARK_DIM].iter().all(|value| value.is_finite()),
            "lm48 contains a non-finite value"
        );
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut ranked = self
            .champions
            .iter()
            .map(|champion| {
                let distance = lm48[..LANDMARK_DIM]
                    .iter()
                    .zip(&champion.landmark_summary)
                    .map(|(left, right)| {
                        let delta = f64::from(*left) - f64::from(*right);
                        delta * delta
                    })
                    .sum::<f64>();
                (distance, champion.program_sha256.as_str(), champion)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.0.total_cmp(&right.0).then_with(|| left.1.cmp(right.1)));
        Ok(ranked
            .into_iter()
            .take(k)
            .map(|(_, _, champion)| champion)
            .collect())
    }
}

fn parse_champion(value: Value) -> Result<Champion> {
    let object = value.as_object().context("expected JSON object")?;
    let string = |key: &str| -> Result<String> {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .with_context(|| format!("missing or invalid {key}"))
    };
    let numbers = |key: &str, expected: usize| -> Result<Vec<f32>> {
        let values = object
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("missing or invalid {key}"))?;
        ensure!(
            values.len() == expected,
            "{key} must contain {expected} values"
        );
        values
            .iter()
            .map(|value| {
                let number = value.as_f64().context("expected number")? as f32;
                ensure!(number.is_finite(), "{key} contains a non-finite value");
                Ok(number)
            })
            .collect()
    };

    let program_sha256 = string("program_sha256")?;
    ensure!(
        program_sha256.len() == 64 && program_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "program_sha256 must be 64 hexadecimal characters"
    );
    let canonical_dsl = string("canonical_dsl")?;
    ensure!(!canonical_dsl.is_empty(), "canonical_dsl must not be empty");
    let feat412 = numbers("feat412", FEAT412_DIM)?;
    let proj16 = numbers("proj16", PROJ16_DIM)?;
    let landmark_summary = numbers("landmark_summary", LANDMARK_DIM)?;
    let family_hash = string("family_hash")?;
    ensure!(!family_hash.is_empty(), "family_hash must not be empty");

    if object.contains_key("data") || object.contains_key("rows") {
        bail!("champion bank rows must not contain dataset rows");
    }

    Ok(Champion {
        program_sha256,
        canonical_dsl,
        feat412,
        proj16,
        landmark_summary,
        family_hash,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufRead, BufReader, Cursor};
    use std::path::{Path, PathBuf};

    use ruzstd::encoding::{compress_to_vec, CompressionLevel};
    use sha2::{Digest, Sha256};

    use super::*;

    fn row(hash: char, first_landmark: f32) -> String {
        let summary = std::iter::once(first_landmark)
            .chain(std::iter::repeat_n(0.0, LANDMARK_DIM - 1))
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"program_sha256":"{}","canonical_dsl":"x","feat412":[{}],"proj16":[{}],"landmark_summary":[{}],"family_hash":"family"}}"#,
            hash.to_string().repeat(64),
            std::iter::repeat_n("0", FEAT412_DIM)
                .collect::<Vec<_>>()
                .join(","),
            std::iter::repeat_n("0", PROJ16_DIM)
                .collect::<Vec<_>>()
                .join(","),
            summary
        )
    }

    fn compressed(text: &str) -> Vec<u8> {
        compress_to_vec(Cursor::new(text.as_bytes()), CompressionLevel::Uncompressed)
    }

    #[test]
    fn malformed_json_reports_one_based_line_number() {
        let data = format!("{}\n{{not json}}\n", row('a', 0.0));
        let error = Bank::from_zstd_reader(Cursor::new(compressed(&data))).unwrap_err();
        assert!(format!("{error:#}").contains("champion bank line 2"));
    }

    #[test]
    fn nearest_uses_first_twelve_coordinates_and_hash_tie_break() {
        let data = format!("{}\n{}\n{}\n", row('b', 1.0), row('a', -1.0), row('c', 3.0));
        let bank = Bank::from_zstd_reader(Cursor::new(compressed(&data))).unwrap();
        let mut lm48 = vec![0.0; 48];
        lm48[12..].fill(1000.0);
        let nearest = bank.nearest(&lm48, 2).unwrap();
        assert_eq!(
            nearest
                .iter()
                .map(|champion| champion.program_sha256.as_str())
                .collect::<Vec<_>>(),
            vec!["a".repeat(64).as_str(), "b".repeat(64).as_str()]
        );
    }

    /// Dormant until RES-07 publishes the pinned raw proposer bank and retrieval oracle.
    ///
    /// The external manifest contract is intentionally small and test-only:
    /// `{ "schema": "jope.rnk09-nearest-parity/v1", "bank": { "path": "...",
    /// "sha256": "..." }, "queries": [{ "lm48": [48 numbers], "k": N,
    /// "nearest_program_sha256": [ordered hashes] }] }`. The manifest itself is pinned
    /// separately by `JOPE_RNK09_PARITY_MANIFEST_SHA256`. No fixture or expected values are
    /// synthesized here; the source owner supplies both after the custody inputs exist.
    #[test]
    #[ignore = "requires custody-pinned raw proposer inputs and RES-07 nearest-k oracle"]
    fn nearest_matches_pinned_proposer_parity_manifest() {
        let manifest_path = required_env_path("JOPE_RNK09_PARITY_MANIFEST");
        let expected_manifest_sha = required_env("JOPE_RNK09_PARITY_MANIFEST_SHA256");
        let manifest_bytes = read_regular(&manifest_path);
        assert_eq!(sha256_hex(&manifest_bytes), expected_manifest_sha);

        let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest["schema"], "jope.rnk09-nearest-parity/v1");
        let bank_entry = &manifest["bank"];
        let bank_relative = bank_entry["path"]
            .as_str()
            .expect("bank.path must be a string");
        let bank_path = PathBuf::from(bank_relative);
        let bank_path = if bank_path.is_absolute() {
            bank_path
        } else {
            manifest_path
                .parent()
                .expect("manifest path must have a parent")
                .join(bank_path)
        };
        let expected_bank_sha = bank_entry["sha256"]
            .as_str()
            .expect("bank.sha256 must be a string");
        let bank_bytes = read_regular(&bank_path);
        assert_eq!(sha256_hex(&bank_bytes), expected_bank_sha);
        assert_no_model_prior_rows(&bank_bytes);

        let bank = Bank::from_zstd_reader(Cursor::new(&bank_bytes)).unwrap();
        let queries = manifest["queries"]
            .as_array()
            .expect("queries must be an array");
        assert!(!queries.is_empty(), "at least one parity query is required");
        for (index, query) in queries.iter().enumerate() {
            let coordinates = query["lm48"]
                .as_array()
                .expect("query lm48 must be an array");
            assert_eq!(coordinates.len(), 48, "query {index} must contain lm48");
            let lm48 = coordinates
                .iter()
                .map(|value| value.as_f64().expect("lm48 values must be numbers") as f32)
                .collect::<Vec<_>>();
            let k = query["k"].as_u64().expect("query k must be unsigned") as usize;
            let expected = query["nearest_program_sha256"]
                .as_array()
                .expect("nearest_program_sha256 must be an array")
                .iter()
                .map(|hash| hash.as_str().expect("expected hash must be a string"))
                .collect::<Vec<_>>();
            assert_eq!(expected.len(), k.min(bank.champions().len()));
            let actual = bank
                .nearest(&lm48, k)
                .unwrap()
                .iter()
                .map(|champion| champion.program_sha256.as_str())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "nearest-k mismatch for query {index}");
        }
    }

    fn required_env(name: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| panic!("{name} is required for this ignored test"))
    }

    fn required_env_path(name: &str) -> PathBuf {
        PathBuf::from(required_env(name))
    }

    fn read_regular(path: &Path) -> Vec<u8> {
        assert!(
            path.is_file() && !path.is_symlink(),
            "regular file required: {}",
            path.display()
        );
        fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn parity_harness_accepts_proposer_role() {
        let mut row: Value = serde_json::from_str(&row('a', 0.0)).unwrap();
        row["role"] = Value::String("proposer".to_owned());
        assert_no_model_prior_rows(&compressed(&serde_json::to_string(&row).unwrap()));
    }

    #[test]
    #[should_panic(expected = "line 1 is not tagged as the separate proposer asset")]
    fn parity_harness_rejects_model_prior_role() {
        let mut row: Value = serde_json::from_str(&row('a', 0.0)).unwrap();
        row["role"] = Value::String("model_prior".to_owned());
        row["coordinate_space"] = Value::String("standardized_lm48_l2_unit".to_owned());
        assert_no_model_prior_rows(&compressed(&serde_json::to_string(&row).unwrap()));
    }

    fn assert_no_model_prior_rows(bank_bytes: &[u8]) {
        let decoder = ruzstd::decoding::StreamingDecoder::new(Cursor::new(bank_bytes)).unwrap();
        for (index, line) in BufReader::new(decoder).lines().enumerate() {
            let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
            assert_eq!(
                value.get("role").and_then(Value::as_str),
                Some("proposer"),
                "line {} is not tagged as the separate proposer asset",
                index + 1
            );
            assert_ne!(
                value.get("role").and_then(Value::as_str),
                Some("model_prior"),
                "line {} is the RES-06 model-prior subset, not proposer input",
                index + 1
            );
            assert_ne!(
                value.get("coordinate_space").and_then(Value::as_str),
                Some("standardized_lm48_l2_unit"),
                "line {} uses standardized model-prior coordinates, not raw proposer lm48",
                index + 1
            );
        }
    }
}
