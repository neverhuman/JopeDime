//! Strict streaming reader for private Rust research records (spec §13.2).
//!
//! Record identity pins are producer attestations; parsing does not qualify a model or corpus
//! split. Synthetic controls require explicit opt-in and must never become training data.

mod json;
mod validation;

#[cfg(test)]
#[path = "record_reader/tests.rs"]
mod tests;

pub use validation::validate_record;

use anyhow::{bail, ensure, Context, Result};
use json::{parse_unique_json, read_bounded_line};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

pub const VERSION: &str = "jope.research.round/v1";
pub const DOPE_SHA256: &str = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814";
pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;

/// Opt-ins accepted by [`RecordReader`] and [`record_inventory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReaderOptions {
    pub allow_synthetic: bool,
    pub allow_incomplete: bool,
    pub require_evaluate_all: bool,
}

/// Incrementally validates one JSON Lines stream and its run/input binding.
pub struct RecordReader<R> {
    reader: R,
    options: ReaderOptions,
    binding: Option<String>,
    count: u64,
    finished: bool,
}

impl<R: BufRead> RecordReader<R> {
    pub fn new(reader: R, options: ReaderOptions) -> Self {
        Self {
            reader,
            options,
            binding: None,
            count: 0,
            finished: false,
        }
    }

    fn next_validated_line(&mut self) -> Result<Option<(Value, Vec<u8>)>> {
        if self.finished {
            return Ok(None);
        }
        let Some(line) = read_bounded_line(&mut self.reader, MAX_LINE_BYTES)? else {
            self.finished = true;
            if self.count == 0 {
                bail!("empty research stream");
            }
            return Ok(None);
        };
        ensure!(
            line.len() <= MAX_LINE_BYTES && line.ends_with(b"\n"),
            "oversized or partial record line"
        );
        let record = parse_unique_json(&line).context("parse research record JSON")?;
        validate_record(
            &record,
            self.options.allow_synthetic,
            self.options.allow_incomplete,
        )?;
        if self.options.require_evaluate_all {
            ensure!(
                record["evaluate_all_proposals"].as_bool() == Some(true)
                    && record["capture_complete"].as_bool() == Some(true),
                "complete all-proposal capture required"
            );
        }
        ensure!(
            record["round"].as_u64() == Some(self.count),
            "duplicate, missing or out-of-order round"
        );
        let binding = serde_json::to_string(&json!([
            record["identity"],
            record["seed"],
            record["tokens"],
            record["mask"],
            record["evaluate_all_proposals"]
        ]))?;
        if let Some(expected) = &self.binding {
            ensure!(
                &binding == expected,
                "run/input identity, conditioning, seed or capture mode changed"
            );
        } else {
            self.binding = Some(binding);
        }
        self.count = self
            .count
            .checked_add(1)
            .context("research stream round count overflow")?;
        Ok(Some((record, line)))
    }
}

impl<R: BufRead> Iterator for RecordReader<R> {
    type Item = Result<Value>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_validated_line() {
            Ok(Some((record, _line))) => Some(Ok(record)),
            Ok(None) => None,
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

/// Open a strict, lazy reader. Each yielded record is validated before it is exposed.
pub fn read_records(
    path: impl AsRef<Path>,
    options: ReaderOptions,
) -> Result<RecordReader<BufReader<File>>> {
    Ok(RecordReader::new(
        BufReader::new(
            File::open(path.as_ref()).with_context(|| {
                format!("open research record stream {}", path.as_ref().display())
            })?,
        ),
        options,
    ))
}

/// Count and hash the same validated source bytes; this does not qualify provenance.
pub fn record_inventory(path: impl AsRef<Path>, options: ReaderOptions) -> Result<Value> {
    let mut reader = read_records(path, options)?;
    let mut digest = Sha256::new();
    let mut rounds = 0u64;
    let mut proposals = 0u64;
    let mut admissions = 0u64;
    let mut dispatches = 0u64;
    let mut outcomes = 0u64;
    let mut successes = 0u64;
    let mut failures = 0u64;
    let mut incomplete_rounds = 0u64;
    let mut identity = None;
    let mut evaluate_all_proposals = None;

    while let Some((record, line)) = reader.next_validated_line()? {
        digest.update(&line);
        rounds += 1;
        proposals += record["proposals"]
            .as_array()
            .expect("validated proposals")
            .len() as u64;
        admissions += record["policy"]["admitted"]
            .as_array()
            .expect("validated admissions")
            .len() as u64;
        dispatches += record["dispatched"]
            .as_array()
            .expect("validated dispatches")
            .len() as u64;
        let round_outcomes = record["outcomes"].as_array().expect("validated outcomes");
        outcomes += round_outcomes.len() as u64;
        for outcome in round_outcomes {
            if outcome["measurement"]["status"] == "success" {
                successes += 1;
            } else {
                failures += 1;
            }
        }
        if record["capture_complete"] == false {
            incomplete_rounds += 1;
        }
        if identity.is_none() {
            identity = Some(record["identity"].clone());
            evaluate_all_proposals = record["evaluate_all_proposals"].as_bool();
        }
    }

    let mut counts = Map::new();
    counts.insert("rounds".into(), json!(rounds));
    counts.insert("proposals".into(), json!(proposals));
    counts.insert("admissions".into(), json!(admissions));
    counts.insert("dispatches".into(), json!(dispatches));
    counts.insert("outcomes".into(), json!(outcomes));
    counts.insert("successes".into(), json!(successes));
    counts.insert("failures".into(), json!(failures));
    counts.insert("incomplete_rounds".into(), json!(incomplete_rounds));
    Ok(json!({
        "schema": "jope.research.inventory/v1",
        "identity": identity.context("empty research stream")?,
        "evaluate_all_proposals": evaluate_all_proposals.context("empty research stream")?,
        "records_sha256": format!("{:x}", digest.finalize()),
        "counts": counts,
    }))
}

/// Require exact stream identity, dispatch mode, bytes and every observed count.
pub fn verify_inventory(actual: &Value, expected: &Value) -> Result<()> {
    validation::require_fields(
        expected,
        "schema identity evaluate_all_proposals records_sha256 counts",
    )?;
    validation::require_fields(
        &expected["counts"],
        "rounds proposals admissions dispatches outcomes successes failures incomplete_rounds",
    )?;
    let actual_json = serde_json::to_string(actual)?;
    let expected_json = serde_json::to_string(expected)?;
    ensure!(
        actual_json == expected_json,
        "research stream differs from the expected committed inventory"
    );
    Ok(())
}

/// Read one expected inventory document using the same byte limit and duplicate-key rule.
pub fn read_expected_inventory(path: impl AsRef<Path>) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path.as_ref())
        .with_context(|| format!("open expected inventory {}", path.as_ref().display()))?
        .take((MAX_LINE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_LINE_BYTES,
        "oversized expected inventory"
    );
    parse_unique_json(&bytes).context("parse expected research inventory JSON")
}

/// Verify an inventory file against the validated stream, mirroring `--expected-inventory`.
pub fn verify_inventory_file(actual: &Value, path: impl AsRef<Path>) -> Result<()> {
    let expected = read_expected_inventory(path)?;
    verify_inventory(actual, &expected)
}

/// Select the concise or detailed count object printed by the Python command-line reader.
pub fn summary_counts(inventory: &Value, detailed: bool) -> Result<Value> {
    let counts = &inventory["counts"];
    validation::require_fields(
        counts,
        "rounds proposals admissions dispatches outcomes successes failures incomplete_rounds",
    )?;
    if detailed {
        return Ok(counts.clone());
    }
    let mut summary = Map::new();
    for key in [
        "rounds",
        "proposals",
        "outcomes",
        "failures",
        "incomplete_rounds",
    ] {
        summary.insert(key.to_owned(), counts[key].clone());
    }
    Ok(Value::Object(summary))
}
