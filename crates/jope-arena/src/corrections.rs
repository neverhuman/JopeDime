//! Protocol corrections: evidence-backed overrides of `benchmark_valid` on published score rows.
//!
//! A published `scores.csv` is the benchmark's record of what it published and is never edited
//! here (the regression fixture is a byte-exact copy of the clone). When this repository's own
//! campaign code contradicts a published flag — the source run of a row wrote
//! `benchmark_valid: false, test_adaptive: true` for its artefacts while the acceptance manifest
//! relabelled the replayed winner `true` — the contradiction is recorded in a
//! `jope.arena-protocol-corrections/v1` file kept *beside* the fixture, and the benchmark-valid
//! ladder applies it before ranking.
//!
//! Rules that keep a correction honest:
//!
//! * it can only demote (`benchmark_valid: false`); nothing here promotes a row;
//! * it is bound to the SHA-256 of the `scores.csv` it was written against, so it cannot be
//!   applied to a re-published file whose rows it was never checked against;
//! * it must cite evidence and it must match at least one row, so a correction that no longer
//!   applies fails loudly instead of silently doing nothing.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::io::{read_bytes, sha256_hex};
use crate::scores::ScoreRow;

/// Schema id of a corrections file.
pub const SCHEMA: &str = "jope.arena-protocol-corrections/v1";

/// A corrections file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Corrections {
    /// [`SCHEMA`].
    pub schema: String,
    /// Track the corrections belong to.
    pub track_id: String,
    /// SHA-256 of the `scores.csv` bytes the corrections were written against.
    pub scores_sha256: String,
    /// The corrections, applied in order.
    pub corrections: Vec<Correction>,
}

/// One correction: every rankable row it selects has its `benchmark_valid` set to `False`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Correction {
    /// Short unique id, quoted in snapshots and ladders.
    pub id: String,
    /// Rows of this method.
    pub method_id: String,
    /// Only rows with this `source_kind`, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kind: Option<String>,
    /// Only rows with this `source_revision`, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    /// Only rows of these datasets, when non-empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dataset_ids: Vec<String>,
    /// The corrected flag. Must be `false`: a correction never promotes a row.
    pub benchmark_valid: bool,
    /// How the row's value was selected (`test-adaptive`).
    pub selection: String,
    /// Why the published flag is wrong, in one paragraph.
    pub reason: String,
    /// Repository paths or `path:line` references a reviewer can open.
    pub evidence: Vec<String>,
}

/// What applying one correction did.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AppliedCorrection {
    /// [`Correction::id`].
    pub id: String,
    /// Rankable rows the selector matched.
    pub rows: usize,
}

/// Provenance of an applied corrections file, recorded in ladders and snapshots.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CorrectionsMeta {
    /// File name (not the path: snapshots are committed and must not carry host paths).
    pub file: String,
    /// SHA-256 of the file bytes.
    pub sha256: String,
    /// Per-correction match counts, in file order.
    pub applied: Vec<AppliedCorrection>,
    /// Rows whose flag was changed from `True` to `False`.
    pub rows_corrected: usize,
}

impl Correction {
    fn selects(&self, row: &ScoreRow) -> bool {
        row.method_id == self.method_id
            && self
                .source_kind
                .as_deref()
                .is_none_or(|kind| row.source_kind == kind)
            && self
                .source_revision
                .as_deref()
                .is_none_or(|revision| row.source_revision == revision)
            && (self.dataset_ids.is_empty() || self.dataset_ids.contains(&row.dataset_id))
    }
}

impl Corrections {
    /// Parse and validate `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let corrections: Self = serde_json::from_slice(bytes).context("corrections JSON")?;
        ensure!(
            corrections.schema == SCHEMA,
            "corrections schema {} is not {SCHEMA}",
            corrections.schema
        );
        ensure!(
            corrections.scores_sha256.len() == 64
                && corrections
                    .scores_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "corrections scores_sha256 {:?} is not a SHA-256",
            corrections.scores_sha256
        );
        ensure!(
            !corrections.corrections.is_empty(),
            "corrections file lists no corrections"
        );
        let mut ids = std::collections::BTreeSet::new();
        for correction in &corrections.corrections {
            ensure!(!correction.id.is_empty(), "correction without id");
            ensure!(
                ids.insert(correction.id.as_str()),
                "duplicate correction id {}",
                correction.id
            );
            ensure!(
                !correction.benchmark_valid,
                "correction {} would promote rows to benchmark_valid; corrections only demote",
                correction.id
            );
            ensure!(
                !correction.method_id.is_empty(),
                "correction {} names no method",
                correction.id
            );
            ensure!(
                !correction.selection.is_empty() && !correction.reason.is_empty(),
                "correction {} needs a selection and a reason",
                correction.id
            );
            ensure!(
                !correction.evidence.is_empty(),
                "correction {} cites no evidence",
                correction.id
            );
        }
        Ok(corrections)
    }

    /// Read `path` (plain, `.gz` or `.zst`); returns the corrections and the file's SHA-256.
    pub fn load(path: &Path) -> Result<(Self, String)> {
        let bytes = read_bytes(path)?;
        let corrections =
            Self::parse(&bytes).with_context(|| format!("corrections {}", path.display()))?;
        Ok((corrections, sha256_hex(&bytes)))
    }

    /// Apply to `rows` of `track_id` read from a `scores.csv` whose bytes hash to
    /// `scores_sha256`. Every correction must select at least one rankable row.
    pub fn apply(
        &self,
        track_id: &str,
        scores_sha256: &str,
        rows: &mut [ScoreRow],
    ) -> Result<(Vec<AppliedCorrection>, usize)> {
        ensure!(
            self.track_id == track_id,
            "corrections are for track {} but the ladder is {track_id}",
            self.track_id
        );
        ensure!(
            self.scores_sha256 == scores_sha256,
            "corrections were written against scores.csv {} but this scores.csv hashes to {scores_sha256}; re-verify every correction before rebinding",
            self.scores_sha256
        );
        let mut applied = Vec::with_capacity(self.corrections.len());
        let mut corrected = 0;
        for correction in &self.corrections {
            let mut matched = 0;
            for row in rows.iter_mut() {
                if !row.is_rankable() || !correction.selects(row) {
                    continue;
                }
                matched += 1;
                if row.benchmark_valid == "True" {
                    row.benchmark_valid = "False".to_owned();
                    corrected += 1;
                }
            }
            ensure!(
                matched > 0,
                "correction {} selects no rankable row; delete it or fix its selector",
                correction.id
            );
            applied.push(AppliedCorrection {
                id: correction.id.clone(),
                rows: matched,
            });
        }
        Ok((applied, corrected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scores::parse_scores;

    const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn file(corrections: &str) -> String {
        format!(
            r#"{{"schema":"{SCHEMA}","track_id":"t","scores_sha256":"{SHA}","corrections":[{corrections}]}}"#
        )
    }

    fn correction(id: &str, extra: &str) -> String {
        format!(
            r#"{{"id":"{id}","method_id":"m",{extra}"benchmark_valid":false,"selection":"test-adaptive","reason":"r","evidence":["e"]}}"#
        )
    }

    fn rows() -> Vec<ScoreRow> {
        let mut text = crate::scores::COLUMNS.join(",");
        text.push('\n');
        text += "t,d1,m,aspfm-generated,rmse,1.0,ok,True,ledger,rev-a,,,\n";
        text += "t,d2,m,aspfm-generated,rmse,1.0,ok,True,manifest,rev-b,,,\n";
        text += "t,d3,m,aspfm-generated,rmse,,missing,True,manifest,rev-b,,,\n";
        text += "t,d1,o,public-development,rmse,1.0,ok,True,report,rev-b,,,\n";
        parse_scores(&text).unwrap()
    }

    #[test]
    fn selectors_demote_only_the_rows_they_name() {
        let text = file(&correction("golden", r#""source_revision":"rev-b","#));
        let corrections = Corrections::parse(text.as_bytes()).unwrap();
        let mut rows = rows();
        let (applied, corrected) = corrections.apply("t", SHA, &mut rows).unwrap();
        assert_eq!(
            applied,
            [AppliedCorrection {
                id: "golden".into(),
                rows: 1
            }]
        );
        assert_eq!(corrected, 1);
        let flags: Vec<&str> = rows.iter().map(|r| r.benchmark_valid.as_str()).collect();
        assert_eq!(
            flags,
            ["True", "False", "True", "True"],
            "only m/rev-b/d2 is rankable and selected; the missing row and method o are untouched"
        );

        // Applying twice matches the same row again but corrects nothing more.
        let (applied, corrected) = corrections.apply("t", SHA, &mut rows).unwrap();
        assert_eq!(applied[0].rows, 1);
        assert_eq!(corrected, 0);
    }

    #[test]
    fn a_correction_that_selects_nothing_is_an_error() {
        let text = file(&correction("stale", r#""dataset_ids":["nope"],"#));
        let corrections = Corrections::parse(text.as_bytes()).unwrap();
        let error = corrections.apply("t", SHA, &mut rows()).unwrap_err();
        assert!(
            error.to_string().contains("selects no rankable row"),
            "{error}"
        );
    }

    #[test]
    fn bindings_and_promotions_are_refused() {
        let text = file(&correction("c", ""));
        let corrections = Corrections::parse(text.as_bytes()).unwrap();
        let other = "1".repeat(64);
        let error = corrections.apply("t", &other, &mut rows()).unwrap_err();
        assert!(error.to_string().contains("written against"), "{error}");
        let error = corrections.apply("u", SHA, &mut rows()).unwrap_err();
        assert!(error.to_string().contains("for track t"), "{error}");

        let promote = file(
            &correction("c", "").replace("\"benchmark_valid\":false", "\"benchmark_valid\":true"),
        );
        let error = Corrections::parse(promote.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("only demote"), "{error}");

        let no_evidence = file(&correction("c", "").replace("[\"e\"]", "[]"));
        assert!(Corrections::parse(no_evidence.as_bytes()).is_err());
        let duplicate = file(&format!("{},{}", correction("c", ""), correction("c", "")));
        assert!(Corrections::parse(duplicate.as_bytes()).is_err());
        let unknown = file(&correction("c", r#""extra":1,"#));
        assert!(Corrections::parse(unknown.as_bytes()).is_err());
    }
}
