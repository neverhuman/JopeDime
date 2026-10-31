//! The shareable summary. The allowlist is the type: it holds only integers, booleans and two
//! unit constants, so no identifier, name, digest, operation or program text can be written.
//!
//! This summary is local to this tool. The product plan's allowlisted shareable-export step
//! may absorb or replace it, so that two canary systems do not diverge.

use crate::fragment::{expansion_stages, stage_texts};
use crate::table::{Lifecycle, MotifTable, TABLE_SCHEMA};
use crate::validate::{MotifResult, Validation, VALIDATION_SCHEMA};
use crate::{Code, NAME_PREFIX};
use anyhow::{ensure, Result};
use feat_math::transpile::lint_shipped_text;
use serde::Serialize;

/// Words the owner bans from anything shipped, matched case-insensitively.
pub const OWNER_BANNED: [&str; 4] = ["search", "tree", "genome", "evolution"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
/// Version tag for the allowlisted shareable fragment summary.
pub enum PublicSchema {
    #[serde(rename = "jope.fragment-summary/v1")]
    /// The first summary schema version.
    V1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
/// Promotion status that can appear in a shareable summary.
pub enum PromotionState {
    #[serde(rename = "required_not_run")]
    /// Promotion requirements have not been run.
    RequiredNotRun,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
/// Counts of checks that passed and failed.
pub struct Tally {
    /// Number of passing results.
    pub passed: u32,
    /// Number of failing results.
    pub failed: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
/// Fragment counts grouped by expanded stage length.
pub struct StageCounts {
    /// Fragments with two expanded stages.
    pub s2: u32,
    /// Fragments with three expanded stages.
    pub s3: u32,
    /// Fragments with four expanded stages.
    pub s4: u32,
    /// Fragments with five expanded stages.
    pub s5: u32,
    /// Fragments with six expanded stages.
    pub s6: u32,
}

/// One pass/fail tally per check class; nothing per host or per control.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PublicChecks {
    /// Results of exact rewrite and re-expansion checks.
    pub round_trip: Tally,
    /// Results of repeated deterministic-engine checks.
    pub determinism: Tally,
    /// Results of fitting in independent compatible contexts.
    pub context_compatibility: Tally,
    /// Results of scanning artifacts for reserved reference names.
    pub name_absence: Tally,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
/// Allowlisted shareable counts and check outcomes; contains no private identities or text.
pub struct PublicExport {
    /// Summary schema version.
    pub schema: PublicSchema,
    /// Total fragments represented by the table.
    pub fragments: u32,
    /// Fragments still in quarantine.
    pub quarantined: u32,
    /// Always zero: no code path promotes an entry.
    pub promoted: u32,
    /// Counts by expanded stage length.
    pub stage_counts: StageCounts,
    /// Pass/fail totals for each public check class.
    pub checks: PublicChecks,
    /// Whether validation verified the source identity against current source records.
    pub source_verified: bool,
    /// Whether promotion requirements were run.
    pub promotion: PromotionState,
}

fn count(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn tally(results: &[MotifResult], check_passed: impl Fn(&MotifResult) -> bool) -> Tally {
    let passed = count(results.iter().filter(|result| check_passed(result)).count());
    Tally {
        passed,
        failed: count(results.len()).saturating_sub(passed),
    }
}

impl PublicChecks {
    /// Tallies over every fragment result.
    pub fn from_results(results: &[MotifResult]) -> Self {
        Self {
            round_trip: tally(results, |r| r.round_trip.passed),
            determinism: tally(results, |r| r.determinism.passed),
            context_compatibility: tally(results, |r| r.context_compatibility.passed),
            name_absence: tally(results, |r| r.name_absence.passed),
        }
    }
}

impl StageCounts {
    /// Fragments of a table per expanded stage count.
    pub fn of(table: &MotifTable) -> Self {
        let mut counts = Self::default();
        for motif in &table.body.motifs {
            match motif.signature.stages {
                2 => counts.s2 += 1,
                3 => counts.s3 += 1,
                4 => counts.s4 += 1,
                5 => counts.s5 += 1,
                6 => counts.s6 += 1,
                _ => {}
            }
        }
        counts
    }
}

impl PublicExport {
    /// The summary of one validation of one table.
    pub fn from_validation(validation: &Validation, table: &MotifTable) -> Self {
        let motifs = &table.body.motifs;
        let stage_counts = StageCounts::of(table);
        let quarantined = motifs
            .iter()
            .filter(|m| m.lifecycle == Lifecycle::Quarantined)
            .count();
        let promoted = motifs
            .iter()
            .filter(|m| m.lifecycle == Lifecycle::Promoted)
            .count();
        Self {
            schema: PublicSchema::V1,
            fragments: count(motifs.len()),
            quarantined: count(quarantined),
            promoted: count(promoted),
            stage_counts,
            checks: PublicChecks::from_results(&validation.motifs),
            source_verified: validation.source_verified,
            promotion: PromotionState::RequiredNotRun,
        }
    }
}

/// Every private value of a table that must never reach shared text.
fn private_values(table: &MotifTable) -> Vec<String> {
    let mut values = vec![
        NAME_PREFIX.to_owned(),
        TABLE_SCHEMA.to_owned(),
        VALIDATION_SCHEMA.to_owned(),
        table.body.source.schema.clone(),
    ];
    let with_prefix = |values: &mut Vec<String>, digest: &str| {
        values.push(digest.to_owned());
        if let Some(prefix) = digest.get(..16) {
            values.push(prefix.to_owned());
        }
    };
    with_prefix(&mut values, &table.table_sha256);
    with_prefix(&mut values, &table.body.source.sha256);
    if let Some(map) = &table.body.parameters.family_map_sha256 {
        with_prefix(&mut values, map);
    }
    for motif in &table.body.motifs {
        values.push(motif.name.clone());
        values.push(motif.id.clone());
        for length in [12, 16] {
            if let Some(prefix) = motif.id.get(..length) {
                values.push(prefix.to_owned());
            }
        }
        values.push(motif.expansion.clone());
        if let Ok(stages) = expansion_stages(&motif.expansion) {
            values.extend(stage_texts(&stages));
        }
        values.push(motif.signature.ops.join(" "));
        values.push(motif.signature.ops.join(","));
        for occurrence in &motif.provenance {
            for digest in [
                &occurrence.dataset_sha256,
                &occurrence.cut_sha256,
                &occurrence.comparison_sha256,
                &occurrence.record_sha256,
                &occurrence.recipe_sha256,
            ] {
                with_prefix(&mut values, digest);
            }
            if let Some(burn) = &occurrence.label.burn_sha256 {
                with_prefix(&mut values, burn);
            }
            if let Some(family) = &occurrence.family {
                values.push(family.clone());
            }
        }
    }
    for (recipe, host) in &table.body.hosts {
        with_prefix(&mut values, recipe);
        values.push(host.clone());
    }
    values.retain(|value| !value.is_empty());
    values
}

/// Defense in depth before a public write: the text holds no private value of the table,
/// passes the shipped-text lint and holds no owner-banned word.
///
/// Matching is by case-insensitive substring. A private value that happens to equal a word
/// of the summary itself (a family label such as `passed`, `checks` or `schema`, or a source
/// label contained in the summary's schema label) therefore makes this scan refuse every
/// summary of that table. That fails closed, by design.
pub fn scan(text: &str, table: &MotifTable) -> Result<()> {
    let lower = text.to_lowercase();
    let leaked = private_values(table)
        .iter()
        .any(|value| lower.contains(&value.to_lowercase()));
    ensure!(!leaked, Code("public_export_leak"));
    ensure!(
        lint_shipped_text(text).is_empty(),
        Code("public_export_leak")
    );
    ensure!(
        !OWNER_BANNED.iter().any(|word| lower.contains(*word)),
        Code("public_export_leak")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;
    use serde_json::Value;
    use std::collections::BTreeSet;

    /// The summary of the labelled fixture table, so the family canaries are planted.
    fn export_text() -> (String, MotifTable) {
        let table = fixture::labelled_table();
        let validation = fixture::passing_validation(&table);
        let export = PublicExport::from_validation(&validation, &table);
        (serde_json::to_string(&export).expect("json"), table)
    }

    fn keys(value: &Value, prefix: &str, out: &mut BTreeSet<String>) {
        if let Value::Object(map) = value {
            for (key, child) in map {
                let path = format!("{prefix}{key}");
                keys(child, &format!("{path}."), out);
                out.insert(path);
            }
        }
    }

    fn strings(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(text) => out.push(text.clone()),
            Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
            Value::Object(map) => map.values().for_each(|item| strings(item, out)),
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    #[test]
    fn public_export_is_type_allowlisted() {
        let (text, table) = export_text();
        let value: Value = serde_json::from_str(&text).expect("json");
        let mut found = BTreeSet::new();
        keys(&value, "", &mut found);
        let expected: BTreeSet<String> = [
            "schema",
            "fragments",
            "quarantined",
            "promoted",
            "stage_counts",
            "stage_counts.s2",
            "stage_counts.s3",
            "stage_counts.s4",
            "stage_counts.s5",
            "stage_counts.s6",
            "checks",
            "checks.round_trip",
            "checks.round_trip.passed",
            "checks.round_trip.failed",
            "checks.determinism",
            "checks.determinism.passed",
            "checks.determinism.failed",
            "checks.context_compatibility",
            "checks.context_compatibility.passed",
            "checks.context_compatibility.failed",
            "checks.name_absence",
            "checks.name_absence.passed",
            "checks.name_absence.failed",
            "source_verified",
            "promotion",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(found, expected);
        let mut values = Vec::new();
        strings(&value, &mut values);
        values.sort();
        assert_eq!(values, ["jope.fragment-summary/v1", "required_not_run"]);
        assert_eq!(value["fragments"], table.body.motifs.len());
        assert_eq!(value["promoted"], 0);
        assert_eq!(value["source_verified"], Value::Bool(false));
    }

    #[test]
    fn public_export_contains_no_planted_canary() {
        let (text, table) = export_text();
        // Positive control: every canary this test looks for is really in the private table.
        fixture::assert_carried(&table);
        let lower = text.to_lowercase();
        for sentinel in fixture::sentinels(&table) {
            assert!(!lower.contains(&sentinel.to_lowercase()), "{sentinel}");
        }
        assert!(!lower.contains(&table.table_sha256[..16]));
        assert!(scan(&text, &table).is_ok());
    }

    #[test]
    fn scan_refuses_any_private_value() {
        let (text, table) = export_text();
        let motif = &table.body.motifs[0];
        let occurrence = &motif.provenance[0];
        let host = table.body.hosts.values().next().expect("a host").clone();
        let family_map = table
            .body
            .parameters
            .family_map_sha256
            .clone()
            .expect("a labelled table");
        let injections = [
            motif.name.clone(),
            motif.id[..12].to_owned(),
            motif.id[..16].to_owned(),
            String::from(fixture::FAMILY_A),
            occurrence.dataset_sha256[..16].to_owned(),
            host,
            motif.expansion.clone(),
            motif.signature.ops.join(","),
            String::from("motif_"),
            table.table_sha256[..16].to_owned(),
            table.body.source.sha256[..16].to_owned(),
            table.body.source.schema.clone(),
            family_map[..16].to_owned(),
        ];
        for injection in injections {
            let leaked = format!("{text} {injection}");
            let error = scan(&leaked, &table).expect_err("refused");
            assert_eq!(code_of(&error), Some("public_export_leak"), "{injection}");
        }
        // The owner ban is case-insensitive.
        for word in OWNER_BANNED {
            let shouted = word.to_uppercase();
            let error = scan(&format!("{text} {shouted}"), &table).expect_err("refused");
            assert_eq!(code_of(&error), Some("public_export_leak"), "{word}");
        }
        // A word only the shipped-text lint bans is refused too.
        assert!(!lint_shipped_text("oracle").is_empty());
        let error = scan(&format!("{text} oracle"), &table).expect_err("refused");
        assert_eq!(code_of(&error), Some("public_export_leak"));
    }

    #[test]
    fn public_export_passes_shipped_text_lint_and_owner_ban() {
        let (text, table) = export_text();
        assert!(lint_shipped_text(&text).is_empty(), "{text}");
        let lower = text.to_lowercase();
        for word in OWNER_BANNED {
            assert!(!lower.contains(word), "{word}");
        }
        let validation = fixture::passing_validation(&table);
        let summary = PublicExport::from_validation(&validation, &table);
        assert_eq!(
            summary.stage_counts.s2 + summary.stage_counts.s3,
            summary.fragments
        );
        assert_eq!(summary.quarantined, summary.fragments);
        assert_eq!(summary.checks.round_trip.passed, summary.fragments);
        assert_eq!(summary.checks.round_trip.failed, 0);
    }
}
