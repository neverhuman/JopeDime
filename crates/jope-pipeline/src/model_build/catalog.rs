//! Per-entry binding to the model-zoo catalog read in JA-01 v2.
//!
//! The committed table `data/model_zoo_entries.v1.json` is generated from the catalog at
//! `jain-model-zoo` `3f34f483…` (catalog sha256 `a8d63e94…`). Every catalog entry keeps its own
//! status fields. Nothing here is a runtime, fit, predict or reload result: `unknown` stays
//! `unknown`, and `runtime_admitted` is `false` for every entry, as the catalog's health document
//! states.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TABLE_SCHEMA: &str = "jope.model-zoo-entry-table/v1";
pub const SOURCE_HEAD: &str = "3f34f483c832823e0d506013e7865b294faa21a2";
pub const SOURCE_CATALOG_SHA256: &str =
    "a8d63e942ac78bb2a94dec4eed38d9a1c599f54d52559a252cf92ecf1e32f410";
pub const EXPECTED_ENTRIES: usize = 655;

const TABLE: &str = include_str!("../../data/model_zoo_entries.v1.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZooEntry {
    pub manifest: String,
    pub manifest_sha256: String,
    pub family: String,
    pub role: String,
    pub task: String,
    pub coverage: String,
    pub optimized: bool,
    pub asset_count: usize,
    pub depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZooEntryTable {
    pub schema: String,
    pub source_head: String,
    pub source_catalog_sha256: String,
    pub entries: Vec<ZooEntry>,
}

/// Whether the entry's implementation is present in source. Reference interfaces are not product
/// code, and no entry is claimed as product-admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceState {
    ReferenceInterfacePresent,
    Unknown,
}

/// Three-valued dimension: a tri-state never collapses an unknown into a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tri {
    Yes,
    No,
    Unknown,
}

/// The status of one catalog entry, with each dimension separate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EntryStatus {
    pub manifest: String,
    pub family: String,
    pub implemented_in_source: SourceState,
    pub runtime_admitted: Tri,
    pub fit_predict_reload_verified: Tri,
    pub export_supported: Tri,
    /// Availability in this build: `Unknown` until a verified probe establishes it.
    pub availability: &'static str,
}

fn status_of(entry: &ZooEntry) -> EntryStatus {
    let implemented_in_source = match entry.role.as_str() {
        "reference-port" | "nested-package" | "reference-workspace" => {
            SourceState::ReferenceInterfacePresent
        }
        _ => SourceState::Unknown,
    };
    EntryStatus {
        manifest: entry.manifest.clone(),
        family: entry.family.clone(),
        implemented_in_source,
        runtime_admitted: Tri::No,
        fit_predict_reload_verified: Tri::Unknown,
        export_supported: Tri::Unknown,
        availability: "unknown",
    }
}

/// Loads and validates the committed table against the source identity it names.
pub fn load() -> Result<ZooEntryTable> {
    let table: ZooEntryTable = serde_json::from_str(TABLE)?;
    ensure!(
        table.schema == TABLE_SCHEMA,
        "table schema {:?}",
        table.schema
    );
    ensure!(
        table.source_head == SOURCE_HEAD,
        "table source head differs"
    );
    ensure!(
        table.source_catalog_sha256 == SOURCE_CATALOG_SHA256,
        "table source catalog differs"
    );
    ensure!(
        table.entries.len() == EXPECTED_ENTRIES,
        "entry count {}",
        table.entries.len()
    );
    let manifests: BTreeSet<&str> = table.entries.iter().map(|e| e.manifest.as_str()).collect();
    ensure!(
        manifests.len() == table.entries.len(),
        "duplicate manifest path"
    );
    for entry in &table.entries {
        ensure!(
            entry.manifest_sha256.len() == 64
                && entry
                    .manifest_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "entry {} has a malformed manifest digest",
            entry.manifest
        );
        ensure!(
            entry.depth == entry.manifest.split('/').count(),
            "entry {} depth does not match its path",
            entry.manifest
        );
        if !matches!(entry.depth, 4 | 5) {
            bail!(
                "entry {} has unexpected depth {}",
                entry.manifest,
                entry.depth
            );
        }
    }
    Ok(table)
}

/// Per-entry statuses, in table order.
pub fn statuses() -> Result<Vec<EntryStatus>> {
    Ok(load()?.entries.iter().map(status_of).collect())
}

/// Counts by status. These are counts of entries, not of runnable or admitted algorithms.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub entries: usize,
    pub reference_interface_present: usize,
    pub runtime_admitted_yes: usize,
    pub fit_predict_reload_verified_unknown: usize,
}

pub fn summary() -> Result<Summary> {
    let statuses = statuses()?;
    Ok(Summary {
        entries: statuses.len(),
        reference_interface_present: statuses
            .iter()
            .filter(|s| s.implemented_in_source == SourceState::ReferenceInterfacePresent)
            .count(),
        runtime_admitted_yes: statuses
            .iter()
            .filter(|s| s.runtime_admitted == Tri::Yes)
            .count(),
        fit_predict_reload_verified_unknown: statuses
            .iter()
            .filter(|s| s.fit_predict_reload_verified == Tri::Unknown)
            .count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_table_is_bound_to_its_source_and_count() {
        let table = load().unwrap();
        assert_eq!(table.entries.len(), 655);
        assert_eq!(table.source_head, SOURCE_HEAD);
    }

    #[test]
    fn every_entry_is_runtime_unadmitted_and_its_verification_is_unknown() {
        for s in statuses().unwrap() {
            assert_eq!(s.runtime_admitted, Tri::No, "{}", s.manifest);
            assert_eq!(
                s.fit_predict_reload_verified,
                Tri::Unknown,
                "{}",
                s.manifest
            );
            assert_eq!(s.export_supported, Tri::Unknown, "{}", s.manifest);
            assert_eq!(s.availability, "unknown", "{}", s.manifest);
        }
    }

    #[test]
    fn the_xgboost_entries_are_present_as_references_and_none_is_admitted() {
        let table = load().unwrap();
        let xgb: Vec<_> = table
            .entries
            .iter()
            .filter(|e| e.manifest.contains("xgboost"))
            .collect();
        assert_eq!(xgb.len(), 84);
        assert!(xgb.iter().all(|e| e.coverage == "pending"));
        assert!(statuses()
            .unwrap()
            .iter()
            .filter(|s| s.manifest.contains("xgboost"))
            .all(|s| s.runtime_admitted == Tri::No));
    }

    #[test]
    fn the_summary_counts_entries_not_algorithms() {
        let s = summary().unwrap();
        assert_eq!(s.entries, 655);
        assert_eq!(s.runtime_admitted_yes, 0);
        assert_eq!(s.fit_predict_reload_verified_unknown, 655);
    }
}
