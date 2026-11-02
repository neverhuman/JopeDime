use super::classify::{decide, Facts};
use super::groups::{flatten, GroupLookup, Ledger};
use super::index::aggregate;
use super::inputs::{load, load_spec, InputRecord};
use super::inventory::{self, E1Pair, Inventory, PinState};
use super::lifecycle::{coverage, Lifecycles};
use super::policy::{Policy, DEFAULT_POLICY_JSON};
use super::refs::{RefSet, Role, ScanStats};
use super::Decision;
use super::{
    report, round4, row, Action, Built, Config, InputsLock, InventoryBucket, InventoryLock,
    InventorySummary, Note, RowInventory, Summary, Totals, INPUTS_LOCK_SCHEMA, PLAN_SUMMARY_SCHEMA,
};
use crate::canon::sha256_hex;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

fn assign_classes(ledger: &Ledger, p: &Policy) -> Vec<(String, &'static str)> {
    let mut out = vec![(String::new(), ""); ledger.groups.len()];
    let mut remaining: BTreeMap<String, BTreeMap<String, i128>> = ledger
        .bucket_classes
        .iter()
        .map(|(b, c)| {
            (
                b.clone(),
                c.iter().map(|(k, v)| (k.clone(), i128::from(*v))).collect(),
            )
        })
        .collect();
    for (i, g) in ledger.groups.iter().enumerate() {
        if let Some(o) = p.class_override_for(&g.bucket, &g.prefix) {
            out[i] = (o.class.clone(), "policy-override");
            if let Some(c) = remaining
                .get_mut(&g.bucket)
                .and_then(|m| m.get_mut(&o.class))
            {
                *c -= i128::from(g.bytes);
            }
        }
    }
    for (i, g) in ledger.groups.iter().enumerate() {
        if !out[i].1.is_empty() {
            continue;
        }
        let classes = ledger.bucket_classes.get(&g.bucket);
        out[i] = match classes.map(BTreeMap::len).unwrap_or(0) {
            0 => ("STANDARD".into(), "bucket-empty-default"),
            1 => (
                classes
                    .and_then(|c| c.keys().next().cloned())
                    .unwrap_or_default(),
                "bucket-uniform",
            ),
            _ => {
                let best = remaining
                    .get(&g.bucket)
                    .and_then(|m| m.iter().max_by_key(|(_, v)| **v).map(|(k, _)| k.clone()))
                    .unwrap_or_else(|| "STANDARD".into());
                (best, "bucket-dominant")
            }
        };
    }
    out
}

fn record_for_bytes(role: &str, path: &str, bytes: &[u8], check: &str) -> InputRecord {
    InputRecord {
        role: role.into(),
        path: path.into(),
        bytes: bytes.len() as u64,
        sha256: sha256_hex(bytes),
        check: check.into(),
    }
}

fn add_note(
    notes: &mut BTreeMap<(String, String), (u64, f64)>,
    kind: &str,
    detail: String,
    gb: f64,
) {
    let e = notes.entry((kind.to_string(), detail)).or_default();
    e.0 += 1;
    e.1 += gb;
}

mod analysis;
mod evaluate;
mod finish;
mod load;
mod state;

/// Performs the build operation and returns an error if it cannot be completed.
pub fn build(cfg: &Config) -> Result<Built> {
    let as_of_unix = crate::civil::parse_rfc3339(&cfg.as_of).with_context(|| {
        format!(
            "--as-of {:?} is not RFC 3339 (YYYY-MM-DDTHH:MM:SSZ)",
            cfg.as_of
        )
    })?;
    let loaded = load::load_inputs(cfg, as_of_unix)?;
    let evaluated = evaluate::evaluate(cfg, loaded)?;
    let analysis = analysis::analyze(&evaluated)?;
    finish::finish(cfg, evaluated, analysis)
}
