//! Builds the dry-run plan from the inputs and writes it.
//!
//! Outputs (in a new, empty directory; nothing is ever overwritten):
//! - `retention-plan.jsonl`: one row per group, sorted by bucket, prefix, direct;
//! - `retention-plan.jsonl.sha256`: the plan's sha256, the identity an owner approves;
//! - `retention-summary.json`: per-action and per-tier totals, hazards, findings, diff;
//! - `brain-inputs.lock.json`: every input's path, size and sha256;
//! - `REPORT.md`: the human summary posted with the plan.

use super::classify::{Decision, Facts};
use super::groups::Group;
use super::index::IndexSummary;
use super::inputs::{InputRecord, InputSpec};
use super::inventory::{DataFile, E1Overlap, GroupInv, ReadStats};
use super::lifecycle::Coverage;
use super::policy::Policy;
use super::refs::{Role, ScanStats};
use super::{round4, Action, PLAN_ROW_SCHEMA};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// `Config` is a record for the config contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct Config {
    /// As of recorded for this config.
    pub as_of: String,
    /// Ledger recorded for this config.
    pub ledger: InputSpec,
    /// Gp index recorded for this config.
    pub gp_index: InputSpec,
    /// Pinned for the record only (count separation, corrections).
    pub extra_inputs: Vec<InputSpec>,
    /// Collection of refs values carried by this config.
    pub refs: Vec<InputSpec>,
    /// A manifest whose `members` map file names to sha256; every `--refs` file in the same
    /// directory must be listed there with the same hash.
    pub roster_manifest: Option<InputSpec>,
    /// Collection of scans values carried by this config.
    pub scans: Vec<(Role, PathBuf)>,
    /// Byte count for the associated scan max.
    pub scan_max_bytes: u64,
    /// Extra directory names never entered by a scan (e.g. this lane's own output directory,
    /// which holds input snapshots).
    pub scan_skip_dirs: Vec<String>,
    /// Optional lifecycle dir value; `None` means it is not available.
    pub lifecycle_dir: Option<PathBuf>,
    /// Local S3 Inventory `manifest.json` copies, one per bucket (optional; without them the plan
    /// is prefix-level only).
    pub inventories: Vec<InputSpec>,
    /// Optional policy value; `None` means it is not available.
    pub policy: Option<PathBuf>,
    /// Optional previous plan value; `None` means it is not available.
    pub previous_plan: Option<PathBuf>,
    /// Optional source commit value; `None` means it is not available.
    pub source_commit: Option<String>,
}

/// A group's S3 Inventory facts as recorded in its plan row.
#[derive(Clone, Debug, Serialize)]
pub struct RowInventory {
    /// Timestamp at which the associated report created occurred.
    pub report_created_at: String,
    /// Current objects and bytes equal the ledger's: dates and classes are exact for the group.
    pub unchanged_since_report: bool,
    /// The ledger disagrees with the report although it shows no write after it (written only
    /// when true): one input is wrong, so the group gets no changing action.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub ledger_conflict: bool,
    #[serde(flatten)]
    /// Facts recorded for this row inventory.
    pub facts: GroupInv,
}

#[derive(Clone, Debug, Serialize)]
/// `InventoryLock` is a record for the inventory lock contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct InventoryLock {
    /// Bucket recorded for this inventory lock.
    pub bucket: String,
    /// Timestamp at which the associated report created occurred.
    pub report_created_at: String,
    /// Manifest recorded for this inventory lock.
    pub manifest: InputRecord,
    /// Collection of files values carried by this inventory lock.
    pub files: Vec<DataFile>,
    /// Read recorded for this inventory lock.
    pub read: ReadStats,
}

#[derive(Clone, Debug, Serialize)]
/// `InventoryBucket` is a record for the inventory bucket contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct InventoryBucket {
    /// Bucket recorded for this inventory bucket.
    pub bucket: String,
    /// Timestamp at which the associated report created occurred.
    pub report_created_at: String,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Current objects recorded for this inventory bucket.
    pub current_objects: u64,
    /// Byte count for the associated current.
    pub current_bytes: u64,
    /// Noncurrent versions recorded for this inventory bucket.
    pub noncurrent_versions: u64,
    /// Byte count for the associated noncurrent.
    pub noncurrent_bytes: u64,
    /// Delete markers recorded for this inventory bucket.
    pub delete_markers: u64,
    /// Rows outside ledger groups recorded for this inventory bucket.
    pub rows_outside_ledger_groups: u64,
    /// The ledger's current objects/bytes for the bucket (a later listing), and the differences.
    pub ledger_objects: u64,
    /// Byte count for the associated ledger.
    pub ledger_bytes: u64,
    /// Ledger minus inventory objects recorded for this inventory bucket.
    pub ledger_minus_inventory_objects: i64,
    /// Byte count for the associated ledger minus inventory.
    pub ledger_minus_inventory_bytes: i64,
}

#[derive(Clone, Debug, Serialize)]
/// `InventorySummary` is a record for the inventory summary contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct InventorySummary {
    /// Collection of buckets values carried by this inventory summary.
    pub buckets: Vec<InventoryBucket>,
    /// In-scope ledger buckets without a report (prefix-level facts only).
    pub ledger_buckets_without_report: Vec<String>,
    /// Groups with inventory recorded for this inventory summary.
    pub groups_with_inventory: u64,
    /// Groups unchanged since report recorded for this inventory summary.
    pub groups_unchanged_since_report: u64,
    /// Groups whose ledger counts disagree with the report without a later write (hazard
    /// `ledger-inventory-conflict`), with both sides' numbers.
    pub ledger_conflicts: Vec<String>,
    /// Noncurrent bytes in t0 groups recorded for this inventory summary.
    pub noncurrent_bytes_in_t0_groups: u64,
    /// Noncurrent bytes in other groups recorded for this inventory summary.
    pub noncurrent_bytes_in_other_groups: u64,
    /// role -> pin state -> distinct keys, for pins into inventoried buckets.
    pub pins: BTreeMap<String, BTreeMap<String, u64>>,
    /// Up to 200 pins whose key is not a current object, with the file that named them.
    pub pins_not_present_sample: Vec<String>,
    /// Collection of e1 overlap values carried by this inventory summary.
    pub e1_overlap: Vec<E1Overlap>,
}

#[derive(Clone, Debug, Serialize)]
/// `PlanRow` is a record for the plan row contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct PlanRow {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Identifier for the associated group.
    pub group_id: String,
    /// Bucket recorded for this plan row.
    pub bucket: String,
    /// Region recorded for this plan row.
    pub region: String,
    /// Prefix recorded for this plan row.
    pub prefix: String,
    /// Direct recorded for this plan row.
    pub direct: bool,
    /// Number of objects represented by this value.
    pub objects: u64,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Gb recorded for this plan row.
    pub gb: f64,
    /// Byte count for the associated mean object.
    pub mean_object_bytes: u64,
    /// Lower bound on last modified.
    pub min_last_modified: Option<String>,
    /// Upper bound on last modified.
    pub max_last_modified: Option<String>,
    /// Storage class basis recorded for this plan row.
    pub storage_class_basis: &'static str,
    /// Tier recorded for this plan row.
    pub tier: &'static str,
    /// Action recorded for this plan row.
    pub action: Action,
    /// Record kind or category represented by this value.
    pub kind: Option<&'static str>,
    /// Optional family value; `None` means it is not available.
    pub family: Option<String>,
    /// Rule recorded for this plan row.
    pub rule: String,
    /// Live recorded for this plan row.
    pub live: bool,
    /// Optional t0 value; `None` means it is not available.
    pub t0: Option<String>,
    /// Mapping of refs by role values carried by this plan row.
    pub refs_by_role: BTreeMap<String, u64>,
    /// Up to five `role:source` files that referenced objects in this group.
    pub ref_sources_sample: Vec<String>,
    /// Mapping of roster contexts values carried by this plan row.
    pub roster_contexts: BTreeMap<String, u64>,
    /// Mapping of pinned programs values carried by this plan row.
    pub pinned_programs: BTreeMap<String, u64>,
    /// Optional index value; `None` means it is not available.
    pub index: Option<IndexSummary>,
    /// Tv recorded for this plan row.
    pub tv: super::classify::Tv,
    /// Collection of e rules fired values carried by this plan row.
    pub e_rules_fired: Vec<String>,
    /// Collection of e candidates values carried by this plan row.
    pub e_candidates: Vec<String>,
    /// Lifecycle recorded for this plan row.
    pub lifecycle: Coverage,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional inventory value; `None` means it is not available.
    pub inventory: Option<RowInventory>,
    /// Cost recorded for this plan row.
    pub cost: super::classify::Cost,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize)]
/// `Totals` is a record for the totals contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct Totals {
    /// Groups recorded for this totals.
    pub groups: u64,
    /// Number of objects represented by this value.
    pub objects: u64,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Gb recorded for this totals.
    pub gb: f64,
    /// Now usd mo recorded for this totals.
    pub now_usd_mo: f64,
    /// After usd mo recorded for this totals.
    pub after_usd_mo: f64,
    /// Delta usd mo recorded for this totals.
    pub delta_usd_mo: f64,
    /// One time in region usd recorded for this totals.
    pub one_time_in_region_usd: f64,
    /// One time off aws usd recorded for this totals.
    pub one_time_off_aws_usd: f64,
    /// Gb after needs e1 recorded for this totals.
    pub gb_after_needs_e1: f64,
    /// Gb already applied by lifecycle recorded for this totals.
    pub gb_already_applied_by_lifecycle: f64,
}

impl Totals {
    fn add(&mut self, r: &PlanRow) {
        self.groups += 1;
        self.objects += r.objects;
        self.bytes += r.bytes;
        self.now_usd_mo += r.cost.now_usd_mo;
        self.after_usd_mo += r.cost.after_usd_mo;
        self.delta_usd_mo += r.cost.delta_usd_mo;
        self.one_time_in_region_usd += r.cost.one_time_in_region_usd;
        self.one_time_off_aws_usd += r.cost.one_time_off_aws_usd;
        if r.cost.after_needs_e1_of_originals {
            self.gb_after_needs_e1 += r.bytes as f64 / 1e9;
        }
        if r.cost.already_applied_by_lifecycle {
            self.gb_already_applied_by_lifecycle += r.bytes as f64 / 1e9;
        }
    }
    fn finish(&mut self) {
        self.gb = round4(self.bytes as f64 / 1e9);
        for x in [
            &mut self.now_usd_mo,
            &mut self.after_usd_mo,
            &mut self.delta_usd_mo,
            &mut self.one_time_in_region_usd,
            &mut self.one_time_off_aws_usd,
            &mut self.gb_after_needs_e1,
            &mut self.gb_already_applied_by_lifecycle,
        ] {
            *x = round4(*x);
        }
    }
}

#[derive(Clone, Debug, Serialize)]
/// `Note` is a record for the note contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct Note {
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Detail recorded for this note.
    pub detail: String,
    /// Groups recorded for this note.
    pub groups: u64,
    /// Gb recorded for this note.
    pub gb: f64,
}

#[derive(Clone, Debug, Serialize)]
/// `InputsLock` is a record for the inputs lock contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct InputsLock {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// As of recorded for this inputs lock.
    pub as_of: String,
    /// Tool recorded for this inputs lock.
    pub tool: String,
    /// Optional source commit value; `None` means it is not available.
    pub source_commit: Option<String>,
    /// Policy recorded for this inputs lock.
    pub policy: InputRecord,
    /// Collection of inputs values carried by this inputs lock.
    pub inputs: Vec<InputRecord>,
    /// Collection of lifecycle files values carried by this inputs lock.
    pub lifecycle_files: Vec<InputRecord>,
    /// Collection of scans values carried by this inputs lock.
    pub scans: Vec<ScanStats>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    /// Collection of inventory values carried by this inputs lock.
    pub inventory: Vec<InventoryLock>,
}

#[derive(Clone, Debug, Serialize)]
/// `Summary` is a record for the summary contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct Summary {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// As of recorded for this summary.
    pub as_of: String,
    /// Mode recorded for this summary.
    pub mode: &'static str,
    /// Granularity recorded for this summary.
    pub granularity: String,
    /// Plan file recorded for this summary.
    pub plan_file: &'static str,
    /// SHA-256 digest identifying the associated plan.
    pub plan_sha256: String,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// SHA-256 digest identifying the associated inputs lock.
    pub inputs_lock_sha256: String,
    /// SHA-256 digest identifying the associated policy.
    pub policy_sha256: String,
    /// Prices recorded for this summary.
    pub prices: String,
    /// Totals recorded for this summary.
    pub totals: Totals,
    /// Mapping of by action values carried by this summary.
    pub by_action: BTreeMap<&'static str, Totals>,
    /// Mapping of by tier values carried by this summary.
    pub by_tier: BTreeMap<&'static str, Totals>,
    /// Expire recorded for this summary.
    pub expire: String,
    /// Mapping of e rule status values carried by this summary.
    pub e_rule_status: BTreeMap<&'static str, String>,
    /// Collection of hazards values carried by this summary.
    pub hazards: Vec<Note>,
    /// Collection of notes values carried by this summary.
    pub notes: Vec<Note>,
    /// Collection of findings values carried by this summary.
    pub findings: Vec<String>,
    /// Mapping of refs values carried by this summary.
    pub refs: BTreeMap<String, u64>,
    /// Refs unmatched in scope recorded for this summary.
    pub refs_unmatched_in_scope: u64,
    /// Collection of refs unmatched sample values carried by this summary.
    pub refs_unmatched_sample: Vec<String>,
    /// Refs other buckets recorded for this summary.
    pub refs_other_buckets: u64,
    /// Refs bucketless resolved recorded for this summary.
    pub refs_bucketless_resolved: u64,
    /// Refs bucketless unresolved recorded for this summary.
    pub refs_bucketless_unresolved: u64,
    /// Prefix mentions recorded for this summary.
    pub prefix_mentions: u64,
    /// Number of index rows represented here.
    pub index_rows: BTreeMap<&'static str, u64>,
    /// Collection of index rows unmatched sample values carried by this summary.
    pub index_rows_unmatched_sample: Vec<String>,
    /// Optional diff vs previous value; `None` means it is not available.
    pub diff_vs_previous: Option<BTreeMap<String, u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional inventory value; `None` means it is not available.
    pub inventory: Option<InventorySummary>,
}

/// `Built` is a record for the built contract in this module: Builds the dry-run plan from the inputs and writes it.
pub struct Built {
    /// Number of rows represented by this value.
    pub rows: Vec<PlanRow>,
    /// Collection of plan jsonl values carried by this built.
    pub plan_jsonl: Vec<u8>,
    /// Summary recorded for this built.
    pub summary: Summary,
    /// Lock recorded for this built.
    pub lock: InputsLock,
    /// Collection of lock JSON values carried by this built.
    pub lock_json: Vec<u8>,
    /// Report md recorded for this built.
    pub report_md: String,
}

mod build;
pub use build::build;

fn row(
    g: &Group,
    f: &Facts,
    d: Decision,
    basis: &'static str,
    ref_sources_sample: Vec<String>,
) -> PlanRow {
    let names = |m: &BTreeMap<Role, u64>| {
        m.iter()
            .map(|(r, n)| (r.as_str().to_string(), *n))
            .collect()
    };
    PlanRow {
        schema: PLAN_ROW_SCHEMA,
        group_id: g.id(),
        bucket: g.bucket.clone(),
        region: g.region.clone(),
        prefix: g.prefix.clone(),
        direct: g.direct,
        objects: g.objects,
        bytes: g.bytes,
        gb: round4(g.bytes as f64 / 1e9),
        mean_object_bytes: g.bytes.checked_div(g.objects).unwrap_or(0),
        min_last_modified: g.min_last_modified.clone(),
        max_last_modified: g.max_last_modified.clone(),
        storage_class_basis: basis,
        tier: d.action.tier(),
        action: d.action,
        kind: d.kind,
        family: d.family,
        rule: d.rule,
        live: d.live,
        t0: d.t0,
        refs_by_role: names(&f.refs_by_role),
        ref_sources_sample,
        roster_contexts: names(&f.ctx_roles),
        pinned_programs: names(&f.prog_roles),
        index: f.index.map(super::index::GroupIndex::summary),
        tv: d.tv,
        e_rules_fired: Vec::new(),
        e_candidates: d.e_candidates,
        lifecycle: f.coverage.clone(),
        inventory: None,
        cost: d.cost,
        reason: d.reason,
    }
}

fn diff_previous(path: &Path, rows: &[PlanRow]) -> Result<BTreeMap<String, u64>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("cannot read previous plan {}", path.display()))?;
    let mut prev: BTreeMap<String, String> = BTreeMap::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let v: Value = serde_json::from_str(line)
            .with_context(|| format!("previous plan line {} does not parse", n + 1))?;
        let id = v
            .get("group_id")
            .and_then(Value::as_str)
            .context("previous plan row lacks group_id")?;
        let action = v
            .get("action")
            .and_then(Value::as_str)
            .context("previous plan row lacks action")?;
        prev.insert(id.to_string(), action.to_string());
    }
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for r in rows {
        seen.insert(r.group_id.clone());
        let key = match prev.get(&r.group_id) {
            Some(a) if a == r.action.as_str() => "unchanged".to_string(),
            Some(a) => format!("{a} -> {}", r.action.as_str()),
            None => format!("new -> {}", r.action.as_str()),
        };
        *out.entry(key).or_default() += 1;
    }
    for (id, a) in &prev {
        if !seen.contains(id) {
            *out.entry(format!("{a} -> gone")).or_default() += 1;
        }
    }
    Ok(out)
}

fn report(
    s: &Summary,
    rows: &[PlanRow],
    lock: &InputsLock,
    ref_counts: &BTreeMap<String, u64>,
) -> String {
    let mut o = String::new();
    o.push_str(&format!(
        "# S3 retention dry-run plan, as of {}\n\n",
        s.as_of
    ));
    o.push_str(&format!("**{}**\n\n", s.mode));
    o.push_str(&format!(
        "- Plan: `{}` sha256 `{}` ({} rows)\n",
        s.plan_file, s.plan_sha256, s.rows
    ));
    o.push_str(&format!(
        "- Inputs lock sha256 `{}`; policy sha256 `{}`\n",
        s.inputs_lock_sha256, s.policy_sha256
    ));
    o.push_str(&format!("- Granularity: {}\n", s.granularity));
    o.push_str(&format!("- Prices: {}\n", s.prices));
    o.push_str(&format!(
        "- Totals: {} objects, {:.1} GB, ${:.2}/mo now (list)\n\n",
        s.totals.objects, s.totals.gb, s.totals.now_usd_mo
    ));
    o.push_str("## Per action\n\n| Action | Groups | Objects | GB | $/mo now | $/mo after | Δ $/mo | one-time in-region $ | one-time from xbabe $ | GB needing E1 of originals | GB already under applied lifecycle |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (a, t) in &s.by_action {
        o.push_str(&format!(
            "| {a} | {} | {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} |\n",
            t.groups,
            t.objects,
            t.gb,
            t.now_usd_mo,
            t.after_usd_mo,
            t.delta_usd_mo,
            t.one_time_in_region_usd,
            t.one_time_off_aws_usd,
            t.gb_after_needs_e1,
            t.gb_already_applied_by_lifecycle
        ));
    }
    o.push_str(&format!("\nEXPIRE: {}\n\n", s.expire));
    o.push_str("## Largest groups per action\n\n");
    for a in Action::ALL {
        let mut sel: Vec<&PlanRow> = rows.iter().filter(|r| r.action == a).collect();
        if sel.is_empty() {
            continue;
        }
        sel.sort_by(|x, y| y.bytes.cmp(&x.bytes).then(x.group_id.cmp(&y.group_id)));
        o.push_str(&format!("### {}\n\n", a.as_str()));
        for r in sel.iter().take(8) {
            o.push_str(&format!("- {}\n", r.reason));
        }
        if sel.len() > 8 {
            let rest: u64 = sel[8..].iter().map(|r| r.bytes).sum();
            o.push_str(&format!(
                "- … {} more groups, {:.1} GB\n",
                sel.len() - 8,
                rest as f64 / 1e9
            ));
        }
        o.push('\n');
    }
    o.push_str("## Hazards in the applied lifecycle configuration\n\n");
    if s.hazards.is_empty() {
        o.push_str("None: no applied rule expires current versions under a T0/TV group, and no cold transition covers a KEEP_HOT group.\n\n");
    }
    for h in &s.hazards {
        o.push_str(&format!(
            "- **{}** ({} groups, {:.1} GB): {}\n",
            h.kind, h.groups, h.gb, h.detail
        ));
    }
    o.push_str("\n## Notes\n\n");
    for n in &s.notes {
        o.push_str(&format!(
            "- {} ({} groups, {:.1} GB): {}\n",
            n.kind, n.groups, n.gb, n.detail
        ));
    }
    o.push_str("\n## Findings\n\n");
    for f in &s.findings {
        o.push_str(&format!("- {f}\n"));
    }
    o.push_str("\n## E-rule status\n\n");
    for (k, v) in &s.e_rule_status {
        o.push_str(&format!("- {k}: {v}\n"));
    }
    o.push_str("\n## References\n\n");
    for (k, v) in ref_counts {
        o.push_str(&format!("- {k}: {v} object refs\n"));
    }
    for sc in &lock.scans {
        o.push_str(&format!(
            "- scan {} `{}`: {} files seen, {} not text, {} over cap (streamed, name no in-scope bucket), {} read ({:.1} GB), {} mention buckets, {} inventories skipped, {} unreadable, {} object refs ({} prose mentions, D1), {} prefix mentions\n",
            sc.role,
            sc.root,
            sc.files_seen,
            sc.files_not_text,
            sc.files_over_cap,
            sc.files_read,
            sc.bytes_read as f64 / 1e9,
            sc.files_mentioning_buckets,
            sc.inventory_skipped,
            sc.files_unreadable,
            sc.object_refs,
            sc.prose_refs,
            sc.prefix_mentions
        ));
    }
    o.push_str(&format!(
        "- unmatched object refs into ledger buckets: {}; refs into other buckets: {}; bucketless keys resolved/unresolved: {}/{}; prefix/bucket mentions (pin nothing): {}\n",
        s.refs_unmatched_in_scope,
        s.refs_other_buckets,
        s.refs_bucketless_resolved,
        s.refs_bucketless_unresolved,
        s.prefix_mentions
    ));
    if let Some(inv) = &s.inventory {
        o.push_str("\n## S3 Inventory\n\n");
        o.push_str(&format!(
            "{} groups have inventory rows; {} are unchanged since their report (exact dates and classes). Noncurrent bytes: {:.1} GB in T0 groups, {:.1} GB elsewhere.\n\n",
            inv.groups_with_inventory,
            inv.groups_unchanged_since_report,
            inv.noncurrent_bytes_in_t0_groups as f64 / 1e9,
            inv.noncurrent_bytes_in_other_groups as f64 / 1e9
        ));
        if !inv.ledger_conflicts.is_empty() {
            o.push_str(&format!(
                "{} groups disagree with the ledger although the ledger shows no write after the report (hazard ledger-inventory-conflict; no changing action):\n\n",
                inv.ledger_conflicts.len()
            ));
            for c in &inv.ledger_conflicts {
                o.push_str(&format!("- {c}\n"));
            }
            o.push('\n');
        }
        o.push_str("| Bucket | Report | Rows | Current obj | Current GB | Noncurrent ver | Noncurrent GB | Delete markers | Ledger - inventory (obj / GB) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for b in &inv.buckets {
            o.push_str(&format!(
                "| {} | {} | {} | {} | {:.1} | {} | {:.1} | {} | {} / {:.2} |\n",
                b.bucket,
                b.report_created_at,
                b.rows,
                b.current_objects,
                b.current_bytes as f64 / 1e9,
                b.noncurrent_versions,
                b.noncurrent_bytes as f64 / 1e9,
                b.delete_markers,
                b.ledger_minus_inventory_objects,
                b.ledger_minus_inventory_bytes as f64 / 1e9
            ));
        }
        o.push_str(&format!(
            "\nIn-scope buckets without a report (prefix-level only): {}\n\n",
            if inv.ledger_buckets_without_report.is_empty() {
                "none".to_string()
            } else {
                inv.ledger_buckets_without_report.join(", ")
            }
        ));
        for (role, states) in &inv.pins {
            o.push_str(&format!(
                "- pins {role}: {}\n",
                states
                    .iter()
                    .map(|(k, n)| format!("{k} {n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        for p in &inv.pins_not_present_sample {
            o.push_str(&format!("  - not present: {p}\n"));
        }
        for e in &inv.e1_overlap {
            o.push_str(&format!(
                "- E1 candidate s3://{}/{} vs s3://{}/{}: {} / {} keys same key+size+ETag ({:.1} GB), {} differ, {} only here, {} only there, {} multipart ETags; not proven (needs GET+sha256)\n",
                e.bucket,
                e.prefix,
                e.counterpart_bucket,
                e.counterpart_prefix,
                e.same_key_size_etag,
                e.objects,
                e.same_key_size_etag_bytes as f64 / 1e9,
                e.same_key_different,
                e.only_here,
                e.only_counterpart,
                e.multipart_etags
            ));
        }
    }
    o.push_str("\n## Inputs\n\n| Role | Path | Bytes | sha256 | Check |\n|---|---|---:|---|---|\n");
    for i in std::iter::once(&lock.policy)
        .chain(&lock.inputs)
        .chain(&lock.lifecycle_files)
    {
        o.push_str(&format!(
            "| {} | `{}` | {} | `{}` | {} |\n",
            i.role, i.path, i.bytes, i.sha256, i.check
        ));
    }
    if let Some(d) = &s.diff_vs_previous {
        o.push_str("\n## Diff vs previous plan\n\n");
        for (k, v) in d {
            o.push_str(&format!("- {k}: {v}\n"));
        }
    } else {
        o.push_str("\nDiff vs previous plan: none (first plan; no approved plan exists).\n");
    }
    o
}

/// Writes the outputs into `dir`, which must not exist or be empty.
pub fn write(b: &Built, dir: &Path) -> Result<()> {
    if dir.exists() {
        ensure!(
            dir.is_dir(),
            "{} exists and is not a directory",
            dir.display()
        );
        ensure!(
            fs::read_dir(dir)?.next().is_none(),
            "{} is not empty; refusing to overwrite a plan",
            dir.display()
        );
    }
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let w = |name: &str, bytes: &[u8]| -> Result<()> {
        fs::write(dir.join(name), bytes).with_context(|| format!("cannot write {name}"))
    };
    w("retention-plan.jsonl", &b.plan_jsonl)?;
    w(
        "retention-plan.jsonl.sha256",
        format!("{}  retention-plan.jsonl\n", b.summary.plan_sha256).as_bytes(),
    )?;
    let mut summary = serde_json::to_vec_pretty(&b.summary)?;
    summary.push(b'\n');
    w("retention-summary.json", &summary)?;
    w("brain-inputs.lock.json", &b.lock_json)?;
    w("REPORT.md", b.report_md.as_bytes())?;
    Ok(())
}
