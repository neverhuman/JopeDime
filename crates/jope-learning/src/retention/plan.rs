//! Builds the dry-run plan from the inputs and writes it.
//!
//! Outputs (in a new, empty directory; nothing is ever overwritten):
//! - `retention-plan.jsonl`: one row per group, sorted by bucket, prefix, direct;
//! - `retention-plan.jsonl.sha256`: the plan's sha256, the identity an owner approves;
//! - `retention-summary.json`: per-action and per-tier totals, hazards, findings, diff;
//! - `brain-inputs.lock.json`: every input's path, size and sha256;
//! - `REPORT.md`: the human summary posted with the plan.

use super::classify::{decide, Decision, Facts};
use super::groups::{flatten, Group, GroupLookup, Ledger};
use super::index::{aggregate, IndexSummary};
use super::inputs::{load, load_spec, InputRecord, InputSpec};
use super::lifecycle::{coverage, Coverage, Lifecycles};
use super::policy::{Policy, DEFAULT_POLICY_JSON};
use super::refs::{RefSet, Role, ScanStats};
use super::{round4, Action, INPUTS_LOCK_SCHEMA, PLAN_ROW_SCHEMA, PLAN_SUMMARY_SCHEMA};
use crate::canon::sha256_hex;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub struct Config {
    pub as_of: String,
    pub ledger: InputSpec,
    pub gp_index: InputSpec,
    /// Pinned for the record only (count separation, corrections).
    pub extra_inputs: Vec<InputSpec>,
    pub refs: Vec<InputSpec>,
    /// A manifest whose `members` map file names to sha256; every `--refs` file in the same
    /// directory must be listed there with the same hash.
    pub roster_manifest: Option<InputSpec>,
    pub scans: Vec<(Role, PathBuf)>,
    pub scan_max_bytes: u64,
    /// Extra directory names never entered by a scan (e.g. this lane's own output directory,
    /// which holds input snapshots).
    pub scan_skip_dirs: Vec<String>,
    pub lifecycle_dir: Option<PathBuf>,
    pub policy: Option<PathBuf>,
    pub previous_plan: Option<PathBuf>,
    pub source_commit: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlanRow {
    pub schema: &'static str,
    pub group_id: String,
    pub bucket: String,
    pub region: String,
    pub prefix: String,
    pub direct: bool,
    pub objects: u64,
    pub bytes: u64,
    pub gb: f64,
    pub mean_object_bytes: u64,
    pub min_last_modified: Option<String>,
    pub max_last_modified: Option<String>,
    pub storage_class_basis: &'static str,
    pub tier: &'static str,
    pub action: Action,
    pub kind: Option<&'static str>,
    pub family: Option<String>,
    pub rule: String,
    pub live: bool,
    pub t0: Option<String>,
    pub refs_by_role: BTreeMap<String, u64>,
    /// Up to five `role:source` files that referenced objects in this group.
    pub ref_sources_sample: Vec<String>,
    pub roster_contexts: BTreeMap<String, u64>,
    pub pinned_programs: BTreeMap<String, u64>,
    pub index: Option<IndexSummary>,
    pub tv: super::classify::Tv,
    pub e_rules_fired: Vec<String>,
    pub e_candidates: Vec<String>,
    pub lifecycle: Coverage,
    pub cost: super::classify::Cost,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Totals {
    pub groups: u64,
    pub objects: u64,
    pub bytes: u64,
    pub gb: f64,
    pub now_usd_mo: f64,
    pub after_usd_mo: f64,
    pub delta_usd_mo: f64,
    pub one_time_in_region_usd: f64,
    pub one_time_off_aws_usd: f64,
    pub gb_after_needs_e1: f64,
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
pub struct Note {
    pub kind: String,
    pub detail: String,
    pub groups: u64,
    pub gb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputsLock {
    pub schema: &'static str,
    pub as_of: String,
    pub tool: String,
    pub source_commit: Option<String>,
    pub policy: InputRecord,
    pub inputs: Vec<InputRecord>,
    pub lifecycle_files: Vec<InputRecord>,
    pub scans: Vec<ScanStats>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub schema: &'static str,
    pub as_of: String,
    pub mode: &'static str,
    pub granularity: String,
    pub plan_file: &'static str,
    pub plan_sha256: String,
    pub rows: u64,
    pub inputs_lock_sha256: String,
    pub policy_sha256: String,
    pub prices: String,
    pub totals: Totals,
    pub by_action: BTreeMap<&'static str, Totals>,
    pub by_tier: BTreeMap<&'static str, Totals>,
    pub expire: String,
    pub e_rule_status: BTreeMap<&'static str, &'static str>,
    pub hazards: Vec<Note>,
    pub notes: Vec<Note>,
    pub findings: Vec<String>,
    pub refs: BTreeMap<String, u64>,
    pub refs_unmatched_in_scope: u64,
    pub refs_unmatched_sample: Vec<String>,
    pub refs_other_buckets: u64,
    pub refs_bucketless_resolved: u64,
    pub refs_bucketless_unresolved: u64,
    pub prefix_mentions: u64,
    pub index_rows: BTreeMap<&'static str, u64>,
    pub index_rows_unmatched_sample: Vec<String>,
    pub diff_vs_previous: Option<BTreeMap<String, u64>>,
}

pub struct Built {
    pub rows: Vec<PlanRow>,
    pub plan_jsonl: Vec<u8>,
    pub summary: Summary,
    pub lock: InputsLock,
    pub lock_json: Vec<u8>,
    pub report_md: String,
}

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

pub fn build(cfg: &Config) -> Result<Built> {
    let as_of_unix = crate::civil::parse_rfc3339(&cfg.as_of).with_context(|| {
        format!(
            "--as-of {:?} is not RFC 3339 (YYYY-MM-DDTHH:MM:SSZ)",
            cfg.as_of
        )
    })?;

    // Policy.
    let (policy_bytes, policy_path) = match &cfg.policy {
        Some(p) => (
            fs::read(p).with_context(|| format!("cannot read policy {}", p.display()))?,
            p.display().to_string(),
        ),
        None => (
            DEFAULT_POLICY_JSON.as_bytes().to_vec(),
            "embedded:retention/policy.v1.json".into(),
        ),
    };
    let policy = Policy::parse(&policy_bytes)?;
    let policy_rec = record_for_bytes("policy", &policy_path, &policy_bytes, "parsed+validated");

    // Ledger and GP index.
    let mut inputs = Vec::new();
    let ledger_in = load_spec(&cfg.ledger)?;
    let ledger_v: Value =
        serde_json::from_slice(&ledger_in.bytes).context("ledger does not parse")?;
    let ledger = flatten(&ledger_v)?;
    inputs.push(ledger_in.record);
    let index_in = load_spec(&cfg.gp_index)?;
    let index_v: Value =
        serde_json::from_slice(&index_in.bytes).context("GP index does not parse")?;
    inputs.push(index_in.record);
    for spec in &cfg.extra_inputs {
        inputs.push(load_spec(spec)?.record);
    }
    let lookup = GroupLookup::new(&ledger.groups);
    let known: BTreeSet<String> = policy.known_score_identities.keys().cloned().collect();
    let agg = aggregate(&index_v, &lookup, &known)?;

    // References.
    let mut refs = RefSet::default();
    let mut members: Option<(PathBuf, BTreeMap<String, String>)> = None;
    if let Some(spec) = &cfg.roster_manifest {
        let m = load_spec(spec)?;
        let v: Value =
            serde_json::from_slice(&m.bytes).context("roster manifest does not parse")?;
        let map: BTreeMap<String, String> = v
            .get("members")
            .and_then(Value::as_object)
            .context("roster manifest lacks members")?
            .iter()
            .filter_map(|(k, e)| {
                e.get("sha256")
                    .and_then(Value::as_str)
                    .map(|s| (k.clone(), s.to_string()))
            })
            .collect();
        let dir = spec
            .path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        members = Some((dir, map));
        inputs.push(m.record);
    }
    let mut ref_counts: BTreeMap<String, u64> = BTreeMap::new();
    for spec in &cfg.refs {
        let role = Role::parse(&spec.role)?;
        let mut loaded = load(&spec.role, &spec.path, spec.expect_prefix.as_deref())?;
        if let Some((dir, map)) = &members {
            if spec.path.parent() == Some(dir.as_path()) {
                let name = spec
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let want = map
                    .get(&name)
                    .with_context(|| format!("{name} is not a member of the roster manifest"))?;
                ensure!(
                    *want == loaded.record.sha256,
                    "{name} sha256 {} != roster manifest {want}",
                    loaded.record.sha256
                );
                loaded.record.check.push_str("; roster-manifest member ok");
            }
        }
        let name = spec.path.display().to_string();
        let (n, _) = refs.add_file(role, &name, &loaded.bytes)?;
        *ref_counts
            .entry(format!("{}:{}", role.as_str(), name))
            .or_default() += n;
        inputs.push(loaded.record);
    }
    // Seed packets folded into the index (jope.pipeline-seeds/v1): the PolicyLM -> GP lineage edge.
    if let Some(dp) = index_v.get("derived_packets").and_then(Value::as_array) {
        for p in dp {
            if let Some(sha) = p.get("program_sha").and_then(Value::as_str) {
                refs.programs
                    .entry(sha.to_ascii_lowercase())
                    .or_default()
                    .insert(Role::Seed);
            }
        }
    }
    let needles = policy.in_scope_bucket_prefixes.clone();
    let mut skip: Vec<&str> = vec![".git", "target", "node_modules", "worktrees", "trash"];
    skip.extend(cfg.scan_skip_dirs.iter().map(String::as_str));
    for (role, root) in &cfg.scans {
        let prose = policy.prose_extensions_for(*role);
        refs.scan_root(*role, root, cfg.scan_max_bytes, &needles, &skip, prose)?;
    }

    // Map references onto groups.
    let mut refs_by_group: BTreeMap<usize, BTreeMap<Role, u64>> = BTreeMap::new();
    let mut sources_by_group: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let (mut unmatched_in_scope, mut other_buckets) = (0u64, 0u64);
    let (mut bucketless_resolved, mut bucketless_unresolved) = (0u64, 0u64);
    let mut unmatched_sample = Vec::new();
    let known_buckets: BTreeSet<&str> = ledger.groups.iter().map(|g| g.bucket.as_str()).collect();
    for r in &refs.objects {
        let found = if r.bucket.is_empty() {
            // A bucketless key counts only when exactly one bucket holds a non-root group for it.
            let hits: BTreeSet<usize> = known_buckets
                .iter()
                .filter_map(|b| lookup.find(b, &r.key))
                .filter(|gi| !ledger.groups[*gi].prefix.is_empty())
                .collect();
            if hits.len() == 1 {
                bucketless_resolved += 1;
                hits.into_iter().next()
            } else {
                bucketless_unresolved += 1;
                continue;
            }
        } else {
            lookup.find(&r.bucket, &r.key)
        };
        match found {
            Some(gi) => {
                *refs_by_group
                    .entry(gi)
                    .or_default()
                    .entry(r.role)
                    .or_default() += 1;
                let s = sources_by_group.entry(gi).or_default();
                if s.len() < 5 {
                    s.insert(format!("{}:{}", r.role.as_str(), r.source));
                }
            }
            None if known_buckets.contains(r.bucket.as_str()) || policy.in_scope(&r.bucket) => {
                unmatched_in_scope += 1;
                if unmatched_sample.len() < 20 {
                    unmatched_sample.push(format!(
                        "{} s3://{}/{}",
                        r.role.as_str(),
                        r.bucket,
                        r.key
                    ));
                }
            }
            None => other_buckets += 1,
        }
    }
    let lifecycles = match &cfg.lifecycle_dir {
        Some(d) => Lifecycles::load_dir(d)?,
        None => Lifecycles::default(),
    };
    let classes = assign_classes(&ledger, &policy);

    // Decide every group.
    let empty = BTreeMap::new();
    let mut rows = Vec::with_capacity(ledger.groups.len());
    for (gi, g) in ledger.groups.iter().enumerate() {
        let gidx = agg.per_group.get(&gi);
        let mut ctx_roles: BTreeMap<Role, u64> = BTreeMap::new();
        let mut prog_roles: BTreeMap<Role, u64> = BTreeMap::new();
        if let Some(i) = gidx {
            for c in &i.contexts {
                for role in refs.contexts.get(c).into_iter().flatten() {
                    *ctx_roles.entry(*role).or_default() += 1;
                }
            }
            for p in &i.programs {
                for role in refs.programs.get(p).into_iter().flatten() {
                    *prog_roles.entry(*role).or_default() += 1;
                }
            }
        }
        let family_scored = gidx.and_then(|i| {
            i.families
                .iter()
                .map(|f| agg.scored_by_family.get(f).copied().unwrap_or(0))
                .max()
        });
        let facts = Facts {
            group: g,
            storage_class: classes[gi].0.clone(),
            refs_by_role: refs_by_group
                .get(&gi)
                .cloned()
                .unwrap_or_else(|| empty.clone()),
            ctx_roles,
            prog_roles,
            index: gidx,
            family_scored_rows: family_scored,
            coverage: coverage(&lifecycles, &g.bucket, &g.prefix, g.direct),
            as_of_unix,
        };
        let d: Decision = decide(&facts, &policy)?;
        if d.action == Action::Expire {
            bail!(
                "internal: EXPIRE reached for {} without object-level evidence",
                g.id()
            );
        }
        if super::classify::hard_t0(&g.bucket, &g.prefix).is_some() && d.action != Action::KeepT0 {
            bail!(
                "internal: hard T0 group {} got {}",
                g.id(),
                d.action.as_str()
            );
        }
        let sources = sources_by_group
            .get(&gi)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        rows.push(row(g, &facts, d, classes[gi].1, sources));
    }

    // Hazards, notes, findings.
    let mut hazards = BTreeMap::new();
    let mut notes = BTreeMap::new();
    for r in &rows {
        let gb = r.bytes as f64 / 1e9;
        if !r.lifecycle.expires_current.is_empty() && (r.action == Action::KeepT0 || r.tv.value) {
            add_note(
                &mut hazards,
                "expire-current-over-protected",
                format!(
                    "{} on {} group(s) with T0/TV=true: {}",
                    r.bucket,
                    r.action.as_str(),
                    r.lifecycle.expires_current.join(" | ")
                ),
                gb,
            );
        }
        if r.action == Action::KeepHot
            && matches!(
                r.lifecycle.cold_class.as_deref(),
                Some("DEEP_ARCHIVE" | "GLACIER" | "GLACIER_IR")
            )
        {
            add_note(
                &mut hazards,
                "cold-transition-over-keep-hot",
                format!("{}: {}", r.bucket, r.lifecycle.full.join(" | ")),
                gb,
            );
        }
        // Owner ruling D2 (ENG:watcher, 2026-10-13T12:42:52Z): T0 keeps the 90-day noncurrent
        // window like everything else (365 d declined), so only a shorter rule is a hazard.
        if r.action == Action::KeepT0
            && r.lifecycle
                .noncurrent_days
                .is_some_and(|d| d < policy.noncurrent_days)
            && policy.in_scope(&r.bucket)
        {
            add_note(
                &mut hazards,
                "t0-noncurrent-below-policy",
                format!(
                    "{}: T0 groups under noncurrent expiry {}d, below the policy's {}d",
                    r.bucket,
                    r.lifecycle.noncurrent_days.unwrap_or(0),
                    policy.noncurrent_days
                ),
                gb,
            );
        }
        if r.action == Action::KeepT0
            && matches!(
                r.lifecycle.cold_class.as_deref(),
                Some("DEEP_ARCHIVE" | "GLACIER" | "GLACIER_IR")
            )
        {
            add_note(
                &mut notes,
                "t0-under-applied-cold-transition",
                format!(
                    "{}: applied rule moves T0 bytes to {} (allowed: T0 means never delete, not never cold; confirm restore latency): {}",
                    r.bucket,
                    r.lifecycle.cold_class.as_deref().unwrap_or("?"),
                    r.lifecycle.full.join(" | ")
                ),
                gb,
            );
        }
        if !r.lifecycle.known && policy.in_scope(&r.bucket) {
            add_note(
                &mut notes,
                "lifecycle-not-supplied",
                format!(
                    "{}: no lifecycle file supplied; hazards unchecked",
                    r.bucket
                ),
                gb,
            );
        }
        if r.storage_class_basis == "bucket-dominant" {
            add_note(
                &mut notes,
                "storage-class-estimated",
                format!(
                    "{}: mixed-class bucket, class per group estimated as the dominant remainder",
                    r.bucket
                ),
                gb,
            );
        }
    }
    let to_notes = |m: BTreeMap<(String, String), (u64, f64)>| -> Vec<Note> {
        m.into_iter()
            .map(|((kind, detail), (n, gb))| Note {
                kind,
                detail,
                groups: n,
                gb: round4(gb),
            })
            .collect()
    };
    let mut findings = Vec::new();
    let mut fam_refs: BTreeMap<(String, String), u64> = BTreeMap::new();
    for r in &rows {
        for (role, n) in &r.refs_by_role {
            let fam = r.family.clone().unwrap_or_else(|| r.bucket.clone());
            *fam_refs.entry((fam, role.clone())).or_default() += n;
        }
        if r.kind == Some("gp-result-unconsumed") && r.action != Action::DistillCold {
            findings.push(format!(
                "policy marks {} as unconsumed, but it is {} ({}): {}",
                r.group_id,
                r.action.as_str(),
                r.t0.clone().unwrap_or_else(|| "referenced or live".into()),
                r.reason.chars().take(160).collect::<String>()
            ));
        }
    }
    for ((fam, role), n) in &fam_refs {
        findings.push(format!(
            "object refs into family {fam} from role {role}: {n}"
        ));
    }
    if !agg.rows_unmatched.is_empty() {
        findings.push(format!(
            "{} GP index rows point at keys outside every ledger group",
            agg.rows_unmatched.len()
        ));
    }

    // Plan bytes and totals.
    let mut plan_jsonl = Vec::new();
    for r in &rows {
        plan_jsonl.extend(serde_json::to_vec(r)?);
        plan_jsonl.push(b'\n');
    }
    let plan_sha = sha256_hex(&plan_jsonl);
    let mut totals = Totals::default();
    let mut by_action: BTreeMap<&'static str, Totals> = Action::ALL
        .iter()
        .map(|a| (a.as_str(), Totals::default()))
        .collect();
    let mut by_tier: BTreeMap<&'static str, Totals> = BTreeMap::new();
    for r in &rows {
        totals.add(r);
        by_action.entry(r.action.as_str()).or_default().add(r);
        by_tier.entry(r.tier).or_default().add(r);
    }
    totals.finish();
    by_action.values_mut().for_each(Totals::finish);
    by_tier.values_mut().for_each(Totals::finish);
    ensure!(
        totals.objects == ledger.total_objects && totals.bytes == ledger.total_bytes,
        "plan totals do not reconcile with the ledger"
    );

    let lock = InputsLock {
        schema: INPUTS_LOCK_SCHEMA,
        as_of: cfg.as_of.clone(),
        tool: crate::TOOL_VERSION.into(),
        source_commit: cfg.source_commit.clone(),
        policy: policy_rec.clone(),
        inputs,
        lifecycle_files: lifecycles
            .files
            .iter()
            .map(|(p, s, n)| InputRecord {
                role: "lifecycle".into(),
                path: p.clone(),
                bytes: *n,
                sha256: s.clone(),
                check: "parsed".into(),
            })
            .collect(),
        scans: refs.scans.clone(),
    };
    let mut lock_json = serde_json::to_vec_pretty(&lock)?;
    lock_json.push(b'\n');

    let diff = match &cfg.previous_plan {
        Some(p) => Some(diff_previous(p, &rows)?),
        None => None,
    };
    let mut ref_totals: BTreeMap<String, u64> = BTreeMap::new();
    for r in &refs.objects {
        *ref_totals.entry(r.role.as_str().to_string()).or_default() += 1;
    }
    ref_totals.insert("contexts".into(), refs.contexts.len() as u64);
    ref_totals.insert("programs".into(), refs.programs.len() as u64);
    let summary = Summary {
        schema: PLAN_SUMMARY_SCHEMA,
        as_of: cfg.as_of.clone(),
        mode: "DRY-RUN: read-only; no S3 calls; this tool has no apply, tag, transition or delete path",
        granularity: format!(
            "prefix groups from the S3 object ledger generated {} (S3 Inventory not yet available)",
            ledger.generated_at
        ),
        plan_file: "retention-plan.jsonl",
        plan_sha256: plan_sha,
        rows: rows.len() as u64,
        inputs_lock_sha256: sha256_hex(&lock_json),
        policy_sha256: policy_rec.sha256.clone(),
        prices: format!("{} (estimates; billed S3 is currently credited)", policy.prices.label),
        totals,
        by_action,
        by_tier,
        expire: "0 groups: EXPIRE needs TV=false, an E1-E5 rule proven with object-level evidence (inventory + GET+sha256), and an agreeing independent reference scan; none is available at prefix level".into(),
        e_rule_status: BTreeMap::from([
            ("E1", "not evaluable: needs S3 Inventory key/size join plus GET+sha256 of both copies (etag is not enough)"),
            ("E2", "not evaluable: no checkpoint->L6 curve table (gp_distilled/v1 not built)"),
            ("E3", "not evaluable: no distilled table hash in a T0 manifest"),
            ("E4", "not evaluable: needs program_sha subset proof over full bodies"),
            ("E5", "delegated to the applied lifecycle rules p0-noncurrent-expire-* (ledger lists current versions only)"),
        ]),
        hazards: to_notes(hazards),
        notes: to_notes(notes),
        findings,
        refs: ref_totals,
        refs_unmatched_in_scope: unmatched_in_scope,
        refs_unmatched_sample: unmatched_sample,
        refs_other_buckets: other_buckets,
        refs_bucketless_resolved: bucketless_resolved,
        refs_bucketless_unresolved: bucketless_unresolved,
        prefix_mentions: refs.prefix_mentions,
        index_rows: BTreeMap::from([
            ("total", agg.rows_total),
            ("s3", agg.rows_s3),
            ("local", agg.rows_local),
            ("unmatched", agg.rows_unmatched.len() as u64),
            ("qualified", agg.qualified_total),
        ]),
        index_rows_unmatched_sample: agg.rows_unmatched.iter().take(20).cloned().collect(),
        diff_vs_previous: diff,
    };
    let report_md = report(&summary, &rows, &lock, &ref_counts);
    Ok(Built {
        rows,
        plan_jsonl,
        summary,
        lock,
        lock_json,
        report_md,
    })
}

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
            "- scan {} `{}`: {} files seen, {} not text, {} over cap, {} read ({:.1} GB), {} mention buckets, {} inventories skipped, {} unreadable, {} object refs ({} prose mentions, D1), {} prefix mentions\n",
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
