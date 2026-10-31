//! Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule
//! status (§8.2), the action, list-price costs and the reason string (§8.7).
//!
//! Order (first match wins): hard or referenced T0 → out of scope → referenced by key from an
//! active roster/seed → live writer → the kind's default. A group whose kind is unknown gets
//! HOLD (§7.3 rule 6). EXPIRE is gated by [`expire_allowed`], which nothing here can satisfy at
//! prefix granularity.

use super::groups::Group;
use super::index::GroupIndex;
use super::inventory::{E1Overlap, GroupInv};
use super::lifecycle::Coverage;
use super::policy::{Kind, Policy};
use super::refs::Role;
use super::{round4, Action};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;

/// Hard T0 rules. They are code, not policy, so no policy file can remove them.
pub fn hard_t0(bucket: &str, prefix: &str) -> Option<String> {
    if bucket.contains("sealed") {
        return Some("hard T0: sealed bucket (NEVER delete)".into());
    }
    if bucket == "veox-jopedime-models-use2" {
        return Some("hard T0: model bundles (frozen v1 / promoted checkpoints)".into());
    }
    for seg in prefix.split('/').filter(|s| !s.is_empty()) {
        let s = seg.to_ascii_lowercase();
        let why = if s.contains("sealed") {
            "sealed"
        } else if s.contains("evidence") {
            "evidence"
        } else if s.starts_with("split") {
            "split / split-registry"
        } else if s.contains("prereg") {
            "prereg"
        } else if s.contains("roster") {
            "roster"
        } else if matches!(
            s.as_str(),
            "schema-registry" | "catalog" | "catalogs" | "full-source-registry"
        ) {
            "registry / catalog"
        } else if matches!(
            s.as_str(),
            "tombstones" | "gp_distilled" | "retention-plans" | "expiry-manifests"
        ) {
            "retention records (plans, tombstones, distilled tables)"
        } else {
            continue;
        };
        return Some(format!("hard T0: {why} prefix segment {seg:?}"));
    }
    None
}

/// EXPIRE needs all three (§8.4): no training value, a proven E-rule, and an agreeing
/// independent reference check.
pub fn expire_allowed(tv: &Tv, e_rule_proven: bool, independent_check_agrees: bool) -> bool {
    !tv.value && e_rule_proven && independent_check_agrees
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
/// `Tv` is a record for the tv contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Tv {
    /// Measured or derived value carried by this result.
    pub value: bool,
    /// Collection of uses values carried by this tv.
    pub uses: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
/// `Cost` is a record for the cost contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Cost {
    /// Class now recorded for this cost.
    pub class_now: String,
    /// Class after recorded for this cost.
    pub class_after: String,
    /// Now usd mo recorded for this cost.
    pub now_usd_mo: f64,
    /// After usd mo recorded for this cost.
    pub after_usd_mo: f64,
    /// Delta usd mo recorded for this cost.
    pub delta_usd_mo: f64,
    /// One time in region usd recorded for this cost.
    pub one_time_in_region_usd: f64,
    /// The same work run from the xbabe hosts: adds internet egress for every byte read.
    pub one_time_off_aws_usd: f64,
    /// The `after` figure needs the loose originals expired under an approved E1 once the
    /// pack copy is verified; until then storage is `now` plus the pack.
    pub after_needs_e1_of_originals: bool,
    /// An applied lifecycle rule already moves this group to `class_after`.
    pub already_applied_by_lifecycle: bool,
}

/// Everything known about one group before deciding.
pub struct Facts<'a> {
    /// Group recorded for this facts.
    pub group: &'a Group,
    /// Storage class recorded for this facts.
    pub storage_class: String,
    /// Mapping of refs by role values carried by this facts.
    pub refs_by_role: BTreeMap<Role, u64>,
    /// Mapping of ctx roles values carried by this facts.
    pub ctx_roles: BTreeMap<Role, u64>,
    /// Mapping of prog roles values carried by this facts.
    pub prog_roles: BTreeMap<Role, u64>,
    /// Optional index value; `None` means it is not available.
    pub index: Option<&'a GroupIndex>,
    /// Number of family scored rows represented here.
    pub family_scored_rows: Option<u64>,
    /// Coverage recorded for this facts.
    pub coverage: Coverage,
    /// As of unix recorded for this facts.
    pub as_of_unix: u64,
    /// S3 Inventory facts for this group, when its bucket has a report.
    pub inv: Option<&'a GroupInv>,
    /// The inventory's current objects and bytes equal the ledger's, so its exact dates and
    /// classes describe this group (the report is older than the ledger).
    pub inv_fresh: bool,
    /// The inventory and the ledger disagree although the ledger shows no write after the report
    /// (`GroupInv::conflicts_with_ledger`): one input is wrong for this group.
    pub inv_conflict: bool,
    /// Key/size/ETag overlap of this group's E1 candidate pair, when both sides are inventoried.
    pub e1: Option<&'a E1Overlap>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
/// `Decision` is a record for the decision contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Decision {
    /// Action recorded for this decision.
    pub action: Action,
    /// Record kind or category represented by this value.
    pub kind: Option<&'static str>,
    /// Optional family value; `None` means it is not available.
    pub family: Option<String>,
    /// Rule recorded for this decision.
    pub rule: String,
    /// Optional t0 value; `None` means it is not available.
    pub t0: Option<String>,
    /// Live recorded for this decision.
    pub live: bool,
    /// Tv recorded for this decision.
    pub tv: Tv,
    /// Collection of e candidates values carried by this decision.
    pub e_candidates: Vec<String>,
    /// Cost recorded for this decision.
    pub cost: Cost,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} KB", b / 1e3)
    } else {
        format!("{b:.0} B")
    }
}

fn roles_str(m: &BTreeMap<Role, u64>) -> String {
    if m.is_empty() {
        return "0".into();
    }
    m.iter()
        .map(|(r, n)| format!("{}:{n}", r.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

fn is_live(max_last_modified: Option<&str>, as_of: u64, window_h: u64) -> bool {
    max_last_modified
        .and_then(crate::civil::parse_rfc3339)
        .is_some_and(|t| t + window_h * 3600 >= as_of)
}

/// The group's exact last write, when an unchanged inventory listed its objects.
fn exact_max<'a>(f: &Facts<'a>) -> Option<&'a str> {
    if f.inv_fresh {
        f.inv.and_then(|i| i.max_last_modified.as_deref())
    } else {
        None
    }
}

/// $/month now: by the inventory's class mix (scaled to the ledger's bytes) when known.
fn storage_now(f: &Facts, p: &Policy) -> Result<f64> {
    let g = f.group;
    match f.inv.filter(|i| i.current_bytes > 0) {
        Some(i) => {
            let mut usd = 0.0;
            for (class, b) in &i.class_bytes {
                let share = *b as f64 / i.current_bytes as f64;
                usd += p.prices.storage(1_000_000_000, class)? * share * g.bytes as f64 / 1e9;
            }
            Ok(usd)
        }
        None => p.prices.storage(g.bytes, &f.storage_class),
    }
}

fn tv_for(f: &Facts, kind: Option<Kind>, t0: bool, p: &Policy) -> Tv {
    let mut uses = Vec::new();
    if t0 {
        uses.push("future-eval-input");
    }
    let active = |m: &BTreeMap<Role, u64>| m.keys().any(|r| r.is_active() || *r == Role::Val263);
    if active(&f.refs_by_role) || active(&f.ctx_roles) || !f.prog_roles.is_empty() {
        uses.push("ce-target");
    }
    // D1: a prose mention is not a pin, but someone may still want those bytes.
    if f.refs_by_role.contains_key(&Role::ProseMention) {
        uses.push("prose-mention");
    }
    let ix = f.index;
    if ix.is_some_and(|i| i.scored_by_context.values().any(|p| p.len() >= 2)) {
        uses.push("negative-example");
    }
    if kind == Some(Kind::Encoding) || ix.is_some_and(|i| i.by_grain.contains_key("context-hwm")) {
        uses.push("knn-teacher");
    }
    if ix.is_some_and(|i| i.scored_rows > 0) {
        uses.push("reservoir");
    }
    if ix.is_some_and(|i| i.rows > 0)
        && f.family_scored_rows.unwrap_or(0) < p.diversity_min_scored_rows
    {
        uses.push("diversity");
    }
    let calib_grains = ["campaign-summary", "throughput-bench", "s3-object"];
    if matches!(kind, Some(Kind::Telemetry | Kind::Receipts | Kind::Report))
        || ix.is_some_and(|i| calib_grains.iter().any(|g| i.by_grain.contains_key(*g)))
    {
        uses.push("calibration");
    }
    // Unscored rows (terminal receipts, head pointers) and unindexed bodies were never parsed.
    if ix.is_none_or(|i| i.rows > i.scored_rows) {
        uses.push("unknown-fail-closed");
    }
    Tv {
        value: !uses.is_empty(),
        uses,
    }
}

fn e_candidates(f: &Facts, kind: Option<Kind>, p: &Policy) -> Vec<String> {
    let g = f.group;
    let mut out = Vec::new();
    if let Some(c) = p.e1_candidate_for(&g.bucket, &g.prefix) {
        match f.e1 {
            Some(o) => out.push(format!(
                "E1 vs {}: S3 Inventory shows {} of {} keys with the same key, size and ETag ({}; {} differ, {} only here, {} only there); NOT proven: needs GET+sha256 of both copies (ETag is not a content hash); {}",
                c.counterpart,
                o.same_key_size_etag,
                o.objects,
                human_bytes(o.same_key_size_etag_bytes),
                o.same_key_different,
                o.only_here,
                o.only_counterpart,
                c.why
            )),
            None => out.push(format!(
                "E1 vs {}: not provable here (needs inventory key/size join and GET+sha256 of both copies); {}",
                c.counterpart, c.why
            )),
        }
    }
    if let Some(i) = f.inv.filter(|i| i.noncurrent_versions > 0) {
        out.push(format!(
            "E5: {} noncurrent versions ({}) at the inventory date; {}",
            i.noncurrent_versions,
            human_bytes(i.noncurrent_bytes),
            match f.coverage.noncurrent_days {
                Some(d) => format!("the applied lifecycle expires them {d} d after they became noncurrent; the brain adds nothing"),
                None => "no applied noncurrent-expiry rule covers them".into(),
            }
        ));
    }
    match kind {
        Some(Kind::Checkpoint) => out.push(
            "E2: needs the checkpoint->L6 curve in gp_distilled/v1 and an authority-reference scan; keep every Nth, best-by-L6 and final".into(),
        ),
        Some(Kind::Telemetry) => out.push(
            "E3 after distillation: needs distilled rows == raw records and the table hash in a T0 manifest".into(),
        ),
        _ => {}
    }
    if f.index
        .is_some_and(|i| i.by_grain.contains_key("context-terminal"))
    {
        out.push("E4: partial-run receipts; needs program_sha subset of kept programs and no unrecorded score (bodies not indexed)".into());
    }
    out
}

fn cost(f: &Facts, action: Action, p: &Policy) -> Result<Cost> {
    let g = f.group;
    let pr = &p.prices;
    let now = storage_now(f, p)?;
    let packs = g.bytes.div_ceil(p.pack_bytes).max(u64::from(g.objects > 0));
    let gets = g.objects as f64 / 1000.0 * pr.get_per_1k;
    let puts = packs as f64 / 1000.0 * pr.put_per_1k;
    let egress = g.bytes as f64 / 1e9 * pr.egress_per_gb;
    let overhead = |n: u64| -> Result<f64> {
        Ok(
            pr.storage(n * pr.archive_overhead_standard_bytes, "STANDARD")?
                + pr.storage(n * pr.archive_overhead_archive_bytes, "DEEP_ARCHIVE")?,
        )
    };
    let cold_done = f.coverage.cold_class.as_deref() == Some("DEEP_ARCHIVE");
    let (class_after, after, one_time, reads, needs_e1, applied) = match action {
        Action::Cold if f.storage_class == "DEEP_ARCHIVE" => {
            (f.storage_class.clone(), now, 0.0, false, false, cold_done)
        }
        Action::Cold => {
            let after = pr.storage(g.bytes, "DEEP_ARCHIVE")? + overhead(g.objects)?;
            let transitions = g.objects as f64 / 1000.0 * pr.transition_deep_archive_per_1k;
            (
                "DEEP_ARCHIVE".into(),
                after,
                transitions,
                false,
                false,
                cold_done,
            )
        }
        Action::DistillCold => {
            let distilled = pr.storage(g.objects * p.distilled_row_bytes, "STANDARD")?;
            let after = pr.storage(g.bytes, "DEEP_ARCHIVE")? + overhead(packs)? + distilled;
            let transitions = packs as f64 / 1000.0 * pr.transition_deep_archive_per_1k;
            (
                "DEEP_ARCHIVE (packs)".into(),
                after,
                gets + puts + transitions,
                true,
                true,
                false,
            )
        }
        Action::Compact => (
            f.storage_class.clone(),
            now,
            gets + puts,
            true,
            false,
            false,
        ),
        _ => (f.storage_class.clone(), now, 0.0, false, false, false),
    };
    Ok(Cost {
        class_now: f.storage_class.clone(),
        class_after,
        now_usd_mo: round4(now),
        after_usd_mo: round4(after),
        delta_usd_mo: round4(after - now),
        one_time_in_region_usd: round4(one_time),
        one_time_off_aws_usd: round4(one_time + if reads { egress } else { 0.0 }),
        after_needs_e1_of_originals: needs_e1,
        already_applied_by_lifecycle: applied,
    })
}

/// Performs the decide operation and returns an error if it cannot be completed.
pub fn decide(f: &Facts, p: &Policy) -> Result<Decision> {
    let g = f.group;
    let rule = p.rule_for(&g.bucket, &g.prefix);
    let kind = rule.map(|r| r.kind);
    let family = rule.and_then(|r| r.family.clone());
    let rule_src = rule.map_or_else(
        || "default".to_string(),
        |r| format!("policy s3://{}/{}", r.bucket, r.prefix),
    );
    let mean = g.bytes.checked_div(g.objects).unwrap_or(0);
    let exact = exact_max(f);
    let live = is_live(
        exact.or(g.max_last_modified.as_deref()),
        f.as_of_unix,
        p.open_run_window_hours,
    );

    let t0_refs: BTreeMap<Role, u64> = f
        .refs_by_role
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let t0_ctx: BTreeMap<Role, u64> = f
        .ctx_roles
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let t0_prog: BTreeMap<Role, u64> = f
        .prog_roles
        .iter()
        .filter(|(r, _)| r.is_t0())
        .map(|(r, n)| (*r, *n))
        .collect();
    let qualified = f.index.map_or(0, |i| i.qualified_rows);
    let t0 = hard_t0(&g.bucket, &g.prefix)
        .or_else(|| {
            (!t0_refs.is_empty()).then(|| {
                format!(
                    "referenced by key from T0 sources ({})",
                    roles_str(&t0_refs)
                )
            })
        })
        .or_else(|| {
            (!t0_ctx.is_empty())
                .then(|| format!("holds GP rows of T0 contexts ({})", roles_str(&t0_ctx)))
        })
        .or_else(|| {
            (!t0_prog.is_empty()).then(|| {
                format!(
                    "holds programs pinned by T0 sources ({})",
                    roles_str(&t0_prog)
                )
            })
        })
        .or_else(|| (qualified > 0).then(|| format!("holds {qualified} qualified rows")))
        .or_else(|| {
            p.soft_t0_for(&g.bucket, &g.prefix)
                .map(|n| format!("policy soft T0: {}", n.why))
        });
    let tv = tv_for(f, kind, t0.is_some(), p);
    let active_key_refs: BTreeMap<Role, u64> = f
        .refs_by_role
        .iter()
        .filter(|(r, _)| r.is_active())
        .map(|(r, n)| (*r, *n))
        .collect();

    let (action, why): (Action, String) = if let Some(t) = &t0 {
        (
            Action::KeepT0,
            format!("{t}; never delete, no transition proposed"),
        )
    } else if !p.in_scope(&g.bucket) {
        (
            Action::Hold,
            "out of JopeDime retention scope (legacy/other system); its own lifecycle is untouched"
                .into(),
        )
    } else if !active_key_refs.is_empty() {
        (
            Action::KeepHot,
            format!(
                "referenced by key/version from active rosters/seeds ({}); packing or archiving would break those references",
                roles_str(&active_key_refs)
            ),
        )
    } else if let Some(i) = f.inv.filter(|_| f.inv_conflict) {
        (
            Action::Hold,
            format!(
                "the ledger ({} obj, {}, last write {}) and the S3 Inventory report ({} current obj, {}, {} latest delete markers) disagree with no write after the report: one input is wrong, so no change until they agree",
                g.objects,
                human_bytes(g.bytes),
                g.max_last_modified.as_deref().unwrap_or("?"),
                i.current_objects,
                human_bytes(i.current_bytes),
                i.delete_markers_latest
            ),
        )
    } else if live && g.direct && exact.is_none() {
        // A direct group only knows its parent prefix's date range unless an unchanged
        // inventory listed its objects.
        (
            Action::Hold,
            format!(
                "direct objects under a prefix written at {}; their own dates are unknown {}",
                g.max_last_modified.as_deref().unwrap_or("?"),
                if f.inv.is_some() {
                    "(they changed after the S3 Inventory report)"
                } else {
                    "until S3 Inventory"
                }
            ),
        )
    } else if live {
        (
            Action::KeepHot,
            format!(
                "writer active (last write {} within {}h of as-of); classify again after the run closes",
                exact.or(g.max_last_modified.as_deref()).unwrap_or("?"),
                p.open_run_window_hours
            ),
        )
    } else {
        match kind {
            Some(Kind::Encoding) => (Action::KeepHot, "encodings: kNN-teacher / training input, read in place".into()),
            Some(Kind::DatasetPayload) => (Action::KeepHot, "dataset/training payload, read in place".into()),
            Some(Kind::GpResult | Kind::Receipts) if mean >= p.compact_max_mean_object_bytes => (
                Action::KeepHot,
                format!("objects already large (mean {}); packing gains nothing", human_bytes(mean)),
            ),
            Some(Kind::GpResult | Kind::Receipts) => (
                Action::Compact,
                format!(
                    "closed run, mean object {}: pack into >= {} zstd shards + index, keep HWM receipt/program/score loose",
                    human_bytes(mean),
                    human_bytes(p.pack_bytes)
                ),
            ),
            Some(Kind::GpResultUnconsumed | Kind::Telemetry) => (
                Action::DistillCold,
                "no recorded consumer; distill every row (incl. failures) to gp_distilled/v1, then pack raw bytes to Deep Archive; nothing deleted".into(),
            ),
            Some(Kind::Archive) if mean < p.cold_min_mean_object_bytes => (
                Action::DistillCold,
                format!("archive objects small (mean {}): pack before Deep Archive", human_bytes(mean)),
            ),
            Some(Kind::Archive) => (Action::Cold, "opaque archive: Deep Archive (180-day minimum, 12-48 h restore)".into()),
            Some(Kind::Checkpoint) => (
                Action::Hold,
                "checkpoints: no checkpoint->L6/authority linkage yet; frozen v1/promoted must stay T0 (§7.3), so no transition".into(),
            ),
            Some(Kind::Report) => (Action::Hold, "research/report bytes: T3 only after the report is sealed (strategy §3)".into()),
            Some(Kind::Replica) => (Action::Hold, "replica: keep/drop is owner decision 6".into()),
            Some(Kind::Custody) => (Action::Hold, "preservation custody: dev/prod duplicate check is owner decision 6".into()),
            Some(Kind::Review) => (Action::Hold, "contents not classified; review before any action".into()),
            None => (Action::Hold, "no policy rule matched; default HOLD + review (§7.3 rule 6)".into()),
        }
    };
    let e = e_candidates(f, kind, p);
    let cost = cost(f, action, p)?;

    let mut reason = format!(
        "{}: {} [{}{}] {} obj, {}, mean {}; {}",
        action.as_str(),
        g.id(),
        kind.map_or("unclassified", Kind::as_str),
        family
            .as_deref()
            .map(|s| format!("/{s}"))
            .unwrap_or_default(),
        g.objects,
        human_bytes(g.bytes),
        human_bytes(mean),
        why
    );
    reason.push_str(&format!(
        "; TV={} ({})",
        tv.value,
        if tv.uses.is_empty() {
            "none".into()
        } else {
            tv.uses.join(",")
        }
    ));
    reason.push_str(&format!("; refs {}", roles_str(&f.refs_by_role)));
    if let Some(i) = f.index {
        reason.push_str(&format!(
            "; index {} rows ({} scored, {} qualified)",
            i.rows, i.scored_rows, i.qualified_rows
        ));
    }
    reason.push_str("; champion: no qualified champion, so no supersession");
    if !f.coverage.full.is_empty() {
        let rules: Vec<&str> = f
            .coverage
            .full
            .iter()
            .map(|s| s.split(':').next().unwrap_or(""))
            .collect();
        reason.push_str(&format!("; lifecycle {}", rules.join(",")));
    }
    if let Some(i) = f.inv {
        reason.push_str(&format!(
            "; inventory {} ({} noncurrent versions{})",
            if f.inv_fresh {
                "unchanged since report"
            } else if f.inv_conflict {
                "disagrees with the ledger without a later write"
            } else {
                "changed since report"
            },
            i.noncurrent_versions,
            exact
                .filter(|_| g.direct)
                .map(|m| format!(", direct objects last written {m}"))
                .unwrap_or_default()
        ));
        if !e.is_empty() {
            reason.push_str("; no E-rule proven (E1 needs GET+sha256; E2-E4 need gp_distilled/v1)");
        }
    } else if !e.is_empty() {
        reason.push_str("; E-rules not provable at prefix level");
    }
    Ok(Decision {
        action,
        kind: kind.map(Kind::as_str),
        family,
        rule: rule_src,
        t0,
        live,
        tv,
        e_candidates: e,
        cost,
        reason,
    })
}
