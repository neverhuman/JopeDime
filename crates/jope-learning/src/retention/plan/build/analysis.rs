use super::state::{Analysis, Evaluated};
use super::*;

pub(super) fn analyze(evaluated: &Evaluated) -> Result<Analysis> {
    let loaded = &evaluated.loaded;
    let rows = &evaluated.rows;
    let inv = &evaluated.inv;
    let policy = &loaded.policy;
    let ledger = &loaded.ledger;
    let agg = &loaded.agg;
    let refs = &loaded.refs;
    let pins = &loaded.pins;
    let pin_sources = &loaded.pin_sources;
    let known_buckets: BTreeSet<&str> = ledger.groups.iter().map(|g| g.bucket.as_str()).collect();
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
    let inventory_summary = match &inv {
        None => None,
        Some(i) => {
            let (mut nc_t0, mut nc_other, mut with, mut unchanged) = (0u64, 0u64, 0u64, 0u64);
            let mut ledger_conflicts = Vec::new();
            for r in &rows {
                let Some(x) = &r.inventory else { continue };
                with += 1;
                unchanged += u64::from(x.unchanged_since_report);
                if x.ledger_conflict {
                    add_note(
                        &mut hazards,
                        "ledger-inventory-conflict",
                        format!(
                            "{}: ledger and S3 Inventory disagree with no write after the report (one input is wrong; these groups get no changing action)",
                            r.bucket
                        ),
                        r.bytes as f64 / 1e9,
                    );
                    ledger_conflicts.push(format!(
                        "{} ({}): ledger {} obj / {} B, last write {}; inventory {} current obj / {} B, {} latest delete markers",
                        r.group_id,
                        r.action.as_str(),
                        r.objects,
                        r.bytes,
                        r.max_last_modified.as_deref().unwrap_or("?"),
                        x.facts.current_objects,
                        x.facts.current_bytes,
                        x.facts.delete_markers_latest
                    ));
                }
                if r.action == Action::KeepT0 {
                    nc_t0 += x.facts.noncurrent_bytes;
                    if x.facts.noncurrent_versions > 0 {
                        add_note(
                            &mut notes,
                            "t0-noncurrent-versions",
                            format!(
                                "{}: noncurrent versions inside T0 groups at the inventory date; the applied rule expires them {} d after they became noncurrent (owner ruling D2 2026-10-13: 90 d for T0 too)",
                                r.bucket,
                                r.lifecycle
                                    .noncurrent_days
                                    .map_or("never".into(), |d| d.to_string())
                            ),
                            x.facts.noncurrent_bytes as f64 / 1e9,
                        );
                    }
                } else {
                    nc_other += x.facts.noncurrent_bytes;
                }
            }
            // Pins: a T0 or active ref whose key is not a current object is a hazard, unless the
            // group changed after the report (the key may be newer than the listing).
            let mut pin_counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
            let mut sample = Vec::new();
            for (role, bucket, key, gi) in &pins {
                let Some(b) = i.buckets.get(bucket) else {
                    continue;
                };
                let state = i
                    .pins
                    .get(&(bucket.clone(), key.clone()))
                    .copied()
                    .unwrap_or(PinState::Absent);
                let state_s = serde_json::to_value(state)?
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                *pin_counts
                    .entry(role.as_str().into())
                    .or_default()
                    .entry(state_s.clone())
                    .or_default() += 1;
                if state == PinState::Present {
                    continue;
                }
                let g = &ledger.groups[*gi];
                let newer = g
                    .max_last_modified
                    .as_deref()
                    .is_some_and(|m| m > b.report_created_at.as_str());
                let (map, kind) = if newer {
                    (&mut notes, "pin-not-in-inventory-group-changed-since")
                } else if role.is_t0() {
                    (&mut hazards, "t0-pin-not-present")
                } else if role.is_active() {
                    (&mut hazards, "active-pin-not-present")
                } else {
                    (&mut notes, "mention-not-present")
                };
                add_note(
                    map,
                    kind,
                    format!("{bucket}: {} refs whose key is {state_s}", role.as_str()),
                    0.0,
                );
                if sample.len() < 200 {
                    sample.push(format!(
                        "{} {state_s} s3://{bucket}/{key}{} (source {})",
                        role.as_str(),
                        if newer {
                            " (group changed since report)"
                        } else {
                            ""
                        },
                        pin_sources
                            .get(&(*role, bucket.clone(), key.clone()))
                            .map_or("?", String::as_str)
                    ));
                }
            }
            let mut ledger_totals: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
            for g in &ledger.groups {
                let e = ledger_totals.entry(g.bucket.as_str()).or_default();
                e.0 += g.objects;
                e.1 += g.bytes;
            }
            let buckets = i
                .buckets
                .iter()
                .map(|(name, t)| {
                    let (lo, lb) = ledger_totals.get(name.as_str()).copied().unwrap_or((0, 0));
                    InventoryBucket {
                        bucket: name.clone(),
                        report_created_at: t.report_created_at.clone(),
                        rows: t.rows,
                        current_objects: t.current_objects,
                        current_bytes: t.current_bytes,
                        noncurrent_versions: t.noncurrent_versions,
                        noncurrent_bytes: t.noncurrent_bytes,
                        delete_markers: t.delete_markers,
                        rows_outside_ledger_groups: t.rows_outside_ledger_groups,
                        ledger_objects: lo,
                        ledger_bytes: lb,
                        ledger_minus_inventory_objects: lo as i64 - t.current_objects as i64,
                        ledger_minus_inventory_bytes: lb as i64 - t.current_bytes as i64,
                    }
                })
                .collect();
            Some(InventorySummary {
                buckets,
                ledger_buckets_without_report: known_buckets
                    .iter()
                    .filter(|b| policy.in_scope(b) && !i.buckets.contains_key(**b))
                    .map(|b| b.to_string())
                    .collect(),
                groups_with_inventory: with,
                groups_unchanged_since_report: unchanged,
                ledger_conflicts,
                noncurrent_bytes_in_t0_groups: nc_t0,
                noncurrent_bytes_in_other_groups: nc_other,
                pins: pin_counts,
                pins_not_present_sample: sample,
                e1_overlap: i.e1.clone(),
            })
        }
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

    Ok(Analysis {
        hazards,
        notes,
        inventory_summary,
        findings,
        plan_jsonl,
        plan_sha,
    })
}
