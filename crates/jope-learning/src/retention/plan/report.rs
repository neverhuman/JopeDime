use super::*;

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
        index: f.index.map(super::super::index::GroupIndex::summary),
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
