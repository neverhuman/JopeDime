use super::*;

pub(crate) fn human_bytes(b: u64) -> String {
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

pub(crate) fn roles_str(m: &BTreeMap<Role, u64>) -> String {
    if m.is_empty() {
        return "0".into();
    }
    m.iter()
        .map(|(r, n)| format!("{}:{n}", r.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn is_live(max_last_modified: Option<&str>, as_of: u64, window_h: u64) -> bool {
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
pub(crate) fn storage_now(f: &Facts, p: &Policy) -> Result<f64> {
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

pub(crate) fn tv_for(f: &Facts, kind: Option<Kind>, t0: bool, p: &Policy) -> Tv {
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

pub(crate) fn e_candidates(f: &Facts, kind: Option<Kind>, p: &Policy) -> Vec<String> {
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

pub(crate) fn cost(f: &Facts, action: Action, p: &Policy) -> Result<Cost> {
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
