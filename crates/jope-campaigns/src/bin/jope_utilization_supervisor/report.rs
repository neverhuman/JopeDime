use super::*;

// ---------------------------------------------------------------------------
// RECORD — historical utilization
// ---------------------------------------------------------------------------

pub(super) fn report(args: &Args, hours: f64) -> Result<()> {
    let path = args.series();
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let cutoff = now_ms().saturating_sub((hours * 3_600_000.0) as u64);
    let mut rows: Vec<Value> = String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|row| u64_at(row, "ts_ms").unwrap_or(0) >= cutoff)
        .collect();
    rows.sort_by_key(|row| u64_at(row, "ts_ms").unwrap_or(0));
    if rows.is_empty() {
        println!("no samples in the last {hours} h ({})", path.display());
        return Ok(());
    }
    let samples = rows.len() as f64;
    let mean_core: f64 = rows
        .iter()
        .filter_map(|r| f64_at(r, "core_pct"))
        .sum::<f64>()
        / samples;
    let mean_slots: f64 = rows
        .iter()
        .filter_map(|r| f64_at(r, "slot_pct"))
        .sum::<f64>()
        / samples;
    let mean_running: f64 = rows
        .iter()
        .filter_map(|r| u64_at(r, "jobs_running"))
        .sum::<u64>() as f64
        / samples;

    // Contiguous runs of non-ok verdicts, which is what "idle gap" means.
    let mut gaps: Vec<(u64, u64, String)> = Vec::new();
    for row in &rows {
        let ts = u64_at(row, "ts_ms").unwrap_or(0);
        let verdict = str_at(row, "verdict").unwrap_or("?").to_owned();
        let bad = verdict == "idle" || verdict == "idle_no_work";
        match gaps.last_mut() {
            Some(last) if bad && last.2 == verdict && ts.saturating_sub(last.1) <= 600_000 => {
                last.1 = ts;
            }
            _ if bad => gaps.push((ts, ts, verdict)),
            _ => {}
        }
    }
    println!(
        "utilization report  window={hours}h  samples={}",
        rows.len()
    );
    println!(
        "  first sample  {}",
        iso_utc(u64_at(&rows[0], "ts_ms").unwrap_or(0))
    );
    println!(
        "  last sample   {}",
        iso_utc(u64_at(rows.last().unwrap(), "ts_ms").unwrap_or(0))
    );
    println!("  mean core-budget occupancy  {mean_core:.1}%");
    println!("  mean slot occupancy         {mean_slots:.1}%");
    println!("  mean running jobs           {mean_running:.0}");
    let ok = rows
        .iter()
        .filter(|r| str_at(r, "verdict") == Some("ok"))
        .count();
    println!("  cycles ok / total           {ok}/{}", rows.len());
    if gaps.is_empty() {
        println!("  idle gaps: none");
    } else {
        println!("  idle gaps ({}):", gaps.len());
        for (start, end, verdict) in &gaps {
            println!(
                "    {} -> {}  {:>5} min  [{verdict}]",
                iso_utc(*start),
                iso_utc(*end),
                (end.saturating_sub(*start)) / 60_000
            );
        }
    }
    Ok(())
}

pub(super) fn human_span(ms: u64) -> String {
    let secs = ms / 1000;
    if secs < 120 {
        format!("{secs}s")
    } else if secs < 7200 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}
