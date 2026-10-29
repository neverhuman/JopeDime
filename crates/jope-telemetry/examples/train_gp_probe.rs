//! Real-data probe for the TRAIN+GP modules.
//!
//! ```text
//! train_gp_probe parity <parse_parity.jsonl>
//! train_gp_probe import <flywheel_root>
//! ```
//!
//! `parity` checks `gp::parse_pipe_dsl` and `gp::program_sha256` against rows written by the
//! committed Python `parse_pipe_dsl`. `import` runs every adapter over a flywheel output tree
//! (`results/flywheel_selfplay_report.json`, `s3_mirror/round-*/`) and prints one JSON line per
//! record kind, plus the A2 collapse indicators per round.

use anyhow::{bail, Context, Result};
use jope_telemetry::{flywheel, gp, import, summary};
use serde_json::{json, Value};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("parity") => parity(Path::new(
            args.get(2).context("parity needs a fixture path")?,
        )),
        Some("import") => import_tree(Path::new(
            args.get(2).context("import needs a flywheel root")?,
        )),
        _ => bail!("usage: train_gp_probe parity <fixture.jsonl> | import <flywheel_root>"),
    }
}

fn parity(fixture: &Path) -> Result<()> {
    let text = std::fs::read_to_string(fixture)?;
    let (mut n, mut parse_bad, mut sha_bad) = (0usize, 0usize, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Value = serde_json::from_str(line)?;
        let dsl = row["dsl"].as_str().context("row without dsl")?;
        let want_stages: Vec<&str> = row["stages"]
            .as_array()
            .context("row without stages")?
            .iter()
            .map(|v| v.as_str().unwrap_or_default())
            .collect();
        let want_head = row["head"].as_str().context("row without head")?;
        let (stages, head) = gp::parse_pipe_dsl(dsl);
        if stages != want_stages || head != want_head {
            parse_bad += 1;
            if parse_bad <= 5 {
                eprintln!("parse mismatch: {dsl:?}\n  rust   {stages:?} {head:?}\n  python {want_stages:?} {want_head:?}");
            }
        }
        let sha: String = gp::program_sha256(dsl)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if Some(sha.as_str()) != row["program_sha256"].as_str() {
            sha_bad += 1;
        }
        n += 1;
    }
    println!(
        "{}",
        json!({"probe": "parse_parity", "rows": n, "parse_mismatches": parse_bad, "sha_mismatches": sha_bad})
    );
    if parse_bad + sha_bad > 0 {
        bail!("parity failed");
    }
    Ok(())
}

fn import_tree(root: &Path) -> Result<()> {
    // Every record also goes to the TRAIN+GP summary at the end.
    let mut recs: Vec<(String, Value)> = Vec::new();
    let report = import::selfplay_report(&root.join("results/flywheel_selfplay_report.json"))?;
    for r in &report.rounds {
        recs.push((flywheel::ROUND_SCHEMA.into(), serde_json::to_value(r)?));
        println!(
            "{}",
            json!({"kind": "round", "round": r.round, "wall_s": r.wall_s, "target_deltas": r.target_deltas,
                   "selfplay_win_rate": r.selfplay_win_rate, "n_arm_contrasts": r.arm_contrasts.len()})
        );
    }
    for q in &report.queue {
        let keep = [
            "optimizer_steps",
            "param_update_l2_norm",
            "mean_popped_batch_age_ms",
            "batches_instant_fallback",
            "total_improved_children",
        ];
        let v: serde_json::Map<String, Value> = keep
            .iter()
            .filter_map(|k| Some(((*k).to_owned(), json!(q.values.get(*k)?))))
            .collect();
        println!(
            "{}",
            json!({"kind": "queue", "round": q.round, "arm": q.arm, "values": v, "availability": q.availability})
        );
    }
    let mut rounds: Vec<_> = std::fs::read_dir(root.join("s3_mirror"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("round-"))
        })
        .collect();
    rounds.sort();
    for dir in rounds {
        let round: u32 = dir.file_name().unwrap().to_string_lossy()[6..].parse()?;
        let mut hwm = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir.join("hwm")) {
            for e in rd.flatten() {
                hwm.push(import::hwm_file(&e.path())?);
            }
        }
        let improvements: Vec<f64> = hwm.iter().filter_map(|h| h.fitness_improvement).collect();
        let missing: std::collections::BTreeSet<&String> = hwm
            .iter()
            .flat_map(|h| &h.missing_identity_fields)
            .collect();
        println!(
            "{}",
            json!({"kind": "hwm", "round": round, "beats": hwm.len(),
                   "improvement_sum": improvements.iter().sum::<f64>(), "missing_identity_fields": missing})
        );
        for e in std::fs::read_dir(dir.join("gp_distilled"))?.flatten() {
            let (_, ret) = import::distilled_round(&e.path())?;
            println!(
                "{}",
                json!({"kind": "retention", "round": ret.round, "counts": ret.counts})
            );
            let c = import::hwm_consistency(round, &hwm, &ret, 60);
            println!("{}", serde_json::to_string(&c)?);
            recs.push((
                import::HWM_CONSISTENCY_SCHEMA.into(),
                serde_json::to_value(&c)?,
            ));
            recs.push((import::RETENTION_SCHEMA.into(), serde_json::to_value(&ret)?));
        }
        for e in std::fs::read_dir(dir.join("scores"))?.flatten() {
            let g = import::retained_scores(&e.path(), round, None)?;
            recs.push((gp::GEN_SCHEMA.into(), serde_json::to_value(&g)?));
            println!(
                "{}",
                json!({"kind": "retained_generation", "round": round, "pop": g.pop_size,
                       "unique_genotype_frac": g.unique_genotype_frac, "edit_distance_mean": g.edit_distance_mean,
                       "edit_distance_p10": g.edit_distance_p10, "op_entropy_nats": g.op_entropy_nats,
                       "ops_distinct": g.ops_distinct, "passers": g.status.gate_passed, "gate_failed": g.status.gate_failed,
                       "fitness_passers": g.fitness_passers, "edit_ops": g.edit_ops})
            );
        }
    }
    let sum = summary::summarize(
        std::iter::empty(),
        recs.iter().map(|(k, v)| (k.as_str(), v)),
    );
    println!("{}", json!({"kind": "summary", "summary": sum}));
    eprintln!("{}", sum.render_text());
    Ok(())
}
