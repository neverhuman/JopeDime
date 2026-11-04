pub fn phase_label(phase: u8) -> &'static str {
    match phase {
        0 => "HYPERION",
        2 => "IMPORT",
        3 => "SELECT",
        4 => "MIMO",
        5 => "FINAL",
        6 => "INVENT",
        _ => "PIPE",
    }
}

fn is_immediate(e: &Event) -> bool {
    matches!(
        e.event.as_str(),
        "start" | "done" | "model" | "warning" | "warn" | "seed"
    )
}

fn progress_line(
    e: &Event,
    context: &RunContext,
    phase: &str,
    phase_elapsed: Option<f64>,
) -> String {
    let mut parts = vec![format!("[{phase}] {}", event_label(e))];
    if e.event == "seed" {
        // native jope inference seed for GP gen-0; #1 = the rank-0 (selection_key argmax). The
        // human line carries the seed count only; seed program text is never printed.
        parts.push(format!(
            "jope inference · seed #{}",
            e.gen.map(|g| g + 1).unwrap_or(1)
        ));
    }
    if e.event == "start" {
        context_parts(context, &mut parts);
    }
    if e.event != "seed" {
        if let Some(model) = &e.model {
            parts.push(format!("model={model}"));
        }
    }
    if let Some(metric) = &e.metric {
        parts.push(format!("metric={metric}"));
    }
    if let Some(status) = &e.hyperion_status {
        parts.push(format!("status={status}"));
    }
    if let Some(engine) = &e.hyperion_engine {
        parts.push(format!("engine={engine}"));
    }
    if let Some(device) = &e.hyperion_device {
        parts.push(format!("device={device}"));
    }
    if let Some(rows) = e.rows {
        parts.push(format!("rows={rows}"));
    }
    if let Some(cols) = e.cols {
        parts.push(format!("cols={cols}"));
    }
    if let Some(dim) = e.output_dim {
        parts.push(format!("output_dim={dim}"));
    }
    if let Some(preview) = &e.hyperion_preview {
        parts.push(format!(
            "preview_points={} concerns={}",
            preview.points.len(),
            preview.concern_ranges.len()
        ));
    }
    if let (Some(gen), Some(gens)) = (e.gen, e.gens) {
        parts.push(format!("gen={}/{}", gen + 1, gens));
    } else if let Some(gens) = e.gens {
        parts.push(format!("gens={gens}"));
    }
    if let (Some(done), Some(total)) = (e.done, e.blocks_total) {
        let label = if e.phase == 2 {
            "blocks"
        } else if e.event == "genprog" {
            "scored"
        } else {
            "done"
        };
        let frac = e
            .done_frac
            .unwrap_or_else(|| done as f64 / total.max(1) as f64);
        parts.push(format!("{label}={done}/{total} {}", bar(frac)));
        let elapsed = phase_elapsed_for_rate(e, phase_elapsed);
        if done > 0 && elapsed > 0.0 {
            parts.push(format!("rate={:.1}/s", done as f64 / elapsed));
        }
    }
    if let Some(eta) = e.eta {
        parts.push(format!("eta={}", fmt_secs(eta.max(0.0))));
    }
    if let Some(accepted) = e.accepted_features {
        parts.push(format!("accepted={accepted}"));
    }
    if let Some(selected) = e.selected_features.or(e.best_size) {
        parts.push(format!("selected={selected}"));
    }
    if let Some(final_features) = e.final_features {
        parts.push(format!("final_features={final_features}"));
    }
    if let Some(top) = &e.top {
        let top = top
            .iter()
            .take(3)
            .map(|(idx, pct)| format!("#{idx}:{pct:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        if !top.is_empty() {
            parts.push(format!("top={top}"));
        }
    }
    if let Some(cv) = e.best_val_auc {
        parts.push(format!("cv={cv:.5}"));
    }
    if let Some(holdout) = e.best_holdout_auc {
        parts.push(format!("holdout={holdout:.5}"));
    }
    if let Some(mean) = e.mean_val_auc {
        parts.push(format!("mean_cv={mean:.5}"));
    }
    if let Some(size) = e.mean_size {
        parts.push(format!("mean_size={size}"));
    }
    if let Some(elapsed) = phase_elapsed {
        parts.push(format!("elapsed={}", fmt_secs(elapsed)));
    }
    parts.push(format!("t={}", fmt_secs(e.t)));
    parts.join(" ")
}

fn event_label(e: &Event) -> &'static str {
    match (e.event.as_str(), e.model.as_ref()) {
        ("model", _) => "candidate",
        ("done", Some(_)) if e.phase == 5 => "selected",
        ("genprog", _) => "progress",
        (other, _) => match other {
            "start" => "start",
            "done" => "done",
            "gen" => "gen",
            "block" => "progress",
            "warning" | "warn" => "warning",
            "seed" => "seed",
            _ => "event",
        },
    }
}

fn context_parts(context: &RunContext, parts: &mut Vec<String>) {
    if let Some(command) = &context.command {
        parts.push(format!("cmd={command}"));
    }
    if let Some(dataset) = &context.dataset {
        parts.push(format!("dataset={dataset}"));
    }
    if let Some(task) = &context.task {
        parts.push(format!("task={task}"));
    }
    if let Some(rows) = context.rows {
        parts.push(format!("rows={rows}"));
    }
    if let Some(hold_rows) = context.hold_rows {
        parts.push(format!("hold={hold_rows}"));
    }
    if let Some(features) = context.features {
        parts.push(format!("features={features}"));
    }
    if let Some(mode) = &context.selection_mode {
        parts.push(format!("selection={mode}"));
    }
}

fn phase_elapsed_for_rate(e: &Event, phase_elapsed: Option<f64>) -> f64 {
    phase_elapsed.unwrap_or(e.t).max(1e-9)
}

fn fmt_secs(secs: f64) -> String {
    if secs >= 60.0 {
        let total = secs.round() as u64;
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        format!("{secs:.1}s")
    }
}

fn bar(frac: f64) -> String {
    let width = 18usize;
    let fill = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!(
        "[{}{}] {:>3.0}%",
        "#".repeat(fill),
        ".".repeat(width.saturating_sub(fill)),
        100.0 * frac.clamp(0.0, 1.0)
    )
}
