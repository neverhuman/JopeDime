use super::evidence::{
    completed, decimal, number, panel, phase, rmse, status_age, status_age_with_limit,
};
use super::prelude::*;
use super::tasks::label;

pub(super) fn binary_replay_detail(run: &Value) -> String {
    format!(
        "{} admitted · {} bindings verified · {} exact replay pairs · {} raw components recomputed",
        decimal_count(run, "binary_admitted_contexts"),
        decimal_count(run, "binary_verified_contexts"),
        decimal_count(&run["binary_verification"], "verified_pairs"),
        decimal_count(
            &run["binary_verification"],
            "independently_computed_raw_components"
        )
    )
}
pub(super) fn comparison_lines(run: &Value) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(
        "POLICY VALUE · MATCHED SEARCH BUDGETS",
        Style::default().fg(VIOLET).bold(),
    )];
    if run["experiment_manifest"].is_string() {
        lines.push(Line::raw(format!(
            "Experiment: {}",
            run["experiment_manifest"].as_str().unwrap_or("unavailable")
        )));
        lines.push(Line::raw(format!(
            "Production selected: {}",
            run["production_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
        )));
        lines.push(Line::raw(format!(
            "Experiment frozen: {}",
            run["experiment_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
        )));
    }
    let matched = &run["matched_quality"];
    if matched.is_object() {
        lines.push(Line::raw(format!(
            "Matched repeats: {} / {} uninterrupted blocks · {} interrupted preserved",
            number(matched, "completed_blocks"),
            number(matched, "required_blocks"),
            number(matched, "interrupted_blocks_preserved")
        )));
        for seed in [20260907, 20260908, 20260909] {
            let key = format!("{seed}:larger_pool_vs_pure_first64");
            lines.push(Line::raw(format!(
                "Seed {seed}: larger / pure_first64 {} · {} / {} families",
                decimal(&matched["comparisons"][&key], "family_balanced_ratio"),
                number(&matched["comparisons"][&key], "families_completed"),
                number(&matched["comparisons"][&key], "families_required")
            )));
        }
        lines.push(Line::raw(format!(
            "Evaluated fits: {} · selector remains undeployed",
            number(matched, "successful_fits_all_attempts")
        )));
        lines.push(Line::raw(format!(
            "Evaluated teachers: {} · selected-seed wins: {} / {}",
            number(matched, "evaluated_teachers_all_attempts"),
            number(
                &matched["attributed_improvements"]["pure_first64"],
                "neural_selection_wins"
            ),
            number(
                &matched["attributed_improvements"]["generic"],
                "neural_selection_wins"
            )
        )));
    }
    if run["binary_admitted_contexts"].is_number() {
        let verified = &run["binary_verification"];
        lines.push(Line::raw(format!("Binary: {} admitted · {} bindings verified · {} exact replay pairs · {} raw components recomputed",run["binary_admitted_contexts"],decimal_count(run,"binary_verified_contexts"),number(verified,"verified_pairs"),number(verified,"independently_computed_raw_components"))));
    }
    let experiment = &run["seed_quality"];
    if experiment.is_object() {
        let frozen_cycle = experiment["selected_checkpoint"]
            .as_str()
            .and_then(|path| path.split('/').find(|part| part.starts_with("cycle-")))
            .unwrap_or("checkpoint");
        lines[0] = Line::styled(
            format!("PURE SEED QUALITY · FROZEN {frozen_cycle} · NO TEACHERS"),
            Style::default().fg(VIOLET).bold(),
        );
        for (key, label) in [
            ("larger_pool_vs_pure_first64", "larger / pure_first64"),
            ("larger_pool_vs_generic", "larger / generic"),
        ] {
            let legacy = if key == "larger_pool_vs_pure_first64" {
                "larger_pool_vs_current"
            } else {
                key
            };
            let best = if experiment["comparisons"][key].is_object() {
                &experiment["comparisons"][key]["best8"]
            } else {
                &experiment["comparisons"][legacy]["best8"]
            };
            lines.push(Line::raw(format!(
                "Best8 {label}: {} · {}W / {}T / {}L · {} / {} families",
                decimal(best, "family_balanced_ratio"),
                number(best, "wins"),
                number(best, "ties"),
                number(best, "losses"),
                number(best, "families_completed"),
                number(best, "families_required")
            )));
        }
        let fit_key = if experiment["top8_physical_fits"].is_number() {
            "top8_physical_fits"
        } else {
            "physical_fits"
        };
        let fail_key = if experiment["top8_failures"].is_number() {
            "top8_failures"
        } else {
            "failures"
        };
        lines.push(Line::raw(format!(
            "Top8: {} fits · {} logical evals · {} failures",
            decimal_count(experiment, fit_key),
            decimal_count(experiment, "logical_evaluations"),
            decimal_count(experiment, fail_key)
        )));
        let downstream = &experiment["downstream"];
        if downstream.is_object() {
            lines.push(Line::raw(format!(
                "GP {} / {} · larger/pure_first64 {} · larger/generic {}",
                number(downstream, "completed_views"),
                number(downstream, "required_views"),
                decimal(
                    &downstream["comparisons"]["current"],
                    "family_balanced_ratio"
                ),
                decimal(
                    &downstream["comparisons"]["generic"],
                    "family_balanced_ratio"
                )
            )));
            lines.push(Line::raw(format!(
                "Selected-seed wins {} vs pure_first64 / {} vs generic; inherited/evolved excluded",
                number(
                    &downstream["direct_neural_selected_winner_improvements"],
                    "current"
                ),
                number(
                    &downstream["direct_neural_selected_winner_improvements"],
                    "generic"
                )
            )));
        }
        lines.push(Line::styled(
            experiment["conclusion"]
                .as_str()
                .unwrap_or("Comparison incomplete; existing production seeder retained")
                .to_owned(),
            Style::default().fg(if experiment["positive_meeting_result"] == true {
                GREEN
            } else {
                GOLD
            }),
        ));
        lines.push(Line::raw(
            "Qualification within previously used families; production seeder unchanged.",
        ));
        return lines;
    }
    for report in run["reports"]
        .as_array()
        .into_iter()
        .flatten()
        .rev()
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let val = &report["validation"];
        let comparisons = &val["evaluation_prefix"];
        lines.push(Line::styled(
            report["cycle"].as_str().unwrap_or("Saved cycle").to_owned(),
            Style::default().fg(INK).bold(),
        ));
        if report["raw_final_scores_only"] == true {
            lines.push(Line::styled(
                "  RAW FINAL SCORES · unequal completed budgets",
                Style::default().fg(GOLD),
            ));
            let rows = report["comparisons"].as_array();
            if let Some(row) = rows
                .into_iter()
                .flatten()
                .find(|v| v["partition"] == "meta_validation")
                .or_else(|| rows.and_then(|v| v.first()))
            {
                lines.push(Line::raw(format!(
                    "  {} cut{} · NN {} / generic {} / shuffled {}",
                    row["family"].as_str().unwrap_or("family"),
                    number(row, "cut"),
                    rmse(&row["neural"], "best_validation_rmse"),
                    rmse(&row["generic_retained"], "best_validation_rmse"),
                    rmse(&row["shuffled_context"], "best_validation_rmse")
                )));
                lines.push(Line::raw(format!(
                    "  Eval budgets {}/{}/{} · matched-prefix report pending",
                    number(&row["neural"], "completed_evaluations"),
                    number(&row["generic_retained"], "completed_evaluations"),
                    number(&row["shuffled_context"], "completed_evaluations")
                )));
            }
            continue;
        }
        lines.push(Line::raw(format!(
            "  NN / generic {} · NN / shuffled {} · {} families",
            decimal(
                &comparisons["neural_over_generic"],
                "family_balanced_rmse_ratio"
            ),
            decimal(
                &comparisons["neural_over_shuffled"],
                "family_balanced_rmse_ratio"
            ),
            number(val, "source_families")
        )));
        lines.push(Line::raw(format!(
            "  {} searches · {} teacher records · {} fitted predictors",
            number(report, "searches"),
            number(report, "records"),
            number(report, "predictors")
        )));
        let wall = &val["wall_prefix_bounds"]["neural_over_generic"];
        if wall["rmse_ratio_lower"].is_number() {
            lines.push(Line::raw(format!(
                "  Wall-matched NN/generic bounds {}–{}",
                decimal(wall, "rmse_ratio_lower"),
                decimal(wall, "rmse_ratio_upper")
            )));
        }
    }
    if run["reports"].as_array().is_none_or(|v| v.is_empty()) {
        lines.push(Line::raw(
            "Awaiting a measured three-arm comparison report.",
        ));
    }
    lines.push(Line::styled(
        "Lower RMSE ratio is better. Policy advantage is not established.",
        Style::default().fg(GOLD),
    ));
    lines.push(Line::raw(
        "Adaptive held-out teaching scores; original final tests stay sealed.",
    ));
    lines
}
pub(super) fn teacher_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let boot = &run["bootstrap"];
    let mut lines = vec![
        Line::styled(
            "BEST GP + TRAINING ENCODING → POLICY → NEXT GP",
            Style::default().fg(GREEN).bold(),
        ),
        Line::raw("Improve the teacher on each cut, then learn its program choices."),
        Line::raw("A stronger teacher and a stronger policy are measured separately."),
        Line::raw(""),
    ];
    for (key, label) in [
        ("meta_train", "Training families"),
        ("meta_validation", "Validation families"),
        ("meta_test", "Test families · sealed"),
    ] {
        let v = &boot["partitions"][key];
        lines.push(self::label(
            label,
            format!(
                "{} / {} with completed GP views",
                number(v, "families_with_completed_view"),
                number(v, "selected_families")
            ),
        ));
    }
    lines.push(label(
        "Coverage receipt",
        if boot["compact_receipt_ready"] == true {
            "coverage ready · saved receipt".into()
        } else {
            status_age(boot)
        },
    ));
    lines.push(label(
        "Best-teacher coverage",
        teacher["cuts_with_teacher"]
            .as_u64()
            .map(|v| format!("{v} unique cuts"))
            .unwrap_or_else(|| "awaiting compact teacher receipt".into()),
    ));
    lines.push(label(
        "New best programs",
        teacher["new_bests"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unavailable".into()),
    ));
    lines.push(label(
        "Teacher telemetry",
        status_age_with_limit(teacher, 15),
    ));
    lines
}
pub(super) fn teacher_evolution_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let mut lines = vec![
        Line::styled(
            "BEST FOUND · EXACT CUT / EVALUATOR IDENTITY",
            Style::default().fg(GREEN).bold(),
        ),
        label(
            "Teacher coverage",
            teacher["cuts_with_teacher"]
                .as_u64()
                .map(|n| {
                    format!(
                        "{n} unique cuts · {} identities",
                        number(teacher, "evaluator_cut_identities")
                    )
                })
                .unwrap_or_else(|| "awaiting durable champion receipt".into()),
        ),
        label(
            "New best / improving cuts",
            format!(
                "{} / {}",
                decimal_count(teacher, "new_bests"),
                decimal_count(teacher, "matched_cut_improvements")
            ),
        ),
        label(
            "Champion reuse",
            teacher["champion_reuse"]["status"]
                .as_str()
                .unwrap_or("unavailable"),
        ),
    ];
    for item in teacher["top_improvements"]
        .as_array()
        .into_iter()
        .flatten()
        .take(2)
    {
        let family = item["family"]
            .as_str()
            .or_else(|| item["dataset"].as_str())
            .unwrap_or("cut");
        lines.push(Line::raw(format!(
            "{} · cut {}",
            family,
            number(item, "cut")
        )));
        lines.push(Line::styled(
            format!(
                "  RMSE {} → {} · {} evaluations",
                rmse(item, "old_rmse"),
                rmse(item, "new_rmse"),
                number(item, "evaluations")
            ),
            Style::default().fg(GREEN),
        ));
        let points = item["trajectory"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v["rmse"].as_f64())
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        if points.len() > 1 {
            let low = points.iter().copied().fold(f64::INFINITY, f64::min);
            let high = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let glyphs = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            let curve = points
                .iter()
                .map(|v| {
                    glyphs[(((v - low) / (high - low).max(1e-20) * 7.).round() as usize).min(7)]
                })
                .collect::<String>();
            lines.push(Line::styled(
                format!(
                    "  Best-found RMSE {curve} · {} ordered updates",
                    points.len()
                ),
                Style::default().fg(VIOLET),
            ));
        }
    }
    lines.push(Line::styled(
        "Best observed programs, without a global-optimum claim.",
        Style::default().fg(MUTED),
    ));
    lines
}
pub(super) fn decimal_count(v: &Value, key: &str) -> String {
    v[key]
        .as_u64()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}
pub(super) fn learning_chart(f: &mut Frame, area: Rect, run: &Value) {
    let samples = run["learning_curve"].as_array();
    let train = samples
        .into_iter()
        .flatten()
        .filter_map(|v| Some((v["step"].as_f64()?, v["sequence_nll"].as_f64()?)))
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    let mut validation = samples
        .into_iter()
        .flatten()
        .filter_map(|v| {
            Some((
                v["validation"]["evaluated_step"].as_f64()?,
                v["validation"]["sequence_nll"].as_f64()?,
            ))
        })
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    validation.dedup_by(|a, b| a.0 == b.0);
    if train.len() < 2 {
        f.render_widget(
            Paragraph::new("Learning curve unavailable until two optimizer receipts are saved.")
                .wrap(Wrap { trim: true })
                .block(panel(" Sequence NLL · saved optimizer history ")),
            area,
        );
        return;
    }
    let xmax = train.last().unwrap().0.max(1.);
    let ymin = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let ymax = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((ymax - ymin) * 0.1).max(0.02);
    let mut datasets = vec![Dataset::default()
        .name("train")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(BLUE))
        .data(&train)];
    if !validation.is_empty() {
        datasets.push(
            Dataset::default()
                .name("validation (bound step)")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(GOLD))
                .data(&validation),
        );
    }
    f.render_widget(
        Chart::new(datasets)
            .block(panel(" Sequence NLL · saved optimizer history "))
            .x_axis(
                Axis::default()
                    .title("optimizer step")
                    .bounds([0., xmax])
                    .labels(["0".to_string(), format!("{xmax:.0}")])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([(ymin - padding).max(0.), ymax + padding])
                    .labels([format!("{ymin:.2}"), format!("{ymax:.2}")])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}
pub(super) fn neural_lines(v: &Value) -> Vec<Line<'static>> {
    if v["encoding"].as_bool() == Some(true) {
        return vec![
            Line::styled(
                phase(v).to_string(),
                Style::default().fg(Color::Cyan).bold(),
            ),
            Line::raw(""),
            label("GPU workers", "8"),
            label(
                "Cut encodings saved",
                format!("{} / {}", number(v, "completed"), number(v, "total")),
            ),
            label(
                "CPU allocation",
                if v["persistent"] == true {
                    "120 CPUs · 4 host CPUs reserved"
                } else {
                    "16 CPUs"
                },
            ),
            label("Storage", "pooled vectors + hash receipts"),
            label("Cut data", "source pointers; memory views"),
            label("Telemetry", status_age(v)),
            Line::raw("Frozen pretrained encoders. No final-test targets enter training context."),
        ];
    }
    vec![
        Line::styled(
            phase(v).to_string(),
            Style::default().fg(Color::Cyan).bold(),
        ),
        Line::raw(""),
        label("Telemetry", status_age(v)),
        label("Synchronized ranks", format!("{} / 8", number(v, "world"))),
        label(
            "Optimizer steps",
            format!("{} / {}", number(v, "step"), number(v, "steps")),
        ),
        label(
            "Global / micro batch",
            format!(
                "{} / {}",
                number(v, "global_batch"),
                number(v, "microbatch")
            ),
        ),
        label("Loss", decimal(v, "loss")),
        label(
            "NLL / pairwise rank",
            format!(
                "{} / {}",
                decimal(v, "sequence_nll"),
                decimal(v, "ranking_bt")
            ),
        ),
        label(
            "Step / reduction (s)",
            format!(
                "{} / {}",
                decimal(v, "step_seconds"),
                decimal(v, "communication_seconds")
            ),
        ),
        label(
            "Examples / second",
            if v["end_to_end_examples_per_second"].is_number() {
                format!(
                    "{} end-to-end",
                    decimal(v, "end_to_end_examples_per_second")
                )
            } else {
                format!("{} · step timing", decimal(v, "examples_per_second"))
            },
        ),
        label(
            "Corpus eligibility",
            if v["corpus_eligible"] == true {
                "qualified"
            } else {
                "qualification only"
            },
        ),
    ]
}
