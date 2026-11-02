use super::*;

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
