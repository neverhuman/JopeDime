use super::*;

pub(crate) fn milestone_lines(run: &Value, gp: &Value, neural: &Value) -> Vec<Line<'static>> {
    let contexts = &run["contexts"];
    let policy =
        if run["policy"]["corpus_sha256"].is_string() && run["policy"]["invalidated"] != true {
            &run["policy"]
        } else if run["fresh_policy"].is_object() {
            &run["fresh_policy"]
        } else {
            neural
        };
    let gp_total = number(gp, "completed_views")
        + number(gp, "failed_views")
        + gp["active"]
            .as_u64()
            .or_else(|| gp["last_recorded_activity"]["active"].as_u64())
            .unwrap_or(0)
        + number(gp, "queued");
    let cycle = &run["cycle"];
    let mut lines = vec![
        label(
            "1  Encodings on disk",
            format!(
                "{} views · {:.1}% · {} encoded families · {} dimensions · {}",
                count_pair(number(contexts, "completed"), number(contexts, "total")),
                100.0 * number(contexts, "completed") as f64
                    / number(contexts, "total").max(1) as f64,
                number(contexts, "families"),
                number(contexts, "dimensions"),
                if contexts["qualification"] == false {
                    "qualified"
                } else {
                    "qualification cohort"
                }
            ),
        ),
        label(
            "2  GP teaching targets",
            format!(
                "{} search jobs · {} evaluations · {} failed",
                count_pair(number(gp, "completed_views"), gp_total),
                number(gp, "completed_evaluations") + number(gp, "running_evaluations"),
                number(gp, "failed_views")
            ),
        ),
        label(
            "3  Policy checkpoint",
            if policy["invalidated"] == true {
                "previous run invalidated · corrected training pending".into()
            } else if policy["encoding"] == true || !policy["step"].is_number() {
                "awaiting training receipt".into()
            } else {
                format!(
                    "{} steps · {} ranks · {}",
                    count_pair(number(policy, "step"), number(policy, "steps")),
                    number(policy, "world"),
                    status_age(policy)
                )
            },
        ),
        label(
            "4  Neural → GP → DL",
            if run["continuous"].is_object() {
                let (status, detail) = campaign_task_state(run, neural);
                format!("{status} · {detail}")
            } else if cycle["complete"] == true {
                "completed · comparisons and fitted predictors saved".into()
            } else if cycle["phase"].is_string() {
                format!(
                    "{} · proposals {} · GP {} · retrain {}",
                    phase(cycle),
                    count_pair(
                        number(&cycle["proposals"], "completed"),
                        number(&cycle["proposals"], "total")
                    ),
                    count_pair(
                        number(&cycle["neural_seeded_gp"], "completed"),
                        number(&cycle["neural_seeded_gp"], "total")
                    ),
                    count_pair(
                        number(&cycle["retraining"], "step"),
                        number(&cycle["retraining"], "steps")
                    )
                )
            } else {
                "pending neural proposals, GP reevaluation, then retraining".into()
            },
        ),
    ];
    if run["encoding_pipeline"]["phase"].is_string() {
        let pipeline = &run["encoding_pipeline"];
        lines.push(label(
            "Next encoding cohort",
            format!(
                "{} · {} contexts · {}",
                phase(pipeline),
                count_pair(number(pipeline, "completed"), number(pipeline, "total")),
                status_age(pipeline)
            ),
        ));
    }
    if run["continuous"].is_object() {
        let (status, detail) = campaign_task_state(run, neural);
        lines.push(label("Continuous learning", format!("{status} · {detail}")));
    }
    if run["campaign_service"]["phase"].is_string() {
        let campaign = &run["campaign_service"];
        lines.push(label(
            if run["continuous"].is_object() {
                "Historical campaign"
            } else {
                "Managed campaign"
            },
            format!(
                "{} cycles complete · {}",
                number(campaign, "completed_cycles"),
                phase(campaign)
            ),
        ));
    }
    if let Some(checkpoint) = run["selected_checkpoint"]["checkpoint"].as_str() {
        lines.push(label(
            "Selected checkpoint",
            checkpoint.trim_start_matches("runs/"),
        ));
    }
    lines.push(label(
        "24-hour pilot",
        if run["pilot_started"] == true {
            "running · inspect pilot receipt"
        } else {
            "not started · qualification work; full pilot gates remain"
        },
    ));
    lines
}
