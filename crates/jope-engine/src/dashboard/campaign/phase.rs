use super::*;

pub(crate) fn goal_phase<'a>(run: &'a Value, neural: &'a Value, gp: &'a Value) -> &'a str {
    if run["continuous"].is_object() {
        if run["operations"]["services"]["jope-parallel-campaign"]["ActiveState"] != "active"
            || now().saturating_sub(number(&run["continuous"], "timestamp")) > 15
        {
            return "Continuous controller unavailable · inspect saved evidence";
        }
        if run["continuous"]["phase"] == "paused" {
            return "Continuous learning paused";
        }
        if completed(neural) {
            return "Continuous learning · scheduling fresh GP targets";
        }
    }
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    if training_hold(run, policy).is_some() {
        return "Automatic training held · inspect task state";
    }
    if number(neural, "timestamp") > 0
        && !completed(neural)
        && now().saturating_sub(number(neural, "timestamp")) <= 15
    {
        return phase(neural);
    }
    if !completed(&run["auxiliary"])
        && number(&run["auxiliary"], "timestamp") > 0
        && now().saturating_sub(number(&run["auxiliary"], "timestamp")) <= 15
    {
        return phase(&run["auxiliary"]);
    }
    let pure = &run["pure_proposal_library"];
    if phase(pure) == "running"
        && number(pure, "timestamp") > 0
        && now().saturating_sub(number(pure, "timestamp")) <= 15
    {
        return "Saving pure policy predictions on eight GPUs";
    }
    let library = &run["proposal_library"];
    if phase(library) == "running"
        && number(library, "timestamp") > 0
        && now().saturating_sub(number(library, "timestamp")) <= 15
    {
        return "Building saved proposal seed library on eight GPUs";
    }
    let pipeline = &run["encoding_pipeline"];
    if !completed(pipeline)
        && number(pipeline, "timestamp") > 0
        && now().saturating_sub(number(pipeline, "timestamp")) <= 15
    {
        return phase(pipeline);
    }
    if !completed(&run["cycle"]) {
        if let Some(value) = run["cycle"]["phase"].as_str() {
            return value;
        }
    }
    if number(gp, "active") > 0 && now().saturating_sub(number(gp, "timestamp")) <= 5 {
        return "GP teaching corpus growing";
    }
    if run["contexts"]["completed"].is_number() {
        return "Encoded corpus ready · next learning cycle";
    }
    phase(run)
}
