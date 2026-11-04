use super::*;

pub(super) fn finite(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} is not finite"))
    }
}

pub(super) fn validate_manifest(manifest: &ArenaManifest) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!(
            "manifest schema {:?} is not {MANIFEST_SCHEMA}",
            manifest.schema
        ));
    }
    if manifest.arena_id.is_empty() || manifest.metric.name.is_empty() {
        return Err("arena_id and metric.name are required".to_owned());
    }
    for (role, contender) in [
        ("incumbent", &manifest.incumbent),
        ("challenger", &manifest.challenger),
    ] {
        if contender.id.is_empty() || !is_hex64(&contender.sha256) {
            return Err(format!("{role} needs an id and a lowercase hex sha256"));
        }
    }
    if manifest.challenger.sha256 == manifest.incumbent.sha256 {
        return Err(
            "the challenger is the incumbent; keep the champion and the learning branch separate"
                .to_owned(),
        );
    }
    match manifest.roster_role.as_str() {
        "confirmation" => {}
        "sealed_final" => {
            return Err(
                "sealed_final looks need an adopted sealed roster (G23); none exists in this build"
                    .to_owned(),
            )
        }
        "development" | "meta_training" => {
            return Err(format!(
                "{} families are not independent confirmation evidence",
                manifest.roster_role
            ))
        }
        other => return Err(format!("unknown roster role {other:?}")),
    }
    let unique: BTreeSet<&str> = manifest.families.iter().map(String::as_str).collect();
    if unique.len() != manifest.families.len() || unique.contains("") {
        return Err("families must be unique non-empty ids".to_owned());
    }
    if manifest.family_weighting != "equal" {
        return Err(format!(
            "family_weighting {:?} is not \"equal\", the only weighting this gate estimates",
            manifest.family_weighting
        ));
    }
    if manifest.min_families < 2 || manifest.min_families > manifest.families.len() {
        return Err(format!(
            "min_families {} must be at least 2 and at most the {} declared families",
            manifest.min_families,
            manifest.families.len()
        ));
    }
    if manifest.max_incomplete_families > manifest.families.len() - manifest.min_families {
        return Err(format!(
            "max_incomplete_families {} leaves fewer than min_families {} of {} families",
            manifest.max_incomplete_families,
            manifest.min_families,
            manifest.families.len()
        ));
    }
    if finite("practical_margin", manifest.practical_margin)? < 0.0 {
        return Err("practical_margin is negative".to_owned());
    }
    let alpha_total = finite("alpha_total", manifest.alpha_total)?;
    if alpha_total <= 0.0 || alpha_total >= 1.0 {
        return Err("alpha_total must lie in (0, 1)".to_owned());
    }
    if manifest.looks.is_empty() {
        return Err("at least one look must be declared".to_owned());
    }
    if manifest.contenders == 0 {
        return Err("contenders must be at least 1".to_owned());
    }
    if manifest.bootstrap.resamples < MIN_RESAMPLES {
        return Err(format!(
            "bootstrap.resamples must be at least {MIN_RESAMPLES}"
        ));
    }
    let mut spent = 0.0;
    for (position, look) in manifest.looks.iter().enumerate() {
        if usize::try_from(look.index).ok() != Some(position + 1) {
            return Err("looks must be numbered 1, 2, ... in order".to_owned());
        }
        let alpha = finite("look alpha", look.alpha)?;
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(format!("look {} alpha must lie in (0, 1)", look.index));
        }
        let per_comparison = alpha / f64::from(manifest.contenders);
        if per_comparison * f64::from(manifest.bootstrap.resamples) < 1.0 {
            return Err(format!(
                "look {} alpha per comparison {per_comparison} is finer than {} resamples resolve",
                look.index, manifest.bootstrap.resamples
            ));
        }
        spent += alpha;
    }
    if spent > alpha_total + ALPHA_SLACK {
        return Err(format!(
            "looks spend alpha {spent}, more than alpha_total {alpha_total}"
        ));
    }
    let max_failure_rate = finite("max_failure_rate", manifest.max_failure_rate)?;
    if !(0.0..=1.0).contains(&max_failure_rate) {
        return Err("max_failure_rate must lie in [0, 1]".to_owned());
    }
    if manifest.safety.panel.is_empty() || manifest.safety.reference.is_empty() {
        return Err("safety.panel and safety.reference are required".to_owned());
    }
    if finite("safety.max_regression", manifest.safety.max_regression)? < 0.0 {
        return Err("safety.max_regression is negative".to_owned());
    }
    Ok(())
}
