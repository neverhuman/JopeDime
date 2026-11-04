use super::*;

/// SplitMix64: a small, fully specified generator, so a bootstrap replays from its seed alone.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` without modulo bias; `n` is positive.
    fn below(&mut self, n: u64) -> u64 {
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let draw = self.next();
            if draw < limit {
                return draw % n;
            }
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    }
}

/// The inverse empirical CDF (type 1) of `sorted` at `p` in (0, 1).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The percentile-bootstrap bounds of the mean of `effects` at one-sided level `alpha`.
fn bootstrap_bounds(effects: &[f64], bootstrap: Bootstrap, alpha: f64) -> (f64, f64) {
    let mut rng = SplitMix64(bootstrap.seed);
    let n = effects.len() as u64;
    let mut means = Vec::with_capacity(bootstrap.resamples as usize);
    let mut draw = vec![0.0; effects.len()];
    for _ in 0..bootstrap.resamples {
        for slot in &mut draw {
            *slot = effects[usize::try_from(rng.below(n)).unwrap_or(0)];
        }
        means.push(mean(&draw));
    }
    means.sort_by(f64::total_cmp);
    (quantile(&means, alpha), quantile(&means, 1.0 - alpha))
}

/// Decide one look of a predeclared arena. Read-only; promotes nothing.
///
/// # Errors
///
/// Either document is not valid JSON of its schema, the manifest is not a usable predeclared
/// confirmation arena, the results do not bind its exact bytes, the look is out of schedule or
/// after a stop, or a row is invalid (an oracle row, an undeclared family, a repeated context, a
/// non-finite metric, more failures than attempts).
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn decide(manifest_bytes: &[u8], results_bytes: &[u8]) -> Result<ChampionReport, String> {
    let manifest: ArenaManifest =
        serde_json::from_slice(manifest_bytes).map_err(|e| format!("manifest: {e}"))?;
    validate_manifest(&manifest)?;
    let results: ArenaResults =
        serde_json::from_slice(results_bytes).map_err(|e| format!("results: {e}"))?;
    if results.schema != RESULTS_SCHEMA {
        return Err(format!(
            "results schema {:?} is not {RESULTS_SCHEMA}",
            results.schema
        ));
    }
    let manifest_sha256 = sha256_hex(manifest_bytes);
    if results.manifest_sha256 != manifest_sha256 {
        return Err("results do not bind the exact manifest bytes".to_owned());
    }

    // The look schedule: this look is declared, every earlier look was taken in order, and none
    // of them stopped the study.
    let look = manifest
        .looks
        .iter()
        .find(|item| item.index == results.look)
        .copied()
        .ok_or_else(|| format!("look {} is not declared", results.look))?;
    let prior: Vec<u32> = results.prior_looks.iter().map(|item| item.index).collect();
    if prior != (1..results.look).collect::<Vec<_>>() {
        return Err(format!(
            "look {} needs exactly the prior looks 1..{}, got {prior:?}",
            results.look, results.look
        ));
    }
    if let Some(stop) = results.prior_looks.iter().find(|item| item.verdict.stops()) {
        return Err(format!(
            "the study stopped at look {} ({})",
            stop.index, stop.verdict
        ));
    }
    let alpha_spent: f64 = manifest
        .looks
        .iter()
        .take_while(|item| item.index <= look.index)
        .map(|item| item.alpha)
        .sum();
    let alpha_per_comparison = look.alpha / f64::from(manifest.contenders);

    // Rows: declared families, unique contexts, selected programs only.
    let declared: BTreeSet<&str> = manifest.families.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut by_family: BTreeMap<&str, Vec<Option<f64>>> = BTreeMap::new();
    let (mut attempts, mut failures) = (0_u64, 0_u64);
    for row in &results.rows {
        if row.selection == Selection::Oracle {
            return Err(format!(
                "context {} is an outer-oracle row: a diagnostic, never promotion evidence",
                row.context_id
            ));
        }
        if !declared.contains(row.family_id.as_str()) {
            return Err(format!("family {} is not declared", row.family_id));
        }
        if !seen.insert((row.family_id.as_str(), row.context_id.as_str())) {
            return Err(format!("context {} is repeated", row.context_id));
        }
        if row.failures > row.attempts {
            return Err(format!(
                "context {} has more failures than attempts",
                row.context_id
            ));
        }
        attempts = attempts.saturating_add(row.attempts);
        failures = failures.saturating_add(row.failures);
        let effect = match (row.challenger, row.incumbent) {
            (Some(challenger), Some(incumbent)) => {
                finite("challenger metric", challenger)?;
                finite("incumbent metric", incumbent)?;
                Some(if manifest.metric.lower_is_better {
                    incumbent - challenger
                } else {
                    challenger - incumbent
                })
            }
            _ => None,
        };
        by_family
            .entry(row.family_id.as_str())
            .or_default()
            .push(effect);
    }
    let cost = Cost {
        cpu_seconds: finite("cost.cpu_seconds", results.cost.cpu_seconds)?,
        wall_seconds: finite("cost.wall_seconds", results.cost.wall_seconds)?,
    };
    if cost.cpu_seconds < 0.0 || cost.wall_seconds < 0.0 {
        return Err("cost is negative".to_owned());
    }
    let safety_regression = finite("safety.challenger", results.safety.challenger)?
        - finite("safety.reference", results.safety.reference)?;

    // Family effects. A family with no rows or an unknown context is incomplete, never imputed.
    let mut family_effects = Vec::new();
    let mut context_effects = Vec::new();
    let mut families_incomplete = Vec::new();
    let contexts_unknown = by_family
        .values()
        .flatten()
        .filter(|effect| effect.is_none())
        .count();
    for family in &manifest.families {
        let known: Option<Vec<f64>> = by_family
            .get(family.as_str())
            .filter(|effects| !effects.is_empty())
            .and_then(|effects| effects.iter().copied().collect());
        match known {
            Some(effects) => {
                family_effects.push(mean(&effects));
                context_effects.extend(effects);
            }
            None => families_incomplete.push(family.clone()),
        }
    }
    let wins = family_effects
        .iter()
        .filter(|effect| **effect > 0.0)
        .count();
    let losses = family_effects
        .iter()
        .filter(|effect| **effect < 0.0)
        .count();
    let ties = family_effects.len() - wins - losses;
    let mut sorted = family_effects.clone();
    sorted.sort_by(f64::total_cmp);
    let (family_effect_mean, family_effect_median, context_effect_mean) = if sorted.is_empty() {
        (None, None, None)
    } else {
        (
            Some(mean(&family_effects)),
            Some(median(&sorted)),
            Some(mean(&context_effects)),
        )
    };
    let failure_rate = (attempts > 0).then(|| failures as f64 / attempts as f64);

    // The verdict, in precedence order; every applicable reason is kept.
    let mut reasons = Vec::new();
    let mut verdict = None;
    if !families_incomplete.is_empty() {
        reasons.push(format!(
            "{} declared families incomplete and left out, not imputed",
            families_incomplete.len()
        ));
    }
    if !results.holds.is_empty() {
        reasons.push(format!("holds in force: {}", results.holds.join(", ")));
        verdict = verdict.or(Some(Verdict::Hold));
    }
    match failure_rate {
        None => {
            reasons.push("no attempts recorded: attempt accounting is absent".to_owned());
            verdict = verdict.or(Some(Verdict::Hold));
        }
        Some(rate) if rate > manifest.max_failure_rate => {
            reasons.push(format!(
                "failure rate {rate} exceeds the limit {}",
                manifest.max_failure_rate
            ));
            verdict = verdict.or(Some(Verdict::Hold));
        }
        Some(_) => {}
    }
    if safety_regression > manifest.safety.max_regression {
        reasons.push(format!(
            "{} regresses by {safety_regression} against {} (lower is better; limit {})",
            manifest.safety.panel, manifest.safety.reference, manifest.safety.max_regression
        ));
        verdict = verdict.or(Some(Verdict::Regression));
    }
    let mut bounds = (None, None);
    if family_effects.len() < manifest.min_families {
        reasons.push(format!(
            "{} complete families, fewer than the {} required",
            family_effects.len(),
            manifest.min_families
        ));
        verdict = verdict.or(Some(Verdict::Inconclusive));
    } else {
        let (lower, upper) =
            bootstrap_bounds(&family_effects, manifest.bootstrap, alpha_per_comparison);
        bounds = (Some(lower), Some(upper));
        let margin = manifest.practical_margin;
        if families_incomplete.len() > manifest.max_incomplete_families {
            // Leaving families out can favour either contender, so the bounds decide nothing.
            reasons.push(format!(
                "{} incomplete families exceed the predeclared {}: selection on missingness",
                families_incomplete.len(),
                manifest.max_incomplete_families
            ));
            verdict = verdict.or(Some(Verdict::Inconclusive));
        } else if lower > margin {
            verdict = verdict.or(Some(Verdict::PromoteEligible));
        } else if upper < margin {
            reasons.push(format!(
                "upper bound {upper} is below the practical margin {margin}"
            ));
            verdict = verdict.or(Some(Verdict::NoPracticalLift));
        } else {
            reasons.push(format!(
                "bounds [{lower}, {upper}] straddle the practical margin {margin}"
            ));
            verdict = verdict.or(Some(Verdict::Inconclusive));
        }
    }

    Ok(ChampionReport {
        schema: REPORT_SCHEMA,
        manifest_sha256,
        results_sha256: sha256_hex(results_bytes),
        arena_id: manifest.arena_id,
        metric: manifest.metric.name,
        incumbent: manifest.incumbent.id,
        challenger: manifest.challenger.id,
        roster_role: manifest.roster_role,
        look: look.index,
        alpha_look: look.alpha,
        alpha_per_comparison,
        alpha_spent,
        families_declared: manifest.families.len(),
        families_complete: family_effects.len(),
        families_incomplete,
        contexts: results.rows.len(),
        contexts_unknown,
        wins,
        ties,
        losses,
        family_effect_mean,
        family_effect_median,
        context_effect_mean,
        family_weighting: manifest.family_weighting,
        max_incomplete_families: manifest.max_incomplete_families,
        lower: bounds.0,
        upper: bounds.1,
        practical_margin: manifest.practical_margin,
        attempts,
        failures,
        failure_rate,
        cost,
        safety_panel: manifest.safety.panel,
        safety_reference: manifest.safety.reference,
        safety_regression,
        verdict: verdict.unwrap_or(Verdict::Inconclusive),
        reasons,
        generalization_claim: GENERALIZATION_CLAIM,
        promoted: false,
    })
}
