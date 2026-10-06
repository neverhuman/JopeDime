//! W §17 metrics from actual search work and emitted winner artifacts.
//! Native measurements cannot qualify the separate compiled parity gate.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub struct Run {
    pub dataset: String,
    pub card: usize,
    pub enabled: bool,
    pub seed: u64,
    pub search: Value,
    pub receipt: Value,
    pub wall: f64,
}

fn count(value: &Value) -> Result<f64, String> {
    value
        .as_u64()
        .map(|v| v as f64)
        .ok_or("missing actual count".into())
}

fn score(value: &Value) -> Result<f64, String> {
    let value = f64::from_bits(value.as_u64().ok_or("missing score bits")?);
    value
        .is_finite()
        .then_some(value)
        .ok_or("nonfinite measured score".into())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    let center = mean(values);
    (values.iter().map(|v| (v - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

fn arm(runs: &[&Run], card: usize) -> Result<Value, String> {
    let mut best = vec![];
    let mut medians = vec![];
    let mut thresholds = vec![];
    let mut nodes = vec![];
    let mut depths = vec![];
    let mut winner_tokens: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut winner_bytes = vec![];
    let mut wall = vec![];
    let mut cells = 0.0;
    let mut domain = 0.0;
    let mut attempts = 0.0;
    let mut hard = 0.0;
    let mut invalid = 0.0;
    let mut elite_count = 0;
    let mut used_count = 0;
    let mut complete = true;
    for run in runs {
        let data = &run.search;
        if data["budget"] != 72
            || data["attempted"] != 72
            || count(&data["completed"])? + count(&data["refused"])? != 72.0
            || data["seed"] != run.seed
            || run.wall < 0.0
            || !run.wall.is_finite()
        {
            return Err("utility budget, seed or wall mismatch".into());
        }
        best.push(score(&data["best_cv_bits"])?);
        medians.push(score(&data["top_ten_median_bits"])?);
        let threshold = &data["evals_to_95_final_best"];
        if !threshold.is_null() && !threshold.as_u64().is_some_and(|v| (1..=72).contains(&v)) {
            return Err("invalid actual literal M3 threshold".into());
        }
        thresholds.push(threshold.clone());
        attempts += count(&data["attempted"])?;
        hard += count(&data["hard_rejected_attempts"])?;
        invalid += count(&data["invalid_offspring_proposals"])?;
        cells += count(&data["all_pass_faults"]["cells"])?;
        domain += count(&data["all_pass_faults"]["domain"])?;
        complete &=
            data["cell_accounting_complete"] == true && data["unobserved_internal_passes"] == 0;
        wall.push(run.wall / 72.0);
        let elites = data["elite"].as_array().ok_or("missing actual elites")?;
        if elites.is_empty() || elites.len() > 10 {
            return Err("invalid actual elite count".into());
        }
        for elite in elites {
            let n = count(&elite["nodes"])?;
            let d = count(&elite["depth"])?;
            if n > 15.0 || d > 4.0 {
                return Err("utility elite exceeds unchanged caps".into());
            }
            nodes.push(n);
            depths.push(d);
            elite_count += 1;
            used_count += usize::from(count(&elite["card_token_occurrences"][card])? > 0.0);
        }
        let recipes = run.receipt["recipes"]
            .as_array()
            .ok_or("missing executed-source inventory")?;
        let mut languages = BTreeSet::new();
        let mut bytes = count(&run.receipt["fitted_source_bytes"])?
            + count(&run.receipt["fitted_codec_bytes"])?;
        for artifact in recipes {
            let language = artifact["language"]
                .as_str()
                .ok_or("missing artifact language")?;
            if !languages.insert(language) {
                return Err("duplicate winner artifact language".into());
            }
            let tokens = count(&artifact["code_tokens"])? + count(&artifact["doc_tokens"])?;
            winner_tokens
                .entry(language.to_owned())
                .or_default()
                .push(tokens);
            bytes += count(&artifact["bytes"])?;
        }
        if languages != BTreeSet::from(["Rust", "Python", "Cpp17"]) {
            return Err("missing winner artifact language".into());
        }
        winner_bytes.push(bytes);
    }
    if cells == 0.0 {
        return Err("no actually evaluated utility cells".into());
    }
    Ok(json!({
        "M1_best_cv_mean":mean(&best), "M1_seed_sd":sample_sd(&best),
        "M2_top10_cv_mean":mean(&medians), "M3_literal_evaluations":thresholds,
        "M4_domain_rate":domain/cells, "M5_hard_reject_rate":hard/attempts,
        "M6_invalid_offspring_rate":invalid/attempts,
        "M7_elite_mean_nodes":mean(&nodes), "M7_elite_mean_depth":mean(&depths),
        "M8_best_tokens":winner_tokens.into_iter().map(|(k,v)|(k,mean(&v))).collect::<BTreeMap<_,_>>(),
        "M8_best_artifact_bytes":mean(&winner_bytes),
        "M9_new_token_elite_share":used_count as f64/elite_count as f64,
        "M10_wall_seconds_per_attempt":mean(&wall),
        "attempts":attempts, "actual_evaluated_cells":cells,
        "cell_accounting_complete":complete
    }))
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("internally computed finite metric")
}

/// Missing parity evidence always remains pending. A native run cannot promote.
pub fn summarize(runs: &[Run]) -> Result<Value, String> {
    let mut identities = BTreeSet::new();
    let mut grouped: BTreeMap<(usize, String), [Vec<&Run>; 2]> = BTreeMap::new();
    for run in runs {
        if run.card > 2 || !identities.insert((run.card, &run.dataset, run.enabled, run.seed)) {
            return Err("duplicate or unknown utility run identity".into());
        }
        grouped.entry((run.card, run.dataset.clone())).or_default()[usize::from(run.enabled)]
            .push(run);
    }
    let expected: BTreeSet<_> = grouped
        .keys()
        .filter(|(card, _)| *card == 0)
        .map(|(_, dataset)| dataset)
        .collect();
    for card in 1..3 {
        let observed: BTreeSet<_> = grouped
            .keys()
            .filter(|(kind, _)| *kind == card)
            .map(|(_, dataset)| dataset)
            .collect();
        if observed != expected {
            return Err("utility cards must share the identical pinned cohort".into());
        }
    }
    let mut cards = vec![];
    for card in 0..3 {
        let mut comparisons = vec![];
        let mut noninferior = true;
        let mut strict_improvements = 0;
        let mut threshold_a = vec![];
        let mut threshold_b = vec![];
        let mut thresholds_complete = true;
        let mut sound = true;
        let mut a_cost = [0.0; 7];
        let mut b_cost = [0.0; 7];
        let mut used = false;
        for ((kind, dataset), arms) in &grouped {
            if *kind != card {
                continue;
            }
            for arm in arms {
                if arm.iter().map(|r| r.seed).collect::<BTreeSet<_>>()
                    != BTreeSet::from([3, 17, 29, 101, 701])
                {
                    return Err("missing exact five-seed utility arm".into());
                }
            }
            let a = arm(&arms[0], card)?;
            let b = arm(&arms[1], card)?;
            let delta = number(&b["M1_best_cv_mean"]) - number(&a["M1_best_cv_mean"]);
            let noise = number(&a["M1_seed_sd"]);
            noninferior &= delta >= -0.25 * noise;
            // Zero-noise ties are not improvements.
            strict_improvements += usize::from(delta > 0.0 && delta >= noise);
            for (metric, allowed) in [
                ("M4_domain_rate", 0.02),
                ("M5_hard_reject_rate", 0.01),
                ("M6_invalid_offspring_rate", 0.01),
            ] {
                sound &= number(&b[metric]) - number(&a[metric]) <= allowed;
            }
            sound &= a["cell_accounting_complete"] == true && b["cell_accounting_complete"] == true;
            for (source, dest) in [(&a, &mut threshold_a), (&b, &mut threshold_b)] {
                for value in source["M3_literal_evaluations"]
                    .as_array()
                    .expect("computed M3")
                {
                    if let Some(value) = value.as_u64() {
                        dest.push(value as f64);
                    } else {
                        thresholds_complete = false;
                    }
                }
            }
            for (metric, dest) in [(&a, &mut a_cost), (&b, &mut b_cost)] {
                for (i, key) in [
                    "M7_elite_mean_nodes",
                    "M7_elite_mean_depth",
                    "M8_best_artifact_bytes",
                    "M10_wall_seconds_per_attempt",
                ]
                .iter()
                .enumerate()
                {
                    dest[i] += number(&metric[*key]);
                }
                for (i, language) in ["Rust", "Python", "Cpp17"].iter().enumerate() {
                    dest[i + 4] += number(&metric["M8_best_tokens"][*language]);
                }
            }
            used |= number(&b["M9_new_token_elite_share"]) > 0.0;
            comparisons
                .push(json!({"dataset":dataset,"A":a,"B":b,"M1_delta":delta,"sigma_seed_A":noise}));
        }
        if comparisons.len() < 6 {
            return Err("fewer than six measured utility datasets".into());
        }
        let faster = thresholds_complete && mean(&threshold_b) <= 0.9 * mean(&threshold_a);
        let quality = noninferior && (strict_improvements >= 2 || faster);
        let cost = b_cost[0] <= a_cost[0]
            && b_cost[1] <= a_cost[1]
            && b_cost[2] <= 1.05 * a_cost[2]
            && b_cost[3] <= 1.10 * a_cost[3]
            && (4..7).all(|i| b_cost[i] <= 1.05 * a_cost[i]);
        cards.push(json!({"card":(["B05","B06","B07"][card]),"datasets":comparisons,
            "noninferior_on_every_dataset":noninferior,"strict_dataset_improvements":strict_improvements,
            "M3_complete":thresholds_complete,"M3_ten_percent_faster":faster,
            "gates":{"G-A":quality,"G-B":"pending compiled L1–L4 and fitted proof","G-C":sound,"G-D":cost,"G-E":used},
            "decision":if used {"keep-experimental"} else {"decline"},
            "decided_by":"owner decision pending; defaults unchanged"}));
    }
    Ok(
        json!({"schema":"feat-math.scalar-utility-metrics.v1","runs":runs.len(),"cards":cards,
        "status":"native metrics; compiled parity and exact-head lifecycle still required",
        "quality_definition":"M1/M2/M3 use raw inner-fold R2 of objective-selected winner/elites; fault penalties affect selection only",
        "noise_definition":"sample standard deviation of the five A seeds on each dataset",
        "cost_definition":"elite nodes/depth; best three-language code+doc tokens and recipe+fitted+codec bytes; wall per attempted evaluation",
        "invalid_offspring_denominator":"all birth and edit attempts, including refused proposals",
        "usage_definition":"share of elite genomes using the card's token or explicit B07 policy selector"}),
    )
}
