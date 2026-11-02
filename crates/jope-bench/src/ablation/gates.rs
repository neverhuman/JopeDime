use super::*;

pub(super) fn summary<'a>(runs: impl Iterator<Item = &'a RunRecord> + Clone) -> Summary {
    Summary {
        m1_rmse: mean_of(runs.clone().map(|r| r.metrics.m1.rmse)),
        m1_r2: mean_of(runs.clone().map(|r| r.metrics.m1.r2)),
        m2_rmse: mean_of(runs.clone().map(|r| r.metrics.m2.rmse)),
        m3: mean_of(runs.clone().map(|r| r.metrics.m3.mean)),
        m6_rate: mean_of(runs.clone().map(|r| r.metrics.m6.rate)),
        m7_stages: mean_of(runs.clone().map(|r| r.metrics.m7.mean_stages)),
        m7_expression_nodes: mean_of_options(
            runs.clone()
                .filter(|r| r.metrics.m7.expression_stages > 0)
                .map(|r| r.metrics.m7.mean_expression_nodes),
        ),
        m8_tokens: mean_of_options(runs.clone().map(|r| r.metrics.m8.tokens())),
        m8_bytes: mean_of_options(runs.clone().map(|r| r.metrics.m8.bytes())),
        m11: mean_of(runs.map(|r| r.metrics.m11.per_evaluation)),
    }
}

pub(super) fn aggregate(
    slot: &'static str,
    profile: Profile,
    datasets: &[Dataset],
    runs: &[RunRecord],
) -> Result<SlotAggregate> {
    let mut per_dataset = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let hash = &dataset.record.dataset_hash;
        let mine = runs.iter().filter(|r| r.slot == slot && &r.dataset == hash);
        let rmse: Vec<f64> = mine.clone().map(|r| r.metrics.m1.rmse).collect();
        ensure!(!rmse.is_empty(), "slot {slot} has no run on {hash}");
        per_dataset.push(DatasetAggregate {
            dataset: hash.clone(),
            seeds: rmse.len(),
            m1_rmse_seed_sd: sample_sd(&rmse),
            mean: summary(mine),
        });
    }
    let means = per_dataset.iter().map(|d| &d.mean);
    let overall = Summary {
        m1_rmse: mean_of(means.clone().map(|m| m.m1_rmse)),
        m1_r2: mean_of(means.clone().map(|m| m.m1_r2)),
        m2_rmse: mean_of(means.clone().map(|m| m.m2_rmse)),
        m3: mean_of(means.clone().map(|m| m.m3)),
        m6_rate: mean_of(means.clone().map(|m| m.m6_rate)),
        m7_stages: mean_of(means.clone().map(|m| m.m7_stages)),
        m7_expression_nodes: mean_of_options(
            means
                .clone()
                .filter(|m| m.m7_expression_nodes.is_some())
                .map(|m| m.m7_expression_nodes),
        ),
        m8_tokens: mean_of_options(means.clone().map(|m| m.m8_tokens)),
        m8_bytes: mean_of_options(means.clone().map(|m| m.m8_bytes)),
        m11: mean_of(means.map(|m| m.m11)),
    };
    Ok(SlotAggregate {
        slot,
        profile: profile.id(),
        datasets: per_dataset,
        overall,
    })
}

/// A clause comparing B with A where lower is better: pass when `b <= a * limit`.
pub(super) fn ratio_clause(
    metric: &'static str,
    rule: &'static str,
    a: f64,
    b: f64,
    limit: f64,
) -> Clause {
    let status = if a == b {
        SAME
    } else if b <= a * limit {
        PASS
    } else {
        FAIL
    };
    Clause {
        metric,
        rule,
        status,
        a: Some(a),
        b: Some(b),
        note: None,
    }
}

/// A clause over a value both profiles may lack for the same reason, such as the M7 expression
/// size when no best program of either profile has a `map` or `derive` stage: equal absence is no
/// difference.
pub(super) fn optional_ratio_clause(
    metric: &'static str,
    rule: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    limit: f64,
) -> Clause {
    match (a, b) {
        (Some(a), Some(b)) => ratio_clause(metric, rule, a, b, limit),
        (None, None) => Clause {
            metric,
            rule,
            status: SAME,
            a: None,
            b: None,
            note: Some("neither profile has a value"),
        },
        _ => Clause {
            metric,
            rule,
            status: NOT_EVALUATED,
            a,
            b,
            note: Some("only one profile has a value"),
        },
    }
}

/// A clause over a value every run should supply, such as M8: an aggregate is `None` when some
/// run lacks it, so the clause is evaluated only when both profiles have a value. A value missing
/// from both profiles was never compared, so it is not evaluated, never no difference.
pub(super) fn every_run_ratio_clause(
    metric: &'static str,
    rule: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    limit: f64,
) -> Clause {
    match (a, b) {
        (Some(a), Some(b)) => ratio_clause(metric, rule, a, b, limit),
        _ => Clause {
            metric,
            rule,
            status: NOT_EVALUATED,
            a,
            b,
            note: Some(
                "a run's best program has no value: the emitters or the token estimator refused \
                 it (see that run's M8 unsupported)",
            ),
        },
    }
}

pub(super) fn not_evaluated(
    metric: &'static str,
    rule: &'static str,
    note: &'static str,
) -> Clause {
    Clause {
        metric,
        rule,
        status: NOT_EVALUATED,
        a: None,
        b: None,
        note: Some(note),
    }
}

/// A gate whose clauses must all hold: it fails on any failing clause, reports no difference
/// when every evaluated clause does, and is not evaluated when no clause could be.
pub(super) fn all_of(rule: &'static str, clauses: Vec<Clause>) -> Gate {
    let evaluated: Vec<&Clause> = clauses
        .iter()
        .filter(|c| c.status != NOT_EVALUATED)
        .collect();
    let status = if evaluated.is_empty() {
        NOT_EVALUATED
    } else if evaluated.iter().any(|c| c.status == FAIL) {
        FAIL
    } else if evaluated.iter().all(|c| c.status == SAME) {
        SAME
    } else {
        PASS
    };
    Gate {
        status,
        complete: evaluated.len() == clauses.len(),
        rule,
        clauses,
        per_dataset: Vec::new(),
    }
}

/// G-A: not worse than A by more than 0.25 sigma on any dataset, and better by at least one
/// sigma on two datasets or M3 at least 10% lower.
pub(super) fn quality_gate(a: &SlotAggregate, b: &SlotAggregate) -> Gate {
    const RULE: &str = "mean M1 RMSE of B is not worse than A by more than 0.25 sigma_seed on any \
                        dataset, and better by at least 1 sigma_seed on at least 2 datasets or \
                        mean M3 at least 10% lower";
    let mut per_dataset = Vec::new();
    for (da, db) in a.datasets.iter().zip(&b.datasets) {
        let delta = db.mean.m1_rmse - da.mean.m1_rmse;
        let sigma = match (da.m1_rmse_seed_sd, db.m1_rmse_seed_sd) {
            (Some(sa), Some(sb)) => Some(((sa * sa + sb * sb) / 2.0).sqrt()),
            _ => None,
        };
        let (worse, better) = match sigma {
            Some(sigma) => (
                delta > 0.0 && delta > 0.25 * sigma,
                delta < 0.0 && -delta >= sigma,
            ),
            None => (false, false),
        };
        per_dataset.push(DatasetDelta {
            dataset: da.dataset.clone(),
            delta_rmse: delta,
            sigma_seed: sigma,
            worse,
            better,
        });
    }
    let identical = per_dataset.iter().all(|d| d.delta_rmse == 0.0) && a.overall.m3 == b.overall.m3;
    let sigma_known = per_dataset.iter().all(|d| d.sigma_seed.is_some());
    let non_inferior = !per_dataset.iter().any(|d| d.worse);
    let better = per_dataset.iter().filter(|d| d.better).count();
    let speed = b.overall.m3 <= 0.9 * a.overall.m3;
    let clause = |metric, rule, holds: bool| Clause {
        metric,
        rule,
        status: if identical {
            SAME
        } else if !sigma_known {
            NOT_EVALUATED
        } else if holds {
            PASS
        } else {
            FAIL
        },
        a: None,
        b: None,
        note: (!sigma_known && !identical).then_some("sigma_seed needs at least two seeds"),
    };
    let clauses = vec![
        clause(
            "M1",
            "not worse by more than 0.25 sigma_seed on any dataset",
            non_inferior,
        ),
        clause(
            "M1",
            "better by at least 1 sigma_seed on at least 2 datasets",
            better >= 2,
        ),
        Clause {
            a: Some(a.overall.m3),
            b: Some(b.overall.m3),
            ..clause("M3", "mean M3 at least 10% lower", speed)
        },
    ];
    let status = if identical {
        SAME
    } else if !sigma_known {
        NOT_EVALUATED
    } else if non_inferior && (better >= 2 || speed) {
        PASS
    } else {
        FAIL
    };
    Gate {
        status,
        complete: identical || sigma_known,
        rule: RULE,
        clauses,
        per_dataset,
    }
}

/// The gates of workplan §17.3 over the two profiles' aggregates.
pub(super) fn gates(
    profile_a: Profile,
    profile_b: Profile,
    a: &SlotAggregate,
    b: &SlotAggregate,
) -> Gates {
    let (oa, ob) = (&a.overall, &b.overall);
    let parity = Gate {
        status: if profile_a == profile_b {
            SAME
        } else {
            NOT_EVALUATED
        },
        complete: profile_a == profile_b,
        rule: "the parity lanes are bit-green with the profile's new elements in the corpus",
        clauses: vec![Clause {
            metric: "parity",
            rule: "crates/feat-math/tests/parity_bits.rs is green",
            status: if profile_a == profile_b {
                SAME
            } else {
                NOT_EVALUATED
            },
            a: None,
            b: None,
            note: Some(
                "evaluated by the parity test, not by the search; the same profile adds no \
                 element",
            ),
        }],
        per_dataset: Vec::new(),
    };
    Gates {
        g_a: quality_gate(a, b),
        g_b: parity,
        g_c: all_of(
            "M4(B) - M4(A) at most 0.02; M5 and M6 increase at most 0.01",
            vec![
                not_evaluated("M4", "increase at most 0.02", "no fault telemetry yet"),
                not_evaluated("M5", "increase at most 0.01", "no fault telemetry yet"),
                Clause {
                    metric: "M6",
                    rule: "increase at most 0.01",
                    status: if oa.m6_rate == ob.m6_rate {
                        SAME
                    } else if ob.m6_rate - oa.m6_rate <= 0.01 {
                        PASS
                    } else {
                        FAIL
                    },
                    a: Some(oa.m6_rate),
                    b: Some(ob.m6_rate),
                    note: None,
                },
            ],
        ),
        g_d: all_of(
            "M7 non-increasing on average; M8 increase at most 5%; M10 (timing) at most 10%",
            vec![
                ratio_clause(
                    "M7",
                    "mean stages non-increasing",
                    oa.m7_stages,
                    ob.m7_stages,
                    1.0,
                ),
                optional_ratio_clause(
                    "M7",
                    "mean expression nodes non-increasing",
                    oa.m7_expression_nodes,
                    ob.m7_expression_nodes,
                    1.0,
                ),
                every_run_ratio_clause(
                    "M8",
                    "token cost increase at most 5%",
                    oa.m8_tokens,
                    ob.m8_tokens,
                    1.05,
                ),
                every_run_ratio_clause(
                    "M8",
                    "artifact bytes increase at most 5%",
                    oa.m8_bytes,
                    ob.m8_bytes,
                    1.05,
                ),
            ],
        ),
        g_e: all_of(
            "M9 > 0: the new elements are used by the best programs",
            vec![not_evaluated(
                "M9",
                "greater than 0",
                "no profile adds elements yet",
            )],
        ),
    }
}

/// [`NO_DIFFERENCE`] when every B run's metrics and cut accounting equal the A run's for the
/// same dataset and seed.
pub(super) fn verdict(runs: &[RunRecord]) -> &'static str {
    let key = |r: &RunRecord| (r.dataset.clone(), r.seed);
    let a: BTreeMap<_, _> = runs
        .iter()
        .filter(|r| r.slot == "A")
        .map(|r| (key(r), r))
        .collect();
    let b: BTreeMap<_, _> = runs
        .iter()
        .filter(|r| r.slot == "B")
        .map(|r| (key(r), r))
        .collect();
    let same = a.len() == b.len()
        && a.iter().all(|(k, ra)| {
            b.get(k)
                .is_some_and(|rb| ra.metrics == rb.metrics && ra.cuts == rb.cuts)
        });
    if same {
        NO_DIFFERENCE
    } else {
        DIFFERENCE
    }
}

// ------------------------------------------------------------------------------------------------
// The paired comparison
// ------------------------------------------------------------------------------------------------
