use super::*;

/// The generation-0 seeds of the informational run: programs with expression stages.
const EXPRESSION_SEEDS: [&str; 15] = [
    // The expression programs of `compat_freeze`'s NAMED_PROGRAMS: they read columns 0 to 4.
    "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))",
    "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))",
    "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))",
    "(pipe X (map expr=\"abs(x) + sqrt(abs(c0)) - log1p(abs(x))\" append=true) (derive expr=\"sin(c0) + cos(c1) * tanh(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sigmoid(x) - min(x, c1) + max(x, c2)\") (derive expr=\"clip(c0, -1, 1) + step(c1, 0.5)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(x, -3, 3)) + log(abs(x) + 1)\" append=true) (derive expr=\"asinh(c0) + atan(c1) - erf(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sign(x) * floor(x) + round(x)\") (derive expr=\"row_sum - row_l1 + row_l2\" append=true) (ridge))",
    "(pipe X (map expr=\"x - row_mean + row_std\" append=true) (derive expr=\"row_min * row_max + c0 / (c1 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - (-x) * 0.5 + x / (c0 + 2)\") (derive expr=\"c3^3 - c4*c4*c4 + c2^0.5\" append=true) (elasticnet alpha=0.005 l1_ratio=0.7))",
    // `x` outside a map: an illegal scope everywhere.
    "(pipe X (derive expr=\"x\" append=true) (ridge))",
    // Guarded operations the scaled data reaches: log of zero, division by zero, sqrt of a
    // negative difference.
    "(pipe X (derive expr=\"log(col(0))\" append=true) (ridge))",
    "(pipe X (derive expr=\"1/col(1)\" append=true) (ridge))",
    "(pipe X (derive expr=\"sqrt(col(0) - col(1))\" append=true) (ridge))",
];

/// Every regression dataset of `datasets/test`, by name, with the rows of its `train.csv` (never
/// `test.csv`), checked against the sha256 its metadata records.
fn test_datasets() -> Vec<(String, Rows, Vec<f64>)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("{}: {error}", root.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("metadata.json").is_file() && path.join("train.csv").is_file())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .filter_map(|dir| {
            let metadata: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json")).unwrap())
                    .unwrap();
            if metadata["task"] != "regression" {
                return None;
            }
            let bytes = std::fs::read(dir.join("train.csv")).unwrap();
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            assert_eq!(
                metadata["files"]["train"]["sha256"].as_str(),
                Some(digest.as_str()),
                "{}: train.csv does not match its metadata",
                dir.display()
            );
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for line in std::str::from_utf8(&bytes).unwrap().lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let mut row: Vec<f64> = line
                    .split(',')
                    .map(|cell| cell.trim().parse::<f64>().unwrap())
                    .collect();
                y.push(row.pop().unwrap());
                x.push(row);
            }
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            Some((name, x, y))
        })
        .collect()
}

/// The kind of a rejection reason: the scope class, or the Hard classes.
fn reason_kind(reason: &str) -> String {
    for kind in ["COLUMN_OOB", "ILLEGAL_SCOPE"] {
        if reason.starts_with("scope:") && reason.contains(kind) {
            return format!("scope {kind}");
        }
    }
    if reason.starts_with("scope:") {
        return "scope, unchecked".to_string();
    }
    reason.to_string()
}

/// The best program of a run, measured after it: how many expression stages it has, whether its
/// rung-0 program passes the scope check, and its rung-0 rates over every traced pass (with
/// `hard` when a pass raised a Hard class), replayed through the public entry points. This is how
/// the run under `Legacy`, which measures nothing, is compared with the others.
fn best_program_faults(outcome: &RaceOutcome, x: &Rows, y: &[f64]) -> String {
    let Some(best) = outcome.ranked.first() else {
        return "-".to_string();
    };
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(&best.dsl).unwrap() else {
        unreachable!("the race ranks pipelines only")
    };
    let expressions = pipeline
        .stages
        .iter()
        .filter(|stage| stage.op == "map" || stage.op == "derive")
        .count();
    let (surrogate, _) = rung0_trace_identity(&best.dsl).unwrap();
    let scope = match validate_scope(&GenomeIr::parse(&surrogate).unwrap(), x[0].len()) {
        Ok(()) => "in scope".to_string(),
        Err(error) => format!("scope: {error}"),
    };
    let rates = replay_rung0(outcome, x, y, &best.dsl).map_or_else(
        || "not fitted".to_string(),
        |replay| {
            format!(
                "{:.4}/{:.4}{}",
                replay.all.domain_rate(),
                replay.all.saturation_rate(),
                if replay.all.is_hard() { " hard" } else { "" }
            )
        },
    );
    format!("{expressions} expression stages, {scope}, rates {rates}")
}

/// Why rung 0 of a rejecting policy would reject a raced candidate, measured after the run the
/// way the policy measures it while racing: its rung-0 program fails the scope check (`scope`),
/// or a pass of its traced rung-0 evaluation raises a Hard class (`hard`), checked in the policy's
/// order on the published rung-0 rows and folds. A fold that cannot be fitted or predicted ends
/// the replay without a verdict, as it ends the policy's evaluation. `None` when rung 0 admits the
/// candidate. This measures every policy's candidates alike, `Legacy`'s included, which rejects
/// nothing and scores such a candidate like any other.
fn rung0_rejection(outcome: &RaceOutcome, x: &Rows, y: &[f64], dsl: &str) -> Option<&'static str> {
    let (surrogate, _) = rung0_trace_identity(dsl).ok()?;
    let program = GenomeIr::parse(&surrogate).ok()?;
    if validate_scope(&program, x[0].len()).is_err() {
        return Some("scope");
    }
    let mut totals = FaultTotals::default();
    for fold in 0..RACE_RUNG_FOLDS[0] {
        let (mut xtr, mut ytr, mut xte) = (Vec::new(), Vec::new(), Vec::new());
        for (&row, &assigned) in outcome.rung0_rows.iter().zip(&outcome.rung0_assignments) {
            if assigned == fold {
                xte.push(x[row].clone());
            } else {
                xtr.push(x[row].clone());
                ytr.push(y[row]);
            }
        }
        let (model, training) = fit_traced(&program, &xtr, &ytr).ok()?;
        totals.add_pass(&training);
        if totals.is_hard() {
            return Some("hard");
        }
        let (_, held_out) = predict_traced(&model, &xte).ok()?;
        totals.add_pass(&held_out);
        if totals.is_hard() {
            return Some("hard");
        }
    }
    None
}

/// Informational only, with no quality gate (plan WP-1.5 acceptance): the rates of `Legacy`,
/// `RejectHard` and `Penalize{0.05, 0, tie-break}` in one seeded lite race per regression dataset
/// of `datasets/test` (collection `test-10`, the real-data half of fixture set D5), seeded with
/// expression programs, since the default vocabulary never generates an expression stage and
/// would measure nothing. Prints one line per run. The policy's own columns (its rejections, their
/// reasons and its domain-rate counters) are `-` under `Legacy`, which checks and measures
/// nothing; the invalid candidates, all raced ones and the offspring among them, are measured
/// after every run alike ([`rung0_rejection`]).
#[test]
#[ignore = "informational: fault-policy rates on datasets/test for the WP-1.5 receipt; run with --run-ignored only"]
fn informational_fault_policy_rates_on_the_test_datasets() {
    let datasets = test_datasets();
    assert_eq!(datasets.len(), 10);
    let policies = [
        ("legacy", FaultPolicy::Legacy),
        ("reject-hard", FaultPolicy::RejectHard),
        ("penalize-0.05-0-tie", PENALIZE),
    ];
    println!(
        "dataset | policy | features | raced | rejected | rejected offspring | reasons | invalid at rung 0: raced, offspring, kinds | pop domain rate | elite domain rate | best cv | best program measured at rung 0 | best program | seconds"
    );
    for (name, x, y) in &datasets {
        for (label, policy) in policies {
            let config = EvolveConfig {
                pop: 24,
                gens: 4,
                folds: 3,
                seed: 0x0DA7_A5E7,
                max_transforms: 3,
                time_budget_secs: 0.0,
                fitness: FitnessMode::LiteRace,
                rank_robust: true,
                seed_genomes: EXPRESSION_SEEDS.map(str::to_string).to_vec(),
                fault_policy: policy,
                ..EvolveConfig::default()
            };
            let log = Log::default();
            let started = std::time::Instant::now();
            let outcome = evolve_lite_race(x, y, &feat_math::r2, &config, &log);
            let seconds = started.elapsed().as_secs_f64();
            let rejections = log.rejections();
            let rejected: std::collections::BTreeSet<&str> =
                rejections.iter().map(|(dsl, _)| dsl.as_str()).collect();
            let offspring: Vec<&str> = outcome
                .candidates
                .iter()
                .filter(|candidate| candidate.r0_generation.is_some_and(|gen| gen > 0))
                .map(|candidate| candidate.dsl.as_str())
                .collect();
            let rejected_offspring = offspring
                .iter()
                .filter(|dsl| rejected.contains(**dsl))
                .count();
            let mut reasons = std::collections::BTreeMap::<String, usize>::new();
            for (_, reason) in &rejections {
                *reasons.entry(reason_kind(reason)).or_default() += 1;
            }
            let last = log.generations().last().cloned();
            let rate = |bits: Option<u64>| bits.map_or(f64::NAN, f64::from_bits);
            let pop_rate = rate(last.as_ref().map(|counters| counters.pop_domain_rate));
            let elite_rate = rate(last.as_ref().map(|counters| counters.elite_domain_rate));
            // `Legacy` checks nothing and measures no rate: its policy columns are `-`.
            let measured = policy != FaultPolicy::Legacy;
            let policy_cell = |value: String| if measured { value } else { "-".to_string() };
            let (mut invalid_raced, mut invalid_offspring) = (0, 0);
            let mut invalid_kinds = std::collections::BTreeMap::<&str, usize>::new();
            for candidate in &outcome.candidates {
                if let Some(kind) = rung0_rejection(&outcome, x, y, &candidate.dsl) {
                    invalid_raced += 1;
                    invalid_offspring +=
                        usize::from(candidate.r0_generation.is_some_and(|gen| gen > 0));
                    *invalid_kinds.entry(kind).or_default() += 1;
                }
            }
            let best = outcome.ranked.first();
            println!(
                "{name} | {label} | {} | {} | {} | {} | {} | {invalid_raced}/{} {invalid_offspring}/{} {invalid_kinds:?} | {} | {} | {} | {} | {} | {seconds:.1}",
                x[0].len(),
                outcome.rung_evals[0],
                policy_cell(rejected.len().to_string()),
                policy_cell(format!("{rejected_offspring}/{}", offspring.len())),
                policy_cell(format!("{reasons:?}")),
                outcome.candidates.len(),
                offspring.len(),
                policy_cell(format!("{pop_rate:.4}")),
                policy_cell(format!("{elite_rate:.4}")),
                best.map_or_else(|| "-".to_string(), |scored| format!("{:.4}", scored.cv_mean)),
                best_program_faults(&outcome, x, y),
                best.map_or("-", |scored| scored.dsl.as_str()),
            );
        }
    }
}
