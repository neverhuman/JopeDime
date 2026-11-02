use super::*;

pub(super) fn load_csv(path: &str) -> Result<(Rows, Vec<f64>), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut x: Rows = Vec::new();
    let mut y = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut row: Vec<f64> = Vec::new();
        for cell in line.split(',') {
            let v: f64 = cell
                .trim()
                .parse()
                .map_err(|e| format!("{path}:{}: {e}", line_no + 1))?;
            row.push(v);
        }
        if row.len() < 2 {
            return Err(format!("{path}:{}: fewer than 2 columns", line_no + 1));
        }
        y.push(row.pop().expect("row has columns"));
        x.push(row);
    }
    if x.is_empty() {
        return Err(format!("{path}: no rows"));
    }
    let width = x[0].len();
    if x.iter().any(|r| r.len() != width) {
        return Err(format!("{path}: ragged rows"));
    }
    Ok((x, y))
}

pub(super) fn r2(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    if ss_tot <= 0.0 {
        return None;
    }
    Some(1.0 - ss_res / ss_tot)
}

pub(super) fn rmse(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mse: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    Some(mse.sqrt())
}

/// Copied from train_only_study.rs::paired_robust_fitness (mean - 0.25*sd,
/// scores clipped to >= -1 so one catastrophic split cannot dominate).
pub(super) fn robust_fitness(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return -1.0;
    }
    let clipped: Vec<f64> = scores.iter().map(|v| v.max(-1.0)).collect();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let sd =
        (clipped.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / clipped.len() as f64).sqrt();
    mean - 0.25 * sd.min(1.0)
}

pub(super) fn worst_split(scores: &[f64]) -> f64 {
    scores
        .iter()
        .map(|v| v.max(-1.0))
        .fold(f64::INFINITY, f64::min)
}

/// rank_scored ordering from train_only_study.rs: fitness desc, worst-split
/// desc, complexity asc, semantic asc (fully deterministic).
pub(super) fn rank(members: &mut [Member]) {
    members.sort_by(|a, b| {
        b.fitness_complete
            .cmp(&a.fitness_complete)
            .then_with(|| robust_fitness(&b.scores).total_cmp(&robust_fitness(&a.scores)))
            .then_with(|| worst_split(&b.scores).total_cmp(&worst_split(&a.scores)))
            .then_with(|| a.complexity.cmp(&b.complexity))
            .then_with(|| a.semantic.cmp(&b.semantic))
    });
}

/// Deterministic stratified index split: rows sorted by target, sliced into
/// deciles, each decile shuffled with its own stream, the leading `hold_frac`
/// of each shuffled decile becoming the validation fold. Fresh `seed` per
/// generation gives a distinct split every time (train_only_study discipline).
pub(super) fn split_8020(y: &[f64], seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x5350_4c49_5430 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        let hold = (ids.len() as f64 * 0.2).round().max(1.0) as usize;
        for (k, id) in ids.into_iter().enumerate() {
            if k < hold {
                valid.push(id);
            } else {
                train.push(id);
            }
        }
    }
    if train.is_empty() || valid.is_empty() {
        let mid = y.len() * 4 / 5;
        return ((0..mid).collect(), (mid..y.len()).collect());
    }
    (train, valid)
}

/// Legacy V1 fold-count adaptation. Strict V2 never changes the declared fold
/// count: an infeasible plan is a structural failure rather than a reason to
/// substitute leave-one-out CV.
pub(super) fn usable_folds(n: usize, configured_k: usize) -> usize {
    if n < 2 {
        return 0;
    }
    let k = configured_k.max(2).min(n);
    if n < 2 * k {
        n
    } else {
        k
    }
}

/// Deterministic stratified k-fold: per-decile shuffle, then deal round-robin.
pub(super) fn kfold(y: &[f64], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut folds = vec![Vec::new(); k];
    let mut card = 0usize;
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x004b_464f_4c44 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        for id in ids {
            folds[card % k].push(id);
            card += 1;
        }
    }
    folds
}

pub(super) fn validate_strict_fold_plan(
    row_count: usize,
    folds: &[Vec<usize>],
) -> Result<(), String> {
    if folds.len() < 2 {
        return Err("strict pooled CV requires at least two folds".to_string());
    }
    let mut coverage = vec![0u8; row_count];
    for (fold_index, fold) in folds.iter().enumerate() {
        if fold.len() < STRICT_MIN_HOLD_ROWS {
            return Err(format!(
                "fold {fold_index} has {} hold rows; minimum is {STRICT_MIN_HOLD_ROWS}",
                fold.len()
            ));
        }
        let fit_rows = row_count.saturating_sub(fold.len());
        if fit_rows < STRICT_MIN_FIT_ROWS {
            return Err(format!(
                "fold {fold_index} has {fit_rows} fit rows; minimum is {STRICT_MIN_FIT_ROWS}"
            ));
        }
        for &row in fold {
            let Some(count) = coverage.get_mut(row) else {
                return Err(format!("fold {fold_index} contains out-of-range row {row}"));
            };
            *count = count.saturating_add(1);
        }
    }
    if let Some((row, count)) = coverage.iter().enumerate().find(|(_, count)| **count != 1) {
        return Err(format!(
            "row {row} appears in {count} hold folds; expected exactly one"
        ));
    }
    Ok(())
}

/// Complete pooled OOF R^2 in original row order. For a fixed truth vector,
/// ranking by this value is exactly equivalent to ranking by pooled OOF RMSE.
/// Any absent, malformed, duplicated, or nonfinite fold fails closed.
pub(super) fn complete_pooled_oof_r2(
    y: &[f64],
    folds: &[Vec<usize>],
    fold_predictions: &[Option<&[f64]>],
) -> Option<f64> {
    if fold_predictions.len() != folds.len() || y.is_empty() {
        return None;
    }
    let mut predictions = vec![f64::NAN; y.len()];
    let mut covered = vec![false; y.len()];
    for (fold, maybe_predictions) in folds.iter().zip(fold_predictions) {
        let values = maybe_predictions.as_ref()?;
        if values.len() != fold.len() {
            return None;
        }
        for (&row, &prediction) in fold.iter().zip(values.iter()) {
            if row >= y.len() || covered[row] || !prediction.is_finite() {
                return None;
            }
            predictions[row] = prediction;
            covered[row] = true;
        }
    }
    if covered.iter().any(|covered| !covered) {
        return None;
    }
    r2(y, &predictions).map(|value| value.max(-1.0))
}

pub(super) fn replace_generation_outcomes(
    population: &mut [Member],
    outcomes: Vec<Option<f64>>,
) -> usize {
    assert_eq!(population.len(), outcomes.len());
    let mut invalid = 0usize;
    for (member, outcome) in population.iter_mut().zip(outcomes) {
        member.scores.clear();
        member.fitness_complete = outcome.is_some();
        if !member.fitness_complete {
            invalid += 1;
        }
        member.scores.push(outcome.unwrap_or(-1.0));
    }
    invalid
}

pub(super) fn expected_search_evaluations(schedule: &[(usize, usize)], folds: usize) -> usize {
    schedule.iter().map(|(width, _)| width * folds).sum()
}

/// T5: deterministic per-generation perturbation of the TRAIN-side rows only.
/// Even generations: Gaussian column noise at 5% of column SD. Odd: 10% row
/// dropout (rows zero-weighted by exclusion from train_ids). Validation rows
/// are always untouched. Returns the CSV path children should load and the
/// possibly-thinned train ids.
pub(super) fn perturbed_generation_csv(
    x: &Rows,
    y: &[f64],
    train_ids: &[usize],
    generation: usize,
    tag: &str,
) -> Result<(String, Vec<usize>), String> {
    let seed = seed_from(&format!("screen:perturb:{tag}:{generation}"));
    let mut rng = Pcg32::new(seed, 0x5045_5254);
    let cols = x[0].len();
    let mut out = x.clone();
    let mut kept: Vec<usize> = train_ids.to_vec();
    if generation.is_multiple_of(2) {
        let mut sd = vec![0.0f64; cols];
        for c in 0..cols {
            let vals: Vec<f64> = train_ids.iter().map(|&i| x[i][c]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            sd[c] =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        }
        for &i in train_ids {
            for c in 0..cols {
                out[i][c] += 0.05 * sd[c] * rng.gaussian();
            }
        }
    } else {
        kept = train_ids
            .iter()
            .copied()
            .filter(|_| rng.next_f64() >= 0.10)
            .collect();
        if kept.len() < train_ids.len() / 2 {
            kept = train_ids.to_vec();
        }
    }
    let path = format!(".perturbed-gen{generation}.csv");
    let mut text = String::new();
    for (i, row) in out.iter().enumerate() {
        for v in row {
            text.push_str(&format!("{v},"));
        }
        text.push_str(&format!(
            "{}
",
            y[i]
        ));
    }
    fs::write(&path, text).map_err(|e| e.to_string())?;
    Ok((path, kept))
}

pub(super) fn subset(x: &Rows, y: &[f64], ids: &[usize]) -> (Rows, Vec<f64>) {
    (
        ids.iter().map(|&i| x[i].clone()).collect(),
        ids.iter().map(|&i| y[i]).collect(),
    )
}

/// Fit on the train subset, score R^2 on the validation subset. Any failure
/// (parse, fit, nonfinite, degenerate target) scores the clip floor -1.0 so a
/// broken gene ranks last instead of aborting the run.
pub(super) fn eval_on(
    member: &Member,
    x: &Rows,
    y: &[f64],
    train_ids: &[usize],
    valid_ids: &[usize],
) -> f64 {
    let Ok(genome) = GenomeIr::parse(&member.canonical) else {
        return -1.0;
    };
    let (tx, ty) = subset(x, y, train_ids);
    let (vx, vy) = subset(x, y, valid_ids);
    match model::fit_predict(&genome, &tx, &ty, &vx) {
        Ok(pred) => r2(&vy, &pred).map_or(-1.0, |v| v.max(-1.0)),
        Err(_) => -1.0,
    }
}

/// Fit on the train subset and return the raw validation predictions, so the
/// parent can pool them across folds. Pooled out-of-fold R^2 is a single ratio
/// over every row; the per-fold mean this replaces normalises each fold by its
/// own ss_tot, which on small folds rewards a gene that drew a lucky slice.
pub(super) fn predict_on(
    member: &Member,
    x: &Rows,
    y: &[f64],
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Option<Vec<f64>> {
    let genome = GenomeIr::parse(&member.canonical).ok()?;
    let (tx, ty) = subset(x, y, train_ids);
    let (vx, _) = subset(x, y, valid_ids);
    model::fit_predict(&genome, &tx, &ty, &vx).ok()
}

/// Child mode: read {genome, train_ids, valid_ids} from stdin, print r2.
/// With "want":"pred" the child prints a JSON array of validation predictions
/// instead, which is what the pooled-OOF finale consumes.
pub(super) fn eval_one(train_path: &str) -> Result<(), String> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| e.to_string())?;
    let req: serde_json::Value = serde_json::from_str(&buf).map_err(|e| e.to_string())?;
    let genome = req["genome"].as_str().ok_or("genome missing")?;
    let ids = |k: &str| -> Vec<usize> {
        req[k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default()
    };
    let (x, y) = load_csv(train_path)?;
    let member = Member {
        canonical: genome.to_string(),
        semantic: String::new(),
        complexity: 0,
        origin: String::new(),
        scores: Vec::new(),
        fitness_complete: false,
    };
    if req["want"].as_str() == Some("pred") {
        let pred = predict_on(&member, &x, &y, &ids("train_ids"), &ids("valid_ids"))
            .ok_or("fit_predict failed")?;
        println!(
            "{}",
            serde_json::to_string(&pred).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let score = eval_on(&member, &x, &y, &ids("train_ids"), &ids("valid_ids"));
    println!("{score}");
    Ok(())
}

/// Why a child evaluation produced no usable number. A model that fits and
/// scores badly is NOT a failure -- it earns the -1.0 clip floor honestly.
/// These variants are the cases where we learned nothing about the gene, and
/// conflating them with a bad score systematically punishes whatever is
/// slowest (wide feature sets, heavy heads) for infrastructure reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) fn strict_generation_completion(
    generation: usize,
    invalid_members: usize,
    failures: EvalFailureCounts,
) -> Result<(), String> {
    if invalid_members == 0 {
        return Ok(());
    }
    Err(format!(
        "generation {generation} contains {invalid_members} members without complete pooled-OOF fitness; fold failures: spawn={}, timeout={}, crashed={}, unparseable={}, invalid_oof={}",
        failures.spawn,
        failures.timeout,
        failures.crashed,
        failures.unparseable,
        failures.invalid_oof,
    ))
}

pub(super) fn eval_timeout_seconds() -> u64 {
    std::env::var("JSS_EVAL_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30)
}

/// Parent mode: run one evaluation in a killable child with a hard deadline
/// (campaign parity: golden42 enforced a 30 s per-candidate timeout at the
/// worker level). Returns the child's raw stdout.
pub(super) fn run_eval_child(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
    want_pred: bool,
) -> Result<String, EvalFailure> {
    use std::io::Write;
    let timeout = eval_timeout_seconds();
    let exe = std::env::current_exe().map_err(|_| EvalFailure::Spawn)?;
    let mut child = std::process::Command::new(exe)
        .arg("eval-one")
        .arg("--train")
        .arg(train_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|_| EvalFailure::Spawn)?;
    let mut payload = serde_json::json!({
        "genome": genome, "train_ids": train_ids, "valid_ids": valid_ids,
    });
    if want_pred {
        payload["want"] = serde_json::Value::String("pred".to_string());
    }
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(payload.to_string().as_bytes());
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(EvalFailure::Crashed);
                }
                let mut out = String::new();
                use std::io::Read;
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                return Ok(out);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(EvalFailure::Timeout);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(EvalFailure::Crashed);
            }
        }
    }
}

/// Score, distinguishing "did not run" (None) from "ran and was bad" (Some).
pub(super) fn eval_checked(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Result<f64, EvalFailure> {
    let out = run_eval_child(train_path, genome, train_ids, valid_ids, false)?;
    out.trim()
        .parse::<f64>()
        .map(|v| v.max(-1.0))
        .map_err(|_| EvalFailure::Unparseable)
}

/// Validation predictions for one fold, for the pooled-OOF finale.
pub(super) fn eval_predictions(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Result<Vec<f64>, EvalFailure> {
    let out = run_eval_child(train_path, genome, train_ids, valid_ids, true)?;
    serde_json::from_str::<Vec<f64>>(out.trim()).map_err(|_| EvalFailure::Unparseable)
}

/// Legacy scoring surface: every failure becomes the clip floor. Retained so
/// the search loop's recorded behaviour is unchanged.
pub(super) fn eval_with_timeout(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> f64 {
    eval_checked(train_path, genome, train_ids, valid_ids).unwrap_or(-1.0)
}
