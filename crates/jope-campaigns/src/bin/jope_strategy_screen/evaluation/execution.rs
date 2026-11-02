use super::*;

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
