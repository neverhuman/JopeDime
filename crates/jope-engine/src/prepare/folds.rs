use super::*;

/// Evaluation row selection uses only original position, split and seed.
pub fn cap_indices(
    n_train: usize,
    n_test: usize,
    task: &str,
    train: &[Vec<String>],
) -> Result<(Vec<usize>, Vec<usize>)> {
    let total = n_train + n_test;
    ensure!(
        n_train > 0 && n_test > 0,
        "both original partitions are required"
    );
    let (nt, ne) = if total <= ROW_LIMIT {
        (n_train, n_test)
    } else {
        let nt = ((ROW_LIMIT as u128 * n_train as u128) / total as u128) as usize;
        (
            nt.clamp(1, ROW_LIMIT - 1),
            ROW_LIMIT - nt.clamp(1, ROW_LIMIT - 1),
        )
    };
    if task == "timeseries" {
        ensure!(n_test < ROW_LIMIT, "sealed horizon exceeds row allowance");
        let nt = n_train.min(ROW_LIMIT - n_test);
        return Ok(((n_train - nt..n_train).collect(), (0..n_test).collect()));
    }
    let rank = |split: &str, n: usize| {
        let mut x = (0..n).collect::<Vec<_>>();
        x.sort_by_key(|i| hash(format!("{SEED}:cap:{split}:{i}").as_bytes()));
        x
    };
    let mut ti = rank("train", n_train);
    let mut ei = rank("evaluation", n_test);
    ei.truncate(ne);
    ei.sort_unstable();
    if matches!(task, "binary" | "multiclass") {
        let mut coverage = BTreeMap::new();
        for &i in &ti {
            coverage.entry(train[i].last().unwrap()).or_insert(i);
        }
        ensure!(coverage.len() <= nt, "cap cannot preserve training classes");
        let selected = coverage.into_values().collect::<BTreeSet<_>>();
        ti.retain(|i| !selected.contains(i));
        ti.truncate(nt - selected.len());
        ti.extend(selected);
    } else {
        ti.truncate(nt);
    }
    ti.sort_unstable();
    Ok((ti, ei))
}
pub fn folds(
    rows: &[Vec<String>],
    task: &str,
    horizon: usize,
    entity_groups: Option<&[String]>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    folds_with_seed(rows, task, horizon, entity_groups, SEED)
}
pub fn folds_with_seed(
    rows: &[Vec<String>],
    task: &str,
    horizon: usize,
    entity_groups: Option<&[String]>,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    ensure!(rows.len() >= 6, "insufficient history for five cuts");
    if task == "timeseries" {
        ensure!(
            horizon > 0 && rows.len() > 5 * horizon,
            "insufficient history for five rolling origins"
        );
        return Ok((0..5)
            .map(|cut| {
                let end = rows.len() - (4 - cut) * horizon;
                ((0..end - horizon).collect(), (end - horizon..end).collect())
            })
            .collect());
    }
    if let Some(g) = entity_groups {
        ensure!(g.len() == rows.len(), "entity group length mismatch");
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    // Union explicit entity IDs with feature duplicates, including conflicting labels.
    let mut uf: Vec<usize> = (0..rows.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        if p[i] != i {
            p[i] = find(p, p[i]);
        }
        p[i]
    }
    let mut seen = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        let mut keys = vec![format!(
            "x:{}",
            hash(&serde_json::to_vec(&r[..r.len() - 1])?)
        )];
        if let Some(g) = entity_groups {
            keys.push(format!("entity:{}", g[i]));
        }
        for key in keys {
            if let Some(j) = seen.insert(key, i) {
                let a = find(&mut uf, i);
                let b = find(&mut uf, j);
                uf[a] = b;
            }
        }
    }
    for i in 0..rows.len() {
        let k = find(&mut uf, i);
        groups.entry(k.to_string()).or_default().push(i);
    }
    ensure!(
        groups.len() >= 5,
        "fewer than five independent row/entity groups"
    );
    let mut strata: BTreeMap<String, Vec<Vec<usize>>> = BTreeMap::new();
    for g in groups.into_values() {
        let labels = g
            .iter()
            .map(|&i| rows[i].last().unwrap().clone())
            .collect::<BTreeSet<_>>();
        let stratum = if matches!(task, "binary" | "multiclass") {
            serde_json::to_string(&labels)?
        } else {
            String::new()
        };
        strata.entry(stratum).or_default().push(g);
    }
    let mut assignment = vec![0; rows.len()];
    let mut loads = [0usize; 5];
    for mut gs in strata.into_values() {
        gs.sort_by_key(|g| hash(format!("{seed}:fold:{:?}", g).as_bytes()));
        let mut local = [0usize; 5];
        for g in gs {
            let fold = (0..5).min_by_key(|&f| (local[f], loads[f], f)).unwrap();
            local[fold] += g.len();
            loads[fold] += g.len();
            for i in g {
                assignment[i] = fold;
            }
        }
    }
    (0..5)
        .map(|cut| {
            let (val, train): (Vec<_>, Vec<_>) =
                (0..rows.len()).partition(|&i| assignment[i] == cut);
            ensure!(
                !train.is_empty() && !val.is_empty(),
                "empty fold after grouping"
            );
            if matches!(task, "binary" | "multiclass") {
                let all = rows
                    .iter()
                    .map(|r| r.last().unwrap())
                    .collect::<BTreeSet<_>>();
                let labels = train
                    .iter()
                    .map(|&i| rows[i].last().unwrap())
                    .collect::<BTreeSet<_>>();
                ensure!(
                    all == labels,
                    "five class-covering folds impossible with these groups"
                );
                let heldout_labels = val
                    .iter()
                    .map(|&i| rows[i].last().unwrap())
                    .collect::<BTreeSet<_>>();
                ensure!(
                    all == heldout_labels,
                    "insufficient_class_groups: every held-out fold must contain every class"
                );
            }
            Ok((train, val))
        })
        .collect()
}
