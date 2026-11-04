fn outcome(left: f64, right: f64, minimize: bool) -> f64 {
    let largest = left.abs().max(right.abs()).max(1.0);
    if (left - right).abs() <= TIE * largest {
        return 0.5;
    }
    if (left < right) == minimize {
        1.0
    } else {
        0.0
    }
}

/// Match count and outcome sum for each unordered method pair (the likelihood's sufficient
/// statistics), in the order Python's dict first saw each pair.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pairs {
    order: Vec<(String, String)>,
    stats: HashMap<(String, String), (u64, f64)>,
}

impl Pairs {
    fn add(&mut self, a: &str, b: &str, outcome: f64) {
        let key = (a.to_owned(), b.to_owned());
        match self.stats.get_mut(&key) {
            Some((count, total)) => {
                *count += 1;
                *total += outcome;
            }
            None => {
                self.order.push(key.clone());
                self.stats.insert(key, (1, 0.0 + outcome));
            }
        }
    }

    /// `(count, total)` of the pair `(a, b)` with `a < b`.
    pub fn get(&self, a: &str, b: &str) -> Option<(u64, f64)> {
        self.stats.get(&(a.to_owned(), b.to_owned())).copied()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Pairs in first-seen order.
    pub fn iter(&self) -> impl Iterator<Item = (&(String, String), (u64, f64))> {
        self.order.iter().map(|key| (key, self.stats[key]))
    }
}

/// One equal-weight match per dataset and pair of scored methods.
pub fn pair_statistics(
    by_dataset: &BTreeMap<String, BTreeMap<String, f64>>,
    minimize: bool,
) -> Pairs {
    let mut pairs = Pairs::default();
    for scores in by_dataset.values() {
        let scores: Vec<(&String, &f64)> = scores.iter().collect();
        for (i, (left, left_value)) in scores.iter().enumerate() {
            for (right, right_value) in &scores[i + 1..] {
                pairs.add(left, right, outcome(**left_value, **right_value, minimize));
            }
        }
    }
    pairs
}

/// Python's two-argument `min`/`max` (the first argument wins ties).
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Regularized Bradley–Terry on the Elo scale by synchronous diagonal Newton steps. Returns the
/// ratings by method, the iterations run and whether the tolerance was reached within `cap`.
pub fn fit(pairs: &Pairs, cap: usize) -> (BTreeMap<String, f64>, usize, bool) {
    let scale = scale();
    let methods: BTreeSet<&str> = pairs
        .order
        .iter()
        .flat_map(|(a, b)| [a.as_str(), b.as_str()])
        .collect();
    let methods: Vec<&str> = methods.into_iter().collect();
    let position: HashMap<&str, usize> = methods.iter().enumerate().map(|(i, m)| (*m, i)).collect();
    let mut keys: Vec<&(String, String)> = pairs.order.iter().collect();
    keys.sort();
    let table: Vec<(usize, usize, f64, f64)> = keys
        .into_iter()
        .map(|key| {
            let (count, total) = pairs.stats[key];
            (
                position[key.0.as_str()],
                position[key.1.as_str()],
                count as f64,
                total,
            )
        })
        .collect();
    let mut ratings = vec![CENTER; methods.len()];
    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..cap {
        let mut gradient = vec![0.0_f64; methods.len()];
        let mut information = vec![0.0_f64; methods.len()];
        for &(a, b, count, total) in &table {
            let delta = py_max(-700.0, py_min(700.0, scale * (ratings[a] - ratings[b])));
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = total - count * probability;
            let curvature = count * probability * (1.0 - probability);
            gradient[a] += residual;
            gradient[b] -= residual;
            information[a] += curvature;
            information[b] += curvature;
        }
        let steps: Vec<f64> = (0..methods.len())
            .map(|m| {
                let newton = (scale * gradient[m] - REGULARIZATION * (ratings[m] - CENTER))
                    / (scale * scale * information[m] + REGULARIZATION);
                py_max(-CLAMP, py_min(CLAMP, newton))
            })
            .collect();
        for (rating, step) in ratings.iter_mut().zip(&steps) {
            *rating += step;
        }
        iterations += 1;
        let largest = steps
            .iter()
            .fold(0.0_f64, |m, s| if s.abs() > m { s.abs() } else { m });
        if largest < TOLERANCE {
            converged = true;
            break;
        }
    }
    let shift = if ratings.is_empty() {
        0.0
    } else {
        CENTER - py::fsum(ratings.iter().copied()).expect("finite ratings") / ratings.len() as f64
    };
    let fitted = methods
        .iter()
        .zip(&ratings)
        .map(|(m, r)| ((*m).to_owned(), r + shift))
        .collect();
    (fitted, iterations, converged)
}

/// Methods with equal match counts against every other method and equal total score.
///
/// Bradley–Terry ratings depend on the data only through match counts and each method's total
/// score, so such methods have equal ratings in exact arithmetic even when their per-opponent
/// records differ. Their published order is decided by floating-point accumulation order and the
/// name tie-break, never by a comparison.
pub fn exchangeable_groups(pairs: &Pairs) -> Vec<Vec<String>> {
    let mut counts: HashMap<&str, HashMap<&str, u64>> = HashMap::new();
    let mut scores: Vec<(&str, f64)> = Vec::new();
    let mut score_at: HashMap<&str, usize> = HashMap::new();
    for ((a, b), (count, total)) in pairs.iter() {
        counts.entry(a).or_default().insert(b, count);
        counts.entry(b).or_default().insert(a, count);
        for (method, amount) in [(a.as_str(), None), (b.as_str(), Some(count))] {
            let slot = *score_at.entry(method).or_insert_with(|| {
                scores.push((method, 0.0));
                scores.len() - 1
            });
            let current = scores[slot].1;
            // scores[a] + total, and (scores[b] + count) - total, as Python evaluated them.
            scores[slot].1 = match amount {
                None => current + total,
                Some(count) => current + count as f64 - total,
            };
        }
    }
    let related = |m: &str, n: &str| {
        let (cm, cn) = (&counts[m], &counts[n]);
        cm.keys()
            .chain(cn.keys())
            .filter(|k| **k != m && **k != n)
            .all(|k| cm.get(k).copied().unwrap_or(0) == cn.get(k).copied().unwrap_or(0))
    };
    let mut distinct: Vec<f64> = scores.iter().map(|(_, s)| *s).collect();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("finite scores"));
    distinct.dedup_by(|a, b| a == b);
    let mut groups = Vec::new();
    for score in distinct {
        let mut members: Vec<&str> = scores
            .iter()
            .filter(|(_, s)| *s == score)
            .map(|(m, _)| *m)
            .collect();
        members.sort_unstable();
        let mut classes: Vec<Vec<&str>> = Vec::new();
        for method in members {
            match classes
                .iter_mut()
                .find(|class| class.iter().all(|n| related(method, n)))
            {
                Some(home) => home.push(method),
                None => classes.push(vec![method]),
            }
        }
        groups.extend(
            classes
                .into_iter()
                .filter(|c| c.len() > 1)
                .map(|c| c.into_iter().map(str::to_owned).collect::<Vec<String>>()),
        );
    }
    groups.sort();
    groups
}

/// Sizes of the connected components of the comparison graph. Order between components is
/// imposed by the regularizer and centring, not by any comparison.
pub fn components(pairs: &Pairs) -> Vec<usize> {
    let mut parent: HashMap<&str, &str> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    fn find<'a>(
        parent: &mut HashMap<&'a str, &'a str>,
        order: &mut Vec<&'a str>,
        node: &'a str,
    ) -> &'a str {
        if !parent.contains_key(node) {
            parent.insert(node, node);
            order.push(node);
        }
        let mut node = node;
        while parent[node] != node {
            let grand = parent[parent[node]];
            parent.insert(node, grand);
            node = grand;
        }
        node
    }
    for (a, b) in &pairs.order {
        let rb = find(&mut parent, &mut order, b);
        let ra = find(&mut parent, &mut order, a);
        parent.insert(ra, rb);
    }
    let mut sizes: HashMap<&str, usize> = HashMap::new();
    for node in order.clone() {
        let root = find(&mut parent, &mut order, node);
        *sizes.entry(root).or_default() += 1;
    }
    let mut sizes: Vec<usize> = sizes.into_values().collect();
    sizes.sort_unstable();
    sizes
}

/// Settings of one ladder.
pub struct LadderOptions<'a> {
    pub ladder_kind: &'a str,
    pub minimize: bool,
    pub names: &'a HashMap<Key, Value>,
    pub pending: &'a HashMap<Key, i64>,
    pub total_datasets: usize,
}

/// A replayed ladder.
#[derive(Clone, Debug)]
pub struct Ladder {
    /// Leaderboard rows (every cell as Python wrote it; ranks and counts are integers).
    pub rows: Vec<Value>,
    /// The leaderboard CSV bytes as `csv.writer(lineterminator="\n")` wrote them.
    pub csv: String,
    pub ratings: BTreeMap<String, f64>,
    pub iterations: usize,
    pub converged: bool,
    pub rankable_rows: usize,
    pub excluded_rows: usize,
    pub datasets_scored: usize,
    pub matches: u64,
    pub components: Vec<usize>,
    pub exchangeable: Vec<Vec<String>>,
    pub identical_scores: Vec<Vec<String>>,
    pub pairs: Pairs,
}

/// The leaderboard of `rows` (every cell spelled as Python's `round`/`repr` write it).
pub fn ladder(rows: &[ScoreRow], options: &LadderOptions<'_>) -> Result<Ladder> {
    let scope = scope_of(options.ladder_kind).ok_or_else(|| shape("ladder kind"))?;
    let mut by_dataset: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut protocols: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut valid: HashMap<String, bool> = HashMap::new();
    let mut rankable_rows = 0;
    let mut excluded = 0;
    for row in rows {
        if !row.rankable() {
            continue;
        }
        let is_valid = row.cell("benchmark_valid") == "True";
        if options.ladder_kind == "benchmark-valid" && !is_valid {
            excluded += 1;
            continue;
        }
        let method = row.cell("method_id").to_owned();
        by_dataset
            .entry(row.cell("dataset_id").to_owned())
            .or_default()
            .insert(method.clone(), float_cell(row.cell("value"))?);
        protocols
            .entry(method.clone())
            .or_default()
            .insert(row.cell("protocol").to_owned());
        let before = valid.get(&method).copied().unwrap_or(true);
        valid.insert(method, before && is_valid);
        rankable_rows += 1;
    }
    let pairs = pair_statistics(&by_dataset, options.minimize);
    let (ratings, iterations, converged) = fit(&pairs, ITERATIONS);
    let mut ranks: HashMap<&str, Vec<u64>> = HashMap::new();
    let mut wins: HashMap<&str, i64> = HashMap::new();
    for scores in by_dataset.values() {
        let mut ordered: Vec<(&String, f64)> = scores.iter().map(|(m, v)| (m, *v)).collect();
        ordered.sort_by(|a, b| {
            let (ka, kb) = if options.minimize {
                (a.1, b.1)
            } else {
                (-a.1, -b.1)
            };
            ka.partial_cmp(&kb)
                .expect("finite values")
                .then_with(|| a.0.cmp(b.0))
        });
        for (rank, (method, _)) in ordered.iter().enumerate() {
            ranks
                .entry(method.as_str())
                .or_default()
                .push(rank as u64 + 1);
        }
        *wins.entry(ordered[0].0.as_str()).or_default() += 1;
    }
    let mut ranked: Vec<(&String, f64)> = ratings.iter().map(|(m, r)| (m, *r)).collect();
    ranked.sort_by(|a, b| {
        (-a.1)
            .partial_cmp(&-b.1)
            .expect("finite ratings")
            .then_with(|| a.0.cmp(b.0))
    });
    let total = options.total_datasets;
    let mut board = Vec::new();
    let mut full = 0;
    for (position, (method, rating)) in ranked.iter().enumerate() {
        let method_ranks = ranks.get(method.as_str()).map_or(&[][..], Vec::as_slice);
        let scored = method_ranks.len();
        let covered = scored == total;
        full += usize::from(covered);
        let cells = options.pending.get(&Key::str(method)).copied().unwrap_or(0);
        if total == 0 {
            return Err(shape(
                "float division by zero: the track declares no datasets",
            ));
        }
        let average = method_ranks.iter().sum::<u64>() as f64 / scored as f64;
        let coverage = 100.0 * scored as f64 / total as f64;
        let name = options
            .names
            .get(&Key::str(method))
            .cloned()
            .unwrap_or_else(|| Value::from(method.as_str()));
        let mut status = format!("{scored}/{total}");
        if cells != 0 {
            status.push_str(&format!(" · {cells} pending"));
        }
        let protocol_list: Vec<&str> = protocols
            .get(method.as_str())
            .map(|p| p.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let mut row = Dict::new();
        row.insert("elo_rank", (position + 1).into());
        row.insert(
            "full_coverage_rank",
            if covered {
                full.to_string()
            } else {
                String::new()
            }
            .into(),
        );
        row.insert("method_id", method.as_str().into());
        row.insert("method", name);
        row.insert("elo", pyjson::float_repr(py::round(*rating, 1)).into());
        row.insert(
            "average_rank",
            pyjson::float_repr(py::round(average, 3)).into(),
        );
        row.insert(
            "dataset_wins",
            Value::Int(wins.get(method.as_str()).copied().unwrap_or(0)),
        );
        row.insert("datasets_scored", scored.into());
        row.insert("datasets_total", total.into());
        row.insert(
            "coverage_pct",
            pyjson::float_repr(py::round(coverage, 1)).into(),
        );
        row.insert(
            "full_coverage",
            if covered { "True" } else { "False" }.into(),
        );
        row.insert("pending_cells", Value::Int(cells));
        row.insert("coverage_status", status.into());
        row.insert("protocols", protocol_list.join("|").into());
        row.insert(
            "benchmark_valid",
            if valid.get(method.as_str()).copied().unwrap_or(false) {
                "True"
            } else {
                "False"
            }
            .into(),
        );
        row.insert("scope", scope.into());
        board.push(Value::Dict(row));
    }
    let mut csv = String::new();
    pycsv::write_row(&mut csv, &LEADERBOARD_COLUMNS);
    for row in &board {
        let cells: Vec<String> = LEADERBOARD_COLUMNS
            .iter()
            .map(|c| pycsv::cell(row.get(c).expect("every column")))
            .collect();
        pycsv::write_row(&mut csv, &cells);
    }
    // Methods with identical score vectors (possible aliases of one another).
    let mut aliases: BTreeMap<Vec<(String, u64)>, Vec<String>> = BTreeMap::new();
    for method in ratings.keys() {
        let signature: Vec<(String, u64)> = by_dataset
            .iter()
            .filter_map(|(dataset, scores)| {
                scores
                    .get(method)
                    .map(|v| (dataset.clone(), (v + 0.0).to_bits()))
            })
            .collect();
        aliases.entry(signature).or_default().push(method.clone());
    }
    let mut identical: Vec<Vec<String>> = aliases.into_values().filter(|g| g.len() > 1).collect();
    identical.sort();
    Ok(Ladder {
        rows: board,
        csv,
        ratings: ratings.clone(),
        iterations,
        converged,
        rankable_rows,
        excluded_rows: excluded,
        datasets_scored: by_dataset.len(),
        matches: pairs.iter().map(|(_, (count, _))| count).sum(),
        components: components(&pairs),
        exchangeable: exchangeable_groups(&pairs),
        identical_scores: identical,
        pairs,
    })
}
