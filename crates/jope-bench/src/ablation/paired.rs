use super::*;

/// One (dataset, seed) pair: A's and B's M1.
pub(crate) struct Pair<'a> {
    dataset: &'a str,
    seed: u64,
    a: &'a Best,
    b: &'a Best,
}

/// The paired comparison of M1 over `runs` ([`PAIRED_RULE`]): overall, and per dataset in the
/// order of `datasets` (dataset hashes). The pairing is one to one on (dataset, seed): a dataset
/// listed twice, a slot other than A or B, a run on a dataset not listed, two runs of one slot
/// with the same dataset and seed, and a run without its partner in the other slot are refused.
/// So is an M1 RMSE that is not finite, before any comparison or arithmetic, and a summary
/// number that would not be finite, so every number a returned [`Paired`] holds is finite.
pub fn paired(datasets: &[String], runs: &[RunRecord]) -> Result<Paired> {
    let mut listed = BTreeSet::new();
    for dataset in datasets {
        ensure!(
            listed.insert(dataset.as_str()),
            "dataset {dataset} is listed twice"
        );
    }
    let mut a_keys = BTreeSet::new();
    let mut b_runs = BTreeMap::new();
    for run in runs {
        let (dataset, seed, slot) = (run.dataset.as_str(), run.seed, run.slot);
        ensure!(
            listed.contains(dataset),
            "dataset {dataset} seed {seed}: the {slot} run is on a dataset the comparison does \
             not list"
        );
        let first = match slot {
            "A" => a_keys.insert((dataset, seed)),
            "B" => b_runs.insert((dataset, seed), run).is_none(),
            other => bail!("dataset {dataset} seed {seed}: run slot {other:?} is neither A nor B"),
        };
        ensure!(
            first,
            "dataset {dataset} seed {seed}: two {slot} runs; runs pair one to one"
        );
        // Before any bit-equality, ordering or arithmetic: two equal NaNs or infinities would
        // otherwise count as a tie, and an infinity against a finite value as a win or a loss.
        let rmse = run.metrics.m1.rmse;
        ensure!(
            rmse.is_finite(),
            "dataset {dataset} seed {seed}: the {slot} run's M1 RMSE is {rmse}; the paired \
             comparison needs finite values"
        );
    }
    let mut all = Vec::new();
    let mut per_dataset = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let mut pairs = Vec::new();
        for a in runs
            .iter()
            .filter(|r| r.slot == "A" && &r.dataset == dataset)
        {
            let b = b_runs
                .remove(&(a.dataset.as_str(), a.seed))
                .with_context(|| {
                    format!(
                        "dataset {dataset} seed {}: an A run needs exactly one B run",
                        a.seed
                    )
                })?;
            pairs.push(Pair {
                dataset,
                seed: a.seed,
                a: &a.metrics.m1,
                b: &b.metrics.m1,
            });
        }
        per_dataset.push(DatasetPaired {
            dataset: dataset.clone(),
            counts: paired_counts(&pairs).with_context(|| format!("dataset {dataset}"))?,
        });
        all.extend(pairs);
    }
    if let Some((dataset, seed)) = b_runs.keys().next() {
        bail!("dataset {dataset} seed {seed}: a B run needs exactly one A run");
    }
    Ok(Paired {
        definitions: PAIRED_DEFINITIONS,
        rule: PAIRED_RULE,
        overall: paired_counts(&all)?,
        per_dataset,
    })
}

pub(crate) fn paired_counts(pairs: &[Pair<'_>]) -> Result<PairedCounts> {
    ensure!(!pairs.is_empty(), "no pair to compare");
    let (mut wins, mut ties, mut losses) = (0, 0, 0);
    for pair in pairs {
        let (a, b) = (pair.a.rmse, pair.b.rmse);
        if a.to_bits() == b.to_bits() {
            ties += 1;
        } else if b < a {
            wins += 1;
        } else if b > a {
            losses += 1;
        } else {
            bail!(
                "dataset {} seed {}: M1 RMSE {a} and {b} do not compare",
                pair.dataset,
                pair.seed
            );
        }
    }
    let a: Vec<f64> = pairs.iter().map(|p| p.a.rmse).collect();
    let b: Vec<f64> = pairs.iter().map(|p| p.b.rmse).collect();
    let differences: Vec<f64> = pairs.iter().map(|p| p.b.rmse - p.a.rmse).collect();
    // The first pair with the smallest difference.
    let mut largest = 0;
    for (index, difference) in differences.iter().enumerate() {
        if *difference < differences[largest] {
            largest = index;
        }
    }
    let others = differences
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != largest)
        .map(|(_, difference)| *difference);
    let counts = PairedCounts {
        pairs: pairs.len(),
        wins,
        ties,
        losses,
        median_a: median(&a).context("no pair")?,
        median_b: median(&b).context("no pair")?,
        median_difference: median(&differences).context("no pair")?,
        mean_difference: mean_of(differences.iter().copied()),
        largest_gain: PairDifference {
            dataset: pairs[largest].dataset.to_owned(),
            seed: pairs[largest].seed,
            difference: differences[largest],
        },
        mean_difference_without_largest_gain: (pairs.len() > 1).then(|| mean_of(others)),
        sign_test_p: sign_test(wins as u64, losses as u64),
        same_winner: pairs
            .iter()
            .filter(|p| p.a.program_sha256 == p.b.program_sha256)
            .count(),
    };
    // Finite inputs can still leave the range of an f64 (a difference, a sum or the mean of two
    // middle values beyond f64::MAX); every number the summary reports must be finite.
    for (what, value) in [
        ("median M1 RMSE of A", counts.median_a),
        ("median M1 RMSE of B", counts.median_b),
        ("median difference", counts.median_difference),
        ("mean difference", counts.mean_difference),
        ("largest gain", counts.largest_gain.difference),
        ("sign test p", counts.sign_test_p),
    ]
    .into_iter()
    .chain(
        counts
            .mean_difference_without_largest_gain
            .map(|value| ("mean difference without the largest gain", value)),
    ) {
        finite(&format!("paired {what}"), value)?;
    }
    Ok(counts)
}

/// The most untied pairs [`sign_test`] takes, `2^20`. A comparison has one pair per dataset and
/// seed, far fewer; the bound keeps the rounding's exponent arithmetic exact in i64.
pub const MAX_SIGN_TEST_PAIRS: u64 = 1 << 20;

/// The two-sided exact sign test of `wins` against `losses` (ties already left out): the
/// probability, for a fair coin over `n = wins + losses` tosses, of a split at least as uneven as
/// the one observed, capped at 1. It is 1 when there is no untied pair.
///
/// The probability is `S / 2^(n-1)` with `S` the sum of `C(n, k)` over `k` up to the smaller
/// count. `S` is computed exactly in integers, and the probability is rounded once to the nearest
/// f64, ties to even (`round_scaled`). So, for every split of at most [`MAX_SIGN_TEST_PAIRS`]
/// untied pairs, the result is the correctly rounded exact probability, and it replays bit for
/// bit on any host. A value under `2^-1022` is subnormal and carries fewer significant bits; the
/// result is 0 only where the exact value is at most half of the smallest positive f64,
/// `2^-1074` (a unanimous split of more than 1075 pairs, for example). The time grows with the
/// smaller count times `n`.
///
/// Outside that domain, more than [`MAX_SIGN_TEST_PAIRS`] pairs or a `wins + losses` that does not
/// fit in a u64, the result is NaN, which the paired summary refuses as not finite.
pub fn sign_test(wins: u64, losses: u64) -> f64 {
    let Some(n) = wins
        .checked_add(losses)
        .filter(|&n| n <= MAX_SIGN_TEST_PAIRS)
    else {
        return f64::NAN;
    };
    let tail = wins.min(losses);
    // An even split, or one short of it for an odd n, is at least half the mass on each side.
    if n == 0 || 2 * tail + 1 >= n {
        return 1.0;
    }
    // C(n, k) and the running sum S, as little-endian 64-bit limbs.
    let mut binomial = vec![1_u64];
    let mut sum = vec![1_u64];
    for k in 0..tail {
        // C(n, k + 1) = C(n, k) (n - k) / (k + 1), an exact division.
        let mut carry = 0_u128;
        for limb in &mut binomial {
            let value = u128::from(*limb) * u128::from(n - k) + carry;
            *limb = value as u64;
            carry = value >> 64;
        }
        if carry > 0 {
            binomial.push(carry as u64);
        }
        let mut rest = 0_u128;
        for limb in binomial.iter_mut().rev() {
            let value = (rest << 64) | u128::from(*limb);
            *limb = (value / u128::from(k + 1)) as u64;
            rest = value % u128::from(k + 1);
        }
        while binomial.len() > 1 && binomial.last() == Some(&0) {
            binomial.pop();
        }
        if sum.len() < binomial.len() {
            sum.resize(binomial.len(), 0);
        }
        let mut carry = 0_u128;
        for (index, limb) in sum.iter_mut().enumerate() {
            let value =
                u128::from(*limb) + u128::from(binomial.get(index).copied().unwrap_or(0)) + carry;
            *limb = value as u64;
            carry = value >> 64;
        }
        if carry > 0 {
            sum.push(carry as u64);
        }
    }
    round_scaled(&sum, n)
}

/// `s / 2^(n-1)`, for an integer `s >= 0` (little-endian 64-bit limbs) and `1 <= n <=`
/// [`MAX_SIGN_TEST_PAIRS`] (so `n` and the bit length of `s`, at most `n + 1`, fit in an i64),
/// rounded once to the nearest f64 with ties to even, and capped at 1. The rounding keeps 53 significant
/// bits in the normal range and whole multiples of `2^-1074` below it, so subnormal results are
/// rounded once too, never twice.
pub(crate) fn round_scaled(s: &[u64], n: u64) -> f64 {
    /// Bit `index` of `s`, which has `length` significant bits; 0 outside them.
    pub(crate) fn bit(s: &[u64], length: u64, index: i64) -> bool {
        index >= 0 && (index as u64) < length && (s[(index / 64) as usize] >> (index % 64)) & 1 == 1
    }
    let Some(top) = s.iter().rposition(|&limb| limb != 0) else {
        return 0.0;
    };
    let length = top as u64 * 64 + u64::from(64 - s[top].leading_zeros());
    // s / 2^(n-1) lies in [2^e, 2^(e+1)) with e = length - n.
    if length >= n {
        return 1.0;
    }
    let exponent = length as i64 - n as i64;
    // The weight of one unit in the last place, and m = s / 2^shift in those units.
    let mut unit = (exponent - 52).max(-1074);
    let shift = unit - 1 + n as i64;
    let mut m = if shift <= 0 {
        // s has at most 53 bits here, so it is exact.
        s[0] << -shift
    } else {
        let mut quotient = 0_u64;
        for index in (shift..length as i64).rev() {
            quotient = (quotient << 1) | u64::from(bit(s, length, index));
        }
        let half = bit(s, length, shift - 1);
        let below = (0..shift - 1).any(|index| bit(s, length, index));
        if half && (below || quotient & 1 == 1) {
            quotient += 1;
        }
        quotient
    };
    if m == 1 << 53 {
        m = 1 << 52;
        unit += 1;
    }
    if unit == -1074 {
        // Subnormal, or the smallest normal binade, where the bits are m itself.
        return f64::from_bits(m);
    }
    f64::from_bits((((unit + 1075) as u64) << 52) | (m - (1 << 52))).min(1.0)
}

/// Each slot's search seconds, for `timing.paired`.
pub(crate) fn paired_timing(runs: &[RunTiming]) -> PairedTiming {
    let seconds = |slot: &str| -> Vec<f64> {
        runs.iter()
            .filter(|t| t.slot == slot)
            .map(|t| t.search_seconds)
            .collect()
    };
    let (a, b) = (seconds("A"), seconds("B"));
    let (total_a, total_b) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    PairedTiming {
        total_seconds_a: total_a,
        total_seconds_b: total_b,
        median_run_seconds_a: median(&a).unwrap_or(0.0),
        median_run_seconds_b: median(&b).unwrap_or(0.0),
        ratio_b_to_a: (total_a > 0.0).then(|| total_b / total_a),
    }
}
