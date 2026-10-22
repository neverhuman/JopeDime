//! Island-model helpers: independent random streams over the same cuts, and the one pure rule
//! that moves champions — between islands every K generations, and between the rounds of
//! `jope pipeline evolve`.

use std::collections::BTreeMap;

/// A measured program: its canonical genome, its held-out RMSE and the score it is ranked by
/// (per cut inside a search, mean over the run's cuts between rounds). The score is the RMSE
/// itself unless the search ranks by a regression objective (`--fitness`), when it is that
/// objective's minimization loss.
#[derive(Clone, Debug, PartialEq)]
pub struct Champion {
    pub genome: String,
    pub rmse: f64,
    pub score: f64,
}

impl Champion {
    /// A champion ranked by its RMSE.
    pub fn by_rmse(genome: impl Into<String>, rmse: f64) -> Self {
        Champion {
            genome: genome.into(),
            rmse,
            score: rmse,
        }
    }
}

/// Up to `k` distinct champions, best first: lowest score, ties broken on the canonical genome.
/// A genome listed more than once keeps its best score (and that measurement's RMSE).
///
/// Pure and deterministic: the result depends on the multiset of champions only, never on the
/// order they are listed in, which is what lets migration and round-to-round seeding replay from
/// the seed alone whatever order threads finished in.
pub fn next_seeds(champions: &[Champion], k: usize) -> Vec<Champion> {
    let mut best: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for champion in champions {
        let key = (champion.score, champion.rmse);
        best.entry(champion.genome.as_str())
            .and_modify(|kept| {
                if key
                    .0
                    .total_cmp(&kept.0)
                    .then(key.1.total_cmp(&kept.1))
                    .is_lt()
                {
                    *kept = key;
                }
            })
            .or_insert(key);
    }
    let mut seeds: Vec<Champion> = best
        .into_iter()
        .map(|(genome, (score, rmse))| Champion {
            genome: genome.to_owned(),
            rmse,
            score,
        })
        .collect();
    seeds.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.genome.cmp(&b.genome)));
    seeds.truncate(k);
    seeds
}

/// Salt that separates island streams from every per-candidate stream.
const ISLAND_SALT: u64 = 0x1D1A_4D5E_ED00_0000;

/// The random-stream seed of island `index`. Island 0 is the run's own `--rng-seed`, so a
/// one-island search is exactly the search without islands.
pub fn island_seed(rng_seed: u64, index: usize) -> u64 {
    if index == 0 {
        return rng_seed;
    }
    let mut z = rng_seed
        .wrapping_add(ISLAND_SALT)
        .wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(genome: &str, rmse: f64) -> Champion {
        Champion::by_rmse(genome, rmse)
    }

    #[test]
    fn next_seeds_ranks_deduplicates_and_caps() {
        let champions = [
            c("b", 2.0),
            c("a", 3.0),
            c("c", 1.0),
            c("a", 1.5),
            c("d", 2.0),
        ];
        let seeds = next_seeds(&champions, 3);
        assert_eq!(seeds, vec![c("c", 1.0), c("a", 1.5), c("b", 2.0)]);
        // Equal scores break on the genome, so `b` precedes `d` whatever the input order.
        assert_eq!(next_seeds(&champions, 10).len(), 4);
        assert_eq!(next_seeds(&champions, 0), vec![]);
        assert_eq!(next_seeds(&[], 4), vec![]);
    }

    #[test]
    fn next_seeds_is_independent_of_input_order() {
        let champions = vec![
            c("(pipe X (ridge))", 4.0),
            c("(pipe X (rank) (ridge))", 2.5),
            c("(pipe X (standardize) (ridge))", 2.5),
            c("(pipe X (rank) (ridge))", 3.0),
            c("(pipe X (blom) (ridge))", 9.0),
        ];
        let forward = next_seeds(&champions, 3);
        let mut reversed = champions.clone();
        reversed.reverse();
        assert_eq!(forward, next_seeds(&reversed, 3));
        let mut rotated = champions.clone();
        rotated.rotate_left(2);
        assert_eq!(forward, next_seeds(&rotated, 3));
        assert_eq!(forward[0], c("(pipe X (rank) (ridge))", 2.5));
        assert_eq!(forward[1], c("(pipe X (standardize) (ridge))", 2.5));
    }

    #[test]
    fn next_seeds_rank_by_the_objective_score_and_keep_the_rmse() {
        let scored = |genome: &str, rmse: f64, score: f64| Champion {
            genome: genome.to_owned(),
            rmse,
            score,
        };
        // Under an objective, the lower-RMSE program can rank second.
        let seeds = next_seeds(&[scored("a", 1.0, 5.0), scored("b", 2.0, 3.0)], 2);
        assert_eq!(seeds, vec![scored("b", 2.0, 3.0), scored("a", 1.0, 5.0)]);
    }

    #[test]
    fn island_zero_is_the_run_stream_and_others_differ() {
        assert_eq!(island_seed(7, 0), 7);
        let seeds: std::collections::BTreeSet<u64> = (0..16).map(|i| island_seed(7, i)).collect();
        assert_eq!(seeds.len(), 16);
        assert_eq!(island_seed(7, 3), island_seed(7, 3));
        assert_ne!(island_seed(7, 3), island_seed(8, 3));
    }
}
