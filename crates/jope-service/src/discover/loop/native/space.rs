use super::super::admission::sample_index;
use super::*;

/// Native heuristic grammar. No learned checkpoint or bank is loaded by this
/// M1 adapter; learned proposer/ranker parity remains the RNK-10/M3 gate.
pub struct NativeSpace;
impl SearchSpace for NativeSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        let genome = feat_math::GenomeIr::parse(dsl)?;
        feat_math::transpile::supports(&genome)
            .map_err(|e| anyhow::anyhow!("unsupported Rust pipeline: {e:?}"))?;
        Ok(genome.canonical())
    }

    fn seeds(&self) -> Result<Vec<String>> {
        let seeds: Vec<_> = feat_math::evolve::STARTERS
            .iter()
            .copied()
            .chain(std::iter::once(feat_math::CHAMPION_GENOME))
            .filter_map(|s| self.canonical(s).ok())
            .collect();
        ensure!(!seeds.is_empty(), "no supported native starters");
        Ok(seeds)
    }

    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        self.propose_observed(parents, count, rng, |_, _| {})
    }
}

impl NativeSpace {
    /// Observe the actual generator arm without adding RNG draws or changing candidates.
    pub(crate) fn propose_observed(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
        mut observe: impl FnMut(&str, &'static str),
    ) -> Result<Vec<String>> {
        use feat_math::evolve::{self, CanonicalMutationArm::*};
        ensure!(!parents.is_empty(), "no parents");
        let arms = [
            ParameterJitter,
            Insert,
            Delete,
            AdjacentSwap,
            HeadReplacement,
            Simplify,
        ];
        let mut proposals = vec![];
        for _ in 0..count.saturating_mul(16) {
            let parent = &parents[sample_index(rng, parents.len())];
            let arm = sample_index(rng, 8);
            let seed = rng.next_u64();
            let proposed = if arm < 6 {
                evolve::mutate_canonical_pipeline_with_arm(parent, seed, 6, false, arms[arm])
            } else if arm == 6 {
                let other = &parents[sample_index(rng, parents.len())];
                evolve::crossover_canonical_pipelines(parent, other, seed, 6)
            } else {
                Ok(evolve::random_canonical_pipeline(seed, 6))
            };
            if let Ok(dsl) = proposed.and_then(|s| self.canonical(&s).map_err(|e| e.to_string())) {
                observe(
                    &dsl,
                    [
                        "native_parameter_jitter",
                        "native_insert",
                        "native_delete",
                        "native_adjacent_swap",
                        "native_head_replacement",
                        "native_simplify",
                        "native_crossover",
                        "native_random",
                    ][arm],
                );
                proposals.push(dsl);
                if proposals.len() == count {
                    break;
                }
            }
        }
        Ok(proposals)
    }
}
