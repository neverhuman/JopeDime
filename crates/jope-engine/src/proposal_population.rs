//! Keep pure model candidates independent of adaptive GP population contents.
use crate::{gp::SeedCandidate, grammar::Vocab, manifest::hash};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PURE_PROTOCOL: &str = "training_context_neural_pool_v1";
pub const ASSEMBLY_PROTOCOL: &str = "preserve_shared_tail_dedup_v1";

/// No incumbent, target score, or population state enters pure deduplication.
pub fn push_pure(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    genome: String,
) -> bool {
    if !seen.insert(genome.clone()) {
        return false;
    }
    if pool.len() < 64 {
        pool.push(SeedCandidate {
            genome,
            origin: "neural_policy".into(),
                    policy_logprob: None,
                });
    }
    true
}

pub fn fill_pure(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    vocab: &Vocab,
    seed: u64,
) -> Result<()> {
    for offset in 0..10000u64 {
        if pool.len() == 64 {
            return Ok(());
        }
        let raw = feat_math::evolve::random_canonical_pipeline(seed.wrapping_add(offset), 6);
        let Ok(genome) = vocab.project_proposal(&raw) else {
            continue;
        };
        if seen.insert(genome.clone()) {
            pool.push(SeedCandidate {
                genome,
                origin: "neural_rejection_fallback".into(),
                    policy_logprob: None,
                });
        }
    }
    anyhow::bail!("unable to complete independent pure neural pool")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Overlap {
    pub pure_rank: usize,
    pub program_sha256: String,
    pub retained_slot: usize,
    pub retained_origin: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Refill {
    pub program_sha256: String,
    pub generator_seed: u64,
    pub origin: String,
}

#[derive(Serialize, Deserialize)]
pub struct Assembly {
    pub schema: String,
    pub protocol: String,
    pub pure_programs_sha256: String,
    pub retained_tail_sha256: String,
    pub overlaps: Vec<Overlap>,
    pub refills: Vec<Refill>,
    #[serde(skip)]
    pub neural_slots: Vec<SeedCandidate>,
}

/// The pure rank order and shared retained16 are immutable inputs. Any overlap
/// is handled only here, with explicit generic replacement provenance.
pub fn assemble(
    pure: &[SeedCandidate],
    retained: &[SeedCandidate],
    vocab: &Vocab,
    seed: u64,
) -> Result<Assembly> {
    ensure!(
        pure.len() == 64 && retained.len() == 16,
        "expected pure64 and retained16"
    );
    let tail = retained
        .iter()
        .enumerate()
        .map(|(i, c)| (c.genome.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    ensure!(tail.len() == 16, "shared tail contains duplicate programs");
    ensure!(
        retained[..10].iter().all(|c| matches!(
            c.origin.as_str(),
            "incumbent_cut_teacher" | "retained_cut_teacher" | "retained_slot_fallback"
        )) && retained[10..]
            .iter()
            .all(|c| c.origin == "random_exploration"),
        "shared64/10/6 origin layout changed"
    );
    ensure!(
        pure.iter()
            .map(|c| &c.genome)
            .collect::<BTreeSet<_>>()
            .len()
            == 64,
        "pure pool is not unique"
    );
    ensure!(
        pure.iter().all(|c| matches!(
            c.origin.as_str(),
            "neural_policy" | "neural_rejection_fallback"
        )),
        "pure pool contains teacher-dependent candidates"
    );
    for candidate in retained {
        vocab.encode(&candidate.genome)?;
    }
    let mut seen = retained
        .iter()
        .map(|c| c.genome.clone())
        .collect::<BTreeSet<_>>();
    let mut neural_slots = Vec::new();
    let mut overlaps = Vec::new();
    for (pure_rank, candidate) in pure.iter().enumerate() {
        vocab.encode(&candidate.genome)?;
        if let Some(&slot) = tail.get(candidate.genome.as_str()) {
            overlaps.push(Overlap {
                pure_rank,
                program_sha256: hash(candidate.genome.as_bytes()),
                retained_slot: 64 + slot,
                retained_origin: retained[slot].origin.clone(),
            });
        } else {
            seen.insert(candidate.genome.clone());
            neural_slots.push(candidate.clone());
        }
    }
    let mut refills = Vec::new();
    for offset in 0..10000u64 {
        if neural_slots.len() == 64 {
            break;
        }
        let generator_seed = seed.wrapping_add(offset);
        let raw = feat_math::evolve::random_canonical_pipeline(generator_seed, 6);
        let Ok(genome) = vocab.project_proposal(&raw) else {
            continue;
        };
        if seen.insert(genome.clone()) {
            let origin = "neural_overlap_fallback".to_string();
            refills.push(Refill {
                program_sha256: hash(genome.as_bytes()),
                generator_seed,
                origin: origin.clone(),
            });
            neural_slots.push(SeedCandidate { genome, origin,
                    policy_logprob: None,
                });
        }
    }
    ensure!(
        neural_slots.len() == 64,
        "unable to fill diverse GP overlap replacements"
    );
    Ok(Assembly {
        schema: "parallel.gp_population_assembly.v1".into(),
        protocol: ASSEMBLY_PROTOCOL.into(),
        pure_programs_sha256: hash(&serde_json::to_vec(
            &pure.iter().map(|c| &c.genome).collect::<Vec<_>>(),
        )?),
        retained_tail_sha256: hash(&serde_json::to_vec(retained)?),
        overlaps,
        refills,
        neural_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pure_fallback_rng_is_independent_of_population_replacements() {
        let vocab: Vocab = crate::manifest::read_json(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"),
        )
        .unwrap();
        let produce = || {
            let mut pool = Vec::new();
            let mut seen = BTreeSet::new();
            fill_pure(&mut pool, &mut seen, &vocab, 909).unwrap();
            pool
        };
        let first = produce();
        let tail = first[..16]
            .iter()
            .enumerate()
            .map(|(i, c)| SeedCandidate {
                genome: c.genome.clone(),
                origin: if i < 10 {
                    "retained_cut_teacher"
                } else {
                    "random_exploration"
                }
                .into(),
                policy_logprob: None,
            })
            .collect::<Vec<_>>();
        let assembled = assemble(&first, &tail, &vocab, 888).unwrap();
        assert_eq!(assembled.refills.len(), 16);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&produce()).unwrap()
        );
        assert!(first
            .iter()
            .all(|c| c.origin == "neural_rejection_fallback"));
    }

    #[cfg(feature = "neural")]
    #[test]
    fn actual_frozen_policy_pure_best_is_identical_after_incumbent_change() {
        use crate::{
            neural::{Config, Policy},
            proposal_ranking::{score_pool, Bindings, Options},
        };
        use tch::{nn, Device};
        tch::set_num_threads(1);
        tch::manual_seed(20260907);
        let vocab: Vocab = crate::manifest::read_json(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"),
        )
        .unwrap();
        let mut config = Config::regression(vocab.tokens.len() as i64);
        config.width = 16;
        config.layers = 1;
        config.heads = 4;
        config.ffn = 32;
        config.dropout = 0.0;
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = Policy::new(vs.root(), config).unwrap();
        let context = (0..2048)
            .map(|i| (i % 13) as f32 / 13.0)
            .collect::<Vec<_>>();
        let bindings = Bindings {
            checkpoint_sha256: "a".repeat(64),
            vocabulary_sha256: "b".repeat(64),
            conditioning_context_sha256: hash(
                &context
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        };
        let mut pure = Vec::new();
        let mut seen = BTreeSet::new();
        fill_pure(&mut pure, &mut seen, &vocab, 101).unwrap();
        let score = || {
            score_pool(
                &policy,
                &vocab,
                &context,
                &pure.iter().map(|c| c.genome.clone()).collect::<Vec<_>>(),
                bindings.clone(),
                Options {
                    device: Device::Cpu,
                    batch_size: 64,
                    keep: 64,
                },
            )
            .unwrap()
        };
        let before = score();
        let pure_ranked = before
            .selected()
            .iter()
            .map(|p| pure.iter().find(|c| c.genome == p.program).unwrap().clone())
            .collect::<Vec<_>>();
        let mut alternate = Vec::new();
        let mut alternate_seen = BTreeSet::new();
        fill_pure(&mut alternate, &mut alternate_seen, &vocab, 202).unwrap();
        let tail = alternate
            .iter()
            .filter(|c| !seen.contains(&c.genome))
            .take(16)
            .enumerate()
            .map(|(i, c)| SeedCandidate {
                genome: c.genome.clone(),
                origin: if i < 10 {
                    "retained_cut_teacher"
                } else {
                    "random_exploration"
                }
                .into(),
                policy_logprob: None,
            })
            .collect::<Vec<_>>();
        let original = assemble(&pure_ranked, &tail, &vocab, 303).unwrap();
        let mut changed = tail.clone();
        changed[0].genome = before.selected()[0].program.clone();
        let revised = assemble(&pure_ranked, &changed, &vocab, 303).unwrap();
        let after = score();
        assert_eq!(
            serde_json::to_vec(&before).unwrap(),
            serde_json::to_vec(&after).unwrap()
        );
        assert_eq!(original.overlaps.len(), 0);
        assert_eq!(revised.overlaps.len(), 1);
        assert_eq!(revised.overlaps[0].pure_rank, 0);
        assert_ne!(
            serde_json::to_vec(&original.neural_slots).unwrap(),
            serde_json::to_vec(&revised.neural_slots).unwrap()
        );
    }

    #[test]
    fn changed_incumbents_change_only_gp_assembly_never_pure_prediction() {
        let vocab: Vocab = crate::manifest::read_json(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"),
        )
        .unwrap();
        let mut stream = Vec::new();
        let mut seen = BTreeSet::new();
        for seed in 1..10000 {
            if let Ok(g) =
                vocab.project_proposal(&feat_math::evolve::random_canonical_pipeline(seed, 6))
            {
                if seen.insert(g.clone()) {
                    stream.push(g);
                }
            }
            if stream.len() == 80 {
                break;
            }
        }
        let mut pure = Vec::new();
        let mut pure_seen = BTreeSet::new();
        // Fixed raw draws include duplicates. Their pure winner/order is frozen
        // before either hypothetical heldout-selected incumbent list exists.
        for genome in stream[..64].iter().chain(stream[..4].iter()) {
            push_pure(&mut pure, &mut pure_seen, genome.clone());
        }
        let before = serde_json::to_vec(&pure).unwrap();
        let winner = pure[0].genome.clone();
        let tail = stream[64..]
            .iter()
            .enumerate()
            .map(|(i, g)| SeedCandidate {
                genome: g.clone(),
                origin: if i < 10 {
                    "retained_cut_teacher"
                } else {
                    "random_exploration"
                }
                .into(),
                policy_logprob: None,
            })
            .collect::<Vec<_>>();
        let a = assemble(&pure, &tail, &vocab, 555).unwrap();
        assert!(a.overlaps.is_empty());
        let mut changed = tail.clone();
        changed[0] = SeedCandidate {
            genome: winner.clone(),
            origin: "incumbent_cut_teacher".into(),
                    policy_logprob: None,
                };
        let b = assemble(&pure, &changed, &vocab, 555).unwrap();
        assert_eq!(before, serde_json::to_vec(&pure).unwrap());
        assert_eq!(pure[0].genome, winner);
        assert_eq!(a.pure_programs_sha256, b.pure_programs_sha256);
        assert_ne!(a.retained_tail_sha256, b.retained_tail_sha256);
        assert_eq!(b.overlaps.len(), 1);
        assert_eq!(b.overlaps[0].pure_rank, 0);
        assert_eq!(b.overlaps[0].retained_slot, 64);
        assert_eq!(b.refills.len(), 1);
        assert_eq!(b.refills[0].origin, "neural_overlap_fallback");
        let population = b
            .neural_slots
            .iter()
            .chain(changed.iter())
            .collect::<Vec<_>>();
        assert_eq!(population.len(), 80);
        assert_eq!(
            population
                .iter()
                .map(|c| &c.genome)
                .collect::<BTreeSet<_>>()
                .len(),
            80
        );
        assert_eq!(population[64].genome, winner);
    }
}
