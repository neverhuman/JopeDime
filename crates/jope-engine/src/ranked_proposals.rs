//! Isolated, manifest-bound seed experiment. No champion store or GP outcomes
//! are inputs to pure predictions; all six controls are frozen together.
mod prelude {
    pub(super) use crate::{
        corpus::{ContextRow, Index},
        grammar::{State, Vocab},
        manifest::{file_hash, hash, read_hashed, read_json, write_json, SEED},
        neural::{Bf16, Config, Policy},
        proposal_ranking::{score_pool, Bindings, Options, Ranking},
        training::mix,
    };
    pub(super) use anyhow::{ensure, Context, Result};
    pub(super) use serde::{Deserialize, Serialize};
    pub(super) use serde_json::{json, Value};
    pub(super) use std::{
        collections::{BTreeMap, BTreeSet, VecDeque},
        fs,
        io::{BufRead, BufReader},
        path::{Path, PathBuf},
        time::Instant,
    };
    pub(super) use tch::{nn, Device, Kind, Tensor};
}

mod assemble;
mod catalog;
mod sampling;
mod worker;

pub use assemble::assemble;
pub use catalog::{Artifact, BankEntry, Candidate, Manifest, Origin, View, PROTOCOL};
pub use sampling::grammar;
pub use worker::{run, worker_device};

#[cfg(test)]
use catalog::insert;
#[cfg(test)]
use prelude::*;
#[cfg(test)]
use worker::run_admitted;

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(family: &str, cut: usize, program: &str) -> BankEntry {
        BankEntry {
            family: family.into(),
            dataset: format!("dataset_{family}"),
            cut,
            record: cut,
            program: program.into(),
            program_sha256: hash(program.as_bytes()),
            token_count: 4,
        }
    }
    #[test]
    fn bank_round_robin_is_deterministic_balanced_and_exact() {
        let entries = vec![
            entry("b", 2, "b2"),
            entry("a", 9, "shared"),
            entry("a", 1, "a1"),
            entry("b", 1, "shared"),
            entry("a", 3, "a3"),
        ];
        let bank = round_robin_bank(entries.clone(), 4);
        assert_eq!(
            bank.iter().map(|e| e.program.as_str()).collect::<Vec<_>>(),
            vec!["a1", "shared", "a3", "b2"]
        );
        assert_eq!(
            serde_json::to_vec(&bank).unwrap(),
            serde_json::to_vec(&round_robin_bank(entries.into_iter().rev().collect(), 4)).unwrap()
        );
        assert!(bank
            .iter()
            .filter(|e| e.family != "a")
            .all(|e| e.family == "b"));
    }
    #[test]
    fn pool_keeps_all128_and_accumulates_origins_before_ranking() {
        let mut pool = Vec::new();
        let mut lookup = BTreeMap::new();
        for draw in 0..128 {
            insert(
                &mut pool,
                &mut lookup,
                format!("program{draw}"),
                Origin {
                    kind: "neural_policy".into(),
                    draw: Some(draw),
                    bank_record: None,
                    source_family: None,
                },
            );
        }
        insert(
            &mut pool,
            &mut lookup,
            "program100".into(),
            Origin {
                kind: "meta_training_champion".into(),
                draw: None,
                bank_record: Some(7),
                source_family: Some("train".into()),
            },
        );
        assert_eq!(pool.len(), 128);
        assert_eq!(pool[100].origins.len(), 2);
    }
    #[test]
    fn frozen_vocabulary_accepts_gcv_without_snapping() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        assert!(vocab.encode("(pipe X (ridge_gcv))").is_ok());
        assert!(vocab
            .encode("(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))")
            .is_err());
        let generic = grammar(&vocab, SEED, true).unwrap();
        assert_eq!(generic.len(), 64);
        assert_eq!(generic[0], "(pipe X (ridge))");
        assert_eq!(generic.iter().collect::<BTreeSet<_>>().len(), 64);
    }
}

#[cfg(test)]
mod manifest_run_tests;
