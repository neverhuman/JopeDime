use super::support::*;
use super::*;

#[test]
fn every_proposer_attempt_has_an_outcome() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let lm = [0.0; 48];
    let cases = [
            "(pipe X (ridge lambda_frac=0.1))",
            "(pipe X (standardize) (ridge lambda_frac=0.1))",
            "(pipe X (standardize) (standardize) (standardize) (standardize) (standardize) (standardize) (ridge lambda_frac=0.1))",
        ];
    let mut unaccounted = vec![];
    let mut trace = Vec::new();
    for (case, dsl) in cases.into_iter().enumerate() {
        let parents = [Program::from_dsl(dsl)];
        for exploring in [false, true] {
            for round in 0..8 {
                let mut req = request(&parents, &lm, exploring);
                req.round = round;
                let batch = proposer.propose(&req, &NoBank, &BTreeSet::new()).unwrap();
                assert_eq!(batch.proposals.len(), if exploring { 16 } else { 8 });
                assert!(batch.attempts[&ProposalFamily::Crossover] > 0);
                let attempted: usize = batch.attempts.values().sum();
                let accounted =
                    batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo;
                if attempted != accounted {
                    unaccounted.push((case, exploring, round, attempted, accounted));
                }
                // Deliberately exclude the repaired outcome counter so the
                // before/after trace binds generation, order and provenance.
                trace.push(serde_json::json!({
                    "case":case,"exploring":exploring,"round":round,
                    "attempts":batch.attempts.iter().map(|(f,n)| (f.name(),n)).collect::<Vec<_>>(),
                    "proposals":batch.proposals.iter().map(|p| serde_json::json!({
                        "dsl":p.program.dsl,"program_sha256":p.program.program_sha256,
                        "memo_key":p.memo_key,"family":p.family.name(),"parents":p.parents,
                    })).collect::<Vec<_>>(),
                }));
            }
        }
    }
    println!(
        "DSC06_GENERATION_TRACE_SHA256={}",
        digest(&serde_json::to_vec(&trace).unwrap())
    );
    assert!(
        unaccounted.is_empty(),
        "missing attempt outcomes in {} rounds: {:?}",
        unaccounted.len(),
        &unaccounted[..unaccounted.len().min(3)]
    );
}

#[test]
fn real_train_parents_preserve_round_determinism_and_language_exclusions() {
    // Real row-free parents with controlled NoBank/zero landmarks; this is
    // software acceptance, not matched evaluated-population M3 evidence.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/flywheel_selfplay/custody/files")
        .join("home")
        .join("ubuntu")
        .join("jope-data/res-01/input/champions.json");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        digest(&bytes),
        "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
    );
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(rows.len(), 1219);
    let parents: Vec<_> = rows
        .iter()
        .map(|row| {
            assert_eq!(row["dl_split"], "DL-Train");
            assert!(hash_valid(row["context_sha256"].as_str().unwrap()));
            assert!(!row["family_id"].as_str().unwrap().is_empty());
            let parent = Program::from_dsl(row["dsl"].as_str().unwrap());
            assert_eq!(
                parent.program_sha256,
                row["program_sha256"].as_str().unwrap()
            );
            parent
        })
        .collect();
    let lm = [0.0; 48];
    for languages in [
        &[Language::Rust][..],
        &[Language::Rust, Language::Python, Language::Javascript][..],
    ] {
        let original = Proposer::new(languages).unwrap();
        let replay = Proposer::new(languages).unwrap();
        let filter = capability::allowed_ops(languages).unwrap();
        let run = |proposer: &Proposer, index: usize, exploring| {
            proposer
                .propose(
                    &Round {
                        seed: 0xd5c06 ^ index as u64,
                        round: index as u64 + 7,
                        exploring,
                        target_family: rows[index]["family_id"].as_str().unwrap(),
                        raw_lm48: &lm,
                        parents: std::slice::from_ref(&parents[index]),
                    },
                    &NoBank,
                    &BTreeSet::new(),
                )
                .map_err(|error| format!("{error:#}"))
        };
        let expected: Vec<_> = (0..rows.len())
            .flat_map(|index| [run(&original, index, false), run(&original, index, true)])
            .collect();
        let mut accepted = 0;
        let mut refused = 0;
        let mut generated = 0;
        for index in (0..rows.len()).rev() {
            for (mode, exploring) in [false, true].into_iter().enumerate() {
                let actual = run(&replay, index, exploring);
                assert_eq!(actual, expected[2 * index + mode], "record {index}");
                match actual {
                    Ok(batch) => {
                        accepted += 1;
                        let width = if exploring { 16 } else { 8 };
                        assert_eq!(batch.requested, width);
                        assert!(batch.proposals.len() <= width);
                        assert_eq!(batch.synthesis_exhausted, batch.proposals.len() < width);
                        assert_eq!(
                            batch.attempts.values().sum::<usize>(),
                            batch.proposals.len()
                                + batch.invalid_or_filtered
                                + batch.duplicates_or_memo
                        );
                        let mut unique = BTreeSet::new();
                        for proposal in batch.proposals {
                            assert_eq!(
                                proposal.program.program_sha256,
                                digest(proposal.program.dsl.as_bytes())
                            );
                            assert!(unique.insert(proposal.memo_key.clone()));
                            let GenomeIr::Pipeline(parsed) =
                                GenomeIr::parse(&proposal.program.dsl).unwrap()
                            else {
                                panic!("proposer must return a pipeline")
                            };
                            for stage in parsed.stages {
                                assert!(filter.allowed_ops().contains(&stage.op.as_str()));
                                assert!(!filter.excluded_ops().contains(&stage.op.as_str()));
                            }
                            generated += 1;
                        }
                    }
                    Err(reason) => {
                        assert!(reason.starts_with("invalid parent:"), "{reason}");
                        refused += 1;
                    }
                }
            }
        }
        assert_eq!(accepted + refused, 2 * rows.len());
        assert!(
            accepted > 0 && generated > 0,
            "no eligible real parents exercised"
        );
        println!("DSC06 real TRAIN parent software controls: languages={languages:?}, records={}, replayed_requests={}, accepted={accepted}, refused={refused}, generated={generated}; reverse-order fresh-instance equality; M3_qualified=false", rows.len(), accepted + refused);
    }
}
