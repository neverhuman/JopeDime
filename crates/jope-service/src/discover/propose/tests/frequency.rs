use super::support::*;
use super::*;

#[test]
fn ten_thousand_proposals_have_real_filter_provenance_and_bounded_source_schedule() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let parents = parents();
    let lm = [0.0; 48];
    let bank = Bank::new();
    let mut counts = BTreeMap::<ProposalFamily, usize>::new();
    for round in 0..1250 {
        let mut req = request(&parents, &lm, false);
        req.round = round;
        let batch = proposer.propose(&req, &bank, &BTreeSet::new()).unwrap();
        assert_eq!(batch.proposals.len(), 8);
        assert!(!batch.synthesis_exhausted);
        assert_eq!(
            batch.attempts.values().sum::<usize>(),
            batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo,
        );
        assert!(
            batch
                .attempts
                .iter()
                .filter(
                    |(f, _)| **f != ProposalFamily::Retrieval && **f != ProposalFamily::Synthesis
                )
                .map(|(_, n)| n)
                .sum::<usize>()
                <= 24
        );
        for proposal in batch.proposals {
            *counts.entry(proposal.family).or_default() += 1;
            assert_eq!(
                proposal.program.program_sha256,
                digest(proposal.program.dsl.as_bytes())
            );
            assert!(proposer.pipeline(&proposal.program).is_ok());
        }
    }
    assert_eq!(counts.values().sum::<usize>(), 10_000);
    assert_eq!(counts[&ProposalFamily::Retrieval], 2500);
    println!(
        "generated-only CPU fixture histogram (not matched evaluated-population M3): {counts:?}"
    );
}

#[test]
fn ten_thousand_train_parent_outputs_report_m3_frequency_comparison() {
    const N: usize = 10_000;
    const MAX_ROUNDS: usize = 4_000;

    let source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/flywheel_selfplay/custody/proposer_frequencies.json"
    ))
    .unwrap();
    assert_eq!(source["schema"], "jope.proposer-frequencies/v1");
    assert_eq!(
        source["population"]["kind"],
        "evaluated GP proposal attempts"
    );
    assert_eq!(source["population"]["records"], 1024);
    assert_eq!(source["population"]["generated_in_round_receipts"], 3712);
    assert_eq!(
        source["population"]["not_individually_operator_labelled"],
        2688
    );
    let target_sha256 = digest(
        include_str!("../../../../research/flywheel_selfplay/custody/proposer_frequencies.json")
            .as_bytes(),
    );
    let exploration_pattern: Vec<bool> = source["round_journals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|round| round["exploration"].as_bool().unwrap())
        .collect();
    assert_eq!(exploration_pattern.len(), 16);

    // This is the committed row-free TRAIN parent index already used by the
    // real-parent determinism controls. No evaluation rows or labels are read.
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
    let parents: Vec<Program> = rows
        .iter()
        .map(|row| {
            assert_eq!(row["dl_split"], "DL-Train");
            assert!(hash_valid(row["context_sha256"].as_str().unwrap()));
            let parent = Program::from_dsl(row["dsl"].as_str().unwrap());
            assert_eq!(
                parent.program_sha256,
                row["program_sha256"].as_str().unwrap()
            );
            parent
        })
        .collect();

    // No raw proposer snapshot/evolving historical bank is supplied, so use
    // NoBank and report this as a generated-only diagnostic, never M3 parity.
    let proposer =
        Proposer::new(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
    let eligible_parent_indices: Vec<usize> = parents
        .iter()
        .enumerate()
        .filter_map(|(index, parent)| proposer.pipeline(parent).ok().map(|_| index))
        .collect();
    assert!(!eligible_parent_indices.is_empty());
    let lm = [0.0; 48];
    let mut counts = BTreeMap::<ProposalFamily, usize>::new();
    let mut generated = 0usize;
    let mut round_index = 0usize;
    while generated < N && round_index < MAX_ROUNDS {
        let parent_index = eligible_parent_indices[round_index % eligible_parent_indices.len()];
        let exploring = exploration_pattern[round_index % exploration_pattern.len()];
        let round = Round {
            seed: 0xd5c06 ^ round_index as u64,
            round: 767 + round_index as u64,
            exploring,
            target_family: rows[parent_index]["family_id"].as_str().unwrap(),
            raw_lm48: &lm,
            parents: std::slice::from_ref(&parents[parent_index]),
        };
        let batch = proposer.propose(&round, &NoBank, &BTreeSet::new()).unwrap();
        for proposal in batch.proposals {
            if generated == N {
                break;
            }
            *counts.entry(proposal.family).or_default() += 1;
            generated += 1;
        }
        round_index += 1;
    }
    assert_eq!(
        generated, N,
        "bounded schedule did not emit 10,000 proposals"
    );

    let comparisons: Vec<serde_json::Value> = source["operators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|operator| {
            let name = operator["operator"].as_str().unwrap();
            let family = [
                ProposalFamily::Retrieval,
                ProposalFamily::Parameter,
                ProposalFamily::ReplaceHead,
                ProposalFamily::Insert,
                ProposalFamily::Delete,
                ProposalFamily::Crossover,
                ProposalFamily::Synthesis,
            ]
            .into_iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("unknown reference family {name}"));
            let count = counts.get(&family).copied().unwrap_or_default();
            let observed = count as f64 / N as f64;
            let reference = operator["fraction"].as_f64().unwrap();
            serde_json::json!({
                "operator": name,
                "observed_count": count,
                "observed_fraction": observed,
                "reference_count": operator["count"],
                "reference_fraction": reference,
                "absolute_delta_percentage_points": (observed - reference).abs() * 100.0,
                "within_two_percentage_points": (observed - reference).abs() <= 0.02,
            })
        })
        .collect();
    let within_tolerance = comparisons
        .iter()
        .all(|row| row["within_two_percentage_points"] == true);
    println!(
        "DSC06_M3_FREQUENCY_REPORT={}",
        serde_json::json!({
            "schema": "jopedime.dsc06-generated-frequency-diagnostic/v1",
            "reference_sha256": target_sha256,
            "reference_population": source["population"],
            "proposal_count": generated,
            "parent_source_sha256": digest(&bytes),
            "parent_count": parents.len(),
            "eligible_parent_count": eligible_parent_indices.len(),
            "parent_split": "DL-Train",
            "parent_rotation": "round_index modulo eligible committed TRAIN parent order",
            "bank": "NoBank; no exact evolving raw proposer snapshot supplied",
            "exploration_pattern_rounds": exploration_pattern.len(),
            "exploration_rounds_in_pattern": exploration_pattern.iter().filter(|&&x| x).count(),
            "generated_distribution_within_two_pp": within_tolerance,
            "population_comparable_to_reference": false,
            "m3_qualified": false,
            "comparisons": comparisons,
        })
    );
}
