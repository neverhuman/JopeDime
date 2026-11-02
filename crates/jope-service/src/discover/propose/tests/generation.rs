use super::support::*;
use super::*;

#[test]
fn deterministic_rounds_are_independent_and_exploration_only_changes_width() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let parents = parents();
    let lm = [0.0; 48];
    let a = proposer
        .propose(&request(&parents, &lm, false), &NoBank, &BTreeSet::new())
        .unwrap();
    let b = proposer
        .propose(&request(&parents, &lm, false), &NoBank, &BTreeSet::new())
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.proposals.len(), 8);
    let wide = proposer
        .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
        .unwrap();
    assert_eq!(wide.proposals.len(), 16);
    let mut other = request(&parents, &lm, false);
    other.round += 1;
    assert_ne!(
        a.proposals,
        proposer
            .propose(&other, &NoBank, &BTreeSet::new())
            .unwrap()
            .proposals
    );
    assert!(!a.raw_bank_available);
}

#[test]
fn raw_bank_contract_family_order_and_source_identity_are_enforced() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let lm = std::array::from_fn(|i| i as f64 / 4.0);
    let bank = Bank::new();
    let result = proposer
        .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
        .unwrap();
    assert_eq!(result.proposals.len(), 8);
    assert_eq!(result.proposals[0].program, bank.records[0].program);
    assert_eq!(
        bank.called.borrow()[0],
        (
            <[f64; 12]>::try_from(&lm[..12]).unwrap(),
            "target".to_string(),
            2
        )
    );
    let mut bad = Bank::new();
    bad.role = BankRole::ModelPrior;
    assert!(proposer
        .propose(&request(&[], &lm, false), &bad, &BTreeSet::new())
        .is_err());
    assert!(bad.called.borrow().is_empty());
    for mutation in 0..4 {
        let mut bad = Bank::new();
        match mutation {
            0 => bad.records[0].family_id = "target".into(),
            1 => bad.records[0].dl_train = false,
            2 => bad.records[0].distance_squared = f64::NAN,
            _ => bad.records.reverse(),
        }
        assert!(proposer
            .propose(&request(&[], &lm, false), &bad, &BTreeSet::new())
            .is_err());
    }
}

#[test]
fn memo_dedup_retains_parents_and_never_returns_known_identity() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let parents = parents();
    let before = parents.clone();
    let lm = [0.0; 48];
    let first = proposer
        .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
        .unwrap();
    let memo = first.proposals.iter().map(|p| p.memo_key.clone()).collect();
    let second = proposer
        .propose(&request(&parents, &lm, true), &NoBank, &memo)
        .unwrap();
    assert_eq!(parents, before);
    assert!(second.proposals.iter().all(|p| !memo.contains(&p.memo_key)));
    assert_eq!(
        second
            .proposals
            .iter()
            .map(|p| &p.memo_key)
            .collect::<BTreeSet<_>>()
            .len(),
        second.proposals.len()
    );
    assert!(second.duplicates_or_memo > 0);
}

#[test]
fn actual_language_intersection_filters_every_generated_operation() {
    let languages = [Language::Rust, Language::Python, Language::Javascript];
    let proposer = Proposer::new(&languages).unwrap();
    let lm = [0.0; 48];
    let batch = proposer
        .propose(&request(&[], &lm, true), &NoBank, &BTreeSet::new())
        .unwrap();
    assert_eq!(batch.proposals.len(), 16);
    let filter = capability::allowed_ops(&languages).unwrap();
    for proposal in batch.proposals {
        let genome = GenomeIr::parse(&proposal.program.dsl).unwrap();
        assert!(filter.accepts_seed(&genome));
        let GenomeIr::Pipeline(pipeline) = genome else {
            panic!("pipeline required")
        };
        assert!(pipeline
            .stages
            .iter()
            .all(|s| !filter.excluded_ops().contains(&s.op.as_str())));
    }
    assert!(Proposer::new(&[Language::Typescript]).is_err());
}

#[test]
fn an_unexportable_retrieval_is_filtered_without_aborting_the_round() {
    let proposer =
        Proposer::new(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
    let mut bank = Bank::new();
    bank.records[0].program = Program::from_dsl("(pipe X (gpr))");
    let lm = [0.0; 48];
    let batch = proposer
        .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
        .unwrap();
    assert_eq!(batch.proposals.len(), 8);
    assert!(batch.invalid_or_filtered > 0);
    assert!(batch
        .proposals
        .iter()
        .all(|p| proposer.pipeline(&p.program).is_ok()));
    assert!(batch
        .proposals
        .iter()
        .all(|p| p.program != bank.records[0].program));
}

#[test]
fn round_key_matches_an_independent_little_endian_sha256_oracle() {
    // Python hashlib over struct.pack("<QQ", 123, 7), retained in the source receipt.
    let mut expected = ChaCha12Rng::from_seed([
        4, 152, 66, 165, 136, 157, 146, 169, 112, 244, 157, 201, 69, 32, 60, 213, 154, 218, 19,
        128, 81, 80, 181, 162, 170, 106, 147, 165, 229, 143, 224, 7,
    ]);
    let mut actual = round_rng(123, 7);
    for _ in 0..8 {
        assert_eq!(actual.next_u64(), expected.next_u64());
    }
}

#[test]
fn typed_cycle_is_not_a_random_twenty_percent_sampler() {
    let counts = (0..24).fold(BTreeMap::<ProposalFamily, usize>::new(), |mut m, step| {
        *m.entry(ProposalFamily::edit(step)).or_default() += 1;
        m
    });
    assert_eq!(
        [
            counts[&ProposalFamily::Parameter],
            counts[&ProposalFamily::ReplaceHead],
            counts[&ProposalFamily::Insert],
            counts[&ProposalFamily::Delete],
            counts[&ProposalFamily::Crossover]
        ],
        [5, 5, 5, 5, 4]
    );
    let source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/flywheel_selfplay/custody/proposer_frequencies.json"
    ))
    .unwrap();
    assert_eq!(
        source["population"]["kind"],
        "evaluated GP proposal attempts"
    );
    assert_eq!(source["population"]["records"], 1024);
    assert_eq!(
        source["population"]["not_individually_operator_labelled"],
        2688
    );
}
