use super::support::*;
use super::*;

#[test]
fn repeated_head_draw_keeps_a_novel_head_slot_when_the_pool_has_an_alternative() {
    // Original TRAIN request130: round775/context2. This is a row-free
    // generator regression, not a model/fitness or release qualification.
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let parents = [Program::from_dsl("(pipe X (poly_interact append=true) (cheb degree=4) (winsor_clip append=true q=0.15759416738534485) (sigmoid append=true) (ridge))")];
    let lm = [
        -7.822644233703613,
        -3.335017681121826,
        2.0916075706481934,
        0.17733953893184662,
        -0.23421242833137512,
        -1.8542579412460327,
        -0.39526352286338806,
        1.5499933958053589,
        0.3646467924118042,
        -0.23557567596435547,
        2.554443597793579,
        -0.7511533498764038,
        1.6888071298599243,
        0.965538501739502,
        -0.9864354729652405,
        -1.153594970703125,
        -1.8711278438568115,
        -0.2296914905309677,
        1.2335679531097412,
        -0.15032795071601868,
        1.465488314628601,
        -1.6561437845230103,
        -0.8125762939453125,
        0.8135456442832947,
        -0.35853514075279236,
        -0.9137781858444214,
        -0.4460272192955017,
        0.6699191927909851,
        0.8401845097541809,
        0.07990206778049469,
        -0.5336035490036011,
        0.4456247091293335,
        0.6210421323776245,
        -0.6575910449028015,
        -0.8605937361717224,
        0.04108719900250435,
        -0.318228542804718,
        -1.1542876958847046,
        0.020905081182718277,
        -0.4025436341762543,
        1.037981390953064,
        0.8037648797035217,
        0.1720881313085556,
        -0.14072567224502563,
        -0.21567422151565552,
        0.2446334958076477,
        -0.9094048738479614,
        0.4294348657131195,
    ];
    let bank = Bank {
            role: BankRole::Proposer,
            records: vec![
                    Retrieved {
                        program: Program::from_dsl("(pipe X (cheb append=true degree=4) (winsor_clip q=0.12691538581586245) (histgbm depth=6 lr=0.14350417339409188 rounds=210))"),
                        family_id: "other-0".into(),
                        dl_train: true,
                        distance_squared: 1.0,
                    },
                    Retrieved {
                        program: Program::from_dsl("(pipe X (blom append=true) (sigmoid append=true) (pair_prod topk=13) (ridge_gcv))"),
                        family_id: "other-1".into(),
                        dl_train: true,
                        distance_squared: 2.0,
                    },
            ],
            called: RefCell::new(vec![]),
        };
    let req = Round {
        seed: 20261031,
        round: 775,
        exploring: false,
        target_family: "original-target",
        raw_lm48: &lm,
        parents: &parents,
    };
    let batch = proposer.propose(&req, &bank, &BTreeSet::new()).unwrap();
    let heads: Vec<_> = batch
        .proposals
        .iter()
        .filter(|p| p.family == ProposalFamily::ReplaceHead)
        .collect();
    assert_eq!(batch.proposals.len(), 8);
    assert_eq!(
        heads.len(),
        2,
        "a repeated head draw consumed a scheduled head slot"
    );
    assert_ne!(heads[0].memo_key, heads[1].memo_key);
    assert_eq!(
        heads[0].program.program_sha256,
        "f5add21fc43bf261f379d1cda8de3500816025f6225458166871b03005baabdc"
    );
    assert!(batch.duplicates_or_memo > 0);
    assert_eq!(
        batch.attempts.values().sum::<usize>(),
        batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo
    );
    assert_eq!(
        batch,
        proposer.propose(&req, &bank, &BTreeSet::new()).unwrap()
    );

    // The first preferred head is already measured, while other heads are
    // novel. Preserve the scheduled head arms without admitting that key.
    let measured = BTreeSet::from([heads[0].memo_key.clone()]);
    let memo_collision = proposer.propose(&req, &bank, &measured).unwrap();
    assert_eq!(memo_collision.proposals.len(), 8);
    assert_eq!(
        memo_collision
            .proposals
            .iter()
            .filter(|p| p.family == ProposalFamily::ReplaceHead)
            .count(),
        2,
        "a memoized first draw consumed a scheduled head slot"
    );
    assert!(memo_collision
        .proposals
        .iter()
        .all(|p| !measured.contains(&p.memo_key)));
    assert_eq!(
        memo_collision.attempts.values().sum::<usize>(),
        memo_collision.proposals.len()
            + memo_collision.invalid_or_filtered
            + memo_collision.duplicates_or_memo
    );

    // A collision does not authorize an existing memo entry or an unbounded
    // retry when every other preferred head has already been measured.
    let mut transforms = proposer.pipeline(&parents[0]).unwrap().stages;
    transforms.pop();
    let (_, templates) = select_landmark_preferred_pools(&lm);
    let mut memo = BTreeSet::new();
    for template in templates {
        let mut parsed = parse_canonical_pipeline(&format!("(pipe X {template})")).unwrap();
        let dsl = assemble(transforms.clone(), parsed.stages.pop().unwrap());
        let key = memo_key(&dsl).unwrap();
        if key != heads[0].memo_key {
            memo.insert(key);
        }
    }
    let exhausted = proposer.propose(&req, &bank, &memo).unwrap();
    assert_eq!(exhausted.proposals.len(), 8);
    assert_eq!(
        exhausted
            .proposals
            .iter()
            .filter(|p| p.family == ProposalFamily::ReplaceHead)
            .count(),
        1
    );
    assert!(exhausted
        .proposals
        .iter()
        .all(|p| !memo.contains(&p.memo_key)));
    assert_eq!(
        exhausted.attempts.values().sum::<usize>(),
        exhausted.proposals.len() + exhausted.invalid_or_filtered + exhausted.duplicates_or_memo
    );
}

#[test]
fn unchanged_head_draw_is_rejected_before_the_next_edit_family() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let lm = [0.0; 48];
    let (_, templates) = select_landmark_preferred_pools(&lm);
    let parents = [Program::from_dsl(format!("(pipe X {})", templates[0]))];
    let parent_key = memo_key(&parents[0].dsl).unwrap();
    let mut controls = 0;
    for seed in 0..128 {
        let mut req = request(&parents, &lm, false);
        req.seed = seed;
        let mut rng = round_rng(req.seed, req.round);
        let mut parameter = proposer.pipeline(&parents[0]).unwrap();
        for stage in &mut parameter.stages {
            mutate_parameters(stage, &mut rng);
        }
        let first = format!("(pipe X {})", templates[index(&mut rng, templates.len())]);
        if memo_key(&first).unwrap() != parent_key {
            continue;
        }
        let batch = proposer.propose(&req, &NoBank, &BTreeSet::new()).unwrap();
        let insert = batch
            .proposals
            .iter()
            .position(|p| p.family == ProposalFamily::Insert)
            .expect("the next applicable edit is insertion");
        assert!(
            batch.proposals[..insert]
                .iter()
                .all(|p| p.family != ProposalFamily::ReplaceHead),
            "an unchanged parent was promoted into a novel head edit for seed {seed}"
        );
        assert!(batch.proposals.iter().all(|p| p.memo_key != parent_key));
        assert_eq!(batch.proposals.len(), 8);
        assert_eq!(
            batch.attempts.values().sum::<usize>(),
            batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo
        );
        assert_eq!(
            batch,
            proposer.propose(&req, &NoBank, &BTreeSet::new()).unwrap()
        );
        controls += 1;
    }
    assert!(controls > 0, "the no-op control must actually execute");
}

#[test]
fn a_changed_head_matching_another_parent_still_retries() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let lm = [0.0; 48];
    let (_, templates) = select_landmark_preferred_pools(&lm);
    let first = Program::from_dsl(format!("(pipe X {})", templates[0]));
    let first_key = memo_key(&first.dsl).unwrap();
    let other = templates
        .iter()
        .map(|t| Program::from_dsl(format!("(pipe X {t})")))
        .find(|p| memo_key(&p.dsl).unwrap() != first_key)
        .unwrap();
    let parents = [first, other];
    let known: BTreeSet<_> = parents.iter().map(|p| memo_key(&p.dsl).unwrap()).collect();
    let mut controls = 0;
    for seed in 0..128 {
        let mut req = request(&parents, &lm, false);
        req.seed = seed;
        let mut rng = round_rng(req.seed, req.round);
        let mut parameter = proposer.pipeline(&parents[0]).unwrap();
        for stage in &mut parameter.stages {
            mutate_parameters(stage, &mut rng);
        }
        let draw = format!("(pipe X {})", templates[index(&mut rng, templates.len())]);
        if memo_key(&draw).unwrap() != first_key {
            continue;
        }
        // Step 1 edits parents[1], so this first choice changes its head
        // but collides with parents[0]. It is a real edit, not a no-op.
        let batch = proposer.propose(&req, &NoBank, &BTreeSet::new()).unwrap();
        let head = batch
            .proposals
            .iter()
            .position(|p| p.family == ProposalFamily::ReplaceHead)
            .unwrap();
        let insert = batch
            .proposals
            .iter()
            .position(|p| p.family == ProposalFamily::Insert)
            .unwrap();
        assert!(head < insert, "a genuine parent collision lost its retry");
        assert!(batch.proposals.iter().all(|p| !known.contains(&p.memo_key)));
        assert_eq!(batch.proposals.len(), 8);
        controls += 1;
    }
    assert!(
        controls > 0,
        "the changed-head control must actually execute"
    );
}

#[test]
fn oversized_head_alternative_is_filtered_without_aborting_the_round() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let mut dsl = String::from("(pipe X ");
    while dsl.len() + "(sigmoid) ".len() + "(ridge))".len() <= MAX_DSL_BYTES {
        dsl.push_str("(sigmoid) ");
    }
    dsl.push_str("(ridge))");
    let parents = [Program::from_dsl(dsl)];
    proposer.pipeline(&parents[0]).unwrap();
    let bank = Bank {
        role: BankRole::Proposer,
        records: vec![],
        called: RefCell::new(vec![]),
    };
    for seed in 0..4 {
        let round = Round {
            seed,
            round: 0,
            exploring: false,
            target_family: "byte-limit-control",
            raw_lm48: &[0.0; 48],
            parents: &parents,
        };
        let batch = proposer.propose(&round, &bank, &BTreeSet::new()).unwrap();
        assert_eq!(batch.proposals.len(), 8);
        assert!(batch.invalid_or_filtered > 0);
        assert!(batch
            .proposals
            .iter()
            .all(|p| p.program.dsl.len() <= MAX_DSL_BYTES));
    }
}
