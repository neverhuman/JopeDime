use jope_learning::{canon::canonical_sha256, support::*};
use std::collections::BTreeMap;

struct Issuer(bool);
impl AdmissionVerifier for Issuer {
    fn verify_train_evidence(&self, _: &TrainEvidence) -> anyhow::Result<bool> {
        Ok(self.0)
    }
}

fn vocab() -> Vocabulary {
    Vocabulary {
        sha256: fixed_vocabulary_sha256().unwrap(),
        max_seq: 640,
    }
}
fn context() -> String {
    "1".repeat(64)
}
fn candidate(dsl: &str, role_context: &str) -> Candidate {
    Candidate {
        context_id: role_context.into(),
        program: vocab().program(dsl).unwrap(),
        source_receipt_sha256: "2".repeat(64),
        parity_receipt_sha256: Some("3".repeat(64)),
        inner_loss: Some(0.2),
        incumbent_inner_loss: Some(0.3),
    }
}
fn parent() -> FrozenSupport {
    let catalog = Catalog {
        schema: "jope.flywheel-full-catalog-trie/v1".into(),
        vocabulary_sha256: vocab().sha256.clone(),
        max_seq: 640,
        programs: vec![
            vocab().program("(pipe X (ridge))").unwrap(),
            vocab().program("(pipe X (standardize) (ridge))").unwrap(),
        ],
    };
    let role_map = BTreeMap::from([(context(), Role::Train)]);
    FrozenSupport::freeze(FreezeRequest {
        expected_catalog_sha256: canonical_sha256(&catalog).unwrap(),
        catalog,
        require_v1_catalog: false,
        expected_role_map_sha256: canonical_sha256(&role_map).unwrap(),
        role_map,
        candidates: vec![],
    })
    .unwrap()
}
fn evidence(c: &Candidate) -> TrainEvidence {
    TrainEvidence {
        context_id: c.context_id.clone(),
        program_sha256: c.program.program_sha256.clone(),
        source_receipt_sha256: c.source_receipt_sha256.clone(),
        parity_receipt_sha256: c.parity_receipt_sha256.clone().unwrap(),
        execution_receipt_sha256: "4".repeat(64),
    }
}
fn invalid_program_candidate(dsl: &str) -> Candidate {
    Candidate {
        context_id: context(),
        program: Program {
            dsl: dsl.to_owned(),
            program_sha256: jope_learning::canon::sha256_hex(dsl.as_bytes()),
            tokens: vec![1, 13, 2],
        },
        source_receipt_sha256: "2".repeat(64),
        parity_receipt_sha256: Some("3".repeat(64)),
        inner_loss: Some(0.2),
        incumbent_inner_loss: Some(0.3),
    }
}

#[test]
fn train_verified_leaf_creates_immutable_successor_and_two_novel_actions() {
    let parent = parent();
    let first = candidate("(pipe X (robust_scale) (ridge))", &context());
    let (decision, child) =
        admit_novel_leaf(&parent, first.clone(), &evidence(&first), &Issuer(true)).unwrap();
    let child = child.unwrap();
    let AdmissionDecision::Admitted { transition } = decision else {
        panic!("TRAIN leaf should be admitted")
    };
    transition.verify().unwrap();
    assert_eq!(child.version, parent.version + 1);
    assert_ne!(child.support_sha256, parent.support_sha256);
    let second = candidate("(pipe X (sigmoid) (ridge))", &context());
    let (decision2, child2) =
        admit_novel_leaf(&child, second.clone(), &evidence(&second), &Issuer(true)).unwrap();
    let AdmissionDecision::Admitted { transition } = decision2 else {
        panic!("second TRAIN leaf should be admitted")
    };
    transition.verify().unwrap();
    let child2 = child2.unwrap();
    assert_eq!(child2.programs.len(), parent.programs.len() + 2);
    assert_eq!(parent.programs.len(), 2, "parent bytes remain immutable");

    let first_program = child2
        .programs
        .iter()
        .find(|program| program.dsl == "(pipe X (robust_scale) (ridge))")
        .unwrap();
    let second_program = child2
        .programs
        .iter()
        .find(|program| program.dsl == "(pipe X (sigmoid) (ridge))")
        .unwrap();
    let mut common = 0;
    while first_program.tokens.get(common) == second_program.tokens.get(common) {
        common += 1;
    }
    let trie = Trie::new(&child2).unwrap();
    let prefix = first_program.tokens[..common].to_vec();
    let choices = trie.children(&prefix).unwrap();
    let first_token = first_program.tokens[common];
    let second_token = second_program.tokens[common];
    let mut initial_logits = vec![0.0; 282];
    initial_logits[second_token as usize] = 4.0f64.ln();
    let initial = Distribution::from_logits(&trie, &prefix, &initial_logits, 1.0).unwrap();
    let p_first = initial.probabilities[initial.choices.binary_search(&first_token).unwrap()];
    let mut changed_logits = initial_logits.clone();
    changed_logits[first_token as usize] = 4.0f64.ln();
    changed_logits[second_token as usize] = 0.0;
    let changed = Distribution::from_logits(&trie, &prefix, &changed_logits, 1.0).unwrap();
    let p_first_changed =
        changed.probabilities[changed.choices.binary_search(&first_token).unwrap()];
    assert!(
        p_first_changed > p_first,
        "novel alternatives respond to logits"
    );
    let (loss, gradient) = (-p_first.ln(), p_first - 1.0);
    let (changed_loss, changed_gradient) = (-p_first_changed.ln(), p_first_changed - 1.0);
    assert!(
        loss > changed_loss && gradient != changed_gradient,
        "fixture loss/gradient respond to relative logits"
    );
    assert!(choices.contains(&first_token) && choices.contains(&second_token));
}

#[test]
fn missing_or_forged_train_proof_is_refused_without_support_mutation() {
    let parent = parent();
    let candidate = candidate("(pipe X (standardize) (ridge))", &context());
    let before = parent.support_sha256.clone();
    let (decision, child) = admit_novel_leaf(
        &parent,
        candidate.clone(),
        &evidence(&candidate),
        &Issuer(false),
    )
    .unwrap();
    assert!(matches!(decision, AdmissionDecision::Refused { .. }));
    assert!(child.is_none());
    assert_eq!(parent.support_sha256, before);
}

#[test]
fn non_train_and_overlength_leaves_are_retained_as_refused_observations() {
    let parent = parent();
    let protected = candidate("(pipe X (standardize) (ridge))", &"9".repeat(64));
    let (decision, child) = admit_novel_leaf(
        &parent,
        protected.clone(),
        &evidence(&protected),
        &Issuer(true),
    )
    .unwrap();
    assert!(matches!(decision, AdmissionDecision::Refused { .. }));
    assert!(child.is_none());

    let long = format!("(pipe X (ridge alpha={}))", "1".repeat(700));
    let overlength = invalid_program_candidate(&long);
    let (decision, child) = admit_novel_leaf(
        &parent,
        overlength.clone(),
        &evidence(&overlength),
        &Issuer(true),
    )
    .unwrap();
    assert!(matches!(decision, AdmissionDecision::Refused { .. }));
    assert!(child.is_none());

    let unexecutable = invalid_program_candidate("(pipe X (not_an_operator))");
    let (decision, child) = admit_novel_leaf(
        &parent,
        unexecutable.clone(),
        &evidence(&unexecutable),
        &Issuer(true),
    )
    .unwrap();
    assert!(matches!(decision, AdmissionDecision::Refused { .. }));
    assert!(child.is_none());
}

#[test]
fn exact_tiny_distribution_normalizes_and_fixed_seed_frequencies_conform() {
    let support = parent();
    let trie = Trie::new(&support).unwrap();
    let programs = &support.programs;
    let mut common = 0;
    while programs[0].tokens.get(common) == programs[1].tokens.get(common) {
        common += 1;
    }
    let prefix = programs[0].tokens[..common].to_vec();
    let choices = trie.children(&prefix).unwrap();
    assert_eq!(choices.len(), 2);
    let mut logits = vec![0.0; 282];
    logits[choices[0] as usize] = 0.0;
    logits[choices[1] as usize] = 0.6f64.ln();
    let law = CloverBehaviorLaw::new(&trie, &support.vocabulary.sha256, 1.0).unwrap();
    let d = law.distribution(&trie, &prefix, &logits).unwrap();
    assert!((d.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mut counts = vec![0usize; choices.len()];
    for i in 0..20_000 {
        let uniform = ((i * 7919 % 20_000) as f64 + 0.5) / 20_000.0;
        let (token, _) = d.select(uniform).unwrap();
        counts[d.choices.binary_search(&token).unwrap()] += 1;
    }
    for (count, expected) in counts.iter().zip(&d.probabilities) {
        assert!((*count as f64 / 20_000.0 - expected).abs() < 0.01);
    }
    let complete_mass: f64 = support
        .programs
        .iter()
        .map(|program| {
            let rows: Vec<_> = (1..program.tokens.len())
                .map(|position| LogitsRow {
                    prefix: program.tokens[..position].to_vec(),
                    logits: vec![0.0; 282],
                })
                .collect();
            law.teacher_forced(&trie, &program.tokens, &rows)
                .unwrap()
                .log_probability
                .exp()
        })
        .sum();
    assert!((complete_mass - 1.0).abs() < 1e-12);
}
