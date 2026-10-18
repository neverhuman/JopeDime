use jope_learning::{
    canon::{canonical_sha256, sha256_hex},
    support::*,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Golden {
    request: FreezeRequest,
    logits: Vec<LogitsRow>,
    scores: Vec<GoldenScore>,
    trace_tokens_sha256: String,
    seeded_traces: usize,
    added_traces: usize,
}
#[derive(Deserialize)]
struct GoldenScore {
    tokens: Vec<u16>,
    per_token: Vec<f64>,
    log_probability: f64,
}

#[test]
fn actual_v1_policy_golden_matches_sampling_and_eos_likelihood() {
    let bytes = include_bytes!("fixtures/support-policy-v1.json");
    assert_eq!(
        sha256_hex(bytes),
        "f1fe0756517983376af62ec21c0b3422fa1f1abdfcfebdd1dde1d0212e725b6c"
    );
    let g: Golden = serde_json::from_slice(bytes).unwrap();
    let added = g.request.candidates[0].program.tokens.clone();
    let support = FrozenSupport::freeze(g.request).unwrap();
    let trie = Trie::new(&support).unwrap();
    let table: BTreeMap<_, _> = g
        .logits
        .into_iter()
        .map(|r| (r.prefix.clone(), r))
        .collect();
    let mut mass = 0.;
    let mut max_delta: f64 = 0.;
    for score in &g.scores {
        let rows: Vec<_> = (1..score.tokens.len())
            .map(|pos| table[&score.tokens[..pos]].clone())
            .collect();
        let actual = Likelihood::teacher_forced(&trie, &score.tokens, &rows, 1.).unwrap();
        max_delta = max_delta.max((actual.log_probability - score.log_probability).abs());
        assert_eq!(actual.per_token.len(), score.per_token.len());
        for (a, b) in actual.per_token.iter().zip(&score.per_token) {
            max_delta = max_delta.max((a - b).abs());
        }
        mass += actual.log_probability.exp();
    }
    assert!(max_delta <= 1e-12, "golden max delta {max_delta}");
    assert!((mass - 1.).abs() <= 1e-12, "support mass {mass}");
    let mut sequences = Vec::new();
    let mut added_draws = 0;
    for seed in 0..g.seeded_traces {
        let mut prefix = vec![1];
        let mut trace = 0.;
        let mut rows = Vec::new();
        while prefix.last() != Some(&2) {
            let row = table[&prefix].clone();
            let d = Distribution::from_logits(&trie, &prefix, &row.logits, 1.).unwrap();
            let random = sha256_hex(format!("row:{seed}:{}", prefix.len()).as_bytes());
            let bits = u64::from_str_radix(&random[..13], 16).unwrap();
            let (token, lp) = d.select(bits as f64 / (1u64 << 52) as f64).unwrap();
            trace += lp;
            rows.push(row);
            prefix.push(token);
        }
        assert_eq!(
            Likelihood::teacher_forced(&trie, &prefix, &rows, 1.)
                .unwrap()
                .log_probability,
            trace
        );
        let expected = g.scores.iter().find(|s| s.tokens == prefix).unwrap();
        assert!((trace - expected.log_probability).abs() <= 1e-12);
        added_draws += usize::from(prefix == added);
        sequences.push(prefix);
    }
    assert_eq!(g.seeded_traces, 1000);
    assert_eq!(added_draws, g.added_traces);
    assert!(added_draws > 0);
    assert_eq!(canonical_sha256(&sequences).unwrap(), g.trace_tokens_sha256);
    println!(
        "1000 exact token traces, {added_draws} added draws, max delta={max_delta}, mass={mass}"
    );
}

#[test]
fn cli_freezes_verifies_and_refuses_overwrite_bad_pin_and_torn_artifact() {
    use std::{fs, process::Command};
    let target =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/support-cli-fixtures");
    fs::create_dir_all(&target).unwrap();
    let directory = tempfile::tempdir_in(&target).unwrap();
    let input = directory.path().join("request.json");
    let output = directory.path().join("support.json");
    fs::write(&input, serde_json::to_vec(&request()).unwrap()).unwrap();
    let bin = env!("CARGO_BIN_EXE_jope_support");
    let freeze = || {
        Command::new(bin)
            .args(["freeze", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    let first = freeze();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let original = fs::read(&output).unwrap();
    let frozen: FrozenSupport = serde_json::from_slice(&original).unwrap();
    assert!(!freeze().status.success());
    assert_eq!(fs::read(&output).unwrap(), original);
    let verify = |sha: &str| {
        Command::new(bin)
            .args(["verify", "--support"])
            .arg(&output)
            .args(["--sha", sha])
            .output()
            .unwrap()
    };
    assert!(verify(&frozen.support_sha256).status.success());
    assert!(!verify(&"0".repeat(64)).status.success());
    fs::write(&output, &original[..original.len() / 2]).unwrap();
    assert!(!verify(&frozen.support_sha256).status.success());
}

fn vocabulary() -> Vocabulary {
    Vocabulary {
        sha256: fixed_vocabulary_sha256().unwrap(),
        max_seq: 640,
    }
}
fn context() -> String {
    "1".repeat(64)
}
fn candidate(dsl: &str) -> Candidate {
    Candidate {
        context_id: context(),
        program: vocabulary().program(dsl).unwrap(),
        source_receipt_sha256: "2".repeat(64),
        parity_receipt_sha256: Some("3".repeat(64)),
        inner_loss: Some(0.2),
        incumbent_inner_loss: Some(0.3),
    }
}
fn request() -> FreezeRequest {
    let catalog = Catalog {
        schema: "jope.flywheel-full-catalog-trie/v1".into(),
        vocabulary_sha256: vocabulary().sha256,
        max_seq: 640,
        programs: vec![
            vocabulary().program("(pipe X (ridge))").unwrap(),
            vocabulary()
                .program("(pipe X (standardize) (ridge lambda=0.1))")
                .unwrap(),
        ],
    };
    let role_map = BTreeMap::from([(context(), Role::Train)]);
    FreezeRequest {
        expected_catalog_sha256: canonical_sha256(&catalog).unwrap(),
        catalog,
        require_v1_catalog: false,
        expected_role_map_sha256: canonical_sha256(&role_map).unwrap(),
        role_map,
        candidates: vec![],
    }
}

#[test]
fn transition_law_adapter_preserves_v1_leaf_bytes_and_g12_refusal() {
    let support = FrozenSupport::freeze(request()).unwrap();
    let original_bytes = serde_json::to_vec(&support).unwrap();
    let original_programs = support.programs.clone();
    let trie = Trie::new(&support).unwrap();
    let law = CloverBehaviorLaw::new(&trie, &support.vocabulary.sha256, 1.0).unwrap();

    assert_eq!(trie.leaf_count(), original_programs.len());
    assert!(!support.qualified);
    assert_eq!(support.qualified_use, "BLOCKED_JEPSON_G12");
    assert!(!support.availability_grants_context_outcome_eligibility);
    let mut total_mass = 0.0;
    for program in &original_programs {
        trie.validate_leaf(&program.tokens).unwrap();
        let rows: Vec<_> = (1..program.tokens.len())
            .map(|position| LogitsRow {
                prefix: program.tokens[..position].to_vec(),
                logits: vec![0.0; 282],
            })
            .collect();
        total_mass += law
            .teacher_forced(&trie, &program.tokens, &rows)
            .unwrap()
            .log_probability
            .exp();
    }
    assert!((total_mass - 1.0).abs() < 1e-12);
    assert_eq!(support.programs, original_programs);
    assert_eq!(serde_json::to_vec(&support).unwrap(), original_bytes);
}

fn logits(trie: &Trie<'_>, tokens: &[u16]) -> Vec<LogitsRow> {
    (1..tokens.len())
        .map(|pos| {
            let prefix = tokens[..pos].to_vec();
            let mut l = vec![f64::NAN; 282];
            for child in trie.children(&prefix).unwrap() {
                l[child as usize] = f64::from(child) / 17.;
            }
            LogitsRow { prefix, logits: l }
        })
        .collect()
}

#[test]
fn external_winner_freezes_version_with_finite_added_leaf() {
    let parent = FrozenSupport::freeze(request()).unwrap();
    let old = canonical_sha256(&parent).unwrap();
    let added = candidate("(pipe X (minmax) (ridge lambda=0.123456789))");
    let next = parent.extend(vec![added.clone()]).unwrap();
    assert_eq!(next.version, 2);
    assert_eq!(next.parent_support_sha256, parent.support_sha256);
    assert_eq!(canonical_sha256(&parent).unwrap(), old);
    assert!(!next.qualified);
    assert!(!next.availability_grants_context_outcome_eligibility);
    let trie = Trie::new(&next).unwrap();
    let score = Likelihood::teacher_forced(
        &trie,
        &added.program.tokens,
        &logits(&trie, &added.program.tokens),
        1.,
    )
    .unwrap();
    assert!(score.log_probability.is_finite());
    assert_eq!(score.per_token.len(), added.program.tokens.len() - 1);
}

#[test]
fn heldout_validation_sealed_unknown_and_missing_roles_refused() {
    for role in [Role::Validation, Role::Heldout, Role::Sealed, Role::Unknown] {
        let mut r = request();
        r.role_map.insert(context(), role);
        r.expected_role_map_sha256 = canonical_sha256(&r.role_map).unwrap();
        r.candidates.push(candidate("(pipe X (minmax) (ridge))"));
        assert!(FrozenSupport::freeze(r)
            .unwrap_err()
            .to_string()
            .contains("TRAIN"));
    }
    let mut r = request();
    r.candidates.push(candidate("(pipe X (minmax) (ridge))"));
    r.role_map.clear();
    r.expected_role_map_sha256 = canonical_sha256(&r.role_map).unwrap();
    assert!(FrozenSupport::freeze(r).is_err());
}

#[test]
fn consuming_path_checks_rehashed_role_poison_and_qualification() {
    let mut r = request();
    r.candidates.push(candidate("(pipe X (minmax) (ridge))"));
    let frozen = FrozenSupport::freeze(r).unwrap();
    let mut poison = frozen.clone();
    poison.role_map.insert(context(), Role::Sealed);
    poison.role_map_sha256 = canonical_sha256(&poison.role_map).unwrap();
    rehash(&mut poison);
    assert!(poison.verify().is_err());
    let mut poison = frozen.clone();
    poison.qualified = true;
    rehash(&mut poison);
    assert!(poison.verify().is_err());
}
fn rehash(s: &mut FrozenSupport) {
    let mut value = serde_json::to_value(&*s).unwrap();
    value.as_object_mut().unwrap().remove("support_sha256");
    s.support_sha256 = canonical_sha256(&value).unwrap();
}

#[test]
fn canonical_hash_tokens_vocabulary_and_length_are_bound() {
    for kind in 0..4 {
        let mut r = request();
        let mut row = candidate("(pipe X (minmax) (ridge))");
        match kind {
            0 => row.program.program_sha256 = "0".repeat(64),
            1 => row.program.tokens.insert(2, 0),
            2 => row.program.tokens.pop().map(|_| ()).unwrap(),
            _ => row.source_receipt_sha256 = "not-a-pin".into(),
        };
        r.candidates.push(row);
        assert!(FrozenSupport::freeze(r).is_err());
    }
    assert!(vocabulary().program(" (pipe X (ridge))").is_err());
    let mut v = vocabulary();
    v.max_seq = 3;
    assert!(v.program("(pipe X (ridge))").is_err());
    v = vocabulary();
    v.sha256 = "0".repeat(64);
    assert!(v.validate().is_err());
}

#[test]
fn exact_deployed_catalog_flag_rejects_fixture_parent() {
    let mut r = request();
    r.require_v1_catalog = true;
    assert!(FrozenSupport::freeze(r).is_err());
}

#[test]
fn duplicate_inputs_are_idempotent_and_reference_provenance_is_retained() {
    let a = candidate("(pipe X (minmax) (ridge))");
    let mut r = request();
    r.candidates = vec![a.clone()];
    let one = FrozenSupport::freeze(r.clone()).unwrap();
    r.candidates.push(a);
    let two = FrozenSupport::freeze(r).unwrap();
    assert_eq!(one.support_sha256, two.support_sha256);
    assert_eq!(one.provenance.len(), 1);
}

#[test]
fn changed_role_map_or_receipt_and_missing_provenance_refused() {
    let mut r = request();
    r.expected_role_map_sha256 = "0".repeat(64);
    assert!(FrozenSupport::freeze(r).is_err());
    let mut r = request();
    r.candidates.push(candidate("(pipe X (minmax) (ridge))"));
    let mut s = FrozenSupport::freeze(r).unwrap();
    s.provenance.clear();
    rehash(&mut s);
    assert!(s.verify().is_err());
}

#[test]
fn nonfinite_evidence_cannot_become_availability_or_zero_reward() {
    let mut r = request();
    let mut row = candidate("(pipe X (minmax) (ridge))");
    row.inner_loss = Some(f64::NAN);
    r.candidates.push(row);
    assert!(FrozenSupport::freeze(r).is_err());
}

#[test]
fn all_complete_leaves_normalize_and_trace_equals_eos_teacher_forcing() {
    let mut r = request();
    r.candidates.push(candidate("(pipe X (minmax) (ridge))"));
    let support = FrozenSupport::freeze(r).unwrap();
    let trie = Trie::new(&support).unwrap();
    let mut mass = 0.;
    for p in &support.programs {
        let rows = logits(&trie, &p.tokens);
        let ll = Likelihood::teacher_forced(&trie, &p.tokens, &rows, 0.7).unwrap();
        mass += ll.log_probability.exp();
        let trace: f64 = rows
            .iter()
            .enumerate()
            .map(|(pos, r)| {
                let d = Distribution::from_logits(&trie, &r.prefix, &r.logits, 0.7).unwrap();
                d.log_probabilities[d.choices.binary_search(&p.tokens[pos + 1]).unwrap()]
            })
            .sum();
        assert_eq!(trace, ll.log_probability);
        assert_eq!(ll.aggregation, "sum_including_eos");
    }
    assert!((mass - 1.).abs() < 1e-12);
}

#[test]
fn categorical_sampler_is_supported_and_added_action_sampleable() {
    let mut r = request();
    let added = candidate("(pipe X (minmax) (ridge))");
    r.candidates.push(added.clone());
    let s = FrozenSupport::freeze(r).unwrap();
    let t = Trie::new(&s).unwrap();
    let added_logits = logits(&t, &added.program.tokens);
    let mut reached = false;
    for seed in 0..1000u64 {
        let mut prefix = vec![1];
        let mut trace = 0.;
        let mut table = Vec::new();
        while prefix.last() != Some(&2) {
            let l = added_logits
                .iter()
                .find(|r| r.prefix == prefix)
                .map(|r| r.logits.clone())
                .unwrap_or(vec![0.; 282]);
            let d = Distribution::from_logits(&t, &prefix, &l, 1.).unwrap();
            let random = sha256_hex(format!("row:{seed}:{}", prefix.len()).as_bytes());
            let bits = u64::from_str_radix(&random[..13], 16).unwrap();
            let u = bits as f64 / (1u64 << 52) as f64;
            let (chosen, ll) = d.select(u).unwrap();
            table.push(LogitsRow {
                prefix: prefix.clone(),
                logits: l,
            });
            trace += ll;
            prefix.push(chosen);
        }
        assert!(t.contains(&prefix));
        assert_eq!(
            Likelihood::teacher_forced(&t, &prefix, &table, 1.)
                .unwrap()
                .log_probability,
            trace
        );
        reached |= prefix == added.program.tokens;
    }
    assert!(reached);
}

#[test]
fn unknown_or_partial_actions_and_nonfinite_supported_logits_refused() {
    let s = FrozenSupport::freeze(request()).unwrap();
    let t = Trie::new(&s).unwrap();
    assert!(t.children(&[]).is_err());
    assert!(t.children(&[1, 281]).is_err());
    let p = &s.programs[0];
    assert!(Likelihood::teacher_forced(&t, &p.tokens[..p.tokens.len() - 1], &[], 1.).is_err());
    let mut rows = logits(&t, &p.tokens);
    rows[0].logits[t.children(&[1]).unwrap()[0] as usize] = f64::NAN;
    assert!(Likelihood::teacher_forced(&t, &p.tokens, &rows, 1.).is_err());
}

#[test]
fn weighted_ce_is_sequence_sum_not_length_mean_and_bad_targets_refused() {
    assert_eq!(
        weighted_archive_ce(&[-2., -10.], &[0.25, 0.75]).unwrap(),
        8.
    );
    assert!(weighted_archive_ce(&[-2., -10.], &[0., 0.]).is_err());
    assert!(weighted_archive_ce(&[-2.], &[-1.]).is_err());
    assert!(weighted_archive_ce(&[f64::NAN], &[1.]).is_err());
}

#[test]
fn kl_requires_same_support_and_frozen_parameter_identity() {
    let h = "a".repeat(64);
    let p = "b".repeat(64);
    assert!(require_same_support_reference(&h, &h, &p, &p).is_ok());
    assert!(require_same_support_reference(&h, &"c".repeat(64), &p, &p).is_err());
    assert!(require_same_support_reference(&h, &h, &p, &h).is_err());
}

#[test]
fn immutable_read_requires_caller_pin_and_detects_semantic_and_byte_changes() {
    let s = FrozenSupport::freeze(request()).unwrap();
    let raw = serde_json::to_vec(&s).unwrap();
    assert!(FrozenSupport::from_bytes(&raw, &s.support_sha256).is_ok());
    assert!(FrozenSupport::from_bytes(&raw, &"0".repeat(64)).is_err());
    assert!(FrozenSupport::from_bytes(&raw[..raw.len() - 1], &s.support_sha256).is_err());
}

#[test]
fn mismatch_keeps_sign_sequence_units_and_missing_outcomes_unknown() {
    let rows: Vec<_> = (0..3)
        .map(|i| MismatchRow {
            context_id: sha256_hex(format!("ctx{i}").as_bytes()),
            parent: "a".repeat(64),
            program_sha256: "b".repeat(64),
            predicted_tokens_including_eos: 4,
            trie_sequence_logprob_float64: -2.,
            grammar_sequence_logprob_float64: -3. - i as f64,
            sampler_recompute_abs_error: 1e-14,
        })
        .collect();
    let r = MismatchReport::from_rows(&rows).unwrap();
    assert_eq!(r.mean_grammar_extra_nll, 2.);
    assert_eq!(r.median_grammar_extra_nll, 2.);
    assert!(!r.qualified);
    assert!(!r.training_objective_changed);
    assert!(MismatchReport::from_rows(&[]).is_err());
}
