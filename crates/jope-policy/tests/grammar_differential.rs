//! W4-T04: the sampling grammar's masks against the production typed parser.
//!
//! Frozen inputs: the committed step-60000 vocabulary `featmath.json` and its documented product
//! restriction `featmath-jain-v1.json` (both bound by sha256 through `bind_vocabulary`), and the
//! automaton in `jope_policy::grammar` (sampling mode, as the decoder uses it). The typed parser is
//! the production admission path: `Vocab::decode` (non-sampling replay, then DSL rendering),
//! `feat_math::GenomeIr::parse`, and the exact `Vocab::encode` round trip that ranking and seed
//! validation require.
//!
//! State domain, per vocabulary: every state reachable within [`DEPTH`] tokens after the
//! regression prefix (exhaustive), and every state along [`WALKS`] seeded grammar-uniform walks.
//! At each state the legal set is nonempty (no dead-end legal prefix), sorted, in range, free of
//! reserved and prefix tokens, and holds EOS exactly when the state is accepting; `advance`
//! accepts exactly the legal set (checked against every id at depth <= 2, a fixed sample of
//! illegal ids elsewhere, plus out-of-range ids) and a refused token leaves the state unchanged;
//! appending EOS decodes exactly when the state is accepting; and a deterministic legal
//! completion of the prefix parses and round-trips. Accepted sequences are also enumerated
//! exhaustively for every head and transform with every slot value and append flag, and for
//! every expression of at most three nodes.
//!
//! This is grammar/parser fixture proof. It says nothing about which programs a trained policy
//! prefers or whether they fit well.
use jope_policy::decode;
use jope_policy::grammar::{bind_vocabulary, State, Vocab, BOS, EOS, MAX_PROGRAM_TOKENS, PAD};
use jope_policy::CHECKPOINT_VOCABULARY_SHA256;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

/// Exhaustive depth after the proposal prefix.
const DEPTH: usize = 3;
/// Seeded grammar-uniform walks per vocabulary.
const WALKS: u64 = 300;
/// The regression proposal prefix `[BOS, featmath, regression]`.
const PREFIX: [usize; 3] = [BOS, 3, 397];
/// Slot-free regression head used to finish partial programs.
const RIDGE: usize = 370;

fn vocabularies() -> Vec<(&'static str, Vocab)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar");
    ["featmath.json", "featmath-jain-v1.json"]
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let bound = bind_vocabulary(&bytes, CHECKPOINT_VOCABULARY_SHA256).unwrap();
            (name, bound.vocab().clone())
        })
        .collect()
}

fn start(vocab: &Vocab) -> (Vec<usize>, State) {
    assert_eq!(vocab.proposal_prefix("regression").unwrap(), PREFIX);
    (PREFIX.to_vec(), vocab.prefix_state("regression").unwrap())
}

/// Tokens that never follow the prefix: PAD, BOS, the engine and every task token.
fn reserved(vocab: &Vocab) -> Vec<usize> {
    let mut ids = vec![PAD, BOS];
    ids.extend(vocab.engine_ids.values());
    ids.extend(vocab.task_ids.values());
    ids
}

/// Ids tried against `advance`: every token and out-of-range ids, or a fixed sample.
fn probe_ids(vocab: &Vocab, legal: &[usize], every: bool) -> Vec<usize> {
    let n = vocab.tokens.len();
    let mut ids = if every {
        (0..n).collect::<Vec<_>>()
    } else {
        let mut ids = legal.to_vec();
        ids.extend((0..n).step_by(7));
        ids.extend([PAD, BOS, EOS]);
        ids
    };
    ids.extend([n, n + 1, 10_000, usize::MAX]);
    ids
}

/// A deterministic legal completion: the first non-function leaf in expressions, the first value
/// or absent token, the first append flag, and the slot-free `ridge` head at a stage.
fn complete(vocab: &Vocab, tokens: &[usize], state: &State) -> Vec<usize> {
    let mut tokens = tokens.to_vec();
    let mut state = state.clone();
    while !state.done() {
        let legal = state.legal(vocab);
        let token = if legal.contains(&RIDGE) {
            RIDGE
        } else {
            *legal
                .iter()
                .find(|t| !vocab.expr.func_arity.contains_key(t))
                .expect("every expression position admits a leaf")
        };
        state.advance(vocab, token).unwrap();
        tokens.push(token);
    }
    tokens
}

/// The production admission path accepts `tokens` exactly; returns the genome.
fn check_accepted(vocab: &Vocab, tokens: &[usize]) -> String {
    assert!(tokens.len() <= MAX_PROGRAM_TOKENS);
    let legal = vocab.replay(tokens).unwrap();
    for (set, token) in legal.iter().zip(&tokens[1..]) {
        assert!(set.contains(token));
    }
    let genome = vocab
        .decode(tokens)
        .unwrap_or_else(|e| panic!("{tokens:?}: {e:#}"));
    feat_math::GenomeIr::parse(&genome).unwrap_or_else(|e| panic!("{genome}: {e:#}"));
    assert_eq!(
        vocab
            .encode(&genome)
            .unwrap_or_else(|e| panic!("{genome}: {e:#}")),
        tokens,
        "{genome} does not re-encode to its sampled tokens"
    );
    assert_eq!(
        decode::valid_genome(vocab, tokens).as_deref(),
        Some(genome.as_str())
    );
    genome
}

/// All state checks for a nonterminal state reached by `tokens`.
fn check_state(vocab: &Vocab, tokens: &[usize], state: &State, every: bool) {
    let legal = state.legal(vocab);
    let phase = state.phase_name();
    assert!(!state.done());
    assert!(
        !legal.is_empty(),
        "dead-end legal prefix {tokens:?} ({phase})"
    );
    assert!(legal.windows(2).all(|w| w[0] < w[1]));
    let reserved = reserved(vocab);
    assert!(legal
        .iter()
        .all(|t| *t < vocab.tokens.len() && !reserved.contains(t)));
    assert_eq!(legal.contains(&EOS), phase == "eos", "{tokens:?}");
    if phase == "eos" {
        assert_eq!(legal, [EOS]);
    }
    for id in probe_ids(vocab, &legal, every) {
        let mut next = state.clone();
        let accepted = next.advance(vocab, id).is_ok();
        assert_eq!(accepted, legal.contains(&id), "{tokens:?} + {id}");
        if !accepted {
            assert_eq!(next.phase_name(), phase);
            assert_eq!(next.legal(vocab), legal);
        }
    }
    let mut with_eos = tokens.to_vec();
    with_eos.push(EOS);
    assert_eq!(
        vocab.decode(&with_eos).is_ok(),
        phase == "eos",
        "{with_eos:?}"
    );
    check_accepted(vocab, &complete(vocab, tokens, state));
}

#[test]
fn masks_agree_with_the_typed_parser_over_the_bounded_state_domain() {
    for (name, vocab) in vocabularies() {
        let (mut checked, mut accepted) = (0usize, 0usize);
        let mut frontier = vec![start(&vocab)];
        for depth in 0..=DEPTH {
            let mut next = Vec::new();
            for (tokens, state) in frontier {
                if state.done() {
                    check_accepted(&vocab, &tokens);
                    accepted += 1;
                    continue;
                }
                check_state(&vocab, &tokens, &state, depth <= 2);
                checked += 1;
                if depth < DEPTH {
                    for token in state.legal(&vocab) {
                        let mut child = state.clone();
                        child.advance(&vocab, token).unwrap();
                        let mut path = tokens.clone();
                        path.push(token);
                        next.push((path, child));
                    }
                }
            }
            frontier = next;
        }
        let mut walked = 0usize;
        for seed in 0..WALKS {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (mut tokens, mut state) = start(&vocab);
            while !state.done() {
                check_state(&vocab, &tokens, &state, false);
                walked += 1;
                let legal = state.legal(&vocab);
                let token = legal[rng.gen_range(0..legal.len())];
                state.advance(&vocab, token).unwrap();
                tokens.push(token);
            }
            check_accepted(&vocab, &tokens);
        }
        eprintln!(
            "{name}: {checked} enumerated states, {accepted} enumerated accepted sequences, {walked} walk states, {WALKS} walks"
        );
        assert!(checked > 2_000 && accepted > 0 && walked > 2 * WALKS as usize);
    }
}

/// Every choice sequence of `op`'s value and append positions, then a legal completion.
fn slot_family(vocab: &Vocab, op: usize) -> Vec<Vec<usize>> {
    let (mut tokens, mut state) = start(vocab);
    state.advance(vocab, op).unwrap();
    tokens.push(op);
    let mut open = vec![(tokens, state)];
    let mut out = Vec::new();
    while let Some((tokens, state)) = open.pop() {
        if matches!(state.phase_name(), "value" | "append") {
            for token in state.legal(vocab) {
                let mut child = state.clone();
                child.advance(vocab, token).unwrap();
                let mut path = tokens.clone();
                path.push(token);
                open.push((path, child));
            }
        } else {
            out.push(complete(vocab, &tokens, &state));
        }
    }
    out
}

/// Every expression of at most `nodes` nodes for expression operator `op`, completed.
fn expression_family(vocab: &Vocab, op: usize, nodes: usize) -> Vec<Vec<usize>> {
    let (mut tokens, mut state) = start(vocab);
    state.advance(vocab, op).unwrap();
    tokens.push(op);
    let mut open = vec![(tokens, state, 0usize)];
    let mut out = Vec::new();
    while let Some((tokens, state, used)) = open.pop() {
        if matches!(state.phase_name(), "value" | "expression") {
            for token in state.legal(vocab) {
                // Close every open argument with leaves within the node budget.
                let arity = vocab.expr.func_arity.get(&token).copied().unwrap_or(0);
                let pending = open_arguments(vocab, &tokens) - 1 + arity;
                if used + 1 + pending > nodes {
                    continue;
                }
                let mut child = state.clone();
                child.advance(vocab, token).unwrap();
                let mut path = tokens.clone();
                path.push(token);
                open.push((path, child, used + 1));
            }
        } else {
            out.push(complete(vocab, &tokens, &state));
        }
    }
    out
}

/// Open argument positions of the expression that ends `tokens` (1 before its first token).
fn open_arguments(vocab: &Vocab, tokens: &[usize]) -> usize {
    let expression = tokens
        .iter()
        .rposition(|t| vocab.ops.contains_key(t))
        .map_or(&[][..], |i| &tokens[i + 1..]);
    expression.iter().fold(1usize, |open, token| {
        open - 1 + vocab.expr.func_arity.get(token).copied().unwrap_or(0)
    })
}

#[test]
fn every_slot_value_and_small_expression_parses_and_round_trips() {
    for (name, vocab) in vocabularies() {
        let (_, state) = start(&vocab);
        let stage = state.legal(&vocab);
        let (mut slots, mut expressions) = (0usize, 0usize);
        for &op in &stage {
            if vocab.ops[&op].takes_expr {
                let family = expression_family(&vocab, op, 3);
                assert!(family.len() > 1_000, "{name} {op}: {}", family.len());
                for tokens in family {
                    check_accepted(&vocab, &tokens);
                    expressions += 1;
                }
            } else {
                for tokens in slot_family(&vocab, op) {
                    check_accepted(&vocab, &tokens);
                    slots += 1;
                }
            }
        }
        eprintln!("{name}: {slots} slot-family programs, {expressions} expression programs");
        assert!(slots > 1_000 && expressions > 2_000);
    }
}

#[test]
fn eos_invalid_ids_and_mode_differences_are_refused_or_documented() {
    for (name, vocab) in vocabularies() {
        let prefix = PREFIX.to_vec();
        let with = |tail: &[usize]| [prefix.as_slice(), tail].concat();
        // Positive examples kept alongside the refusals.
        for tokens in [with(&[RIDGE, EOS]), with(&[372, EOS])] {
            check_accepted(&vocab, &tokens);
        }
        let expr_end = vocab.expr.end_id;
        let refused = [
            with(&[EOS]),                          // EOS at a stage: no head yet
            with(&[206, EOS]),                     // EOS before the head's values
            with(&[53, EOS]),                      // EOS inside a transform
            with(&[59, 50, EOS]),                  // transform only, no head
            with(&[66, 45, 5, EOS]),               // unterminated expression
            with(&[66, 45, 5, 6, EOS]),            // expression without its end token
            with(&[66, expr_end, 50, RIDGE, EOS]), // empty expression
            with(&[RIDGE, EOS, EOS]),              // EOS after acceptance
            with(&[RIDGE, EOS, RIDGE]),            // token after acceptance
            with(&[RIDGE]),                        // accepting state without EOS
            with(&[PAD, RIDGE, EOS]),              // PAD inside a program
            with(&[BOS, RIDGE, EOS]),              // BOS inside a program
            with(&[3, RIDGE, EOS]),                // engine token inside a program
            with(&[397, RIDGE, EOS]),              // task token inside a program
            with(&[401, EOS]),                     // out-of-range ids
            with(&[10_000, EOS]),
            with(&[usize::MAX, EOS]),
            with(&[324, EOS]),        // multiclass-only head under regression
            vec![3, 397, RIDGE, EOS], // BOS missing
            vec![],
        ];
        for tokens in refused {
            assert!(vocab.decode(&tokens).is_err(), "{name}: {tokens:?} decoded");
            assert!(decode::valid_genome(&vocab, &tokens).is_none());
            // The sampling automaton does not accept it either.
            if tokens.len() > 3 && tokens[..3] == PREFIX {
                let mut state = vocab.prefix_state("regression").unwrap();
                let accepted = tokens[3..]
                    .iter()
                    .try_for_each(|&t| state.advance(&vocab, t))
                    .is_ok()
                    && state.done();
                assert!(!accepted, "{name}: sampling automaton accepted {tokens:?}");
            }
        }
        // `generate=false` heads (ridge_gcv) are never sampled, but the non-sampling replay used
        // by decode/encode admits them: incumbents and teachers may carry them. The sampler's
        // support is a subset of the admission language, never the reverse.
        let ridge_gcv = 371;
        let mut state = vocab.prefix_state("regression").unwrap();
        assert!(!state.legal(&vocab).contains(&ridge_gcv));
        assert!(state.advance(&vocab, ridge_gcv).is_err());
        if vocab.ops.contains_key(&ridge_gcv) {
            check_accepted(&vocab, &with(&[ridge_gcv, EOS]));
        }
        // The task token is not rendered: a binary-task sequence decodes to the regression
        // genome but does not round-trip. The decoder's fixed prefix never emits it.
        let binary = [BOS, 3, 398, RIDGE, EOS];
        let genome = vocab.decode(&binary).unwrap();
        assert_eq!(genome, "(pipe X (ridge))");
        assert_ne!(vocab.encode(&genome).unwrap(), binary);
        assert!(vocab.prefix_state("binary").is_err());
    }
}

#[test]
fn longest_sampled_program_fits_the_token_budget() {
    for (name, vocab) in vocabularies() {
        // Upper bound from the tables: the prefix, `max_expr_stages` of the longest expression
        // transform, the remaining transforms at their longest, the longest regression head and
        // EOS. An expression slot holds at most `max_nodes` nodes plus its end token.
        let length = |id: &usize| {
            let op = &vocab.ops[id];
            1 + op
                .slots
                .iter()
                .map(|s| {
                    if s.is_expr {
                        vocab.expr.max_nodes + 1
                    } else {
                        1
                    }
                })
                .sum::<usize>()
                + usize::from(op.allow_append)
        };
        let (_, state) = start(&vocab);
        let stage = state.legal(&vocab);
        let longest = |expr: bool, role: &str| {
            stage
                .iter()
                .filter(|id| vocab.ops[id].takes_expr == expr && vocab.ops[id].role == role)
                .map(length)
                .max()
                .unwrap_or(0)
        };
        let (transforms, expression_stages) = (
            vocab.max_transforms["featmath"],
            vocab.max_expr_stages["featmath"],
        );
        let bound = PREFIX.len()
            + expression_stages * longest(true, "transform")
            + (transforms - expression_stages) * longest(false, "transform")
            + longest(false, "head")
            + 1;
        // A constructive walk reaches the bound: expression stages first, then the longest
        // transforms and head; inside expressions, functions of the highest arity.
        let (mut tokens, mut state) = start(&vocab);
        while !state.done() {
            let legal = state.legal(&vocab);
            let token = *legal
                .iter()
                .max_by_key(|&&t| {
                    let key = if vocab.ops.contains_key(&t) {
                        (2, length(&t))
                    } else if let Some(&arity) = vocab.expr.func_arity.get(&t) {
                        (1, arity)
                    } else {
                        (0, 0)
                    };
                    (key, std::cmp::Reverse(t))
                })
                .unwrap();
            state.advance(&vocab, token).unwrap();
            tokens.push(token);
        }
        check_accepted(&vocab, &tokens);
        eprintln!(
            "{name}: longest program {} tokens, bound {bound}",
            tokens.len()
        );
        assert_eq!(tokens.len(), bound);
        // No accepted program reaches MAX_PROGRAM_TOKENS, so the 128-token limit never truncates
        // a sampled program under the accepted configuration (max_seq 128); only a smaller
        // `max_seq` produces incomplete draws (see `decode_outcomes`).
        assert!(bound < MAX_PROGRAM_TOKENS);
    }
}
