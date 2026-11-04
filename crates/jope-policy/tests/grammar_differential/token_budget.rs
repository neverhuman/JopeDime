use super::*;

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
