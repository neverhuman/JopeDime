use super::*;

impl ProposalLedger {
    /// [`Self::reconcile`], plus the vocabulary-dependent links: every draw starts from the
    /// task's proposal prefix and its recorded outcome is what its tokens decode and parse to (so
    /// a valid program cannot be recorded as a refusal, nor a refusal as a program), each
    /// candidate's tokens decode to its program, its scoring encoding is the vocabulary's
    /// encoding of that program, and its canonical recipe is the parser's.
    pub fn reconcile_with_vocabulary(&self, vocab: &Vocab) -> Result<()> {
        self.reconcile()?;
        let prefix = vocab.proposal_prefix(&self.task)?;
        let program_of = |tokens: &[usize]| -> Result<Option<String>> {
            ensure!(
                tokens.starts_with(&prefix),
                "draw does not start from the task's proposal prefix"
            );
            Ok((tokens.last() == Some(&EOS))
                .then(|| super::super::decode::valid_genome(vocab, tokens))
                .flatten())
        };
        // Replay a draw through the sampling grammar: every drawn token is legal where it was
        // drawn; returns whether the walk accepted (reached EOS) and its final phase.
        let walk = |tokens: &[usize]| -> Result<Option<(bool, &'static str)>> {
            let mut state = vocab.prefix_state(&self.task)?;
            for &token in tokens.get(prefix.len()..).unwrap_or_default() {
                if !state.legal(vocab).contains(&token) {
                    return Ok(None);
                }
                state.advance(vocab, token)?;
            }
            Ok(Some((state.done(), state.phase_name())))
        };
        for d in &self.neural_draws {
            let program = program_of(&d.tokens)?;
            // An incomplete draw stopped before the grammar accepted; every other draw ended at
            // an accepting EOS. Both are legal walks ending in the recorded grammar phase.
            ensure!(
                walk(&d.tokens)?
                    == Some((
                        d.outcome != DrawOutcome::Incomplete,
                        d.grammar_phase.as_str()
                    )),
                "neural draw ({}, {}) is not a legal grammar walk ending in its recorded outcome",
                d.round,
                d.row
            );
            let valid = matches!(
                d.outcome,
                DrawOutcome::Completed | DrawOutcome::Surplus | DrawOutcome::Duplicate
            );
            ensure!(
                valid == program.is_some()
                    && program.map(|p| sha256_hex(p.as_bytes())) == d.program_sha256,
                "neural draw ({}, {}) outcome differs from its tokens",
                d.round,
                d.row
            );
        }
        for d in self.fallback_draws.iter().chain(&self.exploration_draws) {
            let program = program_of(&d.tokens)?;
            ensure!(
                walk(&d.tokens)?.map(|(accepted, _)| accepted)
                    == Some(d.outcome != UniformOutcome::Incomplete),
                "grammar-uniform draw {} is not a legal grammar walk ending in its recorded outcome",
                d.seed
            );
            let exact = program.as_ref().map(|p| {
                vocab.encode_for_task(p, &self.task).ok().as_deref() == Some(&d.tokens[..])
            });
            let consistent = match d.outcome {
                UniformOutcome::Admitted | UniformOutcome::Duplicate => exact == Some(true),
                UniformOutcome::ReencodingMismatch => exact == Some(false),
                UniformOutcome::ParserRejected | UniformOutcome::Incomplete => program.is_none(),
            };
            ensure!(
                consistent && program.map(|p| sha256_hex(p.as_bytes())) == d.program_sha256,
                "grammar-uniform draw {} outcome differs from its tokens",
                d.seed
            );
        }
        for c in &self.candidates {
            ensure!(
                vocab.decode(&c.tokens)? == c.program
                    && canonical_sha256(&c.program)? == c.canonical_recipe_sha256,
                "candidate {} tokens do not decode to its program",
                c.slot
            );
            if let Some(edge) = c.adaptations.get(1) {
                let scored = vocab.encode(&c.program)?;
                ensure!(
                    edge.output_sha256 == tokens_sha256(&scored)?
                        && edge.changed == (scored != c.tokens)
                        && c.prediction.as_ref().map(|p| p.scored_token_count)
                            == Some(scored.len()),
                    "candidate {} scoring encoding differs from the vocabulary's",
                    c.slot
                );
            }
        }
        Ok(())
    }

    /// The ledger describes exactly this seed population (slot order, program, origin, weight).
    pub fn matches_population(&self, population: &[SeedCandidate]) -> Result<()> {
        ensure!(
            population.len() == self.candidates.len(),
            "ledger and population sizes differ"
        );
        for (c, seed) in self.candidates.iter().zip(population) {
            ensure!(
                c.program == seed.genome()
                    && c.origin == seed.origin()
                    && c.prediction.as_ref().map(|p| p.rank_log_weight) == seed.rank_log_weight(),
                "ledger slot {} differs from the seed population",
                c.slot
            );
            // W4-T07 join: a seed slot that names a proposal record names exactly this slot's.
            if let Some(record) = seed.proposal_record_sha256() {
                ensure!(
                    record == sha256_hex(&serde_json::to_vec(c)?),
                    "seed slot {} names another proposal record",
                    c.slot
                );
            }
        }
        Ok(())
    }
}
