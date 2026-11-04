use super::*;

impl ProposalLedger {
    /// Re-derive every count and link from the records alone (no model, no vocabulary).
    pub fn reconcile(&self) -> Result<()> {
        ensure!(
            self.schema == LEDGER_SCHEMA,
            "unknown proposal ledger schema"
        );
        // `propose` samples regression programs only.
        ensure!(self.task == "regression", "unknown proposal task");
        for digest in [
            &self.checkpoint_sha256,
            &self.context_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(
                super::super::is_sha256_hex(digest),
                "invalid proposal ledger identity"
            );
        }
        let s = &self.sampling;
        ensure!(
            (1..=MAX_ROUNDS).contains(&s.rounds_scheduled)
                && (1..=MAX_BATCH).contains(&s.batch)
                && (1..=s.rounds_scheduled).contains(&s.rounds_run)
                && s.temperature.is_finite()
                && s.temperature > 0.0
                && (1..=MAX_PROGRAM_TOKENS).contains(&s.token_budget),
            "sampling record outside the bounded configuration"
        );

        // Every scheduled neural draw has exactly one terminal record, in draw order.
        ensure!(
            self.neural_draws.len() == s.rounds_run * s.batch,
            "scheduled neural draws lack terminal records"
        );
        let mut seen = BTreeSet::new();
        let mut pooled = 0usize;
        for (i, d) in self.neural_draws.iter().enumerate() {
            ensure!(
                d.round == i / s.batch && d.row == i % s.batch,
                "neural draw records are out of order or missing"
            );
            ensure!(
                d.tokens.len() <= s.token_budget,
                "neural draw exceeds the token budget"
            );
            let finished = d.tokens.last() == Some(&EOS);
            match (d.outcome, d.program_sha256.as_deref()) {
                (DrawOutcome::Incomplete, None) => {
                    ensure!(!finished, "an incomplete draw reached EOS");
                    // The decoder ends a row only at EOS or at the token budget.
                    ensure!(
                        d.tokens.len() == s.token_budget,
                        "an incomplete draw did not exhaust its token budget"
                    )
                }
                (DrawOutcome::ParserRejected, None) => {
                    ensure!(finished, "a parser refusal did not reach EOS")
                }
                (DrawOutcome::Completed, Some(sha)) => ensure!(
                    finished && pooled < POOL_SIZE && seen.insert(sha.to_owned()),
                    "completed draw is not a new program entering an unfilled pool"
                ),
                (DrawOutcome::Surplus, Some(sha)) => ensure!(
                    finished && pooled == POOL_SIZE && seen.insert(sha.to_owned()),
                    "surplus draw is not a new program after the pool filled"
                ),
                (DrawOutcome::Duplicate, Some(sha)) => ensure!(
                    finished && seen.contains(sha),
                    "duplicate draw repeats no earlier program"
                ),
                _ => bail!("neural draw outcome and program identity disagree"),
            }
            if d.outcome == DrawOutcome::Completed {
                pooled += 1;
            }
            let last_of_round = d.row + 1 == s.batch;
            if last_of_round && d.round + 1 < s.rounds_run {
                ensure!(
                    pooled < POOL_SIZE,
                    "sampling continued after the pool filled"
                );
            }
        }
        ensure!(
            s.rounds_run == s.rounds_scheduled || pooled == POOL_SIZE,
            "sampling stopped before its scheduled rounds with an unfilled pool"
        );

        // The fallback fill tops the pool up to POOL_SIZE; the exploration fill draws
        // EXPLORATION_SIZE more programs; both from fresh consecutive streams.
        let mut filled = pooled;
        Self::reconcile_fill(
            &self.fallback_draws,
            s.fallback_seed,
            &mut seen,
            &mut filled,
            POOL_SIZE,
            "fallback",
        )?;
        let mut explored = 0;
        Self::reconcile_fill(
            &self.exploration_draws,
            s.exploration_seed,
            &mut seen,
            &mut explored,
            EXPLORATION_SIZE,
            "exploration",
        )?;

        // Each handed-off candidate comes from exactly one admitting draw, and every admitting
        // draw is handed off exactly once.
        ensure!(
            self.candidates.len() == SEED_COUNT,
            "handed-off candidates are not the {SEED_COUNT} seed slots"
        );
        let mut used = BTreeSet::new();
        let mut qualities = Vec::with_capacity(POOL_SIZE);
        for (slot, c) in self.candidates.iter().enumerate() {
            ensure!(
                c.slot == slot && sha256_hex(c.program.as_bytes()) == c.program_sha256,
                "candidate slot or program identity changed"
            );
            let pool_slot = slot < POOL_SIZE;
            let (key, admitted_tokens) = match (&c.source, c.origin.as_str()) {
                (SlotSource::NeuralDraw { round, row }, ORIGIN_NEURAL) if pool_slot => {
                    let d = self
                        .neural_draws
                        .get(round * s.batch + row)
                        .filter(|d| d.round == *round && d.row == *row)
                        .context("neural candidate names no scheduled draw")?;
                    ensure!(
                        d.outcome == DrawOutcome::Completed
                            && d.program_sha256.as_deref() == Some(c.program_sha256.as_str()),
                        "neural candidate is not its draw's completed program"
                    );
                    (format!("neural:{round}:{row}"), &d.tokens)
                }
                (SlotSource::FallbackDraw { seed }, ORIGIN_NEURAL_FALLBACK) if pool_slot => {
                    let d = Self::admitted(&self.fallback_draws, *seed, &c.program_sha256)?;
                    (format!("fallback:{seed}"), &d.tokens)
                }
                (SlotSource::ExplorationDraw { seed }, ORIGIN_RANDOM) if !pool_slot => {
                    let d = Self::admitted(&self.exploration_draws, *seed, &c.program_sha256)?;
                    (format!("exploration:{seed}"), &d.tokens)
                }
                _ => bail!("candidate {slot} origin, source and slot range disagree"),
            };
            ensure!(used.insert(key), "one draw is handed off more than once");
            ensure!(
                c.tokens == *admitted_tokens,
                "candidate tokens differ from its draw"
            );
            // Exactly the declared proposal-stage adaptations, each applied once.
            let expected = if pool_slot { 2 } else { 1 };
            ensure!(
                c.adaptations.len() == expected
                    && c.adaptations[0].kind == EDGE_DECODE
                    && c.adaptations[0].input_sha256 == tokens_sha256(&c.tokens)?
                    && c.adaptations[0].output_sha256 == c.program_sha256
                    && !c.adaptations[0].changed
                    && c.adaptations.get(1).is_none_or(|e| {
                        e.kind == EDGE_SCORE_ENCODING && e.input_sha256 == c.program_sha256
                    }),
                "candidate {slot} proposal-stage adaptations are not exactly one decode (and one scoring encoding)"
            );
            match (&c.prediction, pool_slot) {
                (Some(p), true) => {
                    ensure!(
                        p.rank == slot
                            && p.scored_program_sha256 == c.program_sha256
                            && p.predicted_quality.is_finite()
                            && p.rank_log_weight.is_finite()
                            && p.predicted_runtime_seconds.is_finite()
                            && p.predicted_runtime_seconds >= 0.0
                            && p.reward_heads.iter().all(|h| h.is_finite()),
                        "prediction is not attached to candidate {slot}'s own program"
                    );
                    qualities.push((p.predicted_quality, &c.program_sha256, &c.program));
                }
                (None, false) => {}
                _ => bail!("candidate {slot} prediction presence differs from its slot range"),
            }
        }
        ensure!(
            used.len()
                == self.denominators.completed
                    + self.denominators.fallback_admitted
                    + self.denominators.exploration_admitted,
            "an admitted draw was dropped from the handed-off candidates"
        );
        // Ranking order and the rank weights derived from it.
        for pair in qualities.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            ensure!(
                a.0 > b.0 || (a.0 == b.0 && (a.1, a.2) < (b.1, b.2)),
                "pool slots are not in the documented ranking order"
            );
        }
        let weights = logprob_from_scores(&qualities.iter().map(|q| q.0).collect::<Vec<_>>())?;
        for (c, w) in self.candidates.iter().zip(weights) {
            let p = c.prediction.as_ref().expect("pool slot prediction");
            ensure!(
                p.rank_log_weight == w,
                "rank weight is not derived from the ranked predictions"
            );
        }
        ensure!(
            self.denominators == self.count(),
            "denominators differ from the records"
        );
        Ok(())
    }

    fn admitted<'a>(draws: &'a [UniformDraw], seed: u64, sha: &str) -> Result<&'a UniformDraw> {
        let d = draws
            .iter()
            .find(|d| d.seed == seed)
            .context("grammar-uniform candidate names no draw")?;
        ensure!(
            d.outcome == UniformOutcome::Admitted && d.program_sha256.as_deref() == Some(sha),
            "grammar-uniform candidate is not its draw's admitted program"
        );
        Ok(d)
    }

    fn reconcile_fill(
        draws: &[UniformDraw],
        seed: u64,
        seen: &mut BTreeSet<String>,
        filled: &mut usize,
        target: usize,
        stage: &str,
    ) -> Result<()> {
        ensure!(
            draws.len() as u64 <= FALLBACK_ATTEMPTS,
            "{stage} fill exceeds its draw bound"
        );
        for (offset, d) in draws.iter().enumerate() {
            ensure!(
                d.seed == seed.wrapping_add(offset as u64) && *filled < target,
                "{stage} draws are not consecutive fresh streams of an unfilled fill"
            );
            let finished = d.tokens.last() == Some(&EOS);
            match (d.outcome, d.program_sha256.as_deref()) {
                (UniformOutcome::Admitted, Some(sha)) => {
                    ensure!(
                        finished && seen.insert(sha.to_owned()),
                        "{stage} admitted a program already seen"
                    );
                    *filled += 1;
                }
                (UniformOutcome::Duplicate, Some(sha)) => ensure!(
                    finished && seen.contains(sha),
                    "{stage} duplicate repeats no earlier program"
                ),
                (UniformOutcome::ReencodingMismatch, Some(_)) => {
                    ensure!(finished, "{stage} re-encoding refusal did not reach EOS")
                }
                (UniformOutcome::ParserRejected, None) => {
                    ensure!(finished, "{stage} parser refusal did not reach EOS")
                }
                (UniformOutcome::Incomplete, None) => {
                    ensure!(!finished, "{stage} incomplete draw reached EOS");
                    // A grammar-uniform walk ends only at EOS or at MAX_PROGRAM_TOKENS.
                    ensure!(
                        d.tokens.len() == MAX_PROGRAM_TOKENS,
                        "{stage} incomplete draw did not exhaust its token budget"
                    )
                }
                _ => bail!("{stage} draw outcome and program identity disagree"),
            }
        }
        ensure!(*filled == target, "{stage} fill did not complete");
        Ok(())
    }
}
