use super::*;

fn uniform_draws(records: &[UniformDrawRecord]) -> Vec<UniformDraw> {
    records
        .iter()
        .map(|r| UniformDraw {
            seed: r.seed,
            outcome: r.outcome,
            tokens: r.tokens.clone(),
            program_sha256: r.program_sha256.clone(),
        })
        .collect()
}

/// Everything [`build`] needs from one proposal run.
pub(crate) struct ProposalRun<'a> {
    pub task: &'a str,
    pub checkpoint_sha256: &'a str,
    pub context_sha256: &'a str,
    pub vocabulary_sha256: &'a str,
    pub pool: &'a PurePool,
    pub exploration_seed: u64,
    pub exploration_draws: &'a [UniformDrawRecord],
    pub ranking: &'a Ranking,
    pub population: &'a [SeedCandidate],
    pub vocab: &'a Vocab,
}

/// Record a proposal run; the result is reconciled against the vocabulary before it is returned.
pub(crate) fn build(run: &ProposalRun<'_>) -> Result<ProposalLedger> {
    let pool = run.pool;
    let batch = pool.options.batch;
    ensure!(
        batch > 0 && pool.attempted.is_multiple_of(batch),
        "pool attempts are not whole rounds"
    );
    let neural_draws = pool
        .draws()
        .iter()
        .map(|d| NeuralDraw {
            round: d.draw.round,
            row: d.draw.row,
            outcome: d.outcome,
            tokens: d.tokens.clone(),
            grammar_phase: d.grammar_phase.into(),
            program_sha256: d.program_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let fallback_draws = uniform_draws(pool.fallback_draws());
    let exploration_draws = uniform_draws(run.exploration_draws);
    let selected = run.ranking.selected();
    ensure!(
        run.population.len() == SEED_COUNT && selected.len() == POOL_SIZE,
        "proposal population or ranking size changed"
    );
    let mut candidates = Vec::with_capacity(SEED_COUNT);
    for (slot, seed) in run.population.iter().enumerate() {
        let program = seed.genome().to_owned();
        let program_sha256 = sha256_hex(program.as_bytes());
        let (source, tokens) = match seed.origin() {
            ORIGIN_NEURAL => {
                let d = neural_draws
                    .iter()
                    .find(|d| {
                        d.outcome == DrawOutcome::Completed
                            && d.program_sha256.as_deref() == Some(program_sha256.as_str())
                    })
                    .context("neural candidate without its completed draw")?;
                (
                    SlotSource::NeuralDraw {
                        round: d.round,
                        row: d.row,
                    },
                    d.tokens.clone(),
                )
            }
            origin => {
                let (draws, fallback) = match origin {
                    ORIGIN_NEURAL_FALLBACK => (&fallback_draws, true),
                    ORIGIN_RANDOM => (&exploration_draws, false),
                    _ => bail!("unknown proposal origin {origin}"),
                };
                let d = draws
                    .iter()
                    .find(|d| {
                        d.outcome == UniformOutcome::Admitted
                            && d.program_sha256.as_deref() == Some(program_sha256.as_str())
                    })
                    .context("grammar-uniform candidate without its admitted draw")?;
                let source = if fallback {
                    SlotSource::FallbackDraw { seed: d.seed }
                } else {
                    SlotSource::ExplorationDraw { seed: d.seed }
                };
                (source, d.tokens.clone())
            }
        };
        let mut adaptations = vec![AdaptationEdge {
            kind: EDGE_DECODE.into(),
            input_sha256: tokens_sha256(&tokens)?,
            output_sha256: program_sha256.clone(),
            changed: false,
        }];
        let prediction = if slot < POOL_SIZE {
            let p = &selected[slot];
            // The score subject is this very program; the ranking re-encodes the text.
            ensure!(
                p.program == program && p.program_sha256 == program_sha256,
                "ranked prediction belongs to another program"
            );
            let scored = run.vocab.encode(&program)?;
            ensure!(
                scored.len() == p.token_count,
                "scored token count differs from the scoring encoding"
            );
            adaptations.push(AdaptationEdge {
                kind: EDGE_SCORE_ENCODING.into(),
                input_sha256: program_sha256.clone(),
                output_sha256: tokens_sha256(&scored)?,
                changed: scored != tokens,
            });
            Some(SlotPrediction {
                rank: slot,
                predicted_quality: p.predicted_quality,
                reward_heads: p.reward_heads,
                predicted_runtime_seconds: p.predicted_runtime_seconds,
                rank_log_weight: seed
                    .rank_log_weight()
                    .context("ranked pool slot without its rank weight")?,
                scored_program_sha256: p.program_sha256.clone(),
                scored_token_count: p.token_count,
            })
        } else {
            None
        };
        candidates.push(SlotRecord {
            slot,
            origin: seed.origin().into(),
            source,
            tokens,
            canonical_recipe_sha256: canonical_sha256(&program)?,
            program,
            program_sha256,
            adaptations,
            prediction,
        });
    }
    let mut ledger = ProposalLedger {
        schema: LEDGER_SCHEMA.into(),
        task: run.task.into(),
        checkpoint_sha256: run.checkpoint_sha256.into(),
        context_sha256: run.context_sha256.into(),
        vocabulary_sha256: run.vocabulary_sha256.into(),
        sampling: SamplingRecord {
            sampling_seed: pool.sampling_seed,
            fallback_seed: pool.fallback_seed,
            exploration_seed: run.exploration_seed,
            rounds_scheduled: pool.options.rounds,
            rounds_run: pool.attempted / batch,
            batch,
            temperature: pool.options.temperature,
            token_budget: pool.token_budget,
        },
        neural_draws,
        fallback_draws,
        exploration_draws,
        candidates,
        denominators: Denominators {
            scheduled_neural_draws: 0,
            completed: 0,
            surplus: 0,
            duplicate: 0,
            parser_rejected: 0,
            incomplete: 0,
            complete_sequences: 0,
            valid_programs: 0,
            unique_programs: 0,
            fallback_draws: 0,
            fallback_admitted: 0,
            exploration_draws: 0,
            exploration_admitted: 0,
            scored: 0,
            rank_survivors: 0,
            handed_off: 0,
            unique_canonical_handed_off: 0,
            shadow: 0,
        },
    };
    ledger.denominators = ledger.count();
    // The pool's own counters agree with the records.
    let d = &ledger.denominators;
    ensure!(
        d.scheduled_neural_draws == pool.attempted
            && d.valid_programs == pool.valid
            && d.duplicate == pool.within_neural_duplicates
            && d.completed == pool.unique_neural
            && d.fallback_draws as u64 == pool.fallback_attempts
            && run.ranking.chosen_count() == POOL_SIZE,
        "proposal counters differ from the draw records"
    );
    ledger.reconcile_with_vocabulary(run.vocab)?;
    Ok(ledger)
}
