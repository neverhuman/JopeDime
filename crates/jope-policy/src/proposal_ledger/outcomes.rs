use super::*;

impl ProposalLedger {
    /// Every draw's terminal outcome, for the candidate ledger of the search the population
    /// seeds (W4-T03): each record names the draw, its outcome, the digests of its tokens and
    /// program and, when it was handed off, its slot and slot record identity.
    pub fn draw_outcomes(&self) -> Result<DrawOutcomes> {
        fn label<T: Serialize>(outcome: &T) -> Result<String> {
            serde_json::to_value(outcome)?
                .as_str()
                .map(str::to_owned)
                .context("draw outcome is not a label")
        }
        let mut entries = Vec::with_capacity(
            self.neural_draws.len() + self.fallback_draws.len() + self.exploration_draws.len(),
        );
        for d in &self.neural_draws {
            let draw = SlotSource::NeuralDraw {
                round: d.round,
                row: d.row,
            };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        for d in &self.fallback_draws {
            let draw = SlotSource::FallbackDraw { seed: d.seed };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        for d in &self.exploration_draws {
            let draw = SlotSource::ExplorationDraw { seed: d.seed };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        let mut draws = Vec::with_capacity(entries.len());
        for (draw, outcome, tokens, program_sha256) in entries {
            let slot = self
                .candidates
                .iter()
                .find(|c| c.source == draw)
                .map(|c| c.slot);
            draws.push(DrawOutcomeRecord {
                slot_record_sha256: slot.map(|s| self.slot_record_sha256(s)).transpose()?,
                draw,
                outcome,
                tokens_sha256: tokens_sha256(tokens)?,
                program_sha256: program_sha256.clone(),
                slot,
            });
        }
        Ok(DrawOutcomes {
            ledger_sha256: self.sha256()?,
            draws,
            denominators: self.denominators.clone(),
        })
    }

    /// Export all existing draws and exact denominators, with stable I02 IDs and optional
    /// downstream decision references. A companion cannot invent a new draw or fitting grant.
    pub fn decision_links(
        &self,
        joins: &BTreeMap<String, Vec<String>>,
    ) -> Result<ProposalDecisionLinks> {
        self.reconcile()?;
        let outcomes = self.draw_outcomes()?;
        let mut used = BTreeSet::new();
        let mut draws = Vec::new();
        for draw in outcomes.draws {
            let key = match draw.draw {
                SlotSource::NeuralDraw { round, row } => format!("neural/{round}/{row}"),
                SlotSource::FallbackDraw { seed } => format!("fallback/{seed:016x}"),
                SlotSource::ExplorationDraw { seed } => format!("exploration/{seed:016x}"),
            };
            let payload = event_wire::CanonicalPayload::parse_and_canonicalize(
                &serde_json::to_vec(&serde_json::json!({
                    "schema": "jope.selfplay.draw-request/v1", "ledger_sha256": outcomes.ledger_sha256, "draw_key": key
                }))?,
            )?;
            let draw_id = payload.domain_separated_id()?.sha256_hex().to_owned();
            let decision_ids = joins.get(&draw_id).cloned().unwrap_or_default();
            ensure!(
                decision_ids.windows(2).all(|w| w[0] < w[1])
                    && decision_ids.iter().all(|s| super::super::is_sha256_hex(s)),
                "decision links must be sorted unique IDs"
            );
            used.insert(draw_id.clone());
            draws.push(ProposalDrawLink {
                draw_id, draw_key: key, outcome: draw.outcome, tokens_sha256: draw.tokens_sha256,
                program_sha256: draw.program_sha256, slot_record_sha256: draw.slot_record_sha256,
                missing_decision_reason: decision_ids.is_empty().then(|| "no downstream decision delivered; original shadow/outcome retained without fitting authority".into()),
                decision_ids,
            });
        }
        ensure!(
            joins.keys().all(|key| used.contains(key)),
            "orphan proposal draw/decision link"
        );
        Ok(ProposalDecisionLinks {
            schema: DECISION_LINKS_SCHEMA.into(),
            ledger_sha256: outcomes.ledger_sha256,
            denominators: outcomes.denominators,
            draws,
        })
    }

    /// Every draw that did not become a handed-off candidate, without fitting authority.
    pub fn shadow(&self) -> Vec<ShadowRecord<'_>> {
        let neural = self
            .neural_draws
            .iter()
            .filter(|d| d.outcome != DrawOutcome::Completed)
            .map(|d| ShadowRecord {
                stage: "neural",
                outcome: format!("{:?}", d.outcome),
                tokens: &d.tokens,
                program_sha256: d.program_sha256.as_deref(),
                fitting_authority: false,
            });
        fn uniform<'a>(
            stage: &'static str,
            draws: &'a [UniformDraw],
        ) -> impl Iterator<Item = ShadowRecord<'a>> + 'a {
            draws
                .iter()
                .filter(|d| d.outcome != UniformOutcome::Admitted)
                .map(move |d| ShadowRecord {
                    stage,
                    outcome: format!("{:?}", d.outcome),
                    tokens: &d.tokens,
                    program_sha256: d.program_sha256.as_deref(),
                    fitting_authority: false,
                })
        }
        neural
            .chain(uniform("fallback", &self.fallback_draws))
            .chain(uniform("exploration", &self.exploration_draws))
            .collect()
    }
}
