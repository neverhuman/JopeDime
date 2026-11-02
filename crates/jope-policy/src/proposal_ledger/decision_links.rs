use super::*;

/// A companion references the existing ledger, including every shadow/refused draw. It adds
/// downstream decision joins without changing the bytes or denominators of proposal-ledger/v1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalDecisionLinks {
    pub schema: String,
    pub ledger_sha256: String,
    pub denominators: Denominators,
    pub draws: Vec<ProposalDrawLink>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalDrawLink {
    pub draw_id: String,
    pub draw_key: String,
    #[serde(rename = "draw_disposition")]
    pub outcome: String,
    pub tokens_sha256: String,
    pub program_sha256: Option<String>,
    pub slot_record_sha256: Option<String>,
    pub decision_ids: Vec<String>,
    /// No delivery/decision is not permission to fit a shadow proposal.
    pub missing_decision_reason: Option<String>,
}

impl ProposalDecisionLinks {
    /// Serialize this companion using the canonical JSON codec.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        Ok(
            event_wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(self)?)?
                .as_bytes()
                .to_vec(),
        )
    }

    /// Verify that these decision joins exactly match the companion derived from `ledger`.
    pub fn verify(&self, ledger: &ProposalLedger) -> Result<()> {
        let mut joins = BTreeMap::new();
        for draw in &self.draws {
            ensure!(
                joins
                    .insert(draw.draw_id.clone(), draw.decision_ids.clone())
                    .is_none(),
                "duplicate proposal draw link"
            );
        }
        ensure!(
            *self == ledger.decision_links(&joins)?,
            "altered proposal decision companion"
        );
        Ok(())
    }
}

#[cfg(test)]
mod decision_link_tests {
    use super::*;

    #[test]
    fn companion_retains_every_draw_without_changing_legacy_ledger() -> Result<()> {
        let mut config = crate::config::PolicyConfig::regression(401);
        config.width = 16;
        config.heads = 4;
        config.ffn = 32;
        config.layers = 1;
        config.dropout = 0.;
        config.max_seq = 128;
        let model = crate::model::PolicyModel::random_weights(config.clone(), 5)?;
        let checkpoint = crate::config::CheckpointBinding::unbound_for_testing(
            config,
            crate::CHECKPOINT_VOCABULARY_SHA256,
        );
        let vocabulary = crate::grammar::bind_vocabulary(
            include_bytes!("../../../jope-engine/assets/grammar/featmath.json"),
            crate::CHECKPOINT_VOCABULARY_SHA256,
        )?;
        let context = crate::context::PolicyContext::from_frozen_values(vec![0.; 2048])?;
        let hash = "a".repeat(64);
        let context_hash = crate::context_sha256(&context);
        let binding = crate::seeds::SeedBinding::new(&hash, 0, &hash, &context_hash)?;
        let proposal = crate::seeds::propose_unbound(
            &model,
            &checkpoint,
            &vocabulary,
            &context,
            &binding,
            &crate::seeds::ProposalSeeds::derive(7),
            &crate::decode::SamplingOptions {
                rounds: 1,
                batch: 8,
                temperature: 1.,
            },
        )?;
        let ledger = proposal.ledger();
        let before = serde_json::to_vec(ledger)?;
        let companion = ledger.decision_links(&BTreeMap::new())?;
        companion.verify(ledger)?;
        assert_eq!(
            serde_json::from_slice::<ProposalDecisionLinks>(&companion.canonical_bytes()?)?,
            companion
        );
        assert_eq!(companion.draws.len(), ledger.draw_outcomes()?.draws.len());
        assert_eq!(companion.denominators, ledger.denominators);
        assert!(companion
            .draws
            .iter()
            .any(|d| d.slot_record_sha256.is_none()));
        assert!(companion
            .draws
            .iter()
            .all(|d| d.missing_decision_reason.is_some()));
        let joins = [(companion.draws[0].draw_id.clone(), vec!["b".repeat(64)])].into();
        let linked = ledger.decision_links(&joins)?;
        linked.verify(ledger)?;
        assert_eq!(linked.draws[0].decision_ids.len(), 1);
        let mut missing = linked.clone();
        missing.draws.pop();
        assert!(missing.verify(ledger).is_err());
        let mut changed = linked;
        changed.draws[0].outcome = "invented".into();
        assert!(changed.verify(ledger).is_err());
        assert!(ledger
            .decision_links(&[("c".repeat(64), vec!["d".repeat(64)])].into())
            .is_err());
        assert_eq!(serde_json::to_vec(ledger)?, before);
        Ok(())
    }
}
