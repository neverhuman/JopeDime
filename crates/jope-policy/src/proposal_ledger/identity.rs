use super::*;

impl ProposalLedger {
    /// sha256 of this ledger's canonical JSON.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Identity of one handed-off candidate's proposal record: what a GP lineage envelope's
    /// `seed_population_slot` source joins to.
    pub fn slot_record_sha256(&self, slot: usize) -> Result<String> {
        let record = self
            .candidates
            .get(slot)
            .context("slot outside the handed-off candidates")?;
        Ok(sha256_hex(&serde_json::to_vec(record)?))
    }
    /// Canonical recipe identities in slot order.
    pub fn canonical_recipes(&self) -> Vec<&str> {
        self.candidates
            .iter()
            .map(|c| c.canonical_recipe_sha256.as_str())
            .collect()
    }

    /// The candidate records keyed by canonical recipe: canonical duplicates among handed-off
    /// candidates are visible rather than counted as independent proposals.
    pub fn canonical_groups(&self) -> BTreeMap<&str, Vec<usize>> {
        let mut groups = BTreeMap::<&str, Vec<usize>>::new();
        for c in &self.candidates {
            groups
                .entry(c.canonical_recipe_sha256.as_str())
                .or_default()
                .push(c.slot);
        }
        groups
    }
}
