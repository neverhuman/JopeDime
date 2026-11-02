use super::*;

impl BoundedRunContract {
    /// Bounded parse. A legacy fixed manifest is reported as legacy, never coerced.
    pub fn parse(bytes: &[u8]) -> std::result::Result<Self, PlanRefusal> {
        need!(
            bytes.len() <= MAX_CONTRACT_BYTES,
            Invalid,
            "contract exceeds {MAX_CONTRACT_BYTES} bytes"
        );
        let value: Value = serde_json::from_slice(bytes).map_err(invalid)?;
        if value["schema"] == LEGACY_CONTRACT_SCHEMA {
            return Err(PlanRefusal::Legacy(format!(
                "`{LEGACY_CONTRACT_SCHEMA}` runs only through Bind/Tick/Serve with its original semantics"
            )));
        }
        let contract: Self = serde_json::from_value(value).map_err(invalid)?;
        contract.validate_structure()?;
        Ok(contract)
    }
    pub fn canonical_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }
    pub fn world_size(&self) -> usize {
        self.participants.len()
    }
    pub fn participant(&self, global_rank: u32) -> Option<&Participant> {
        self.participants.get(global_rank as usize)
    }
}
