//! Adapter for callers that already hold an immutable verified evidence snapshot.
use jope_records::research::{EvidenceLocator, EvidenceResolver, EvidenceValue};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default)]
/// `EvidenceView` is a record for the evidence view contract in this module: Adapter for callers that already hold an immutable verified evidence snapshot.
pub struct EvidenceView {
    values: BTreeMap<EvidenceLocator, EvidenceValue>,
    revoked: BTreeSet<EvidenceLocator>,
}

impl EvidenceView {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        values: impl IntoIterator<Item = EvidenceValue>,
        revoked: impl IntoIterator<Item = EvidenceLocator>,
    ) -> Self {
        Self {
            values: values.into_iter().map(|v| (v.locator.clone(), v)).collect(),
            revoked: revoked.into_iter().collect(),
        }
    }
}

impl EvidenceResolver for EvidenceView {
    type Error = std::convert::Infallible;
    fn is_revoked(&self, locator: &EvidenceLocator) -> Result<bool, Self::Error> {
        Ok(self.revoked.contains(locator))
    }
    fn resolve(&self, locator: &EvidenceLocator) -> Result<Option<EvidenceValue>, Self::Error> {
        Ok(self.values.get(locator).cloned())
    }
}
