//! Deterministic, budgeted migration admission for asynchronous islands.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MigrationProposal {
    pub source_island: u32,
    pub destination_island: u32,
    pub proposal_sequence: u64,
    pub genome_id: String,
    pub effort: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MigrationAdmission {
    pub admitted: Vec<MigrationProposal>,
    pub rejected: Vec<(MigrationProposal, &'static str)>,
    pub effort_used: u64,
    pub bytes_used: u64,
}

/// Makes async completion timing irrelevant: callers provide proposals and this function admits
/// them by declared sequence, then source and destination IDs, under both resource caps.
pub fn admit(
    mut proposals: Vec<MigrationProposal>,
    effort_cap: u64,
    byte_cap: u64,
) -> MigrationAdmission {
    proposals.sort_by(|a, b| {
        (
            a.proposal_sequence,
            a.source_island,
            a.destination_island,
            &a.genome_id,
        )
            .cmp(&(
                b.proposal_sequence,
                b.source_island,
                b.destination_island,
                &b.genome_id,
            ))
    });
    let mut result = MigrationAdmission {
        admitted: Vec::new(),
        rejected: Vec::new(),
        effort_used: 0,
        bytes_used: 0,
    };
    for proposal in proposals {
        let effort = result.effort_used.checked_add(proposal.effort);
        let bytes = result.bytes_used.checked_add(proposal.bytes);
        let reason = if effort.is_none_or(|n| n > effort_cap) {
            Some("effort cap")
        } else if bytes.is_none_or(|n| n > byte_cap) {
            Some("byte cap")
        } else {
            None
        };
        if let Some(reason) = reason {
            result.rejected.push((proposal, reason));
        } else {
            result.effort_used = effort.unwrap();
            result.bytes_used = bytes.unwrap();
            result.admitted.push(proposal);
        }
    }
    result
}
