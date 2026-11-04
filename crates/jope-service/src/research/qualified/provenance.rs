use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperatorOrigin {
    Seed,
    Proposal(ProposalFamily),
    Native(&'static str),
}

impl OperatorOrigin {
    pub(super) fn family(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::Proposal(family) => family.name(),
            Self::Native(family) => family,
        }
    }
}

/// Supplied by the admitted proposer/seed path, with its roster and Plan pins
/// checked before capture. No implementation is installed by default.
pub(crate) trait OperatorProvenance: Send + Sync {
    fn validate_plan(&self, plan: &Plan) -> Result<()>;
    fn observe_round(&self, _round: u64, _dsls: &[String]) -> Result<()> {
        Ok(())
    }
    fn origin(&self, round: u64, dsl_sha256: &str) -> Result<OperatorOrigin>;
}

/// Hash-bound provenance collected directly from the frozen seed roster and
/// proposer `Proposal` objects. Missing or duplicate origins fail closed.
pub(crate) struct AdmittedOperatorProvenance {
    plan_sha256: String,
    origins: BTreeMap<(u64, String), OperatorOrigin>,
}

impl AdmittedOperatorProvenance {
    pub(crate) fn from_seed_roster(plan: &Plan, seeds: &[String]) -> Result<Self> {
        ensure!(!seeds.is_empty(), "admitted seed roster is empty");
        let mut origins = BTreeMap::new();
        for dsl in seeds {
            ensure!(!dsl.is_empty(), "empty seed program");
            let sha = digest(dsl.as_bytes());
            ensure!(
                origins.insert((0, sha), OperatorOrigin::Seed).is_none(),
                "duplicate seed program"
            );
        }
        Ok(Self {
            plan_sha256: plan_sha256(plan)?,
            origins,
        })
    }

    pub(crate) fn record_proposals(
        &mut self,
        plan: &Plan,
        round: u64,
        proposals: &[Proposal],
    ) -> Result<()> {
        ensure!(round > 0, "proposal provenance requires a positive round");
        ensure!(
            self.plan_sha256 == plan_sha256(plan)?,
            "operator provenance Plan changed"
        );
        let mut pending = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            ensure!(
                proposal.program.program_sha256 == digest(proposal.program.dsl.as_bytes()),
                "proposer program digest mismatch"
            );
            pending.push((
                (round, proposal.program.program_sha256.clone()),
                OperatorOrigin::Proposal(proposal.family),
            ));
        }
        ensure!(
            pending
                .iter()
                .map(|(key, _)| key)
                .collect::<BTreeSet<_>>()
                .len()
                == pending.len(),
            "duplicate proposal provenance"
        );
        ensure!(
            pending
                .iter()
                .all(|(key, _)| !self.origins.contains_key(key)),
            "proposal provenance already recorded"
        );
        self.origins.extend(pending);
        Ok(())
    }
}

impl OperatorProvenance for AdmittedOperatorProvenance {
    fn validate_plan(&self, plan: &Plan) -> Result<()> {
        ensure!(
            self.plan_sha256 == plan_sha256(plan)?,
            "operator provenance Plan differs"
        );
        Ok(())
    }

    fn origin(&self, round: u64, dsl_sha256: &str) -> Result<OperatorOrigin> {
        self.origins
            .get(&(round, dsl_sha256.to_owned()))
            .copied()
            .context("program has no admitted seed/proposer origin")
    }
}
