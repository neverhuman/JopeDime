use super::*;

pub(crate) struct NativeRoundInputs {
    pub(crate) inputs: NativeCaptureInputs,
    pub(crate) conditioning: TokenProjection,
}
impl NativeRoundInputs {
    fn program(&self, dsl: &str) -> Result<RoundProgram> {
        let p = self.inputs.program(dsl, "native_input")?;
        Ok(RoundProgram {
            dsl: dsl.into(),
            features412: p
                .feat412
                .try_into()
                .map_err(|_| anyhow::anyhow!("feature width"))?,
            projection16: p
                .proj16
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("projection width"))?,
        })
    }
}
impl RoundInputProvider for NativeRoundInputs {
    fn observe(&self, context: &RoundContext<'_>, proposals: &[String]) -> Result<RoundInputs> {
        Ok(RoundInputs {
            pool: context
                .measured
                .iter()
                .map(|m| self.program(m.dsl))
                .collect::<Result<_>>()?,
            pool_gates: context
                .measured
                .iter()
                .map(|m| m.surrogate_fitness < 100.0)
                .collect(),
            proposals: proposals
                .iter()
                .map(|dsl| self.program(dsl))
                .collect::<Result<_>>()?,
            conditioning: self.conditioning.clone(),
        })
    }
}

pub(crate) struct LiveProvenance {
    plan_sha256: String,
    origins: Mutex<BTreeMap<(u64, String), OperatorOrigin>>,
    pending: Mutex<BTreeMap<String, OperatorOrigin>>,
}
impl LiveProvenance {
    pub(crate) fn new(plan: &Plan, seeds: &BTreeSet<String>) -> Result<Self> {
        Ok(Self {
            plan_sha256: digest(&serde_json::to_vec(plan)?),
            origins: Mutex::new(
                seeds
                    .iter()
                    .map(|dsl| ((0, digest(dsl.as_bytes())), OperatorOrigin::Seed))
                    .collect(),
            ),
            pending: Mutex::new(BTreeMap::new()),
        })
    }
    fn generated(&self, dsl: &str, arm: &'static str) {
        self.pending
            .lock()
            .expect("provenance mutex poisoned")
            .entry(digest(dsl.as_bytes()))
            .or_insert(OperatorOrigin::Native(arm));
    }
}
impl OperatorProvenance for LiveProvenance {
    fn validate_plan(&self, plan: &Plan) -> Result<()> {
        ensure!(
            self.plan_sha256 == digest(&serde_json::to_vec(plan)?),
            "provenance Plan changed"
        );
        Ok(())
    }
    fn observe_round(&self, round: u64, dsls: &[String]) -> Result<()> {
        if round == 0 {
            return Ok(());
        }
        let mut pending = self.pending.lock().expect("provenance mutex poisoned");
        let selected = dsls
            .iter()
            .map(|dsl| {
                let sha = digest(dsl.as_bytes());
                Ok((
                    (round, sha.clone()),
                    *pending
                        .get(&sha)
                        .context("program was not observed at the native generator")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.origins
            .lock()
            .expect("provenance mutex poisoned")
            .extend(selected);
        pending.clear();
        Ok(())
    }
    fn origin(&self, round: u64, sha: &str) -> Result<OperatorOrigin> {
        self.origins
            .lock()
            .expect("provenance mutex poisoned")
            .get(&(round, sha.into()))
            .copied()
            .context("unobserved native program origin")
    }
}
pub(crate) struct RecordingSpace {
    pub(crate) provenance: Arc<LiveProvenance>,
}
impl SearchSpace for RecordingSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        NativeSpace.canonical(dsl)
    }
    fn seeds(&self) -> Result<Vec<String>> {
        NativeSpace.seeds()
    }
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        NativeSpace.propose_observed(parents, count, rng, |dsl, arm| {
            self.provenance.generated(dsl, arm)
        })
    }
}
