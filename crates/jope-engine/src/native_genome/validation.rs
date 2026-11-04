use super::*;

impl GenomeTask {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.native_genome_task.v1"
                && self.evaluations > 0
                && self.evaluations <= 1000
                && self.seconds > 0
                && self.seconds <= 600,
            "native genome task budget/schema invalid"
        );
        crate::manifest::relative(&self.output)?;
        ensure!(
            match self.stage {
                GenomeEvaluationStage::Direct => {
                    self.evaluations == 1 && self.direct_parent.is_none()
                }
                GenomeEvaluationStage::Continuation => self.direct_parent.is_some(),
            },
            "native genome direct/continuation contract mismatch"
        );
        Ok(())
    }
    pub(crate) fn candidate(&self, root: &Path) -> Result<Candidate> {
        self.validate()?;
        let c: Candidate = load(root, &self.candidate)?;
        ensure!(
            c.schema == "jope.native_candidate.v1"
                && !c.run.is_empty()
                && c.context.partition == "meta_train"
                && hash(c.genome.as_bytes()) == c.genome_sha256
                && feat_math::GenomeIr::parse(&c.genome)?.canonical() == c.genome,
            "candidate identity, canonical program or sealed partition differs"
        );
        let prepared = resolve(root, &c.context.prepared)?;
        // Route R8: the boundary is decided on exactly the bound preparation bytes.
        let p: crate::prepare::Prepared = serde_json::from_slice(&read_verified(
            &prepared.join("prepared.json"),
            &c.context.prepared_sha256,
            "candidate preparation",
        )?)?;
        ensure!(
            p.task == "regression"
                && p.meta_partition == "meta_train"
                && p.family == c.context.family
                && p.dataset == c.context.dataset
                && (p.family_partition_frozen || self.preparation_overlay.is_some())
                && (p.clean_evaluation || (self.qualification && p.qualification_only)),
            "candidate scientific boundary changed"
        );
        if let Some(overlay) = &self.preparation_overlay {
            ensure!(
                !self.qualification,
                "scoped overlay cannot relabel qualification material"
            );
            crate::preparation_overlay::verify(root, overlay, &c, &p)?;
        }
        let v: crate::grammar::Vocab = load(root, &self.vocabulary)?;
        ensure!(
            v.encode(&c.genome)?.len() <= crate::corpus::SEQUENCE,
            "candidate cannot retain exact frozen tokens"
        );
        let proposal: crate::native_graph::VerifiedResult = load(root, &c.proposal)?;
        ensure!(
            proposal.schema == "jope.native_graph_result.v1"
                && proposal.run == c.run
                && proposal.generation == c.generation,
            "candidate original proposal scope differs"
        );
        let capture: Value = load(root, &c.origin_capture)?;
        ensure!(
            capture["schema"] == "jope.all_valid_neural.v1"
                && capture["final_tests_opened"] == false,
            "candidate source capture differs"
        );
        let row = capture["rows"]
            .as_array()
            .and_then(|rows| rows.get(c.origin_row))
            .context("candidate source occurrence absent")?;
        ensure!(
            row["program_sha256"]
                == hash(row["genome"].as_str().context("source genome")?.as_bytes()),
            "source genome digest differs"
        );
        if c.arm == Arm::Generic {
            let draw = c.generic_draw.context("generic source draw absent")?;
            ensure!(
                c.sampling_seed
                    == crate::native_loop::generation_seed(
                        &c.run,
                        c.generation,
                        &format!(
                            "generic:{}:{}:{}",
                            c.context.prepared_sha256, c.context.cut, c.matched_slot
                        )
                    )?,
                "generic RNG is not generation-derived"
            );
            ensure!(
                draw < 10000 && capture["arm"] == "model",
                "generic comparison origin differs"
            );
            let raw =
                feat_math::evolve::random_canonical_pipeline(c.sampling_seed.wrapping_add(draw), 6);
            ensure!(
                v.project_proposal(&raw)? == c.genome,
                "generic comparison RNG/program differs"
            );
        } else {
            ensure!(
                c.generic_draw.is_none()
                    && row["genome"] == c.genome
                    && capture["sampling_seed"] == c.sampling_seed
                    && capture["arm"] == serde_json::to_value(c.arm)?,
                "model/shuffled original occurrence differs"
            );
        }
        let producer: Value = load(root, &proposal.output)?;
        let original: BTreeMap<PathBuf, String> =
            serde_json::from_value(producer["artifacts"].clone())?;
        ensure!(
            original
                .iter()
                .any(|(path, sha)| c.origin_capture.path.ends_with(path)
                    && sha == &c.origin_capture.sha256),
            "candidate capture absent from original proposal output closure"
        );
        let _: Value = load(root, &c.parent_checkpoint)?;
        Ok(c)
    }
}
