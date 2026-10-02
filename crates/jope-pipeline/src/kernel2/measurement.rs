use super::*;
use crate::search::{FIT_FAILED, MEASURED, PANICKED, PARSE_FAILED, SCORE_FAILED};
use feat_math::{
    kernel::{
        admission::prefer_fewer_faults,
        search::admit_child,
        telemetry::{predict_traced, ZeroWidthInput},
        width::{check_input_width, WidthMismatch},
    },
    metrics::{r2, rmse},
};

impl Kernel2Evaluator {
    /// Measure one attempted candidate. Use `measure` for a budgeted search.
    /// The fit pass and held-out prediction pass are pooled exactly once.
    pub fn evaluate(
        &self,
        genome: &str,
        cut: &Cut,
        generation: usize,
        origin: &'static str,
        clock: Clock,
    ) -> Kernel2Observation {
        let started = clock.start();
        let mut record = empty_record(genome, cut.index, generation, origin);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.traced_measure(genome, cut, &mut record)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                if error.downcast_ref::<ZeroWidthInput>().is_some()
                    || error.downcast_ref::<WidthMismatch>().is_some()
                {
                    record.observation.status = INADMISSIBLE;
                }
                record.observation.failure = Some(format!("{error:#}"));
            }
            Err(_) => {
                record.observation.status = PANICKED;
                record.observation.failure = Some("candidate evaluation panicked".into());
            }
        }
        record.refresh_faults();
        record.observation.seconds = started.elapsed().as_secs_f64();
        record
    }

    /// The allowance is charged before admission, including a refused candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn measure(
        &self,
        genome: &str,
        cut: &Cut,
        generation: usize,
        origin: &'static str,
        clock: Clock,
        budget: &mut EvaluationBudget,
    ) -> Option<Kernel2Observation> {
        budget
            .charge(1)
            .then(|| self.evaluate(genome, cut, generation, origin, clock))
    }

    fn traced_measure(&self, text: &str, cut: &Cut, record: &mut Kernel2Observation) -> Result<()> {
        record.observation.status = PARSE_FAILED;
        GenomeIr::parse(text)?;
        record.observation.status = INADMISSIBLE;
        let width = rectangular_width(&cut.train_x)?;
        rectangular_width(&cut.hold_x)?;
        let genome = admit_child(text, &self.profile, width)?;
        if let Some(error) =
            ZeroWidthInput::find(&cut.train_x).or_else(|| ZeroWidthInput::find(&cut.hold_x))
        {
            return Err(error.into());
        }
        record.observation.status = FIT_FAILED;
        ensure!(
            cut.train_x.len() == cut.train_y.len(),
            "unaligned training rows and targets"
        );
        let model = super::fitting::fit_observed(&genome, &cut.train_x, &cut.train_y, record)?;
        reject_hard(record)?;
        check_input_width(&model, width)?;
        let train = model.predict(&cut.train_x)?;
        if let Some(row) = cut.hold_x.first() {
            check_input_width(&model, row.len())?;
        }
        let (hold, held_out) = predict_traced(&model, &cut.hold_x)?;
        record.faults.add_pass(&held_out);
        reject_hard(record)?;
        record.observation.status = SCORE_FAILED;
        ensure!(
            train.len() == cut.train_y.len() && hold.len() == cut.hold_y.len(),
            "prediction and target lengths differ"
        );
        let (train_score, hold_score, error) = (
            r2(&cut.train_y, &train),
            r2(&cut.hold_y, &hold),
            rmse(&cut.hold_y, &hold),
        );
        ensure!(
            train_score.is_finite() && hold_score.is_finite() && error.is_finite(),
            "nonfinite candidate scores"
        );
        let selection_score = match self.fault_policy {
            FaultPolicy::Penalize {
                lambda_domain,
                lambda_saturation,
                ..
            } => record
                .faults
                .penalize(hold_score, lambda_domain, lambda_saturation),
            FaultPolicy::RejectHard => Some(hold_score),
            _ => unreachable!("fault policy validation invariant violated"),
        };
        ensure!(selection_score.is_some(), "nonfinite penalized score");
        record.observation.train_r2 = Some(train_score);
        record.observation.hold_r2 = Some(hold_score);
        record.observation.rmse = Some(error);
        record.selection_score = selection_score;
        record.observation.status = MEASURED;
        Ok(())
    }

    /// Stable ranking uses anchored WP1.5 runs; near ties are not a comparator.
    pub fn rank<'a>(&self, observations: &'a [Kernel2Observation]) -> Vec<&'a Kernel2Observation> {
        let mut ranked: Vec<_> = observations
            .iter()
            .filter(|o| {
                o.observation.measured()
                    && o.selection_score.is_some_and(f64::is_finite)
                    && !o.faults.is_hard()
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.selection_score
                .unwrap()
                .total_cmp(&a.selection_score.unwrap())
                .then(a.observation.genome.cmp(&b.observation.genome))
        });
        if matches!(
            self.fault_policy,
            FaultPolicy::Penalize {
                tie_break: true,
                ..
            }
        ) {
            prefer_fewer_faults(
                &mut ranked,
                |o| o.selection_score.unwrap(),
                |o| o.faults.faulted_cells(),
            );
        }
        ranked
    }
}

pub(super) fn reject_hard(record: &mut Kernel2Observation) -> Result<()> {
    if record.faults.is_hard() {
        record.observation.status = INADMISSIBLE;
        anyhow::bail!("hard fault: {}", record.faults.hard_classes);
    }
    Ok(())
}

fn empty_record(
    genome: &str,
    cut: usize,
    generation: usize,
    origin: &'static str,
) -> Kernel2Observation {
    Kernel2Observation {
        observation: Observation {
            genome: genome.into(),
            program_sha256: sha256_hex(genome.as_bytes()),
            cut,
            generation,
            origin,
            train_r2: None,
            hold_r2: None,
            rmse: None,
            seconds: 0.0,
            status: FIT_FAILED,
            failure: None,
            island: None,
        },
        domain_fault_rate: 0.0,
        saturation_fault_rate: 0.0,
        constant_output: false,
        faulted_cells: 0,
        selection_score: None,
        faults: FaultTotals::default(),
    }
}
