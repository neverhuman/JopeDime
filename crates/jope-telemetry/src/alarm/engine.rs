use super::*;

impl Engine {
    pub fn new(run_id: impl Into<String>, th: Thresholds) -> Self {
        Self {
            run_id: run_id.into(),
            th,
            collapse_streak: HashMap::new(),
            grad_window: VecDeque::new(),
            grad_spike_streak: 0,
            val_hist: Vec::new(),
            elo_hist: HashMap::new(),
            step_secs: VecDeque::new(),
            last_step_at: None,
            notifier: None,
        }
    }

    pub fn with_notifier(mut self, n: CommsNotifier) -> Self {
        self.notifier = Some(n);
        self
    }

    fn fire(&mut self, id: AlarmId, at: u64, reasons: Vec<String>, now_secs: f64) -> Alarm {
        let a = Alarm {
            schema: ALARM_SCHEMA.into(),
            id,
            code: id.code().into(),
            severity: id.severity(),
            run_id: self.run_id.clone(),
            at,
            reasons,
        };
        eprintln!(
            "JTEL-ALARM {} {:?} run={} at={} {}",
            a.code,
            a.severity,
            a.run_id,
            a.at,
            a.reasons.join("; ")
        );
        if let Some(n) = &mut self.notifier {
            n.notify(&a, now_secs);
        }
        a
    }

    /// A1: any freeze invariant broken. `FailClosed`; the caller raises.
    pub fn observe_dope_guard(&mut self, o: &DopeGuardObservation, now: f64) -> Option<Alarm> {
        let mut r = Vec::new();
        if o.hashes_equal == Some(false) {
            r.push(format!(
                "DOPE hash changed: {}",
                o.changed_tensors.join(",")
            ));
        }
        if o.trainable_dope_params > 0 {
            r.push(format!(
                "{} DOPE params have requires_grad=True",
                o.trainable_dope_params
            ));
        }
        if o.dope_params_in_optimizer > 0 {
            r.push(format!(
                "{} DOPE params in optimizer groups",
                o.dope_params_in_optimizer
            ));
        }
        if o.dope_grads_present > 0 {
            r.push(format!("{} DOPE params hold a .grad", o.dope_grads_present));
        }
        if o.dope_training_mode {
            r.push("dope_model.training is True".into());
        }
        if o.trainable_jope_params == Some(0) {
            // Not a freeze violation, but nothing is learning: report under A1 for visibility.
            r.push("no trainable JopeDime params".into());
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A1DopeFreeze, o.epoch, r, now))
    }

    /// A2: a collapse condition held for N consecutive generations of one context.
    pub fn observe_generation(&mut self, g: &GenSummary, now: f64) -> Option<Alarm> {
        let th = &self.th;
        let mut r = Vec::new();
        let lt = |v: Option<f64>, t: f64| v.is_some_and(|v| v < t);
        if lt(g.unique_genotype_frac, th.min_unique_genotype_frac) {
            r.push(format!(
                "unique_genotype_frac={:.3}",
                g.unique_genotype_frac.unwrap()
            ));
        }
        if lt(
            g.behavior.fitness_profile_unique_frac,
            th.min_behavior_unique_frac,
        ) {
            r.push(format!(
                "fitness_profile_unique_frac={:.3}",
                g.behavior.fitness_profile_unique_frac.unwrap()
            ));
        }
        if lt(g.op_entropy_norm, th.min_op_entropy_norm) {
            r.push(format!("op_entropy_norm={:.3}", g.op_entropy_norm.unwrap()));
        }
        if g.dedup_reject_frac
            .is_some_and(|v| v > th.max_dedup_reject_frac)
        {
            r.push(format!(
                "dedup_reject_frac={:.3}",
                g.dedup_reject_frac.unwrap()
            ));
        }
        if g.fitness_passers.n >= 4 && lt(g.fitness_passers.iqr, th.min_passer_fitness_iqr) {
            r.push(format!(
                "passer_fitness_iqr={:.2e}",
                g.fitness_passers.iqr.unwrap()
            ));
        }
        if g.provenance_passers.n > 0 && lt(g.provenance_passers.novel, th.min_novel_passer_frac) {
            r.push(format!(
                "novel_passer_frac={:.3}",
                g.provenance_passers.novel.unwrap()
            ));
        }
        let streak = self.collapse_streak.entry(g.context.clone()).or_default();
        if r.is_empty() {
            *streak = 0;
            return None;
        }
        *streak += 1;
        if *streak < th.collapse_consecutive_gens {
            return None;
        }
        let s = *streak;
        r.push(format!("{s} consecutive generations"));
        Some(self.fire(AlarmId::A2GpCollapse, g.generation as u64, r, now))
    }

    /// A3 (step part) and A10 bookkeeping. `wall_secs` is the step's completion time.
    pub fn observe_step(
        &mut self,
        step: u64,
        loss: f64,
        grad_norm: Option<f64>,
        wall_secs: f64,
    ) -> Option<Alarm> {
        if let Some(prev) = self.last_step_at.replace(wall_secs) {
            if self.step_secs.len() == 256 {
                self.step_secs.pop_front();
            }
            self.step_secs.push_back(wall_secs - prev);
        }
        let mut r = Vec::new();
        if !loss.is_finite() {
            r.push(format!("non-finite loss {loss}"));
        }
        if let Some(g) = grad_norm {
            if !g.is_finite() {
                r.push(format!("non-finite grad norm {g}"));
            } else {
                let spiking = self.grad_window.len() >= self.th.grad_spike_window / 2
                    && median(self.grad_window.iter().copied())
                        .is_some_and(|m| m > 0.0 && g > self.th.grad_spike_factor * m);
                self.grad_spike_streak = if spiking {
                    self.grad_spike_streak + 1
                } else {
                    0
                };
                if self.grad_spike_streak >= self.th.grad_spike_consecutive {
                    r.push(format!(
                        "grad norm {g:.3e} > {}x running median for {} steps",
                        self.th.grad_spike_factor, self.grad_spike_streak
                    ));
                }
                if !spiking {
                    if self.grad_window.len() == self.th.grad_spike_window {
                        self.grad_window.pop_front();
                    }
                    self.grad_window.push_back(g);
                }
            }
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A3Divergence, step, r, wall_secs))
    }

    /// A3 (epoch part): validation loss rising for N epochs while training loss falls.
    pub fn observe_epoch(
        &mut self,
        epoch: u64,
        train_loss: f64,
        val_loss: f64,
        now: f64,
    ) -> Option<Alarm> {
        self.val_hist.push((train_loss, val_loss));
        let n = self.th.val_rise_epochs as usize;
        if self.val_hist.len() <= n {
            return None;
        }
        let w = &self.val_hist[self.val_hist.len() - n - 1..];
        let val_rising = w.windows(2).all(|p| p[1].1 > p[0].1);
        let train_falling = w.windows(2).all(|p| p[1].0 < p[0].0);
        (val_rising && train_falling).then(|| {
            self.fire(
                AlarmId::A3Divergence,
                epoch,
                vec![format!(
                    "val loss rising {n} epochs while train loss falls (overfit)"
                )],
                now,
            )
        })
    }

    /// A4: the fused model picks the same winner everywhere, or predicts a constant.
    pub fn observe_selection(&mut self, o: &SelectionObservation, now: f64) -> Option<Alarm> {
        let th = &self.th;
        let mut r = Vec::new();
        if o.top1_share
            .is_some_and(|v| v > th.max_selection_top1_share)
        {
            r.push(format!("top1_share={:.3}", o.top1_share.unwrap()));
        }
        if o.unique_frac
            .is_some_and(|v| v < th.min_selection_unique_frac)
        {
            r.push(format!(
                "selection_unique_frac={:.3}",
                o.unique_frac.unwrap()
            ));
        }
        if o.across_dataset_prediction_spread
            .is_some_and(|v| v < th.min_prediction_spread)
        {
            r.push(format!(
                "prediction_spread={:.2e}",
                o.across_dataset_prediction_spread.unwrap()
            ));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A4SelectionCollapse, o.epoch, r, now))
    }

    /// A5: Elo of one task type down for N consecutive rounds within one `elo_fit_id` series.
    /// `series` names the fit series and task; values from different fits never share a key.
    pub fn observe_elo(
        &mut self,
        series: &str,
        round: u64,
        elo: f64,
        below_baseline: bool,
        now: f64,
    ) -> Option<Alarm> {
        let h = self.elo_hist.entry(series.to_owned()).or_default();
        h.push(elo);
        let n = self.th.elo_drop_rounds as usize;
        let mut r = Vec::new();
        if h.len() > n && h[h.len() - n - 1..].windows(2).all(|p| p[1] < p[0]) {
            r.push(format!("{series}: Elo down {n} consecutive rounds"));
        }
        if below_baseline && round >= 3 {
            r.push(format!("{series}: below H baseline at round {round}"));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A5EloRegression, round, r, now))
    }

    /// A6: the optimizer's LR differs from the schedule the config declares.
    pub fn observe_lr(&mut self, step: u64, lr: f64, expected: f64, now: f64) -> Option<Alarm> {
        let rel = (lr - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
        (rel > self.th.lr_rel_tol).then(|| {
            self.fire(
                AlarmId::A6ScheduleDrift,
                step,
                vec![format!("lr={lr:.6e} expected={expected:.6e} rel={rel:.2e}")],
                now,
            )
        })
    }

    /// A7: evaluator timeouts, failures, or non-reproducible fitness.
    pub fn observe_eval_batch(&mut self, o: &EvalBatchObservation, now: f64) -> Option<Alarm> {
        if o.n == 0 {
            return None;
        }
        let mut r = Vec::new();
        let tr = o.timeouts as f64 / o.n as f64;
        let fr = o.failures as f64 / o.n as f64;
        if tr > self.th.max_eval_timeout_rate {
            r.push(format!("timeout_rate={tr:.3}"));
        }
        if fr > self.th.max_eval_fail_rate {
            r.push(format!("fail_rate={fr:.3}"));
        }
        if o.repro_mismatches > 0 {
            r.push(format!(
                "{} fitness re-evaluations differed",
                o.repro_mismatches
            ));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A7EvaluatorHealth, o.batch, r, now))
    }

    /// A8: any evaluation touched the sealed partition. `FailClosed`.
    pub fn observe_sealed_touch(&mut self, at: u64, count: u64, now: f64) -> Option<Alarm> {
        (count > 0).then(|| {
            self.fire(
                AlarmId::A8SealedTouch,
                at,
                vec![format!("{count} sealed-partition evaluations")],
                now,
            )
        })
    }

    /// A9: a flywheel target other than Rust was emitted. `FailClosed` for emission.
    pub fn observe_rust_only(&mut self, at: u64, violations: u64, now: f64) -> Option<Alarm> {
        (violations > 0).then(|| {
            self.fire(
                AlarmId::A9RustOnlyTarget,
                at,
                vec![format!("{violations} non-Rust targets")],
                now,
            )
        })
    }

    /// A10: called by the drain on its 1 s tick while the producer process is alive.
    pub fn heartbeat(&mut self, now: f64) -> Option<Alarm> {
        let last = self.last_step_at?;
        let med = median(self.step_secs.iter().copied()).unwrap_or(0.0);
        let limit = (self.th.heartbeat_median_factor * med).max(self.th.heartbeat_min_secs);
        let idle = now - last;
        (idle > limit).then(|| {
            self.fire(
                AlarmId::A10TrainerHeartbeat,
                0,
                vec![format!("no step for {idle:.0}s (limit {limit:.0}s)")],
                now,
            )
        })
    }
}
