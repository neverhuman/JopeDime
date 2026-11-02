use super::*;

/// Mutable state of the fit-based checks of one fragment.
struct MotifRun<'a> {
    table: &'a MotifTable,
    motif: &'a MotifDef,
    tables: &'a [Synthetic],
    determinism: CheckResult,
    context: CheckResult,
    not_exercised: u32,
    backend: Vec<BackendRow>,
    artifacts: Vec<Artifact>,
}

impl MotifRun<'_> {
    /// C2 on hosts: up to `limit` distinct hosts in provenance order, fitted on S2 and
    /// otherwise S3. A host refused on both counts as not exercised.
    fn hosts(&mut self, limit: usize) -> Result<()> {
        let table = self.table;
        let motif = self.motif;
        let tables = self.tables;
        let mut chosen: Vec<(&str, TaskLabel)> = Vec::new();
        for occurrence in &motif.provenance {
            if chosen.len() == limit {
                break;
            }
            let recipe = occurrence.recipe_sha256.as_str();
            if !chosen.iter().any(|(seen, _)| *seen == recipe) {
                chosen.push((recipe, occurrence.task));
            }
        }
        for (recipe, task) in chosen {
            let host = table
                .body
                .hosts
                .get(recipe)
                .ok_or(Code("provenance_inconsistent"))?;
            let ir = GenomeIr::parse(host).map_err(|_| Code("provenance_inconsistent"))?;
            legacy_artifacts(&ir, &mut self.artifacts);
            let mut exercised = false;
            for shape in [Shape::S2, Shape::S3] {
                match fit_twice(&ir, task, data(tables, shape)?) {
                    Outcome::Refused => {}
                    Outcome::Nondeterministic => {
                        self.determinism.exercised += 1;
                        self.determinism.failures.push(
                            Failure::new("fit_not_deterministic")
                                .recipe(recipe)
                                .context("host")
                                .table(shape)
                                .task(task),
                        );
                        exercised = true;
                        break;
                    }
                    Outcome::Fitted(fitted) => {
                        self.determinism.exercised += 1;
                        bound_artifacts(&fitted.model, &mut self.artifacts);
                        exercised = true;
                        break;
                    }
                }
            }
            if !exercised {
                self.not_exercised += 1;
            }
        }
        Ok(())
    }

    /// The context, as fitted or with the task's default head written out, equals a recorded
    /// host. Such a fit repeats a host: it is determinism and name-absence evidence, but not
    /// an independent context.
    fn repeats_a_host(&self, texts: &[&str]) -> bool {
        let hosts = &self.table.body.hosts;
        texts.iter().any(|text| {
            GenomeIr::parse(text).is_ok_and(|ir| {
                let canonical = ir.canonical();
                hosts.values().any(|host| *host == canonical)
            })
        })
    }

    /// C3 with C2 on every successful context fit. For every task where a host qualified:
    /// Q1 `(pipe X <input> E)` with the default head, Q2 with a leading `(standardize)`, and
    /// for regression the legacy-dialect Q3 `(pipe X E (ridge))`.
    ///
    /// For each task, Q1 must fit on S2 or S3 (`context_not_fitted`), and at least one
    /// context that does not repeat a recorded host must fit on S2 or S3
    /// (`context_not_independent`). Only fits of such independent contexts count as
    /// exercised. Everything else is recorded coverage.
    fn contexts(&mut self) -> Result<()> {
        let motif = self.motif;
        let tables = self.tables;
        let input = input_stage()?;
        let schema = classification_schema();
        for task in &motif.support.tasks {
            let task = *task;
            let spec = task.task_spec()?;
            let prefixes = [
                ("q1", input.clone()),
                ("q2", format!("{input} (standardize)")),
            ];
            let mut q1_fitted = false;
            let mut independent_fitted = false;
            for (label, prefix) in prefixes {
                let text = format!("(pipe X {prefix} {})", motif.expansion);
                let explicit = format!(
                    "(pipe X {prefix} {} ({}))",
                    motif.expansion,
                    default_head(task)
                );
                let refusal = |code: &str| Failure::new(code).context(label).task(task);
                let Ok(ir) = GenomeIr::parse(&text) else {
                    self.context.failures.push(refusal("parse_failed"));
                    continue;
                };
                let canonical = ir.canonical();
                let again = GenomeIr::parse(&canonical).map(|parsed| parsed.canonical());
                if again.ok().as_deref() != Some(canonical.as_str()) {
                    self.context.failures.push(refusal("not_idempotent"));
                    continue;
                }
                let slotted = matches!(
                    &ir,
                    GenomeIr::Pipeline(expr) if PipelineSlots::from_pipeline(expr).is_ok()
                );
                if !slotted {
                    self.context.failures.push(refusal("slot_refused"));
                    continue;
                }
                if validate_v3_genome(&ir, &spec, &schema).is_err() {
                    self.context.failures.push(refusal("grammar_refused"));
                    continue;
                }
                let independent = !self.repeats_a_host(&[text.as_str(), explicit.as_str()]);
                let legacy = legacy_artifacts(&ir, &mut self.artifacts);
                let mut primary = false;
                let mut bound = BoundCoverage::default();
                for data in tables {
                    match fit_twice(&ir, task, data) {
                        Outcome::Refused => {}
                        Outcome::Nondeterministic => {
                            self.determinism.exercised += 1;
                            self.determinism.failures.push(
                                Failure::new("fit_not_deterministic")
                                    .context(label)
                                    .table(data.shape)
                                    .task(task),
                            );
                        }
                        Outcome::Fitted(fitted) => {
                            self.determinism.exercised += 1;
                            if independent {
                                self.context.exercised += 1;
                            }
                            let coverage = bound_artifacts(&fitted.model, &mut self.artifacts);
                            if !primary && matches!(data.shape, Shape::S2 | Shape::S3) {
                                primary = true;
                                bound = coverage;
                            }
                        }
                    }
                }
                if label == "q1" {
                    q1_fitted = primary;
                }
                independent_fitted |= independent && primary;
                self.backend.push(BackendRow {
                    context: label.to_owned(),
                    task,
                    independent,
                    legacy_plan: legacy.plan,
                    legacy_fit: None,
                    legacy_cpp17: legacy.cpp17,
                    legacy_javascript: legacy.javascript,
                    bound_emit: bound.bound_emit,
                    fitted_cpp17: bound.fitted_cpp17,
                });
            }
            if !q1_fitted {
                self.context
                    .failures
                    .push(Failure::new("context_not_fitted").context("q1").task(task));
            } else if !independent_fitted {
                self.context
                    .failures
                    .push(Failure::new("context_not_independent").task(task));
            }
            if task == TaskLabel::Regression {
                self.legacy_context()?;
            }
        }
        Ok(())
    }

    /// Q3: legacy planner coverage and, where the planner accepts the text, a legacy fit on
    /// S2's numeric rows. Recorded only.
    fn legacy_context(&mut self) -> Result<()> {
        let text = format!("(pipe X {} (ridge))", self.motif.expansion);
        let Ok(ir) = GenomeIr::parse(&text) else {
            return Ok(());
        };
        let independent = !self.repeats_a_host(&[text.as_str()]);
        let legacy = legacy_artifacts(&ir, &mut self.artifacts);
        let s2 = data(self.tables, Shape::S2)?;
        let legacy_fit = legacy.plan.then(|| {
            s2.numeric_rows()
                .is_some_and(|rows| feat_math::fit(&ir, &rows, &s2.regression).is_ok())
        });
        self.backend.push(BackendRow {
            context: String::from("q3"),
            task: TaskLabel::Regression,
            independent,
            legacy_plan: legacy.plan,
            legacy_fit,
            legacy_cpp17: legacy.cpp17,
            legacy_javascript: legacy.javascript,
            bound_emit: false,
            fitted_cpp17: false,
        });
        Ok(())
    }
}

/// C1 to C4 for one fragment, with every artifact the checks produced.
pub(super) fn check_motif(
    table: &MotifTable,
    index: usize,
    motif: &MotifDef,
    options: &ValidateOptions,
    tables: &[Synthetic],
) -> Result<(MotifResult, Vec<Artifact>)> {
    let round_trip = round_trip(table, motif)?;
    let mut run = MotifRun {
        table,
        motif,
        tables,
        determinism: CheckResult::default(),
        context: CheckResult::default(),
        not_exercised: 0,
        backend: Vec::new(),
        artifacts: Vec::new(),
    };
    run.hosts(usize::try_from(options.hosts_per_fragment)?)?;
    run.contexts()?;
    let name_absence = name_absence(motif, &run.artifacts);
    let result = MotifResult {
        index: u32::try_from(index)?,
        round_trip,
        determinism: run.determinism.finish("determinism_not_exercised"),
        context_compatibility: run.context.finish("context_not_exercised"),
        name_absence,
        not_exercised: run.not_exercised,
        backend: run.backend,
    };
    Ok((result, run.artifacts))
}
