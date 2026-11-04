use super::*;

impl Receipt {
    /// Structural invariants the schema cannot express.
    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if ![
            RECEIPT_SCHEMA_ID,
            OBJECTIVE_SCHEMA_ID,
            V2_SCHEMA_ID,
            LEGACY_SCHEMA_ID,
        ]
        .contains(&self.schema.as_str())
        {
            errors.push(format!(
                "schema is {:?}, expected {RECEIPT_SCHEMA_ID:?} or {OBJECTIVE_SCHEMA_ID:?}",
                self.schema
            ));
        }
        // An objective changes what "best" means, so the schema says which ranking was used.
        if self.search.objective.is_some() != (self.schema == OBJECTIVE_SCHEMA_ID) {
            errors.push(format!(
                "a {} receipt {} an objective; only {OBJECTIVE_SCHEMA_ID} records one",
                self.schema,
                if self.search.objective.is_some() {
                    "records"
                } else {
                    "lacks"
                }
            ));
        }
        // Under an objective, every best program carries the loss it was best by.
        if self.search.objective.is_some() {
            if let Some(best) = &self.search.best {
                if best.validation_objective.is_none() {
                    errors.push("the best program lacks its validation_objective".into());
                }
            }
            for cut in &self.search.cuts {
                if cut.best_program.is_some() && cut.best_validation_objective.is_none() {
                    errors.push(format!(
                        "cut {}: the best program lacks its best_validation_objective",
                        cut.cut
                    ));
                }
            }
        }
        for stage in &self.stages {
            if !STATUSES.contains(&stage.status.as_str()) {
                errors.push(format!(
                    "stage {}: status {:?} is not one of {STATUSES:?}",
                    stage.name, stage.status
                ));
            }
            if stage.status != "passed" && stage.reason.as_ref().is_none_or(String::is_empty) {
                errors.push(format!(
                    "stage {}: a {} stage must record a reason",
                    stage.name, stage.status
                ));
            }
        }
        for required in ["dataset", "prepare", "propose", "search", "select", "emit"] {
            if !self.stages.iter().any(|s| s.name == required) {
                errors.push(format!("stage {required} is missing"));
            }
        }
        if self.search.cuts.len() != self.prepared.cuts.len() {
            errors.push(format!(
                "{} cut searches for {} prepared cuts",
                self.search.cuts.len(),
                self.prepared.cuts.len()
            ));
        }
        for cut in &self.search.cuts {
            if cut
                .successful_measurements
                .checked_add(cut.failed_or_unavailable_measurements)
                != Some(cut.completed_evaluations)
            {
                errors.push(format!(
                    "cut {}: {} evaluations but {} + {} outcomes",
                    cut.cut,
                    cut.completed_evaluations,
                    cut.successful_measurements,
                    cut.failed_or_unavailable_measurements
                ));
            }
            if cut.successful_measurements > 0 && cut.best_program.is_none() {
                errors.push(format!(
                    "cut {}: {} measurement(s) but no best program",
                    cut.cut, cut.successful_measurements
                ));
            }
        }
        let measured: u64 = self
            .search
            .cuts
            .iter()
            .map(|c| c.successful_measurements)
            .fold(0u64, u64::saturating_add);
        if measured > 0 && self.search.selection == crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "{measured} successful measurement(s) but the selection is the unmeasured fallback"
            ));
        }
        // A passed emit stage with nothing emitted would be a receipt that says the run produced
        // code while listing none, which no reader could catch without re-deriving it.
        if let Some(emit) = self.stages.iter().find(|s| s.name == "emit") {
            if emit.status == "passed" && self.artifacts.is_empty() {
                errors.push("stage emit passed but the receipt names no artifact".to_owned());
            }
            if emit.status == "failed" && !self.artifacts.is_empty() {
                errors.push(format!(
                    "stage emit failed but the receipt names {} artifact(s)",
                    self.artifacts.len()
                ));
            }
        }
        if self.ok && self.stages.iter().any(|s| s.status == "failed") {
            errors.push("ok is true but a stage failed".to_owned());
        }
        if let Some(best) = &self.search.best {
            if best.genome != self.search.selected_genome {
                errors.push("the selected genome is not the best one".to_owned());
            }
            if best.cuts_measured == 0 {
                errors.push("a best program measured on no cut".to_owned());
            }
        } else if self.search.selection != crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "selection {:?} without a best program",
                self.search.selection
            ));
        }
        let current = self.schema == RECEIPT_SCHEMA_ID || self.schema == OBJECTIVE_SCHEMA_ID;
        if current || self.schema == V2_SCHEMA_ID {
            self.v2_errors(&mut errors);
        }
        if current {
            self.v3_errors(&mut errors);
        } else if self.search.seeding.is_some() || self.search.islands.is_some() {
            errors.push(format!(
                "{} receipts carry no seeding or islands record",
                self.schema
            ));
        }
        errors
    }

    /// v3: the seeding record is internally consistent and every seed was attempted on every
    /// requested cut, so its place in the selection rests on this run's measurements alone.
    fn v3_errors(&self, errors: &mut Vec<String>) {
        use std::collections::BTreeSet;
        match &self.search.islands {
            None => errors.push("v3 requires an islands record".into()),
            Some(islands) => {
                if islands.count == 0 || (islands.count == 1 && !islands.migrations.is_empty()) {
                    errors.push(format!(
                        "{} island(s) with {} migration(s)",
                        islands.count,
                        islands.migrations.len()
                    ));
                }
                for migration in &islands.migrations {
                    if migration.from_island >= islands.count
                        || migration.to_island >= islands.count
                        || migration.from_island == migration.to_island
                        || !self.prepared.cuts.contains(&migration.cut)
                        || !migration.rmse.is_finite()
                        || migration.rmse < 0.0
                        || islands.migration_interval == 0
                        || (migration.after_generation + 1) % islands.migration_interval != 0
                    {
                        errors.push(format!(
                            "migration of {} on cut {} is not a migration between this run's \
                             islands at its interval",
                            migration.program_sha256, migration.cut
                        ));
                    }
                }
            }
        }
        let Some(seeding) = &self.search.seeding else {
            errors.push("v3 requires a seeding record".into());
            return;
        };
        // The cap is `--seed-limit`'s ceiling: the default admits [`crate::search::ELITES`].
        if seeding.seeds.len() > crate::seeds::MAX_SEED_LIMIT
            || seeding.seeds.len() as u64 > seeding.offered
        {
            errors.push(format!(
                "{} seed(s) from {} offered exceeds the cap of {}",
                seeding.seeds.len(),
                seeding.offered,
                crate::seeds::MAX_SEED_LIMIT
            ));
        }
        let mut distinct = BTreeSet::new();
        for seed in &seeding.seeds {
            if crate::sha256_hex(seed.genome.as_bytes()) != seed.program_sha256
                || !distinct.insert(&seed.genome)
                || seed.genome == crate::search::BOOTSTRAP_GENOME
            {
                errors.push(format!(
                    "seed {}: digest, duplicate or bootstrap mismatch",
                    seed.program_sha256
                ));
            }
            match seed.source.as_str() {
                crate::seeds::FROM_INCUMBENT if seeding.parent_receipt_sha256.is_none() => {
                    errors.push("an incumbent seed without a parent receipt digest".into())
                }
                crate::seeds::FROM_SEEDS_FILE if seeding.seeds_file_sha256.is_none() => {
                    errors.push("a seeds-file seed without a seeds file digest".into())
                }
                crate::seeds::FROM_INCUMBENT | crate::seeds::FROM_SEEDS_FILE => {}
                other => errors.push(format!("seed source {other:?} is unknown")),
            }
            let covered = self.search.coverage.as_ref().is_some_and(|coverage| {
                coverage
                    .iter()
                    .any(|p| p.program_sha256 == seed.program_sha256 && p.missing_cuts.is_empty())
            });
            if !covered {
                errors.push(format!(
                    "seed {} was not attempted on every requested cut",
                    seed.program_sha256
                ));
            }
        }
    }

    fn v2_errors(&self, errors: &mut Vec<String>) {
        use std::collections::BTreeSet;
        let search = &self.search;
        let expected: BTreeSet<u64> = self.prepared.cuts.iter().copied().collect();
        let actual: BTreeSet<u64> = search.cuts.iter().map(|c| c.cut).collect();
        if expected.is_empty()
            || expected.len() != self.prepared.cuts.len()
            || actual != expected
            || actual.len() != search.cuts.len()
        {
            errors.push("cut identities must match the distinct prepared cuts".into());
        }
        if search.budget_policy.as_deref() != Some(crate::search::BUDGET_POLICY)
            || search.evaluations == 0
            || search.evaluations > crate::search::MAX_EVALUATIONS as u64
            || search.seconds == 0
            || search.seconds > crate::search::MAX_SECONDS
        {
            errors.push("invalid v2 total budget or missing budget policy".into());
        }
        let rule = if search.objective.is_some() {
            crate::select::MIN_MEAN_VALIDATION_OBJECTIVE
        } else {
            crate::select::MIN_MEAN_VALIDATION_RMSE
        };
        if search.selection != rule
            || search.compared_cuts.as_ref() != Some(&expected.iter().copied().collect::<Vec<_>>())
        {
            errors.push("v2 selection must compare all requested cuts".into());
        }
        let Some(coverage) = &search.coverage else {
            errors.push("v2 requires program coverage".into());
            return;
        };
        let mut programs = BTreeSet::new();
        for program in coverage {
            let all: Vec<u64> = program
                .measured_cuts
                .iter()
                .chain(&program.failed_cuts)
                .chain(&program.missing_cuts)
                .copied()
                .collect();
            if !programs.insert(&program.program_sha256)
                || all.len() != expected.len()
                || all.iter().copied().collect::<BTreeSet<_>>() != expected
            {
                errors.push(
                    "program coverage must partition the requested cuts without duplicates".into(),
                );
            }
        }
        for cut in &search.cuts {
            let Some(a) = &cut.accounting else {
                errors.push(format!("cut {}: v2 requires phase accounting", cut.cut));
                continue;
            };
            let (x, y) = (&a.exploration, &a.comparison);
            for phase in [x, y] {
                if phase
                    .successful_measurements
                    .checked_add(phase.failed_measurements)
                    != Some(phase.completed_evaluations)
                    || phase.completed_evaluations > phase.evaluation_limit
                    || !phase.seconds.is_finite()
                    || phase.seconds < 0.0
                    || !phase.seconds_limit.is_finite()
                    || phase.seconds_limit < 0.0
                {
                    errors.push(format!(
                        "cut {}: invalid phase outcomes or admission limit",
                        cut.cut
                    ));
                }
            }
            let reserve = search.cuts.len() > 1;
            let exploration_limit = if reserve {
                search.evaluations.div_ceil(2)
            } else {
                search.evaluations
            };
            let exploration_seconds = search.seconds as f64 / if reserve { 2.0 } else { 1.0 };
            let comparison_limit = if reserve {
                search.evaluations.saturating_sub(x.completed_evaluations)
            } else {
                0
            };
            let comparison_seconds = if reserve {
                (search.seconds as f64 - x.seconds).max(0.0)
            } else {
                0.0
            };
            if x.evaluation_limit != exploration_limit
                || x.seconds_limit != exploration_seconds
                || y.evaluation_limit != comparison_limit
                || (y.seconds_limit - comparison_seconds).abs() > 1e-9
                || cut.completed_evaluations > search.evaluations
                || x.completed_evaluations.checked_add(y.completed_evaluations)
                    != Some(cut.completed_evaluations)
                || x.successful_measurements
                    .checked_add(y.successful_measurements)
                    != Some(cut.successful_measurements)
                || x.failed_measurements.checked_add(y.failed_measurements)
                    != Some(cut.failed_or_unavailable_measurements)
                || !a.active_seconds.is_finite()
                || a.active_seconds < 0.0
                || (a.active_seconds - x.seconds - y.seconds).abs() > 1e-9
                || a.deadline_overrun != (a.active_seconds > search.seconds as f64)
            {
                errors.push(format!(
                    "cut {}: total budget/phase accounting mismatch",
                    cut.cut
                ));
            }
            let exploration_stop_valid = match x.stopped_by.as_str() {
                crate::search::STOPPED_BY_EVALUATIONS => {
                    x.completed_evaluations == x.evaluation_limit
                }
                crate::search::STOPPED_BY_SECONDS => {
                    x.seconds >= x.seconds_limit && x.completed_evaluations < x.evaluation_limit
                }
                crate::search::STOPPED_BY_EXHAUSTED => x.completed_evaluations < x.evaluation_limit,
                _ => false,
            };
            if !exploration_stop_valid {
                errors.push(format!("cut {}: exploration stop reason mismatch", cut.cut));
            }
            let expected_stop = if !reserve {
                x.stopped_by.as_str()
            } else if a.active_seconds >= search.seconds as f64 {
                crate::search::STOPPED_BY_SECONDS
            } else if cut.completed_evaluations >= search.evaluations {
                crate::search::STOPPED_BY_EVALUATIONS
            } else {
                crate::search::STOPPED_BY_POOL
            };
            if cut.stopped_by != expected_stop
                || (reserve && y.stopped_by != cut.stopped_by)
                || (!reserve && *y != Phase::not_requested())
            {
                errors.push(format!("cut {}: phase stop reason mismatch", cut.cut));
            }
            if coverage
                .iter()
                .filter(|p| p.measured_cuts.contains(&cut.cut))
                .count() as u64
                != cut.successful_measurements
                || coverage
                    .iter()
                    .filter(|p| p.failed_cuts.contains(&cut.cut))
                    .count() as u64
                    != cut.failed_or_unavailable_measurements
            {
                errors.push(format!("cut {}: coverage/outcome mismatch", cut.cut));
            }
        }
        if let Some(best) = &search.best {
            if best.cuts_measured != expected.len() as u64
                || best.program_sha256 != search.selected_program_sha256
                || crate::sha256_hex(best.genome.as_bytes()) != best.program_sha256
                || !coverage.iter().any(|p| {
                    p.program_sha256 == best.program_sha256
                        && p.measured_cuts.len() == expected.len()
                })
            {
                errors.push("selected program lacks complete bound cut coverage".into());
            }
        }
    }
}
