use super::*;

impl RunService {
    pub(in crate::http::runs) fn prepare(&self, spec: &RunSpec) -> Result<Prepared> {
        if !matches!(
            spec.task,
            Task::Regression | Task::Binary | Task::Multiclass | Task::Auto
        ) {
            return Err(unavailable(
                "This run backend supports regression, binary and multiclass classification.",
            ));
        }
        if jope_runtime::backend::DeviceRequest::parse(&spec.device).is_none() {
            return Err(invalid("Unknown device."));
        }
        // Unknown comparator keys are refused here, before any dataset read or fit.
        let comparators = resolve_comparators(&spec.comparators, spec.effort)?;
        let mut backend =
            BackendPlan::resolve(spec, self.activity.admitted_ranker(), 1).map_err(|error| {
                let code = crate::run::service::error_code(&error);
                // Keep the public catalog code without exposing private backend diagnostics.
                issue(
                    code,
                    format!("{code}: The requested run backend could not be prepared."),
                )
            })?;
        if spec.test_dataset.is_some()
            || spec.split.kind != SplitKind::Random
            || spec.split.group_column.is_some()
            || spec.split.time_column.is_some()
        {
            return Err(unavailable(
                "This run backend supports random FIT/internal assessment splits without a separate test dataset.",
            ));
        }
        let language_filter = run_language_filter(&spec.languages)?;
        if spec.task == Task::Regression {
            crate::run::service::run_objective(&spec.objective).map_err(|_| {
                invalid(
                    "The objective is invalid or requires unsupported ordered/auxiliary inputs.",
                )
            })?;
        } else if spec.task == Task::Binary {
            if let ObjectiveSpec::Name(name) = &spec.objective {
                if !name.trim_start().starts_with('{') {
                    crate::objective::binary::BinaryObjective::parse(
                        name,
                        ["0".into(), "1".into()],
                    )
                    .map_err(|_| invalid("The objective is invalid for binary classification."))?;
                }
            }
        } else if spec.task == Task::Multiclass {
            if let ObjectiveSpec::Name(name) = &spec.objective {
                if !name.trim_start().starts_with('{') {
                    crate::objective::multiclass::MulticlassObjective::parse(
                        name,
                        vec!["0".into(), "1".into(), "2".into()],
                    )
                    .map_err(|_| {
                        invalid("The objective is invalid for multiclass classification.")
                    })?;
                }
            }
        }
        if !spec.split.holdout_fraction.is_finite()
            || !(0.0..1.0).contains(&spec.split.holdout_fraction)
            || spec.split.holdout_fraction == 0.0
        {
            return Err(invalid(
                "holdout_fraction must be finite and strictly between zero and one.",
            ));
        }
        let record = self
            .store
            .get(&RecordKey::Dataset(spec.dataset.clone()))
            .map_err(io)?
            .ok_or_else(|| issue(ErrorCode::NotFound, "Registered dataset does not exist."))?;
        let dataset: RegisteredDataset = serde_json::from_value(record.data).map_err(io)?;
        if dataset.id != spec.dataset {
            return Err(io("Dataset identity mismatch."));
        }
        if dataset.csv.bytes > self.config.limits.max_dataset_bytes.value {
            return Err(invalid("Dataset exceeds configured maximum bytes."));
        }
        let fit_estimate =
            super::q1_fit_rows::fit_row_count(dataset.rows, spec.split.holdout_fraction);
        if fit_estimate > self.config.limits.rung1_max_rows.value {
            return Err(invalid("FIT exceeds limits.rung1_max_rows."));
        }
        let available = SystemSensors::without_gpu(&self.home)
            .mem_available()
            .map_err(|_| unavailable("Memory admission sensor unavailable."))?;
        if dataset
            .csv
            .bytes
            .saturating_mul(4)
            .saturating_add(WORKER_RAM)
            > available
        {
            return Err(issue(
                ErrorCode::BackendUnavailable,
                "Insufficient available memory for this bounded run.",
            ));
        }
        let source = self.store.read_blob(&dataset.csv).map_err(io)?;
        let mut table = jope_runtime::table::parse_csv(&source, &Default::default()).map_err(io)?;
        let matrix =
            jmat::Matrix::open(self.store.read_blob(&dataset.matrix).map_err(io)?).map_err(io)?;
        let target = table
            .headers
            .iter()
            .position(|v| v == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        if spec.ignore_columns.contains(&spec.target) {
            return Err(invalid("The target cannot be ignored."));
        }
        let ignored: std::collections::BTreeSet<_> = spec.ignore_columns.iter().collect();
        if ignored.len() != spec.ignore_columns.len()
            || ignored.iter().any(|v| !table.headers.contains(v))
        {
            return Err(invalid(
                "Ignored columns must be distinct registered columns.",
            ));
        }
        let (resolved_task, task_inferred, task_reason) = match spec.task {
            Task::Regression => (
                Task::Regression,
                false,
                "Explicit regression task.".to_owned(),
            ),
            Task::Binary => (
                Task::Binary,
                false,
                "Explicit binary classification task.".to_owned(),
            ),
            Task::Multiclass => (
                Task::Multiclass,
                false,
                "Explicit multiclass classification task.".to_owned(),
            ),
            Task::Auto => {
                let mut distinct = std::collections::BTreeSet::new();
                for row in &table.rows {
                    let Some(label) = row[target]
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    else {
                        return Err(invalid(&format!(
                            "Target column `{}` cannot contain missing values.",
                            spec.target
                        )));
                    };
                    distinct.insert(label);
                }
                let all_numeric = distinct
                    .iter()
                    .all(|label| crate::dataset::finite_number(label).is_some());
                if distinct.len() == 2 {
                    (
                        Task::Binary,
                        true,
                        "Inferred binary task from two distinct target labels.".to_owned(),
                    )
                } else if all_numeric {
                    (
                        Task::Regression,
                        true,
                        format!(
                            "Inferred regression task from numeric target with {} distinct values.",
                            distinct.len()
                        ),
                    )
                } else if distinct.len() >= 3 {
                    (
                        Task::Multiclass,
                        true,
                        format!(
                            "Inferred multiclass task from {} non-numeric target labels.",
                            distinct.len()
                        ),
                    )
                } else {
                    return Err(invalid(&format!(
                        "Target column `{}` must have at least 2 distinct values.",
                        spec.target
                    )));
                }
            }
        };
        if resolved_task == Task::Binary {
            return self.prepare_binary(
                spec,
                dataset,
                table,
                target,
                &ignored,
                language_filter,
                comparators,
                backend,
                task_inferred,
                task_reason,
            );
        }
        if resolved_task == Task::Multiclass {
            return self.prepare_multiclass(
                spec,
                dataset,
                table,
                target,
                &ignored,
                comparators,
                backend,
                task_inferred,
                task_reason,
            );
        }
        let (objective, resolved_objective) = crate::run::service::run_objective(&spec.objective)
            .map_err(|_| {
            invalid("The objective is invalid or requires unsupported ordered/auxiliary inputs.")
        })?;
        let numeric: Vec<bool> = (0..table.headers.len())
            .map(|index| matrix.columns()[index].kind == jmat::ColumnKind::Numeric)
            .collect();
        let selection = numeric_feature_selection(&table.headers, &numeric, target, &ignored)
            .map_err(|detail| invalid(&detail))?;
        if table.rows.iter().any(|row| {
            row[target]
                .as_deref()
                .and_then(crate::dataset::finite_number)
                .is_none()
        }) {
            return Err(invalid(&format!(
                "Target column `{}` must be a finite number in every row.",
                spec.target
            )));
        }
        let leaks = crate::dataset::leakage::target_leakage(&table, &spec.target);
        let id_columns = crate::dataset::group_leak::id_like_columns(
            &table.headers,
            &spec.target,
            &spec.ignore_columns,
        );
        let id_values: Vec<(String, Vec<Option<String>>)> = id_columns
            .iter()
            .filter_map(|name| {
                let index = table.headers.iter().position(|header| header == name)?;
                let values = table
                    .rows
                    .iter()
                    .map(|row| row.get(index).cloned().flatten())
                    .collect();
                Some((name.clone(), values))
            })
            .collect();
        let selected = selection.kept;
        let features: Vec<_> = selected
            .iter()
            .filter(|&&i| i != target)
            .map(|&i| table.headers[i].clone())
            .collect();
        table.headers = selected.iter().map(|&i| table.headers[i].clone()).collect();
        table.rows = table
            .rows
            .into_iter()
            .map(|row| selected.iter().map(|&i| row[i].clone()).collect())
            .collect();
        let fit_count =
            super::q1_fit_rows::fit_row_count(table.rows.len() as u64, spec.split.holdout_fraction)
                as usize;
        let effort = SearchEffort::from_name(match spec.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .unwrap()
        .limits_with_budget(BudgetOverrides {
            evaluations: spec.budget.evaluations.map(|v| v.0),
            wall_seconds: spec.budget.wall_seconds.map(|v| v.0),
            ..Default::default()
        });
        if effort.evaluations == 0
            || effort.wall_seconds == 0
            || fit_count < usize::from(effort.rung1_folds)
            || fit_count > self.config.limits.rung1_max_rows.value as usize
            || fit_count >= table.rows.len()
        {
            return Err(invalid(
                "Budget or FIT row count is outside the backend's configured limits.",
            ));
        }
        let target_column = table
            .headers
            .iter()
            .position(|name| name == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        let (fit_ids, holdout_ids) = random_split(
            table.rows.len(),
            fit_count,
            spec.seed.0,
            "public-run/fit-v1",
        )
        .map_err(io)?;
        let encode = |ids: &[usize]| -> Result<Vec<u8>> {
            let mut part = table.clone();
            part.rows = ids.iter().map(|&i| table.rows[i].clone()).collect();
            jmat::encode(&part).map_err(io)
        };
        let fit = encode(&fit_ids)?;
        let holdout = encode(&holdout_ids)?;
        let spanning =
            crate::dataset::group_leak::spanning_columns(&id_values, &fit_ids, &holdout_ids);
        let baseline = {
            let training =
                crate::dataset::baseline::target_values(&table.rows, target_column, &fit_ids)
                    .ok_or_else(|| invalid("Target column is not finite on the training rows."))?;
            let evaluation =
                crate::dataset::baseline::target_values(&table.rows, target_column, &holdout_ids)
                    .ok_or_else(|| invalid("Target column is not finite on the evaluation rows."))?;
            crate::dataset::baseline::training_mean_rmse(&training, &evaluation)
                .ok_or_else(|| invalid("The training-mean baseline needs both row sets."))?
        };
        let governor = Governor::new(
            std::num::NonZeroU64::new(u64::MAX).unwrap(),
            NonZeroU32::new(1).unwrap(),
            self.config.limits.disk_reserve.value,
        );
        let estimates = governor.preflight(
            Requirements {
                matrix_ram_bytes: dataset.csv.bytes.saturating_mul(4),
                per_worker_ram_bytes: WORKER_RAM,
                matrix_disk_bytes: (fit.len() + holdout.len()) as u64,
                artifact_bytes: MODEL_LIMIT,
                ..Default::default()
            },
            &mut SystemSensors::without_gpu(&self.home),
        )?;
        let timeout = self
            .config
            .limits
            .eval_timeout
            .value
            .as_millis()
            .min(u128::from(effort.wall_seconds) * 1000)
            .max(1) as u64;
        let mut normalized = serde_json::to_value(spec).map_err(io)?;
        normalized["dry_run"] = json!(false);
        normalized["plan_digest"] = Value::Null;
        let plan_digest = hash(
            &serde_json::to_vec(
                &json!({"backend":"cpu-ridge-seeded-rmse/v1", "spec":normalized,
            "dataset":dataset.csv, "fit":hash(&fit), "binary":self.binary, "timeout_ms":timeout,
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates,
            "ranker_bundle":backend.bundle_digest,"placement":backend.placement}),
            )
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let mut public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"regression","inferred":task_inferred,"reason":task_reason},
            "objective":resolved_objective,
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effort.rung1_folds,"rounds_in_flight":1},
            "guidance":backend.guidance,"landmarks":{"mode":spec.landmarks,"sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":language_filter.excluded_ops(),"intersection_empty":false},
            "placement":backend.placement,
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":backend.warnings})).map_err(io)?;
        if !selection.skipped.is_empty() {
            public.warnings.push(
                skipped_column_warning(&selection.skipped)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        for leak in &leaks {
            public.warnings.push(
                crate::dataset::leakage::warning(leak)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        for column in &spanning {
            public.warnings.push(
                crate::dataset::group_leak::warning(column)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        public.warnings.push(
            crate::dataset::baseline::warning(&spec.target, holdout_ids.len(), baseline)
                .map_err(|error| invalid(&error.to_string()))?,
        );
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: backend.bundle_digest.clone(),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective,
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: effort.evaluations,
            wall_seconds: effort.wall_seconds,
            evaluation_wall_ms: timeout,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(&fit, &search)
            .map_err(|_| invalid("Unable to prepare numeric FIT landmarks."))?;
        public.landmarks.sample_rows = backend.landmark_rows as u32;
        public.comparators = Some(comparators);
        let rows = |ids: Vec<usize>| {
            ids.into_iter()
                .map(|row| AuditRowId {
                    dataset: spec.dataset.clone(),
                    row: row as u64,
                })
                .collect()
        };
        Ok(Prepared {
            public,
            search,
            fit,
            holdout,
            fit_rows: rows(fit_ids),
            holdout_rows: rows(holdout_ids),
            backend,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_binary(
        &self,
        spec: &RunSpec,
        dataset: RegisteredDataset,
        mut table: jope_runtime::table::CsvTable,
        target: usize,
        ignored: &std::collections::BTreeSet<&String>,
        language_filter: feat_math::evolve::api::capability::LanguageFilter,
        mut comparators: ResolvedComparators,
        mut backend: BackendPlan,
        task_inferred: bool,
        task_reason: String,
    ) -> Result<Prepared> {
        let _ = binary_run_roster(&spec.languages)?;
        let mut distinct = std::collections::BTreeSet::new();
        for row in &mut table.rows {
            let Some(trimmed) = row[target]
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
            else {
                return Err(invalid(&format!(
                    "Target column `{}` cannot contain missing values.",
                    spec.target
                )));
            };
            distinct.insert(trimmed.clone());
            row[target] = Some(trimmed);
        }
        let all_numeric = distinct
            .iter()
            .all(|label| crate::dataset::finite_number(label).is_some());
        let mut classes: Vec<String> = distinct.into_iter().collect();
        if all_numeric {
            classes.sort_by(|a, b| {
                let left = crate::dataset::finite_number(a).unwrap_or(0.0);
                let right = crate::dataset::finite_number(b).unwrap_or(0.0);
                left.total_cmp(&right).then_with(|| a.cmp(b))
            });
            for window in classes.windows(2) {
                if crate::dataset::finite_number(&window[0])
                    == crate::dataset::finite_number(&window[1])
                {
                    return Err(invalid(&format!(
                        "Binary target column `{}` has duplicate numeric class encodings.",
                        spec.target
                    )));
                }
            }
        }
        if classes.len() != 2 {
            return Err(invalid(&format!(
                "Binary target column `{}` must have exactly 2 distinct classes; found {}.",
                spec.target,
                classes.len()
            )));
        }
        let class_order = [classes[0].clone(), classes[1].clone()];
        let (objective, resolved_objective) =
            crate::run::service::run_binary_objective(&spec.objective, class_order.clone())
                .map_err(|_| {
                    invalid(
                        "The objective is invalid for binary classification or its class_order does not match the dataset.",
                    )
                })?;
        if comparators.enabled {
            comparators = ResolvedComparators {
                enabled: false,
                tier: None,
                ids: Vec::new(),
                ceiling_seconds: None,
                reason: "binary task".into(),
            };
        }
        let kept: Vec<usize> = (0..table.headers.len())
            .filter(|&index| !ignored.contains(&&table.headers[index]))
            .collect();
        let features: Vec<String> = kept
            .iter()
            .filter(|&&index| index != target)
            .map(|&index| table.headers[index].clone())
            .collect();
        if features.is_empty() {
            return Err(invalid("At least one feature column is required."));
        }
        let leaks = crate::dataset::leakage::target_leakage(&table, &spec.target);
        let id_columns = crate::dataset::group_leak::id_like_columns(
            &table.headers,
            &spec.target,
            &spec.ignore_columns,
        );
        let id_values: Vec<(String, Vec<Option<String>>)> = id_columns
            .iter()
            .filter_map(|name| {
                let index = table.headers.iter().position(|header| header == name)?;
                let values = table
                    .rows
                    .iter()
                    .map(|row| row.get(index).cloned().flatten())
                    .collect();
                Some((name.clone(), values))
            })
            .collect();
        table.headers = kept.iter().map(|&i| table.headers[i].clone()).collect();
        table.rows = table
            .rows
            .into_iter()
            .map(|row| kept.iter().map(|&i| row[i].clone()).collect())
            .collect();
        let fit_count =
            super::q1_fit_rows::fit_row_count(table.rows.len() as u64, spec.split.holdout_fraction)
                as usize;
        let effort = SearchEffort::from_name(match spec.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .unwrap()
        .limits_with_budget(BudgetOverrides {
            evaluations: spec.budget.evaluations.map(|v| v.0),
            wall_seconds: spec.budget.wall_seconds.map(|v| v.0),
            ..Default::default()
        });
        if effort.evaluations == 0
            || effort.wall_seconds == 0
            || fit_count < 4
            || fit_count > self.config.limits.rung1_max_rows.value as usize
            || fit_count >= table.rows.len()
        {
            return Err(invalid(
                "Budget or FIT row count is outside the backend's configured limits.",
            ));
        }
        let target_column = table
            .headers
            .iter()
            .position(|name| name == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        let row_classes: Vec<usize> = table
            .rows
            .iter()
            .map(|row| usize::from(row[target_column].as_deref() == Some(class_order[1].as_str())))
            .collect();
        let (fit_ids, holdout_ids) = crate::discover::splits::stratified_split(
            &row_classes,
            fit_count,
            spec.seed.0,
            "public-run/fit-v1",
        )
        .map_err(io)?;
        let fit_c0 = fit_ids.iter().filter(|&&r| row_classes[r] == 0).count();
        let fit_c1 = fit_ids.len() - fit_c0;
        let hold_c0 = holdout_ids.iter().filter(|&&r| row_classes[r] == 0).count();
        let hold_c1 = holdout_ids.len() - hold_c0;
        if fit_c0 < 2 || fit_c1 < 2 || hold_c0 < 1 || hold_c1 < 1 {
            return Err(invalid(
                "Each binary target class needs at least 2 FIT rows and 1 holdout row.",
            ));
        }
        let encode = |ids: &[usize]| -> Result<Vec<u8>> {
            let mut part = table.clone();
            part.rows = ids.iter().map(|&i| table.rows[i].clone()).collect();
            jmat::encode(&part).map_err(io)
        };
        let fit = encode(&fit_ids)?;
        let holdout = encode(&holdout_ids)?;
        let spanning =
            crate::dataset::group_leak::spanning_columns(&id_values, &fit_ids, &holdout_ids);
        let parsed_objective =
            crate::objective::binary::BinaryObjective::parse_json(&objective).map_err(io)?;
        let p_pos = fit_c1 as f64 / fit_ids.len() as f64;
        let prior_probs = vec![[1.0 - p_pos, p_pos]; holdout_ids.len()];
        let holdout_labels: Vec<&str> = holdout_ids
            .iter()
            .map(|&r| class_order[row_classes[r]].as_str())
            .collect();
        let prior_baseline = parsed_objective
            .evaluate(&holdout_labels, &prior_probs)
            .ok()
            .map(|s| s.value);
        let effective_cv_folds = (effort.rung1_folds as usize).min(fit_c0.min(fit_c1)).max(2) as u8;
        let governor = Governor::new(
            std::num::NonZeroU64::new(u64::MAX).unwrap(),
            NonZeroU32::new(1).unwrap(),
            self.config.limits.disk_reserve.value,
        );
        let estimates = governor.preflight(
            Requirements {
                matrix_ram_bytes: dataset.csv.bytes.saturating_mul(4),
                per_worker_ram_bytes: WORKER_RAM,
                matrix_disk_bytes: (fit.len() + holdout.len()) as u64,
                artifact_bytes: MODEL_LIMIT,
                ..Default::default()
            },
            &mut SystemSensors::without_gpu(&self.home),
        )?;
        let timeout = self
            .config
            .limits
            .eval_timeout
            .value
            .as_millis()
            .min(u128::from(effort.wall_seconds) * 1000)
            .max(1) as u64;
        let mut normalized = serde_json::to_value(spec).map_err(io)?;
        normalized["dry_run"] = json!(false);
        normalized["plan_digest"] = Value::Null;
        let plan_digest = hash(
            &serde_json::to_vec(&json!({"backend":"cpu-native-binary/v1", "spec":normalized,
            "dataset":dataset.csv, "fit":hash(&fit), "binary":self.binary, "timeout_ms":timeout,
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates,
            "ranker_bundle":backend.bundle_digest,"placement":backend.placement}))
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let excluded_ops: Vec<String> = if spec.languages.contains(&Language::Javascript) {
            vec![
                "tab_boost".into(),
                "extra_trees".into(),
                "gam_boost".into(),
                "portfolio".into(),
            ]
        } else {
            language_filter
                .excluded_ops()
                .iter()
                .map(|s| (*s).to_owned())
                .collect()
        };
        let mut public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"binary","inferred":task_inferred,"reason":task_reason},
            "objective":resolved_objective,
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effective_cv_folds,"rounds_in_flight":1},
            "guidance":backend.guidance,"landmarks":{"mode":spec.landmarks,"sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":excluded_ops,"intersection_empty":false},
            "placement":backend.placement,
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":backend.warnings})).map_err(io)?;
        for leak in &leaks {
            public.warnings.push(
                crate::dataset::leakage::warning(leak)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        for column in &spanning {
            public.warnings.push(
                crate::dataset::group_leak::warning(column)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        if let Some(score_val) = prior_baseline {
            public.warnings.push(
                crate::warnings::catalog("BASELINE_MEAN")
                    .expect("BASELINE_MEAN is catalogued")
                    .event(
                        WarningState::Raised,
                        Some(score_val),
                        None,
                        format!(
                            "Baseline predicts the training-set positive prior of `{}` ({p_pos:.6}). On the same {} evaluation rows its {} is {score_val:.6}.",
                            spec.target,
                            holdout_ids.len(),
                            public.objective.name,
                        ),
                        "Compare the model with this number. It uses the same evaluation rows and no others.",
                    )
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: backend.bundle_digest.clone(),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective,
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: effort.evaluations,
            wall_seconds: effort.wall_seconds,
            evaluation_wall_ms: timeout,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(&fit, &search)
            .map_err(|_| invalid("Unable to prepare binary FIT context."))?;
        public.landmarks.sample_rows = backend.landmark_rows as u32;
        public.comparators = Some(comparators);
        let rows = |ids: Vec<usize>| {
            ids.into_iter()
                .map(|row| AuditRowId {
                    dataset: spec.dataset.clone(),
                    row: row as u64,
                })
                .collect()
        };
        Ok(Prepared {
            public,
            search,
            fit,
            holdout,
            fit_rows: rows(fit_ids),
            holdout_rows: rows(holdout_ids),
            backend,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_multiclass(
        &self,
        spec: &RunSpec,
        dataset: RegisteredDataset,
        mut table: jope_runtime::table::CsvTable,
        target: usize,
        ignored: &std::collections::BTreeSet<&String>,
        mut comparators: ResolvedComparators,
        mut backend: BackendPlan,
        task_inferred: bool,
        task_reason: String,
    ) -> Result<Prepared> {
        let _ = multiclass_run_roster(&spec.languages)?;
        let mut distinct = std::collections::BTreeSet::new();
        for row in &mut table.rows {
            let Some(trimmed) = row[target]
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
            else {
                return Err(invalid(&format!(
                    "Target column `{}` cannot contain missing values.",
                    spec.target
                )));
            };
            distinct.insert(trimmed.clone());
            row[target] = Some(trimmed);
        }
        let all_numeric = distinct
            .iter()
            .all(|label| crate::dataset::finite_number(label).is_some());
        let mut classes: Vec<String> = distinct.into_iter().collect();
        if all_numeric {
            classes.sort_by(|a, b| {
                let left = crate::dataset::finite_number(a).unwrap_or(0.0);
                let right = crate::dataset::finite_number(b).unwrap_or(0.0);
                left.total_cmp(&right).then_with(|| a.cmp(b))
            });
            for window in classes.windows(2) {
                if crate::dataset::finite_number(&window[0])
                    == crate::dataset::finite_number(&window[1])
                {
                    return Err(invalid(&format!(
                        "Multiclass target column `{}` has duplicate numeric class encodings.",
                        spec.target
                    )));
                }
            }
        }
        if classes.len() < 3 {
            return Err(invalid(&format!(
                "Multiclass target column `{}` must have at least 3 distinct classes; found {}.",
                spec.target,
                classes.len()
            )));
        }
        let (objective, resolved_objective) =
            crate::run::service::run_multiclass_objective(&spec.objective, classes.clone())
                .map_err(|_| {
                    invalid(
                        "The objective is invalid for multiclass classification or its class_order does not match the dataset.",
                    )
                })?;
        if comparators.enabled {
            comparators = ResolvedComparators {
                enabled: false,
                tier: None,
                ids: Vec::new(),
                ceiling_seconds: None,
                reason: "multiclass task".into(),
            };
        }
        let kept: Vec<usize> = (0..table.headers.len())
            .filter(|&index| !ignored.contains(&&table.headers[index]))
            .collect();
        let features: Vec<String> = kept
            .iter()
            .filter(|&&index| index != target)
            .map(|&index| table.headers[index].clone())
            .collect();
        if features.is_empty() {
            return Err(invalid("At least one feature column is required."));
        }
        let leaks = crate::dataset::leakage::target_leakage(&table, &spec.target);
        let id_columns = crate::dataset::group_leak::id_like_columns(
            &table.headers,
            &spec.target,
            &spec.ignore_columns,
        );
        let id_values: Vec<(String, Vec<Option<String>>)> = id_columns
            .iter()
            .filter_map(|name| {
                let index = table.headers.iter().position(|header| header == name)?;
                let values = table
                    .rows
                    .iter()
                    .map(|row| row.get(index).cloned().flatten())
                    .collect();
                Some((name.clone(), values))
            })
            .collect();
        table.headers = kept.iter().map(|&i| table.headers[i].clone()).collect();
        table.rows = table
            .rows
            .into_iter()
            .map(|row| kept.iter().map(|&i| row[i].clone()).collect())
            .collect();
        let fit_count =
            super::q1_fit_rows::fit_row_count(table.rows.len() as u64, spec.split.holdout_fraction)
                as usize;
        let effort = SearchEffort::from_name(match spec.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .unwrap()
        .limits_with_budget(BudgetOverrides {
            evaluations: spec.budget.evaluations.map(|v| v.0),
            wall_seconds: spec.budget.wall_seconds.map(|v| v.0),
            ..Default::default()
        });
        let num_classes = classes.len();
        if effort.evaluations == 0
            || effort.wall_seconds == 0
            || fit_count < num_classes.saturating_mul(2)
            || fit_count > self.config.limits.rung1_max_rows.value as usize
            || fit_count >= table.rows.len()
        {
            return Err(invalid(
                "Budget or FIT row count is outside the backend's configured limits.",
            ));
        }
        let target_column = table
            .headers
            .iter()
            .position(|name| name == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        let class_index_map: std::collections::BTreeMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();
        let row_classes: Vec<usize> = table
            .rows
            .iter()
            .map(|row| {
                row[target_column]
                    .as_deref()
                    .and_then(|s| class_index_map.get(s).copied())
                    .ok_or_else(|| invalid("Target label missing from multiclass order."))
            })
            .collect::<Result<Vec<_>>>()?;
        let (fit_ids, holdout_ids) = crate::discover::splits::stratified_split(
            &row_classes,
            fit_count,
            spec.seed.0,
            "public-run/fit-v1",
        )
        .map_err(io)?;
        let mut fit_class_counts = vec![0usize; num_classes];
        for &r in &fit_ids {
            fit_class_counts[row_classes[r]] += 1;
        }
        let mut holdout_class_counts = vec![0usize; num_classes];
        for &r in &holdout_ids {
            holdout_class_counts[row_classes[r]] += 1;
        }
        if fit_class_counts.iter().any(|&c| c < 2) || holdout_class_counts.iter().any(|&c| c < 1) {
            return Err(invalid(
                "Each multiclass target class needs at least 2 FIT rows and 1 holdout row.",
            ));
        }
        let min_fit_class = fit_class_counts.iter().copied().min().unwrap_or(0);
        let encode = |ids: &[usize]| -> Result<Vec<u8>> {
            let mut part = table.clone();
            part.rows = ids.iter().map(|&i| table.rows[i].clone()).collect();
            jmat::encode(&part).map_err(io)
        };
        let fit = encode(&fit_ids)?;
        let holdout = encode(&holdout_ids)?;
        let spanning =
            crate::dataset::group_leak::spanning_columns(&id_values, &fit_ids, &holdout_ids);
        let parsed_objective =
            crate::objective::multiclass::MulticlassObjective::parse_json(&objective)
                .map_err(io)?;
        let prior_vec: Vec<f64> = fit_class_counts
            .iter()
            .map(|&c| c as f64 / fit_ids.len() as f64)
            .collect();
        let prior_probs = vec![prior_vec; holdout_ids.len()];
        let holdout_labels: Vec<&str> = holdout_ids
            .iter()
            .map(|&r| classes[row_classes[r]].as_str())
            .collect();
        let prior_baseline = parsed_objective
            .evaluate(&holdout_labels, &prior_probs)
            .ok()
            .map(|s| s.value);
        let effective_cv_folds = (effort.rung1_folds as usize).min(min_fit_class).max(2) as u8;
        let governor = Governor::new(
            std::num::NonZeroU64::new(u64::MAX).unwrap(),
            NonZeroU32::new(1).unwrap(),
            self.config.limits.disk_reserve.value,
        );
        let estimates = governor.preflight(
            Requirements {
                matrix_ram_bytes: dataset.csv.bytes.saturating_mul(4),
                per_worker_ram_bytes: WORKER_RAM,
                matrix_disk_bytes: (fit.len() + holdout.len()) as u64,
                artifact_bytes: MODEL_LIMIT,
                ..Default::default()
            },
            &mut SystemSensors::without_gpu(&self.home),
        )?;
        let timeout = self
            .config
            .limits
            .eval_timeout
            .value
            .as_millis()
            .min(u128::from(effort.wall_seconds) * 1000)
            .max(1) as u64;
        let mut normalized = serde_json::to_value(spec).map_err(io)?;
        normalized["dry_run"] = json!(false);
        normalized["plan_digest"] = Value::Null;
        let plan_digest = hash(
            &serde_json::to_vec(
                &json!({"backend":"cpu-native-multiclass/v1", "spec":normalized,
            "dataset":dataset.csv, "fit":hash(&fit), "binary":self.binary, "timeout_ms":timeout,
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates,
            "ranker_bundle":backend.bundle_digest,"placement":backend.placement}),
            )
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let excluded_ops: Vec<String> = Vec::new();
        let mut public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"multiclass","inferred":task_inferred,"reason":task_reason},
            "objective":resolved_objective,
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effective_cv_folds,"rounds_in_flight":1},
            "guidance":backend.guidance,"landmarks":{"mode":spec.landmarks,"sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":excluded_ops,"intersection_empty":false},
            "placement":backend.placement,
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":backend.warnings})).map_err(io)?;
        for leak in &leaks {
            public.warnings.push(
                crate::dataset::leakage::warning(leak)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        for column in &spanning {
            public.warnings.push(
                crate::dataset::group_leak::warning(column)
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        if let Some(score_val) = prior_baseline {
            public.warnings.push(
                crate::warnings::catalog("BASELINE_MEAN")
                    .expect("BASELINE_MEAN is catalogued")
                    .event(
                        WarningState::Raised,
                        Some(score_val),
                        None,
                        format!(
                            "Baseline predicts the training-set empirical {}-class prior of `{}`. On the same {} evaluation rows its {} is {score_val:.6}.",
                            num_classes,
                            spec.target,
                            holdout_ids.len(),
                            public.objective.name,
                        ),
                        "Compare the model with this number. It uses the same evaluation rows and no others.",
                    )
                    .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: backend.bundle_digest.clone(),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective,
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: effort.evaluations,
            wall_seconds: effort.wall_seconds,
            evaluation_wall_ms: timeout,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(&fit, &search)
            .map_err(|_| invalid("Unable to prepare multiclass FIT context."))?;
        public.landmarks.sample_rows = backend.landmark_rows as u32;
        public.comparators = Some(comparators);
        let rows = |ids: Vec<usize>| {
            ids.into_iter()
                .map(|row| AuditRowId {
                    dataset: spec.dataset.clone(),
                    row: row as u64,
                })
                .collect()
        };
        Ok(Prepared {
            public,
            search,
            fit,
            holdout,
            fit_rows: rows(fit_ids),
            holdout_rows: rows(holdout_ids),
            backend,
        })
    }
}

/// Resolve the requested comparator policy at the run effort. `off` and every policy at `fast`
/// effort disable the stage; `auto` and `all` select the full roster at the matching tier;
/// `only` selects a validated subset. Unknown keys are refused at every effort, before any fit.
pub(in crate::http::runs) fn resolve_comparators(
    policy: &ComparatorPolicy,
    effort: Effort,
) -> Result<ResolvedComparators> {
    use jope_comparators::{ComparatorId, ComparatorTier};
    let ids: Vec<ComparatorId> = match policy {
        ComparatorPolicy::Off => Vec::new(),
        ComparatorPolicy::Auto | ComparatorPolicy::All => ComparatorId::ALL.to_vec(),
        ComparatorPolicy::Only { ids } => {
            let mut selected = Vec::with_capacity(ids.len());
            for key in ids {
                let id = ComparatorId::from_key(key).ok_or_else(|| {
                    let valid: Vec<_> = ComparatorId::ALL.iter().map(|id| id.key()).collect();
                    let mut problem = invalid(&format!(
                        "Unknown comparator `{key}`. Valid comparators: {}.",
                        valid.join(", ")
                    ));
                    problem.field = Some("comparators.ids".into());
                    problem
                })?;
                if !selected.contains(&id) {
                    selected.push(id);
                }
            }
            if selected.is_empty() {
                let mut problem =
                    invalid("comparators mode `only` needs at least one comparator id.");
                problem.field = Some("comparators.ids".into());
                return Err(problem);
            }
            selected
        }
    };
    let disabled = |reason: &str| ResolvedComparators {
        enabled: false,
        tier: None,
        ids: Vec::new(),
        ceiling_seconds: None,
        reason: reason.into(),
    };
    if policy.is_off() {
        return Ok(disabled("not requested"));
    }
    let effort_name = match effort {
        Effort::Fast => return Ok(disabled("fast effort")),
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::Ultra => "ultra",
    };
    let tier = ComparatorTier::from_effort(effort_name)
        .ok_or_else(|| io("Comparator tier is missing for this effort."))?;
    Ok(ResolvedComparators {
        enabled: true,
        tier: Some(tier.as_str().into()),
        ids: ids.iter().map(|id| id.key().to_owned()).collect(),
        ceiling_seconds: Some(tier.budget().stage_ceiling.as_secs()),
        reason: "requested".into(),
    })
}

#[derive(Debug)]
struct NumericColumns {
    kept: Vec<usize>,
    skipped: Vec<String>,
}

/// Keep the numeric target and numeric features. Non-numeric features are skipped
/// and named. A non-numeric target, or a table with no numeric feature, is an error
/// that names the columns (#2161).
fn numeric_feature_selection(
    headers: &[String],
    numeric: &[bool],
    target: usize,
    ignored: &std::collections::BTreeSet<&String>,
) -> std::result::Result<NumericColumns, String> {
    if headers.len() != numeric.len() || target >= headers.len() {
        return Err("Target column is not in this table.".into());
    }
    if !numeric[target] {
        return Err(format!(
            "Target column `{}` is not numeric. This engine needs a numeric target.",
            headers[target]
        ));
    }
    let mut kept = Vec::new();
    let mut skipped = Vec::new();
    for (index, name) in headers.iter().enumerate() {
        if ignored.contains(&name) {
            continue;
        }
        if index == target || numeric[index] {
            kept.push(index);
        } else {
            skipped.push(name.clone());
        }
    }
    if !kept.iter().any(|&index| index != target) {
        let detail = if skipped.is_empty() {
            "there is no other column".to_owned()
        } else {
            format!(
                "these columns are not numeric: {}",
                skipped
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        return Err(format!(
            "At least one numeric feature is required. {detail}."
        ));
    }
    Ok(NumericColumns { kept, skipped })
}

fn skipped_column_warning(
    skipped: &[String],
) -> std::result::Result<jope_api::Warning, crate::warnings::WarningInputError> {
    let list = skipped
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let plural = if skipped.len() == 1 { "" } else { "s" };
    crate::warnings::catalog("NON_NUMERIC_COLUMNS_SKIPPED")
        .expect("NON_NUMERIC_COLUMNS_SKIPPED is catalogued")
        .event(
            WarningState::Raised,
            None,
            None,
            format!(
                "Ignoring non-numeric column{plural} {list}. jAIn uses numeric columns for now."
            ),
            "Remove those columns from the file, or pass --ignore-column, to silence this note.",
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(headers: &[&str]) -> Vec<String> {
        headers.iter().map(|header| (*header).to_owned()).collect()
    }

    #[test]
    fn text_feature_is_skipped_and_named() {
        let headers = names(&["region", "units", "sales"]);
        let ignored = std::collections::BTreeSet::new();
        let selection =
            numeric_feature_selection(&headers, &[false, true, true], 2, &ignored).unwrap();
        assert_eq!(selection.kept, vec![1, 2]);
        assert_eq!(selection.skipped, vec!["region".to_owned()]);
        let warning = skipped_column_warning(&selection.skipped).unwrap();
        assert_eq!(warning.code, "NON_NUMERIC_COLUMNS_SKIPPED");
        assert!(warning.message.contains("`region`"), "{}", warning.message);
    }

    #[test]
    fn text_only_features_name_the_columns() {
        let headers = names(&["region", "sales"]);
        let ignored = std::collections::BTreeSet::new();
        let error = numeric_feature_selection(&headers, &[false, true], 1, &ignored).unwrap_err();
        assert!(error.contains("`region`"), "{error}");
        assert!(error.contains("At least one numeric feature"), "{error}");
    }

    #[test]
    fn text_target_is_named() {
        let headers = names(&["units", "region"]);
        let ignored = std::collections::BTreeSet::new();
        let error = numeric_feature_selection(&headers, &[true, false], 1, &ignored).unwrap_err();
        assert!(error.contains("Target column `region`"), "{error}");
    }

    #[test]
    fn an_ignored_text_column_is_not_reported_again() {
        let headers = names(&["region", "units", "sales"]);
        let region = headers[0].clone();
        let ignored = std::collections::BTreeSet::from([&region]);
        let selection =
            numeric_feature_selection(&headers, &[false, true, true], 2, &ignored).unwrap();
        assert!(selection.skipped.is_empty());
        assert_eq!(selection.kept, vec![1, 2]);
    }

    #[test]
    fn comparators_are_off_by_default_and_fast_never_runs_them() {
        let off = resolve_comparators(&ComparatorPolicy::Off, Effort::Medium).unwrap();
        assert!(!off.enabled);
        assert_eq!(off.reason, "not requested");
        assert!(off.ids.is_empty());

        let fast_all = resolve_comparators(&ComparatorPolicy::All, Effort::Fast).unwrap();
        assert!(!fast_all.enabled);
        assert_eq!(fast_all.reason, "fast effort");
        assert!(fast_all.ids.is_empty());

        let fast_only = resolve_comparators(
            &ComparatorPolicy::Only {
                ids: vec!["xgboost_style".into()],
            },
            Effort::Fast,
        )
        .unwrap();
        assert!(!fast_only.enabled);
        assert_eq!(fast_only.reason, "fast effort");
    }

    #[test]
    fn unknown_comparator_is_refused_before_any_fit_even_at_fast() {
        let error = resolve_comparators(
            &ComparatorPolicy::Only {
                ids: vec!["not-a-model".into()],
            },
            Effort::Fast,
        )
        .unwrap_err();
        let text = format!("{error:?}");
        assert!(text.contains("not-a-model"), "{text}");
        assert!(text.contains("Unknown comparator"), "{text}");
    }

    #[test]
    fn all_and_only_at_medium_enable_the_named_roster() {
        let auto = resolve_comparators(&ComparatorPolicy::Auto, Effort::High).unwrap();
        assert!(auto.enabled);
        assert_eq!(auto.tier.as_deref(), Some("high"));

        let all = resolve_comparators(&ComparatorPolicy::All, Effort::Medium).unwrap();
        assert!(all.enabled);
        assert_eq!(all.tier.as_deref(), Some("medium"));
        assert_eq!(all.reason, "requested");
        assert_eq!(all.ids.len(), 10);
        assert!(all.ids.iter().any(|id| id == "xgboost_style"));

        let only = resolve_comparators(
            &ComparatorPolicy::Only {
                ids: vec!["random_forest".into(), "random_forest".into()],
            },
            Effort::Medium,
        )
        .unwrap();
        assert!(only.enabled);
        assert_eq!(only.ids, vec!["random_forest".to_owned()]);
    }
}
