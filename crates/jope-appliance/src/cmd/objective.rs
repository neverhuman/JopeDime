pub(crate) fn describe_spec_with_options(
    text: &str,
    task: Option<&str>,
    classes: &[String],
) -> anyhow::Result<serde_json::Value> {
    if let Some(t) = task {
        anyhow::ensure!(
            matches!(t, "regression" | "binary" | "multiclass"),
            "objective --task must be regression, binary or multiclass, got {t:?}"
        );
    }

    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        let parsed_val = serde_json::from_str::<serde_json::Value>(trimmed).ok();
        let schema_str = parsed_val
            .as_ref()
            .and_then(|v| v.get("schema"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let class_len = parsed_val
            .as_ref()
            .and_then(|v| v.get("class_order"))
            .and_then(serde_json::Value::as_array)
            .map(Vec::len);

        if task == Some("multiclass")
            || schema_str.starts_with("jope.multiclass-fitness/")
            || class_len.is_some_and(|n| n >= 3)
        {
            let fitness = if classes.is_empty() {
                jope_runtime::fitness_multiclass::MulticlassFitness::parse(text)?
            } else {
                jope_runtime::fitness_multiclass::MulticlassFitness::parse_for_classes(
                    text,
                    classes.to_vec(),
                )?
            };
            return jope_runtime::service::describe_multiclass(&fitness);
        }

        if task == Some("binary")
            || schema_str.starts_with("jope.classification-fitness/")
            || class_len == Some(2)
        {
            let fitness = if classes.is_empty() {
                jope_runtime::fitness_classification::ClassificationFitness::parse(text)?
            } else {
                let pair: [String; 2] = classes.to_vec().try_into().map_err(|_| {
                    anyhow::anyhow!("binary classification requires exactly 2 --classes")
                })?;
                jope_runtime::fitness_classification::ClassificationFitness::parse_for_classes(
                    text, pair,
                )?
            };
            return jope_runtime::service::describe_classification(&fitness);
        }

        anyhow::ensure!(
            classes.is_empty(),
            "regression objectives do not accept --classes"
        );
        let fitness = jope_runtime::fitness::Fitness::parse(text)?;
        return jope_runtime::service::describe(&fitness);
    }

    if task == Some("multiclass") || classes.len() >= 3 {
        let resolved_classes = if classes.is_empty() {
            vec!["0".to_owned(), "1".to_owned(), "2".to_owned()]
        } else {
            classes.to_vec()
        };
        let fitness = jope_runtime::fitness_multiclass::MulticlassFitness::from_shorthand(
            trimmed,
            resolved_classes,
        )?;
        return jope_runtime::service::describe_multiclass(&fitness);
    }

    if task == Some("binary") || classes.len() == 2 {
        let pair: [String; 2] = if classes.is_empty() {
            ["0".to_owned(), "1".to_owned()]
        } else {
            classes.to_vec().try_into().expect("checked len == 2")
        };
        let fitness = jope_runtime::fitness_classification::ClassificationFitness::from_shorthand(
            trimmed, pair,
        )?;
        return jope_runtime::service::describe_classification(&fitness);
    }

    anyhow::ensure!(
        classes.is_empty(),
        "classification --classes requires at least 2 classes (2 for binary, >= 3 for multiclass)"
    );

    if task != Some("regression") {
        let base_name = trimmed.split(':').next().unwrap_or_default().trim();
        if jope_runtime::fitness_multiclass::Metric::is_multiclass_only_shorthand(base_name) {
            let fitness = jope_runtime::fitness_multiclass::MulticlassFitness::from_shorthand(
                trimmed,
                vec!["0".to_owned(), "1".to_owned(), "2".to_owned()],
            )?;
            return jope_runtime::service::describe_multiclass(&fitness);
        }
        if jope_runtime::fitness_classification::Metric::from_shorthand_name(base_name).is_some() {
            let fitness =
                jope_runtime::fitness_classification::ClassificationFitness::from_shorthand(
                    trimmed,
                    ["0".to_owned(), "1".to_owned()],
                )?;
            return jope_runtime::service::describe_classification(&fitness);
        }
    }

    let fitness = jope_runtime::fitness::Fitness::parse(text)?;
    jope_runtime::service::describe(&fitness)
}

pub(crate) fn run(
    spec: &str,
    task: Option<&str>,
    classes: &[String],
    json_out: bool,
) -> anyhow::Result<()> {
    let text = if let Some(path) = spec.strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        spec.to_owned()
    };
    let value = describe_spec_with_options(&text, task, classes)?;
    if json_out {
        super::dataset::emit(&value, true)
    } else {
        println!(
            "{} ({})",
            value["name"].as_str().unwrap_or("objective"),
            value["fitness_sha256"].as_str().unwrap_or("")
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::describe_spec_with_options;

    fn describe_spec(text: &str) -> anyhow::Result<serde_json::Value> {
        describe_spec_with_options(text, None, &[])
    }

    #[test]
    fn builtin_objective_description_is_stable_json() {
        let value = describe_spec("mae").unwrap();
        assert_eq!(value["name"], "mae");
        assert_eq!(value["fitness_sha256"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn binary_shorthands_and_json_specs_are_described() {
        for (spec, expected_name, expected_dir) in [
            ("roc_auc", "roc_auc", "maximize"),
            ("auroc", "roc_auc", "maximize"),
            ("logloss", "logloss", "minimize"),
            ("brier", "brier", "minimize"),
            ("pr_auc", "pr_auc", "maximize"),
            ("balanced_accuracy", "balanced_accuracy", "maximize"),
            ("f1:threshold=0.35", "f1", "maximize"),
        ] {
            let value = describe_spec(spec).unwrap();
            assert_eq!(
                value["schema"],
                jope_runtime::fitness_classification::FITNESS_VERSION
            );
            assert_eq!(value["task"], "binary");
            assert_eq!(value["name"], expected_name);
            assert_eq!(value["direction"], expected_dir);
            assert_eq!(value["class_order"], serde_json::json!(["0", "1"]));
            assert_eq!(value["fitness_sha256"].as_str().unwrap().len(), 64);
        }

        let json_spec = r#"{
            "schema": "jope.classification-fitness/v1",
            "metric": "cost_matrix_loss",
            "class_order": ["legit", "fraud"],
            "aggregation": "mean",
            "undefined": "fail_candidate",
            "cost_matrix": [[0.0, 1.0], [10.0, 0.0]]
        }"#;
        let described = describe_spec(json_spec).unwrap();
        assert_eq!(described["task"], "binary");
        assert_eq!(described["name"], "cost_matrix_loss");
        assert_eq!(described["direction"], "minimize");
        assert_eq!(
            described["class_order"],
            serde_json::json!(["legit", "fraud"])
        );

        let err = describe_spec("f1:threshold=1.5").unwrap_err().to_string();
        assert!(err.contains("threshold"), "{err}");
    }

    #[test]
    fn multiclass_shorthands_explicit_classes_and_json_specs_are_described() {
        for (spec, expected_name, expected_dir) in [
            ("macro_f1", "macro_f1", "maximize"),
            ("accuracy", "accuracy", "maximize"),
        ] {
            let value = describe_spec(spec).unwrap();
            assert_eq!(
                value["schema"],
                jope_runtime::fitness_multiclass::FITNESS_VERSION
            );
            assert_eq!(value["task"], "multiclass");
            assert_eq!(value["name"], expected_name);
            assert_eq!(value["direction"], expected_dir);
            assert_eq!(value["class_order"], serde_json::json!(["0", "1", "2"]));
            assert_eq!(value["fitness_sha256"].as_str().unwrap().len(), 64);
        }

        let classes = vec!["red".into(), "green".into(), "blue".into()];
        let logloss = describe_spec_with_options("logloss", Some("multiclass"), &classes).unwrap();
        assert_eq!(logloss["task"], "multiclass");
        assert_eq!(logloss["name"], "logloss");
        assert_eq!(logloss["direction"], "minimize");
        assert_eq!(
            logloss["class_order"],
            serde_json::json!(["red", "green", "blue"])
        );

        let json_spec = r#"{
            "schema": "jope.multiclass-fitness/v1",
            "metric": "cost_matrix_loss",
            "class_order": ["low", "med", "high"],
            "aggregation": "mean",
            "undefined": "fail_candidate",
            "cost_matrix": [[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [5.0, 2.0, 0.0]]
        }"#;
        let described = describe_spec(json_spec).unwrap();
        assert_eq!(described["task"], "multiclass");
        assert_eq!(described["name"], "cost_matrix_loss");
        assert_eq!(
            described["class_order"],
            serde_json::json!(["low", "med", "high"])
        );
    }

    #[test]
    fn cli_objective_validate_accepts_task_and_classes_flags() {
        use clap::Parser;
        let parsed = crate::cli::Cli::try_parse_from([
            "jopedime",
            "objective",
            "validate",
            "--task",
            "multiclass",
            "--classes",
            "red,green,blue",
            "logloss",
            "--json",
        ])
        .unwrap();
        let crate::cli::ApplianceCommand::Objective {
            command:
                crate::cli::ObjectiveCommand::Validate {
                    spec,
                    task,
                    classes,
                    json,
                },
        } = parsed.command
        else {
            panic!("expected Objective Validate subcommand");
        };
        assert_eq!(spec, "logloss");
        assert_eq!(task.as_deref(), Some("multiclass"));
        assert_eq!(classes, vec!["red", "green", "blue"]);
        assert!(json);
    }
}
