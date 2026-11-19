pub(crate) fn describe_spec(text: &str) -> anyhow::Result<serde_json::Value> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        let looks_like_classification = serde_json::from_str::<serde_json::Value>(trimmed)
            .ok()
            .is_some_and(|value| {
                value.get("class_order").is_some()
                    || value
                        .get("schema")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|s| s.starts_with("jope.classification-fitness/"))
            });
        if looks_like_classification {
            let fitness = jope_runtime::fitness_classification::ClassificationFitness::parse(text)?;
            return jope_runtime::service::describe_classification(&fitness);
        }
        let fitness = jope_runtime::fitness::Fitness::parse(text)?;
        return jope_runtime::service::describe(&fitness);
    }

    let base_name = trimmed.split(':').next().unwrap_or_default().trim();
    if jope_runtime::fitness_classification::Metric::from_shorthand_name(base_name).is_some() {
        let fitness = jope_runtime::fitness_classification::ClassificationFitness::from_shorthand(
            trimmed,
            ["0".to_owned(), "1".to_owned()],
        )?;
        return jope_runtime::service::describe_classification(&fitness);
    }

    let fitness = jope_runtime::fitness::Fitness::parse(text)?;
    jope_runtime::service::describe(&fitness)
}

pub(crate) fn run(spec: &str, json_out: bool) -> anyhow::Result<()> {
    let text = if let Some(path) = spec.strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        spec.to_owned()
    };
    let value = describe_spec(&text)?;
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
    use super::describe_spec;

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
}
