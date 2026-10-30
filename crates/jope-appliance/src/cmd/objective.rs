pub(crate) fn run(spec: &str, json_out: bool) -> anyhow::Result<()> {
    let text = if let Some(path) = spec.strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        spec.to_owned()
    };
    let fitness = jope_runtime::fitness::Fitness::parse(&text)?;
    let value = jope_runtime::service::describe(&fitness)?;
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
    #[test]
    fn builtin_objective_description_is_stable_json() {
        let fitness = jope_runtime::fitness::Fitness::parse("mae").unwrap();
        let value = jope_runtime::service::describe(&fitness).unwrap();
        assert_eq!(value["name"], "mae");
        assert_eq!(value["fitness_sha256"].as_str().unwrap().len(), 64);
    }
}
