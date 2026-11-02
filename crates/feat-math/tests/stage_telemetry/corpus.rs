use super::*;

#[test]
fn the_program_corpus_is_compat_freezes() {
    let named: Vec<String> = NAMED_PROGRAMS
        .iter()
        .map(|(name, source)| format!("    ({name:?}, {source:?}),"))
        .collect();
    assert_eq!(named, compat_freeze_entries("NAMED_PROGRAMS"));
    for (copy, name) in [
        (IP_SCRUB_PIPELINES, "IP_SCRUB_PIPELINES"),
        (IP_SCRUB_SYMBOLIC, "IP_SCRUB_SYMBOLIC"),
    ] {
        let lines: Vec<String> = copy
            .iter()
            .map(|source| format!("    {source:?},"))
            .collect();
        assert_eq!(lines, compat_freeze_entries(name), "{name}");
    }
    let fixture = fixture_json("compat_freeze/model_serde.v1.json");
    let models = fixture["models"].as_array().expect("models");
    let recorded: Vec<&str> = models
        .iter()
        .map(|model| model["name"].as_str().expect("name"))
        .collect();
    let set = program_set();
    let names: Vec<&str> = set.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, recorded);
    // The fixture records exactly the programs skipped here as skipped.
    for ((name, source), model) in set.iter().zip(models) {
        let skipped = model["serde_json"]
            .as_str()
            .expect("serde_json")
            .starts_with("skipped:");
        assert_eq!(skipped, fit_skipped(source), "{name}");
    }
    assert_eq!(set.len(), 55);
}
