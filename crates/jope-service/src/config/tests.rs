use super::*;

use super::*;

fn map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).into(), (*value).into()))
        .collect()
}

fn setting_value(config: &Config, key: &str) -> String {
    match key {
        "defaults.effort" => config.defaults.effort.value.clone(),
        "defaults.languages" => config.defaults.languages.value.join(","),
        "defaults.reasoning" => config.defaults.reasoning.value.clone(),
        "defaults.workers" => match config.defaults.workers.value {
            WorkerCount::Auto => "auto".into(),
            WorkerCount::Count(count) => count.to_string(),
        },
        "defaults.device" => config.defaults.device.value.clone(),
        "limits.max_dataset_bytes" => config.limits.max_dataset_bytes.value.to_string(),
        "limits.disk_reserve" => config.limits.disk_reserve.value.to_string(),
        "limits.rung0_max_rows" => config.limits.rung0_max_rows.value.to_string(),
        "limits.rung1_max_rows" => config.limits.rung1_max_rows.value.to_string(),
        "limits.eval_timeout" => format!("{}s", config.limits.eval_timeout.value.as_secs()),
        "engine.idle_exit" => format!("{}s", config.engine.idle_exit.value.as_secs()),
        "ui.color" => config.ui.color.value.clone(),
        "ui.theme" => config.ui.theme.value.clone(),
        _ => panic!("unknown config key {key}"),
    }
}

#[test]
fn defaults_match_appendix_c_and_keep_default_origins() {
    let config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert_eq!(config.defaults.effort.value, "medium");
    assert_eq!(config.defaults.effort.origin, Origin::BuiltInDefault);
    assert_eq!(config.defaults.languages.value, ["python"]);
    assert_eq!(config.defaults.reasoning.value, "concise");
    assert_eq!(config.defaults.workers.value, WorkerCount::Auto);
    assert_eq!(config.defaults.device.value, "auto");
    assert_eq!(config.limits.max_dataset_bytes.value, 4 * (1 << 30));
    assert_eq!(config.limits.disk_reserve.value, 1 << 30);
    assert_eq!(config.limits.rung0_max_rows.value, 20_000);
    assert_eq!(config.limits.rung1_max_rows.value, 100_000);
    assert_eq!(config.limits.eval_timeout.value, Duration::from_secs(30));
    assert_eq!(config.engine.idle_exit.value, Duration::from_secs(30 * 60));
    assert_eq!(config.ui.color.value, "auto");
    assert_eq!(config.ui.theme.value, "auto");
}

#[test]
fn each_key_obeys_flag_environment_file_default_precedence_and_origin() {
    let file = r#"
[defaults]
effort = "fast"
languages = ["rust"]
reasoning = "off"
workers = 2
device = "cpu"
[limits]
max_dataset_bytes = "2GiB"
disk_reserve = "2GiB"
rung0_max_rows = 10
rung1_max_rows = 20
eval_timeout = "5s"
[engine]
idle_exit = "5m"
[ui]
color = "always"
theme = "dark"
"#;
    let cases = [
        (
            "defaults.effort",
            "--effort",
            "JOPEDIME_DEFAULTS_EFFORT",
            "high",
            "fast",
        ),
        (
            "defaults.languages",
            "--languages",
            "JOPEDIME_DEFAULTS_LANGUAGES",
            "go,rust",
            "rust",
        ),
        (
            "defaults.reasoning",
            "--reasoning",
            "JOPEDIME_DEFAULTS_REASONING",
            "detailed",
            "off",
        ),
        (
            "defaults.workers",
            "--workers",
            "JOPEDIME_DEFAULTS_WORKERS",
            "4",
            "2",
        ),
        (
            "defaults.device",
            "--device",
            "JOPEDIME_DEFAULTS_DEVICE",
            "cuda",
            "cpu",
        ),
        (
            "limits.max_dataset_bytes",
            "--max-dataset-bytes",
            "JOPEDIME_LIMITS_MAX_DATASET_BYTES",
            "8GiB",
            "2147483648",
        ),
        (
            "limits.disk_reserve",
            "--disk-reserve",
            "JOPEDIME_LIMITS_DISK_RESERVE",
            "3GiB",
            "2147483648",
        ),
        (
            "limits.rung0_max_rows",
            "--rung0-max-rows",
            "JOPEDIME_LIMITS_RUNG0_MAX_ROWS",
            "30",
            "10",
        ),
        (
            "limits.rung1_max_rows",
            "--rung1-max-rows",
            "JOPEDIME_LIMITS_RUNG1_MAX_ROWS",
            "40",
            "20",
        ),
        (
            "limits.eval_timeout",
            "--eval-timeout",
            "JOPEDIME_LIMITS_EVAL_TIMEOUT",
            "10s",
            "5s",
        ),
        (
            "engine.idle_exit",
            "--idle-exit",
            "JOPEDIME_ENGINE_IDLE_EXIT",
            "10m",
            "300s",
        ),
        (
            "ui.color",
            "--color",
            "JOPEDIME_UI_COLOR",
            "never",
            "always",
        ),
        ("ui.theme", "--theme", "JOPEDIME_UI_THEME", "light", "dark"),
    ];
    for (key, flag, env, value, file_value) in cases {
        let flags = map(&[(flag, value)]);
        let environment = map(&[(env, "always-invalid-if-parsed")]);
        let config = Config::from_sources(Some(file), &flags, &environment).unwrap();
        assert_eq!(config.origin_for(key), Some(Origin::Flag), "{key}");
        assert_eq!(setting_value(&config, key), normalized(key, value), "{key}");

        let flags = BTreeMap::new();
        let environment = map(&[(env, value)]);
        let config = Config::from_sources(Some(file), &flags, &environment).unwrap();
        assert_eq!(config.origin_for(key), Some(Origin::Environment), "{key}");
        assert_eq!(setting_value(&config, key), normalized(key, value), "{key}");

        let config = Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap();
        assert_eq!(config.origin_for(key), Some(Origin::File), "{key}");
        assert_eq!(setting_value(&config, key), file_value, "{key}");

        let config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
        assert_eq!(
            config.origin_for(key),
            Some(Origin::BuiltInDefault),
            "{key}"
        );
    }
}

fn normalized(key: &str, value: &str) -> String {
    match key {
        "limits.max_dataset_bytes" | "limits.disk_reserve" => {
            parse_bytes(value).unwrap().to_string()
        }
        "limits.eval_timeout" | "engine.idle_exit" => {
            format!("{}s", parse_duration(value).unwrap().as_secs())
        }
        _ => value.to_owned(),
    }
}

#[test]
fn byte_sizes_and_durations_parse() {
    let config = Config::from_sources(
        Some("[limits]\nmax_dataset_bytes = \"4GiB\"\neval_timeout = \"30m\""),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(config.limits.max_dataset_bytes.value, 4 * (1 << 30));
    assert_eq!(
        config.limits.eval_timeout.value,
        Duration::from_secs(30 * 60)
    );
}

#[test]
fn invalid_values_name_their_key() {
    let error = Config::from_sources(
        Some("[limits]\nmax_dataset_bytes = \"four gigs\""),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("limits.max_dataset_bytes"));

    for (key, file) in [
        ("defaults.effort", "[defaults]\neffort = \"banana\""),
        ("defaults.reasoning", "[defaults]\nreasoning = \"banana\""),
        ("limits", "limits = 123"),
        ("defaults", "defaults = 123"),
        ("engine", "engine = 123"),
        ("ui", "ui = 123"),
    ] {
        let error =
            Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
        assert_eq!(error.key, key, "{file}");
        assert!(error.to_string().contains(key), "{file}");
    }
}

#[test]
fn invalid_enum_values_are_rejected_at_each_precedence_source() {
    for (key, flag, env, file) in [
        (
            "defaults.effort",
            "--effort",
            "JOPEDIME_DEFAULTS_EFFORT",
            "[defaults]\neffort = \"banana\"",
        ),
        (
            "defaults.reasoning",
            "--reasoning",
            "JOPEDIME_DEFAULTS_REASONING",
            "[defaults]\nreasoning = \"banana\"",
        ),
    ] {
        let error =
            Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
        assert_eq!(error.key, key, "{file}");

        let error =
            Config::from_sources(None, &BTreeMap::new(), &map(&[(env, "banana")])).unwrap_err();
        assert_eq!(error.key, key, "{env}");

        let error =
            Config::from_sources(None, &map(&[(flag, "banana")]), &BTreeMap::new()).unwrap_err();
        assert_eq!(error.key, key, "{flag}");
    }
}

#[test]
fn valid_flag_override_wins_over_invalid_lower_priority_file_value() {
    let file = "[defaults]\neffort = \"banana\"\nreasoning = \"banana\"";
    let flags = map(&[("--effort", "high"), ("--reasoning", "detailed")]);
    let config = Config::from_sources(Some(file), &flags, &BTreeMap::new()).unwrap();
    assert_eq!(config.defaults.effort.value, "high");
    assert_eq!(config.defaults.effort.origin, Origin::Flag);
    assert_eq!(config.defaults.reasoning.value, "detailed");
    assert_eq!(config.defaults.reasoning.origin, Origin::Flag);
}

#[test]
fn unknown_keys_warn_without_rejecting_known_values() {
    let config = Config::from_sources(
        Some("[defaults]\neffort = \"high\"\nextra = true\n[future]\nvalue = 1"),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(config.defaults.effort.value, "high");
    assert_eq!(config.warnings.len(), 2);
    assert!(config
        .warnings
        .iter()
        .any(|warning| warning.message.contains("defaults.extra")));
    assert!(config
        .warnings
        .iter()
        .any(|warning| warning.message.contains("future")));
}
