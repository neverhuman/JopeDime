use super::*;

use super::{complete, cycle_index, split_before_cursor, CompletionData, CompletionSource};

fn temp_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("jope-completion-")
        .tempdir()
        .expect("create test directory")
}

#[test]
fn command_completion_includes_canonical_commands_and_aliases() {
    let suggestions = complete("\\ta", 3, &CompletionData::default());
    assert!(suggestions.iter().any(|item| item.insert == "\\target"));

    let suggestions = complete("\\fit", 4, &CompletionData::default());
    assert!(suggestions.iter().any(|item| item.insert == "\\fitness"));

    let suggestions = complete("\\help tar", 9, &CompletionData::default());
    assert!(suggestions.iter().any(|item| item.insert == "target"));
}

#[test]
fn flags_and_enum_values_are_contextual() {
    for (line, expected) in [
        ("\\open --n", "--name"),
        ("\\target --t", "--task"),
        ("\\effort --t", "--time"),
    ] {
        let suggestions = complete(line, line.len(), &CompletionData::default());
        assert!(
            suggestions.iter().any(|item| item.insert == expected),
            "{line}"
        );
        assert!(suggestions
            .iter()
            .all(|item| item.source == CompletionSource::Flag));
    }

    let suggestions = complete("\\run --", 7, &CompletionData::default());
    assert!(suggestions.iter().any(|item| item.insert == "--seed"));
    assert!(suggestions
        .iter()
        .all(|item| item.source == CompletionSource::Flag));

    let suggestions = complete("\\effort f", 9, &CompletionData::default());
    assert_eq!(suggestions[0].insert, "fast");

    let suggestions = complete("\\landmarks n", 12, &CompletionData::default());
    assert_eq!(suggestions[0].insert, "none");

    let suggestions = complete("\\target --task b", 16, &CompletionData::default());
    assert_eq!(suggestions[0].insert, "binary");

    let line = "\\reasoning d";
    let suggestions = complete(line, line.len(), &CompletionData::default());
    assert_eq!(suggestions[0].insert, "detailed");

    let line = "\\export --kind so";
    let suggestions = complete(line, line.len(), &CompletionData::default());
    assert_eq!(suggestions[0].insert, "source");

    let suggestions = complete("\\lang rust,p", 12, &CompletionData::default());
    assert!(suggestions.iter().any(|item| item.insert == "rust,python"));
}

#[test]
fn column_run_and_algorithm_sources_use_supplied_context() {
    let columns = vec![
        "age".to_owned(),
        "income".to_owned(),
        "median price".to_owned(),
        "region".to_owned(),
    ];
    let run_ids = vec!["run_01ABC".to_owned()];
    let algorithm_ids = vec!["alg_0007".to_owned()];
    let data = CompletionData {
        columns: &columns,
        run_ids: &run_ids,
        algorithm_ids: &algorithm_ids,
        ..CompletionData::default()
    };

    let suggestions = complete("\\target med", 11, &data);
    assert_eq!(suggestions[0].insert, "\"median price\"");
    for line in [
        "\\target --ignore med",
        "\\target --group med",
        "\\target --time med",
    ] {
        let suggestions = complete(line, line.len(), &data);
        assert_eq!(suggestions[0].insert, "\"median price\"", "{line}");
        assert_eq!(suggestions[0].source, CompletionSource::Column, "{line}");
    }
    let line = "\\target target --ignore age,inc";
    let suggestions = complete(line, line.len(), &data);
    assert_eq!(suggestions[0].insert, "age,income");
    assert_eq!(suggestions[0].source, CompletionSource::Column);
    let suggestions = complete("\\status run_", 12, &data);
    assert_eq!(suggestions[0].source, CompletionSource::RunId);
    let suggestions = complete("\\attach run_", 12, &data);
    assert_eq!(suggestions[0].source, CompletionSource::RunId);
    let suggestions = complete("\\diff alg_", 10, &data);
    assert_eq!(suggestions[0].source, CompletionSource::AlgorithmId);
    let suggestions = complete("\\code rust alg_", 15, &data);
    assert_eq!(suggestions[0].source, CompletionSource::AlgorithmId);
    let suggestions = complete("\\export --alg alg_", 18, &data);
    assert_eq!(suggestions[0].source, CompletionSource::AlgorithmId);
    let line = "\\predict --model alg_";
    let suggestions = complete(line, line.len(), &data);
    assert_eq!(suggestions[0].source, CompletionSource::AlgorithmId);
}

#[test]
fn path_completion_is_relative_and_quotes_spaces_and_unicode() {
    let temp = temp_dir();
    let folder = temp.path().join("données space");
    std::fs::create_dir_all(&folder).expect("create unicode path");
    let file = folder.join("house price.csv");
    let dot_file = temp.path().join("data.csv");
    std::fs::write(&file, "").expect("create path candidate");
    std::fs::write(&dot_file, "").expect("create ./ path candidate");
    let paths = vec![file, dot_file];
    let data = CompletionData {
        paths: &paths,
        working_directory: Some(temp.path()),
        ..CompletionData::default()
    };

    let suggestions = complete("\\open données", "\\open données".len(), &data);
    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].insert, "\"données space/house price.csv\"");
    assert_eq!(suggestions[0].source, CompletionSource::Path);

    let suggestions = complete("\\open @données", "\\open @données".len(), &data);
    assert_eq!(suggestions[0].insert, "@\"données space/house price.csv\"");

    for line in [
        "\\open ./dat",
        "\\predict --input ./dat",
        "\\export --out ./dat",
    ] {
        let suggestions = complete(line, line.len(), &data);
        assert!(
            suggestions.iter().any(|item| item.insert == "./data.csv"),
            "{line}: {suggestions:?}"
        );
    }

    for line in [
        "\\predict --input données",
        "\\predict --output données",
        "\\export --out données",
    ] {
        let suggestions = complete(line, line.len(), &data);
        assert_eq!(
            suggestions[0].insert, "\"données space/house price.csv\"",
            "{line}"
        );
        assert_eq!(suggestions[0].source, CompletionSource::Path, "{line}");
    }
}

#[test]
fn quoted_path_suggestions_round_trip_quotes_and_backslashes() {
    let temp = temp_dir();
    let quoted = temp.path().join("data \"v2\".csv");
    let backslash = temp.path().join("folder\\name space.csv");
    std::fs::write(&quoted, "").expect("create quoted path candidate");
    std::fs::write(&backslash, "").expect("create backslash path candidate");
    let paths = vec![quoted, backslash];
    let data = CompletionData {
        paths: &paths,
        working_directory: Some(temp.path()),
        ..CompletionData::default()
    };

    let initial = complete("\\open ", "\\open ".len(), &data);
    assert!(initial
        .iter()
        .any(|item| item.insert == "\"data \\\"v2\\\".csv\""));
    assert!(initial
        .iter()
        .any(|item| item.insert == "\"folder\\\\name space.csv\""));

    for suggestion in initial {
        let command = format!("\\open {}", suggestion.insert);
        let round_trip = complete(&command, command.len(), &data);
        assert!(
            round_trip
                .iter()
                .any(|item| item.insert == suggestion.insert),
            "{} -> {round_trip:?}",
            suggestion.insert
        );
    }
}

#[test]
fn completion_only_returns_values_and_never_executes_input() {
    let temp = temp_dir();
    let marker = temp.path().join("should-not-exist");
    let suggestions = complete("\\run ", 5, &CompletionData::default());
    assert!(suggestions
        .iter()
        .all(|item| item.source == CompletionSource::Flag));
    assert!(!marker.exists());
}

#[test]
fn shift_tab_cycles_backward_and_wraps() {
    assert_eq!(cycle_index(None, 3, true), Some(2));
    assert_eq!(cycle_index(Some(0), 3, true), Some(2));
    assert_eq!(cycle_index(Some(1), 3, true), Some(0));
    assert_eq!(cycle_index(Some(0), 3, false), Some(1));
    assert_eq!(cycle_index(None, 0, true), None);
}

#[test]
fn windows_backslashes_remain_argument_characters() {
    let (previous, current) = split_before_cursor(r"\open C:\data\x.csv");
    assert_eq!(previous, vec!["\\open".to_owned()]);
    assert_eq!(current, r"C:\data\x.csv");
}
