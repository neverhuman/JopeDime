//! Pure, context-driven suggestions for the inline terminal composer.

use std::path::{Path, PathBuf};

/// Category of completion candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionSource {
    Command,
    Flag,
    Enum,
    Column,
    RunId,
    AlgorithmId,
    Path,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
    /// Text to replace the current token with.
    pub insert: String,
    /// Kind of data that produced this suggestion.
    pub source: CompletionSource,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompletionData<'a> {
    /// Names from the currently drafted dataset.
    pub columns: &'a [String],
    /// Run identifiers visible in the current store.
    pub run_ids: &'a [String],
    /// Algorithm identifiers visible in the current run.
    pub algorithm_ids: &'a [String],
    /// Filesystem entries already enumerated by the caller.
    pub paths: &'a [PathBuf],
    /// Used to render path suggestions relative to the working directory.
    pub working_directory: Option<&'a Path>,
}

/// Canonical command and flag entries shared by terminal parsing and completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandSpec {
    /// Command spelling without its leading `\\` or `/`.
    pub name: &'static str,
    /// Flags accepted by the command.
    pub flags: &'static [&'static str],
}

/// Canonical terminal command and flag vocabulary for parser/completion use.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        flags: &[],
    },
    CommandSpec {
        name: "open",
        flags: &["--name", "--json"],
    },
    CommandSpec {
        name: "target",
        flags: &["--task", "--ignore", "--group", "--time"],
    },
    CommandSpec {
        name: "objective",
        flags: &["--json"],
    },
    CommandSpec {
        name: "fitness",
        flags: &["--json"],
    },
    CommandSpec {
        name: "effort",
        flags: &["--evals", "--time"],
    },
    CommandSpec {
        name: "landmarks",
        flags: &[],
    },
    CommandSpec {
        name: "lang",
        flags: &["--best-effort"],
    },
    CommandSpec {
        name: "language",
        flags: &["--best-effort"],
    },
    CommandSpec {
        name: "plan",
        flags: &["--json"],
    },
    CommandSpec {
        name: "run",
        flags: &["--seed", "--json"],
    },
    CommandSpec {
        name: "pause",
        flags: &["--json"],
    },
    CommandSpec {
        name: "continue",
        flags: &["--json"],
    },
    CommandSpec {
        name: "cancel",
        flags: &["--json"],
    },
    CommandSpec {
        name: "restart",
        flags: &["--same-seed", "--new-seed", "--warm", "--json"],
    },
    CommandSpec {
        name: "status",
        flags: &["--json"],
    },
    CommandSpec {
        name: "metrics",
        flags: &["--detailed", "--json"],
    },
    CommandSpec {
        name: "reasoning",
        flags: &[],
    },
    CommandSpec {
        name: "code",
        flags: &["--json"],
    },
    CommandSpec {
        name: "diff",
        flags: &["--json"],
    },
    CommandSpec {
        name: "export",
        flags: &["--kind", "--lang", "--alg", "--out", "--force", "--json"],
    },
    CommandSpec {
        name: "predict",
        flags: &["--input", "--output", "--model", "--json"],
    },
    CommandSpec {
        name: "runs",
        flags: &["--all", "--json"],
    },
    CommandSpec {
        name: "attach",
        flags: &["--json"],
    },
    CommandSpec {
        name: "doctor",
        flags: &["--json"],
    },
    CommandSpec {
        name: "quit",
        flags: &["--pause"],
    },
];

const TASKS: &[&str] = &["regression", "binary", "multiclass"];
const EFFORTS: &[&str] = &["fast", "medium", "high", "ultra"];
const LANDMARKS: &[&str] = &["full", "none"];
const LANGUAGES: &[&str] = &["rust", "python", "javascript", "typescript"];
const REASONING: &[&str] = &["off", "concise", "detailed"];
const EXPORT_KINDS: &[&str] = &["bundle", "model", "source"];

#[derive(Clone, Copy)]
enum ValueSource {
    Command,
    Enum(&'static [&'static str]),
    Languages,
    Column,
    RunId,
    AlgorithmId,
    Path,
    AlgorithmOrPath,
}

/// Return suggestions for the token at `cursor` without invoking any command.
///
/// `paths` must be entries enumerated by the caller. Keeping enumeration out of
/// this function makes completion deterministic and side-effect free.
pub fn complete(input: &str, cursor: usize, data: &CompletionData<'_>) -> Vec<Suggestion> {
    if cursor > input.len() || !input.is_char_boundary(cursor) {
        return Vec::new();
    }

    let (mut previous, prefix) = split_before_cursor(&input[..cursor]);
    if previous.is_empty() {
        return complete_commands(&prefix);
    }

    let raw_command = previous.remove(0);
    let command_name = raw_command
        .trim_start_matches(['\\', '/'])
        .to_ascii_lowercase();
    let Some(command) = COMMANDS
        .iter()
        .find(|command| command.name == command_name.as_str())
    else {
        return Vec::new();
    };

    if prefix.starts_with('-') {
        return complete_flags(command, &previous, &prefix);
    }

    if let Some(path_prefix) = prefix.strip_prefix('@') {
        return complete_paths(path_prefix, "@", data);
    }

    let value_source = previous
        .last()
        .and_then(|last| value_source_for_flag(command_name.as_str(), last))
        .or_else(|| positional_source(command_name.as_str(), previous.len(), &prefix));

    if let Some(source) = value_source {
        return complete_source(source, &prefix, data);
    }

    if prefix.is_empty() {
        return complete_flags(command, &previous, &prefix);
    }

    Vec::new()
}

/// Pick the next item in completion order. Shift-Tab uses `backwards = true`.
pub fn cycle_index(current: Option<usize>, count: usize, backwards: bool) -> Option<usize> {
    if count == 0 {
        return None;
    }
    match current {
        None if backwards => Some(count - 1),
        None => Some(0),
        Some(index) if backwards => Some((index % count + count - 1) % count),
        Some(index) => Some((index + 1) % count),
    }
}

fn complete_commands(prefix: &str) -> Vec<Suggestion> {
    let marker = if prefix.starts_with('/') { "/" } else { "\\" };
    let search = prefix.trim_start_matches(['\\', '/']);
    COMMANDS
        .iter()
        .map(|command| format!("{marker}{}", command.name))
        .filter(|candidate| candidate[1..].starts_with(search))
        .map(|insert| Suggestion {
            insert,
            source: CompletionSource::Command,
        })
        .collect()
}

fn complete_flags(command: &CommandSpec, used: &[String], prefix: &str) -> Vec<Suggestion> {
    command
        .flags
        .iter()
        .filter(|flag| !used.iter().any(|arg| arg.as_str() == **flag))
        .filter(|flag| flag.starts_with(prefix))
        .map(|flag| Suggestion {
            insert: (*flag).to_owned(),
            source: CompletionSource::Flag,
        })
        .collect()
}

fn value_source_for_flag(command: &str, flag: &str) -> Option<ValueSource> {
    match (command, flag) {
        ("target", "--task") => Some(ValueSource::Enum(TASKS)),
        ("target", "--ignore" | "--group" | "--time") => Some(ValueSource::Column),
        ("effort", "--evals" | "--time") | ("run", "--seed") => None,
        ("open", "--name") => None,
        ("export", "--kind") => Some(ValueSource::Enum(EXPORT_KINDS)),
        ("export", "--lang") => Some(ValueSource::Languages),
        ("export", "--alg") => Some(ValueSource::AlgorithmId),
        ("export", "--out") | ("predict", "--input" | "--output") => Some(ValueSource::Path),
        ("predict", "--model") => Some(ValueSource::AlgorithmOrPath),
        _ => None,
    }
}

fn positional_source(command: &str, completed_args: usize, prefix: &str) -> Option<ValueSource> {
    if matches!(command, "open") && completed_args == 0 {
        return Some(ValueSource::Path);
    }
    if prefix.starts_with('@') {
        return Some(ValueSource::Path);
    }
    match command {
        "help" if completed_args == 0 => Some(ValueSource::Command),
        "target" if completed_args == 0 => Some(ValueSource::Column),
        "effort" if completed_args == 0 => Some(ValueSource::Enum(EFFORTS)),
        "landmarks" if completed_args == 0 => Some(ValueSource::Enum(LANDMARKS)),
        "lang" | "language" if completed_args == 0 => Some(ValueSource::Languages),
        "reasoning" if completed_args == 0 => Some(ValueSource::Enum(REASONING)),
        "status" | "attach" if completed_args == 0 => Some(ValueSource::RunId),
        "code" if completed_args == 0 => Some(ValueSource::Languages),
        "code" if completed_args == 1 => Some(ValueSource::AlgorithmId),
        "diff" if completed_args < 2 => Some(ValueSource::AlgorithmId),
        _ => None,
    }
}

fn complete_source(
    source: ValueSource,
    prefix: &str,
    data: &CompletionData<'_>,
) -> Vec<Suggestion> {
    match source {
        ValueSource::Command => filtered(
            COMMANDS.iter().map(|command| command.name),
            prefix,
            CompletionSource::Command,
        ),
        ValueSource::Enum(values) => {
            filtered(values.iter().copied(), prefix, CompletionSource::Enum)
        }
        ValueSource::Languages => complete_languages(prefix),
        ValueSource::Column => filtered(
            data.columns.iter().map(String::as_str),
            prefix,
            CompletionSource::Column,
        ),
        ValueSource::RunId => filtered(
            data.run_ids.iter().map(String::as_str),
            prefix,
            CompletionSource::RunId,
        ),
        ValueSource::AlgorithmId => filtered(
            data.algorithm_ids.iter().map(String::as_str),
            prefix,
            CompletionSource::AlgorithmId,
        ),
        ValueSource::Path => complete_paths(prefix, "", data),
        ValueSource::AlgorithmOrPath => {
            let mut suggestions = complete_source(ValueSource::AlgorithmId, prefix, data);
            suggestions.extend(complete_paths(prefix, "", data));
            suggestions
        }
    }
}

fn complete_languages(prefix: &str) -> Vec<Suggestion> {
    let (base, search) = prefix
        .rfind(',')
        .map(|index| (&prefix[..=index], &prefix[index + 1..]))
        .unwrap_or(("", prefix));
    LANGUAGES
        .iter()
        .filter(|language| language.starts_with(search))
        .map(|language| Suggestion {
            insert: format!("{base}{language}"),
            source: CompletionSource::Enum,
        })
        .collect()
}

fn complete_paths(prefix: &str, marker: &str, data: &CompletionData<'_>) -> Vec<Suggestion> {
    let Some(cwd) = data.working_directory else {
        return Vec::new();
    };
    data.paths
        .iter()
        .filter_map(|path| {
            let relative = path.strip_prefix(cwd).ok()?;
            let relative = relative.to_string_lossy();
            relative.starts_with(prefix).then(|| Suggestion {
                insert: format!("{marker}{}", quote_path(&relative)),
                source: CompletionSource::Path,
            })
        })
        .collect()
}

fn quote_path(path: &str) -> String {
    if path.chars().any(char::is_whitespace) {
        format!("\"{path}\"")
    } else {
        path.to_owned()
    }
}

fn filtered<'a>(
    values: impl Iterator<Item = &'a str>,
    prefix: &str,
    source: CompletionSource,
) -> Vec<Suggestion> {
    values
        .filter(|value| value.starts_with(prefix))
        .map(|value| Suggestion {
            insert: value.to_owned(),
            source,
        })
        .collect()
}

/// Split the already-typed prefix into completed words and the current word.
/// Quotes group path names containing spaces; backslashes outside quotes remain
/// ordinary characters so Windows paths are not treated as command escapes.
fn split_before_cursor(input: &str) -> (Vec<String>, String) {
    let mut completed = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut started = false;

    for character in input.chars() {
        match (quote, character) {
            (None, '"' | '\'') => {
                quote = Some(character);
                started = true;
            }
            (Some(active), character) if character == active => quote = None,
            (None, character) if character.is_whitespace() => {
                if started {
                    completed.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            _ => {
                current.push(character);
                started = true;
            }
        }
    }

    if started {
        (completed, current)
    } else {
        (completed, String::new())
    }
}

#[cfg(test)]
mod tests {
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
        let columns = vec!["median price".to_owned(), "region".to_owned()];
        let run_ids = vec!["run_01ABC".to_owned()];
        let algorithm_ids = vec!["alg_0007".to_owned()];
        let data = CompletionData {
            columns: &columns,
            run_ids: &run_ids,
            algorithm_ids: &algorithm_ids,
            ..CompletionData::default()
        };

        let suggestions = complete("\\target med", 11, &data);
        assert_eq!(suggestions[0].insert, "median price");
        for line in [
            "\\target --ignore med",
            "\\target --group med",
            "\\target --time med",
        ] {
            let suggestions = complete(line, line.len(), &data);
            assert_eq!(suggestions[0].insert, "median price", "{line}");
            assert_eq!(suggestions[0].source, CompletionSource::Column, "{line}");
        }
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
        std::fs::write(&file, "").expect("create path candidate");
        let paths = vec![file];
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
}
