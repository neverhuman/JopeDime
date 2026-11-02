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
        ValueSource::Column => complete_columns(prefix, data),
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

fn complete_columns(prefix: &str, data: &CompletionData<'_>) -> Vec<Suggestion> {
    let (list_prefix, active_prefix) = prefix
        .rsplit_once(',')
        .map(|(list_prefix, active_prefix)| {
            let selected = list_prefix
                .split(',')
                .map(serialize_token)
                .collect::<Vec<_>>()
                .join(",");
            (format!("{selected},"), active_prefix)
        })
        .unwrap_or_else(|| (String::new(), prefix));
    data.columns
        .iter()
        .filter(|column| column.starts_with(active_prefix))
        .map(|column| Suggestion {
            insert: format!("{list_prefix}{}", serialize_token(column)),
            source: CompletionSource::Column,
        })
        .collect()
}

fn complete_paths(prefix: &str, marker: &str, data: &CompletionData<'_>) -> Vec<Suggestion> {
    let Some(cwd) = data.working_directory else {
        return Vec::new();
    };
    let mut match_prefix = prefix;
    while let Some(remainder) = match_prefix.strip_prefix("./") {
        match_prefix = remainder;
    }
    let spelling_prefix = &prefix[..prefix.len() - match_prefix.len()];
    data.paths
        .iter()
        .filter_map(|path| {
            let relative = path.strip_prefix(cwd).ok()?;
            let relative = relative.to_string_lossy();
            relative.starts_with(match_prefix).then(|| Suggestion {
                insert: format!(
                    "{marker}{}",
                    serialize_token(&format!("{spelling_prefix}{relative}"))
                ),
                source: CompletionSource::Path,
            })
        })
        .collect()
}

fn serialize_token(value: &str) -> String {
    if !value
        .chars()
        .any(|character| character.is_whitespace() || matches!(character, '\'' | '"'))
    {
        return value.to_owned();
    }

    let mut serialized = String::with_capacity(value.len() + 2);
    serialized.push('"');
    for character in value.chars() {
        if matches!(character, '\\' | '"') {
            serialized.push('\\');
        }
        serialized.push(character);
    }
    serialized.push('"');
    serialized
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
    let mut characters = input.chars().peekable();

    while let Some(character) = characters.next() {
        match (quote, character) {
            (Some('"'), '\\') => match characters.peek().copied() {
                Some('"' | '\\') => {
                    current.push(characters.next().expect("peeked escape character"));
                    started = true;
                }
                _ => {
                    current.push('\\');
                    started = true;
                }
            },
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
mod tests;
