//! Parsing for the terminal's registry-backed command line.

use std::fmt;

use jope_api::registry::{terminal_command, TerminalCommand};

/// A parsed command and its arguments. The registry entry is canonical even
/// when the user entered an alias or the slash prefix.
#[derive(Clone, Debug)]
pub struct ParsedCommand {
    /// The canonical registry record for the entered command or alias.
    pub command: &'static TerminalCommand,
    /// Tokenized arguments, excluding the command name and `--json` flag.
    pub arguments: Vec<String>,
    /// Whether the engine response should be emitted as raw JSON.
    pub json: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The input contains no command text.
    EmptyInput,
    /// The first non-whitespace character is not a supported command prefix.
    MissingPrefix,
    /// The input contains a prefix but no command name.
    MissingCommandName,
    /// The registry has no command or alias with the entered name.
    UnknownCommand(String),
    /// A quoted argument has no matching closing delimiter.
    UnterminatedQuote(char),
    /// An inline objective object is not valid JSON.
    InvalidObjectiveJson(String),
    /// `--json` was used with a client-only or draft-only command.
    JsonNotSupported,
    /// `--json` was repeated.
    DuplicateJsonFlag,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("empty command"),
            Self::MissingPrefix => f.write_str("command must begin with `\\` or `/`"),
            Self::MissingCommandName => f.write_str("missing command name after prefix"),
            Self::UnknownCommand(name) => write!(f, "unknown command `{name}`"),
            Self::UnterminatedQuote(quote) => {
                write!(f, "unterminated {quote} quote in command arguments")
            }
            Self::InvalidObjectiveJson(detail) => write!(f, "invalid objective JSON: {detail}"),
            Self::JsonNotSupported => {
                f.write_str("`--json` is only valid for engine-backed commands")
            }
            Self::DuplicateJsonFlag => f.write_str("`--json` may only be specified once"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse one terminal command. Argument words use single or double quotes;
/// backslashes in unquoted arguments are ordinary characters, so Windows paths
/// remain intact. Inline objective JSON remains one unchanged argument.
pub fn parse(input: &str) -> Result<ParsedCommand, ParseError> {
    let input = input.trim_start();
    if input.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    let prefix = input.chars().next().expect("nonempty input");
    if prefix != '\\' && prefix != '/' {
        return Err(ParseError::MissingPrefix);
    }

    let after_prefix = &input[prefix.len_utf8()..];
    let name_end = after_prefix
        .find(char::is_whitespace)
        .unwrap_or(after_prefix.len());
    if name_end == 0 {
        return Err(ParseError::MissingCommandName);
    }

    let entered_name = &input[..prefix.len_utf8() + name_end];
    let command = terminal_command(entered_name)
        .ok_or_else(|| ParseError::UnknownCommand(entered_name.to_owned()))?;
    let mut arguments = tokenize(&after_prefix[name_end..], command.name == r"\objective")?;

    let mut json = false;
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] != "--json" {
            index += 1;
            continue;
        }
        if command.engine_operations.is_empty() {
            return Err(ParseError::JsonNotSupported);
        }
        if json {
            return Err(ParseError::DuplicateJsonFlag);
        }
        json = true;
        arguments.remove(index);
    }

    Ok(ParsedCommand {
        command,
        arguments,
        json,
    })
}

pub(crate) fn tokenize(input: &str, objective_json: bool) -> Result<Vec<String>, ParseError> {
    let mut arguments = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut quote = None;
    let mut chars = input.char_indices().peekable();

    while let Some((offset, ch)) = chars.next() {
        if objective_json && quote.is_none() && !started && ch == '{' {
            let mut values = serde_json::Deserializer::from_str(&input[offset..])
                .into_iter::<serde::de::IgnoredAny>();
            values
                .next()
                .expect("an opening brace starts a JSON value")
                .map_err(|error| ParseError::InvalidObjectiveJson(error.to_string()))?;
            let end = offset + values.byte_offset();
            while chars.peek().is_some_and(|(index, _)| *index < end) {
                chars.next();
            }
            if chars.peek().is_some_and(|(_, ch)| !ch.is_whitespace()) {
                return Err(ParseError::InvalidObjectiveJson(
                    "separate arguments after the object with whitespace".to_owned(),
                ));
            }
            arguments.push(input[offset..end].to_owned());
            continue;
        }
        match quote {
            Some(delimiter) if ch == delimiter => quote = None,
            Some('"') if ch == '\\' => match chars.peek().map(|(_, ch)| *ch) {
                Some('"' | '\\') => word.push(chars.next().expect("peeked character").1),
                _ => word.push(ch),
            },
            Some(_) => word.push(ch),
            None if ch.is_whitespace() => {
                if started {
                    arguments.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            None if ch == '\'' || ch == '"' => {
                quote = Some(ch);
                started = true;
            }
            None => {
                word.push(ch);
                started = true;
            }
        }
    }

    if let Some(delimiter) = quote {
        return Err(ParseError::UnterminatedQuote(delimiter));
    }
    if started {
        arguments.push(word);
    }
    Ok(arguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::registry::terminal_commands;

    #[test]
    fn every_registry_command_resolves_with_both_prefixes() {
        let commands = terminal_commands().collect::<Vec<_>>();
        assert_eq!(commands.len(), 24);

        for command in commands {
            for prefix in ['\\', '/'] {
                let entered = format!("{prefix}{}", &command.name[1..]);
                let parsed = parse(&entered).unwrap_or_else(|error| panic!("{entered}: {error}"));
                assert_eq!(parsed.command.name, command.name, "{entered}");
                assert!(parsed.arguments.is_empty(), "{entered}");
                assert!(!parsed.json, "{entered}");
            }

            for alias in command.aliases {
                for prefix in ['\\', '/'] {
                    let entered = format!("{prefix}{}", &alias[1..]);
                    let parsed =
                        parse(&entered).unwrap_or_else(|error| panic!("{entered}: {error}"));
                    assert_eq!(parsed.command.name, command.name, "{entered}");
                }
            }
        }
    }

    #[test]
    fn quoted_arguments_and_windows_paths_are_preserved() {
        let parsed = parse(r#"\open "C:\data\sales file.csv" --name 'Q1 sample'"#).unwrap();
        assert_eq!(
            parsed.arguments,
            [r"C:\data\sales file.csv", "--name", "Q1 sample"]
        );

        let parsed = parse(r"\open C:\data\x.csv").unwrap();
        assert_eq!(parsed.arguments, [r"C:\data\x.csv"]);
    }

    #[test]
    fn file_objective_argument_is_not_expanded_or_split() {
        let parsed = parse(r"\objective @objective.json").unwrap();
        assert_eq!(parsed.arguments, ["@objective.json"]);

        let parsed = parse(r#"\objective "@objectives/Q1 objective.json""#).unwrap();
        assert_eq!(parsed.arguments, ["@objectives/Q1 objective.json"]);
    }

    #[test]
    fn inline_json_objectives_preserve_syntax_for_all_aliases() {
        let payload = r#"{
  "name": "rmse", "nested": {"label": "温度 \"quoted\"", "weights": [1, 2]}
}"#;
        for command in [r"\objective", "/objective", r"\fitness", "/fitness"] {
            let parsed = parse(&format!("{command} {payload}")).unwrap();
            assert_eq!(parsed.command.name, r"\objective");
            assert_eq!(parsed.arguments, [payload]);
            assert!(!parsed.json);
        }
    }

    #[test]
    fn json_output_flag_does_not_change_inline_objective_strings() {
        let payload = r#"{"name":"--json","nested":{"value":"two words"}}"#;
        for input in [
            format!(r"\objective --json {payload}"),
            format!(r"\objective {payload} --json"),
        ] {
            let parsed = parse(&input).unwrap();
            assert_eq!(parsed.arguments, [payload]);
            assert!(parsed.json);
        }
    }

    #[test]
    fn malformed_inline_objective_json_has_a_specific_error() {
        for payload in [r#"{"name":"#, r#"{name:"rmse"}"#, "{}suffix"] {
            let error = parse(&format!(r"\objective {payload}")).unwrap_err();
            assert!(matches!(error, ParseError::InvalidObjectiveJson(_)));
            assert!(error.to_string().starts_with("invalid objective JSON:"));
        }
    }

    #[test]
    fn json_flag_is_extracted_only_for_engine_commands() {
        let parsed = parse(r"\status RUN-1 --json").unwrap();
        assert!(parsed.json);
        assert_eq!(parsed.arguments, ["RUN-1"]);
    }

    #[test]
    fn errors_have_stable_messages() {
        let cases = [
            ("", "empty command"),
            ("open data.csv", "command must begin with `\\` or `/`"),
            ("\\", "missing command name after prefix"),
            ("\\unknown", "unknown command `\\unknown`"),
            (
                "\\open 'unfinished",
                "unterminated ' quote in command arguments",
            ),
            (
                "\\help --json",
                "`--json` is only valid for engine-backed commands",
            ),
            (
                "\\status --json --json",
                "`--json` may only be specified once",
            ),
        ];

        for (input, expected) in cases {
            let error = parse(input).expect_err(input);
            assert_eq!(error.to_string(), expected, "{input:?}");
        }
    }
}
