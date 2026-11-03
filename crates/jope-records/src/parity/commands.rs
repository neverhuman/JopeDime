use super::*;

pub fn kebab(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::new();
    for (i, ch) in chars.iter().enumerate() {
        if i > 0 && ch.is_uppercase() {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            if prev.is_lowercase()
                || prev.is_numeric()
                || (prev.is_uppercase() && next.is_some_and(char::is_lowercase))
            {
                out.push('-');
            }
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// A top-level variant of `enum <name> { ... }`.
#[derive(Clone, Debug)]
pub struct Variant {
    pub variant: String,
    pub name: String,
    pub neural: bool,
    pub hidden: bool,
}

/// Top-level variants of `enum <enum_name> { ... }` with their outer attributes. The source is read
/// as `read_text(errors="replace")` read it (universal newlines, so a CRLF file parses), and the
/// patterns use Python's classes ([`py::regex`]).
pub fn parse_clap_enum(path: &Path, enum_name: &str) -> Result<Vec<Variant>> {
    let src = py::read_text_replace(path)?;
    let block = py::regex(&format!(
        r"(?s)enum {} \{{\n(.*?)\n\}}\n",
        regex::escape(enum_name)
    ));
    let body = block.captures(&src).ok_or_else(|| {
        Error::Shape(format!(
            "{}: enum {enum_name} not found",
            py::os_display(path.as_os_str())
        ))
    })?[1]
        .to_owned();
    let attr = py::regex(r"^    #\[");
    // Python treats U+001C through U+001F as whitespace; Rust's Unicode `\s` class does not.
    let variant = py::regex(r"^    ([A-Z][A-Za-z0-9]*)[\s\x1c-\x1f]*(\{|,|\()");
    let named = py::regex(r#"name\s*=\s*"([^"]+)""#);
    let mut rows = vec![];
    let mut attrs: Vec<String> = vec![];
    for line in py::splitlines(&body) {
        if attr.is_match(line) {
            attrs.push(py::strip(line).to_owned());
            continue;
        }
        if let Some(m) = variant.captures(line) {
            let variant = m[1].to_owned();
            let mut name = kebab(&variant);
            for a in &attrs {
                if let Some(n) = named.captures(a) {
                    if a.starts_with("#[command") {
                        name = n[1].to_owned();
                    }
                }
            }
            rows.push(Variant {
                variant,
                name,
                neural: attrs.iter().any(|a| a.contains(r#"feature = "neural""#)),
                hidden: attrs.iter().any(|a| a.contains("hide = true")),
            });
            attrs.clear();
        }
    }
    Ok(rows)
}

/// Subcommand names from clap `--help` output (the `Commands:` block).
pub fn parse_help_commands(text: &str) -> Vec<String> {
    let name = py::regex(r"^  ([a-z0-9][a-z0-9-]*)(\s|$)");
    let mut names = vec![];
    let mut inside = false;
    for line in py::splitlines(text) {
        if py::strip(line) == "Commands:" {
            inside = true;
            continue;
        }
        if inside {
            if py::strip(line).is_empty() || line.starts_with("Options") {
                if !names.is_empty() {
                    break;
                }
                continue;
            }
            if let Some(m) = name.captures(line) {
                if &m[1] != "help" {
                    names.push(m[1].to_owned());
                }
            }
        }
    }
    names
}

/// Command paths (`data build`, `train`, ...) from an argparse CLI built with `add_parser`.
pub fn parse_argparse_commands(path: &Path) -> Result<Vec<String>> {
    Ok(argparse_commands(&py::read_text_replace(path)?))
}

/// [`parse_argparse_commands`] for the source text.
pub fn argparse_commands(src: &str) -> Vec<String> {
    let subparsers = py::regex(r"(\w+)\s*=\s*(\w+)\.add_subparsers\(");
    let parsers = py::regex(r#"(\w+)\s*=\s*(\w+)\.add_parser\(\s*"([^"]+)""#);
    let owner: BTreeMap<String, String> = subparsers
        .captures_iter(src)
        .map(|m| (m[1].to_owned(), m[2].to_owned()))
        .collect();
    let mut command_of: BTreeMap<String, String> = BTreeMap::new();
    let mut out = vec![];
    for m in parsers.captures_iter(src) {
        let (var, subs, name) = (&m[1], &m[2], &m[3]);
        let prefix = owner
            .get(subs)
            .and_then(|parent| command_of.get(parent))
            .cloned()
            .unwrap_or_default();
        let full = py::strip(&format!("{prefix} {name}")).to_owned();
        command_of.insert(var.to_owned(), full.clone());
        out.push(full);
    }
    out
}
