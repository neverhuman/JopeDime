//! Renders the `deploy/systemd/**/*.in` unit templates.
//!
//! A template is a systemd unit whose site-specific roots are the four
//! placeholders in [`PLACEHOLDERS`]: `@JOPE_HOME@` (the source checkout),
//! `@JOPE_STATE@` (mutable run state), `@JOPE_BUNDLE@` (the directory the
//! built binaries are installed in) and `@JOPE_USER@` (the account the unit
//! runs as). Rendering is plain substitution — no escaping, no conditionals —
//! so `deploy/systemd/render.sh` and this module produce identical bytes, and
//! any other `@NAME@` token is an error rather than a unit that starts with a
//! literal `@` in an `ExecStart=`.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The placeholder names, without their `@` delimiters.
pub const PLACEHOLDERS: [&str; 4] = ["JOPE_HOME", "JOPE_STATE", "JOPE_BUNDLE", "JOPE_USER"];

/// Suffix that marks a file as a template.
pub const TEMPLATE_SUFFIX: &str = ".in";

/// Values for the four placeholders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Roots {
    /// Source checkout used by the unit's working directory and paths.
    pub home: PathBuf,
    /// Mutable state directory used by the service.
    pub state: PathBuf,
    /// Directory containing the installed service binaries.
    pub bundle: PathBuf,
    /// Account name assigned to the rendered unit.
    pub user: String,
}

impl Roots {
    /// The substitution for `@name@`, or `None` for an unknown name.
    pub fn value(&self, name: &str) -> Option<String> {
        match name {
            "JOPE_HOME" => Some(super::to_string(&self.home)),
            "JOPE_STATE" => Some(super::to_string(&self.state)),
            "JOPE_BUNDLE" => Some(super::to_string(&self.bundle)),
            "JOPE_USER" => Some(self.user.clone()),
            _ => None,
        }
    }
}

/// Why a template could not be rendered.
#[derive(Debug)]
pub enum RenderError {
    /// `@NAME@` appeared and `NAME` is not one of [`PLACEHOLDERS`].
    UnknownPlaceholder { file: PathBuf, name: String },
    /// A placeholder value is empty, which would splice `//` or `User=` into a unit.
    EmptyValue { name: &'static str },
    /// Reading a template or writing its rendering failed.
    Io { path: PathBuf, source: io::Error },
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPlaceholder { file, name } => write!(
                f,
                "{}: unknown placeholder @{name}@ (known: {})",
                file.display(),
                PLACEHOLDERS.join(", ")
            ),
            Self::EmptyValue { name } => write!(f, "placeholder {name} has an empty value"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for RenderError {}

/// Substitutes every `@NAME@` in `template`. `file` only labels errors.
pub fn render(template: &str, roots: &Roots, file: &Path) -> Result<String, RenderError> {
    for name in PLACEHOLDERS {
        if roots.value(name).is_some_and(|value| value.is_empty()) {
            return Err(RenderError::EmptyValue { name });
        }
    }
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find('@') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let token_end = after
            .find(|c: char| !(c.is_ascii_uppercase() || c == '_'))
            .unwrap_or(after.len());
        let name = &after[..token_end];
        if !name.is_empty() && after[token_end..].starts_with('@') {
            match roots.value(name) {
                Some(value) => out.push_str(&value),
                None => {
                    return Err(RenderError::UnknownPlaceholder {
                        file: file.to_path_buf(),
                        name: name.to_owned(),
                    })
                }
            }
            rest = &after[token_end + 1..];
        } else {
            // A bare '@' (systemd's ExecStart=@… argv[0] form, e-mail
            // addresses in comments) is ordinary text.
            out.push('@');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Every `*.in` file below `templates`, in sorted order, as paths relative to it.
pub fn templates_under(templates: &Path) -> Result<Vec<PathBuf>, RenderError> {
    fn walk(dir: &Path, base: &Path, into: &mut Vec<PathBuf>) -> Result<(), RenderError> {
        let entries = fs::read_dir(dir).map_err(|source| RenderError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| RenderError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.is_dir() {
                walk(&path, base, into)?;
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(TEMPLATE_SUFFIX))
            {
                into.push(path.strip_prefix(base).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(templates, templates, &mut found)?;
    found.sort();
    Ok(found)
}

/// Renders every template below `templates` into `out`, keeping the relative
/// directory (`selfplay/x.service.in` → `<out>/selfplay/x.service`). Returns
/// the rendered paths in sorted order.
pub fn render_tree(
    templates: &Path,
    out: &Path,
    roots: &Roots,
) -> Result<Vec<PathBuf>, RenderError> {
    let mut rendered = Vec::new();
    for relative in templates_under(templates)? {
        let source = templates.join(&relative);
        let text = fs::read_to_string(&source).map_err(|source_error| RenderError::Io {
            path: source.clone(),
            source: source_error,
        })?;
        let unit = render(&text, roots, &source)?;
        let target_name = relative
            .to_string_lossy()
            .trim_end_matches(TEMPLATE_SUFFIX)
            .to_owned();
        let target = out.join(target_name);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|source| RenderError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::write(&target, unit).map_err(|source| RenderError::Io {
            path: target.clone(),
            source,
        })?;
        rendered.push(target);
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots() -> Roots {
        Roots {
            home: PathBuf::from("/opt/jope/checkout"),
            state: PathBuf::from("/var/lib/jope"),
            bundle: PathBuf::from("/opt/jope/bundle/bin"),
            user: "jope".into(),
        }
    }

    #[test]
    fn substitutes_all_four_and_leaves_bare_at_signs() {
        let unit = "[Service]\nUser=@JOPE_USER@\nWorkingDirectory=@JOPE_STATE@\nExecStart=@JOPE_BUNDLE@/tool --root @JOPE_STATE@ --spec @JOPE_HOME@/m.json\n# mail me@example.org, ExecStart=@/bin/x\n";
        let out = render(unit, &roots(), Path::new("t.in")).unwrap();
        assert_eq!(
            out,
            "[Service]\nUser=jope\nWorkingDirectory=/var/lib/jope\nExecStart=/opt/jope/bundle/bin/tool --root /var/lib/jope --spec /opt/jope/checkout/m.json\n# mail me@example.org, ExecStart=@/bin/x\n"
        );
    }

    #[test]
    fn unknown_placeholder_is_an_error() {
        let err = render("X=@JOPE_NOPE@\n", &roots(), Path::new("t.in")).unwrap_err();
        assert!(
            matches!(err, RenderError::UnknownPlaceholder { ref name, .. } if name == "JOPE_NOPE"),
            "{err}"
        );
    }

    #[test]
    fn empty_value_is_an_error() {
        let mut r = roots();
        r.user = String::new();
        let err = render("User=@JOPE_USER@\n", &r, Path::new("t.in")).unwrap_err();
        assert!(
            matches!(err, RenderError::EmptyValue { name: "JOPE_USER" }),
            "{err}"
        );
    }

    #[test]
    fn renders_a_tree_keeping_subdirectories() {
        let dir = std::env::temp_dir().join(format!("jope-units-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("in/selfplay")).unwrap();
        fs::write(
            dir.join("in/selfplay/a.service.in"),
            "ExecStart=@JOPE_BUNDLE@/a\n",
        )
        .unwrap();
        fs::write(dir.join("in/selfplay/a.timer.in"), "[Timer]\nOnBootSec=1\n").unwrap();
        fs::write(dir.join("in/README.md"), "not a template\n").unwrap();
        let rendered = render_tree(&dir.join("in"), &dir.join("out"), &roots()).unwrap();
        assert_eq!(
            rendered,
            vec![
                dir.join("out/selfplay/a.service"),
                dir.join("out/selfplay/a.timer")
            ]
        );
        assert_eq!(
            fs::read_to_string(dir.join("out/selfplay/a.service")).unwrap(),
            "ExecStart=/opt/jope/bundle/bin/a\n"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
