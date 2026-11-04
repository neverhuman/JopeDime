use super::*;

/// Why [`realpath`] or [`resolve`] could not resolve a path.
#[derive(Debug)]
pub enum ResolveError {
    /// `os.lstat` (strict mode) or `os.readlink` raised for `path`.
    Os {
        error: std::io::Error,
        path: PathBuf,
    },
    /// A symlink loop: pathlib raised `RuntimeError("Symlink loop from '<path>'")`.
    Loop { path: PathBuf },
    /// The working directory a relative path is resolved against could not be read:
    /// `os.getcwd()` raised (`[Errno 2] No such file or directory` for a deleted one).
    Cwd(std::io::Error),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Os { error, path } => f.write_str(&os_error_path(error, path)),
            ResolveError::Cwd(error) => f.write_str(&os_error_bare(error)),
            ResolveError::Loop { path } => {
                write!(f, "Symlink loop from {}", os_repr(path.as_os_str()))
            }
        }
    }
}

/// `os.path.realpath(path, strict=strict)`, CPython 3.12's `posixpath._joinrealpath` component by
/// component: each name is `lstat`ed and a symlink is replaced by its target, resolved in turn,
/// **before** a later `..` removes the last component of the path resolved so far, so
/// `outer/link/../x` with `link -> guarded/sub` is `guarded/x`, not the lexical `outer/x`.
/// Empty and `.` components are skipped and a leading `//` is `/` (the path is resolved from the
/// root). In strict mode a missing component (any `lstat` error) refuses naming the path up to it,
/// as Python's `OSError` did; otherwise the name is kept as it is spelled and resolution goes on.
/// `os.readlink` errors refuse in both modes. A symlink met again while it is being resolved is a
/// loop: in strict mode that refuses ([`ResolveError::Loop`], pathlib's `RuntimeError`); otherwise
/// the rest of the path is appended unresolved, as Python did. The result is [`abspath`] of the
/// resolved path (relative input is resolved against the working directory).
pub fn realpath(path: &Path, strict: bool) -> std::result::Result<PathBuf, ResolveError> {
    use std::collections::HashMap;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    fn os(bytes: &[u8]) -> &Path {
        Path::new(OsStr::from_bytes(bytes))
    }
    /// `posixpath.join(a, b)`.
    fn join(a: &[u8], b: &[u8]) -> Vec<u8> {
        if b.starts_with(b"/") {
            return b.to_vec();
        }
        let mut out = a.to_vec();
        if !(out.is_empty() || out.ends_with(b"/")) {
            out.push(b'/');
        }
        out.extend_from_slice(b);
        out
    }
    /// `posixpath.split(p)`.
    fn split(p: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let i = p.iter().rposition(|&b| b == b'/').map_or(0, |i| i + 1);
        let (head, tail) = p.split_at(i);
        let mut head = head.to_vec();
        if !head.is_empty() && head.iter().any(|&b| b != b'/') {
            while head.ends_with(b"/") {
                head.pop();
            }
        }
        (head, tail.to_vec())
    }
    type Seen = HashMap<Vec<u8>, Option<Vec<u8>>>;
    /// `_joinrealpath(path, rest, strict, seen)`: the resolved path and whether it is complete.
    fn join_realpath(
        mut path: Vec<u8>,
        rest: &[u8],
        strict: bool,
        seen: &mut Seen,
    ) -> std::result::Result<(Vec<u8>, bool), ResolveError> {
        let mut rest = rest;
        if rest.starts_with(b"/") {
            rest = &rest[1..];
            path = b"/".to_vec();
        }
        while !rest.is_empty() {
            let (name, tail) = match rest.iter().position(|&b| b == b'/') {
                Some(i) => (&rest[..i], &rest[i + 1..]),
                None => (rest, &b""[..]),
            };
            rest = tail;
            if name.is_empty() || name == b"." {
                continue;
            }
            if name == b".." {
                if path.is_empty() {
                    path = b"..".to_vec();
                } else {
                    let (head, last) = split(&path);
                    path = head;
                    if last == b".." {
                        path = join(&join(&path, b".."), b"..");
                    }
                }
                continue;
            }
            let newpath = join(&path, name);
            let is_link = match std::fs::symlink_metadata(os(&newpath)) {
                Ok(meta) => meta.file_type().is_symlink(),
                Err(error) if strict => {
                    return Err(ResolveError::Os {
                        error,
                        path: os(&newpath).to_path_buf(),
                    })
                }
                Err(_) => false,
            };
            if !is_link {
                path = newpath;
                continue;
            }
            match seen.get(&newpath) {
                Some(Some(resolved)) => {
                    path = resolved.clone();
                    continue;
                }
                Some(None) if strict => {
                    return Err(ResolveError::Loop {
                        path: os(&newpath).to_path_buf(),
                    })
                }
                Some(None) => return Ok((join(&newpath, rest), false)),
                None => {}
            }
            seen.insert(newpath.clone(), None);
            let target = std::fs::read_link(os(&newpath)).map_err(|error| ResolveError::Os {
                error,
                path: os(&newpath).to_path_buf(),
            })?;
            let (resolved, ok) = join_realpath(path, target.as_os_str().as_bytes(), strict, seen)?;
            path = resolved;
            if !ok {
                return Ok((join(&path, rest), false));
            }
            seen.insert(newpath, Some(path.clone()));
        }
        Ok((path, true))
    }
    let (resolved, _) = join_realpath(
        vec![],
        path.as_os_str().as_bytes(),
        strict,
        &mut HashMap::new(),
    )?;
    abspath(os(&resolved)).map_err(ResolveError::Cwd)
}

/// `Path(path).resolve(strict=strict)` (Python 3.12): [`realpath`], and in non-strict mode a
/// `stat` of the result, so a symlink loop refuses in both modes ([`ResolveError::Loop`], the
/// path `stat` named) while any other `stat` error is ignored, as pathlib ignored it. Every port
/// that resolves a path for a safety decision (custody, roots, "inside the tree") resolves it
/// here or with `std::fs::canonicalize`, which is the same resolution in strict mode; none
/// normalises `..` lexically before symlinks are resolved.
pub fn resolve(path: &Path, strict: bool) -> std::result::Result<PathBuf, ResolveError> {
    const ELOOP: i32 = if cfg!(target_os = "macos") { 62 } else { 40 };
    let resolved = realpath(path, strict)?;
    if !strict {
        if let Err(error) = std::fs::metadata(&resolved) {
            if error.raw_os_error() == Some(ELOOP) {
                return Err(ResolveError::Loop { path: resolved });
            }
        }
    }
    Ok(resolved)
}

/// `re.fullmatch("[0-9a-f]{n}", value or "")` for a JSON value (a non-string never matches).
pub fn is_hex(value: Option<&Value>, len: usize) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// `PurePosixPath(text)`: its root (`/`, or `//` exactly as POSIX keeps it) and its parts
/// without empty or `.` components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosixPath {
    pub root: Option<&'static str>,
    pub parts: Vec<String>,
}

impl PosixPath {
    pub fn parse(text: &str) -> Self {
        let root = if text.starts_with("//") && !text.starts_with("///") {
            Some("//")
        } else if text.starts_with('/') {
            Some("/")
        } else {
            None
        };
        let parts = text
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned)
            .collect();
        PosixPath { root, parts }
    }

    pub fn is_absolute(&self) -> bool {
        self.root.is_some()
    }

    /// `".." in path.parts`.
    pub fn has_parent_reference(&self) -> bool {
        self.parts.iter().any(|part| part == "..")
    }

    /// `base / path` for a relative path.
    pub fn join_onto(&self, base: &Path) -> PathBuf {
        let mut out = base.to_path_buf();
        for part in &self.parts {
            out.push(part);
        }
        out
    }

    /// Whether the relative parts begin with `prefix`.
    pub fn starts_with(&self, prefix: &[&str]) -> bool {
        self.root.is_none()
            && self.parts.len() >= prefix.len()
            && self.parts.iter().zip(prefix).all(|(a, b)| a == b)
    }

    /// `self / other`: an absolute `other` replaces the path, as in Python.
    pub fn join(&self, other: &str) -> Self {
        self.joinpath(&PosixPath::parse(other))
    }

    /// `self / other` for a parsed path: an absolute `other` (either root) replaces the path.
    pub fn joinpath(&self, other: &PosixPath) -> Self {
        if other.is_absolute() {
            return other.clone();
        }
        let mut parts = self.parts.clone();
        parts.extend(other.parts.iter().cloned());
        PosixPath {
            root: self.root,
            parts,
        }
    }

    /// `path.parent`: the path without its final part (a root, or `.`, is its own parent).
    pub fn parent(&self) -> Self {
        let mut parts = self.parts.clone();
        parts.pop();
        PosixPath {
            root: self.root,
            parts,
        }
    }

    /// `self.relative_to(other)` (Python 3.12, `walk_up=False`): the rest of the path when `other`
    /// is the path itself or one of its parents. pathlib compares them as spelled after parsing,
    /// so the roots must be the same (`/`, POSIX's distinct `//`, or none; `///` is `/`) and so
    /// must the leading parts. Otherwise the `ValueError` message, both paths as `str()` prints
    /// them.
    pub fn relative_to(&self, other: &PosixPath) -> std::result::Result<Self, String> {
        let n = other.parts.len();
        if self.root == other.root && self.parts.len() >= n && self.parts[..n] == other.parts[..] {
            return Ok(PosixPath {
                root: None,
                parts: self.parts[n..].to_vec(),
            });
        }
        Err(format!(
            "{} is not in the subpath of {}",
            pyjson::str_repr(&self.to_string()),
            pyjson::str_repr(&other.to_string())
        ))
    }

    /// `path.name`: the final component, or `""`.
    pub fn name(&self) -> &str {
        self.parts.last().map_or("", String::as_str)
    }

    /// `path.suffix` (Python 3.12): the final component's extension, if it has one that is not
    /// its whole name.
    pub fn suffix(&self) -> &str {
        let name = self.name();
        match name.rfind('.') {
            Some(i) if 0 < i && i < name.len() - 1 => &name[i..],
            _ => "",
        }
    }
}

/// `str(PurePosixPath(...))`.
impl std::fmt::Display for PosixPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self.parts.join("/");
        match (self.root, body.is_empty()) {
            (Some(root), _) => write!(f, "{root}{body}"),
            (None, true) => f.write_str("."),
            (None, false) => f.write_str(&body),
        }
    }
}
