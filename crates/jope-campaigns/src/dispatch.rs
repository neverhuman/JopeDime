//! One binary, many programs.
//!
//! The 55 former `src/bin/*.rs` programs (plus the former root binary
//! `jope-dime-regressor`) are compiled into two binaries, `jope-campaigns` and
//! `jope-selfplay`. Each binary carries a [`Table`] of [`Program`]s and picks
//! one to run in two ways:
//!
//! * **argv\[0\]** — when the binary is invoked through a symlink (or hard
//!   link) named after a program, e.g. `@JOPE_BUNDLE@/jope_selfplay_queue
//!   --once` from a systemd unit, the program named by the file name runs and
//!   sees exactly the argv it always saw (`deploy/systemd/BINARIES.md` lists
//!   the links; `deploy/systemd/render.sh --install-symlinks <bindir>` creates
//!   them);
//! * **first argument** — `jope-selfplay jope_selfplay_queue --once` runs the
//!   same program with argv\[0\] rewritten to the program name, so `--help`
//!   and error messages keep the historical name.
//!
//! With no program the binary prints its program list (`--help`) or the bare
//! names, one per line (`--list`, consumed by `render.sh --install-symlinks`).

use std::path::Path;

/// One former `src/bin` program.
pub struct Program {
    /// The historical binary name (the file stem under `src/bin`).
    pub name: &'static str,
    /// One line for the program list.
    pub about: &'static str,
    /// The program's former `fn main`, given its full argv (argv\[0\] = program name).
    pub run: fn(Vec<String>) -> anyhow::Result<()>,
}

/// The program table of one binary.
pub struct Table {
    /// The binary's own name (`jope-campaigns` / `jope-selfplay`).
    pub binary: &'static str,
    /// One line describing the binary.
    pub about: &'static str,
    pub programs: &'static [Program],
}

/// The file stem of an invoked path (`/opt/jope/bin/jope_selfplay_queue` ->
/// `jope_selfplay_queue`; a Windows-style `.exe` suffix is dropped too).
pub fn program_name(argv0: &str) -> String {
    let stem = Path::new(argv0)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(argv0);
    stem.strip_suffix(".exe").unwrap_or(stem).to_owned()
}

impl Table {
    /// The program called `name`; `-` and `_` are interchangeable so that
    /// `search-v3` and `search_v3` both resolve.
    pub fn find(&self, name: &str) -> Option<&'static Program> {
        let wanted = name.replace('-', "_");
        self.programs
            .iter()
            .find(|program| program.name == name)
            .or_else(|| {
                self.programs
                    .iter()
                    .find(|program| program.name.replace('-', "_") == wanted)
            })
    }

    /// Every program name, in table order.
    pub fn names(&self) -> Vec<&'static str> {
        self.programs.iter().map(|program| program.name).collect()
    }

    /// The `--help` text: the binary, how to invoke a program, and the list.
    pub fn usage(&self) -> String {
        let width = self
            .programs
            .iter()
            .map(|program| program.name.len())
            .max()
            .unwrap_or(0);
        let column = self.binary.len() + 17;
        let mut text = format!(
            "{binary} — {about}\n\nusage: {form:<column$}\n       {link:<column$} (through a link named after the program)\n       {list:<column$} (program names only, one per line)\n\n{count} programs:\n",
            binary = self.binary,
            about = self.about,
            form = format!("{} <program> [args…]", self.binary),
            link = "<program> [args…]",
            list = format!("{} --list", self.binary),
            count = self.programs.len()
        );
        for program in self.programs {
            text.push_str(&format!(
                "  {name:<width$}  {about}\n",
                name = program.name,
                about = program.about
            ));
        }
        text
    }

    /// Dispatch a full argv (argv\[0\] included) and run the chosen program.
    pub fn run(&self, argv: Vec<String>) -> anyhow::Result<()> {
        let invoked = argv.first().map(|argv0| program_name(argv0));
        if let Some(program) = invoked.as_deref().and_then(|name| self.find(name)) {
            return (program.run)(argv);
        }
        match argv.get(1).map(String::as_str) {
            None | Some("--help") | Some("-h") | Some("help") => {
                print!("{}", self.usage());
                Ok(())
            }
            Some("--list") => {
                for name in self.names() {
                    println!("{name}");
                }
                Ok(())
            }
            Some("--version") | Some("-V") => {
                println!("{} {}", self.binary, env!("CARGO_PKG_VERSION"));
                Ok(())
            }
            Some(name) => match self.find(name) {
                Some(program) => {
                    let mut program_argv = Vec::with_capacity(argv.len() - 1);
                    program_argv.push(program.name.to_owned());
                    program_argv.extend(argv.into_iter().skip(2));
                    (program.run)(program_argv)
                }
                None => anyhow::bail!(
                    "{}: unknown program `{name}`; run `{} --help` for the list",
                    self.binary,
                    self.binary
                ),
            },
        }
    }

    /// The binary's `fn main`: dispatch the process argv.
    pub fn main(&self) -> anyhow::Result<()> {
        self.run(std::env::args().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello(argv: Vec<String>) -> anyhow::Result<()> {
        anyhow::ensure!(argv[0] == "hello_prog", "argv0 was {}", argv[0]);
        anyhow::ensure!(argv.get(1).map(String::as_str) == Some("--flag"));
        Ok(())
    }

    fn fail(_argv: Vec<String>) -> anyhow::Result<()> {
        anyhow::bail!("boom")
    }

    static TABLE: Table = Table {
        binary: "demo",
        about: "a test table",
        programs: &[
            Program {
                name: "hello_prog",
                about: "says hello",
                run: hello,
            },
            Program {
                name: "fail-prog",
                about: "fails",
                run: fail,
            },
        ],
    };

    #[test]
    fn argv0_file_name_selects_the_program() {
        TABLE
            .run(vec!["/opt/bin/hello_prog".into(), "--flag".into()])
            .unwrap();
        TABLE
            .run(vec!["hello_prog.exe".into(), "--flag".into()])
            .unwrap();
    }

    #[test]
    fn first_argument_selects_the_program_and_becomes_argv0() {
        TABLE
            .run(vec!["demo".into(), "hello_prog".into(), "--flag".into()])
            .unwrap();
        let error = TABLE
            .run(vec!["demo".into(), "fail_prog".into()])
            .unwrap_err();
        assert_eq!(error.to_string(), "boom");
    }

    #[test]
    fn unknown_program_is_an_error_and_help_forms_are_not() {
        let error = TABLE.run(vec!["demo".into(), "nope".into()]).unwrap_err();
        assert!(error.to_string().contains("unknown program `nope`"));
        TABLE.run(vec!["demo".into()]).unwrap();
        TABLE.run(vec!["demo".into(), "--help".into()]).unwrap();
        TABLE.run(vec!["demo".into(), "--list".into()]).unwrap();
    }

    #[test]
    fn usage_lists_every_program_once() {
        let usage = TABLE.usage();
        assert!(usage.starts_with("demo — a test table\n"));
        assert!(usage.contains("2 programs:\n"));
        assert_eq!(usage.matches("hello_prog").count(), 1);
        assert_eq!(usage.matches("fail-prog").count(), 1);
        assert_eq!(TABLE.names(), vec!["hello_prog", "fail-prog"]);
        assert_eq!(program_name("/a/b/x.exe"), "x");
    }
}
