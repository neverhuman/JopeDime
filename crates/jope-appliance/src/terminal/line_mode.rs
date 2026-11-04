//! Plain transcript output: no raw mode, cursor addressing, colors, or redraws.

use std::io::{self, Write};

use super::sanitize_user_string;

/// Resolve the fallback before constructing a terminal backend.
pub fn required(plain: bool, term: Option<&str>, no_color: bool, width: u16) -> bool {
    plain || no_color || term.is_some_and(|term| term.eq_ignore_ascii_case("dumb")) || width < 40
}

/// A line-oriented transcript writer, shared by startup fallback and narrow resizes.
pub struct LineMode<W> {
    writer: W,
    links: bool,
}

impl<W: Write> LineMode<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            links: false,
        }
    }

    /// Render trusted VEOX URLs as OSC 8 hyperlinks (see `super::links`).
    pub fn with_links(mut self, links: bool) -> Self {
        self.links = links;
        self
    }

    /// Preserve logical line breaks while stripping all terminal controls in user content.
    pub fn transcript(&mut self, lines: &[String]) -> io::Result<()> {
        for line in lines {
            for logical in line.split('\n') {
                let clean = sanitize_user_string(logical);
                writeln!(self.writer, "{}", super::links::linkify(&clean, self.links))?;
            }
        }
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fallback_is_independent_and_boundary_is_exact() {
        assert!(!required(false, Some("xterm-256color"), false, 40));
        for (plain, term, no_color, width) in [
            (true, Some("xterm"), false, 80),
            (false, Some("dumb"), false, 80),
            (false, Some("DUMB"), false, 80),
            (false, Some("xterm"), true, 80),
            (false, Some("xterm"), false, 39),
        ] {
            assert!(required(plain, term, no_color, width));
        }
    }

    #[test]
    fn output_preserves_lines_and_removes_escapes_including_hostile_user_text() {
        let mut mode = LineMode::new(Vec::new());
        mode.transcript(&[
            "first\nsecond".into(),
            "\x1b[31mred\x1b[0m \x1b]8;;https://invalid\x07link\x1b]8;;\x07\r\x08".into(),
        ])
        .unwrap();
        assert_eq!(mode.writer, b"first\nsecond\nred link\n");
        assert!(!mode.writer.contains(&0x1b));
    }

    #[test]
    fn links_are_added_only_for_trusted_urls_after_stripping_user_escapes() {
        let mut mode = LineMode::new(Vec::new()).with_links(true);
        mode.transcript(&[
            "Terms https://veox.ai/terms".into(),
            "\x1b]8;;https://evil.example\x07https://veox.ai/x\x1b]8;;\x07 https://evil.example"
                .into(),
        ])
        .unwrap();
        let text = String::from_utf8(mode.writer).unwrap();
        assert_eq!(
            text,
            format!(
                "Terms {}\n{} https://evil.example\n",
                super::super::links::osc8("https://veox.ai/terms", "https://veox.ai/terms"),
                super::super::links::osc8("https://veox.ai/x", "https://veox.ai/x"),
            )
        );
    }

    #[test]
    fn output_errors_propagate() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert_eq!(
            LineMode::new(Broken)
                .transcript(&["finished".into()])
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}
