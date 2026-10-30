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
}

impl<W: Write> LineMode<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Preserve logical line breaks while stripping all terminal controls in user content.
    pub fn transcript(&mut self, lines: &[String]) -> io::Result<()> {
        for line in lines {
            for logical in line.split('\n') {
                writeln!(self.writer, "{}", sanitize_user_string(logical))?;
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
