//! Helpers for exercising the appliance through a real pseudo-terminal.
#![allow(dead_code)] // Shared helpers are intentionally broader than the smoke test's current use.

use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};

/// Keys that need terminal control bytes rather than ordinary text input.
#[derive(Clone, Copy, Debug)]
pub enum Key {
    Enter,
    Escape,
    Tab,
    Backspace,
    Up,
    Down,
    Right,
    Left,
}

impl Key {
    fn bytes(self) -> &'static [u8] {
        match self {
            Self::Enter => b"\r",
            Self::Escape => b"\x1b",
            Self::Tab => b"\t",
            Self::Backspace => b"\x7f",
            Self::Up => b"\x1b[A",
            Self::Down => b"\x1b[B",
            Self::Right => b"\x1b[C",
            Self::Left => b"\x1b[D",
        }
    }
}

/// A child process connected to a PTY, with its output captured in the background.
pub struct PtySession {
    master: Box<dyn MasterPty + Send>,
    writer: Box<dyn Write + Send>,
    child: Box<dyn Child + Send + Sync>,
    output: Arc<Mutex<Vec<u8>>>,
    reader_thread: Option<JoinHandle<()>>,
}

impl PtySession {
    /// Spawn the package's `jopedime` binary with the requested arguments.
    pub fn spawn_jopedime(args: &[&str]) -> io::Result<Self> {
        let binary = option_env!("CARGO_BIN_EXE_jopedime")
            .ok_or_else(|| io::Error::other("jopedime binary path is integration-test only"))?;
        Self::spawn(binary, args)
    }

    /// Spawn jopedime with selected environment overrides for isolated engine homes.
    pub fn spawn_jopedime_with_env(args: &[&str], env: &[(&str, &str)]) -> io::Result<Self> {
        let binary = option_env!("CARGO_BIN_EXE_jopedime")
            .ok_or_else(|| io::Error::other("jopedime binary path is integration-test only"))?;
        Self::spawn_with_env(binary, args, env)
    }

    /// Spawn a program attached to a PTY. Kept public for other appliance integration tests.
    pub fn spawn(program: impl AsRef<OsStr>, args: &[&str]) -> io::Result<Self> {
        Self::spawn_with_env(program, args, &[])
    }

    /// Spawn a program attached to a PTY with selected environment overrides.
    pub fn spawn_with_env(
        program: impl AsRef<OsStr>,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> io::Result<Self> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(other)?;

        let mut command = CommandBuilder::new(program);
        command.args(args);
        command.env("TERM", "xterm-256color");
        for (key, value) in env {
            command.env(key, value);
        }
        let child = pair.slave.spawn_command(command).map_err(other)?;
        let mut reader = pair.master.try_clone_reader().map_err(other)?;
        let writer = pair.master.take_writer().map_err(other)?;
        let output = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&output);
        let reader_thread = thread::spawn(move || {
            let mut chunk = [0_u8; 4096];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => captured
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .extend_from_slice(&chunk[..count]),
                }
            }
        });

        Ok(Self {
            master: pair.master,
            writer,
            child,
            output,
            reader_thread: Some(reader_thread),
        })
    }

    /// Type ordinary text into the child's terminal input.
    pub fn type_text(&mut self, text: &str) -> io::Result<()> {
        self.writer.write_all(text.as_bytes())?;
        self.writer.flush()
    }

    /// Press a terminal key such as Enter, Escape, or an arrow key.
    pub fn press_key(&mut self, key: Key) -> io::Result<()> {
        self.writer.write_all(key.bytes())?;
        self.writer.flush()
    }

    /// Change the terminal dimensions, expressed as columns then rows.
    pub fn resize(&mut self, cols: u16, rows: u16) -> io::Result<()> {
        self.master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(other)
    }

    /// Return the current PTY dimensions.
    pub fn size(&self) -> io::Result<(u16, u16)> {
        let size = self.master.get_size().map_err(other)?;
        Ok((size.cols, size.rows))
    }

    /// Return captured terminal output with ANSI sequences and control bytes removed.
    pub fn read_screen_as_text(&self) -> String {
        let output = self
            .output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        strip_terminal_controls(&output)
    }

    /// Wait for text to appear in the terminal output, returning the plain-text screen snapshot.
    pub fn wait_for_text(&mut self, needle: &str, timeout: Duration) -> io::Result<String> {
        let deadline = Instant::now() + timeout;
        loop {
            let screen = self.read_screen_as_text();
            if screen.contains(needle) {
                return Ok(screen);
            }

            if let Some(status) = self.child.try_wait().map_err(other)? {
                self.join_reader();
                let screen = self.read_screen_as_text();
                if screen.contains(needle) {
                    return Ok(screen);
                }
                return Err(io::Error::other(format!(
                    "PTY child exited ({status}) before output contained {needle:?}; output: {screen}"
                )));
            }

            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("timed out waiting for {needle:?}; output: {screen}"),
                ));
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn join_reader(&mut self) {
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.join_reader();
    }
}

fn other(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn strip_terminal_controls(bytes: &[u8]) -> String {
    let mut plain = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            0x1b if bytes.get(index + 1) == Some(&b'[') => {
                index += 2;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if (0x40..=0x7e).contains(&byte) {
                        break;
                    }
                }
            }
            0x1b if bytes.get(index + 1) == Some(&b']') => {
                index += 2;
                while index < bytes.len() {
                    if bytes[index] == 0x07 {
                        index += 1;
                        break;
                    }
                    if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'\\') {
                        index += 2;
                        break;
                    }
                    index += 1;
                }
            }
            0x1b => index += 1 + usize::from(index + 1 < bytes.len()),
            b'\r' => {
                plain.push(b'\n');
                index += 1;
                if bytes.get(index) == Some(&b'\n') {
                    index += 1;
                }
            }
            byte if byte == b'\n' || byte == b'\t' || byte >= 0x20 => {
                plain.push(byte);
                index += 1;
            }
            _ => index += 1,
        }
    }
    String::from_utf8_lossy(&plain).into_owned()
}
