//! Process-scoped terminal ownership, atomic signal flags, and panic-safe restoration.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock, Weak};

use ratatui::crossterm::{
    cursor::{Hide, Show},
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute,
    style::ResetColor,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

static OWNER: Mutex<()> = Mutex::new(());
static PANIC_HOOK: Once = Once::new();
static CURRENT: OnceLock<Mutex<Weak<RestoreState>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitSignal {
    Interrupt,
    Terminate,
    Hangup,
}

struct RestoreState {
    inline: AtomicBool,
    fullscreen: AtomicBool,
}

fn display_reset(writer: &mut impl Write) -> io::Result<()> {
    execute!(writer, Show, ResetColor, DisableBracketedPaste)
}

impl RestoreState {
    fn restore(&self) -> io::Result<()> {
        if !self.inline.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        // Always attempt both operations, even when one fails (e.g. a closed output pipe).
        let raw = disable_raw_mode();
        let screen = if self.fullscreen.swap(false, Ordering::SeqCst) {
            execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen)
        } else {
            Ok(())
        };
        let display = display_reset(&mut io::stdout());
        raw.and(screen).and(display)
    }
}

/// A single terminal session. Plain sessions never enable raw mode or emit reset commands.
pub struct TerminalGuard {
    state: Arc<RestoreState>,
    signal: Arc<AtomicUsize>,
    registrations: Vec<signal_hook::SigId>,
    // Unlock only after state and subscriptions have been dropped.
    _lease: MutexGuard<'static, ()>,
}

impl TerminalGuard {
    pub fn acquire(inline: bool) -> io::Result<Self> {
        let lease = match OWNER.try_lock() {
            Ok(lease) => lease,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(io::Error::other("another terminal session is active"));
            }
        };
        let state = Arc::new(RestoreState {
            inline: AtomicBool::new(false),
            fullscreen: AtomicBool::new(false),
        });
        let mut guard = Self {
            state,
            signal: Arc::new(AtomicUsize::new(0)),
            registrations: Vec::new(),
            _lease: lease,
        };
        // Handlers do only atomic stores. Registration precedes raw mode and input threads.
        use signal_hook::consts::{SIGINT, SIGTERM};
        let mut signals = vec![(SIGINT, 1), (SIGTERM, 2)];
        #[cfg(unix)]
        signals.push((signal_hook::consts::SIGHUP, 3));
        #[cfg(windows)]
        signals.push((signal_hook::consts::SIGBREAK, 2));
        for (signal, value) in signals {
            guard.registrations.push(signal_hook::flag::register_usize(
                signal,
                Arc::clone(&guard.signal),
                value,
            )?);
        }
        let current = CURRENT.get_or_init(|| Mutex::new(Weak::new()));
        PANIC_HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                let state = CURRENT.get().and_then(|current| {
                    current
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .upgrade()
                });
                if let Some(state) = state {
                    let _ = state.restore();
                }
                previous(info);
            }));
        });
        *current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(&guard.state);
        if inline {
            // Mark before either fallible operation so partial initialization is cleaned up.
            guard.state.inline.store(true, Ordering::SeqCst);
            enable_raw_mode()?;
            execute!(io::stdout(), Hide, EnableBracketedPaste)?;
        }
        Ok(guard)
    }

    /// Own the full terminal while retaining panic/signal and partial-startup cleanup.
    pub fn acquire_fullscreen() -> io::Result<Self> {
        let guard = Self::acquire(true)?;
        guard.state.fullscreen.store(true, Ordering::SeqCst);
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        Ok(guard)
    }

    /// Used on a narrow resize; restoration happens before the first plain transcript write.
    pub fn leave_inline(&self) -> io::Result<()> {
        self.state.restore()
    }

    pub fn signal(&self) -> Option<ExitSignal> {
        match self.signal.load(Ordering::SeqCst) {
            1 => Some(ExitSignal::Interrupt),
            2 => Some(ExitSignal::Terminate),
            3 => Some(ExitSignal::Hangup),
            _ => None,
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.state.restore();
        for registration in self.registrations.drain(..) {
            signal_hook::low_level::unregister(registration);
        }
        if let Some(current) = CURRENT.get() {
            *current
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Weak::new();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoration_commands_show_cursor_reset_colors_and_disable_paste() {
        let mut output = Vec::new();
        display_reset(&mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\x1b[?25h"));
        assert!(output.contains("\x1b[0m"));
        assert!(output.contains("\x1b[?2004l"));
        assert!(!output.contains("?1049"), "never use alternate screen");
    }
}
