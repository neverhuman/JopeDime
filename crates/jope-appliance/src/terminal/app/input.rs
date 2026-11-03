use super::*;
pub(super) fn submit_draft(
    state: &mut State,
    command: &mut impl FnMut(&str) -> io::Result<()>,
) -> io::Result<Option<Exit>> {
    let draft = std::mem::take(&mut state.draft);
    if matches!(draft.trim(), "\\quit" | "/quit") {
        return Ok(Some(Exit::Quit));
    }
    command(&draft)?;
    Ok(None)
}

pub(super) fn handle_input_event(
    event: Event,
    screen: &mut Screen,
    guard: &TerminalGuard,
    width: &mut u16,
    state: &mut State,
    command: &mut impl FnMut(&str) -> io::Result<()>,
) -> io::Result<Option<Exit>> {
    #[cfg(all(test, target_os = "linux"))]
    trace_first_event_for_test(&event);
    match event {
        Event::Resize(columns, _) => {
            *width = columns;
            if columns < 40 {
                screen.downgrade(guard)?;
            }
        }
        Event::Paste(text) => state.draft.push_str(&sanitize_user_string(&text)),
        Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if state.draft.is_empty() {
                    return Ok(Some(Exit::Quit));
                }
                state.draft.pop();
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                state.draft.clear();
                screen.transcript(&["Use \\quit or Ctrl-D to exit.".into()], *width)?;
            }
            KeyCode::Enter => return submit_draft(state, command),
            KeyCode::Backspace => {
                state.draft.pop();
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                state.draft.push(character);
            }
            _ => {}
        },
        _ => {}
    }
    Ok(None)
}

pub(super) fn drain_queued_input(
    screen: &mut Screen,
    guard: &TerminalGuard,
    width: &mut u16,
    state: &mut State,
    command: &mut impl FnMut(&str) -> io::Result<()>,
) -> io::Result<Option<Exit>> {
    drain_event_queue(
        || {
            if event::poll(Duration::ZERO)? {
                event::read().map(Some)
            } else {
                Ok(None)
            }
        },
        |event| handle_input_event(event, screen, guard, width, state, command),
    )
}

pub(super) fn reconcile_width_and_drain(
    screen: &mut Screen,
    guard: &TerminalGuard,
    width: &mut u16,
    state: &mut State,
    command: &mut impl FnMut(&str) -> io::Result<()>,
) -> io::Result<Option<Exit>> {
    let was_inline = matches!(screen, Screen::Inline(_));
    screen.reconcile_width(guard, width)?;
    if was_inline && matches!(screen, Screen::Plain(_)) {
        drain_queued_input(screen, guard, width, state, command)
    } else {
        Ok(None)
    }
}

pub(super) fn drain_event_queue(
    mut next: impl FnMut() -> io::Result<Option<Event>>,
    mut handle: impl FnMut(Event) -> io::Result<Option<Exit>>,
) -> io::Result<Option<Exit>> {
    while let Some(event) = next()? {
        if let Some(exit) = handle(event)? {
            return Ok(Some(exit));
        }
    }
    Ok(None)
}

#[cfg(all(test, target_os = "linux"))]
pub(super) static STOP_BEFORE_EVENT_POLL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(all(test, target_os = "linux"))]
pub(super) static TRACE_FIRST_EVENT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(test, target_os = "linux"))]
pub(super) fn stop_before_event_poll_for_test() -> io::Result<()> {
    use std::sync::atomic::Ordering;
    if !STOP_BEFORE_EVENT_POLL.swap(false, Ordering::SeqCst) {
        return Ok(());
    }
    let pid = std::process::id().to_string();
    signal_hook::low_level::raise(signal_hook::consts::SIGWINCH)?;
    TRACE_FIRST_EVENT.store(true, Ordering::SeqCst);
    println!("\nPROBE_BEFORE_POLL {pid}");
    io::Write::flush(&mut io::stdout())?;
    signal_hook::low_level::raise(signal_hook::consts::SIGSTOP)?;
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
pub(super) fn trace_first_event_for_test(event: &Event) {
    use std::sync::atomic::Ordering;
    if !TRACE_FIRST_EVENT.swap(false, Ordering::SeqCst) {
        return;
    }
    let kind = match event {
        Event::Resize(..) => "resize",
        Event::Key(key) if key.code == KeyCode::Enter => "enter",
        Event::Key(_) => "key",
        _ => "other",
    };
    println!("\nPROBE_FIRST_EVENT={kind}");
    let _ = io::Write::flush(&mut io::stdout());
}
