mod model;
mod ui;
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use model::*;
use ratatui::{
    backend::{CrosstermBackend, TestBackend},
    Terminal,
};
use std::{
    env, fs,
    io::{self, IsTerminal},
    path::PathBuf,
    time::{Duration, Instant},
};

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
    }
}
fn config(path: &str) -> Result<std::collections::HashMap<String, String>, String> {
    let text = fs::read_to_string(path).map_err(|_| "cannot read --env file")?;
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once('=').ok_or("invalid env assignment")?;
        if ![
            "JOPE_WATCH_SNAPSHOT",
            "JOPE_WATCH_RUN",
            "JOPE_WATCH_ANALYSIS",
            "JOPE_WATCH_STATE",
            "JOPE_WATCH_PHASE",
            "JOPE_WATCH_INTERVAL",
        ]
        .contains(&k)
        {
            return Err("unknown env field; credentials do not belong in monitor config".into());
        }
        out.insert(k.to_string(), v.to_string());
    }
    Ok(out)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut path = None;
    let mut snap = None;
    let mut size = (150, 44);
    let mut check = false;
    let mut bench = None;
    let mut limit = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--env" => {
                let cfg = config(&args.next().ok_or("missing env path")?)?;
                if path.is_none() {
                    path = cfg.get("JOPE_WATCH_SNAPSHOT").cloned();
                }
            }
            "--snapshot" => path = Some(args.next().ok_or("missing snapshot path")?),
            "--render" => snap = Some(args.next().ok_or("missing render path")?),
            "--size" => {
                let s = args.next().ok_or("missing dimensions")?;
                let (w, h) = s.split_once('x').ok_or("expected WxH")?;
                size = (
                    w.parse::<u16>()?.clamp(20, 300),
                    h.parse::<u16>()?.clamp(8, 120),
                );
            }
            "--check" => check = true,
            "--bench" => {
                bench = Some(
                    args.next()
                        .ok_or("missing iterations")?
                        .parse::<usize>()?
                        .clamp(1, 100000),
                )
            }
            "--exit-after" => limit = Some(args.next().ok_or("missing seconds")?.parse::<u64>()?),
            "--help" | "-h" => {
                println!("jope-lambda-watch --env .env [--snapshot PATH]\n  --check validates one snapshot\n  --render PATH --size 150x44 writes ANSI + plain terminal frames\n  --bench N measures pure in-memory frames\nKeys: q quit, arrows/j/k select, / filter, f failures, Space freeze, Enter tree/KPI, ? help");
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    let path = PathBuf::from(
        path.or_else(|| env::var("JOPE_WATCH_SNAPSHOT").ok())
            .ok_or("set --env or --snapshot")?,
    );
    if check || snap.is_some() || bench.is_some() {
        let s = load(&path)?;
        if check {
            println!(
                "valid snapshot: {} views, sequence {}, mode {}",
                s.views.len(),
                s.sequence,
                s.mode
            );
            return Ok(());
        }
        let mut t = Terminal::new(TestBackend::new(size.0, size.1))?;
        let mut u = ui::Ui::default();
        let start = Instant::now();
        let n = bench.unwrap_or(1);
        for _ in 0..n {
            t.draw(|f| ui::draw(f, &s, &mut u, None, now_us()))?;
        }
        if bench.is_some() {
            println!(
                "{{\"frames\":{n},\"elapsed_ms\":{},\"microseconds_per_frame\":{},\"views\":{}}}",
                start.elapsed().as_secs_f64() * 1000.0,
                start.elapsed().as_secs_f64() * 1e6 / n as f64,
                s.views.len()
            );
        }
        if let Some(out) = snap {
            let b = t.backend().buffer();
            let mut ansi = String::new();
            let mut text = String::new();
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let c = &b[(x, y)];
                    if let ratatui::style::Color::Rgb(r, g, b) = c.fg {
                        ansi.push_str(&format!("\x1b[38;2;{r};{g};{b}m"));
                    }
                    if let ratatui::style::Color::Rgb(r, g, b) = c.bg {
                        ansi.push_str(&format!("\x1b[48;2;{r};{g};{b}m"));
                    }
                    ansi.push_str(c.symbol());
                    text.push_str(c.symbol());
                }
                ansi.push_str("\x1b[0m\n");
                text.push('\n');
            }
            fs::write(&out, ansi)?;
            fs::write(format!("{out}.txt"), text)?;
        }
        return Ok(());
    }
    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        return Err(
            "interactive mode requires a terminal; SSH with -t or use --render/--check".into(),
        );
    }
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
        hook(p);
    }));
    enable_raw_mode()?;
    let _guard = Guard;
    execute!(io::stdout(), EnterAlternateScreen, Hide)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut u = ui::Ui::default();
    let mut current = Snapshot::default();
    let mut error = Some("waiting for telemetry snapshot".into());
    let mut signature = None;
    let start = Instant::now();
    let mut last_draw = Instant::now() - Duration::from_secs(1);
    loop {
        if limit.is_some_and(|n| start.elapsed().as_secs() >= n) {
            break;
        }
        if !u.paused {
            let sig = fs::metadata(&path)
                .ok()
                .map(|m| (m.modified().ok(), m.len()));
            if sig != signature || error.is_some() {
                match load(&path) {
                    Ok(s) => {
                        if current.campaign == s.campaign
                            && current.feed_id == s.feed_id
                            && s.sequence < current.sequence
                        {
                            error =
                                Some("feed sequence regressed; retaining last good frame".into());
                        } else {
                            current = s;
                            error = None;
                            signature = sig;
                        }
                    }
                    Err(e) => error = Some(e),
                }
            }
            if error.is_none() {
                pulse(&path, &mut current);
            }
        }
        if last_draw.elapsed() >= Duration::from_millis(250) {
            terminal.draw(|f| ui::draw(f, &current, &mut u, error.as_deref(), now_us()))?;
            last_draw = Instant::now();
        }
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                        break;
                    }
                    if u.quality {
                        match k.code {
                            KeyCode::Char('q') => break,
                            KeyCode::Enter | KeyCode::Esc => {
                                u.quality = false;
                                u.detail_scroll = 0;
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                u.detail_scroll = u.detail_scroll.saturating_add(1)
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                u.detail_scroll = u.detail_scroll.saturating_sub(1)
                            }
                            KeyCode::PageDown => {
                                u.detail_scroll = u.detail_scroll.saturating_add(10)
                            }
                            KeyCode::PageUp => u.detail_scroll = u.detail_scroll.saturating_sub(10),
                            KeyCode::Char(' ') => u.paused = !u.paused,
                            _ => {}
                        }
                    } else if u.searching {
                        match k.code {
                            KeyCode::Esc => {
                                u.searching = false;
                                u.query.clear();
                            }
                            KeyCode::Enter => u.searching = false,
                            KeyCode::Backspace => {
                                u.query.pop();
                            }
                            KeyCode::Char(c) if !c.is_control() && u.query.len() < 128 => {
                                u.query.push(c)
                            }
                            _ => {}
                        }
                        u.selected = 0;
                        u.offset = 0;
                        u.anchor = None;
                    } else {
                        if matches!(
                            k.code,
                            KeyCode::Down
                                | KeyCode::Up
                                | KeyCode::PageDown
                                | KeyCode::PageUp
                                | KeyCode::Char('j')
                                | KeyCode::Char('k')
                                | KeyCode::Char('f')
                        ) {
                            u.anchor = None;
                        }
                        match k.code {
                            KeyCode::Char('q') => break,
                            KeyCode::Down | KeyCode::Char('j') => {
                                u.selected = u.selected.saturating_add(1)
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                u.selected = u.selected.saturating_sub(1)
                            }
                            KeyCode::PageDown => u.selected = u.selected.saturating_add(10),
                            KeyCode::PageUp => u.selected = u.selected.saturating_sub(10),
                            KeyCode::Char('/') => u.searching = true,
                            KeyCode::Char('f') => {
                                u.failures = !u.failures;
                                u.selected = 0;
                            }
                            KeyCode::Char(' ') => u.paused = !u.paused,
                            KeyCode::Char('?') => u.help = !u.help,
                            KeyCode::Enter => u.quality = !u.quality,
                            _ => {}
                        }
                    }
                    last_draw = Instant::now() - Duration::from_secs(1);
                }
                Event::Resize(_, _) => terminal.clear()?,
                _ => {}
            }
        }
    }
    Ok(())
}
