//! Clickable links (OSC 8) for URLs that the jAIn CLI itself prints.
//!
//! Every displayed string is first stripped of all terminal controls by
//! [`super::sanitize_user_string`], so no user- or data-provided escape sequence ever reaches
//! the terminal. Links are then added back only by [`linkify`], and only for a URL that is
//! itself visible, whose target is exactly that visible text, and whose host belongs to VEOX
//! (or a loopback development workspace). A hyperlink can therefore never hide a different
//! destination behind its text, and the URL always stays visible and copyable when the
//! terminal does not support OSC 8.
//!
//! Detection is conservative: links are emitted only for terminals known to render OSC 8
//! (`JAIN_HYPERLINKS=1` forces them on, `JAIN_HYPERLINKS=0` or a `dumb` terminal turns them
//! off).

use std::io::{self, Write};

/// The terms of service shown at first run.
pub const TERMS_URL: &str = "https://veox.ai/terms";

const OSC8_CLOSE: &str = "\u{1b}]8;;\u{1b}\\";

/// Wrap `text` in an OSC 8 hyperlink to `url`. `url` must already be trusted and free of
/// control characters; [`linkify`] is the only caller that adds links to displayed text.
pub fn osc8(url: &str, text: &str) -> String {
    debug_assert!(!url.chars().any(char::is_control));
    format!("\u{1b}]8;;{url}\u{1b}\\{text}{OSC8_CLOSE}")
}

/// Whether the current terminal renders OSC 8 hyperlinks.
pub fn supported_from_env() -> bool {
    supported(|name| std::env::var(name).ok())
}

/// Terminal support heuristics, from environment variables supplied by `var`.
pub fn supported(var: impl Fn(&str) -> Option<String>) -> bool {
    match var("JAIN_HYPERLINKS").as_deref().map(str::trim) {
        Some("1" | "true" | "on" | "yes") => return true,
        Some("0" | "false" | "off" | "no") => return false,
        _ => {}
    }
    if var("TERM").is_some_and(|term| term.eq_ignore_ascii_case("dumb")) {
        return false;
    }
    if var("TERM_PROGRAM").is_some_and(|program| {
        matches!(
            program.as_str(),
            "iTerm.app" | "WezTerm" | "vscode" | "ghostty" | "Hyper" | "Tabby"
        )
    }) {
        return true;
    }
    if var("WT_SESSION").is_some_and(|value| !value.is_empty())
        || var("KITTY_WINDOW_ID").is_some_and(|value| !value.is_empty())
        || var("WEZTERM_EXECUTABLE").is_some_and(|value| !value.is_empty())
    {
        return true;
    }
    var("VTE_VERSION")
        .and_then(|value| value.trim().parse::<u32>().ok())
        .is_some_and(|version| version >= 5000)
}

/// Whether `url` may be rendered as a clickable link: HTTPS on veox.ai or a subdomain, or a
/// loopback development workspace.
pub fn trusted(url: &str) -> bool {
    let (scheme, rest) = match url.split_once("://") {
        Some(parts) => parts,
        None => return false,
    };
    if url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return false;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') {
        return false;
    }
    let host = match authority.rsplit_once(':') {
        Some((host, port)) if port.bytes().all(|b| b.is_ascii_digit()) => host,
        _ => authority,
    }
    .to_ascii_lowercase();
    match scheme {
        "https" => {
            host == "veox.ai"
                || host
                    .strip_suffix(".veox.ai")
                    .is_some_and(|sub| !sub.is_empty() && !sub.ends_with('.'))
                || matches!(host.as_str(), "localhost" | "127.0.0.1")
        }
        "http" => matches!(host.as_str(), "localhost" | "127.0.0.1"),
        _ => false,
    }
}

fn url_char(c: char) -> bool {
    c.is_ascii_graphic() && !matches!(c, '<' | '>' | '"' | '\'' | '`' | '\\' | '{' | '}' | '|')
}

/// Byte ranges of `http://` and `https://` URLs in `text`, excluding trailing punctuation.
pub fn find_urls(text: &str) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut search = 0;
    while search < text.len() {
        let rest = &text[search..];
        let Some(offset) = ["https://", "http://"]
            .iter()
            .filter_map(|scheme| rest.find(scheme))
            .min()
        else {
            break;
        };
        let start = search + offset;
        // A URL glued onto a preceding word (`xhttps://`) is not a separate link.
        if text[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric())
        {
            search = start + 1;
            continue;
        }
        let mut end = start
            + text[start..]
                .char_indices()
                .find(|(_, c)| !url_char(*c))
                .map_or(text.len() - start, |(index, _)| index);
        while end > start && text[..end].ends_with(['.', ',', ';', ':', '!', '?', ')', ']']) {
            end -= 1;
        }
        if text[start..end]
            .split_once("://")
            .is_some_and(|(_, host)| !host.is_empty())
        {
            found.push((start, end));
        }
        search = end.max(start + 1);
    }
    found
}

/// Add OSC 8 links to trusted, visible URLs in already-sanitized text. Without terminal
/// support (`enabled == false`) the text is returned unchanged.
pub fn linkify(clean: &str, enabled: bool) -> String {
    if !enabled {
        return clean.to_owned();
    }
    let mut output = String::with_capacity(clean.len());
    let mut last = 0;
    for (start, end) in find_urls(clean) {
        let url = &clean[start..end];
        if !trusted(url) {
            continue;
        }
        output.push_str(&clean[last..start]);
        output.push_str(&osc8(url, url));
        last = end;
    }
    output.push_str(&clean[last..]);
    output
}

/// A trusted URL drawn at a screen position by the full-screen interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spot {
    pub x: u16,
    pub y: u16,
    pub url: String,
}

/// Trusted URLs in one rendered row of plain text, as (column, url). Columns count cells,
/// which equal characters for the ASCII-only URLs this returns.
pub fn spots_in(text: &str) -> Vec<(u16, String)> {
    find_urls(text)
        .into_iter()
        .filter(|(start, end)| trusted(&text[*start..*end]))
        .filter_map(|(start, end)| {
            let column = ratatui::text::Span::raw(&text[..start]).width();
            Some((u16::try_from(column).ok()?, text[start..end].to_owned()))
        })
        .collect()
}

/// Re-emit drawn URL cells as OSC 8 hyperlinks after ratatui has drawn a frame.
///
/// ratatui 0.29 measures every cell symbol's display width, so embedding escape sequences in
/// cells corrupts its layout. Instead the frame is drawn as plain text and each visible URL is
/// written again, with the same characters and colors, wrapped in a hyperlink. The cursor
/// position is saved and restored, so the composer cursor is unchanged.
pub fn overlay(
    out: &mut impl Write,
    spots: &[Spot],
    buffer: &ratatui::buffer::Buffer,
) -> io::Result<()> {
    use ratatui::crossterm::{cursor, queue, style};
    if spots.is_empty() {
        return Ok(());
    }
    queue!(out, cursor::SavePosition)?;
    for spot in spots {
        if !trusted(&spot.url) {
            continue;
        }
        let area = buffer.area;
        let width = spot.url.chars().count() as u16;
        if spot.y >= area.bottom() || spot.x.saturating_add(width) > area.right() {
            continue;
        }
        // Only overlay text that is actually on screen at that position.
        let drawn: String = (0..width)
            .map(|offset| buffer[(spot.x + offset, spot.y)].symbol())
            .collect();
        if drawn != spot.url {
            continue;
        }
        let cell = &buffer[(spot.x, spot.y)];
        queue!(
            out,
            cursor::MoveTo(spot.x, spot.y),
            style::SetForegroundColor(cell.fg.into()),
            style::SetBackgroundColor(cell.bg.into()),
            style::SetAttribute(style::Attribute::Underlined),
            style::Print(osc8(&spot.url, &spot.url)),
            style::SetAttribute(style::Attribute::Reset),
            style::ResetColor
        )?;
    }
    queue!(out, cursor::RestorePosition)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn support_heuristics_cover_known_terminals_and_overrides() {
        assert!(supported(env(&[("TERM_PROGRAM", "iTerm.app")])));
        assert!(supported(env(&[("TERM_PROGRAM", "WezTerm")])));
        assert!(supported(env(&[("TERM_PROGRAM", "vscode")])));
        assert!(supported(env(&[("WT_SESSION", "abc")])));
        assert!(supported(env(&[("KITTY_WINDOW_ID", "1")])));
        assert!(supported(env(&[("VTE_VERSION", "6003")])));
        assert!(!supported(env(&[("VTE_VERSION", "4205")])));
        assert!(!supported(env(&[("TERM", "xterm-256color")])));
        assert!(!supported(env(&[])));
        assert!(supported(env(&[("JAIN_HYPERLINKS", "1")])));
        assert!(!supported(env(&[
            ("JAIN_HYPERLINKS", "0"),
            ("TERM_PROGRAM", "iTerm.app")
        ])));
        assert!(!supported(env(&[("TERM", "dumb"), ("WT_SESSION", "abc")])));
    }

    #[test]
    fn only_veox_and_loopback_urls_are_trusted() {
        for url in [
            "https://veox.ai/terms",
            "https://app.veox.ai/cli/connect?user_code=ABCD-1234",
            "https://dev-xbabe0.veox.ai",
            "http://127.0.0.1:8080/runs/x",
            "https://localhost:3000",
        ] {
            assert!(trusted(url), "{url}");
        }
        for url in [
            "https://evil.example/terms",
            "https://veox.ai.evil.example",
            "https://evilveox.ai",
            "http://app.veox.ai",
            "https://user@veox.ai",
            "javascript:alert(1)",
            "https://veox.ai/\u{7}",
            "https://.veox.ai",
        ] {
            assert!(!trusted(url), "{url}");
        }
    }

    #[test]
    fn finds_urls_without_trailing_punctuation() {
        let text = "Open https://app.veox.ai/cli/connect?user_code=AB-12. Then (https://veox.ai/terms), done";
        let urls: Vec<&str> = find_urls(text).iter().map(|(s, e)| &text[*s..*e]).collect();
        assert_eq!(
            urls,
            [
                "https://app.veox.ai/cli/connect?user_code=AB-12",
                "https://veox.ai/terms"
            ]
        );
        assert!(find_urls("xhttps://veox.ai").is_empty());
        assert!(find_urls("https://").is_empty());
    }

    #[test]
    fn linkify_wraps_only_trusted_visible_urls() {
        let clean = "Terms: https://veox.ai/terms or https://evil.example/x";
        assert_eq!(linkify(clean, false), clean);
        let linked = linkify(clean, true);
        assert_eq!(
            linked,
            format!(
                "Terms: {} or https://evil.example/x",
                osc8("https://veox.ai/terms", "https://veox.ai/terms")
            )
        );
        // Stripping our own sequences restores exactly the visible text.
        assert_eq!(crate::terminal::sanitize_user_string(&linked), clean);
    }

    #[test]
    fn hostile_hyperlinks_never_survive_sanitize_then_linkify() {
        // A data-provided link hiding an untrusted target behind trusted-looking text.
        let hostile =
            "\u{1b}]8;;https://evil.example\u{1b}\\https://veox.ai/terms\u{1b}]8;;\u{1b}\\";
        let shown = linkify(&crate::terminal::sanitize_user_string(hostile), true);
        assert!(!shown.contains("evil.example"));
        assert_eq!(
            shown,
            osc8("https://veox.ai/terms", "https://veox.ai/terms"),
            "only our own link, whose target is its visible text, remains"
        );
        let untrusted = linkify(
            &crate::terminal::sanitize_user_string(
                "\u{1b}]8;;https://evil.example\u{7}click\u{1b}]8;;\u{7}",
            ),
            true,
        );
        assert_eq!(untrusted, "click");
    }

    #[test]
    fn spots_report_cell_columns_for_wide_prefixes() {
        assert_eq!(
            spots_in("中 https://veox.ai/terms x https://evil.example"),
            vec![(3, "https://veox.ai/terms".to_owned())]
        );
    }

    #[test]
    fn overlay_rewrites_only_matching_cells_and_restores_the_cursor() {
        use ratatui::{buffer::Buffer, layout::Rect, style::Style};
        let mut buffer = Buffer::empty(Rect::new(0, 0, 40, 2));
        buffer.set_string(2, 0, "https://veox.ai/terms", Style::default());
        let spots = [
            Spot {
                x: 2,
                y: 0,
                url: "https://veox.ai/terms".into(),
            },
            // Stale: the text is no longer on screen there.
            Spot {
                x: 2,
                y: 1,
                url: "https://veox.ai/privacy".into(),
            },
        ];
        let mut out = Vec::new();
        overlay(&mut out, &spots, &buffer).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\u{1b}7"), "{text:?}");
        assert!(text.ends_with("\u{1b}8"), "{text:?}");
        assert_eq!(
            text.matches("\u{1b}]8;;https://veox.ai/terms\u{1b}\\")
                .count(),
            1
        );
        assert!(!text.contains("privacy"));
        let mut empty = Vec::new();
        overlay(&mut empty, &[], &buffer).unwrap();
        assert!(empty.is_empty());
    }
}
