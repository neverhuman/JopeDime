//! `/copy` and `/mouse`: getting any text out of the TUI (#2147).
//!
//! The full-screen interface does not capture the mouse by default, so the terminal's own
//! selection works on everything drawn. `/copy` additionally puts a chosen item on the
//! clipboard with OSC 52 (supported by most terminals: iTerm2, kitty, WezTerm, Alacritty,
//! Windows Terminal, foot, tmux with `set-clipboard on`) and always prints short items plainly
//! as well, so a terminal that ignores OSC 52 still shows the whole text to select.
//! `/transcript` shows the entire session as plain, unwrapped text on the normal screen.
use super::*;
use std::io::Write;

/// Largest text sent through OSC 52; many terminals refuse longer sequences.
const MAX_OSC52_BYTES: usize = 74_994;

/// What `/copy` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Item {
    /// Everything shown since the previous command (the default).
    Last,
    Link,
    Run,
    Result,
    All,
}

impl Item {
    pub(super) fn parse(args: &[String]) -> anyhow::Result<Self> {
        Ok(match args {
            [] => Self::Last,
            [one] => match one.as_str() {
                "link" | "url" => Self::Link,
                "run" | "id" => Self::Run,
                "result" => Self::Result,
                "all" | "transcript" => Self::All,
                "last" => Self::Last,
                _ => anyhow::bail!("usage: /copy [link|run|result|all]"),
            },
            _ => anyhow::bail!("usage: /copy [link|run|result|all]"),
        })
    }
}

/// The OSC 52 sequence that sets the system clipboard to `text`.
pub(super) fn osc52(text: &str) -> String {
    format!("\u{1b}]52;c;{}\u{7}", base64(text.as_bytes()))
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The transcript lines `item` names. `lines` is the plain transcript, oldest first, with
/// submitted commands as `> text` (the last one being the `/copy` itself).
pub(super) fn select(lines: &[String], item: Item, run: Option<&str>) -> Option<String> {
    let input = |line: &String| line.starts_with("> ");
    // Everything before the `/copy` that is being executed.
    let before = match lines.iter().rposition(input) {
        Some(last)
            if lines[last]
                .trim_start_matches("> ")
                .trim_start_matches(['/', '\\'])
                .starts_with("copy") =>
        {
            &lines[..last]
        }
        _ => lines,
    };
    let text = match item {
        Item::All => before.join("\n"),
        Item::Last => {
            let start = before.iter().rposition(input).map_or(0, |i| i + 1);
            before[start..].join("\n")
        }
        Item::Link => before.iter().rev().find_map(|line| {
            crate::terminal::links::find_urls(line)
                .last()
                .map(|(start, end)| line[*start..*end].to_owned())
        })?,
        Item::Run => run.map(str::to_owned).or_else(|| {
            before.iter().rev().find_map(|line| {
                line.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .find(|word| word.starts_with("run_") && word.len() > 20)
                    .map(str::to_owned)
            })
        })?,
        Item::Result => {
            let start = before
                .iter()
                .rposition(|line| line.contains("Run finished"))?;
            let end = before[start..]
                .iter()
                .position(input)
                .map_or(before.len(), |offset| start + offset);
            before[start..end].join("\n")
        }
    };
    let text = text.trim_matches('\n').to_owned();
    (!text.trim().is_empty()).then_some(text)
}

impl super::Session {
    pub(super) fn copy(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        let item = Item::parse(args)?;
        let run = self.snapshot.as_ref().map(|s| s.run_id.to_string());
        let lines = self.screen.transcript();
        let Some(text) = select(&lines, item, run.as_deref()) else {
            return Ok(vec![match item {
                Item::Link => "There is no link on screen to copy yet.".into(),
                Item::Run => "There is no run to copy yet.".into(),
                Item::Result => "There is no finished run result to copy yet.".into(),
                _ => "There is nothing to copy yet.".into(),
            }]);
        };
        let count = text.lines().count();
        let mut out = Vec::new();
        if text.len() <= MAX_OSC52_BYTES {
            let mut stdout = io::stdout();
            let _ = stdout
                .write_all(osc52(&text).as_bytes())
                .and_then(|()| stdout.flush());
            out.push(format!(
                "Copied {count} line{} to the clipboard (through your terminal; if it did not arrive, select the text below or use /transcript).",
                if count == 1 { "" } else { "s" }
            ));
        } else {
            out.push(format!(
                "That is {count} lines, too long for the terminal clipboard: /transcript shows it as plain text to select."
            ));
        }
        // Short items are shown again whole, on lines of their own, so they can be selected
        // even where OSC 52 is off. Long ones stay in the transcript and /transcript.
        if matches!(item, Item::Link | Item::Run) || count == 1 {
            out.push(text);
        }
        Ok(out)
    }

    pub(super) fn set_mouse(&mut self, capture: bool) {
        if self.mouse != capture {
            self.mouse = capture;
            self.mouse_changed = true;
        }
        self.screen.mouse = capture;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(str::to_owned).collect()
    }

    #[test]
    fn osc52_carries_base64_of_the_exact_text() {
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(osc52("hi\n"), "\u{1b}]52;c;aGkK\u{7}");
    }

    #[test]
    fn copy_selects_the_last_output_link_run_result_or_everything() {
        let transcript = lines(
            "> /demo regression_parkinsons\n\
             Demo regression_parkinsons · 120 rows\n\
             run run_01JBD8Q6N4V2T5Y8K3M7P9R1S0 started\n\
             ✓ Run finished time budget reached · champion alg_0001\n\
             Stopped at the time budget (10 s) after 15 of 128 candidates.\n\
             > /login\n\
             Open https://app.veox.ai/cli/connect?user_code=ABCD-1234\n\
             Pairing code: ABCD-1234\n\
             > /copy",
        );
        assert_eq!(
            select(&transcript, Item::Last, None).unwrap(),
            "Open https://app.veox.ai/cli/connect?user_code=ABCD-1234\nPairing code: ABCD-1234"
        );
        assert_eq!(
            select(&transcript, Item::Link, None).unwrap(),
            "https://app.veox.ai/cli/connect?user_code=ABCD-1234"
        );
        assert_eq!(
            select(&transcript, Item::Run, None).unwrap(),
            "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0"
        );
        assert_eq!(
            select(&transcript, Item::Run, Some("run_X")).unwrap(),
            "run_X"
        );
        assert_eq!(
            select(&transcript, Item::Result, None).unwrap(),
            "✓ Run finished time budget reached · champion alg_0001\nStopped at the time budget (10 s) after 15 of 128 candidates."
        );
        let all = select(&transcript, Item::All, None).unwrap();
        assert!(all.starts_with("> /demo") && all.ends_with("Pairing code: ABCD-1234"));
        assert_eq!(select(&lines("> /copy"), Item::Last, None), None);
        assert!(Item::parse(&["nope".into()]).is_err());
        assert_eq!(Item::parse(&[]).unwrap(), Item::Last);
    }

    #[test]
    fn long_urls_are_copied_whole_even_when_the_screen_wraps_them() {
        let url = format!("https://app.veox.ai/runs/{}", "a".repeat(300));
        let transcript = vec![
            format!("Explore it in your workspace: {url}"),
            "> /copy link".into(),
        ];
        assert_eq!(select(&transcript, Item::Link, None).unwrap(), url);
    }
}
