//! Semantic terminal colors, environment detection, and safe user text display.

use std::env;

pub mod app;
pub mod cards;
pub mod commands;
pub mod complete;
pub mod composer;

#[path = "highlight.rs"]
pub mod highlight;
pub mod line_mode;
pub mod restore;
pub mod runs;
pub mod screen;
pub mod status;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThemeMode {
    Light,
    Dark,
    Auto,
}

impl ThemeMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "light" => Some(Self::Light),
            "dark" => Some(Self::Dark),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    fn resolve(self, colorfgbg: Option<&str>) -> Self {
        match self {
            Self::Auto => {
                if colorfgbg
                    .and_then(|value| value.rsplit(';').next())
                    .and_then(|background| background.parse::<u8>().ok())
                    .is_some_and(|background| background >= 7)
                {
                    Self::Light
                } else {
                    Self::Dark
                }
            }
            mode => mode,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorLevel {
    TrueColor,
    Ansi256,
    Ansi16,
    None,
}

pub fn detect_color_level(
    colorterm: Option<&str>,
    term: Option<&str>,
    no_color: bool,
) -> ColorLevel {
    if no_color || term.is_some_and(|term| term.eq_ignore_ascii_case("dumb")) {
        return ColorLevel::None;
    }

    if colorterm.is_some_and(|value| {
        value.eq_ignore_ascii_case("truecolor") || value.eq_ignore_ascii_case("24bit")
    }) {
        return ColorLevel::TrueColor;
    }

    match term {
        Some(term) if term.to_ascii_lowercase().contains("256color") => ColorLevel::Ansi256,
        Some(term) if !term.is_empty() => ColorLevel::Ansi16,
        _ => ColorLevel::None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThemeToken {
    Fg,
    Muted,
    Accent,
    Good,
    Warn,
    Bad,
    CodeKeyword,
    CodeString,
    CodeNumber,
    CodeComment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ansi16Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThemeColor {
    Default,
    Ansi16(Ansi16Color),
    Indexed(u8),
    Rgb(u8, u8, u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Theme {
    mode: ThemeMode,
    level: ColorLevel,
}

impl Theme {
    pub fn from_env(mode: ThemeMode) -> Self {
        let colorterm = env::var("COLORTERM").ok();
        let term = env::var("TERM").ok();
        let no_color = env::var_os("NO_COLOR").is_some();
        let colorfgbg = env::var("COLORFGBG").ok();
        Self {
            mode: mode.resolve(colorfgbg.as_deref()),
            level: detect_color_level(colorterm.as_deref(), term.as_deref(), no_color),
        }
    }

    pub fn new(mode: ThemeMode, level: ColorLevel, colorfgbg: Option<&str>) -> Self {
        Self {
            mode: mode.resolve(colorfgbg),
            level,
        }
    }

    pub fn mode(self) -> ThemeMode {
        self.mode
    }

    pub fn color_level(self) -> ColorLevel {
        self.level
    }

    pub fn color(self, token: ThemeToken) -> Option<ThemeColor> {
        if self.level == ColorLevel::None {
            return None;
        }

        let light = self.mode == ThemeMode::Light;
        let fallback = match token {
            ThemeToken::Fg => return Some(ThemeColor::Default),
            ThemeToken::Muted | ThemeToken::CodeComment => {
                if light {
                    Ansi16Color::Black
                } else {
                    Ansi16Color::BrightBlack
                }
            }
            ThemeToken::Accent => {
                if light {
                    Ansi16Color::Blue
                } else {
                    Ansi16Color::Cyan
                }
            }
            ThemeToken::Good | ThemeToken::CodeString => Ansi16Color::Green,
            ThemeToken::Warn => Ansi16Color::Yellow,
            ThemeToken::Bad => Ansi16Color::Red,
            ThemeToken::CodeKeyword => Ansi16Color::Blue,
            ThemeToken::CodeNumber => Ansi16Color::Magenta,
        };

        Some(match self.level {
            ColorLevel::None => unreachable!("handled above"),
            ColorLevel::Ansi16 => ThemeColor::Ansi16(fallback),
            ColorLevel::Ansi256 => ThemeColor::Indexed(match fallback {
                Ansi16Color::Default => 7,
                Ansi16Color::Black => 0,
                Ansi16Color::Red => 1,
                Ansi16Color::Green => 2,
                Ansi16Color::Yellow => 3,
                Ansi16Color::Blue => 4,
                Ansi16Color::Magenta => 5,
                Ansi16Color::Cyan => 6,
                Ansi16Color::White => 7,
                Ansi16Color::BrightBlack => 8,
                Ansi16Color::BrightRed => 9,
                Ansi16Color::BrightGreen => 10,
                Ansi16Color::BrightYellow => 11,
                Ansi16Color::BrightBlue => 12,
                Ansi16Color::BrightMagenta => 13,
                Ansi16Color::BrightCyan => 14,
                Ansi16Color::BrightWhite => 15,
            }),
            ColorLevel::TrueColor => {
                let (red, green, blue) = match fallback {
                    Ansi16Color::Default => unreachable!("foreground returned above"),
                    Ansi16Color::Black => (45, 45, 45),
                    Ansi16Color::Red => (220, 50, 47),
                    Ansi16Color::Green => (40, 160, 70),
                    Ansi16Color::Yellow => (200, 150, 0),
                    Ansi16Color::Blue => (55, 100, 210),
                    Ansi16Color::Magenta => (180, 70, 190),
                    Ansi16Color::Cyan => (0, 175, 190),
                    Ansi16Color::White => (230, 230, 230),
                    Ansi16Color::BrightBlack => (125, 125, 125),
                    Ansi16Color::BrightRed => (255, 100, 100),
                    Ansi16Color::BrightGreen => (100, 220, 100),
                    Ansi16Color::BrightYellow => (255, 220, 100),
                    Ansi16Color::BrightBlue => (120, 160, 255),
                    Ansi16Color::BrightMagenta => (220, 130, 230),
                    Ansi16Color::BrightCyan => (100, 220, 230),
                    Ansi16Color::BrightWhite => (255, 255, 255),
                };
                ThemeColor::Rgb(red, green, blue)
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Symbols {
    pub success: &'static str,
    pub active: &'static str,
    pub warning: &'static str,
    pub failure: &'static str,
}

pub fn symbols_for_locale(locale: Option<&str>) -> Symbols {
    let utf8 = locale.is_some_and(|locale| {
        let normalized = locale.to_ascii_lowercase();
        normalized.contains("utf-8") || normalized.contains("utf8")
    });
    if utf8 {
        Symbols {
            success: "✓",
            active: "●",
            warning: "⚠",
            failure: "✗",
        }
    } else {
        Symbols {
            success: "+",
            active: "*",
            warning: "!",
            failure: "x",
        }
    }
}

pub fn symbols_from_env() -> Symbols {
    symbols_for_locale(
        env::var("LC_ALL")
            .ok()
            .or_else(|| env::var("LC_CTYPE").ok())
            .or_else(|| env::var("LANG").ok())
            .as_deref(),
    )
}

/// Strip terminal control sequences and all remaining control characters from user text.
pub fn sanitize_user_string(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut output = String::with_capacity(input.len());
    let mut index = 0;

    while index < chars.len() {
        let character = chars[index];
        if character == '\u{1b}' || character == '\u{9b}' || character == '\u{9d}' {
            let is_esc = character == '\u{1b}';
            let kind = if is_esc && index + 1 < chars.len() {
                chars[index + 1]
            } else if character == '\u{9b}' {
                '['
            } else if character == '\u{9d}' {
                ']'
            } else {
                '\0'
            };

            if kind == '[' {
                index += if is_esc { 2 } else { 1 };
                while index < chars.len() {
                    let code = chars[index] as u32;
                    index += 1;
                    if (0x40..=0x7e).contains(&code) {
                        break;
                    }
                }
                continue;
            }

            if kind == ']' {
                index += if is_esc { 2 } else { 1 };
                while index < chars.len() {
                    if chars[index] == '\u{7}' {
                        index += 1;
                        break;
                    }
                    if chars[index] == '\u{1b}'
                        && chars.get(index + 1).is_some_and(|next| *next == '\\')
                    {
                        index += 2;
                        break;
                    }
                    index += 1;
                }
                continue;
            }

            // An isolated or unknown ESC must not eat the following printable character.
            index += 1;
            continue;
        }

        if !character.is_control() {
            output.push(character);
        }
        index += 1;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_each_color_level_and_disables_color_when_requested() {
        assert_eq!(
            detect_color_level(Some("truecolor"), Some("xterm-256color"), false),
            ColorLevel::TrueColor
        );
        assert_eq!(
            detect_color_level(Some("24bit"), Some("xterm"), false),
            ColorLevel::TrueColor
        );
        assert_eq!(
            detect_color_level(None, Some("xterm-256color"), false),
            ColorLevel::Ansi256
        );
        assert_eq!(
            detect_color_level(None, Some("xterm"), false),
            ColorLevel::Ansi16
        );
        assert_eq!(
            detect_color_level(None, Some("dumb"), false),
            ColorLevel::None
        );
        assert_eq!(
            detect_color_level(Some("truecolor"), Some("xterm"), true),
            ColorLevel::None
        );
        assert_eq!(detect_color_level(None, None, false), ColorLevel::None);
    }

    #[test]
    fn parses_theme_modes_and_resolves_auto_from_terminal_background() {
        assert_eq!(ThemeMode::parse("light"), Some(ThemeMode::Light));
        assert_eq!(ThemeMode::parse("dark"), Some(ThemeMode::Dark));
        assert_eq!(ThemeMode::parse("auto"), Some(ThemeMode::Auto));
        assert_eq!(ThemeMode::parse("sepia"), None);
        assert_eq!(
            Theme::new(ThemeMode::Auto, ColorLevel::Ansi16, Some("15;15")).mode(),
            ThemeMode::Light
        );
        assert_eq!(
            Theme::new(ThemeMode::Auto, ColorLevel::Ansi16, Some("15;0")).mode(),
            ThemeMode::Dark
        );
        assert_eq!(
            Theme::new(ThemeMode::Auto, ColorLevel::Ansi16, None).mode(),
            ThemeMode::Dark
        );
    }

    #[test]
    fn maps_tokens_at_each_color_depth_and_none() {
        let sixteen = Theme::new(ThemeMode::Dark, ColorLevel::Ansi16, None);
        let fallbacks = [
            (ThemeToken::Fg, ThemeColor::Default),
            (
                ThemeToken::Muted,
                ThemeColor::Ansi16(Ansi16Color::BrightBlack),
            ),
            (ThemeToken::Accent, ThemeColor::Ansi16(Ansi16Color::Cyan)),
            (ThemeToken::Good, ThemeColor::Ansi16(Ansi16Color::Green)),
            (ThemeToken::Warn, ThemeColor::Ansi16(Ansi16Color::Yellow)),
            (ThemeToken::Bad, ThemeColor::Ansi16(Ansi16Color::Red)),
            (
                ThemeToken::CodeKeyword,
                ThemeColor::Ansi16(Ansi16Color::Blue),
            ),
            (
                ThemeToken::CodeString,
                ThemeColor::Ansi16(Ansi16Color::Green),
            ),
            (
                ThemeToken::CodeNumber,
                ThemeColor::Ansi16(Ansi16Color::Magenta),
            ),
            (
                ThemeToken::CodeComment,
                ThemeColor::Ansi16(Ansi16Color::BrightBlack),
            ),
        ];
        for (token, fallback) in fallbacks {
            assert_eq!(sixteen.color(token), Some(fallback));
        }
        assert_eq!(
            Theme::new(ThemeMode::Light, ColorLevel::Ansi16, None).color(ThemeToken::Accent),
            Some(ThemeColor::Ansi16(Ansi16Color::Blue))
        );
        let indexed = Theme::new(ThemeMode::Dark, ColorLevel::Ansi256, None);
        let truecolor = Theme::new(ThemeMode::Dark, ColorLevel::TrueColor, None);
        let no_color = Theme::new(ThemeMode::Dark, ColorLevel::None, None);
        for (token, _) in fallbacks {
            if token == ThemeToken::Fg {
                assert_eq!(indexed.color(token), Some(ThemeColor::Default));
                assert_eq!(truecolor.color(token), Some(ThemeColor::Default));
            } else {
                assert!(matches!(indexed.color(token), Some(ThemeColor::Indexed(_))));
                assert!(matches!(
                    truecolor.color(token),
                    Some(ThemeColor::Rgb(_, _, _))
                ));
            }
            assert_eq!(no_color.color(token), None);
        }
    }

    #[test]
    fn uses_unicode_symbols_only_for_utf8_locales() {
        let unicode = symbols_for_locale(Some("en_US.UTF-8"));
        assert_eq!(unicode.success, "✓");
        assert_eq!(unicode.active, "●");
        assert_eq!(unicode.warning, "⚠");
        assert_eq!(unicode.failure, "✗");

        let ascii = symbols_for_locale(Some("C"));
        assert_eq!(ascii.success, "+");
        assert_eq!(ascii.active, "*");
        assert_eq!(ascii.warning, "!");
        assert_eq!(ascii.failure, "x");
        assert_eq!(symbols_for_locale(None), ascii);
    }

    #[test]
    fn strips_escape_sequences_and_controls_without_losing_plain_text() {
        assert_eq!(sanitize_user_string("a\u{1b}b"), "ab");
        assert_eq!(sanitize_user_string("a\u{1b}[31mred\u{1b}[0m!"), "ared!");
        assert_eq!(
            sanitize_user_string("a\u{1b}]8;;https://example.test\u{7}link\u{1b}]8;;\u{1b}\\z"),
            "alinkz"
        );
        assert_eq!(sanitize_user_string("a\u{1b}]title\u{7}b"), "ab");
        assert_eq!(sanitize_user_string("a\u{7}b\n\tc"), "abc");
        assert_eq!(sanitize_user_string("a\u{9b}31mred"), "ared");
        assert_eq!(sanitize_user_string("a\u{9d}title\u{7}b"), "ab");
        assert_eq!(sanitize_user_string("plain λ text"), "plain λ text");
    }
}
