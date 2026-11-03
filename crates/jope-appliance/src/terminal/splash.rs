//! Animated JAIN landing logo.
//!
//! Open, fine-stroked letters in cool graphite and silver, with a blue glint.
//! The letter texture stays fixed while a single soft highlight drifts across it.
//! Frames are a pure function of elapsed time so they can be tested exactly.
use std::time::Duration;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::{ColorLevel, ThemeMode};

const ROWS: usize = 10;
/// Open pixel skeleton; rendering adds a modest horizontal weight to each stroke.
const LETTERS: [&[&str; ROWS]; 4] = [
    &[
        "########", "......#.", "......#.", "......#.", "......#.", "......#.", "......#.",
        "#.....#.", ".#...#..", "..###...",
    ],
    &[
        "....##.....",
        "....#.#....",
        "....#.#....",
        "...#...#...",
        "...#...#...",
        "..#.....#..",
        "..#######..",
        ".#.......#.",
        ".#.......#.",
        "#.........#",
    ],
    &[
        "#######", "..##...", "..##...", "..##...", "..##...", "..##...", "..##...", "..##...",
        "..##...", "#######",
    ],
    &[
        "#.........#",
        "##........#",
        "#.#.......#",
        "#..#......#",
        "#...#.....#",
        "#...#.....#",
        "#....#....#",
        "#.....#...#",
        "#......#..#",
        "#.......###",
    ],
];
const GAP: usize = 0;

/// Resting marks: light, unstructured scatter. Repeats weight the lightest marks.
const REST: [char; 14] = [
    '.', '\'', '`', '.', '-', '.', '\'', '.', '`', '-', '.', '`', '\'', '.',
];

/// Fine glints keep the letter strokes airy even at the highlight's peak.
const GLINT: [[char; 2]; 6] = [
    [':', ';'],
    ['·', ':'],
    ['+', '˙'],
    ['*', '+'],
    ['✧', '*'],
    ['·', '✦'],
];

const DIGITS: [char; 10] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

/// One-shot timeline: a brief hold, then one wave crosses.
const PLAYBACK_SPEED: f32 = 2.0;
const HOLD: f32 = 0.9;
const PASS: f32 = 4.0;
const CYCLE: f32 = HOLD + PASS;
/// Crest brightness below this leaves the resting mark in place.
const QUIET: f32 = 0.18;

fn width_px() -> usize {
    LETTERS.iter().map(|letter| letter[0].len()).sum::<usize>() + GAP * (LETTERS.len() - 1)
}

/// Widen strokes by one pixel while keeping the counters and letter spacing open.
fn letter_ink(letter: &[&str; ROWS], x: usize, y: usize) -> bool {
    let row = letter[y].as_bytes();
    row[x] == b'#' || (x > 0 && row[x - 1] == b'#')
}

fn ink(px: usize, row: usize) -> bool {
    let mut x = px;
    for letter in LETTERS {
        let width = letter[row].len();
        if x < width {
            return letter_ink(letter, x, row);
        }
        x -= width;
        if x < GAP {
            return false;
        }
        x -= GAP;
    }
    false
}

/// Arrival times follow connected ink (including diagonals), never straight through counters.
/// Letter starts overlap; J enters at its lower-left tip and A at its lower-left foot.
fn arrivals() -> &'static [f32] {
    static MAP: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let width = width_px();
        let mut times = vec![f32::INFINITY; width * ROWS];
        let mut offset = 0;
        for (index, letter) in LETTERS.iter().enumerate() {
            let w = letter[0].len();
            let seed = match index {
                0 => (0, 7),
                1 | 2 => (0, ROWS - 1),
                _ => (0, 0),
            };
            let mut distances = vec![usize::MAX; w * ROWS];
            let mut queue = std::collections::VecDeque::from([seed]);
            distances[seed.1 * w + seed.0] = 0;
            while let Some((x, y)) = queue.pop_front() {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= ROWS as isize {
                            continue;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        if letter_ink(letter, nx, ny) && distances[ny * w + nx] == usize::MAX {
                            distances[ny * w + nx] = distances[y * w + x] + 1;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
            let longest = distances
                .iter()
                .copied()
                .filter(|d| *d != usize::MAX)
                .max()
                .unwrap_or(1)
                .max(1);
            for y in 0..ROWS {
                for x in 0..w {
                    if distances[y * w + x] != usize::MAX {
                        times[y * width + offset + x] = 0.06 + index as f32 * 0.18
                            - if index > 0 { 0.035 } else { 0.0 }
                            + if index == 3 { 0.06 } else { 0.0 }
                            + distances[y * w + x] as f32 / longest as f32 * 0.28;
                    }
                }
            }
            offset += w + GAP;
        }
        times
    })
}

fn through(arrival: f32, progress: f32) -> f32 {
    let age = progress - arrival;
    let spread = if age < 0.0 { 0.025 } else { 0.085 };
    (-(age / spread).powi(2)).exp()
}

/// Deterministic per-cell noise in [0, 1).
fn hash(x: usize, y: usize, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

fn pick<const N: usize>(marks: &[char; N], noise: f32) -> char {
    marks[((noise * N as f32) as usize).min(N - 1)]
}

/// Progress of the single pass, or `None` before it starts and after it finishes.
fn pass(t: f32) -> Option<f32> {
    let phase = t - HOLD;
    (phase >= 0.0 && phase < PASS).then(|| phase / PASS)
}

/// A broad satin highlight, with a soft leading edge and a longer trailing glow.
fn wave(u: f32, v: f32, t: f32, progress: f32) -> f32 {
    // Ease the travel so the wave glides in and out rather than starting abruptly.
    let eased = progress * progress * (3.0 - 2.0 * progress);
    let crest = -0.6 + eased * 2.2;
    let bow = 0.05 * (v * 2.6 + t * 1.1).sin() + 0.10 * (v - 0.5);
    let d = u + bow - crest;
    if d > 0.0 {
        (-(d / 0.18).powi(2)).exp()
    } else {
        (-(d / 0.28).powi(2)).exp()
    }
}

/// A soft, smooth shimmer for cells under the crest: each cell has its own slow rate and phase.
fn shimmer(x: usize, y: usize, t: f32) -> f32 {
    let rate = 0.6 + 0.6 * hash(x, y, 11);
    let phase = hash(x, y, 13) * std::f32::consts::TAU;
    0.5 + 0.5 * (t * rate + phase).sin()
}

/// Sparse, smooth 220ms pinpricks. Only the moving wave reveals them.
fn speckle(x: usize, y: usize, t: f32) -> f32 {
    if hash(x, y, 23) > 0.14 {
        return 0.0;
    }
    let phase = (t + hash(x, y, 29) * 2.7).rem_euclid(2.7);
    if phase >= 0.22 {
        return 0.0;
    }
    (std::f32::consts::PI * phase / 0.22).sin().powi(2)
}

/// Which logo size fits, as (width in columns, rows per pixel), or `None` for text only.
pub fn fit(width: u16, height: u16) -> Option<(usize, usize)> {
    let width = usize::from(width);
    let height = usize::from(height);
    let px = width_px();
    // Two columns per pixel keeps pixels square on a typical 1:2 terminal cell.
    // Large: square pixels two rows tall. Normal: one row per pixel, squeezed to the width.
    if height >= ROWS * 2 + 2 && width >= px * 4 {
        Some((px * 4, 2))
    } else if height >= ROWS + 2 && width >= px {
        Some(((px * 2).min(width), 1))
    } else {
        None
    }
}

pub struct Palette {
    level: ColorLevel,
    light: bool,
}

impl Palette {
    pub fn new(level: ColorLevel, mode: ThemeMode) -> Self {
        Self {
            level,
            light: mode == ThemeMode::Light,
        }
    }

    /// Color rises gently from graphite to blue without changing the stroke weight.
    fn style(&self, energy: f32) -> Style {
        let bold = Modifier::empty();
        let color = match self.level {
            ColorLevel::None => return Style::default().add_modifier(bold),
            ColorLevel::TrueColor => {
                let (rest, crest) = if self.light {
                    ((108, 112, 120), (40, 92, 190))
                } else {
                    ((134, 141, 150), (120, 174, 255))
                };
                // Ease-in so most of the wave stays near the resting tone.
                let f = energy.clamp(0.0, 1.0).powf(1.6);
                let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * f) as u8;
                Color::Rgb(
                    mix(rest.0, crest.0),
                    mix(rest.1, crest.1),
                    mix(rest.2, crest.2),
                )
            }
            ColorLevel::Ansi256 => {
                const DARK: [u8; 6] = [245, 246, 249, 110, 111, 75];
                const LIGHT: [u8; 6] = [243, 242, 241, 67, 68, 26];
                let step = ((energy.clamp(0.0, 1.0) * 5.0).round() as usize).min(5);
                Color::Indexed(if self.light { LIGHT } else { DARK }[step])
            }
            ColorLevel::Ansi16 => match (self.light, energy > 0.7) {
                (false, false) => Color::DarkGray,
                (false, true) => Color::LightBlue,
                (true, false) => Color::DarkGray,
                (true, true) => Color::Blue,
            },
        };
        Style::default().fg(color).add_modifier(bold)
    }
}

/// Render the logo at elapsed time `t`, `columns` wide with `rows_per_px` rows per pixel row.
pub fn frame(
    elapsed: Duration,
    columns: usize,
    rows_per_px: usize,
    palette: &Palette,
) -> Vec<Line<'static>> {
    let t = elapsed.as_secs_f32() * PLAYBACK_SPEED;
    let px = width_px();
    let rows = ROWS * rows_per_px;
    let progress = pass(t);
    let arrivals = arrivals();
    let rest = palette.style(if t >= CYCLE { 1.0 } else { 0.0 });
    (0..rows)
        .map(|y| {
            let row = y / rows_per_px;
            let v = y as f32 / (rows.max(2) - 1) as f32;
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(columns);
            for x in 0..columns {
                if !ink(x * px / columns.max(1), row) {
                    spans.push(Span::raw(" "));
                    continue;
                }
                let mark = pick(&REST, hash(x, y, 1));
                let Some(progress) = progress else {
                    spans.push(Span::styled(mark.to_string(), rest));
                    continue;
                };
                let arrival = arrivals[row * px + x * px / columns.max(1)];
                // Infected strokes retain blue as their transient glyphs fade back to texture.
                let retained = ((progress - arrival) / 0.06).clamp(0.0, 1.0);
                let rest = palette.style(retained);
                let u = x as f32 / (columns.max(2) - 1) as f32;
                // The wash sits beneath a stronger pulse conducted by the letter strokes.
                let mut b = (0.38 * wave(u, v, t, progress)).max(through(arrival, progress));
                if b < 0.02 {
                    // Untouched by the wave: exactly the resting mark.
                    spans.push(Span::styled(mark.to_string(), rest));
                    continue;
                }
                // Shimmer only in the brighter part of the crest, and only a little.
                if b > 0.45 {
                    b *= 0.82 + 0.18 * shimmer(x, y, t);
                }
                // A fixed dither keeps the crest edge organic without adding motion.
                let mut b = (b + (hash(x, y, 3) - 0.5) * 0.08).clamp(0.0, 1.0);
                let flash = if b > 0.25 { speckle(x, y, t) } else { 0.0 };
                b += (1.0 - b) * flash;
                let mark = if flash > 0.55 {
                    '·'
                } else if b > 0.35 && hash(x, y, 31) < 0.12 {
                    pick(&DIGITS, hash(x, y, 37))
                } else if b < QUIET {
                    mark
                } else {
                    let level = (b - QUIET) / (1.0 - QUIET) * GLINT.len() as f32;
                    pick(
                        &GLINT[(level as usize).min(GLINT.len() - 1)],
                        hash(x, y, 17),
                    )
                };
                spans.push(Span::styled(
                    mark.to_string(),
                    palette.style(b.max(retained)),
                ));
            }
            Line::from(spans)
        })
        .collect()
}

/// Whether the landing should animate; `JAIN_NO_ANIMATION` turns motion off.
pub fn animated() -> bool {
    std::env::var_os("JAIN_NO_ANIMATION").is_none_or(|value| value.is_empty() || value == "0")
}

/// Whether the one-shot animation still needs frequent redraws, including its final frame.
pub fn running(elapsed: Duration) -> bool {
    elapsed.as_secs_f32() * PLAYBACK_SPEED < CYCLE + 0.1
}

/// Finished blue logo, used when animation is off.
pub fn still() -> Duration {
    Duration::from_secs_f32(CYCLE / PLAYBACK_SPEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    fn at(seconds: f32) -> Vec<String> {
        let palette = Palette::new(ColorLevel::None, ThemeMode::Dark);
        text(&frame(
            Duration::from_secs_f32(seconds / PLAYBACK_SPEED),
            100,
            1,
            &palette,
        ))
    }

    #[test]
    fn letters_share_one_height_and_fit_tiers() {
        assert!(LETTERS
            .iter()
            .all(|letter| letter.iter().all(|row| row.len() == letter[0].len())));
        assert_eq!(fit(220, 40), Some((width_px() * 4, 2)));
        assert_eq!(fit(110, 30), Some((width_px() * 2, 1)));
        assert_eq!(fit(80, 24).map(|fit| fit.1), Some(1));
        assert_eq!(fit(30, 24), None);
        assert_eq!(fit(100, 8), None);
    }

    #[test]
    fn calm_is_still_light_and_random() {
        let first = at(0.2);
        assert_eq!(first, at(HOLD - 0.2));
        assert_eq!(first, at(CYCLE + HOLD / 2.0));
        assert!(first
            .iter()
            .all(|row| row.chars().all(|c| c == ' ' || REST.contains(&c))));
        let distinct: std::collections::BTreeSet<char> =
            first.iter().flat_map(|row| row.chars()).collect();
        assert!(distinct.len() >= 5);
        assert_eq!(first.len(), ROWS);
        assert!(first.iter().all(|row| row.chars().count() == 100));
    }

    #[test]
    fn a_single_wave_crosses_and_the_logo_settles() {
        let glints: Vec<char> = GLINT
            .iter()
            .flatten()
            .chain(DIGITS.iter())
            .copied()
            .collect();
        let lit_columns = |rows: &[String]| {
            let mut columns: Vec<usize> = rows
                .iter()
                .flat_map(|row| {
                    row.chars()
                        .enumerate()
                        .filter(|(_, c)| glints.contains(c))
                        .map(|(x, _)| x)
                })
                .collect();
            columns.sort_unstable();
            columns
        };
        // Mid-pass: a broad connected highlight covers several letters.
        let mid = lit_columns(&at(HOLD + PASS * 0.5));
        assert!(!mid.is_empty());
        let span = mid.last().unwrap() - mid.first().unwrap();
        assert!((20..90).contains(&span), "crest spans {span} columns");
        // The crest moves left to right.
        let early = lit_columns(&at(HOLD + PASS * 0.35));
        let late = lit_columns(&at(HOLD + PASS * 0.65));
        assert!(early[early.len() / 2] < late[late.len() / 2]);
        // After the pass the logo is exactly at rest again.
        assert_eq!(at(CYCLE + 0.5), at(0.5));
    }

    #[test]
    fn plays_once_then_keeps_original_texture_in_blue() {
        let palette = Palette::new(ColorLevel::TrueColor, ThemeMode::Dark);
        let done = frame(
            Duration::from_secs_f32(CYCLE / PLAYBACK_SPEED),
            100,
            1,
            &palette,
        );
        let later = frame(Duration::from_secs(100), 100, 1, &palette);
        assert_eq!(done, later);
        assert_eq!(text(&done), at(0.2));
        assert!(done
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.content != " ")
            .all(|span| span.style.fg == Some(Color::Rgb(120, 174, 255))));
        assert!(pass(CYCLE + HOLD + PASS / 2.0).is_none());
        assert!(!running(Duration::from_secs(100)));
    }

    #[test]
    fn pulse_follows_strokes_and_hops_to_a_lower_left() {
        let width = width_px();
        let map = arrivals();
        let at = |x, y| map[y * width + x];
        assert!(at(0, 7) < at(2, 9));
        assert!(at(2, 9) < at(7, 7));
        assert!(at(7, 7) < at(7, 0));
        assert!(at(7, 0) < at(0, 0));
        let a = LETTERS[0][0].len() + GAP;
        assert!(at(2, 9) < at(a, 9));
        assert!(at(a, 9) < at(0, 0), "A starts before J finishes");
        // Every successor begins while the previous letter is still being infected.
        let mut offset = 0;
        let mut previous_end = None;
        for letter in LETTERS {
            let times: Vec<f32> = (0..ROWS)
                .flat_map(|y| (offset..offset + letter[0].len()).map(move |x| at(x, y)))
                .filter(|time| time.is_finite())
                .collect();
            let start = times.iter().copied().fold(f32::INFINITY, f32::min);
            let end = times.iter().copied().fold(0.0, f32::max);
            if let Some(previous_end) = previous_end {
                assert!(start < previous_end);
            }
            previous_end = Some(end);
            offset += letter[0].len() + GAP;
        }
        assert!(at(a, 9) < at(a + 5, 0));
        assert!(at(a + 5, 0) < at(a + LETTERS[1][0].len() - 1, 9));
        for y in 0..ROWS {
            for x in 0..width {
                assert_eq!(map[y * width + x].is_finite(), ink(x, y));
            }
        }
        assert!(through(at(a, 9), at(a, 9)) > 0.99);
        assert!(through(at(a + 5, 0), at(a, 9)) < 0.01);
    }

    #[test]
    fn speckles_are_sparse_and_brief() {
        let mut flashing = 0;
        for y in 0..20 {
            for x in 0..100 {
                let peak = (0.11 - hash(x, y, 29) * 2.7).rem_euclid(2.7);
                if speckle(x, y, peak) > 0.9 {
                    flashing += 1;
                    assert_eq!(speckle(x, y, peak + 0.23), 0.0);
                }
            }
        }
        assert!((100..400).contains(&flashing));
    }

    #[test]
    fn frames_are_deterministic_and_move() {
        let peak = HOLD + PASS * 0.5;
        assert_eq!(at(peak), at(peak));
        assert_ne!(at(peak), at(peak + 0.5));
    }
}
