//! Terminal geometry derived from VEOX's existing 79 x 20 logo.svg paths.
//! Cubic outlines are flattened to eight segments per curve; even/odd fill keeps O open.
use super::super::{ColorLevel, Theme, ThemeMode};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

const OUTLINES: &[&[(f32, f32)]] = &[
    &[
        (50.2769, 0.70312),
        (51.2734, 0.74279),
        (52.2243, 0.86178),
        (53.1296, 1.06009),
        (53.9891, 1.33773),
        (54.8029, 1.6947),
        (55.571, 2.131),
        (56.2933, 2.64662),
        (56.9698, 3.24157),
        (57.5827, 3.90195),
        (58.1138, 4.61387),
        (58.5632, 5.37731),
        (58.9309, 6.19226),
        (59.2168, 7.05871),
        (59.421, 7.97666),
        (59.5436, 8.94609),
        (59.5844, 9.96699),
        (59.5436, 10.9879),
        (59.421, 11.9574),
        (59.2168, 12.8753),
        (58.9309, 13.7418),
        (58.5632, 14.5567),
        (58.1138, 15.3202),
        (57.5827, 16.0321),
        (56.9698, 16.6924),
        (56.2933, 17.2874),
        (55.571, 17.803),
        (54.8029, 18.2393),
        (53.9891, 18.5963),
        (53.1296, 18.8739),
        (52.2243, 19.0723),
        (51.2734, 19.1912),
        (50.2769, 19.2309),
        (49.2792, 19.1912),
        (48.3256, 19.0723),
        (47.4164, 18.8739),
        (46.5525, 18.5963),
        (45.7343, 18.2393),
        (44.9624, 17.803),
        (44.2375, 17.2874),
        (43.5601, 16.6924),
        (42.9473, 16.0321),
        (42.4161, 15.3202),
        (41.9667, 14.5567),
        (41.599, 13.7418),
        (41.313, 12.8753),
        (41.1087, 11.9574),
        (40.9862, 10.9879),
        (40.9453, 9.96699),
        (40.9862, 8.94612),
        (41.1087, 7.97677),
        (41.313, 7.05893),
        (41.599, 6.19257),
        (41.9667, 5.37767),
        (42.4161, 4.61422),
        (42.9473, 3.90219),
        (43.5601, 3.24157),
        (44.237, 2.64662),
        (44.9604, 2.131),
        (45.7302, 1.6947),
        (46.5466, 1.33773),
        (47.4094, 1.06009),
        (48.3187, 0.86178),
        (49.2745, 0.74279),
        (50.2769, 0.70312),
    ],
    &[
        (50.2769, 3.99939),
        (49.6682, 4.02439),
        (49.0894, 4.09938),
        (48.5407, 4.22437),
        (48.0219, 4.39935),
        (47.5331, 4.62433),
        (47.0743, 4.8993),
        (46.6454, 5.22427),
        (46.2465, 5.59923),
        (45.8866, 6.01746),
        (45.5747, 6.47219),
        (45.3109, 6.96341),
        (45.095, 7.4911),
        (44.9272, 8.05524),
        (44.8074, 8.65582),
        (44.7355, 9.29282),
        (44.7115, 9.96621),
        (44.7355, 10.6388),
        (44.8075, 11.2736),
        (44.9274, 11.8713),
        (45.0953, 12.4324),
        (45.3112, 12.9575),
        (45.575, 13.447),
        (45.8868, 13.9015),
        (46.2465, 14.3216),
        (46.6452, 14.6995),
        (47.0739, 15.0269),
        (47.5327, 15.304),
        (48.0216, 15.5307),
        (48.5404, 15.7071),
        (49.0893, 15.833),
        (49.6681, 15.9086),
        (50.2769, 15.9338),
        (50.8844, 15.9086),
        (51.4599, 15.833),
        (52.0038, 15.7071),
        (52.5169, 15.5307),
        (52.9996, 15.304),
        (53.4525, 15.0269),
        (53.8762, 14.6995),
        (54.2713, 14.3216),
        (54.6283, 13.9009),
        (54.9377, 13.4449),
        (55.1995, 12.9535),
        (55.4137, 12.4268),
        (55.5804, 11.8648),
        (55.6994, 11.2673),
        (55.7708, 10.6345),
        (55.7946, 9.96621),
        (55.7708, 9.29709),
        (55.6994, 8.6621),
        (55.5804, 8.06178),
        (55.4137, 7.49668),
        (55.1995, 6.96733),
        (54.9377, 6.47428),
        (54.6283, 6.01807),
        (54.2713, 5.59923),
        (53.8755, 5.22427),
        (53.4502, 4.8993),
        (52.9952, 4.62433),
        (52.5107, 4.39935),
        (51.9966, 4.22437),
        (51.453, 4.09938),
        (50.8797, 4.02439),
        (50.2769, 3.99939),
    ],
    &[
        (78.0923, 0.9668),
        (72.89, 9.7747),
        (78.4025, 18.87),
        (74.1313, 18.87),
        (70.1708, 12.216),
        (66.1853, 18.87),
        (61.9141, 18.87),
        (67.426, 9.7747),
        (62.2232, 0.9668),
        (66.471, 0.9668),
        (70.1697, 7.30955),
        (73.845, 0.9668),
        (78.0923, 0.9668),
    ],
    &[
        (36.7145, 4.22193),
        (36.7145, 0.9668),
        (22.9258, 0.9668),
        (22.9258, 18.87),
        (36.7145, 18.87),
        (36.7145, 15.5911),
        (26.6247, 15.5911),
        (26.6247, 11.2832),
        (36.2475, 11.2832),
        (36.2475, 8.09949),
        (26.6247, 8.09949),
        (26.6247, 4.22193),
        (36.7145, 4.22193),
    ],
    &[
        (9.52286, 19.4946),
        (11.6876, 14.9949),
        (11.693, 14.9765),
        (11.7011, 14.9574),
        (4.97045, 0.9668),
        (0.60938, 0.9668),
        (9.52286, 19.4946),
    ],
    &[
        (13.99, 10.2008),
        (9.62891, 10.2008),
        (14.0713, 0.9668),
        (18.4323, 0.9668),
        (13.99, 10.2008),
    ],
];

fn inside(x: f32, y: f32) -> bool {
    let mut filled = false;
    for outline in OUTLINES {
        let mut previous = outline[outline.len() - 1];
        for &point in *outline {
            if (point.1 > y) != (previous.1 > y)
                && x < (previous.0 - point.0) * (y - point.1) / (previous.1 - point.1) + point.0
            {
                filled = !filled;
            }
            previous = point;
        }
    }
    filled
}

fn wave(x: f32, y: f32, seconds: f32, animated: bool) -> f32 {
    if !animated {
        return 0.0;
    }
    let sweep = (seconds % 7.2) / 5.2 * 1.6 - 0.3;
    let distance =
        x - sweep + (y - 0.5) * 0.10 + 0.055 * (y * std::f32::consts::TAU - seconds * 0.9).sin();
    let pulse = 0.88 + 0.12 * (seconds * std::f32::consts::TAU / 2.4).sin();
    let crest = (-(distance / 0.15).powi(2)).exp() * pulse;
    let wake = 0.16 * (-((distance + 0.23) / 0.11).powi(2)).exp();
    (crest + wake).min(1.0)
}

fn character(coverage: u8, energy: f32) -> char {
    if coverage == 0 {
        return ' ';
    }
    // Quiet punctuation becomes denser ASCII at the wave's crest. Occupancy and
    // the brand silhouette stay fixed; only the material within the strokes changes.
    const INK: [char; 6] = ['.', ':', '+', '*', '#', '@'];
    let base = usize::from(coverage >= 3);
    INK[base + (energy * 4.0).round() as usize]
}

fn ink(theme: Theme, energy: f32) -> Style {
    if theme.color_level() == ColorLevel::None {
        return Style::default();
    }
    let light = theme.mode() == ThemeMode::Light;
    let color = match theme.color_level() {
        ColorLevel::TrueColor => {
            let (base, peak) = if light {
                ([168.0, 174.0, 181.0], [56.0, 66.0, 79.0])
            } else {
                ([49.0, 60.0, 73.0], [235.0, 231.0, 219.0])
            };
            let rgb =
                std::array::from_fn::<_, 3, _>(|i| (base[i] + (peak[i] - base[i]) * energy) as u8);
            Color::Rgb(rgb[0], rgb[1], rgb[2])
        }
        ColorLevel::Ansi256 => Color::Indexed(if light {
            248 - (energy * 11.0) as u8
        } else {
            237 + (energy * 18.0) as u8
        }),
        _ => Color::DarkGray,
    };
    let style = Style::default().fg(color);
    if energy > 0.55 {
        style.add_modifier(Modifier::BOLD)
    } else {
        style
    }
}

#[derive(Default)]
pub(super) struct Monument {
    size: (u16, u16),
    mask: Vec<Vec<u8>>,
}

impl Monument {
    pub(super) fn draw(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        theme: Theme,
        seconds: f32,
        animated: bool,
    ) {
        // Cache vector coverage until resize; only ASCII density and ink animate.
        if self.size != (area.width, area.height) {
            self.size = (area.width, area.height);
            self.mask = (0..area.height)
                .map(|y| {
                    (0..area.width)
                        .map(|x| {
                            let mut mask = 0;
                            for sy in 0..2 {
                                for sx in 0..2 {
                                    let px = (f32::from(x) + (sx as f32 + 0.5) / 2.0)
                                        / f32::from(area.width)
                                        * 79.0;
                                    let py = (f32::from(y) + (sy as f32 + 0.5) / 2.0)
                                        / f32::from(area.height)
                                        * 20.0;
                                    if inside(px, py) {
                                        mask |= 1 << (sy * 2 + sx);
                                    }
                                }
                            }
                            (mask as u8).count_ones() as u8
                        })
                        .collect()
                })
                .collect();
        }
        let lines = self
            .mask
            .iter()
            .enumerate()
            .map(|(y, row)| {
                Line::from(
                    row.iter()
                        .enumerate()
                        .map(|(x, glyph)| {
                            let energy = wave(
                                x as f32 / f32::from(area.width),
                                y as f32 / f32::from(area.height),
                                seconds,
                                animated,
                            );
                            Span::styled(
                                character(*glyph, energy).to_string(),
                                if *glyph == 0 {
                                    Style::default()
                                } else {
                                    ink(theme, energy)
                                },
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(lines), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn animated_frames_preserve_geometry_and_resize_rebuilds_it() {
        use ratatui::{backend::TestBackend, Terminal};
        let mut terminal = Terminal::new(TestBackend::new(100, 14)).unwrap();
        let mut logo = Monument::default();
        let theme = Theme::new(ThemeMode::Dark, ColorLevel::TrueColor, None);
        terminal
            .draw(|frame| logo.draw(frame, frame.area(), theme, 0.0, true))
            .unwrap();
        let first = terminal.backend().buffer().clone();
        terminal
            .draw(|frame| logo.draw(frame, frame.area(), theme, 2.3, true))
            .unwrap();
        let second = terminal.backend().buffer();
        assert!(first
            .content
            .iter()
            .zip(&second.content)
            .all(|(a, b)| (a.symbol() == " ") == (b.symbol() == " ")));
        assert!(first
            .content
            .iter()
            .zip(&second.content)
            .any(|(a, b)| a.fg != b.fg));
        assert!(
            first
                .content
                .iter()
                .zip(&second.content)
                .any(|(a, b)| a.symbol() != b.symbol()),
            "wave must change ASCII density"
        );
        assert!(
            second
                .content
                .iter()
                .any(|cell| cell.modifier.contains(Modifier::BOLD)),
            "wave crest must become bold"
        );
        assert!(second.content.iter().all(|cell| cell.symbol().is_ascii()));
        terminal.backend_mut().resize(80, 10);
        terminal
            .draw(|frame| logo.draw(frame, frame.area(), theme, 2.3, true))
            .unwrap();
        assert_eq!(logo.size, (80, 10));
        assert_eq!(logo.mask.len(), 10);
    }

    #[test]
    fn shimmer_changes_ink_while_monochrome_stays_static() {
        let color = Theme::new(ThemeMode::Dark, ColorLevel::TrueColor, None);
        assert_ne!(
            ink(color, wave(0.5, 0.5, 0.0, true)),
            ink(color, wave(0.5, 0.5, 2.3, true))
        );
        assert_eq!(
            ink(color, wave(0.5, 0.5, 0.0, false)),
            ink(color, wave(0.5, 0.5, 2.3, false))
        );
        let mono = Theme::new(ThemeMode::Dark, ColorLevel::None, None);
        assert_eq!(ink(mono, wave(0.5, 0.5, 2.3, true)), Style::default());
        assert!(inside(42.5, 10.0));
        assert!(!inside(50.0, 10.0), "O counter must stay open");
    }
}
