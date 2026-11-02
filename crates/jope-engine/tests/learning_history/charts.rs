use super::*;

#[test]
fn navigation_reaches_both_ends_and_all_view_sizes_remain_usable() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    for i in 0..30 {
        let p = format!("runs/train-{i:02}");
        append(root, &p, &[row(&p, 1, 100 + i), row(&p, 2, 101 + i)]);
    }
    let snapshot = Cache::default().refresh(root);
    let mut nav = Navigation::default();
    nav.edge(&snapshot, false);
    assert_eq!(nav.selected.as_ref(), Some(&snapshot.runs[0].path));
    nav.move_by(&snapshot, -20);
    assert_eq!(nav.selected.as_ref(), Some(&snapshot.runs[0].path));
    nav.edge(&snapshot, true);
    assert_eq!(nav.selected.as_ref(), Some(&snapshot.runs[29].path));
    nav.move_by(&snapshot, 20);
    assert_eq!(nav.selected.as_ref(), Some(&snapshot.runs[29].path));
    let mut history = History::default();
    history.learning = Arc::new(snapshot);
    history.learning_navigation = nav;
    for (width, height) in [(160, 55), (100, 36), (48, 15), (20, 8)] {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    2,
                    &Hardware::default(),
                    jope_parallel::dashboard::DashboardData {
                        gp: &Value::Null,
                        neural: &Value::Null,
                        run: &json!({}),
                        events: "",
                        history: &history,
                    },
                )
            })
            .unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        if width >= 48 {
            assert!(text.contains("Sequence NLL"));
            assert!(text.contains("Deep learning"));
        }
        if width >= 100 {
            assert!(text.contains("Validation is adaptive"));
            assert!(text.contains("unbound validation rows"));
        }
    }
}

#[test]
fn full_history_plot_retains_spikes_and_both_endpoints() {
    let mut points = (0..10000).map(|i| (i as f64, 1.)).collect::<Vec<_>>();
    points[7133].1 = 100.;
    points[9999].1 = 0.1;
    let plot = envelope(&points, 80, [0., 9999.]);
    assert!(plot.len() <= 162);
    assert!(plot.contains(&points[0]));
    assert!(plot.contains(&points[7133]));
    assert!(plot.contains(&points[9999]));
}

#[test]
fn moving_average_follows_the_global_trend_and_damps_a_spike() {
    let mut points = (0..100)
        .map(|i| (i as f64, 10.0 - i as f64 * 0.05))
        .collect::<Vec<_>>();
    points[50].1 = 80.0;
    let smooth = moving_average(&points);
    assert_eq!(smooth.first().map(|p| p.0), Some(0.0));
    assert!((smooth.first().unwrap().1 - 10.0).abs() < 1.0);
    assert!((smooth.last().unwrap().1 - points.last().unwrap().1).abs() < 1.0);
    let mid = smooth.iter().find(|p| (p.0 - 50.0).abs() < 0.1).unwrap();
    assert!(
        mid.1 < 20.0,
        "spike should not dominate the global line: {}",
        mid.1
    );
    assert!(smooth.windows(2).all(|w| w[0].0 <= w[1].0));
}

#[test]
fn selected_run_keeps_the_three_global_history_charts() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let mut cycles = Vec::new();
    for i in 0..4 {
        let cycle = format!("runs/campaign/cycle-{i:04}");
        write(&root.join(&cycle).join("cycle.json"), json!({}));
        let path = format!("{cycle}/train-2");
        write(&root.join(&path).join("complete.json"), json!({"step":2}));
        append(
            root,
            &path,
            &[
                row(&path, 1, 1_000 + i * 3_600),
                row(&path, 2, 1_100 + i * 3_600),
            ],
        );
        cycles.push(json!({"cycle":cycle}));
    }
    write(
        &root.join("runs/campaign/completed-cycles.json"),
        json!({"cycles":cycles}),
    );
    let snapshot = Cache::default().refresh(root);
    let mut nav = Navigation::default();
    nav.edge(&snapshot, true);
    let mut history = History::default();
    history.learning = Arc::new(snapshot);
    history.learning_navigation = nav;
    let mut terminal = Terminal::new(TestBackend::new(160, 48)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                2,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &Value::Null,
                    neural: &Value::Null,
                    run: &json!({}),
                    events: "",
                    history: &history,
                },
            )
        })
        .unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|c| c.symbol())
        .collect::<String>();
    assert!(text.contains("global SMA"));
    assert!(text.contains("hours since first saved"));
    assert!(!text.contains("minutes since run began"));
    assert!(text.contains("Training Sequence NLL"));
    assert!(text.contains("Validation Sequence NLL"));
    assert!(text.contains("Validation pairwise loss"));
    let line = text.chars().filter(|c| "─│╭╮╰╯╱╲".contains(*c)).count();
    assert!(
        line >= 40,
        "expected a connected trend line, found {line} stroke cells in:\n{text}"
    );
}

#[test]
fn trend_row_uses_the_chart_height_for_a_large_loss_drop() {
    let height = 10;
    let top = trend_row(8.0, 0.4, 8.0, height, true);
    let bot = trend_row(0.4, 0.4, 8.0, height, true);
    assert_eq!(top, 0);
    assert_eq!(bot, 9);
    assert!(trend_row(1.0, 0.4, 8.0, height, true) < 8);
}

#[test]
fn column_line_covers_every_horizontal_cell() {
    let points = (0..40)
        .map(|i| (i as f64, 8.0 - i as f64 * 0.1))
        .collect::<Vec<_>>();
    let line = column_line(&points, 20, [0., 39.]);
    assert_eq!(line.len(), 20);
    assert!((line[0].0 - 0.0).abs() < 1e-9);
    assert!((line[19].0 - 39.0).abs() < 1e-9);
    assert!(line.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(line[0].1 > line[19].1);
}
