use super::*;

#[test]
fn grapheme_editing_handles_combining_zwj_flags_and_cjk() {
    let clusters = ["a", "e\u{301}", "👨‍👩‍👧‍👦", "🇨🇦", "中"];
    let mut c = Composer::default();
    c.set_text(&clusters.concat()).unwrap();
    key(&mut c, KeyCode::Home, KeyModifiers::NONE);
    let mut expected = 0;
    for g in clusters {
        key(&mut c, KeyCode::Right, KeyModifiers::NONE);
        expected += g.len();
        assert_eq!(c.cursor_byte(), expected);
    }
    for end in (0..clusters.len()).rev() {
        key(&mut c, KeyCode::Backspace, KeyModifiers::NONE);
        assert_eq!(c.text(), clusters[..end].concat());
    }
    c.set_text("ab").unwrap();
    key(&mut c, KeyCode::Left, KeyModifiers::NONE);
    key(&mut c, KeyCode::Char('\u{301}'), KeyModifiers::NONE);
    assert_eq!(c.text(), "a\u{301}b");
    assert_eq!(c.cursor_byte(), 3);
    key(&mut c, KeyCode::Left, KeyModifiers::NONE);
    assert_eq!(c.cursor_byte(), 0);
    key(&mut c, KeyCode::Delete, KeyModifiers::NONE);
    assert_eq!(c.text(), "b");
    c.set_text("").unwrap();
    for ch in "👩‍💻".chars() {
        key(&mut c, KeyCode::Char(ch), KeyModifiers::NONE);
    }
    key(&mut c, KeyCode::Left, KeyModifiers::NONE);
    assert_eq!(c.cursor_byte(), 0);
    key(&mut c, KeyCode::Delete, KeyModifiers::NONE);
    assert_eq!(c.text(), "");
}
#[test]
fn multiline_bindings_paste_submission_and_controls_are_separate() {
    let mut c = Composer::default();
    c.set_text("one two").unwrap();
    key(&mut c, KeyCode::Char('w'), KeyModifiers::CONTROL);
    assert_eq!(c.text(), "one ");
    key(&mut c, KeyCode::Char('u'), KeyModifiers::CONTROL);
    assert_eq!(c.text(), "");
    c.set_text("one two\nthree four").unwrap();
    key(&mut c, KeyCode::Char('a'), KeyModifiers::CONTROL);
    assert_eq!(c.cursor_byte(), 8);
    key(&mut c, KeyCode::Char('k'), KeyModifiers::CONTROL);
    assert_eq!(c.text(), "one two\n");
    key(&mut c, KeyCode::Char('e'), KeyModifiers::CONTROL);
    assert_eq!(c.cursor_byte(), c.text().len());
    assert_eq!(
        key(&mut c, KeyCode::Enter, KeyModifiers::ALT),
        ComposerAction::None
    );
    assert_eq!(
        key(&mut c, KeyCode::Enter, KeyModifiers::SHIFT),
        ComposerAction::None
    );
    assert_eq!(
        c.handle(
            Event::Paste("pasted\r\n\ttext\x1b[31m!\r".into()),
            &CompletionData::default()
        )
        .unwrap(),
        ComposerAction::None
    );
    assert!(c.text().ends_with("pasted\n\ttext!\n"));
    assert!(!c.text().contains('\x1b'));
    let text = c.text().to_owned();
    assert_eq!(
        key(&mut c, KeyCode::Enter, KeyModifiers::CONTROL),
        ComposerAction::None
    );
    assert_eq!(c.text(), text);
    assert_eq!(
        key(&mut c, KeyCode::Char('l'), KeyModifiers::CONTROL),
        ComposerAction::Redraw
    );
    assert_eq!(c.text(), text);
    assert_eq!(
        key(&mut c, KeyCode::Enter, KeyModifiers::NONE),
        ComposerAction::Submit(text)
    );
    assert_eq!(c.text(), "");
    assert_eq!(
        key(&mut c, KeyCode::Char('d'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
    assert_eq!(
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
    c.set_text("e\u{301}x").unwrap();
    key(&mut c, KeyCode::Home, KeyModifiers::NONE);
    key(&mut c, KeyCode::Char('d'), KeyModifiers::CONTROL);
    assert_eq!(c.text(), "x");
    assert_eq!(
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
    assert_eq!(c.text(), "x");
    c.set_text("one two three").unwrap();
    key(&mut c, KeyCode::Char('b'), KeyModifiers::ALT);
    assert_eq!(&c.text()[c.cursor_byte()..], "three");
    key(&mut c, KeyCode::Char('b'), KeyModifiers::ALT);
    assert_eq!(&c.text()[c.cursor_byte()..], "two three");
    key(&mut c, KeyCode::Char('f'), KeyModifiers::ALT);
    assert_eq!(&c.text()[c.cursor_byte()..], " three");
}
#[test]
fn ctrl_c_exits_with_a_draft_completion_or_history_search() {
    let mut c = Composer::default();
    c.set_text("unsent draft").unwrap();
    assert_eq!(
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
    assert_eq!(c.text(), "unsent draft");
    c.set_text("\\o").unwrap();
    key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
    assert!(c.completion().is_some());
    assert_eq!(
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
    key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    assert!(c.search.is_some());
    assert_eq!(
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
        ComposerAction::ExitRequested
    );
}
#[test]
fn completion_cycles_original_token_and_replaces_quoted_suffix() {
    let mut c = Composer::default();
    c.set_text("\\o").unwrap();
    key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
    let first = c.text().to_owned();
    assert!(c.completion().unwrap().0.len() >= 2);
    key(&mut c, KeyCode::BackTab, KeyModifiers::SHIFT);
    assert_ne!(c.text(), first);
    key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(c.text(), first);
    let before = c.view(60, 6);
    assert_eq!(
        c.handle(Event::Resize(60, 24), &CompletionData::default())
            .unwrap(),
        ComposerAction::Redraw
    );
    assert_eq!(c.view(60, 6), before);
    c.set_text("\\opeBAD other").unwrap();
    key(&mut c, KeyCode::Home, KeyModifiers::NONE);
    for _ in 0..4 {
        key(&mut c, KeyCode::Right, KeyModifiers::NONE);
    }
    key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(c.text(), "\\open other");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("folder/name with space.csv");
    let paths = [path];
    let data = CompletionData {
        paths: &paths,
        working_directory: Some(directory.path()),
        ..Default::default()
    };
    c.set_text("\\open \"folder/naBAD\" --name x").unwrap();
    c.cursor = "\\open \"folder/na".len();
    c.handle(
        Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
        &data,
    )
    .unwrap();
    assert!(c.text().contains("name with space.csv"));
    assert!(c.text().ends_with(" --name x"));
    assert!(!c.text().contains("BAD"));
}
