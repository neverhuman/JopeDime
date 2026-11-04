use super::*;

#[test]
fn view_tracks_cells_scrolls_and_never_changes_edit_state() {
    let mut c = Composer::default();
    c.set_text("e\u{301}中👩‍💻").unwrap();
    assert_eq!(c.view(40, 4).cursor, (7, 0));
    c.set_text("first\nsecond\nthird\nfourth\nfifth").unwrap();
    let text = c.text().to_owned();
    let cursor = c.cursor_byte();
    let view = c.view(40, 4);
    assert_eq!(view.cursor.1, 3);
    assert!(view.lines.last().unwrap().ends_with("fifth"));
    for width in [0, 1, 2, 3, 10, 80] {
        let v = c.view(width, 4);
        if width > 0 {
            assert!(v.cursor.0 < width);
        }
        assert_eq!(c.text(), text);
        assert_eq!(c.cursor_byte(), cursor);
    }
    c.set_text("abcd").unwrap();
    assert_eq!(c.view(6, 4).cursor, (2, 1));
    c.set_text("abcd\nz").unwrap();
    c.cursor = 4;
    assert_eq!(c.view(6, 4).cursor, (2, 1));
}
