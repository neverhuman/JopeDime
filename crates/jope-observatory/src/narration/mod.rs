//! Read-only delivery of retained search commentary; no search-control authority.
mod http;
mod index;
mod mcp;
use anyhow::Result;
pub use http::{router, serve, validate_listen, App};
pub use index::{
    sequence, Index, Limits, Page, Status, MAX_PAGE_BYTES, MAX_PAGE_RECORDS, MAX_RECORD_BYTES,
};
use jope_narrator::Cursor;
/// Serve the selected narration index over newline-delimited MCP JSON-RPC.
pub use mcp::mcp;
use std::io::Write;

/// Escape terminal controls without modifying retained evidence or JSON text.
pub fn terminal_text(text: &str) -> String {
    let mut output = String::new();
    for c in text.chars() {
        if c.is_control() && c != '\n' && c != '\t' {
            output.push_str(&format!("\\u{{{:x}}}", c as u32));
        } else {
            output.push(c);
        }
    }
    output
}

/// Write one validated page as JSONL or terminal-safe narration text.
pub fn print_page(
    index: &mut Index,
    after: &Cursor,
    limit: usize,
    text: bool,
    out: &mut impl Write,
) -> Result<Page> {
    let page = index.page(after, limit, MAX_PAGE_BYTES)?;
    for event in &page.events {
        if text {
            writeln!(out, "{}", terminal_text(&event.narration.text))?;
        } else {
            serde_json::to_writer(&mut *out, event)?;
            out.write_all(b"\n")?;
        }
    }
    out.flush()?;
    Ok(page)
}
