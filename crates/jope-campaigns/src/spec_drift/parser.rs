use super::*;

/// Strip whole-line comments so a commented-out value is never read as a
/// reservation. This is load-bearing: `python/test-support/jope/research/test_exact_ops.py` documents the
/// old 49152 value in a comment directly above the live 12288, and a naive
/// scanner reports the comment.
pub(super) fn strip_comments(src: &str, is_py: bool) -> Vec<String> {
    src.lines()
        .map(|ln| {
            let t = ln.trim_start();
            let is_comment = if is_py {
                t.starts_with('#')
            } else {
                t.starts_with("//")
            };
            if is_comment {
                String::new()
            } else {
                ln.to_owned()
            }
        })
        .collect()
}

/// Rust `#[cfg(test)]` modules assert against these same literals, so scanning
/// them yields phantom render sites. Cut the file at the test module.
pub(super) fn cut_test_module(src: &str) -> &str {
    // Test-only helper functions may live beside production renderers. Only
    // cut at the actual test module; cutting at the first `#[cfg(test)]`
    // silently hid every production renderer below such a helper.
    for marker in ["\n#[cfg(test)]\nmod tests", "\nmod tests"] {
        if let Some(i) = src.find(marker) {
            return &src[..i];
        }
    }
    src
}

/// Value of `key` in a window of rendered-TOML-inside-source-code.
///
/// Handles all three literal styles in this tree simultaneously:
///   Python list item : `"ram_mb = 12288",`
///   Rust owned string: `"ram_mb = 49152".to_owned(),`
///   Rust raw block   : `ram_mb = 16384\n\`
pub(super) fn find_value(window: &str, key: &str) -> Option<String> {
    for ln in window.lines() {
        // Every occurrence, not just the first: `placement = "single_node"`
        // contains "node", and stopping at the first hit would hide a real
        // `node = "xbabe2"` later on the same line.
        let mut from = 0usize;
        while let Some(rel) = ln[from..].find(key) {
            let pos = from + rel;
            from = pos + key.len();
            // Reject substring hits such as `packed_ram_mb` or `single_node`.
            if pos > 0 {
                let prev = ln.as_bytes()[pos - 1];
                if prev == b'_' || prev.is_ascii_alphanumeric() {
                    continue;
                }
            }
            let rest = ln[pos + key.len()..].trim_start();
            let Some(rest) = rest.strip_prefix('=') else {
                continue;
            };
            let v: String = rest
                .trim_start()
                .trim_start_matches('\\')
                .trim_start_matches('"')
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

/// The `[resource]` section of an unescaped window: everything from the
/// `[resource]` header to the next TOML section header.
///
/// Needed because a bare `find_value(window, "node")` is not a reliable test for
/// a `[resource] node` pin. It matched Rust `format!` NAMED ARGUMENTS that
/// happen to be spelled `node = node,` below the template, and missed the real
/// pin whenever the value was a placeholder (`node = {node:?}`), whose `{` makes
/// the extracted value empty. Measured on this tree: three of the five
/// `crates/jope-campaigns/src/test_adaptive` render sites emit `[resource] node` and were NOT flagged,
/// while the two that were flagged matched on the format-argument list rather
/// than on the spec text. Scoping the test to the section, and looking for the
/// key at the start of a line, is what TOML actually means.
pub(super) fn resource_section(window: &str) -> &str {
    let Some(start) = window.find("[resource]") else {
        return "";
    };
    let rest = &window[start + "[resource]".len()..];
    match rest.find("\n[") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// Is `key` assigned at the start of a line inside this section?
pub(super) fn section_declares(section: &str, key: &str) -> bool {
    section.lines().any(|ln| {
        let t = ln.trim();
        t.strip_prefix(key)
            .is_some_and(|r| r.trim_start().starts_with('='))
    })
}

pub(super) fn enclosing_function(lines: &[String], idx: usize, is_py: bool) -> String {
    for i in (0..=idx).rev() {
        let t = lines[i].trim_start();
        let rest = if is_py {
            t.strip_prefix("def ")
        } else {
            t.strip_prefix("fn ")
                .or_else(|| t.strip_prefix("pub fn "))
                .or_else(|| t.strip_prefix("async fn "))
                .or_else(|| t.strip_prefix("pub async fn "))
        };
        if let Some(rest) = rest {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return name;
            }
        }
    }
    "<module>".to_owned()
}

/// `Some(entry)` when the site is registered; the inner `Option<Family>` is the
/// family it belongs to, or `None` for a deliberately out-of-scope renderer.
pub(super) fn registry_entry(file: &str, function: &str) -> Option<Option<Family>> {
    REGISTRY
        .iter()
        .find(|(f, fun, _)| *f == file && *fun == function)
        .map(|(_, _, fam)| *fam)
}
