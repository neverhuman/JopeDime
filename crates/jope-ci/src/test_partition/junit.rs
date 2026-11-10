//! A minimal, fail-closed XML reader for nextest's JUnit reports.
//!
//! The Python read them with `xml.etree.ElementTree`; this crate adds no XML dependency (a
//! `Cargo.lock` change mints a new ADR-0004 evaluator identity), so this reader keeps exactly what
//! the partition proof needs: the element tree, its names and attributes. Text is checked (entity
//! references must be known) and discarded. A malformed document is refused, as expat refused it;
//! an internal DTD subset, which nextest never writes, is refused as unsupported.

/// One element: its tag, attributes (entity references resolved) and child elements.
#[derive(Debug)]
pub struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Element>,
}

impl Element {
    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// `element.find(name) is not None`: a direct child with this tag.
    pub fn has_child(&self, name: &str) -> bool {
        self.children.iter().any(|child| child.name == name)
    }

    /// `element.iter(name)`: this element and its descendants with this tag, in document order.
    pub fn iter<'a>(&'a self, name: &str, out: &mut Vec<&'a Element>) {
        if self.name == name {
            out.push(self);
        }
        for child in &self.children {
            child.iter(name, out);
        }
    }
}

struct Reader<'a> {
    text: &'a str,
    pos: usize,
}

/// Parse a whole document: optional prolog, exactly one root element, then only whitespace,
/// comments and processing instructions.
pub fn parse(text: &str) -> Result<Element, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut reader = Reader { text, pos: 0 };
    reader.misc(true)?;
    if reader.at_end() {
        return Err("no element found".to_owned());
    }
    let root = reader.element()?;
    reader.misc(false)?;
    if !reader.at_end() {
        return Err(reader.error("junk after document element"));
    }
    Ok(root)
}

impl Reader<'_> {
    fn rest(&self) -> &str {
        &self.text[self.pos..]
    }

    fn at_end(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn error(&self, what: &str) -> String {
        format!("not well-formed ({what}) at byte {}", self.pos)
    }

    fn skip_whitespace(&mut self) {
        let rest = self.rest();
        let skipped = rest.len() - rest.trim_start_matches([' ', '\t', '\r', '\n']).len();
        self.pos += skipped;
    }

    /// Skip to just past `end`, or refuse an unterminated construct.
    fn skip_past(&mut self, end: &str, what: &str) -> Result<(), String> {
        match self.rest().find(end) {
            Some(at) => {
                self.pos += at + end.len();
                Ok(())
            }
            None => Err(self.error(&format!("unclosed {what}"))),
        }
    }

    /// Whitespace, comments and processing instructions outside the root (and, before it, the
    /// XML declaration and a DOCTYPE without an internal subset).
    fn misc(&mut self, prolog: bool) -> Result<(), String> {
        loop {
            self.skip_whitespace();
            let rest = self.rest();
            if rest.starts_with("<!--") {
                self.skip_past("-->", "comment")?;
            } else if rest.starts_with("<?") {
                self.skip_past("?>", "processing instruction")?;
            } else if prolog && rest.starts_with("<!DOCTYPE") {
                let end = rest
                    .find('>')
                    .ok_or_else(|| self.error("unclosed DOCTYPE"))?;
                if rest[..end].contains('[') {
                    return Err(self.error("internal DTD subset is not supported"));
                }
                self.pos += end + 1;
            } else if self.at_end() || rest.starts_with('<') {
                return Ok(());
            } else {
                return Err(self.error("syntax error"));
            }
        }
    }

    fn name(&mut self) -> Result<String, String> {
        let rest = self.rest();
        let end = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '/' | '>' | '=' | '<'))
            .unwrap_or(rest.len());
        if end == 0 {
            return Err(self.error("invalid token"));
        }
        let name = rest[..end].to_owned();
        self.pos += end;
        Ok(name)
    }

    /// An element starting at `<`, through its matching end tag.
    fn element(&mut self) -> Result<Element, String> {
        if !self.rest().starts_with('<') {
            return Err(self.error("syntax error"));
        }
        self.pos += 1;
        let name = self.name()?;
        let mut attributes: Vec<(String, String)> = vec![];
        loop {
            let before = self.pos;
            self.skip_whitespace();
            let rest = self.rest();
            if rest.starts_with("/>") {
                self.pos += 2;
                return Ok(Element {
                    name,
                    attributes,
                    children: vec![],
                });
            }
            if rest.starts_with('>') {
                self.pos += 1;
                break;
            }
            if self.pos == before {
                return Err(self.error("invalid token"));
            }
            let key = self.name()?;
            self.skip_whitespace();
            if !self.rest().starts_with('=') {
                return Err(self.error("invalid token"));
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.quoted()?;
            if attributes.iter().any(|(existing, _)| *existing == key) {
                return Err(self.error("duplicate attribute"));
            }
            attributes.push((key, value));
        }
        let mut children = vec![];
        loop {
            let rest = self.rest();
            if rest.starts_with("</") {
                self.pos += 2;
                let end = self.name()?;
                self.skip_whitespace();
                if end != name || !self.rest().starts_with('>') {
                    return Err(self.error("mismatched tag"));
                }
                self.pos += 1;
                return Ok(Element {
                    name,
                    attributes,
                    children,
                });
            } else if rest.starts_with("<!--") {
                self.skip_past("-->", "comment")?;
            } else if rest.starts_with("<![CDATA[") {
                self.skip_past("]]>", "CDATA section")?;
            } else if rest.starts_with("<?") {
                self.skip_past("?>", "processing instruction")?;
            } else if rest.starts_with('<') {
                children.push(self.element()?);
            } else if rest.is_empty() {
                return Err(self.error("unclosed token"));
            } else {
                let end = rest.find('<').unwrap_or(rest.len());
                unescape(&rest[..end]).map_err(|what| self.error(&what))?;
                self.pos += end;
            }
        }
    }

    fn quoted(&mut self) -> Result<String, String> {
        let rest = self.rest();
        let quote = match rest.chars().next() {
            Some(q @ ('"' | '\'')) => q,
            _ => return Err(self.error("invalid token")),
        };
        let end = rest[1..]
            .find(quote)
            .ok_or_else(|| self.error("unclosed token"))?;
        let raw = &rest[1..=end];
        if raw.contains('<') {
            return Err(self.error("invalid token"));
        }
        let value = unescape(raw).map_err(|what| self.error(&what))?;
        self.pos += end + 2;
        Ok(value)
    }
}

/// Resolve the predefined entities and character references; any other `&` is refused.
fn unescape(raw: &str) -> Result<String, String> {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let end = tail
            .find(';')
            .ok_or_else(|| "not well-formed (invalid token)".to_owned())?;
        let entity = &tail[..end];
        let resolved = match entity {
            "lt" => '<',
            "gt" => '>',
            "amp" => '&',
            "quot" => '"',
            "apos" => '\'',
            _ => {
                let code = if let Some(hex) = entity.strip_prefix("#x") {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(decimal) = entity.strip_prefix('#') {
                    decimal.parse().ok()
                } else {
                    return Err(format!("undefined entity &{entity};"));
                };
                code.and_then(char::from_u32)
                    .filter(|c| *c != '\0')
                    .ok_or_else(|| format!("reference to invalid character number &{entity};"))?
            }
        };
        out.push(resolved);
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested_cases_attributes_and_children() {
        let root = parse(concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- report -->",
            "<testsuites><testsuite name='s'>",
            "<testcase classname=\"a&amp;b\" name=\"t&#x31;\"><failure message=\"x\">",
            "<![CDATA[<skipped/> & raw]]></failure><system-out>a &lt; b</system-out></testcase>",
            "<testcase classname=\"c\" name=\"t2\"><skipped/></testcase>",
            "</testsuite></testsuites>\n"
        ))
        .unwrap();
        let mut cases = vec![];
        root.iter("testcase", &mut cases);
        assert_eq!(2, cases.len());
        assert_eq!(Some("a&b"), cases[0].attribute("classname"));
        assert_eq!(Some("t1"), cases[0].attribute("name"));
        assert!(cases[0].has_child("failure") && !cases[0].has_child("skipped"));
        assert!(cases[1].has_child("skipped"));
    }

    #[test]
    fn refuses_malformed_documents() {
        for bad in [
            "",
            "   ",
            "<a>",
            "<a></b>",
            "<a/><b/>",
            "<a x='1' x='2'/>",
            "<a x='<'/>",
            "<a>&bogus;</a>",
            "<a x=1/>",
            "text<a/>",
            "<!DOCTYPE a [<!ENTITY e 'x'>]><a/>",
            "<a><!-- open</a>",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} parsed");
        }
    }
}
