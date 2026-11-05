//! A small, dependency-free PDF of a local run's result, for the workspace's Report tab.
//!
//! Text only (Helvetica, Latin-1), one or more Letter pages. It carries the run's identity,
//! the selected model and its measured scores — never a row of the dataset.

/// Lines of the report for a finished local run.
pub fn lines(
    snapshot: &jope_api::RunSnapshot,
    finished: Option<&jope_api::RunFinishedBody>,
    engine_version: &str,
    generated_at: &str,
) -> Vec<String> {
    let value = serde_json::to_value(snapshot).unwrap_or_default();
    let word = |v: &serde_json::Value| v.as_str().unwrap_or("?").to_owned();
    let score = |v: Option<f64>| {
        v.filter(|v| v.is_finite())
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "not measured".into())
    };
    let mut lines = vec![
        "jAIn run report".to_owned(),
        String::new(),
        format!("Run: {}", snapshot.run_id),
        format!(
            "State: {} ({})",
            word(&value["state"]),
            word(&value["phase"])
        ),
        format!("Effort: {}", word(&value["effort"])),
        format!(
            "Evaluations: {} · elapsed {} s",
            snapshot.budget_used.evaluations.0,
            word(&value["budget_used"]["wall_seconds"])
        ),
        format!("Engine: JopeDime {engine_version} (local run on the user's machine)"),
        format!("Generated: {generated_at}"),
        String::new(),
    ];
    match &snapshot.best {
        Some(best) => {
            lines.push(format!("Selected model: {}", best.algorithm_id));
            lines.push(format!(
                "Cross-validated {}: {}",
                crate::terminal::sanitize_user_string(&best.metric),
                score(Some(best.cv_value))
            ));
        }
        None => lines.push("No model was selected.".into()),
    }
    if let Some(finished) = finished {
        let reason = serde_json::to_value(finished.completion_reason).unwrap_or_default();
        lines.push(format!("Finished: {}", word(&reason)));
        lines.push(format!("CV score: {}", score(finished.scores.cv)));
        lines.push(format!("Holdout score: {}", score(finished.scores.holdout)));
        lines.push(format!("Test score: {}", score(finished.scores.test)));
    }
    if !snapshot.warnings.is_empty() {
        lines.push(String::new());
        lines.push("Warnings:".into());
        for warning in &snapshot.warnings {
            lines.push(format!(
                "  {}: {}",
                crate::terminal::sanitize_user_string(&warning.code),
                crate::terminal::sanitize_user_string(&warning.message)
            ));
        }
    }
    lines.push(String::new());
    lines.push(
        "Scores are the engine's own measurements on this machine; the workspace did not \
         re-evaluate them."
            .into(),
    );
    lines
}

fn escape(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '(' | ')' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '·' => out.push_str("\\267"),
            c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
            c if (c as u32) >= 0xA0 && (c as u32) <= 0xFF => {
                out.push_str(&format!("\\{:03o}", c as u32))
            }
            '—' | '–' => out.push('-'),
            _ => out.push('?'),
        }
    }
    out
}

fn wrap(line: &str, width: usize) -> Vec<String> {
    if line.chars().count() <= width {
        return vec![line.to_owned()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for word in line.split(' ') {
        if !current.is_empty() && current.chars().count() + 1 + word.chars().count() > width {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    out.push(current);
    out
}

/// A valid PDF 1.4 document of `lines`.
pub fn pdf(lines: &[String]) -> Vec<u8> {
    const PER_PAGE: usize = 50;
    let wrapped: Vec<String> = lines.iter().flat_map(|l| wrap(l, 90)).collect();
    let pages: Vec<&[String]> = if wrapped.is_empty() {
        vec![&[][..]]
    } else {
        wrapped.chunks(PER_PAGE).collect()
    };
    // Objects: 1 catalog, 2 pages, 3 font, then (page, content) per page.
    let mut objects: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".into(),
        String::new(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".into(),
    ];
    let mut kids = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        let page_id = 4 + index * 2;
        kids.push(format!("{page_id} 0 R"));
        let mut stream = String::from("BT\n/F1 11 Tf\n14 TL\n56 740 Td\n");
        for (n, line) in page.iter().enumerate() {
            if n == 0 && index == 0 {
                stream.push_str(&format!(
                    "/F1 16 Tf\n({}) Tj\n/F1 11 Tf\nT*\n",
                    escape(line)
                ));
            } else {
                stream.push_str(&format!("({}) Tj\nT*\n", escape(line)));
            }
        }
        stream.push_str("ET\n");
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
            page_id + 1
        ));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}endstream",
            stream.len()
        ));
    }
    objects[1] = format!(
        "<< /Type /Pages /Kids [{}] /Count {} >>",
        kids.join(" "),
        kids.len()
    );
    let mut out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
    );
    for offset in offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_report_is_a_well_formed_pdf_with_a_correct_xref() {
        let lines: Vec<String> = (0..120)
            .map(|n| format!("Line {n} (escaped) \\ · ok"))
            .collect();
        let bytes = pdf(&lines);
        assert!(bytes.starts_with(b"%PDF-1.4"));
        assert!(bytes.ends_with(b"%%EOF\n"));
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/Count 3"), "120 lines make three pages");
        assert!(text.contains("(Line 7 \\(escaped\\) \\\\ \\267 ok) Tj"));
        // Every xref offset points at its object header.
        let start: usize = text
            .rsplit("startxref\n")
            .next()
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let table = &bytes[start..];
        let table = String::from_utf8_lossy(table);
        for (n, entry) in table
            .lines()
            .skip(3)
            .take_while(|l| l.ends_with(" n "))
            .enumerate()
        {
            let offset: usize = entry[..10].parse().unwrap();
            assert!(bytes[offset..].starts_with(format!("{} 0 obj", n + 1).as_bytes()));
        }
        // The upload check the workspace and `cloud publish --report` apply.
        assert!(bytes[bytes.len().saturating_sub(1024)..]
            .windows(5)
            .any(|w| w == b"%%EOF"));
    }
}
