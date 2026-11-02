use super::*;

/// The report and whether the paper, its Figure 1, the committed SVG figures, the curated
/// experiment digest, and the README agree with their records; with `sources`, also whether every
/// claim source on this host still hashes to its pin. With `write_figure`, Figure 1 is
/// re-rendered first.
pub fn check(root: &Path, sources: bool, write_figure: bool) -> Result<(String, bool)> {
    let writeup = py::read_text(&root.join(WRITEUP))?;
    let ledger_text = py::read_text(&root.join(LEDGER))?;
    let trace = py::read_text(&root.join(TRACE))?;
    let readme_text = py::read_text(&root.join(README))?;
    let exp_jsonl = py::read_text(&root.join(EXPERIMENTS_JSONL))?;
    let exp_readme = py::read_text(&root.join(EXPERIMENTS_README))?;
    let exp_synthesis = py::read_text(&root.join(EXPERIMENTS_SYNTHESIS))?;
    let (mut problems, mut summary) = check_paper(&writeup, &ledger_text, &trace)?;
    problems.extend(crate::paper_figure::check(
        root,
        &writeup,
        &ledger_text,
        write_figure,
    )?);
    problems.extend(check_svg_figures(
        root,
        &trace,
        &writeup,
        &readme_text,
        &ledger_text,
    )?);
    problems.extend(check_experiments(
        root,
        &exp_jsonl,
        &exp_readme,
        &exp_synthesis,
        &ledger_text,
    )?);
    summary = format!(
        "{summary}; Figure 1 against its data and C26, C27, C76 and C190; 4 committed SVG figures; 18 experiment records"
    );
    if sources {
        let (found, line) = check_sources(&ledger_text)?;
        problems.extend(found);
        summary = format!("{summary}\n{line}");
    }
    let g = glance(root)?;
    problems.extend(check_readme(&readme_text, &g));
    let mut out = String::new();
    for p in &problems {
        out.push_str(p);
        out.push('\n');
    }
    out.push_str(&format!(
        "{summary}; README: version {} and the At a glance counts\n",
        g.version
    ));
    out.push_str(if problems.is_empty() {
        "paper-trace: OK\n"
    } else {
        "paper-trace: FAIL\n"
    });
    Ok((out, problems.is_empty()))
}
