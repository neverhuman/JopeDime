use super::*;

/// The `--adapter` argument that recorded this engine, rendered so a shell reads it back
/// exactly; a token that would need quoting refuses rather than print a wrong command.

pub(super) fn adapter_flag(spec: &Value) -> Result<String> {
    // `re.compile(r"^[A-Za-z0-9_./=:+-]+$").match`: Python's `$` also matches before a final
    // newline, so a token ending in one newline passed there, and passes here.
    let safe = Regex::new(r"^[A-Za-z0-9_./=:+-]+\n?\z").expect("static pattern");
    let context = format!("{SAMECUT} adapter spec");
    // `tokens = [spec["name"], *spec["command"]]`: every token checked as it is, none skipped.
    let name = sub(spec, "name", &context)?;
    let command = sub(spec, "command", &context)?;
    let unpacked = py::iterate(command).ok_or_else(|| {
        raised(
            &context,
            format!(
                "TypeError: Value after * must be an iterable, not {}",
                py::type_name(command)
            ),
        )
    })?;
    for token in std::iter::once(name).chain(unpacked.iter()) {
        // `SAFE_TOKEN.match(token)`
        let text = token.as_str().ok_or_else(|| {
            raised(
                &context,
                format!(
                    "TypeError: expected string or bytes-like object, got '{}'",
                    py::type_name(token)
                ),
            )
        })?;
        if !safe.is_match(text) {
            return refuse(format!(
                "adapter {}: token {} is not shell-safe to render",
                name.repr(),
                token.repr()
            ));
        }
    }
    let command = py::join(" ", command).map_err(|e| raised(&context, e))?;
    Ok(format!("--adapter \"{}={command}\"", name.py_str()))
}

pub(super) fn samecut_row(r: &Dict, versions: &BTreeMap<String, String>) -> Result<String> {
    let method_id = cell(r, "method_id")?;
    let b = if method_id == SEARCHED { "**" } else { "" };
    let mut name = cell(r, "method")?.to_owned();
    if let Some(engine) = method_id.strip_prefix("adapter.") {
        let version = versions
            .get(engine)
            .map_or("version unknown", String::as_str);
        name = format!("{name} ({version})");
    }
    Ok(format!(
        "| {b}{}{b} | {b}{name}{b} | {b}{}{b} | {} | {} | {}/{} |",
        cell(r, "elo_rank")?,
        cell(r, "elo")?,
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?
    ))
}

pub(super) fn render_samecut(p: &Paths) -> Result<String> {
    let e = load_samecut(p)?;
    let (deep, reports, ours) = (&e.deep, &e.reports, &e.ours);
    let first = &reports[0];
    let (plan, protocol, search, host) = (
        at(first, "plan")?,
        at(first, "protocol")?,
        at(first, "search")?,
        at(first, "host")?,
    );
    let n = reports.len();
    let mut per_dataset = vec![];
    let mut search_seconds: Vec<Num> = vec![];
    let (mut finalist, mut best) = (0usize, 0usize);
    // `sorted(reports, key=lambda r: r["dataset_id"])` (stable). The ids are strings; ids of any
    // other type (which Python sorted only when they were mutually comparable numbers) are
    // refused rather than ordered by their text.
    let mut ordered: Vec<(&str, &Value)> = vec![];
    for r in reports {
        let id = sub(r, "dataset_id", SAMECUT)?;
        match (id.as_str(), reports.len()) {
            (Some(id), _) => ordered.push((id, r)),
            (None, 1) => ordered.push(("", r)),
            (None, _) => {
                return Err(Error::Shape(format!(
                    "{SAMECUT}: report dataset_id {} is not a string, so the reports cannot be ordered as the Python ordered them",
                    id.repr()
                )))
            }
        }
    }
    ordered.sort_by(|a, b| a.0.cmp(b.0));
    for (_, r) in ordered {
        let methods = items(at(r, "methods")?, &format!("{SAMECUT} report methods"))?;
        // `next(m for m in r["methods"] if m["method"]["kind"] == "search")`: every method before
        // the first search one is read, and one that cannot be read refuses.
        let mut s_index = None;
        for (i, m) in methods.iter().enumerate() {
            if kind_of(m)? == "search" {
                s_index = Some(i);
                break;
            }
        }
        let s_index = s_index.ok_or_else(|| shape("search method"))?;
        let sm = &methods[s_index];
        // "Best" is the ladder's own rule: the lowest pooled RMSE among the finalists that
        // scored, ties by method id: `min(scored, key=lambda m: (rmse, id))`, which keeps the
        // first of equal keys and compares the ids only on a tie. The RMSEs are compared as
        // Python compared them, exactly ([`py_cmp`]: an int RMSE and a float one that round to the
        // same double are not a tie).
        let mut top: Option<(usize, Num, Value)> = None;
        for (i, m) in methods.iter().enumerate() {
            let deep_stage = get(m, "deep");
            if !(deep_stage.truthy() && sub(deep_stage, "status", SAMECUT)? == &Value::from("ok")) {
                continue;
            }
            let rmse = num_of(
                sub(sub(deep_stage, "oof", SAMECUT)?, "rmse", SAMECUT)?,
                &format!("{SAMECUT} report deep oof rmse"),
            )?;
            let id = sub(at(m, "method")?, "id", SAMECUT)?.clone();
            let better = match &top {
                None => true,
                // Tuple `<`: the first pair that is neither identical nor equal decides, so the ids
                // decide between equal RMSEs, and between two NaNs (every JSON `NaN` is one object).
                Some((_, best_rmse, best_id)) => {
                    if py_eq(rmse, *best_rmse) || (rmse.is_nan() && best_rmse.is_nan()) {
                        id_less(&id, best_id)?
                    } else {
                        py_lt(rmse, *best_rmse)
                    }
                }
            };
            if better {
                top = Some((i, rmse, id));
            }
        }
        let Some((top_index, rmse, top_id)) = top else {
            return refuse(format!(
                "{SAMECUT}: {} has no scored finalist",
                s(sub(r, "dataset_id", SAMECUT)?)
            ));
        };
        let top_id = top_id.py_str();
        let produce = num_of(
            at(at(sm, "method")?, "produce_seconds")?,
            &format!("{SAMECUT} report search method produce_seconds"),
        )?;
        search_seconds.push(produce);
        let produce = produce.as_f64();
        let rank = get(sm, "deep_rank");
        if !rank.is_null() {
            finalist += 1;
        }
        let is_search = top_index == s_index;
        if is_search {
            best += 1;
        }
        let marker = if is_search { "**" } else { "" };
        let rank_text = if rank.is_null() {
            // `f'screen {s['screen_rank']}'`
            format!("screen {}", s(sub(sm, "screen_rank", SAMECUT)?))
        } else {
            s(rank)
        };
        per_dataset.push(format!(
            "| `{}` | {} | {} | {marker}{top_id}{marker} | {} | {rank_text} | {} s |",
            s(at(r, "dataset_id")?),
            s(at(r, "rows")?),
            s(at(r, "features")?),
            general4(rmse.as_f64()),
            fixed(produce, 1)
        ));
    }
    // `sum(s) / n`, the median of `sorted(s)` (the middle value, or `sum` of the middle two over
    // 2) and `max(s)`, with Python's arithmetic and its ordering of a NaN.
    let what = format!("{SAMECUT} report search method produce_seconds");
    let mean_s = py_div(py_sum(&search_seconds, &what)?, n);
    let sorted_s = py_sorted(&search_seconds, &what)?;
    let median_s = if n % 2 == 1 {
        sorted_s[n / 2].as_f64()
    } else {
        py_div(py_sum(&sorted_s[n / 2 - 1..n / 2 + 1], &what)?, 2)
    };
    let max_s = py_max(&search_seconds).as_f64();
    let git_sha = first_nine(at(first, "git_sha")?, &format!("{SAMECUT} report git_sha"))?;
    let mut out = vec![
        SAMECUT_BEGIN.to_owned(),
        "### Same-cut — every method on identical held-out rows, reproducible from this tree".to_owned(),
        String::new(),
        format!(
            "`{}` · {n} regression datasets from `datasets/test` · primary metric **RMSE** (minimize) · fold plan k={}, seed {}, screen fold {}, top {} deepened · search budget {} programs × {} rounds, re-run on every cut · recorded at `{}` with {} workers on {} cores.",
            s(at(deep, "track_id")?), s(at(plan, "k")?), s(at(plan, "seed")?),
            s(at(protocol, "screen_fold")?), s(at(protocol, "top")?), s(at(search, "pop")?),
            s(at(search, "gens")?), git_sha,
            s(at(host, "workers")?), s(at(host, "cores")?)
        ),
        String::new(),
        format!(
            "Every row here is a number this tree produced and can reproduce: the ten single-head baselines, four external engines (XGBoost, LightGBM, CatBoost and a scikit-learn random forest, each run as a subprocess with its library defaults, a fixed seed and one thread, handed a cut's training rows and its held-out features only, and scored by this benchmark from the predictions it returns) and the program the search wrote for each dataset, all fitted on the same training rows and scored on the same held-out rows by one evaluator (`{SAMECUT_DOC}`). The search sees only a cut's training rows, and its deep row is the *procedure* re-run on every cut, not one program chosen with knowledge of the other cuts' held-out rows. Bold is the searched program."
        ),
        String::new(),
        "**The datasets are a favourable selection.** They are the `test-10` collection, chosen from the published ladder where the historical engine ranked first (`docs/datasets.md`), and they are used here because they are the only committed regression data. Identical cuts make the comparison between methods fair on these ten; they say nothing about a broader panel. The frozen 30-dataset PMLB panel of the experiment contract (`docs/asm.md`) excludes them on purpose; a same-cut run over it is the next evidence step.".to_owned(),
        String::new(),
        "| rank | method | Elo | avg rank | dataset wins | coverage |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- |".to_owned(),
    ];
    for r in &e.deep_rows {
        out.push(samecut_row(r, &e.versions)?);
    }
    let ours_rank = int_cell(ours, "elo_rank")?;
    let mut standing = vec![];
    for name in SAMECUT_ADAPTERS {
        let engine = &e.engines[name];
        let side = if ours_rank < int_cell(engine, "elo_rank")? {
            "above"
        } else {
            "below"
        };
        standing.push(format!(
            "{side} {} (rank {})",
            cell(engine, "method")?,
            cell(engine, "elo_rank")?
        ));
    }
    let top = s(at(protocol, "top")?);
    out.extend([
        String::new(),
        format!(
            "**Where the search stands today:** rank {} of {} on the deep ladder, Elo {}, lowest pooled RMSE on {best} of {n} datasets (the ladder's own rule), a finalist on {finalist}; on the other {} it did not make the top-{top} screen. Against the four external engines it ranks {}. Coverage counts datasets where a method reached the deep stage, so the ladder ranks finalists only. The screen ladder (every method, one cut) is committed beside it.",
            cell(ours, "elo_rank")?, s(at(deep, "methods")?), cell(ours, "elo")?,
            n - finalist, standing.join(", ")
        ),
        String::new(),
        format!(
            "**Seconds per program.** Writing the program took the search {} s per dataset on average (median {} s, longest {} s) on the screen cut's training rows, wall time under whatever else the host was running. The owner's stated reference point for a large language model writing a program for the same task, not measured here, is about {LLM_REFERENCE_HOURS} hour and {} tokens; that comparison is one of cost, not of quality, and the quality standing is the ladder above.",
            fixed(mean_s, 1), fixed(median_s, 1), fixed(max_s, 1), thousands(LLM_REFERENCE_TOKENS)
        ),
        String::new(),
        "| dataset | rows | features | lowest pooled RMSE | RMSE | search rank (penalised score) | search |".to_owned(),
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |".to_owned(),
    ]);
    out.extend(per_dataset);
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    out.extend([
        String::new(),
        format!(
            "Sources: `{deep_csv}` (sha256 `{}…`) recomputed byte-for-byte from `{deep_scores}`, itself re-derived from the {n} reports under `{}` by `crates/jope-bench/tests/committed_evidence.rs`, which also re-evaluates one dataset's baselines on the same cuts. Provenance and the exact commands: `{SAMECUT}/SOURCES.md`. Reproduce:",
            &p.sha256(&deep_csv)?[..16], samecut_path("reports")
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \\".to_owned(),
    ]);
    for spec in &e.specs {
        out.push(format!("  {} \\", adapter_flag(spec)?));
    }
    out.extend([
        format!(
            "  --top {top} --search --search-pop {} --search-gens {} --resample-search --workers {}",
            s(at(search, "pop")?), s(at(search, "gens")?), s(at(host, "workers")?)
        ),
        "cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \\".to_owned(),
        format!("  --track-root target/same-cut/track --track {SAMECUT_TRACK_ID}"),
        format!(
            "cargo run -p jope-arena -- elo --track {} --root target/same-cut/track \\",
            s(at(deep, "track_id")?)
        ),
        format!("  --check --committed {deep_csv}"),
        "```".to_owned(),
        SAMECUT_END.to_owned(),
    ]);
    Ok(out.join("\n"))
}
