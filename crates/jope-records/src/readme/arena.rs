use super::*;

/// A snapshot and the CSV it hashes, cross-checked. Fails closed on any disagreement.
pub(crate) fn load_ladder(
    p: &Paths,
    snapshot: &str,
    csv: &str,
    ladder: &str,
) -> Result<(Value, Vec<Dict>)> {
    let snap = p.json(snapshot)?;
    let text = py::read_text(&p.at(csv))?;
    let rows = pycsv::dict_reader(&py::splitlines(&text).join("\n"), Newlines::LineFeed)?.rows;
    // `snap.get("schema")`
    dict_get(&snap, "schema", snapshot)?;
    if get(&snap, "schema") != &Value::from(SNAPSHOT_SCHEMA) {
        return refuse(format!(
            "{snapshot}: schema {} is not {SNAPSHOT_SCHEMA}",
            get(&snap, "schema").repr()
        ));
    }
    if get(&snap, "ladder") != &Value::from(ladder) {
        return refuse(format!(
            "{snapshot}: ladder {} is not {}",
            get(&snap, "ladder").repr(),
            crate::pyjson::str_repr(ladder)
        ));
    }
    // The snapshot records the hash of the ladder it was derived from. If the committed CSV has
    // moved since, the two files disagree and neither is safe to publish.
    let actual = p.sha256(csv)?;
    if get(&snap, "leaderboard_sha256") != &Value::from(actual.as_str()) {
        return refuse(format!(
            "{csv} sha256 {actual}\ndoes not match {snapshot} leaderboard_sha256 {}.",
            s(get(&snap, "leaderboard_sha256"))
        ));
    }
    if at(&snap, "methods")? != &Value::from(rows.len()) {
        return refuse(format!(
            "{snapshot}: {} methods but the CSV has {} rows",
            s(at(&snap, "methods")?),
            rows.len()
        ));
    }
    Ok((snap, rows))
}

pub(crate) struct Evidence {
    track: Value,
    published: Value,
    published_rows: Vec<Dict>,
    valid: Value,
    valid_rows: Vec<Dict>,
    corrections: Value,
}

pub(crate) fn load_evidence(p: &Paths) -> Result<Evidence> {
    let track = p.json(TRACK)?;
    let (published, published_rows) = load_ladder(p, SNAPSHOT, LEADERBOARD, "published")?;
    let (valid, valid_rows) = load_ladder(p, VALID_SNAPSHOT, VALID_LEADERBOARD, "benchmark-valid")?;
    let corrections = p.json(CORRECTIONS)?;
    if !at(&published, "corrections")?.is_null()
        || at(&published, "excluded_rows")? != &Value::Int(0)
    {
        return refuse(format!(
            "{SNAPSHOT}: the published ladder must apply no corrections"
        ));
    }
    // `corrections.get("schema")`
    dict_get(&corrections, "schema", CORRECTIONS)?;
    if get(&corrections, "schema") != &Value::from(CORRECTIONS_SCHEMA) {
        return refuse(format!("{CORRECTIONS}: schema is not {CORRECTIONS_SCHEMA}"));
    }
    // `a == b == c` is `a == b and b == c`: `c` is read only when `a == b`.
    let (a, b) = (
        at(&corrections, "scores_sha256")?,
        at(&published, "scores_sha256")?,
    );
    if !(a == b && b == at(&valid, "scores_sha256")?) {
        return refuse("the two ladders and the corrections do not bind the same scores.csv");
    }
    let meta = at(&valid, "corrections")?;
    if meta.is_null() || at(meta, "sha256")? != &Value::from(p.sha256(CORRECTIONS)?) {
        return refuse(format!(
            "{VALID_SNAPSHOT} was not computed with the committed {CORRECTIONS}"
        ));
    }
    if at(&valid, "excluded_rows")? != at(meta, "rows_corrected")? {
        // Equal while every published row is True. If the benchmark ever publishes a False row,
        // the ladder excludes it too and this sentence must be rewritten, not silently
        // under-counted.
        return refuse(format!(
            "the benchmark-valid ladder excluded {} rows but the corrections demoted {}; the README prose assumes they are the same set",
            s(at(&valid, "excluded_rows")?),
            s(at(meta, "rows_corrected")?)
        ));
    }
    for row in &valid_rows {
        // `r["benchmark_valid"] != "True"`: a short row's missing cell is None, which differs.
        let flag = row.get("benchmark_valid").ok_or_else(|| {
            raised(
                VALID_LEADERBOARD,
                py::exception::key_error("benchmark_valid"),
            )
        })?;
        if flag != &Value::from("True") {
            return refuse("a benchmark-invalid method survived into the benchmark-valid ladder");
        }
    }
    // Every correction must show up as fewer scored datasets for its method, so re-adding the
    // rows, or editing the corrections without regenerating the ladder, fails here. The ids,
    // methods and row counts are read as the Python read them: iterated, hashed and added as they
    // are, never skipped or defaulted, and added as Python adds them ([`Num::plus`]: integers
    // exact at any size, a float only once a float operand appears).
    let applied = applied_rows(meta, &format!("{VALID_SNAPSHOT} corrections"))?;
    let listed = items(
        sub(&corrections, "corrections", CORRECTIONS)?,
        &format!("{CORRECTIONS} corrections"),
    )?;
    let context = format!("{CORRECTIONS} corrections");
    let mut ids = std::collections::HashSet::new();
    for c in &listed {
        let id = sub(c, "id", &context)?;
        ids.insert(py::Key::of(id).ok_or_else(|| raised(&context, py::exception::unhashable(id)))?);
    }
    let applied_ids: std::collections::HashSet<py::Key> =
        applied.iter().map(|(k, _, _)| k.clone()).collect();
    if applied_ids != ids {
        return refuse("the corrections applied to the ladder are not the committed corrections");
    }
    // `corrected_by_method[m] = corrected_by_method.get(m, 0) + applied[c["id"]]`, in insertion
    // order like the Python dict.
    let mut by_method: Vec<(py::Key, Value, Num)> = vec![];
    for c in &listed {
        let method = sub(c, "method_id", &context)?;
        let key = py::Key::of(method)
            .ok_or_else(|| raised(&context, py::exception::unhashable(method)))?;
        let rows = py_lookup(&applied, sub(c, "id", &context)?).expect("the id sets are equal");
        let position = by_method.iter().position(|(k, _, _)| *k == key);
        let total = position.map_or(Num::Int(0), |i| by_method[i].2);
        let total = total
            .add(rows)
            .map_err(|e| raised(format_args!("{VALID_SNAPSHOT} corrections applied rows"), e))?;
        match position {
            Some(i) => by_method[i].2 = total,
            None => by_method.push((key, method.clone(), total)),
        }
    }
    let scored = |rows: &[Dict]| -> Result<BTreeMap<String, i64>> {
        rows.iter()
            .map(|r| {
                Ok((
                    cell(r, "method_id")?.to_owned(),
                    int_cell(r, "datasets_scored")?,
                ))
            })
            .collect()
    };
    let (scored_published, scored_valid) = (scored(&published_rows)?, scored(&valid_rows)?);
    for (key, method, n) in &by_method {
        // `scored.get(method, 0)`: the ladders are keyed by the CSV's strings.
        let count = |scored: &BTreeMap<String, i64>| {
            key.as_str()
                .and_then(|m| scored.get(m))
                .copied()
                .unwrap_or(0)
        };
        let (before, after) = (count(&scored_published), count(&scored_valid));
        // `before - after != n`: the difference of two `i64` counts exact in `i128` (it needs 65
        // bits at the extremes, where `i64` arithmetic wrapped), compared with the total exactly.
        let difference = Num::Int(i128::from(before) - i128::from(after));
        if !py_eq(difference, *n) {
            return refuse(format!(
                "{}: {} rows corrected but scored datasets went {before} -> {after}",
                method.py_str(),
                n.py_str()
            ));
        }
    }
    Ok(Evidence {
        track,
        published,
        published_rows,
        valid,
        valid_rows,
        corrections,
    })
}

pub(crate) fn table_row(r: &Dict) -> Result<String> {
    let ours = OURS.contains(&cell(r, "method_id")?);
    let bold = |text: &str| {
        if ours {
            format!("**{text}**")
        } else {
            text.to_owned()
        }
    };
    Ok(format!(
        "| {} | {} | {} | {} | {} | {}/{} | `{}` |",
        bold(cell(r, "elo_rank")?),
        bold(cell(r, "method")?),
        bold(cell(r, "elo")?),
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?,
        cell(r, "protocols")?
    ))
}

/// The top N plus every one of our entries, in rank order, with an ellipsis row per gap.
pub(crate) fn ladder_table(rows: &[Dict]) -> Result<Vec<String>> {
    let mut shown: std::collections::BTreeSet<usize> = (0..TOP_N).collect();
    for (i, r) in rows.iter().enumerate() {
        if OURS.contains(&cell(r, "method_id")?) {
            shown.insert(i);
        }
    }
    let mut out = vec![
        "| rank | method | Elo | avg rank | dataset wins | coverage | protocol |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- | --- |".to_owned(),
    ];
    let mut previous: i64 = -1;
    for i in shown {
        if i as i64 > previous + 1 {
            out.push("| … | | | | | | |".to_owned());
        }
        out.push(table_row(rows.get(i).ok_or_else(|| shape("ladder row"))?)?);
        previous = i as i64;
    }
    Ok(out)
}

pub(crate) fn render_arena(p: &Paths) -> Result<String> {
    let e = load_evidence(p)?;
    let (track, corrections, published, valid) = (&e.track, &e.corrections, &e.published, &e.valid);
    let empty = Value::Dict(Dict::new());
    let no_list = Value::List(vec![]);
    // `track.get("split", {})`, `track.get("leaderboard_policy", {})`; each is then used as a dict.
    let split = dict_get(track, "split", TRACK)?.unwrap_or(&empty);
    let policy = dict_get(track, "leaderboard_policy", TRACK)?.unwrap_or(&empty);
    let split_at = format!("{TRACK} split");
    let policy_at = format!("{TRACK} leaderboard_policy");
    // `", ".join(split.get("excluded_datasets", [])) or "none"`
    let excluded = dict_get(split, "excluded_datasets", &split_at)?.unwrap_or(&no_list);
    let excluded = py::join(", ", excluded)
        .map_err(|e| raised(format_args!("{split_at} excluded_datasets"), e))?;
    let excluded = if excluded.is_empty() {
        "none".to_owned()
    } else {
        excluded
    };
    let excluded_rows = s(at(valid, "excluded_rows")?);
    let find = |rows: &[Dict], id: &str| -> Result<Option<Dict>> {
        let mut found = None;
        for r in rows {
            if cell(r, "method_id")? == id {
                found = Some(r.clone());
            }
        }
        Ok(found)
    };
    let applied = applied_rows(
        at(valid, "corrections")?,
        &format!("{VALID_SNAPSHOT} corrections"),
    )?;
    // `{c["id"]: c for c in corrections["corrections"]}`
    let context = format!("{CORRECTIONS} corrections");
    let mut by_id = PyDict::new();
    for c in items(sub(corrections, "corrections", CORRECTIONS)?, &context)? {
        let id = sub(&c, "id", &context)?.clone();
        py_insert(&mut by_id, &id, c, &context)?;
    }
    let (Some(golden46), Some(golden42)) = (
        py_lookup(
            &by_id,
            &Value::from("golden46-new3-20260723-v4-test-adaptive"),
        ),
        py_lookup(
            &by_id,
            &Value::from("golden42-evolution-envelope-test-selected"),
        ),
    ) else {
        return refuse(
            "the README prose is written for the two committed corrections; update both together",
        );
    };
    // `c["evidence"][0].split(" ")[0]`
    let first_ref = |c: &Value| -> Result<String> {
        let evidence = sub(c, "evidence", &context)?;
        let first =
            py::first_item(evidence).map_err(|e| raised(format_args!("{context} evidence"), e))?;
        let text = first.as_str().ok_or_else(|| {
            raised(
                format_args!("{context} evidence"),
                py::exception::no_attribute(&first, "split"),
            )
        })?;
        archived(p, text.split(' ').next().unwrap_or_default())
    };
    let public = if get(split, "test_labels_public").truthy() {
        "public"
    } else {
        "**not public**"
    };
    // `track['primary_metric'].upper()`
    let metric = at(track, "primary_metric")?;
    let metric = metric
        .as_str()
        .ok_or_else(|| {
            raised(
                format_args!("{TRACK} primary_metric"),
                py::exception::no_attribute(metric, "upper"),
            )
        })?
        .to_uppercase();
    let mut out = vec![
        ARENA_BEGIN.to_owned(),
        format!(
            "### {} — {}, {} datasets",
            s(at(track, "public_name")?),
            s(at(track, "task")?),
            s(at(published, "datasets_scored")?)
        ),
        String::new(),
        format!(
            "`{}` · primary metric **{metric}** ({}) · Elo by `{}` · test labels are {public}; {} datasets are restricted; excluded datasets: {excluded}.",
            s(at(track, "id")?),
            s(at(track, "metric_direction")?),
            dict_get(policy, "elo_method", &policy_at)?.map_or_else(|| "unknown".to_owned(), s),
            split.get("restricted_dataset_count").map_or_else(|| "0".to_owned(), s),
        ),
        String::new(),
        "#### Benchmark-valid — the ladder this repository claims".to_owned(),
        String::new(),
        format!(
            "**{} methods** · {} rankable scores · {excluded_rows} published rows excluded as test-selected (see the next block). Bold rows are this repository's entries. *Benchmark-valid* here means that no artefact in this repository contradicts the row's published flag; it is not a claim that every comparator's protocol was verified.",
            s(at(valid, "methods")?),
            thousands(rankable_rows(valid)?),
        ),
        String::new(),
    ];
    out.extend(ladder_table(&e.valid_rows)?);
    let together = policy
        .get("protocols_ranked_together")
        .map_or_else(|| "False".to_owned(), s)
        .to_lowercase();
    out.extend([
        String::new(),
        format!(
            "**Read the protocol column before the Elo column.** `leaderboard_policy.protocols_ranked_together` is `{together}`, so entries produced under different protocols share one ladder. `jope-dime-regressor` is one fixed-seed reproducible run of this repository's engine under `public-development`, the same protocol as every other comparator. The same-cut benchmark is what turns this into a like-for-like comparison."
        ),
        String::new(),
        "#### Best known program per dataset — as published; test-selected, not a held-out result"
            .to_owned(),
        String::new(),
    ]);
    let Some(pr) = find(&e.published_rows, "jope-dime")? else {
        return refuse("jope-dime is missing from the published ladder");
    };
    if find(&e.valid_rows, "jope-dime")?.is_some() {
        // The prose below is written for the case where every jope-dime row is corrected.
        return refuse(
            "jope-dime still has benchmark-valid rows; the best-known block's prose assumes it has none",
        );
    }
    // `applied[c['id']]`
    let applied_of = |c: &Value| -> Result<String> {
        let id = sub(c, "id", &context)?;
        py_lookup(&applied, id).map(s).ok_or_else(|| {
            raised(
                format_args!("{VALID_SNAPSHOT} corrections applied"),
                format!("KeyError: {}", id.repr()),
            )
        })
    };
    let track_id = s(at(track, "id")?);
    out.extend([
        format!(
            "The benchmark published `jope-dime` at rank {}, Elo {}, {} dataset wins, {}/{} coverage, every row `benchmark_valid: True`. This repository's own code says otherwise about every one of those rows, so the benchmark-valid ladder above carries none of them:",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "- **{} Golden42 rows** are each the per-dataset minimum RMSE over the ledger's test-scored candidate evaluations (63,108 across the 42 datasets), not the score of one program frozen before the test targets were opened. The red-team lane in this tree names that quantity the *best-test ASPFM envelope*, role `oracle_reference_only`, and keeps it out of its fair ladder (`{}`, `crates/jope-campaigns/src/commands/train_only_redteam.rs:402-407`).",
            applied_of(golden42)?, first_ref(golden42)?
        ),
        format!(
            "- **{} Golden46 rows** come from run `{}`, whose artefacts this repository's campaign code writes with `benchmark_valid: false, test_adaptive: true` (`{}`); the acceptance manifest relabelled the replayed winners `true`.",
            applied_of(golden46)?, s(at(golden46, "source_revision")?), first_ref(golden46)?
        ),
        String::new(),
        "Both are the best program found per dataset when candidates were scored on the test targets: reported here as best-known results, never as a held-out claim. The reproducible fixed-seed run of the same engine, `jope-dime-regressor`, is what stands on the ladder above. The same-cut benchmark replaces every row on this page with numbers this tree can reproduce.".to_owned(),
        String::new(),
        "| ladder | `jope-dime` | Elo | dataset wins | coverage |".to_owned(),
        "| --- | ---: | ---: | ---: | --- |".to_owned(),
        format!(
            "| as published | rank {} | {} | {} | {}/{} |",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        format!(
            "| benchmark-valid | not ranked | — | — | 0/{} (every row test-selected) |",
            cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "Sources: published `{LEADERBOARD}` (sha256 `{}…`), recomputed byte-for-byte from `scores.csv`; benchmark-valid `{VALID_LEADERBOARD}` (sha256 `{}…`), the same `scores.csv` with `{CORRECTIONS}` (sha256 `{}…`) applied. Both are asserted by `crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:",
            &p.sha256(LEADERBOARD)?[..16], &p.sha256(VALID_LEADERBOARD)?[..16], &p.sha256(CORRECTIONS)?[..16]
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} --check"),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} \\"),
        format!("  --ladder benchmark-valid --corrections {CORRECTIONS} \\"),
        format!("  --check --committed {VALID_LEADERBOARD}"),
        "```".to_owned(),
        ARENA_END.to_owned(),
    ]);
    let _ = REGRESSION;
    Ok(out.join("\n"))
}
