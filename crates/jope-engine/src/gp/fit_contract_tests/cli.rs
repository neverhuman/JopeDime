use super::*;

pub(super) fn run_cli(argv: &[String]) -> Result<()> {
    let (operation, rest) = argv.split_first().context("fixture CLI operation absent")?;
    let mut values = BTreeMap::new();
    let mut qualification = false;
    let mut index = 0;
    while index < rest.len() {
        let key = rest[index]
            .strip_prefix("--")
            .context("fixture CLI flag expected")?;
        if key == "qualification" {
            qualification = true;
            index += 1;
            continue;
        }
        let value = rest
            .get(index + 1)
            .context("fixture CLI flag value absent")?;
        ensure!(
            values.insert(key.to_owned(), value.clone()).is_none(),
            "repeated fixture CLI flag --{key}"
        );
        index += 2;
    }
    let text = |key: &str| {
        values
            .get(key)
            .cloned()
            .with_context(|| format!("fixture CLI --{key} absent"))
    };
    let path = |key: &str| text(key).map(PathBuf::from);
    match operation.as_str() {
        "fit" => fit(&path("train")?, &text("genome")?, &path("output")?),
        "fit-view" if std::env::var(FAULT).as_deref() == Ok("fail_export") => {
            anyhow::bail!("fixture: fitted-view export unavailable")
        }
        // The public verb's declared root (W1 route R2); without one only a
        // contained view is fitted.
        "fit-view" => match values.get("root") {
            Some(root) => fit_view_in_root(
                Path::new(root),
                &path("prepared")?,
                text("cut")?.parse()?,
                &text("genome")?,
                &path("output")?,
            ),
            None => fit_view(
                &path("prepared")?,
                text("cut")?.parse()?,
                &text("genome")?,
                &path("output")?,
            ),
        },
        // The isolated export's worker (W1 follow-up 2c).
        "fit-memory" if std::env::var(FAULT).as_deref() == Ok("fail_export") => {
            anyhow::bail!("fixture: fitted-view export unavailable")
        }
        "fit-memory" => fit_memory(
            text("fd")?.parse()?,
            &text("training-sha256")?,
            &text("preprocessing-sha256")?,
            &text("genome")?,
            &path("output")?,
        ),
        // Same library call the original test made in-process, run inside a
        // configured client so its spawned worker inherits the fit bindings.
        "fit-view-isolated" => match values.get("root") {
            Some(root) => fit_view_isolated_in_root(
                &path("executable")?,
                Path::new(root),
                &path("prepared")?,
                text("cut")?.parse()?,
                &text("genome")?,
                &path("output")?,
            ),
            None => fit_view_isolated(
                &path("executable")?,
                &path("prepared")?,
                text("cut")?.parse()?,
                &text("genome")?,
                &path("output")?,
            ),
        },
        "evaluate-worker" => evaluate(
            &path("train")?,
            &path("validation")?,
            &path("genome")?,
            &path("output")?,
        ),
        "evaluate-memory" => {
            let prepared = values.get("prepared").map(PathBuf::from);
            let cut = values
                .get("cut")
                .map(|cut| cut.parse::<usize>())
                .transpose()?;
            ensure!(
                prepared.is_some() == cut.is_some(),
                "--prepared and --cut go together"
            );
            // The product entry's guard (W5-T05), before any fault starts a helper.
            guard_evaluator_group()?;
            evaluator_fault()?;
            evaluate_prepared_memory(
                text("fd")?.parse()?,
                &path("genome")?,
                &path("output")?,
                prepared.as_deref().zip(cut),
            )
        }
        "search" => search_seeded_with_rng_in_root(
            &path("root")?,
            &path("prepared")?,
            text("cut")?.parse()?,
            &path("output")?,
            crate::gp::SearchOptions {
                evaluations: text("evaluations")?.parse()?,
                seconds: text("seconds")?.parse()?,
                qualification,
                seeds: values.get("seeds").map(Path::new),
                vocabulary: values.get("vocabulary").map(Path::new),
            },
            values
                .get("rng-seed")
                .map(|seed| seed.parse::<u64>())
                .transpose()?
                .unwrap_or(SEED),
        ),
        // The shipped `score` subcommand's library call (outer scoring).
        "score" => score(
            &path("model")?,
            &path("test")?,
            &path("receipt")?,
            &path("output")?,
        ),
        // The cycle pipeline's predictor handoff for one completed search.
        "export-selected" => {
            let root = path("root")?;
            let executable = path("executable")?;
            let executable_sha = file_hash(&executable)?;
            let seeds = values.get("seeds").map(PathBuf::from);
            let vocabulary = values.get("vocabulary").map(PathBuf::from);
            let receipt = handoff::export_selected(&handoff::SelectedExport {
                root: &root,
                search_output: &path("output")?,
                prepared: &path("prepared")?,
                cut: text("cut")?.parse()?,
                seeds: seeds.as_deref(),
                vocabulary: vocabulary.as_deref(),
                executable: &executable,
                executable_sha: &executable_sha,
                predictor: &path("predictor")?,
            })?;
            println!("{receipt}");
            Ok(())
        }
        other => anyhow::bail!("unsupported fixture CLI operation {other}"),
    }
}
