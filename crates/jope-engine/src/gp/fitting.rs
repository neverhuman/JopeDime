pub fn fit(train: &Path, genome: &str, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "fitted predictor already exists");
    let table = Table::read(train, &serde_json::Value::Null)?;
    ensure!(table.rows.len() <= ROW_LIMIT, "fit row limit exceeded");
    let pp = Preprocessing::fit(&table.rows, "regression")?;
    let transformed = pp.transform(&table.rows)?;
    let (x, y) = xy(&transformed);
    let g = GenomeIr::parse(genome)?;
    let preprocessing = serde_json::to_vec_pretty(&pp)?;
    let feature_view = raw_feature_view(&csv_sha256(&transformed)?, &preprocessing)?;
    let fitted =
        crate::fit_boundary::fit(&g.canonical(), x, y, preprocessing, feature_view, output)?;
    let p = Predictor {
        schema: "parallel.predictor.v1".into(),
        task: "regression".into(),
        canonical: g.canonical(),
        training_sha256: file_hash(train)?,
        feature_count: table.width - 1,
        preprocessing: pp,
        model: fitted.model,
        fit_completion: Some(fitted.completion),
    };
    write_json(output, &p)
}
/// Fit a selected program directly from the immutable cut pointers. The
/// predictor stores the cut's training-only preprocessing for later raw inputs.
/// Without a declared root only a contained view (the cut's own tables) is
/// projected; a pointer view is refused, never resolved from the view's
/// ancestors (W1 route R2). The public `fit-view` verb is [`fit_view_in_root`].
pub fn fit_view(prepared: &Path, cut: usize, genome: &str, output: &Path) -> Result<()> {
    fit_view_with_root(None, prepared, cut, genome, output)
}

/// The public `fit-view` verb (W1 route R2): the cut's fitting rows are
/// projected once, from the preparation it read, under the declared `root`;
/// only that projection and the verified preparation state reach the fit
/// authority. No ancestor of the view is searched for its backing table.
pub fn fit_view_in_root(
    root: &Path,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    fit_view_with_root(Some(root), prepared, cut, genome, output)
}

/// The fitting-only projection of one cut (W1 route R2): under the declared
/// `root`, or with none only from a contained view's own tables. `p` is the
/// preparation the caller read; `prepared.json` is not reopened.
fn project_fitting_view(
    root: Option<&Path>,
    prepared: &Path,
    p: &Prepared,
    cut: usize,
) -> Result<Vec<Vec<f64>>> {
    match root {
        Some(root) => crate::prepare::load_bound_view_in_root(root, prepared, p, cut, true),
        None => crate::prepare::load_bound_contained_view(prepared, p, cut, true),
    }
}

fn fit_view_with_root(
    root: Option<&Path>,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "fitted predictor already exists");
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.schema == "parallel.prepared.v1" && p.task == "regression",
        "regression preparation required"
    );
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let preprocessing_path = prepared.join(format!("cut-{cut}/preprocessing.json"));
    // Route R8: the stored state and the fitted feature view bind the same
    // verified bytes; the path is read once.
    let preprocessing_bytes = read_verified(
        &preprocessing_path,
        &c.preprocessing_sha256,
        "preprocessing",
    )
    .context("preprocessing changed")?;
    let table = project_fitting_view(root, prepared, &p, cut)?;
    fit_projected(
        &table,
        &c.training_sha256,
        preprocessing_bytes,
        genome,
        output,
    )
}

/// Fit exactly the given projected training rows (W1 follow-up 2c). Shared by
/// the in-process fitted-view fit and the sealed-memory export worker.
fn fit_projected(
    rows: &[Vec<f64>],
    training_sha256: &str,
    preprocessing_bytes: Vec<u8>,
    genome: &str,
    output: &Path,
) -> Result<()> {
    ensure!(
        csv_sha256(rows)? == training_sha256,
        "projected training rows differ from the cut's training identity"
    );
    let preprocessing: Preprocessing = serde_json::from_slice(&preprocessing_bytes)?;
    let (x, y) = xy(rows);
    let g = GenomeIr::parse(genome)?;
    let feature_view = raw_feature_view(training_sha256, &preprocessing_bytes)?;
    let fitted = crate::fit_boundary::fit(
        &g.canonical(),
        x,
        y,
        preprocessing_bytes,
        feature_view,
        output,
    )?;
    write_json(
        output,
        &Predictor {
            schema: "parallel.predictor.v1".into(),
            task: "regression".into(),
            canonical: g.canonical(),
            training_sha256: training_sha256.into(),
            feature_count: rows.first().context("empty projected training rows")?.len() - 1,
            preprocessing,
            model: fitted.model,
            fit_completion: Some(fitted.completion),
        },
    )
}
