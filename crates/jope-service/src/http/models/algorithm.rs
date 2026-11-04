use super::*;

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::http) struct AlgorithmQuery {
    language: Option<Language>,
    diff_from: Option<AlgorithmId>,
    after: Option<AlgorithmId>,
    limit: Option<usize>,
}
pub(super) fn algorithm(
    state: &DatasetState,
    run: RunId,
    alg: Option<AlgorithmId>,
    query: AlgorithmQuery,
    list: bool,
) -> Result<Value> {
    if query.diff_from.is_some() || query.limit.is_some_and(|n| !(1..=256).contains(&n)) {
        return Err(bad());
    }
    let (snapshot, model, _) = champion(state, &run, alg.as_ref())?;
    let best = snapshot.best.as_ref().ok_or_else(missing)?;
    let language = query.language.unwrap_or(Language::Rust);
    let name = language_name(language);
    let source = String::from_utf8(model.export_source(name).map_err(|_| unavailable())?)
        .map_err(|_| integrity())?;
    let result = Algorithm {
        algorithm_id: best.algorithm_id.clone(),
        summary: "Frozen completed CPU regression model".into(),
        metric: best.metric.clone(),
        cv_value: best.cv_value,
        model_artifact: best.model_artifact.clone(),
        model_digest: model.id().into(),
        conditioning_digest: "none".into(),
        bundle_digest: "none".into(),
        revision: snapshot.revision,
        source: vec![AlgorithmSource { language, source }],
    };
    if list {
        Ok(
            json!({"algorithms":if query.after.as_ref().is_some_and(|id| id >= &best.algorithm_id) {vec![]} else {vec![result]},"next_after":null}),
        )
    } else {
        serde_json::to_value(result).map_err(|_| integrity())
    }
}
pub(super) fn language_name(v: Language) -> &'static str {
    match v {
        Language::Rust => "rust",
        Language::Python => "python",
        Language::Javascript => "javascript",
        Language::Typescript => "typescript",
    }
}
pub(in crate::http) async fn get_algorithm(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<(String, String)>, axum::extract::rejection::PathRejection>,
    query: std::result::Result<Query<AlgorithmQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let (Ok(Path((run, alg))), Ok(Query(query))) = (path, query) else {
        return response(&bad());
    };
    let (Ok(run), Ok(alg)) = (run.parse::<RunId>(), alg.parse::<AlgorithmId>()) else {
        return response(&bad());
    };
    finish_json(
        tokio::task::spawn_blocking(move || algorithm(&state, run, Some(alg), query, false)).await,
    )
}
pub(in crate::http) async fn list_algorithms(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
    query: std::result::Result<Query<AlgorithmQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let (Ok(Path(run)), Ok(Query(query))) = (path, query) else {
        return response(&bad());
    };
    let Ok(run) = run.parse::<RunId>() else {
        return response(&bad());
    };
    finish_json(
        tokio::task::spawn_blocking(move || algorithm(&state, run, None, query, true)).await,
    )
}
pub(super) fn finish_json(
    result: std::result::Result<Result<Value>, tokio::task::JoinError>,
) -> Response {
    match result {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => response(&e),
        Err(_) => response(&integrity()),
    }
}
