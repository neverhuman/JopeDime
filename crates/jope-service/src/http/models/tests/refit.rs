use super::*;

#[tokio::test]
async fn export_refit_matches_the_completed_model_without_fitting_again() {
    let (_dir, state, run, id, _) = fixture();
    let mut body = export_request();
    body.refit = Refit::FitHoldout;
    let target = format!("/v1/runs/{run}/exports");
    assert_eq!(
        export(
            &state,
            run.clone(),
            body.clone(),
            identity("wrong-mode", 13, &target, &body)
        )
        .status(),
        422
    );
    state
        .store
        .transaction(|tx| {
            let key = RecordKey::Artifact(id.clone());
            let mut r = tx.get(&key)?.unwrap();
            r.data["refit"] = json!("fit_holdout");
            tx.put(&key, &r)
        })
        .unwrap();
    let response = export(
        &state,
        run,
        body.clone(),
        identity("completed-mode", 13, &target, &body),
    );
    assert_eq!(response.status(), 200);
    let manifest: ExportManifest = serde_json::from_slice(&bytes(response).await).unwrap();
    assert_eq!(manifest.refit, Refit::FitHoldout);
    assert_eq!(manifest.parity_status, ParityStatus::Unavailable);
}
#[test]
fn explicit_fit_uses_only_the_model_bound_by_the_completed_run_receipt() {
    let (_dir, state, run, fit_id, _) = fixture();
    let (fit, _) = object(&state, &fit_id).unwrap();
    // Controlled selector fixture; operational proof uses real native fits.
    let mut manifest = fit.manifest().clone();
    manifest.metadata["refit"] = json!("fit_holdout");
    let full = ModelObject::new(manifest, fit.parts().clone(), MODEL_BYTES).unwrap();
    let full_id: ArtifactId = format!("art_{}", &full.id()[7..31]).parse().unwrap();
    let refs = std::iter::once(full.manifest_bytes())
        .chain(full.parts().values().map(Vec::as_slice))
        .map(|b| state.store.publish_blob(b).unwrap())
        .collect();
    let completed = json!({"run_id":run,"algorithm_id":"alg_0001","model_artifact":full_id,"model_digest":full.id(),"model_path":"unused","refit":"fit_holdout","scores":{"cv":0.25,"holdout":0.2,"test":null},"completion_reason":"budget_evaluations","budget_bound_by":"evaluations","budget_used":{"evaluations":"1","wall_seconds":"0.1"},"nondeterminism":{"count":"0","experiment_ids":[]},"holdout_model_digest":fit.id(),"holdout_scope":"holdout_scored_with_fit_only_model","test_scope":null,"fit_rows":4,"holdout_rows":1,"test_rows":0,"audit":null});
    let _: crate::discover::finalize::FinalizationReceipt =
        serde_json::from_value(completed.clone()).unwrap();
    let command = RecordKey::Command {
        principal: "discover.finalization.v1".into(),
        idempotency_key: run.to_string(),
    };
    state.store.transaction(|tx|{
        let fit_key=RecordKey::Artifact(fit_id.clone());
        let mut fit_record=tx.get(&fit_key)?.unwrap();
        fit_record.data["refit"]=json!(Refit::Fit);
        tx.put(&fit_key,&fit_record)?;
        tx.put(&RecordKey::Artifact(full_id.clone()),&Record::metadata(json!({"kind":"finalized_model","model_digest":full.id(),"refit":"fit_holdout"}),refs))?;
        let key=RecordKey::Run(run.clone());let mut record=tx.get(&key)?.unwrap();record.data["snapshot"]["best"]["model_artifact"]=json!(full_id);tx.put(&key,&record)?;
        tx.put(&command,&Record::metadata(json!({"state":"completed","receipt":completed}),vec![]))
    }).unwrap();
    let alg = "alg_0001".parse().unwrap();
    assert_eq!(
        export_champion(&state, &run, &alg, Refit::Fit)
            .unwrap()
            .0
            .id(),
        fit.id()
    );
    assert_eq!(
        export_champion(&state, &run, &alg, Refit::FitHoldout)
            .unwrap()
            .0
            .id(),
        full.id()
    );
    state
        .store
        .transaction(|tx| {
            let mut record = tx.get(&command)?.unwrap();
            record.data["receipt"]["run_id"] = json!(RunId::new());
            tx.put(&command, &record)
        })
        .unwrap();
    assert_eq!(
        export_champion(&state, &run, &alg, Refit::Fit)
            .unwrap_err()
            .code,
        ErrorCode::ModelIntegrityFailed
    );
}
