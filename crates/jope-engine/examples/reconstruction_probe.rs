//! Read-only reconstruction qualification; never admits data or opens test.csv.
use anyhow::{Context, Result};
use jope_parallel::{
    manifest::*,
    prepare::{reconstruct_numeric, Table},
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!(
        args.len() == 3,
        "usage: reconstruction_probe candidates.json regression_source_root output.json"
    );
    let input = PathBuf::from(&args[0]);
    let source = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    anyhow::ensure!(!output.exists(), "probe receipt already exists");
    let candidates: Vec<Value> = read_json(&input)?;
    let rows=rayon::ThreadPoolBuilder::new().num_threads(4).build()?.install(||candidates.par_iter().map(|candidate|{
        let id=candidate["id"].as_str().or_else(||candidate["dataset"].as_str()).unwrap_or("");
        let path=source.join(id);
        let result=(||->Result<Value>{
            anyhow::ensure!(id.len()==16&&id.chars().all(|c|c.is_ascii_hexdigit()),"invalid dataset identity");
            let meta:Value=read_json(&path.join("meta.json"))?;
            let training=Table::read(&path.join("train.csv"),&meta)?;
            let (raw,receipt)=reconstruct_numeric(&training,&meta)?;
            let kinds=meta["feature_encodings"].as_object().context("feature encodings")?;
            Ok(json!({"id":id,"reconstructed":true,"rows":raw.rows.len(),"features":raw.width-1,
                "source_dataset_id":meta["source_dataset_id"],"source":receipt["source"],"source_sha256":receipt["source_sha256"],
                "metadata_sha256":file_hash(&path.join("meta.json"))?,"training_sha256":file_hash(&path.join("train.csv"))?,
                "row_mapping_sha256":hash(&serde_json::to_vec(&receipt["source_row_indices"])?),
                "max_match_abs_error":receipt["max_match_abs_error"],"equivalent_source_matches":receipt["equivalent_source_matches"],
                "historical_scaler_feature_indices":receipt["historical_scaler_feature_indices"],
                "categorical_features":kinds.iter().filter(|(_,v)|v["kind"]=="categorical_codes").map(|(k,_)|k).collect::<Vec<_>>(),
                "provenance_admission_performed":false}))
        })();
        result.unwrap_or_else(|error|json!({"id":id,"reconstructed":false,"error":format!("{error:#}"),"provenance_admission_performed":false}))
    }).collect::<Vec<_>>());
    write_json(
        &output,
        &json!({"schema":"parallel.reconstruction_probe.v1","input_sha256":file_hash(&input)?,
        "original_test_targets_opened":false,"datasets":rows,"provenance_admission_performed":false}),
    )?;
    println!(
        "{}",
        json!({"datasets":rows.len(),"reconstructed":rows.iter().filter(|r|r["reconstructed"]==true).count(),"receipt":output})
    );
    Ok(())
}
