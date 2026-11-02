use super::*;

fn validate_config(cfg: &RouterPolicyConfig) -> Result<()> {
    if cfg.samples == 0 {
        bail!("router policy samples must be at least 1");
    }
    if cfg.top_n == 0 {
        bail!("router policy top_n must be at least 1");
    }
    if cfg.max_transforms == 0 {
        bail!("router policy max_transforms must be at least 1");
    }
    if cfg.max_transforms > 8 {
        bail!("router policy max_transforms must be at most 8");
    }
    Ok(())
}

fn embedding_from_json(value: &serde_json::Value) -> Option<Vec<f64>> {
    if let Some(values) = json_number_array(value) {
        return Some(values);
    }
    let obj = value.as_object()?;
    for key in [
        "router",
        "router128",
        "embedding",
        "dataset_vector",
        "vector",
        "vec",
        "features",
    ] {
        if let Some(values) = obj.get(key).and_then(json_number_array) {
            return Some(values);
        }
    }
    None
}

fn json_number_array(value: &serde_json::Value) -> Option<Vec<f64>> {
    let array = value.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        out.push(item.as_f64()?);
    }
    Some(out)
}

fn validate_supported_pipeline(source: &str) -> std::result::Result<String, Reject> {
    let parsed = GenomeIr::parse(source).map_err(|_| Reject::Parse)?;
    if !matches!(parsed, GenomeIr::Pipeline(_)) {
        return Err(Reject::Parse);
    }
    emit(&parsed, &EmitOptions::default()).map_err(|_| Reject::Unsupported)?;
    Ok(parsed.canonical())
}
