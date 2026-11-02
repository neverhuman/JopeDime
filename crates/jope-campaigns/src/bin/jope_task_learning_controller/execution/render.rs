use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_spec(
    args: &Args,
    job_id: &str,
    output_name: &str,
    task: Task,
    mode: TrainerMode,
    seed: u64,
    corpus: &CorpusPointer,
    warm_path: &Path,
    warm_digest: &Sha256Digest,
    accepted_parent: &Sha256Digest,
    recipe: &str,
) -> Result<String> {
    let packed: Value = read_json(&corpus.packed_manifest)?;
    let packed_root = corpus
        .packed_manifest
        .parent()
        .context("packed manifest has no parent")?;
    let catalog = PathBuf::from(
        packed
            .get("catalog_data")
            .and_then(Value::as_str)
            .context("packed manifest lacks catalog data")?,
    );
    let catalog_sha = packed
        .get("catalog_data_sha256")
        .and_then(Value::as_str)
        .context("packed manifest lacks catalog digest")?;
    ensure!(
        catalog.is_file() && sha256_file(&catalog)? == catalog_sha,
        "packed catalog content drift"
    );
    let mode_name = match mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    let quoted = |value: &str| serde_json::to_string(value).expect("JSON string");
    let resource_block = TRAINER.render_resource_block();
    let placement_block = TRAINER.render_placement_block();
    let limits_block = TRAINER.render_limits_block();
    let mut text = format!(
        "[job]\nid = {}\nproject = \"{PROJECT}\"\ntier = \"normal\"\n\
         command = \"/usr/bin/python3\"\n\
         args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \"{{input:manifest}}\", \
         \"--output\", \"{{output:training}}/{output_name}\", \"--device\", \"cuda\", \
         \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \"{recipe}\", \"--seed\", \"{seed}\", \
         \"--task\", \"{}\", \"--mode\", \"{mode_name}\", \"--packed-manifest\", \
         \"{{input:packed_manifest}}\", \"--packed-catalog-data\", \"{{input:packed_catalog}}\", \
         \"--accepted-parent-sha256\", \"{}\"]\n\n\
         {resource_block}\n{placement_block}\n\
         [env]\nPYTHONPATH = {}\nOMP_NUM_THREADS = \"{}\"\n\
         JOPE_TRAINER_INTENT_SCHEMA = \"TrainerIntent.v3\"\n\n\
         [preempt]\npreemptible = {}\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n\
         {limits_block}\n",
        quoted(job_id),
        task.as_str(),
        accepted_parent.as_str(),
        quoted(&args.prime_source.to_string_lossy()),
        TRAINER.omp_threads,
        TRAINER.preemptible,
    );
    let mut inputs = vec![
        (
            "script",
            args.trainer_script.clone(),
            "scripts/test_exact_train.py".to_owned(),
            None,
        ),
        (
            "common",
            args.common_script.clone(),
            "scripts/test_exact_common.py".to_owned(),
            None,
        ),
        (
            "packed_source",
            args.packed_script.clone(),
            "scripts/test_exact_packed_corpus.py".to_owned(),
            None,
        ),
        (
            "manifest",
            corpus.manifest.clone(),
            "corpus/corpus-manifest.json".to_owned(),
            Some(corpus.manifest_sha256.clone()),
        ),
        (
            "checkpoint",
            warm_path.to_owned(),
            format!("checkpoint/{}.model.pt", warm_digest.as_str()),
            Some(warm_digest.as_str().to_owned()),
        ),
        (
            "packed_manifest",
            corpus.packed_manifest.clone(),
            "packed/manifest.json".to_owned(),
            Some(corpus.packed_manifest_sha256.clone()),
        ),
        (
            "packed_catalog",
            catalog,
            "packed/catalog-data.f32le".to_owned(),
            Some(catalog_sha.to_owned()),
        ),
        (
            "trainer_source",
            args.prime_source.join("jope/policy/train_v3.py"),
            "provenance/train_v3.py".to_owned(),
            None,
        ),
    ];
    let files = packed
        .get("files")
        .and_then(Value::as_object)
        .context("packed manifest lacks files")?;
    for (index, (name, metadata)) in files.iter().enumerate() {
        let path = packed_root.join(name);
        let digest = metadata
            .get("sha256")
            .and_then(Value::as_str)
            .context("packed file lacks digest")?;
        inputs.push((
            Box::leak(format!("packed_{index}").into_boxed_str()),
            path,
            format!("packed/{name}"),
            Some(digest.to_owned()),
        ));
    }
    for (name, source, target, pinned) in inputs {
        ensure!(
            source.is_file(),
            "trainer input is absent: {}",
            source.display()
        );
        let digest = pinned.unwrap_or(sha256_file(&source)?);
        ensure!(
            sha256_file(&source)? == digest,
            "trainer input content drift: {}",
            source.display()
        );
        text.push_str(&format!(
            "[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n\n",
            quoted(name),
            quoted(&source.to_string_lossy()),
            quoted(&target),
            quoted(&digest),
        ));
    }
    text.push_str("[[output]]\nname = \"training\"\nsource = \"training\"\ndirectory = true\n");
    Ok(text)
}
