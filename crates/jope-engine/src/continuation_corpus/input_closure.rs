use super::*;

pub fn input_closure(
    root: &Path,
    manifest: &Artifact,
    graph_state: &Path,
) -> Result<BTreeMap<PathBuf, String>> {
    let set: TargetSet = native_genome::load(root, manifest)?;
    crate::manifest::relative(&set.source_graph_state)?;
    ensure!(
        root.join(&set.source_graph_state) == graph_state,
        "source graph state differs from target-set binding"
    );
    let mut pending = vec![
        manifest.clone(),
        set.parent_corpus,
        set.vocabulary,
        set.admission_policy,
        set.frozen_scientific_policy,
    ];
    pending.extend(set.targets);
    pending.extend(set.preparation_overlay);
    let mut all = BTreeMap::new();
    fn refs(v: &Value, out: &mut Vec<Artifact>) {
        if let Some(o) = v.as_object() {
            if let (Some(path), Some(sha)) = (
                o.get("path").and_then(Value::as_str),
                o.get("sha256").and_then(Value::as_str),
            ) {
                if !Path::new(path).is_absolute()
                    && sha.len() == 64
                    && sha
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    out.push(Artifact {
                        path: path.into(),
                        sha256: sha.into(),
                    });
                }
            }
            for x in o.values() {
                refs(x, out)
            }
        } else if let Some(a) = v.as_array() {
            for x in a {
                refs(x, out)
            }
        }
    }
    while let Some(a) = pending.pop() {
        if let Some(old) = all.get(&a.path) {
            ensure!(old == &a.sha256, "source closure conflicting identity");
            continue;
        }
        ensure!(
            all.len() < 1_000_000,
            "source closure exceeds one million files"
        );
        let path = resolve(root, &a.path)?;
        ensure!(
            fs::symlink_metadata(&path)?.is_file(),
            "source closure special file"
        );
        ensure!(
            crate::manifest::file_hash(&path)? == a.sha256,
            "source closure file changed"
        );
        let length = fs::metadata(&path)?.len();
        let typed_large =
            length > 16 * 1024 * 1024 && path.extension().is_some_and(|x| x == "json");
        if length <= 16 * 1024 * 1024 || typed_large {
            let maximum = if typed_large {
                native_genome::DATASET_METADATA_LIMIT
            } else {
                16 * 1024 * 1024
            };
            let parsed =
                serde_json::from_slice::<Value>(&native_genome::checked(root, &a, maximum)?);
            if typed_large {
                ensure!(parsed.is_ok(), "large JSON corpus metadata is malformed");
            }
            if let Ok(value) = parsed {
                if typed_large {
                    ensure!(
                        matches!(
                            value["schema"].as_str(),
                            Some(
                                "parallel.teaching_corpus.v1"
                                    | "parallel.teaching_shards.v1"
                                    | "parallel.context_index.v1"
                                    | "jain.corpus_files.v1"
                                    | "jain.corpus_parts.v1"
                            )
                        ),
                        "large JSON source is not a supported typed corpus metadata manifest"
                    );
                }
                refs(&value, &mut pending);
                match value["schema"].as_str() {
                    Some("jope.native_graph_result.v1") => {
                        let result: crate::native_graph::VerifiedResult =
                            serde_json::from_value(value)?;
                        let key = hash(&serde_json::to_vec(&(
                            &result.run,
                            result.generation,
                            &result.task_id,
                        ))?);
                        let archived = graph_state
                            .join("archived-admissions")
                            .join(format!("{key}.json"));
                        // Completed predecessors must have finished the durable archive transition.
                        // Never capture an active pathname that the coordinator later removes.
                        let admission_path = archived;
                        let admission_ref = Artifact {
                            path: admission_path.strip_prefix(root)?.into(),
                            sha256: result.admission_sha256,
                        };
                        let admission: crate::native_graph::Admission =
                            native_genome::load(root, &admission_ref)?;
                        ensure!(
                            admission.manifest.sha256 == result.task_manifest_sha256,
                            "source admission task differs"
                        );
                        let task_path = resolve(root, &admission.manifest.path)?;
                        let (task, input) = crate::smartcluster_worker::verify_task_inputs(
                            root,
                            &task_path,
                            &admission.manifest.sha256,
                        )?;
                        let stem = task_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .context("source task stem")?;
                        pending.push(Artifact {
                            path: task_path
                                .with_file_name(format!("{stem}.input.json"))
                                .strip_prefix(root)?
                                .into(),
                            sha256: task.input_manifest_sha256,
                        });
                        pending.push(admission_ref);
                        for (path, sha256) in input.input_artifacts {
                            pending.push(Artifact { path, sha256 })
                        }
                        if let Some(c) = input.closure {
                            pending.push(Artifact {
                                path: c.path,
                                sha256: c.sha256,
                            })
                        }
                        if let crate::smartcluster_worker::Operation::Proposal { output, .. } =
                            input.operation
                        {
                            let producer: Value = native_genome::load(root, &result.output)?;
                            let artifacts: BTreeMap<PathBuf, String> =
                                serde_json::from_value(producer["artifacts"].clone())?;
                            for (path, sha256) in artifacts {
                                pending.push(Artifact {
                                    path: output.join(path),
                                    sha256,
                                })
                            }
                        }
                    }
                    Some("jope.native_genome_completion.v1") => {
                        let result: native_genome::GenomeResult = serde_json::from_value(value)?;
                        let base = result
                            .search
                            .path
                            .parent()
                            .context("original search parent")?;
                        for (path, sha256) in result.artifacts {
                            pending.push(Artifact {
                                path: base.join(path),
                                sha256,
                            })
                        }
                        for (path, sha256) in native_genome::required_inputs(root, &result.task)? {
                            pending.push(Artifact { path, sha256 })
                        }
                    }
                    _ => (),
                }
            }
        }
        all.insert(a.path, a.sha256);
    }
    Ok(all)
}
