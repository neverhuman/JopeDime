use super::*;

/// A seeded, vocabulary-adapted search whose seed slots name proposal records (W4-T07), with
/// the caller's input files removed afterwards. Returns the output and the input bytes.
pub(crate) fn frozen_input_search(
    root: &Path,
    prepared: &Path,
    authority: &Fixture,
    name: &str,
) -> (PathBuf, Vec<u8>, Vec<u8>) {
    let seeds = root.join(format!("{name}-seeds.json"));
    seed_population(
        prepared,
        &seeds,
        &[PROGRAM, "(pipe X (standardize) (ridge))"],
        |_| None,
    );
    let mut population: Value = read_json(&seeds).unwrap();
    for (slot, candidate) in population["candidates"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        candidate["proposal_record_sha256"] =
            json!(hash(format!("proposal record {slot}").as_bytes()));
    }
    write_json(&seeds, &population).unwrap();
    let vocabulary = root.join(format!("{name}-vocabulary.json"));
    fs::copy(vocabulary_path(), &vocabulary).unwrap();
    let output = root.join(name);
    success(
        search_client(authority, root, prepared, &output, 2)
            .arg("--seeds")
            .arg(&seeds)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let bytes = (fs::read(&seeds).unwrap(), fs::read(&vocabulary).unwrap());
    // The caller's copies are gone: only the search output remains.
    fs::remove_file(&seeds).unwrap();
    fs::remove_file(&vocabulary).unwrap();
    (output, bytes.0, bytes.1)
}

/// W5-T01: a seeded search freezes the exact seed population and vocabulary bytes its request
/// pins, so its lineage (including each seed slot's proposal-record join) verifies from the
/// output alone. Reproduction: the output held only the digests, and the lineage verifier
/// needed the caller's original paths.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn a_seeded_search_freezes_the_inputs_its_lineage_is_verified_against() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, vocabulary) = frozen_input_search(&root, &prepared, &authority, "frozen");
    let inputs = output.join("search-inputs");
    let (frozen_seeds, frozen_vocabulary) =
        (inputs.join("seeds.json"), inputs.join("vocabulary.json"));
    assert_eq!(
        (
            fs::read(&frozen_seeds).ok(),
            fs::read(&frozen_vocabulary).ok()
        ),
        (Some(seeds), Some(vocabulary)),
        "the search did not freeze the inputs its request pins"
    );
    let verified = lineage::verify_search(
        &root,
        &output,
        Some(frozen_seeds.as_path()),
        Some(frozen_vocabulary.as_path()),
    )
    .unwrap();
    assert_eq!(verified.lineage.chains[0].source, "seed_population_slot");
    // The W4-T07 join reached the candidate ledger.
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(
        rows[0]["lineage"]["source"]["proposal_record_sha256"],
        json!(hash(b"proposal record 0"))
    );
    authority.stop_service();
}
