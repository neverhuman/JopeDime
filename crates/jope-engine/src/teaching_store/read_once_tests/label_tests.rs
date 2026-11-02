use super::*;

/// Route R8 at `teaching_store::teacher_programs` (:1243-1248 before the
/// fix): the index whose labels are read is the index whose digest the
/// provenance names, read once.
#[test]
fn teacher_labels_come_from_the_named_index_even_if_it_is_replaced_after_its_read() {
    let c = label_case("labels-honest");
    // Positive control.
    let programs = c.programs().unwrap();
    assert!(!programs.is_empty() && !forged(&programs));
    let _hook = replace_on_verification(&c.corpus.join("index.json"), c.forged.clone());
    let raced = c.programs();
    assert!(
        matches!(&raced, Ok(p) if !p.is_empty() && !forged(p)),
        "the labels were read from an index other than the one read and checked: {raced:?}"
    );
}

/// The same receiver, other direction: an index the provenance does not name
/// is refused, even if the named index replaces it after its read.
#[test]
fn teacher_labels_of_an_unnamed_index_are_refused_even_if_it_is_replaced_after_its_read() {
    let c = label_case("labels-forged");
    fs::write(c.corpus.join("index.json"), &c.forged).unwrap();
    // Negative control: refused without a replacement.
    let refused = c.programs().unwrap_err();
    assert!(refused.contains("label index changed"), "{refused}");
    let _hook = replace_on_verification(&c.corpus.join("index.json"), c.bound.clone());
    let raced = c.programs();
    assert!(
        matches!(&raced, Err(e) if e.contains("label index changed")),
        "teacher labels of an index the provenance does not name were returned: {raced:?}"
    );
}

/// The same receiver, bound to its caller (`load_resident_verified`,
/// `read_index`): the labels read are those of the index the caller verified.
#[test]
fn teacher_labels_are_those_of_the_callers_verified_index() {
    let mut c = label_case("labels-caller");
    // Positive control.
    assert!(!forged(&c.programs().unwrap()));
    c.index.records_sha256 = "0".repeat(64);
    let refused = c.programs();
    assert!(
        matches!(&refused, Err(e) if e.contains("label index changed")),
        "labels were read for an index other than the caller's verified index: {refused:?}"
    );
}

#[test]
fn the_corpus_vocabulary_is_the_bytes_its_index_binds_even_if_replaced_after_its_check() {
    let (s, corpus) = published("vocabulary-raced");
    let root = s.f.root();
    let path = corpus.join("vocabulary.json");
    let honest: crate::grammar::Vocab = read_json(&path).unwrap();
    let replacement = permuted_vocabulary(&path);
    assert_ne!(replacement, fs::read(&path).unwrap());
    // Positive control.
    assert_eq!(vocabulary_tokens(root, &corpus).unwrap(), honest.tokens);
    let _hook = replace_on_verification(&path, replacement);
    let raced = vocabulary_tokens(root, &corpus);
    assert!(
        matches!(&raced, Ok(tokens) if *tokens == honest.tokens),
        "the vocabulary consumed is not the one the corpus index binds: {raced:?}"
    );
}

/// A legacy (`parallel.teaching_corpus.v1`) index binds its vocabulary by
/// digest only; `load` never opens the file, so the reader must check it.
#[test]
fn a_legacy_corpus_vocabulary_other_than_its_index_binding_is_refused() {
    let f = Fixture::new();
    let corpus = f.root().join("legacy-corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::copy(f.vocab(), corpus.join("vocabulary.json")).unwrap();
    write_json(
        &corpus.join("index.json"),
        &json!({"schema":"parallel.teaching_corpus.v1","qualification":true,
            "vocabulary_sha256":file_hash(&corpus.join("vocabulary.json")).unwrap(),
            "context_index_sha256":"0".repeat(64),"records_sha256":"0".repeat(64),
            "record_bytes":crate::corpus::RECORD_BYTES,"records":0,"legal_sets":[],
            "contexts":[],"groups":[]}),
    )
    .unwrap();
    // Positive control.
    let honest: crate::grammar::Vocab = read_json(&f.vocab()).unwrap();
    assert_eq!(vocabulary_tokens(f.root(), &corpus).unwrap(), honest.tokens);
    fs::write(
        corpus.join("vocabulary.json"),
        permuted_vocabulary(&f.vocab()),
    )
    .unwrap();
    let refused = vocabulary_tokens(f.root(), &corpus);
    assert!(
        matches!(&refused, Err(e) if e.contains("vocabulary changed")),
        "a vocabulary the legacy index does not bind was consumed: {refused:?}"
    );
}

#[test]
fn a_resident_view_is_never_served_for_another_corpus_directory() {
    let (s, corpus) = published("resident-a");
    let root = s.f.root();
    let bytes = fs::read(corpus.join("index.json")).unwrap();
    // Positive control: a cold and a warm load of the corpus.
    store::load_resident_verified(root, &corpus, &bytes).unwrap();
    store::load_resident_verified(root, &corpus, &bytes).unwrap();
    // Another directory with the same index bytes and a changed payload.
    let other = root.join("resident-b");
    fs::create_dir_all(&other).unwrap();
    for name in ["index.json", "vocabulary.json", "provenance.json"] {
        fs::copy(corpus.join(name), other.join(name)).unwrap();
    }
    fs::write(other.join("groups.bin"), b"changed payload").unwrap();
    // Its cold load refuses.
    assert!(store::load(root, &other).is_err());
    let warm = store::load_resident_verified(root, &other, &bytes);
    assert!(
        warm.is_err(),
        "a resident view of another corpus directory was served for a changed payload"
    );
}

#[test]
fn legacy_teacher_labels_are_authenticated_by_the_verified_index_never_by_provenance() {
    let (s, legacy) = legacy_export("legacy-bound");
    let root = s.f.root();
    // Positive control.
    let programs = legacy_programs(root, &legacy).unwrap();
    assert!(!programs.is_empty() && !forged(&programs));
    // Forged labels, and a provenance naming their digest; its back-reference
    // to the index is unchanged.
    let labels = fs::read_to_string(legacy.join("labels.jsonl")).unwrap();
    let forged_labels = labels
        .lines()
        .map(|line| {
            let mut row: Value = serde_json::from_str(line).unwrap();
            row["program"] = json!("(pipe X (forged) (ridge))");
            format!("{}\n", serde_json::to_string(&row).unwrap())
        })
        .collect::<String>();
    fs::write(legacy.join("labels.jsonl"), &forged_labels).unwrap();
    let mut provenance: Value = read_json(&legacy.join("provenance.json")).unwrap();
    provenance["labels_sha256"] = json!(hash(forged_labels.as_bytes()));
    write_json(&legacy.join("provenance.json"), &provenance).unwrap();
    let refused = legacy_programs(root, &legacy);
    assert!(
        matches!(&refused, Err(e) if e.contains("teacher labels changed")),
        "forged legacy labels were returned on provenance.json's word: {refused:?}"
    );
}

#[test]
fn legacy_teacher_labels_without_an_index_binding_are_unbound_and_refused() {
    let (s, legacy) = legacy_export("legacy-unbound");
    let root = s.f.root();
    // An index written before the binding existed: no labels digest. Its
    // provenance names it by back-reference, as legacy corpora always did.
    let mut index: Value = read_json(&legacy.join("index.json")).unwrap();
    index.as_object_mut().unwrap().remove("labels_sha256");
    write_json(&legacy.join("index.json"), &index).unwrap();
    let mut provenance: Value = read_json(&legacy.join("provenance.json")).unwrap();
    provenance["index_sha256"] = json!(file_hash(&legacy.join("index.json")).unwrap());
    write_json(&legacy.join("provenance.json"), &provenance).unwrap();
    // The corpus itself still opens (its index is the legacy index).
    assert!(store::read_index(root, &legacy).is_ok());
    let refused = legacy_programs(root, &legacy);
    assert!(
        matches!(&refused, Err(e) if e.contains("unbound")),
        "labels of a legacy corpus bound only by provenance.json were trusted: {refused:?}"
    );
}
