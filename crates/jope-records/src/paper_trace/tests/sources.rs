use super::super::source_validation::source_file;
use super::*;

#[test]
fn source_parts_resolve_to_single_files_only() {
    let roots: BTreeMap<String, String> = [("LANE".to_owned(), "/lane".to_owned())].into();
    let file = |p: &str| source_file(p, &roots).map(|f| f.display().to_string());
    assert_eq!(
        file("LANE:a/b.json#.x.y").as_deref(),
        Some("/lane/a/b.json")
    );
    assert_eq!(file("LANE:c.md:4-5,9").as_deref(), Some("/lane/c.md"));
    assert_eq!(
        file("LANE:d.jsonl (rows with x)").as_deref(),
        Some("/lane/d.jsonl")
    );
    assert_eq!(file("LANE:e-0000{10..40}.json#.loss"), None);
    assert_eq!(file("LANE:r2*.json"), None);
    assert_eq!(file("NOPE:a.json"), None);
    assert_eq!(file("derived from C10 and C23"), None);
    assert_eq!(file("LANE:../outside.json"), None);
    assert_eq!(file("LANE:a/../../outside.json"), None);
    assert_eq!(file("LANE:/etc/passwd"), None);

    let ledger = r#"{"source_conventions": {"roots": {"LANE": "/usr"}}, "claims": [
      {"id": "C01", "statement": "s", "value": "1", "source": "LANE:../etc/passwd",
       "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "status": "established"},
      {"id": "C02", "statement": "s", "value": "2", "source": "LANE:/etc/passwd",
       "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "status": "established"}
    ]}"#;
    let (problems, _) = check_sources(ledger).expect("parses");
    assert!(
        problems
            .iter()
            .any(|p| p.contains("C01 cites unsafe path") && p.contains("../etc/passwd")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("C02 cites unsafe path") && p.contains("/etc/passwd")),
        "{problems:?}"
    );
}
