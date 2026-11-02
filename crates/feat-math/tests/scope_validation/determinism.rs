use super::*;

// ---------------------------------------------------------------------------------------------
// Determinism and shipped text
// ---------------------------------------------------------------------------------------------

/// Every hand-written case and every corpus program, at every sweep width.
fn sweep() -> Vec<(String, usize)> {
    let mut sources: Vec<String> = HAND_CASES.iter().map(|s| s.to_string()).collect();
    sources.extend(corpus().into_iter().map(|(_, canonical)| canonical));
    sources
        .into_iter()
        .flat_map(|source| SWEEP_WIDTHS.map(|width| (source.clone(), width)))
        .collect()
}

#[test]
fn the_same_input_gives_the_same_result_and_the_same_bytes() {
    let mut digest = Sha256::new();
    let cases = sweep();
    for (source, width) in &cases {
        let genome = program(source);
        let first = validate_scope(&genome, *width);
        for _ in 0..3 {
            assert_eq!(
                validate_scope(&genome, *width),
                first,
                "{source} at {width}"
            );
        }
        // The same program, read back from its canonical text.
        let again = program(&genome.canonical());
        assert_eq!(validate_scope(&again, *width), first, "{source} at {width}");
        let text = match &first {
            Ok(()) => "ok".to_string(),
            Err(error) => error.to_string(),
        };
        digest.update(format!("{source}\t{width}\t{text}\n"));
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    eprintln!("DETERMINISM cases={} sha256={hex}", cases.len());
    // Every displayed result, reason bytes included, is pinned across runs and hosts: a change to
    // any of them fails here, and is made on purpose, with this digest and the WP-1.6 receipt.
    assert_eq!(
        (cases.len(), hex.as_str()),
        (
            780,
            "4579e907329496ced7ac5bfe8cabb519096eaa08b40a6eaba116c65a7317c63f"
        ),
        "the displayed results changed"
    );
}

#[test]
fn every_message_passes_the_shipped_text_lint() {
    let mut messages = BTreeSet::new();
    for (source, width) in sweep() {
        if let Err(error) = check(&source, width) {
            messages.insert(error.to_string());
            messages.insert(error.reason);
        }
    }
    assert!(messages.len() > 20, "{} messages", messages.len());
    for message in &messages {
        let hits = feat_math::transpile::lint_shipped_text(message);
        assert!(hits.is_empty(), "{message}: {hits:?}");
    }
}
