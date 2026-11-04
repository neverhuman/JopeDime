use super::*;

const DAT: &str = "epoch loss ce\n2 1.25 0\n3 0.95 0\n4 0.9001 0\n5 0.9005 0\n6 0.9002 0\n";

/// A ledger whose C26, C27 and C76 agree with `DAT` when the plateau starts at epoch 4.
fn ledger(dat: &str, c27: &str, c76: &str) -> String {
    let pin = py::sha256_hex(dat.as_bytes());
    format!(
        r#"{{"claims": [
              {{"id": "C26", "value": "1.2500000000 -> 0.9001000000 (ep 4) -> 0.9002000000 (ep 6)"}},
              {{"id": "C27", "value": "{c27}", "sha256": "{pin}"}},
              {{"id": "C76", "statement": "accuracy over 32 lineages", "value": "{c76}"}}
            ]}}"#
    )
}

fn problems(dat: &str, ledger_text: &str) -> Vec<String> {
    let mut found = vec![];
    let f = figure(dat.as_bytes(), ledger_text, &mut found).expect("parses");
    // Synthetic data plateaus at epoch 4, not 10; drop the start-epoch rule's finding.
    found.retain(|p| !p.contains("plateau from epoch") && !p.contains("already inside"));
    assert!(render(&f).ends_with("</svg>\n"));
    found
}

#[test]
fn the_figure_follows_its_data_and_its_claims() {
    let c76 = "0.5056656 [0.4412160, 0.5684545]";
    let ok = ledger(DAT, "x", c76);
    assert_eq!(problems(DAT, &ok), Vec::<String>::new());

    // Data edited after the pin: the hash and C26's last value no longer agree.
    let drifted = DAT.replace("6 0.9002", "6 0.9003");
    let found = problems(&drifted, &ok);
    assert!(
        found.iter().any(|p| p.contains("not C27's pin")),
        "{found:?}"
    );
    assert!(
        found.iter().any(|p| p.contains("not C26's value")),
        "{found:?}"
    );

    // An unreadable C76 value.
    let found = problems(DAT, &ledger(DAT, "x", "about half"));
    assert!(found.iter().any(|p| p.contains("C76 is not")), "{found:?}");

    // Valid and unreadable C190 values.
    let pin = py::sha256_hex(DAT.as_bytes());
    let with_c190 = format!(
        r#"{{"claims": [
              {{"id": "C26", "value": "1.2500000000 -> 0.9001000000 (ep 4) -> 0.9002000000 (ep 6)"}},
              {{"id": "C27", "value": "x", "sha256": "{pin}"}},
              {{"id": "C76", "statement": "accuracy over 32 lineages", "value": "{c76}"}},
              {{"id": "C190", "value": "8,908 S3 rows; shadow-pool-v2 (1,136 contexts, 255 clusters) R0->R4 pairwise 0.634318 -> 0.680342 (+0.046025 [+0.039928, +0.051692] GREATER)"}}
            ]}}"#
    );
    assert_eq!(problems(DAT, &with_c190), Vec::<String>::new());
    let mut found = vec![];
    let f = figure(DAT.as_bytes(), &with_c190, &mut found).expect("parses");
    let svg = render(&f);
    assert!(svg.contains("(b) Ranking: fresh programs vs. shadow pool"));
    assert!(svg.contains("fresh-program q-hat at chance; shadow-pool lift is non-qualifying."));
    assert!(svg.contains("v1 q-hat (fresh)"));
    assert!(svg.contains("R0 0.634 -> R4 0.680"));
    assert!(svg.contains("lift +0.046 [+0.040, +0.052]"));
    assert!(svg.contains("(255 clusters, non-qualifying)"));
    assert!(!svg.contains("(b) Fresh-program pairwise ranking accuracy"));
    // Only the vertical R0->R4 segment uses the surrogate stroke on <line>; no level caps around R4.
    assert_eq!(svg.matches(r##"stroke="#2e6b3e""##).count(), 2);

    let bad_c190 = with_c190.replace("R0->R4 pairwise", "broken");
    let found = problems(DAT, &bad_c190);
    assert!(found.iter().any(|p| p.contains("C190 is not")), "{found:?}");
}

#[test]
fn the_plateau_is_recomputed_from_epoch_ten() {
    let mut dat = String::from("epoch loss\n");
    for e in 2..=12 {
        let loss = if e < 9 {
            1.0
        } else {
            0.9 + f64::from(e) * 1e-5
        };
        dat.push_str(&format!("{e} {loss}\n"));
    }
    let mut found = vec![];
    let led = ledger(
        &dat,
        "[0.9001000, 0.9001200] over 3 epochs",
        "0.5 [0.4, 0.6]",
    );
    figure(dat.as_bytes(), &led, &mut found).expect("parses");
    // Epoch 9 (0.90009) is below the band, so the plateau's start is accepted.
    assert!(!found.iter().any(|p| p.contains("plateau")), "{found:?}");
    let led = ledger(
        &dat,
        "[0.9001000, 0.9001300] over 3 epochs",
        "0.5 [0.4, 0.6]",
    );
    let mut found = vec![];
    figure(dat.as_bytes(), &led, &mut found).expect("parses");
    assert!(
        found.iter().any(|p| p.contains("not C27's value")),
        "{found:?}"
    );
}

#[test]
fn the_rendering_is_deterministic_and_draws_every_epoch() {
    let mut found = vec![];
    let led = ledger(DAT, "x", "0.5056656 [0.4412160, 0.5684545]");
    let f = figure(DAT.as_bytes(), &led, &mut found).expect("parses");
    let svg = render(&f);
    assert_eq!(svg, render(&f));
    // One marker per epoch plus the ranking point.
    assert_eq!(svg.matches("<circle").count(), 5 + 1);
    assert!(svg.contains(">0.506<") && svg.contains(">[0.441, 0.568]<"));
    assert!(svg.contains(">32 lineages<"));
}
