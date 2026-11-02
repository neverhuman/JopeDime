use super::*;

#[test]
fn controls_expand_to_their_own_canonical_bytes() {
    let table = fixture::table();
    let champion = feat_math::CHAMPION_GENOME.trim();
    let portable = fixture::PORTABLE_4LANG.trim();
    let roster = "(pipe X (input categorical=onehot max_onehot=256 min_count=1 \
                  missing=indicator numeric=median) (ridge))";
    for literal in [champion, portable, roster] {
        let expanded = expand(literal, &table).expect("expands");
        assert_eq!(expanded.canonical, literal);
        assert_eq!(expanded.recipe_sha256, sha256_hex(literal.as_bytes()));
    }
    for member in &feat_math::members::CORE_V1 {
        if member.pipe.is_empty() {
            continue;
        }
        let expanded = expand(member.pipe, &table).expect("expands");
        assert_eq!(expanded.canonical, fixture::canonical(member.pipe));
    }
}

#[test]
fn rewritten_controls_emit_the_emitter_lane_pinned_bytes() {
    let champion = feat_math::CHAMPION_GENOME.trim();
    let portable = fixture::PORTABLE_4LANG.trim();
    let set = fixture::set_of(vec![
        fixture::qualified(champion, TaskLabel::Regression, "d1", "r1"),
        fixture::qualified(portable, TaskLabel::Regression, "d2", "r2"),
    ]);
    let table = mine(&set, None, MineParameters::default()).expect("mines");
    assert!(!table.body.motifs.is_empty());
    let controls = [
        (champion, [PINNED_CHAMPION_PY, PINNED_CHAMPION_RS, "", ""]),
        (
            portable,
            [
                PINNED_PORTABLE_PY,
                PINNED_PORTABLE_RS,
                PINNED_PORTABLE_CPP,
                PINNED_PORTABLE_JS,
            ],
        ),
    ];
    let options = EmitOptions::default();
    for (control, [python, rust, cpp, javascript]) in controls {
        let mut rewritten_any = false;
        for motif in &table.body.motifs {
            let Ok(rewritten) = rewrite(control, motif, At::All) else {
                continue;
            };
            rewritten_any = true;
            assert!(rewritten.contains(&motif.name));
            let expanded = expand(&rewritten, &table).expect("expands");
            assert_eq!(expanded.canonical, control);
            let ir = GenomeIr::parse(&expanded.canonical).expect("parses");
            let emitted = transpile::emit(&ir, &options).expect("emits");
            assert_eq!(sha256_hex(emitted.python.as_bytes()), python.trim());
            assert_eq!(sha256_hex(emitted.rust.as_bytes()), rust.trim());
            if !cpp.is_empty() {
                let source = transpile::emit_cpp17(&ir, &options).expect("C++17 emits");
                assert_eq!(sha256_hex(source.as_bytes()), cpp.trim());
            }
            if !javascript.is_empty() {
                let source = transpile::emit_javascript(&ir, &options).expect("JavaScript emits");
                assert_eq!(sha256_hex(source.as_bytes()), javascript.trim());
            }
        }
        assert!(rewritten_any);
    }
}
