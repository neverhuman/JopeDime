#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;
    use serde_json::Value;

    fn reseal(table: &mut MotifTable) {
        table.table_sha256 = table.body.digest().expect("digest");
    }

    fn verify_code(table: &MotifTable) -> Option<&'static str> {
        table.verify().err().and_then(|error| code_of(&error))
    }

    #[test]
    fn motif_id_is_domain_separated_sha256_of_expansion() {
        let expansion = "(cheb append=true degree=3) (pair_prod append=true topk=24)";
        let id = motif_id(expansion);
        let mut framed = b"jope-motif-expansion/v1\0".to_vec();
        framed.extend_from_slice(expansion.as_bytes());
        assert_eq!(id, sha256_hex(&framed));
        assert_ne!(id, sha256_hex(expansion.as_bytes()));
        assert!(is_hex64(&id));
        let name = motif_name(&id);
        assert_eq!(name, format!("motif_{}", &id[..16]));
        assert_eq!(name.len(), NAME_PREFIX.len() + 16);
        assert_ne!(motif_id(expansion), motif_id("(standardize) (sigmoid)"));
    }

    #[test]
    fn names_use_reserved_prefix_and_collide_with_no_vocabulary() {
        let table = fixture::table();
        assert!(!table.body.motifs.is_empty());
        for motif in &table.body.motifs {
            assert!(motif.name.starts_with(NAME_PREFIX));
            assert!(name_is_reserved(&motif.name));
            assert!(registry::operator(&motif.name).is_none());
            assert!(!is_head_op(&motif.name));
            assert!(!is_target_op(&motif.name));
            assert!(stage_class(&motif.name).is_none());
            assert!(members::find(&motif.name).is_none());
            for op in &motif.signature.ops {
                assert!(!motif.name.contains(op.as_str()));
            }
        }
        for descriptor in registry::operator_registry() {
            assert!(!name_is_reserved(descriptor.name));
        }
        for member in &members::CORE_V1 {
            assert!(!name_is_reserved(member.name));
        }
    }

    #[test]
    fn table_digest_detects_any_body_edit() {
        let table = fixture::table();
        assert!(table.verify().is_ok());
        let edits: [fn(&mut TableBody); 5] = [
            |body| body.inputs.qualified += 1,
            |body| body.motifs[0].support.records += 1,
            |body| body.motifs[0].provenance[0].stage_offset += 1,
            |body| body.parameters.min_datasets += 1,
            |body| {
                let first = body.hosts.keys().next().cloned().expect("a host");
                body.hosts.insert(first, String::from("(pipe X (ridge))"));
            },
        ];
        for edit in edits {
            let mut edited = table.clone();
            edit(&mut edited.body);
            assert_eq!(verify_code(&edited), Some("table_digest_mismatch"));
        }
        let mut relabelled = table.clone();
        relabelled.schema = String::from("jope.motif-table/v2");
        assert_eq!(verify_code(&relabelled), Some("table_schema"));
    }

    #[test]
    fn unknown_fields_are_refused_at_every_level() {
        let table = fixture::table();
        let bytes = table.to_json().expect("serialize");
        assert_eq!(MotifTable::load(&bytes).expect("round trip"), table);
        let base: Value = serde_json::from_slice(&bytes).expect("json");
        let paths: [&[&str]; 12] = [
            &[],
            &["body"],
            &["body", "engine"],
            &["body", "source"],
            &["body", "parameters"],
            &["body", "inputs"],
            &["body", "motifs", "0"],
            &["body", "motifs", "0", "signature"],
            &["body", "motifs", "0", "signature", "declared_tasks"],
            &["body", "motifs", "0", "promotion"],
            &["body", "motifs", "0", "support"],
            &["body", "motifs", "0", "provenance", "0", "label"],
        ];
        for path in paths {
            let mut value = base.clone();
            let mut node = &mut value;
            for key in path {
                node = match key.parse::<usize>() {
                    Ok(index) => &mut node[index],
                    Err(_) => &mut node[*key],
                };
            }
            node.as_object_mut()
                .expect("an object")
                .insert(String::from("unexpected"), Value::Bool(true));
            let bytes = serde_json::to_vec(&value).expect("serialize");
            let error = MotifTable::load(&bytes).expect_err("unknown field refused");
            assert_eq!(code_of(&error), Some("table_schema"), "at {path:?}");
        }
        let mut occurrence = base.clone();
        occurrence["body"]["motifs"][0]["provenance"][0]
            .as_object_mut()
            .expect("an object")
            .insert(String::from("unexpected"), Value::Bool(true));
        let bytes = serde_json::to_vec(&occurrence).expect("serialize");
        assert!(MotifTable::load(&bytes).is_err());
        let mut gate = base;
        gate["body"]["motifs"][0]["promotion"]["ablation"] = Value::from("passed");
        let bytes = serde_json::to_vec(&gate).expect("serialize");
        assert!(MotifTable::load(&bytes).is_err());
    }

    #[test]
    fn verify_rejects_nested_noncanonical_or_renamed_entries() {
        let table = fixture::table();

        let mut renamed = table.clone();
        renamed.body.motifs[0].name = String::from("motif_0000000000000000");
        reseal(&mut renamed);
        assert_eq!(verify_code(&renamed), Some("fragment_name_mismatch"));

        let mut foreign_id = table.clone();
        foreign_id.body.motifs[0].id = motif_id("(standardize) (sigmoid)");
        reseal(&mut foreign_id);
        assert_eq!(verify_code(&foreign_id), Some("fragment_id_mismatch"));

        let set_expansion = |expansion: &str| {
            let mut edited = table.clone();
            let motif = &mut edited.body.motifs[0];
            motif.expansion = expansion.to_owned();
            motif.id = motif_id(expansion);
            motif.name = motif_name(&motif.id);
            reseal(&mut edited);
            edited
        };
        // Parameters out of canonical order.
        let noncanonical =
            set_expansion("(cheb degree=3 append=true) (pair_prod append=true topk=24)");
        assert_eq!(verify_code(&noncanonical), Some("expansion_not_canonical"));
        let single = set_expansion("(cheb append=true degree=3)");
        assert_eq!(verify_code(&single), Some("expansion_not_canonical"));
        let nested = set_expansion("(motif_0123456789abcdef) (pair_prod append=true topk=24)");
        assert_eq!(verify_code(&nested), Some("nested_reference"));
    }

    #[test]
    fn hand_written_table_without_provenance_is_refused() {
        let table = fixture::table();
        let mut bare = table.clone();
        for motif in &mut bare.body.motifs {
            motif.provenance.clear();
        }
        bare.body.hosts.clear();
        reseal(&mut bare);
        assert_eq!(verify_code(&bare), Some("provenance_inconsistent"));

        let mut empty_first = table;
        empty_first.body.motifs[0].provenance.clear();
        reseal(&mut empty_first);
        assert_eq!(verify_code(&empty_first), Some("provenance_inconsistent"));
    }

    #[test]
    fn verify_rejects_inconsistent_provenance() {
        let table = fixture::table();

        let mut missing_host = table.clone();
        missing_host.body.motifs[0].provenance[0].recipe_sha256 = fixture::digest("absent");
        reseal(&mut missing_host);
        assert_eq!(verify_code(&missing_host), Some("provenance_inconsistent"));

        let mut shifted = table.clone();
        shifted.body.motifs[0].provenance[0].stage_offset += 1;
        reseal(&mut shifted);
        assert_eq!(verify_code(&shifted), Some("provenance_inconsistent"));

        let mut extra = table.clone();
        let text = fixture::canonical("(pipe X (standardize) (ridge))");
        extra.body.hosts.insert(sha256_hex(text.as_bytes()), text);
        reseal(&mut extra);
        assert_eq!(verify_code(&extra), Some("provenance_inconsistent"));

        let mut mislabelled = table.clone();
        let key = mislabelled
            .body
            .hosts
            .keys()
            .next()
            .cloned()
            .expect("a host");
        mislabelled
            .body
            .hosts
            .insert(key, fixture::canonical("(pipe X (ridge))"));
        reseal(&mut mislabelled);
        assert_eq!(verify_code(&mislabelled), Some("provenance_inconsistent"));

        let mut adaptive = table.clone();
        adaptive.body.motifs[0].provenance[0].label = fixture::adaptive_label();
        reseal(&mut adaptive);
        assert_eq!(verify_code(&adaptive), Some("provenance_inconsistent"));

        let mut family = table;
        family.body.motifs[0].provenance[0].family = Some(String::from(fixture::FAMILY_A));
        reseal(&mut family);
        assert_eq!(verify_code(&family), Some("provenance_inconsistent"));
    }

    #[test]
    fn verify_rejects_inconsistent_input_counts_and_source() {
        let table = fixture::table();
        assert!(table.verify().is_ok());
        let edits: [fn(&mut TableBody); 6] = [
            // Under-reporting the excluded adaptive rows.
            |body| body.inputs.excluded_adaptive -= 1,
            |body| body.inputs.used += 1,
            |body| body.inputs.qualified += 1,
            |body| body.source.programs += 1,
            |body| body.source.sha256 = String::from("abc"),
            |body| body.source.schema.clear(),
        ];
        for edit in edits {
            let mut edited = table.clone();
            edit(&mut edited.body);
            reseal(&mut edited);
            assert_eq!(verify_code(&edited), Some("provenance_inconsistent"));
        }
    }

    #[test]
    fn verify_recomputes_support_and_signature() {
        let table = fixture::table();

        let mut datasets = table.clone();
        datasets.body.motifs[0].support.datasets += 1;
        reseal(&mut datasets);
        assert_eq!(verify_code(&datasets), Some("support_mismatch"));

        let mut seeds = table.clone();
        seeds.body.motifs[0].support.every_host_is_a_known_seed ^= true;
        reseal(&mut seeds);
        assert_eq!(verify_code(&seeds), Some("support_mismatch"));

        let mut filtered = table.clone();
        filtered.body.parameters.min_datasets = 50;
        reseal(&mut filtered);
        assert_eq!(verify_code(&filtered), Some("support_mismatch"));

        let mut tags = table.clone();
        tags.body.motifs[0]
            .signature
            .tags
            .push(String::from("extra"));
        reseal(&mut tags);
        assert_eq!(verify_code(&tags), Some("signature_drift"));

        let mut observed = table.clone();
        observed.body.motifs[0].signature.observed_tasks.clear();
        reseal(&mut observed);
        assert_eq!(verify_code(&observed), Some("signature_drift"));

        // Order is recomputed as well.
        let mut swapped = table;
        assert!(swapped.body.motifs.len() >= 2);
        swapped.body.motifs.swap(0, 1);
        reseal(&mut swapped);
        assert_eq!(verify_code(&swapped), Some("table_order"));
    }

    #[test]
    fn promoted_entries_are_refused_in_this_build() {
        let mut table = fixture::table();
        table.body.motifs[0].lifecycle = Lifecycle::Promoted;
        reseal(&mut table);
        assert_eq!(verify_code(&table), Some("promotion_evidence_missing"));
    }

    #[test]
    fn engine_binding_mismatch_is_reported_as_drift() {
        let table = fixture::table();
        assert_eq!(table.body.engine, EngineBinding::current());
        let edits: [fn(&mut EngineBinding); 3] = [
            |engine| engine.registry_sha256 = "0".repeat(64),
            |engine| engine.emitter_version = String::from("0.0.0"),
            |engine| engine.bound_artifact_schema = String::from("battle-artifact-v0"),
        ];
        for edit in edits {
            let mut drifted = table.clone();
            edit(&mut drifted.body.engine);
            reseal(&mut drifted);
            assert_eq!(verify_code(&drifted), Some("engine_drift"));
        }
    }
}
