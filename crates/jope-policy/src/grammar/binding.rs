/// Bind vocabulary bytes to a checkpoint. Accepts the checkpoint's own vocabulary, or a
/// documented product restriction whose base is that vocabulary and whose restriction block
/// declares an unchanged token table. Everything else is refused.
pub fn bind_vocabulary(
    bytes: &[u8],
    checkpoint_vocabulary_sha256: &str,
) -> Result<BoundVocabulary> {
    let sha256 = sha256_hex(bytes);
    let bound = checkpoint_vocabulary_sha256.to_ascii_lowercase();
    let identity = if sha256 == bound {
        VocabularyIdentity::Checkpoint
    } else if let Some((_, base)) = PRODUCT_RESTRICTIONS
        .iter()
        .find(|(restricted, base)| *restricted == sha256 && *base == bound)
    {
        VocabularyIdentity::ProductRestriction {
            base_sha256: (*base).to_string(),
        }
    } else {
        bail!(
            "refused: vocabulary {sha256} is neither the checkpoint vocabulary {bound} nor a documented product restriction of it"
        );
    };
    let vocab = Vocab::from_json(bytes)?;
    match &identity {
        VocabularyIdentity::Checkpoint => ensure!(
            vocab.restriction.is_none(),
            "checkpoint vocabulary must not carry a restriction block"
        ),
        VocabularyIdentity::ProductRestriction { base_sha256 } => {
            let r = vocab
                .restriction
                .as_ref()
                .context("product vocabulary lacks its restriction block")?;
            ensure!(
                r.get("schema").and_then(Value::as_str) == Some("parallel.vocab_restriction.v1")
                    && r.get("base_vocabulary_sha256").and_then(Value::as_str)
                        == Some(base_sha256.as_str())
                    && r.get("token_table").and_then(Value::as_str) == Some("unchanged"),
                "product vocabulary restriction block does not declare a token-compatible restriction of {base_sha256}"
            );
        }
    }
    Ok(BoundVocabulary {
        vocab: Arc::new(vocab),
        sha256,
        identity,
    })
}

/// Full token-compatibility proof of a restriction against its base: identical token table,
/// expression tables, engines, tasks and append ids; every kept operator identical to the
/// base operator; every removed operator listed in the restriction block.
pub fn verify_restriction(base: &Vocab, restricted: &Vocab) -> Result<()> {
    ensure!(
        base.tokens == restricted.tokens
            && base.expr == restricted.expr
            && base.engines == restricted.engines
            && base.engine_ids == restricted.engine_ids
            && base.tasks == restricted.tasks
            && base.task_ids == restricted.task_ids
            && base.append_ids == restricted.append_ids
            && base.max_transforms == restricted.max_transforms
            && base.max_expr_stages == restricted.max_expr_stages,
        "restriction changed the token table or grammar constants"
    );
    ensure!(
        base.engine_transforms
            .keys()
            .eq(restricted.engine_transforms.keys())
            && base.engine_heads.keys().eq(restricted.engine_heads.keys()),
        "restriction changed the engine operator tables"
    );
    for (id, op) in &restricted.ops {
        ensure!(
            base.ops.get(id) == Some(op),
            "restriction changed operator {id}"
        );
    }
    let removed = restricted
        .restriction
        .as_ref()
        .and_then(|r| r.get("removed_ops"))
        .and_then(Value::as_array)
        .context("restriction lists no removed_ops")?
        .iter()
        .map(|r| {
            r.get("token_id")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .context("removed op without token_id")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let actual = base
        .ops
        .keys()
        .filter(|id| !restricted.ops.contains_key(id))
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        removed == actual,
        "restriction removed_ops disagrees with the operator tables"
    );
    for (engine, ids) in &restricted.engine_transforms {
        let expected = base
            .engine_transforms
            .get(engine)
            .with_context(|| format!("restriction lists transforms for unknown engine {engine}"))?
            .iter()
            .filter(|id| restricted.ops.contains_key(id))
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            *ids == expected,
            "restriction reordered {engine} transforms"
        );
    }
    for (engine, ids) in &restricted.engine_heads {
        let expected = base
            .engine_heads
            .get(engine)
            .with_context(|| format!("restriction lists heads for unknown engine {engine}"))?
            .iter()
            .filter(|id| restricted.ops.contains_key(id))
            .copied()
            .collect::<Vec<_>>();
        ensure!(*ids == expected, "restriction reordered {engine} heads");
    }
    Ok(())
}
