use super::*;

pub(crate) fn build_repeat(
    purpose: &str,
    repeat: usize,
    seed: u64,
    targets: &[f64],
) -> Result<FoldRepeat> {
    if targets.len() < FOLDS || targets.iter().any(|value| !value.is_finite()) {
        bail!("fold bank needs at least five finite targets");
    }
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut strata = vec![Vec::new(); 10];
    let row_count = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        strata[(rank * 10 / row_count).min(9)].push(row);
    }
    let mut validation_indices = vec![Vec::new(); FOLDS];
    for (stratum, rows) in strata.iter_mut().enumerate() {
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"train-only-redteam-fold-order-v1\0");
            hash.update(seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hash.finalize().to_vec()
        });
        for (position, row) in rows.iter().enumerate() {
            validation_indices[(position + stratum) % FOLDS].push(*row);
        }
    }
    let mut coverage = vec![0_u8; targets.len()];
    for fold in &mut validation_indices {
        fold.sort_unstable();
        if fold.is_empty() {
            bail!("fold bank contains an empty fold");
        }
        for row in fold {
            coverage[*row] = coverage[*row].saturating_add(1);
        }
    }
    if coverage.iter().any(|count| *count != 1) {
        bail!("every row must appear exactly once as validation per repeat");
    }
    let repeat_sha256 = receipt_digest(&json!({
        "purpose": purpose,
        "repeat": repeat,
        "seed": seed,
        "validation_indices": validation_indices,
    }))?;
    Ok(FoldRepeat {
        purpose: purpose.to_string(),
        repeat,
        seed,
        validation_indices,
        repeat_sha256,
    })
}

pub(crate) fn build_fold_bank(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
    config: &TrainOnlyRedteamConfig,
) -> Result<FoldBankReceipt> {
    let make = |purpose: &str, seeds: &[u64]| {
        seeds
            .iter()
            .enumerate()
            .map(|(repeat, seed)| build_repeat(purpose, repeat, *seed, targets))
            .collect::<Result<Vec<_>>>()
    };
    let bank = FoldBankReceipt {
        schema: "TrainOnlyRedteamFoldBank.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        rows: targets.len(),
        folds: FOLDS,
        search: make("search", &config.fold_banks.search_seeds)?,
        selector: make("selector", &config.fold_banks.selector_seeds)?,
        audit: make("audit", &config.fold_banks.audit_seeds)?,
    };
    validate_fold_bank(&bank, config)?;
    Ok(bank)
}

pub(crate) fn validate_fold_bank(
    bank: &FoldBankReceipt,
    config: &TrainOnlyRedteamConfig,
) -> Result<()> {
    if bank.schema != "TrainOnlyRedteamFoldBank.v1"
        || bank.folds != FOLDS
        || bank.search.len() != 2
        || bank.selector.len() != 2
        || bank.audit.len() != 1
    {
        bail!("fold-bank shape drift");
    }
    let expected = [
        ("search", config.fold_banks.search_seeds.as_slice()),
        ("selector", config.fold_banks.selector_seeds.as_slice()),
        ("audit", config.fold_banks.audit_seeds.as_slice()),
    ];
    for (purpose, seeds) in expected {
        let repeats = match purpose {
            "search" => &bank.search,
            "selector" => &bank.selector,
            _ => &bank.audit,
        };
        for (repeat, expected_seed) in repeats.iter().zip(seeds) {
            if repeat.purpose != purpose || repeat.seed != *expected_seed {
                bail!("fold-bank purpose or seed drift");
            }
            let rebuilt = receipt_digest(&json!({
                "purpose": repeat.purpose,
                "repeat": repeat.repeat,
                "seed": repeat.seed,
                "validation_indices": repeat.validation_indices,
            }))?;
            if rebuilt != repeat.repeat_sha256 {
                bail!("fold-bank repeat digest drift");
            }
            let mut coverage = vec![0_u8; bank.rows];
            for fold in &repeat.validation_indices {
                for row in fold {
                    if *row >= bank.rows {
                        bail!("fold-bank row is out of bounds");
                    }
                    coverage[*row] = coverage[*row].saturating_add(1);
                }
            }
            if coverage.iter().any(|count| *count != 1) {
                bail!("fold-bank coverage drift");
            }
        }
    }
    Ok(())
}
