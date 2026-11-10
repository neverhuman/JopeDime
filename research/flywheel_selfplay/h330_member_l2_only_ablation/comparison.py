"""Pinned H327 REG versus H330 L2ONLY descriptions; no decision function."""
def compare(historical, current, paired, gap):
    assert historical['wave_id'] == 'H327'
    assert len(historical['arms']['REG']['cells']) == len(current['L2ONLY']['cells']) == 9
    for a,b in zip(historical['arms']['BASE']['cells'], current['BASE']['cells']):
        assert (a['partition'],a['root_seed'],a['family_mean_gain']) == (b['partition'],b['root_seed'],b['family_mean_gain'])
    cells=[]
    for a,b in zip(historical['arms']['REG']['cells'],current['L2ONLY']['cells']):
        assert (a['partition'],a['root_seed']) == (b['partition'],b['root_seed'])
        cells.append(dict(partition=a['partition'],root_seed=a['root_seed'],
            REG_gain=a['family_mean_gain'],L2ONLY_gain=b['family_mean_gain'],
            REG_harm=a['harmed_rate'],L2ONLY_harm=b['harmed_rate'],
            REG_drop10=a['drop_top']['10'],L2ONLY_drop10=b['drop_top']['10'],
            REG_elo=a['elo'],L2ONLY_elo=b['elo']))
    keys=['median_gain','ci95','median_harmed_rate','median_elo','drop_top']
    return dict(scope='DESCRIPTIVE_ONLY_NO_EXTRA_DECISION',historical_wave='H327',
        historical_verdict=historical['verdict'],historical_rng=327191,current_rng=330191,
        interval_note='Historical REG intervals use RNG327191; L2ONLY uses RNG330191. No additional REG-vs-L2ONLY hypothesis test.',
        arms={a:{k:record[k] for k in keys} for a,record in [('REG',historical['arms']['REG']['aggregate']),('L2ONLY',current['L2ONLY']['aggregate'])]},
        paired_vs_BASE={a:{k:record[k] for k in ['median','ci95','lcb95','per_seed_medians']} for a,record in [('REG',historical['paired_drop_top10']),('L2ONLY',paired)]},
        member_gap={a:record for a,record in [('REG',historical['member_overfit_diagnostic']['arms']['REG']),('L2ONLY',gap['arms']['L2ONLY'])]},cells=cells)
