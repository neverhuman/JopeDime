#!/usr/bin/env python3
"""Audit compact H53b receipts without opening holdout or outer payloads."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/'jope-data/h53b/runs'
wave=json.loads((root/'wave.json').read_text())
assert wave['schema']=='jope.h53b.wave/v1'
assert len(wave['completed'])==6 and wave['complete'], 'wave incomplete'
rows=[]
for x in wave['completed']:
 p=root/f"scale-{x['scale']:g}"/f"seed{x['seed']}"/'summary.json'
 d=json.loads(p.read_text())
 assert d['rounds_completed']>0 and d['optimizer_updates_completed']==2*d['rounds_completed']
 assert d['r882_adapter_transfer_to_candidate'] is False
 assert d['dope_sha256_before']==d['dope_sha256_after']
 assert d['fresh_adapter_sha256_before_training']!=d['fresh_adapter_sha256_after_training']
 assert d['holdout_payloads_opened']==0 and d['outer_rows_delivered']==0
 rows.append({'seed':d['seed'],'scale':d['initialization_scale'],'rounds':d['rounds_completed'],'final_utility':d['final_family_utility'],'final_elo':d['elo_vs_r882_diagnostic']['ratings'].get('H53b'),'gap_harness':d['family_utility_gap_vs_harness_r882'],'gap_live_target':d['family_utility_gap_vs_live_target'],'family_bootstrap':d['family_bootstrap'],'device':d['device'],'wall_seconds':d['wall_seconds'],'r882_candidate_ceiling':d['post_r882_candidate_ceiling'],'rounds_to_cross':d['rounds_to_cross_at_diagnostic_reads']})
assert len(rows)==6
print(json.dumps({'cells':rows},sort_keys=True,indent=2))
