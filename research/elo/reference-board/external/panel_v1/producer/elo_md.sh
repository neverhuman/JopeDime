#!/bin/bash
# refetch kumo results, regather, write ELO_VS_METHODS.md
cd ~/jope_plan/elo/panel_results
ssh xbabe3 "cat ~/jope_elo/kumo/res_kumo.*.jsonl" > kumo/res_kumo_x3.jsonl.tmp && mv kumo/res_kumo_x3.jsonl.tmp kumo/res_kumo_x3.jsonl
ssh xbabe4 "/usr/local/bin/docker exec jope-elo-kumo bash -c 'cat /k/res_kumo.*.jsonl'" > kumo/res_kumo_x4.jsonl.tmp && mv kumo/res_kumo_x4.jsonl.tmp kumo/res_kumo_x4.jsonl
cd .. && python3 gather_panel.py >/dev/null && python3 - <<'PY'
import json,os,datetime
o=json.load(open("panel_results/elo_panel_v1.json"))
L=["# Elo vs methods — panel_v1 (240 regression + 240 binary, family 5-fold)","",
f"Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC. Elo from pairwise per-dataset wins (RMSE lower is better; AUC higher is better). TabPFN excluded; strongest Kumo option = KumoTabular-large, 8 estimators.",""]
done=True
for t in ("regression","binary"):
  e=o[t]["elo"]; L+= [f"## {t.capitalize()} ({o[t]['n_datasets']} datasets)","","| Rank | Method | Elo | Coverage |","|---|---|---|---|"]
  for i,r in enumerate(e,1):
    m=r["method"]; L.append(f"| {i} | {'**'+m+'**' if 'jope' in m else m} | {r['elo']} | {r['coverage']}/{o[t]['n_datasets']} |")
    if r["coverage"]<o[t]['n_datasets']-5: done=False
  fr=[i for i,r in enumerate(e,1) if 'jope' in r['method']]
  L+=["",f"Fused model rank: **#{fr[0]} of {len(e)}**" if fr else "",""]
L.append("Status: COMPLETE" if done else "Status: PROVISIONAL — Kumo still running")
open(os.path.expanduser("~/jope_plan/ns/ELO_VS_METHODS.md"),"w").write("\n".join(L)+"\n")
print("\n".join(L))
PY
