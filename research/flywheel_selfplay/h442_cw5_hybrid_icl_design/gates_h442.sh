#!/bin/bash
# H442 integrity gates (PREREG section 3). Run AFTER driver_h442.sh and BEFORE readout. Create-once.
#   gates_h442.sh PREREG_COMMIT -> $EV/results/H442.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H442
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h442/$(basename "$STAGE")}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h442_cw5_hybrid_icl_design
OUT=$EV/results/H442.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
SHA_VAL=8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0
H431_M20=/home/ubuntu/jope-data/h335-cl/h431/pub-r2/mon_M_h020

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H442.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/READOUT_CW5.json" ] && [ ! -e "$EV/results/FROZEN.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
n_ok=$(jq -s '[.[] | select(.cid != null and .ok == true)] | length' "$R/ctxs"/ctx_*.json)

replay_json=$(python3 -B - "$R/ctxs" "$H431_M20" <<'PY'
import json, sys
from pathlib import Path
cw_dir, h431_dir = Path(sys.argv[1]), Path(sys.argv[2])
n_f30, cid_ok, max_rel = 0, True, 0.0
for p in sorted(cw_dir.glob("ctx_*.json")):
    r = json.loads(p.read_text())
    if r["F"] <= 30:
        hp = h431_dir / p.name
        if not hp.exists():
            cid_ok = False
            continue
        hr = json.loads(hp.read_text())
        if hr.get("cid") != r["cid"]:
            cid_ok = False
        u0 = float(r["u"]["R0"])
        uh = float(hr["u"]["ridgeres"])
        rel = abs(u0 - uh) / max(1.0, abs(uh))
        if rel > max_rel:
            max_rel = rel
        n_f30 += 1
print(json.dumps({"f_le_30_n": n_f30, "cid_match_all": cid_ok, "max_rel_diff": max_rel}))
PY
)

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile un "$R/unit_tests.json" \
  --slurpfile st <(cat "$R/ctxs"/status_*.json) \
  --argjson rp "$replay_json" \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" --argjson nok "$n_ok" \
  --arg sm20 "$SHA_M20" --arg sval "$SHA_VAL" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $un[0] as $un |
  {schema: "jope.h442.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t, contexts_ok: $nok, ckpt_sha256: $sm20, val_sha256: $sval,
     total_threads: ([$st[].threads] | add), total_cpu_s: ([$st[].cpu_s] | add),
     r0_h431_replay: $rp
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: (([$st[].ckpt_sha256] | unique) == [$sm20] and ([$st[].val_sha256] | unique) == [$sval]),
     g3_cpu_only_16_threads: (([$st[].device] | unique) == ["cpu"] and ([$st[].cuda_visible] | any | not) and ([$st[].threads] | add) <= 16 and $t.wall_s <= 14400),
     g4_complete_106: ($nok == 106),
     g5_unit_tests: ($un.all_passed == true),
     g6_confirm_untouched: $view,
     g7_pub_stage: ($stage and $mode == "pub"),
     g8_r0_replays_h431_m20: ($rp.f_le_30_n > 0 and $rp.cid_match_all == true and $rp.max_rel_diff <= 1e-9)
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
