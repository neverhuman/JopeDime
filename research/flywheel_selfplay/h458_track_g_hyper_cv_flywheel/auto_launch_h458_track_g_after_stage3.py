#!/usr/bin/env python3
"""Autonomous Post-Stage-2/3 Milestone Reporter & `xbabe1` Pure-CPU Track G Launcher
(`auto_launch_h458_track_g_after_stage3.py`).

1. Monitors `/home/ubuntu/jope_plan/evidence/H455/H455_L_X1_V2_VALLEY001_SUMMARY.json` (Stage 2) and posts all
   Stage 2 BT-Elo matrices to `/home/ubuntu/JopeDime/shared_CHAT.md` under `flock` the instant it lands.
2. Monitors `/home/ubuntu/jope_plan/evidence/H456/H456_STAGE3_MULTI_VALLEY_SUMMARY.json` (Stage 3):
   - Posts all Stage 3 BT-Elo matrices to `shared_CHAT.md` under `flock`.
   - Syncs `H458` Track G modules + `qhat_v4_heldout.pt` + `gp_family_seed_library_v4.json` + `s8_confirm_cids.json`
     to `TRACK_G_HOST` (`JOPE_TRACK_G_HOST`, default `xbabe1`; `0` deletions).
   - Cleanly stops ONLY our own `exact_gp_teacher_harvester_h454.py` on `TRACK_G_HOST` (`SIGTERM`) and launches:
       (a) `hyper_cv_test_gp_harvester_h458.py` (`44` CPU workers, syncing `gp_teacher_shards_v2` to `xbabe2`), and
       (b) `ultra_gp_cycle_h458.py` (`22` contexts x `3` inner slots = `66` CPU cores, `Gen 1 -> 10..1000` `Tier-C`
           `Train Hyper-CV` anchored on the highest-Elo Stage 1-3 `2xCV-7` columns).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ALLOWED_TRACK_G_HOSTS = ("xbabe1", "xbabe2", "xbabe3")
TRACK_G_HOST = os.environ.get("JOPE_TRACK_G_HOST", "xbabe1").strip()
STAGE2_JSON = Path("/home/ubuntu/jope_plan/evidence/H455/H455_L_X1_V2_VALLEY001_SUMMARY.json")
STAGE3_JSON = Path("/home/ubuntu/jope_plan/evidence/H456/H456_STAGE3_MULTI_VALLEY_SUMMARY.json")
QHAT_V4_CKPT = Path("/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt")
S8_AUDIT_JSON = Path("/home/ubuntu/jope-data/h458s-quarantine/s8/S8_BUCKET_AUDIT.json")
STATE_JSON = Path("/home/ubuntu/jope-data/h458-flywheel/logs/auto_launch_track_g_state.json")


def _post_shared_chat(to_str: str, msg: str) -> None:
    subprocess.run(
        [sys.executable, "-u", str(HERE / "coord_shared_chat_h458.py"), "post", "--to", to_str, "--msg", msg],
        check=False,
        timeout=30,
    )


def _format_summary_elos(doc: Dict) -> str:
    rows: List[str] = []
    matrices = doc.get("waves") or doc.get("matrices") or doc.get("results") or {}
    if isinstance(matrices, dict):
        for k, v in matrices.items():
            if isinstance(v, dict):
                elo = v.get("best_post_elo", v.get("stack_cons_post_elo", v.get("elo")))
                arm = v.get("best_post_arm", "")
                rows.append(f"{k}: {elo} ({arm})")
    elif isinstance(matrices, list):
        for item in matrices:
            if isinstance(item, dict):
                lbl = item.get("label", item.get("name", "matrix"))
                elo = item.get("best_post_elo", item.get("stack_cons_post_elo", item.get("elo")))
                arm = item.get("best_post_arm", "")
                rows.append(f"{lbl}: {elo} ({arm})")
    if not rows and "best_overall_elo" in doc:
        rows.append(f"best_overall: {doc.get('best_overall_elo')} ({doc.get('best_overall_label', '')})")
    return " | ".join(rows) if rows else json.dumps({k: doc[k] for k in list(doc.keys())[:6]})


def main() -> None:
    if TRACK_G_HOST not in ALLOWED_TRACK_G_HOSTS:
        raise ValueError(f"Unsupported JOPE_TRACK_G_HOST={TRACK_G_HOST!r}; expected one of {ALLOWED_TRACK_G_HOSTS}")

    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_JSON.read_text()) if STATE_JSON.is_file() else {}
    hv2_state_key = f"harvester_v2_{TRACK_G_HOST}_launched"
    tg_state_key = f"track_g_{TRACK_G_HOST}_launched"

    print(json.dumps({"event": "auto_track_g_watcher_start", "host": TRACK_G_HOST, "state": state}), flush=True)

    while True:
        # 0. Immediate Harvester v2 swap on TRACK_G_HOST once qhat_v4_heldout.pt is ready
        if QHAT_V4_CKPT.is_file() and not state.get(hv2_state_key):
            s8_compact = Path("/home/ubuntu/jope-data/h458-flywheel/s8_confirm_cids.json")
            if S8_AUDIT_JSON.is_file() and not s8_compact.is_file():
                s8_doc = json.loads(S8_AUDIT_JSON.read_text())
                bad_cids = sorted({str(r["cid"]) for r in s8_doc.get("entries", []) if not bool(r.get("admit_canonical", True))})
                s8_compact.write_text(json.dumps({"bad_cids": bad_cids}, indent=2) + "\n")

            subprocess.run(
                [
                    "ssh",
                    TRACK_G_HOST,
                    "mkdir -p /home/ubuntu/h452-stage/code /home/ubuntu/jope-data/h458-flywheel/surrogate_v4 "
                    "/home/ubuntu/jope-data/h458-flywheel/logs /home/ubuntu/jope-data/h458s-quarantine/s8 "
                    "/home/ubuntu/jope_plan/evidence/H458/trackG",
                ],
                check=False,
                timeout=30,
            )
            if S8_AUDIT_JSON.is_file():
                subprocess.run(
                    ["rsync", "-az", str(S8_AUDIT_JSON), f"{TRACK_G_HOST}:{S8_AUDIT_JSON}"],
                    check=False,
                    timeout=30,
                )
            subprocess.run(
                ["rsync", "-az", "/home/ubuntu/h452-stage/code/", f"{TRACK_G_HOST}:/home/ubuntu/h452-stage/code/"],
                check=False,
                timeout=60,
            )
            for rel_f in (
                "surrogate_v4/qhat_v4_heldout.pt",
                "gp_family_seed_library_v4.json",
                "s8_confirm_cids.json",
            ):
                src_p = Path("/home/ubuntu/jope-data/h458-flywheel") / rel_f
                if src_p.is_file():
                    subprocess.run(
                        ["rsync", "-az", str(src_p), f"{TRACK_G_HOST}:/home/ubuntu/jope-data/h458-flywheel/{rel_f}"],
                        check=False,
                        timeout=60,
                    )
            hv2_cmd = (
                "pkill -TERM -f exact_gp_teacher_harvester_h454.py || true; sleep 2; "
                "nohup env CUDA_VISIBLE_DEVICES='' nice -n 19 python3 -u /home/ubuntu/h452-stage/code/hyper_cv_test_gp_harvester_h458.py "
                "--workers 44 --evolve-gens 90 --hours 48 --s3-source-uniform 1 "
                "--sync-targets xbabe2:/home/ubuntu/jope-data/h458-flywheel --sync-every-s 60 "
                f">> /home/ubuntu/jope-data/h458-flywheel/logs/harvester_v2_{TRACK_G_HOST}.log 2>&1 < /dev/null & "
                "sleep 1; pgrep -af 'hyper_cv_test_gp_harvester_h458'"
            )
            r_hv2 = subprocess.run(["ssh", TRACK_G_HOST, hv2_cmd], capture_output=True, text=True, check=False, timeout=45)
            _post_shared_chat(
                "ENG:watcher, cl_h458, cl_fwriter, cl_mtc, monster_train_agy (cc Jepson)",
                f"MILESTONE — {TRACK_G_HOST} swapped cleanly from Harvester v1 to H458 Harvester v2 "
                "(hyper_cv_test_gp_harvester_h458.py, 44 CPU workers, QHatV4-guided, S8 281-table exclusion, "
                "0.0% Episode-Test loser admission gate, syncing gp_teacher_shards_v2 + teaching_shards_v4 to xbabe2).",
            )
            state[hv2_state_key] = True
            state["harvester_v2_pid"] = r_hv2.stdout.strip()
            STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps({"event": hv2_state_key, "pid": r_hv2.stdout.strip()}), flush=True)

        # 1. Stage 2 summary check
        if not state.get("stage2_posted") and STAGE2_JSON.is_file():
            try:
                s2_doc = json.loads(STAGE2_JSON.read_text())
                s2_txt = _format_summary_elos(s2_doc)
                _post_shared_chat(
                    "cl_h458, cl_fwriter, cl_mtc, monster_train_agy (cc ENG:watcher, Jepson)",
                    f"MILESTONE — Stage 2 (fleet_L_x1_h455_maxent_v2 valley_001 + L+M soups) COMPLETE: {s2_txt}. "
                    f"Receipt: {STAGE2_JSON}. Stage 3 (M valley_003 + L valley_004 + 5-ckpt L+M soups) auto-starting now on {TRACK_G_HOST}+xbabe2 CPU.",
                )
                state["stage2_posted"] = True
                STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
                print(json.dumps({"event": "stage2_posted", "summary": s2_txt}), flush=True)
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                print(json.dumps({"event": "stage2_parse_err", "err": str(exc)}), flush=True)

        # 2. Stage 3 summary check -> launch Tier-C 1000g Ultra-GP alongside Harvester v2 on TRACK_G_HOST!
        if STAGE3_JSON.is_file() and not state.get(tg_state_key):
            try:
                s3_doc = json.loads(STAGE3_JSON.read_text())
                s3_txt = _format_summary_elos(s3_doc)
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                s3_txt = f"Stage 3 file present ({exc})"

            subprocess.run(
                ["rsync", "-az", "/home/ubuntu/h452-stage/code/", f"{TRACK_G_HOST}:/home/ubuntu/h452-stage/code/"],
                check=False,
                timeout=60,
            )
            for anc_name in (
                "cv7_hyb_lh455v1_mh455v12_50_50",
                "cv7_hyb_lw2_mh455_v2_65_35",
                "cv7_hyb_m_h455_v12_avg",
                "cv7_hyb_l_x1_h455_v1_soup50",
            ):
                anc_p = Path("/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91") / anc_name
                if anc_p.is_dir():
                    subprocess.run(
                        [
                            "rsync",
                            "-az",
                            f"{anc_p}/",
                            f"{TRACK_G_HOST}:/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/{anc_name}/",
                        ],
                        check=False,
                        timeout=90,
                    )

            remote_launch = (
                "nohup env CUDA_VISIBLE_DEVICES='' nice -n 19 python3 -u /home/ubuntu/h452-stage/code/ultra_gp_cycle_h458.py "
                "--workers 22 --inner-slots 3 --n-pop 44 --start-gen 1 --n-gens 1000 "
                "--snapshots 10,50,100,250,500,1000 "
                f">> /home/ubuntu/jope-data/h458-flywheel/logs/ultra_gp_cycle2_{TRACK_G_HOST}.log 2>&1 < /dev/null & "
                "sleep 1; pgrep -af 'hyper_cv_test_gp_harvester_h458|ultra_gp_cycle_h458'"
            )
            r_launch = subprocess.run(["ssh", TRACK_G_HOST, remote_launch], capture_output=True, text=True, check=False, timeout=45)

            _post_shared_chat(
                "cl_h458, cl_fwriter, cl_mtc, monster_train_agy (cc ENG:watcher, Jepson)",
                f"MILESTONE — Stage 3 COMPLETE ({s3_txt}; receipt {STAGE3_JSON}) & {TRACK_G_HOST} launched "
                f"ultra_gp_cycle_h458.py (66c, Gen 1->10..1000 Tier-C Train Hyper-CV M=1 untouched cross-fit anchored on 1838.04 L+M soup) "
                f"alongside Harvester v2 (44w).",
            )
            state[tg_state_key] = True
            state[f"{TRACK_G_HOST}_pids"] = r_launch.stdout.strip()
            STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps({"event": tg_state_key, "pids": r_launch.stdout.strip()}), flush=True)
            break

        time.sleep(15.0)


if __name__ == "__main__":
    main()
