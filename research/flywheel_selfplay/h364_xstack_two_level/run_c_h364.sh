#!/bin/bash
# H364 orchestration (PREREG §3), run on xbabe2. Every step refuses unless the PREREG in the staged code root is
# byte-identical to the H364 PREREG on origin/v1/next (part A: published before any P201d view or GP run). The steps that
# read P201d outputs (pull onward) also refuse unless the staged DESIGN.json (part B: the frozen second-level constants,
# chosen on P201b + P201c only) is byte-identical to the one on origin/v1/next.
#   H364_CODE=/home/ubuntu/h364-stage/code-TAG H364_GPC=/home/ubuntu/h364-stage/gpc-TAG run_c_h364.sh STEP
#   panel   P201d list (xbabe2 H335 sandbox); must equal the PREREG cids sha256
#   encode  P201d SEARCH views (xbabe2 H335 sandbox, H335_VIEWS_RW=1; DOPE sha checked before/after)
#   stage   -> xbabe0: GP code root + marker, the P201d panel and views (sha256-checked on arrival); the frozen H317
#           code, superset worker and screener inputs already on xbabe0 are verified in place
#   launch  xbabe0: systemd-run --user h364-C-gp (docker sandbox, 12 AGG shards x 8 workers, 96 cores)
#   ---- part B gate from here on ----
#   pull    xbabe0 run dir -> xbabe2 (same path); every file's sha256 compared with xbabe0's; 190 records required
#   prep    phase_c_h364.py prepare (library inputs: SEARCH envelope + train-only AUDIT envelope per context)
#   lib     systemd-run --user h364-lib: the 4 classical learners, SEARCH OOF + SEARCH->AUDIT fits (--audit fit)
#   score   phase_c_h364.py score (X0 harness check, X0 / 2L / 2L_noguard / library alone) then decide (verdict.json)
set -euo pipefail
STEP=${1:?step}
CODE=${H364_CODE:?H364_CODE}
GPC=${H364_GPC:?H364_GPC}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h364_xstack_two_level
CIDS_SHA=9a1ace6cfd9abdf6c436d980b61d98b59825e56ea3719cd67911a0f2aa25de98
N=190
DEADLINE=2026-11-05T20:00:00Z
DATA=/home/ubuntu/jope-data/h335-cl
RUN=$DATA/runs/H364_C_xbabe0
H364D=/home/ubuntu/jope-data/h364-2l
EV=/home/ubuntu/jope_plan/evidence/H364
HOST=xbabe0
SSH=(ssh -o BatchMode=yes "$HOST")
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
(cd "$CODE" && sha256sum --quiet -c CODE.sha256)
(cd "$GPC/code" && sha256sum --quiet -c ../CODE.sha256)
git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
  +refs/heads/v1/next:refs/remotes/origin/v1/next
cmp -s "$CODE/PREREG.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
  || { echo "staged PREREG.md is not the H364 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
PREREG_COMMIT=$(git -C "$REPO" log -1 --format=%H origin/v1/next -- "$WDIR/PREREG.md")
case $STEP in pull|prep|lib|score)
  { [ -s "$CODE/DESIGN.json" ] && cmp -s "$CODE/DESIGN.json" <(git -C "$REPO" show "origin/v1/next:$WDIR/DESIGN.json"); } \
    || { echo "part B (DESIGN.json) is not staged and published on origin/v1/next: no P201d output is read" >&2; exit 3; }
esac
case $STEP in encode|stage|launch|lib|score)
  [ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; } ;;
esac
mkdir -p "$EV/logs"
case $STEP in
  panel|encode)
    cd /home/ubuntu
    if [ "$STEP" = panel ]; then
      H335_CODE=$GPC/code nice -n 19 /home/ubuntu/h335-stage/code/sandbox_h335.sh python3 -B gp_c_h364.py panel
      [ "$(jq -r .cids_sha256 "$DATA/h364/P201d.json")" = "$CIDS_SHA" ] || { echo "P201d != PREREG panel" >&2; exit 5; }
      chmod a-w "$DATA/h364/P201d.json"
    else
      H335_VIEWS_RW=1 H335_CODE=$GPC/code nice -n 19 /home/ubuntu/h335-stage/code/sandbox_h335.sh \
        python3 -B gp_c_h364.py encode --threads 8
    fi ;;
  stage)
    if "${SSH[@]}" "systemctl --user list-units --state=active --no-legend 'h364-*' | grep -q ."; then
      echo "refused: an h364-* unit is active on $HOST" >&2; exit 4
    fi
    MAN=$EV/staging_$(date -u +%Y%m%dT%H%M%SZ)
    [ -e "$GPC/PREREG_ON_V1NEXT" ] || printf '%s %s\n' "$PREREG_COMMIT" "$(sha256sum "$CODE/PREREG.md" | cut -d' ' -f1)" \
      > "$GPC/PREREG_ON_V1NEXT"
    "${SSH[@]}" "mkdir -p $(dirname "$GPC") $DATA/h364 $DATA/staging"
    grep -oE '"[a-z0-9_]+\.py": "[0-9a-f]{64}"' "$GPC/code/h335_core.py" | sed -E 's/"([^"]+)": "([0-9a-f]+)"/\2  \1/' \
      > "$MAN.h317pins.sha256"
    [ "$(wc -l < "$MAN.h317pins.sha256")" = 18 ] || { echo "pin list != 18" >&2; exit 5; }
    scp -q "$MAN.h317pins.sha256" "$HOST:$DATA/staging/h364_h317pins.sha256"
    "${SSH[@]}" "cd /home/ubuntu/h317-stage/code && sha256sum --quiet -c $DATA/staging/h364_h317pins.sha256"
    "${SSH[@]}" "[ ! -e $GPC/code ]" || { echo "$HOST:$GPC/code exists (create-once)" >&2; exit 3; }
    rsync -a "$GPC/" "$HOST:$GPC/"
    "${SSH[@]}" "cd $GPC/code && sha256sum --quiet -c ../CODE.sha256"
    "${SSH[@]}" "sha256sum $DATA/bin/bootstrap.s1main.superset-2a848c8f8085" | grep -q '^2a848c8f8085a392'
    scp -q "$DATA/neural_manifest.sha256" "$HOST:$DATA/staging/h364_neural_manifest.sha256"
    "${SSH[@]}" "cd $DATA/neural && sha256sum --quiet -c $DATA/staging/h364_neural_manifest.sha256"
    jq -r '.cids[]' "$DATA/h364/P201d.json" > "$MAN.cids"
    [ "$(wc -l < "$MAN.cids")" = "$N" ] || { echo "P201d list != $N" >&2; exit 5; }
    sed 's/$/.npz/' "$MAN.cids" > "$MAN.views.files"
    (cd "$DATA/views-search" && xargs -a "$MAN.views.files" sha256sum > "$MAN.views.sha256")
    (cd "$DATA/h364" && sha256sum P201d.json > "$MAN.panel.sha256")
    rsync -a "$DATA/h364/P201d.json" "$HOST:$DATA/h364/"
    rsync -a --files-from="$MAN.views.files" "$DATA/views-search/" "$HOST:$DATA/views-search/"
    scp -q "$MAN.views.sha256" "$HOST:$DATA/staging/h364_views.sha256"
    scp -q "$MAN.panel.sha256" "$HOST:$DATA/staging/h364_panel.sha256"
    "${SSH[@]}" "cd $DATA/views-search && sha256sum --quiet -c $DATA/staging/h364_views.sha256 && cd $DATA/h364 \
      && sha256sum --quiet -c $DATA/staging/h364_panel.sha256"
    echo "stage OK -> $HOST (manifest prefix $MAN; PREREG $PREREG_COMMIT)" ;;
  launch)
    "${SSH[@]}" "systemd-run --user --unit h364-C-gp -p MemoryMax=100G -p MemorySwapMax=0 \
      -p StandardOutput=append:$DATA/h364/h364-C-gp.log -p StandardError=append:$DATA/h364/h364-C-gp.log \
      $GPC/code/run_gp_c_xbabe0.sh" ;;
  pull)
    if "${SSH[@]}" "systemctl --user is-active -q h364-C-gp"; then echo "h364-C-gp still active" >&2; exit 4; fi
    mkdir -p "$RUN"
    rsync -a "$HOST:$RUN/" "$RUN/"
    rsync -a "$HOST:$DATA/h364/h364-C-gp.log" "$EV/logs/h364-C-gp.xbabe0.log"
    "${SSH[@]}" "cd $RUN && find . -type f | sort | xargs -d '\n' sha256sum" > "$EV/logs/C_run.xbabe0.sha256"
    (cd "$RUN" && sha256sum --quiet -c "$EV/logs/C_run.xbabe0.sha256")
    n=$(find "$RUN/stages/C_AGG" -name '*__364101.jsonl' | wc -l)
    echo "pull OK: $(wc -l < "$EV/logs/C_run.xbabe0.sha256") files identical; $n/$N stage records"
    [ "$n" = "$N" ] || { echo "INCOMPLETE: $n/$N stage records" >&2; exit 7; } ;;
  prep)
    cd "$CODE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3 -B phase_c_h364.py prepare ;;
  lib)
    cat > "$EV/run_lib.sh" <<EOS
#!/bin/bash
set -euo pipefail
cd "$CODE" && sha256sum --quiet -c CODE.sha256
echo "CODE.sha256 \$(sha256sum CODE.sha256 | cut -d' ' -f1) start \$(date -u +%FT%TZ)"
exec taskset -c 38-53,102-117 "$CODE/sandbox_h354.sh" "$H364D" python3 -B "$CODE/lib_oof_h354.py" \\
  --inputs "$H364D/inputs" --out "$H364D/lib" --jobs 8 --audit fit \\
  --learners catboost.v1,lightgbm.v1,xgboost.v1,random_forest.v1
EOS
    chmod +x "$EV/run_lib.sh"
    systemd-run --user --unit=h364-lib --collect -p MemoryMax=48G -p MemorySwapMax=0 -p Nice=19 -p CPUQuota=3200% \
      --setenv=PYTHONDONTWRITEBYTECODE=1 -p StandardOutput=append:"$EV/logs/lib.log" \
      -p StandardError=append:"$EV/logs/lib.err" "$EV/run_lib.sh" ;;
  score)
    cd "$CODE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c 38-53,102-117 nice -n 19 python3 -B phase_c_h364.py score --jobs 16 \
      && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3 -B phase_c_h364.py decide ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
