#!/bin/bash
# H357 orchestration (PREREG §3), run on xbabe2. Every step refuses unless the PREREG in the staged code root is
# byte-identical to the H357 PREREG on origin/v1/next (published before any P201c fit).
#   H357_CODE=/home/ubuntu/h357-stage/code-TAG H357_GPC=/home/ubuntu/h357-stage/gpc-TAG run_c_h357.sh STEP
#   panel   P201c list (xbabe2 H335 sandbox); must equal the PREREG cids sha256
#   encode  P201c SEARCH views (xbabe2 H335 sandbox, H335_VIEWS_RW=1; DOPE sha checked before/after)
#   stage   -> xbabe0: GP code root + marker, the P201c panel and views (sha256-checked on arrival); the frozen H317
#           code, superset worker and screener inputs already on xbabe0 are verified in place
#   launch  xbabe0: systemd-run --user h357-C-gp (docker sandbox, 12 AGG shards x 8 workers, 96 cores)
#   pull    xbabe0 run dir -> xbabe2 (same path); every file's sha256 compared with xbabe0's; 194 records required
#   prep    phase_c_h357.py prepare
#   lib     systemd-run --user h357-lib: the 4 classical learners, SEARCH OOF + SEARCH->AUDIT fits (--audit fit)
#   auditreq, stagex, auditx, pullx  Amendment 1 (PREREG §3.1): AUDIT predictions for GP members X_CL uses but X0
#           never audited (request on xbabe2, SEARCH-only; frozen engine on xbabe0, unit h357-C-auditx; pulled back)
#   score   phase_c_h357.py arms + decide
set -euo pipefail
STEP=${1:?step}
CODE=${H357_CODE:?H357_CODE}
GPC=${H357_GPC:?H357_GPC}
REPO=/home/ubuntu/JopeDime
PREREG_PATH=research/flywheel_selfplay/h357_xstack_cl_confirm/PREREG.md
CIDS_SHA=36fbf6a92204487a31f2533905a57393e9c317fa7484ab7cbde5026e34468f6d
DEADLINE=2026-11-05T20:00:00Z
DATA=/home/ubuntu/jope-data/h335-cl
RUN=$DATA/runs/H357_C_xbabe0
H357D=/home/ubuntu/jope-data/h357-xc
EV=/home/ubuntu/jope_plan/evidence/H357
HOST=xbabe0
SSH=(ssh -o BatchMode=yes "$HOST")
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
(cd "$CODE" && sha256sum --quiet -c CODE.sha256)
(cd "$GPC/code" && sha256sum --quiet -c ../CODE.sha256)
git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
  +refs/heads/v1/next:refs/remotes/origin/v1/next
cmp -s "$CODE/PREREG.md" <(git -C "$REPO" show "origin/v1/next:$PREREG_PATH") \
  || { echo "staged PREREG.md is not the H357 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
PREREG_COMMIT=$(git -C "$REPO" log -1 --format=%H origin/v1/next -- "$PREREG_PATH")
case $STEP in encode|stage|launch|lib|stagex|auditx)
  [ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; } ;;
esac
mkdir -p "$EV/logs"
case $STEP in
  panel|encode)
    cd /home/ubuntu
    if [ "$STEP" = panel ]; then
      H335_CODE=$GPC/code nice -n 19 /home/ubuntu/h335-stage/code/sandbox_h335.sh python3 -B gp_c_h357.py panel
      [ "$(jq -r .cids_sha256 "$DATA/h357/P201c.json")" = "$CIDS_SHA" ] || { echo "P201c != PREREG panel" >&2; exit 5; }
      chmod a-w "$DATA/h357/P201c.json"
    else
      H335_VIEWS_RW=1 H335_CODE=$GPC/code nice -n 19 /home/ubuntu/h335-stage/code/sandbox_h335.sh \
        python3 -B gp_c_h357.py encode --threads 8
    fi ;;
  stage)
    if "${SSH[@]}" "systemctl --user list-units --state=active --no-legend 'h357-*' | grep -q ."; then
      echo "refused: an h357-* unit is active on $HOST" >&2; exit 4
    fi
    MAN=$EV/staging_$(date -u +%Y%m%dT%H%M%SZ)
    [ -e "$GPC/PREREG_ON_V1NEXT" ] || printf '%s %s\n' "$PREREG_COMMIT" "$(sha256sum "$CODE/PREREG.md" | cut -d' ' -f1)" \
      > "$GPC/PREREG_ON_V1NEXT"
    "${SSH[@]}" "mkdir -p $(dirname "$GPC") $DATA/h357 $DATA/staging"
    grep -oE '"[a-z0-9_]+\.py": "[0-9a-f]{64}"' "$GPC/code/h335_core.py" | sed -E 's/"([^"]+)": "([0-9a-f]+)"/\2  \1/' \
      > "$MAN.h317pins.sha256"
    [ "$(wc -l < "$MAN.h317pins.sha256")" = 18 ] || { echo "pin list != 18" >&2; exit 5; }
    scp -q "$MAN.h317pins.sha256" "$HOST:$DATA/staging/h357_h317pins.sha256"
    "${SSH[@]}" "cd /home/ubuntu/h317-stage/code && sha256sum --quiet -c $DATA/staging/h357_h317pins.sha256"
    "${SSH[@]}" "[ ! -e $GPC/code ]" || { echo "$HOST:$GPC/code exists (create-once)" >&2; exit 3; }
    rsync -a "$GPC/" "$HOST:$GPC/"
    "${SSH[@]}" "cd $GPC/code && sha256sum --quiet -c ../CODE.sha256"
    "${SSH[@]}" "sha256sum $DATA/bin/bootstrap.s1main.superset-2a848c8f8085" | grep -q '^2a848c8f8085a392'
    scp -q "$DATA/neural_manifest.sha256" "$HOST:$DATA/staging/h357_neural_manifest.sha256"
    "${SSH[@]}" "cd $DATA/neural && sha256sum --quiet -c $DATA/staging/h357_neural_manifest.sha256"
    jq -r '.cids[]' "$DATA/h357/P201c.json" > "$MAN.cids"
    [ "$(wc -l < "$MAN.cids")" = 194 ] || { echo "P201c list != 194" >&2; exit 5; }
    sed 's/$/.npz/' "$MAN.cids" > "$MAN.views.files"
    (cd "$DATA/views-search" && xargs -a "$MAN.views.files" sha256sum > "$MAN.views.sha256")
    (cd "$DATA/h357" && sha256sum P201c.json > "$MAN.panel.sha256")
    rsync -a "$DATA/h357/P201c.json" "$HOST:$DATA/h357/"
    rsync -a --files-from="$MAN.views.files" "$DATA/views-search/" "$HOST:$DATA/views-search/"
    scp -q "$MAN.views.sha256" "$HOST:$DATA/staging/h357_views.sha256"
    scp -q "$MAN.panel.sha256" "$HOST:$DATA/staging/h357_panel.sha256"
    "${SSH[@]}" "cd $DATA/views-search && sha256sum --quiet -c $DATA/staging/h357_views.sha256 && cd $DATA/h357 \
      && sha256sum --quiet -c $DATA/staging/h357_panel.sha256"
    echo "stage OK -> $HOST (manifest prefix $MAN; PREREG $PREREG_COMMIT)" ;;
  launch)
    "${SSH[@]}" "systemd-run --user --unit h357-C-gp -p MemoryMax=100G -p MemorySwapMax=0 \
      -p StandardOutput=append:$DATA/h357/h357-C-gp.log -p StandardError=append:$DATA/h357/h357-C-gp.log \
      $GPC/code/run_gp_c_xbabe0.sh" ;;
  pull)
    if "${SSH[@]}" "systemctl --user is-active -q h357-C-gp"; then echo "h357-C-gp still active" >&2; exit 4; fi
    mkdir -p "$RUN"
    rsync -a "$HOST:$RUN/" "$RUN/"
    rsync -a "$HOST:$DATA/h357/h357-C-gp.log" "$EV/logs/h357-C-gp.xbabe0.log"
    "${SSH[@]}" "cd $RUN && find . -type f | sort | xargs -d '\n' sha256sum" > "$EV/logs/C_run.xbabe0.sha256"
    (cd "$RUN" && sha256sum --quiet -c "$EV/logs/C_run.xbabe0.sha256")
    n=$(find "$RUN/stages/C_AGG" -name '*__357101.jsonl' | wc -l)
    echo "pull OK: $(wc -l < "$EV/logs/C_run.xbabe0.sha256") files identical; $n/194 stage records"
    [ "$n" = 194 ] || { echo "INCOMPLETE: $n/194 stage records" >&2; exit 7; } ;;
  prep)
    cd "$CODE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3 -B phase_c_h357.py prepare ;;
  auditreq)  # Amendment 1 (PREREG §3.1): which GP members does X_CL use that X0 never audited (SEARCH-only)?
    cd "$CODE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 python3 -B phase_c_h357.py audit-request ;;
  stagex)  # Amendment 1: a new GP code root (create-once on xbabe0) + the request, sha256-checked on arrival
    if "${SSH[@]}" "systemctl --user list-units --state=active --no-legend 'h357-*' | grep -q ."; then
      echo "refused: an h357-* unit is active on $HOST" >&2; exit 4
    fi
    [ -e "$GPC/PREREG_ON_V1NEXT" ] || printf '%s %s\n' "$PREREG_COMMIT" "$(sha256sum "$CODE/PREREG.md" | cut -d' ' -f1)" \
      > "$GPC/PREREG_ON_V1NEXT"
    "${SSH[@]}" "[ ! -e $GPC/code ]" || { echo "$HOST:$GPC/code exists (create-once)" >&2; exit 3; }
    "${SSH[@]}" "mkdir -p $(dirname "$GPC")"
    rsync -a "$GPC/" "$HOST:$GPC/"
    "${SSH[@]}" "cd $GPC/code && sha256sum --quiet -c ../CODE.sha256"
    REQ=$H357D/audit_extra_request.json
    "${SSH[@]}" "[ ! -e $DATA/h357/audit_extra_request.json ]" || { echo "request exists on $HOST" >&2; exit 3; }
    rsync -a "$REQ" "$HOST:$DATA/h357/audit_extra_request.json"
    "${SSH[@]}" "sha256sum $DATA/h357/audit_extra_request.json" | grep -q "^$(sha256sum "$REQ" | cut -d' ' -f1) "
    echo "stagex OK -> $HOST" ;;
  auditx)
    "${SSH[@]}" "systemd-run --user --unit h357-C-auditx -p MemoryMax=100G -p MemorySwapMax=0 \
      -p StandardOutput=append:$DATA/h357/h357-C-auditx.log -p StandardError=append:$DATA/h357/h357-C-auditx.log \
      $GPC/code/run_auditx_xbabe0.sh" ;;
  pullx)
    if "${SSH[@]}" "systemctl --user is-active -q h357-C-auditx"; then echo "h357-C-auditx still active" >&2; exit 4; fi
    RX=$DATA/runs/H357_C_xbabe0_auditx
    mkdir -p "$RX"
    rsync -a "$HOST:$RX/" "$RX/"
    rsync -a "$HOST:$DATA/h357/h357-C-auditx.log" "$EV/logs/h357-C-auditx.xbabe0.log"
    "${SSH[@]}" "cd $RX && find . -type f | sort | xargs -d '\n' sha256sum" > "$EV/logs/C_auditx.xbabe0.sha256"
    (cd "$RX" && sha256sum --quiet -c "$EV/logs/C_auditx.xbabe0.sha256")
    n=$(wc -l < "$RX/done.jsonl"); want=$(jq '.contexts | length' "$H357D/audit_extra_request.json")
    echo "pullx OK: $(wc -l < "$EV/logs/C_auditx.xbabe0.sha256") files identical; $n/$want contexts done"
    [ "$n" = "$want" ] || { echo "INCOMPLETE: $n/$want audit-extra contexts" >&2; exit 7; } ;;
  lib)
    cat > "$EV/run_lib.sh" <<EOS
#!/bin/bash
set -euo pipefail
cd "$CODE" && sha256sum --quiet -c CODE.sha256
echo "CODE.sha256 \$(sha256sum CODE.sha256 | cut -d' ' -f1) start \$(date -u +%FT%TZ)"
exec taskset -c 38-53,102-117 "$CODE/sandbox_h354.sh" "$H357D" python3 -B "$CODE/lib_oof_h354.py" \\
  --inputs "$H357D/inputs" --out "$H357D/lib" --jobs 8 --audit fit \\
  --learners catboost.v1,lightgbm.v1,xgboost.v1,random_forest.v1
EOS
    chmod +x "$EV/run_lib.sh"
    systemd-run --user --unit=h357-lib --collect -p MemoryMax=48G -p MemorySwapMax=0 -p Nice=19 -p CPUQuota=3200% \
      --setenv=PYTHONDONTWRITEBYTECODE=1 -p StandardOutput=append:"$EV/logs/lib.log" \
      -p StandardError=append:"$EV/logs/lib.err" "$EV/run_lib.sh" ;;
  score)
    cd "$CODE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 taskset -c 38-53,102-117 \
      python3 -B phase_c_h357.py arms --jobs 16 \
      && PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B phase_c_h357.py decide ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
