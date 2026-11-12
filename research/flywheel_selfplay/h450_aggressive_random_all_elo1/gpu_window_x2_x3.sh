#!/bin/bash
# Bounded 2-GPU window manager for xbabe2 (RTX 3090) and xbabe3 (RTX 3090) ONLY.
# STRICT RULE: NEVER touches xbabe1 (RTX 4090 running external ZImagePipeline) or xbabe0!
#   gpu_window_x2_x3.sh status
#   gpu_window_x2_x3.sh acquire   -> saves live @cl_apex --resume jobs on xbabe2 & xbabe3, waits for 0 .tmp writes,
#                                    posts window to AGENT_CHAT.md under flock, cleanly SIGTERMs, verifies <1000 MiB VRAM.
#   gpu_window_x2_x3.sh release   -> relaunches saved @cl_apex --resume jobs with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
#                                    verifies active CUDA memory, AND waits until BOTH w2 (xbabe2) and w0 (xbabe3) append
#                                    a fresh `{"sync": ..., "loaded": true}` entry to `log.jsonl` before posting `resumed`!
set -euo pipefail
CMD=${1:?status|acquire|release}
EV=/home/ubuntu/jope_plan/evidence/H450
SAVE_JSON=$EV/logs/cl_apex_saved_jobs_x2_x3.json
REPO=/home/ubuntu/JopeDime
mkdir -p "$EV/logs"

run_host() {
  local h=$1; shift
  if [ "$h" = "xbabe2" ]; then
    bash -c "$*"
  else
    ssh -o BatchMode=yes "$h" "$*"
  fi
}

case "$CMD" in
status)
  for h in xbabe2 xbabe3; do
    echo "=== $h ==="
    run_host "$h" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true'
  done
  ;;

acquire)
  [ ! -e "$SAVE_JSON" ] || { echo "ERROR: $SAVE_JSON already exists; run release first or inspect" >&2; exit 3; }

  python3 - "$SAVE_JSON" <<'PY'
import json, subprocess, sys, time
out_path = sys.argv[1]

def sh(host, cmd):
    if host == "xbabe2":
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, check=True).stdout

saved = []
for host in ("xbabe2", "xbabe3"):
    probe = sh(host, r"""python3 - <<'INNER'
import glob, json, os
jobs = []
for pid_d in sorted(glob.glob('/proc/[0-9]*')):
    try:
        cmdline = open(pid_d + '/cmdline', 'rb').read().decode('utf-8', 'replace').split('\0')
        if cmdline and cmdline[-1] == '':
            cmdline = cmdline[:-1]
        if len(cmdline) >= 3 and 'icl_train.py' in ' '.join(cmdline) and '--resume' in cmdline:
            if cmdline[0].endswith('flock'):
                continue
            pid = int(os.path.basename(pid_d))
            stat_s = open(pid_d + '/stat').read()
            ppid = int(stat_s.rsplit(')', 1)[1].split()[1])
            p_cmd = []
            if os.path.exists(f'/proc/{ppid}/cmdline'):
                p_cmd = open(f'/proc/{ppid}/cmdline', 'rb').read().decode('utf-8', 'replace').split('\0')
                if p_cmd and p_cmd[-1] == '':
                    p_cmd = p_cmd[:-1]
            cwd = os.path.realpath(pid_d + '/cwd')
            out_dir = cmdline[cmdline.index('--out') + 1] if '--out' in cmdline else ''
            ck_last = os.path.join(out_dir, 'ckpt_last.pt') if out_dir else ''
            log_p = os.path.join(out_dir, 'log.jsonl') if out_dir else ''
            log_lines = sum(1 for _ in open(log_p)) if (log_p and os.path.exists(log_p)) else 0
            ck_mtime = os.path.getmtime(ck_last) if ck_last and os.path.exists(ck_last) else None
            ck_size = os.path.getsize(ck_last) if ck_last and os.path.exists(ck_last) else None
            launch_argv = p_cmd if (p_cmd and 'flock' in p_cmd[0]) else cmdline
            jobs.append({
                'pid': pid,
                'ppid': ppid if (p_cmd and 'flock' in p_cmd[0]) else None,
                'cwd': cwd,
                'launch_argv': launch_argv,
                'python_argv': cmdline,
                'out_dir': out_dir,
                'ckpt_last': ck_last,
                'ckpt_mtime': ck_mtime,
                'ckpt_size': ck_size,
                'log_lines_before': log_lines,
            })
    except Exception:
        pass
print(json.dumps(jobs))
INNER""")
    for j in json.loads(probe.strip()):
        j["host"] = host
        saved.append(j)

with open(out_path, "w") as f:
    json.dump({"saved_utc": time.strftime("%FT%TZ", time.gmtime()), "jobs": saved}, f, indent=2)
print(f"Saved {len(saved)} live @cl_apex --resume GPU job(s) on xbabe2/xbabe3 to {out_path}")
for j in saved:
    print(" ", j["host"], "pid=", j["pid"], "out=", j["out_dir"], "log_lines=", j["log_lines_before"])
PY

  MSG="$(date -u +%FT%TZ) agy_apex_w10 -> cl_apex, cl_fwriter: Acquiring owner-approved ~38m 2x RTX 3090 window on xbabe2 and xbabe3 ONLY for Wave H450 (xbabe1 untouched); cleanly pausing ICL_L_g1_w2 (xbabe2) and ICL_L_g1_w0 (xbabe3) at 0-.tmp boundary and will auto-resume both and verify sync loaded=true in both log.jsonl files before reporting resumed."
  (
    cd "$REPO"
    flock -x 9
    printf '\n%s\n' "$MSG" >> AGENT_CHAT.md
  ) 9>"$REPO/AGENT_CHAT.md.lock"

  python3 - "$SAVE_JSON" <<'PY'
import json, subprocess, sys
doc = json.load(open(sys.argv[1]))

def sh(host, cmd):
    if host == "xbabe2":
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, check=True).stdout

for j in doc["jobs"]:
    h = j["host"]
    od = j["out_dir"]
    pid = j["pid"]
    sh(h, f"for _ in $(seq 1 45); do ls {od}/*.tmp /home/ubuntu/jope-data/apex-icl/sync/L_g1/*.tmp >/dev/null 2>&1 || break; sleep 1; done")
    sh(h, f"kill -TERM {pid} 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 {pid} 2>/dev/null || break; sleep 1; done")
    print(f"Stopped {h} pid={pid} ({od})")

for h in ("xbabe2", "xbabe3"):
    out = sh(h, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
    used_mib = int(out.splitlines()[0].strip())
    print(f"{h}: GPU memory.used = {used_mib} MiB")
    if used_mib > 1000:
        raise SystemExit(f"ERROR: {h} still has {used_mib} MiB VRAM in use!")
print("X2_AND_X3_GPUS_ACQUIRED_OK")
PY
  ;;

release)
  [ -f "$SAVE_JSON" ] || { echo "ERROR: missing $SAVE_JSON" >&2; exit 3; }
  RESUME_SUMMARY=$(python3 - "$SAVE_JSON" <<'PY'
import json, shlex, subprocess, sys, time
save_path = sys.argv[1]
doc = json.load(open(save_path))

def sh(host, cmd):
    if host == "xbabe2":
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, check=True).stdout

pre_lines = {}
for j in doc["jobs"]:
    h = j["host"]
    log_p = j["out_dir"] + "/log.jsonl"
    cnt = int(sh(h, f"wc -l < {shlex.quote(log_p)} 2>/dev/null || echo 0").strip())
    pre_lines[h] = cnt

for j in doc["jobs"]:
    h = j["host"]
    cwd = j["cwd"]
    argv_q = " ".join(shlex.quote(x) for x in j["launch_argv"])
    log_p = j["out_dir"] + "/agy_auto_resume.log"
    cmd = (
        f"( cd {shlex.quote(cwd)} && exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"setsid nohup {argv_q} ) >> {shlex.quote(log_p)} 2>&1 < /dev/null & echo $!"
    )
    new_pid = sh(h, cmd).strip()
    sys.stderr.write(f"Relaunched {h} ({j['out_dir']}) -> wrapper pid {new_pid}\n")

time.sleep(12)
for j in doc["jobs"]:
    h = j["host"]
    procs = sh(h, "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader").strip()
    sys.stderr.write(f"{h} CUDA procs @ +12s: {procs}\n")
    if not procs:
        raise SystemExit(f"ERROR: {h} GPU job did not attach to CUDA! Check {j['out_dir']}/agy_auto_resume.log")

# Wait up to 720s (--sync-s 300 + --sync-wait 240) until BOTH w2 and w0 emit a post-resume `sync` line with `loaded: true`!
sync_ok = {}
t_wait0 = time.time()
while time.time() - t_wait0 < 720.0 and len(sync_ok) < len(doc["jobs"]):
    for j in doc["jobs"]:
        h = j["host"]
        if h in sync_ok:
            continue
        log_p = j["out_dir"] + "/log.jsonl"
        start_ln = pre_lines[h] + 1
        tail_lines = sh(h, f"tail -n +{start_ln} {shlex.quote(log_p)} 2>/dev/null || true").splitlines()
        for ln in tail_lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
                if "sync" in rec and rec.get("loaded") is True:
                    procs_now = sh(h, "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").strip()
                    sync_ok[h] = {"sync": rec["sync"], "step": rec.get("step"), "wait_s": rec.get("wait_s"), "cuda": procs_now}
                    sys.stderr.write(f"Verified {h} sync loaded=true: {sync_ok[h]}\n")
                    break
            except Exception:
                pass
    if len(sync_ok) < len(doc["jobs"]):
        time.sleep(6.0)

if len(sync_ok) < len(doc["jobs"]):
    raise SystemExit(f"ERROR: timed out waiting for sync loaded=true on all jobs; got {sync_ok}")

print(json.dumps(sync_ok, sort_keys=True))
PY
)
  mv "$SAVE_JSON" "$EV/logs/cl_apex_saved_jobs_x2_x3.resumed.$(date -u +%Y%m%dT%H%M%SZ).json"
  MSG="$(date -u +%FT%TZ) agy_apex_w10 -> cl_apex, cl_fwriter: Completed Wave H450 2-GPU window; resumed ICL_L_g1_w2 (xbabe2) and ICL_L_g1_w0 (xbabe3) from ckpt_last.pt AND verified post-resume sync with loaded=true in both log.jsonl files: $RESUME_SUMMARY"
  (
    cd "$REPO"
    flock -x 9
    printf '\n%s\n' "$MSG" >> AGENT_CHAT.md
  ) 9>"$REPO/AGENT_CHAT.md.lock"
  echo "X2_AND_X3_RESUMED_AND_SYNC_VERIFIED: $RESUME_SUMMARY"
  ;;

*)
  echo "Usage: $0 {status|acquire|release}" >&2
  exit 2
  ;;
esac
