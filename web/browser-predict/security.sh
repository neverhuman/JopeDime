#!/usr/bin/env bash
# The standalone browser lock is absent from the root workspace dependency graph.
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$(cargo deny --version)" = 'cargo-deny 0.20.2'
target="$PWD/target"
if printenv CARGO_TARGET_DIR >/dev/null; then target=$(printenv CARGO_TARGET_DIR); fi
evidence="$target/browser-predict/security"
mkdir -p "$evidence"
cargo deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features list --layout crate > "$evidence/graph.log"
missing=""
for package in jope-browser-predict feat-math jope-runtime; do
  grep -qE "^$package@" "$evidence/graph.log" || missing="$missing $package"
done
if [ -n "$missing" ]; then
  echo "standalone browser dependency graph is incomplete; missing:$missing" >&2
  exit 1
fi
cargo deny --manifest-path web/browser-predict/Cargo.toml --config deny.toml --locked --all-features check advisories bans licenses sources > "$evidence/check.log" 2>&1 || {
  status=$?
  cat "$evidence/check.log" >&2
  exit "$status"
}
python3 - "$evidence" <<'PY'
import hashlib,json,pathlib,subprocess,sys
out=pathlib.Path(sys.argv[1])
files=['deny.toml','web/browser-predict/Cargo.toml','web/browser-predict/Cargo.lock','web/browser-predict/security.sh','web/browser-predict/check.sh']
receipt={'schema':'jope.browser-dependency-scan/v1','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dirty_source':subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).splitlines(),'tool':subprocess.check_output(['cargo','deny','--version'],text=True).strip(),'all_features':True,'locked':True,'checks':['advisories','bans','licenses','sources'],'exit':0,'inputs':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in files},'evidence':{p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ['graph.log','check.log']}}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
echo 'Standalone browser lock: advisory/ban/license/source checks passed; three graph canaries present'
