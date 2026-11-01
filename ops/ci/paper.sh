#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
evidence="target/ci-evidence/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/paper"
build=target/innovationzero-paper
mkdir -p "$build"
if ! git cat-file -e "$JOPE_SOURCE_SHA^{commit}" 2>/dev/null; then
  git fetch --no-tags --depth=1 origin "$JOPE_SOURCE_SHA"
fi
python3 - <<'PY'
import json, os, pathlib, subprocess
git = lambda *args: subprocess.check_output(['git', *args], text=True).strip()
source = os.environ['JOPE_SOURCE_SHA']; checkout = git('rev-parse', 'HEAD')
assert checkout == os.environ['GITHUB_SHA'], 'native event checkout required'
source_tree = git('rev-parse', source + '^{tree}')
checkout_tree = git('rev-parse', 'HEAD^{tree}')
assert source_tree == checkout_tree, 'PR merge tree differs from reviewed source; update branch first'
pathlib.Path('target/innovationzero-paper/source-identity.json').write_text(json.dumps({
    'schema':'jope.publication-source/v1', 'source_sha':source, 'checkout_sha':checkout,
    'event_sha':os.environ['GITHUB_SHA'], 'source_tree':source_tree, 'checkout_tree':checkout_tree,
    'run_id':os.environ['GITHUB_RUN_ID'], 'attempt':int(os.environ['GITHUB_RUN_ATTEMPT']),
    'event':os.environ['GITHUB_EVENT_NAME']}, indent=2) + '\n')
print('Publication source/checkout trees match:', source_tree)
PY
export JOPE_PAPER_NATIVE_EVIDENCE="$build/native/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT"
bash deploy/ci/publication-tools.sh python3 deploy/ci/receipt.py --lane paper --output "$evidence" --backend none \
  --evidence "$build/InnovationZero.pdf" \
  --evidence "$build/InnovationZero-Supplement.pdf" \
  --evidence "$build/InnovationZero-Archive.pdf" \
  --evidence "$build/InnovationZero.log" \
  --evidence "$build/InnovationZero-Supplement.log" \
  --evidence "$build/InnovationZero-Archive.log" \
  --evidence "$build/build-environment.txt" \
  --evidence "$build/source-identity.json" \
  --evidence "$JOPE_PAPER_NATIVE_EVIDENCE/dependencies.json" \
  --evidence "$JOPE_PAPER_NATIVE_EVIDENCE/format.log" \
  -- make -C docs/paper/innovationzero check self-test pdf
