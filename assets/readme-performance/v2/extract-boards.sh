#!/usr/bin/env bash
# Write boards.json: the README's rows for JopeDime's own models on the dev91-audit-v1 reference board.
# Reads the board's committed ladder.json from Git objects at a pinned research-branch commit; no fits, no
# scores and no dataset reads. The board lives on the research branch v1/next, which CI does not check out, so
# render.sh reads only the extracted file (pinned by sha). Re-run this after the board changes, review the
# prose in render.sh against the new rows, and then update both pins.
#   usage: extract-boards.sh [--check|--write]   (needs the pinned commit's objects: git fetch origin v1/next)
set -euo pipefail
root=${JOPE_PERFORMANCE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'usage: extract-boards.sh [--check|--write]' >&2; exit 2 ;; esac
cd "$root"
out=assets/readme-performance/v2/boards.json
commit=7f7ff32833de92327c71ed6abcd683d0decf4660
path=research/elo/reference-board/boards/dev91-audit-v1/ladder.json
fail() { echo "extract-boards: $*" >&2; exit 1; }
git cat-file -e "$commit^{commit}" 2>/dev/null || fail "commit $commit is not present; fetch origin v1/next first"
blob=$(git rev-parse "$commit:$path")
sha=$(git cat-file blob "$blob" | sha256sum); sha=${sha%% *}
ours='["h364.XSTACK_2L","h335.H335_AGG"]'
frontier='["h335.Opus5p5_Extra","h335.Gemini4_argon_high","h335.GPT6p1_Sol_Max","h414.kimik3p1_max","h335.GPT6_Astra_Max","h412.grok4p7_xhigh"]'
json=$(git cat-file blob "$blob" | jq -S --arg commit "$commit" --arg path "$path" --arg blob "$blob" --arg sha "$sha" \
  --argjson ours "$ours" --argjson frontier "$frontier" '
  def ranked: [.elo.elo | to_entries | sort_by(-.value.elo) | to_entries[] |
    {rank: (.key + 1), id: .value.key, elo: .value.value.elo, ci95: .value.value.ci95}];
  ranked as $r | ($r | map({key: .id, value: .}) | from_entries) as $by |
  if (.elo.anchor != "h335.H257_REPLICA") or (.n_contexts != 91) or (.n_families != 19) or (.incomplete | length != 0)
     or ($r | length) != (.entrants | length) or any(($ours + $frontier)[]; $by[.] == null)
  then error("dev91-audit-v1 ladder shape changed") else . end |
  {schema: "jope.readme-reference-boards/v1",
   source: {commit: $commit, path: $path, git_blob: $blob, sha256: $sha},
   board: .board, anchor: .elo.anchor, contexts: .n_contexts, families: .n_families, entrants: ($r | length),
   leader: $r[0],
   above_ours: [$r[] | select(.rank < $by[$ours[0]].rank) | .id],
   ours: [$ours[] | $by[.]],
   frontier: [$frontier[] | $by[.]] | sort_by(.rank)}')
if [[ $mode == --check ]]; then
  [[ -f $out ]] || fail "$out is missing; run with --write"
  [[ $(cat "$out") == "$json" ]] || fail "$out differs from the board at $commit; run with --write and review render.sh"
  echo "$out matches $path at ${commit:0:10}."
else
  printf '%s\n' "$json" > "$out"
  echo "wrote $out from $path at ${commit:0:10} (blob $blob)."
fi
