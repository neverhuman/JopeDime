#!/usr/bin/env bash
# docs/paper/innovationzero/scripts/check.sh — the publication's source and output contract.
#
#   scripts/check.sh                    check sources (no TeX needed)
#   scripts/check.sh --prepare DIR      check sources, then write DIR/archive-index.tex
#   scripts/check.sh --pdf-dir DIR      check sources and the three built PDFs and their logs in DIR
#   scripts/check.sh --self-test        prove that every refusal below fires on a broken fixture
#
# Refusals, each a separate check:
#   archive    every artifact in evidence/artifact-manifest.json exists under artifacts/ as a regular
#              file with the declared byte count, SHA-256 and source commit; ids and paths unique
#   programs   the ten emitted programs match evidence/pilot-2026-10-22/programs.tsv
#   elo        evidence/elo-summary.json matches the archived RESULTS.json, and the panel is not
#              described as containing a qualified InnovationZero (K = 1) entrant
#   frontier   data/frontier-entrants.csv agrees with the archived Elo, cost and size receipts
#   derived    data/frontier-*.csv regenerate exactly from the archived match matrix
#   tex        every \input and \includegraphics resolves, every citation is in references.bib,
#              every section file is included, no placeholder or shell escape
#   vocabulary the manuscript does not use the repository's restricted vocabulary (deploy/ci/ip-scrub.sh)
#   pdf        no overfull box, unresolved reference or citation, missing glyph, unembedded or Type 3 font
#
# Bash, jq and coreutils only (ADR-0006: no new Python). PAPER_DIR overrides the paper directory,
# which the self-test uses to run these checks against disposable copies.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
P=${PAPER_DIR:-$here}
export LC_ALL=C

fail() { printf 'publication check (%s): %s\n' "$1" "$2" >&2; exit 1; }
build_root() { printf '%s' "${IZ_BUILD_DIR:-$(git -C "$here" rev-parse --show-toplevel)/target/innovationzero-paper}"; }
commas() { printf '%s' "$1" | sed -e ':a' -e 's/\B[0-9]\{3\}\>/,&/' -e 'ta'; }
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
size() { stat -c %s -- "$1"; }

# The restricted vocabulary of deploy/ci/ip-scrub.sh, applied to the manuscript. Proper names of
# other groups' systems (AlphaEvolve) and the word "permutation" are the only exemptions; citation
# keys are not prose and are removed before scanning.
VOCAB='\bGP\b|genome|genetic|evolut|evolv|\bpopulation|chromosome|mutation|crossover|lexicase|fitness|keijzer|koza|expression tree|parse tree|syntax tree|symbolic regression'
REQUIRED_IDS="challenge-master challenge-participant opus-source astra-source sol-source gemini-source elo-results match-matrix elo-historical gemini-costs"

check_archive() {
  local manifest="$P/evidence/artifact-manifest.json"
  [ -f "$manifest" ] || fail archive "missing evidence/artifact-manifest.json"
  jq -e '.artifacts | type == "array" and length > 0' "$manifest" >/dev/null || fail archive "no artifacts declared"
  [ -z "$(jq -r '.artifacts[].id' "$manifest" | sort | uniq -d)" ] || fail archive "duplicate artifact id"
  [ -z "$(jq -r '.artifacts[].path' "$manifest" | sort | uniq -d)" ] || fail archive "duplicate artifact path"
  local id path bytes digest commit file
  while IFS=$'\t' read -r id path bytes digest commit; do
    case "$path" in artifacts/*) ;; *) fail archive "artifact outside artifacts/: $path" ;; esac
    case "/$path/" in */../*|*/./*) fail archive "non-canonical artifact path: $path" ;; esac
    file="$P/$path"
    [ -f "$file" ] && [ ! -L "$file" ] || fail archive "missing regular file: $path"
    [ "$(size "$file")" = "$bytes" ] || fail archive "byte count mismatch: $path"
    [ "$(sha "$file")" = "$digest" ] || fail archive "SHA-256 mismatch: $path"
    [ -n "$commit" ] && [ "$commit" != null ] || fail archive "missing source commit: $path"
  done < <(jq -r '.artifacts[] | [.id, .path, (.bytes|tostring), .sha256, (.source_commit // "")] | @tsv' "$manifest")
  for id in $REQUIRED_IDS; do
    jq -e --arg id "$id" 'any(.artifacts[]; .id == $id)' "$manifest" >/dev/null || fail archive "required artifact missing: $id"
  done
}

check_programs() {
  local list="$P/evidence/pilot-2026-10-22/programs.tsv" file digest bytes lines rest n=0
  [ -f "$list" ] || fail programs "missing evidence/pilot-2026-10-22/programs.tsv"
  while IFS=$'\t' read -r file digest bytes lines rest; do
    [ "$file" = file ] && continue
    n=$((n + 1))
    [ -f "$P/evidence/pilot-2026-10-22/programs/$file" ] || fail programs "missing program: $file"
    [ "$(sha "$P/evidence/pilot-2026-10-22/programs/$file")" = "$digest" ] || fail programs "SHA-256 mismatch: $file"
    [ "$(size "$P/evidence/pilot-2026-10-22/programs/$file")" = "$bytes" ] || fail programs "byte count mismatch: $file"
    [ "$(wc -l < "$P/evidence/pilot-2026-10-22/programs/$file")" = "$lines" ] || fail programs "line count mismatch: $file"
  done < "$list"
  [ "$n" = 10 ] || fail programs "expected ten programs, found $n"
}

check_elo() {
  local summary="$P/evidence/elo-summary.json" results="$P/artifacts/benchmark/RESULTS.json" manifest="$P/evidence/artifact-manifest.json"
  local want_sha want_bytes trimmed
  want_sha=$(jq -r '.source.sha256' "$summary"); want_bytes=$(jq -r '.source.bytes' "$summary")
  # The archived copy keeps one extra trailing LF; either framing must match the recorded identity.
  trimmed=$(head -c "$want_bytes" "$results" | sha256sum | cut -d' ' -f1)
  { [ "$(sha "$results")" = "$want_sha" ] && [ "$(size "$results")" = "$want_bytes" ]; } \
    || { [ "$trimmed" = "$want_sha" ] && [ "$(size "$results")" = $((want_bytes + 1)) ]; } \
    || fail elo "RESULTS.json bytes differ from the identity recorded in elo-summary.json"
  [ "$(jq -r '.source.revision' "$summary")" = "$(jq -r '.artifacts[] | select(.path == "artifacts/benchmark/RESULTS.json") | .source_commit' "$manifest")" ] \
    || fail elo "Elo source revision drift"
  jq -e --slurpfile r "$results" '
      (.ratings | length) == ($r[0].elo | length)
      and all(.ratings[]; . as $x | $r[0].elo[$x.artifact] != null
            and $r[0].elo[$x.artifact].rating == $x.rating and $r[0].elo[$x.artifact].ci95 == $x.ci95
            and $x.ci95[0] <= $x.rating and $x.rating <= $x.ci95[1])
      and .pairwise == $r[0].pairwise' "$summary" >/dev/null || fail elo "Elo estimates, intervals or pairwise results drifted"
  jq -e '.panel.innovationzero_k1_entrant == false and .panel.families == 64 and .panel.folds == 3
         and .panel.seeds == [41,73,109] and .panel.cells_per_entrant == 576
         and .evidence_status == "repository-reported exploratory"' "$summary" >/dev/null \
    || fail elo "exploratory panel identity drifted or now claims a qualified InnovationZero entrant"
}

check_frontier() {
  local csv="$P/data/frontier-entrants.csv" results="$P/artifacts/benchmark/RESULTS.json"
  local hist="$P/artifacts/benchmark/HISTORICAL_THREE_ENTRANT.json" gem="$P/artifacts/benchmark/GEMINI_COSTS.json"
  local manifest="$P/evidence/artifact-manifest.json" expected
  expected=$(jq -rn --slurpfile r "$results" --slurpfile h "$hist" --slurpfile g "$gem" --slurpfile m "$manifest" '
    def r2: . * 100 | round / 100 | tostring | if test("\\.") then (if test("\\.[0-9]$") then . + "0" else . end) else . + ".00" end;
    def r1: . * 10 | round / 10 | tostring | if test("\\.") then . else . + ".0" end;
    def bytes($id): $m[0].artifacts[] | select(.id == $id) | .bytes;
    [["Opus5p5_Extra", "opus-source"], ["Gemini4_argon_high", "gemini-source"],
     ["GPT6p1_Sol_Max", "sol-source"], ["GPT6_Astra_Max", "astra-source"]][] as [$a, $id]
    | ($r[0].elo[$a]) as $e
    | (if $a == "Gemini4_argon_high" then $g[0] else $h[0].costs[$a] end) as $c
    | [$a, ($e.rating | r2), ($e.ci95[0] | r2), ($e.ci95[1] | r2), ($c.fit_wall_seconds | r1),
       ($c.peak_memory_bytes / 1e6 | r1), (bytes($id) | tostring), (bytes($id) / 3.5 | round | tostring)] | join(",")')
  [ "$(awk -F, 'NR > 1 { print $1 "," $3 "," $4 "," $5 "," $18 "," $19 "," $20 "," $21 }' "$csv")" = "$expected" ] \
    || fail frontier "data/frontier-entrants.csv disagrees with the archived Elo, cost or source-size receipts"
}

check_derived() {
  local tmp
  mkdir -p -- "$(build_root)"
  tmp=$(mktemp -d "${TMPDIR_PAPER:-$(build_root)/derive}.XXXXXX")
  bash "$P/scripts/derive-frontier.sh" "$tmp" 2>/dev/null || { rm -rf -- "$tmp"; fail derived "derive-frontier.sh refused the archived match matrix"; }
  local f
  for f in frontier-pairwise.csv frontier-family-strip.csv frontier-ratios.csv; do
    cmp -s "$tmp/$f" "$P/data/$f" || { rm -rf -- "$tmp"; fail derived "data/$f does not regenerate from the archived inputs"; }
  done
  rm -rf -- "$tmp"
}

strip_tex() { sed -E 's/(^|[^\\])%.*$/\1/' "$1"; }

check_tex() {
  local bib="$P/references.bib" keys cited f rel
  [ -f "$bib" ] || fail tex "missing references.bib"
  keys=$(grep -oE '^@[A-Za-z]+\{[^,[:space:]]+' "$bib" | sed 's/.*{//' | sort)
  [ -n "$keys" ] || fail tex "empty bibliography"
  [ -z "$(printf '%s\n' "$keys" | uniq -d)" ] || fail tex "duplicate bibliography key"
  local docs=("$P/main.tex" "$P/supplement.tex" "$P/archive.tex")
  local files=("${docs[@]}" "$P"/sections/*.tex "$P"/appendices/*.tex "$P"/supplement/*.tex "$P"/figures/*.tex)
  local inputs=""
  for f in "${files[@]}"; do
    [ -f "$f" ] || fail tex "missing manuscript file: ${f#"$P"/}"
    if strip_tex "$f" | grep -qE '\b(TODO|TBD|TBC|PLACEHOLDER|XXX)\b'; then fail tex "placeholder in ${f#"$P"/}"; fi
    if grep -q 'write18' "$f"; then fail tex "shell escape in ${f#"$P"/}"; fi
    while read -r rel; do
      [ -n "$rel" ] || continue
      case "$rel" in archive-index|archive-index.tex) continue ;; esac
      inputs="$inputs ${rel%.tex} "
      [ -f "$P/${rel%.tex}.tex" ] || fail tex "unresolved \\input{$rel} in ${f#"$P"/}"
    done < <(strip_tex "$f" | grep -oE '\\(input|include)\{[^}]+\}' | sed -E 's/.*\{//; s/\}$//')
    while read -r rel; do
      [ -n "$rel" ] || continue
      [ -f "$P/$rel" ] || fail tex "unresolved graphic $rel in ${f#"$P"/}"
    done < <(strip_tex "$f" | grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' | sed -E 's/.*\{//; s/\}$//')
  done
  cited=$(cat "${files[@]}" | sed -E 's/(^|[^\\])%.*$/\1/' \
    | grep -oE '\\(cite|citep|citet|citealp|citeauthor|citeyear|nocite)\*?(\[[^]]*\]){0,2}\{[^}]+\}' \
    | sed -E 's/.*\{//; s/\}$//' | tr ',' '\n' | tr -d ' ' | sort -u)
  for f in $cited; do printf '%s\n' "$keys" | grep -qxF "$f" || fail tex "unknown citation key: $f"; done
  for f in "$P"/sections/*.tex "$P"/appendices/*.tex "$P"/supplement/*.tex; do
    rel=${f#"$P"/}; rel=${rel%.tex}
    case "$inputs" in *" $rel "*) ;; *) fail tex "section file not included: $rel" ;; esac
  done
}

check_vocabulary() {
  local f hits
  for f in "$P"/main.tex "$P"/supplement.tex "$P"/archive.tex "$P"/sections/*.tex "$P"/appendices/*.tex \
           "$P"/supplement/*.tex "$P"/figures/*.tex "$P"/README.md "$P"/EDITORIAL_DECISIONS.md "$P"/evidence/literature.md; do
    [ -f "$f" ] || continue
    hits=$(sed -E 's/\\cite[a-z]*\*?(\[[^]]*\]){0,2}\{[^}]*\}//g; s/`[^`]*`//g; s/AlphaEvolve//g; s/[Pp]ermutation//g' "$f" \
      | grep -inE "$VOCAB" || true)
    [ -z "$hits" ] || fail vocabulary "restricted vocabulary in ${f#"$P"/}: $(printf '%s' "$hits" | head -1 | cut -c1-120)"
  done
}

check_sources() {
  check_archive; check_programs; check_elo; check_frontier; check_derived; check_tex; check_vocabulary
}

tex_escape() { sed -e 's/\\/\\textbackslash{}/g' -e 's/[&%$#_{}]/\\&/g' -e 's/~/\\textasciitilde{}/g'; }

prepare_archive() {
  local out=$1 manifest="$P/evidence/artifact-manifest.json"
  mkdir -p -- "$out"
  {
    echo "% Generated by scripts/check.sh --prepare from the checked manifests; do not edit."
    local id title kind path bytes digest commit lines
    while IFS='|' read -r id title kind; do
      IFS=$'\t' read -r path bytes digest commit < <(jq -r --arg id "$id" \
        '.artifacts[] | select(.id == $id) | [.path, (.bytes|tostring), .sha256, .source_commit] | @tsv' "$manifest")
      lines=$(wc -l < "$P/$path")
      printf '\\archiveentry{%s}{%s}{%s}{%s}{%s}{%s}{%s}{%s}\n' "$kind" "$title" "$(printf '%s' "$id" | tex_escape)" \
        "$path" "$(commas "$bytes")" "$(commas "$lines")" "$digest" "$commit"
    done <<'EOF'
challenge-participant|The participant packet, exactly as every entrant received it|text
challenge-master|The organizer's master file: packet, appendix and red-team record|text
opus-source|Opus5p5\_Extra: cross-fitted boosted-ridge stack|rust
gemini-source|Gemini4\_argon\_high: five-expert simplex stack|rust
sol-source|GPT6p1\_Sol\_Max: ridge-initialized histogram boosting|rust
astra-source|GPT6\_Astra\_Max: ridge, histogram trees and a convex mixture|rust
EOF
    echo '\archivepart{Ten programs written by the engine}{The programs of Table~\ref{M-tab:programs}, one per pilot dataset, emitted as standalone Rust and printed as compiled.}'
    local file title
    while IFS=$'\t' read -r file digest bytes lines title; do
      [ "$file" = file ] && continue
      printf '\\archiveentry{rust}{Dataset %s: %s}{%s}{%s}{%s}{%s}{%s}{%s}\n' "${file:0:8}" "$title" "$(printf '%s' "${file%.rs}" | tex_escape)" \
        "evidence/pilot-2026-10-22/programs/$file" "$(commas "$bytes")" "$(commas "$lines")" "$digest" "this publication"
    done < "$P/evidence/pilot-2026-10-22/programs.tsv"
  } > "$out/archive-index.tex"
}

check_pdfs() {
  local dir=$1 name defects fonts row cols
  for name in InnovationZero InnovationZero-Supplement InnovationZero-Archive; do
    [ -s "$dir/$name.pdf" ] && [ "$(size "$dir/$name.pdf")" -gt 1000 ] || fail pdf "missing PDF: $name"
    [ -f "$dir/$name.log" ] || fail pdf "missing TeX log: $name"
    defects=$(grep -E 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined|Missing character:|multiply defined|LaTeX Font Warning: Font shape|LaTeX Error:|Package .* Error:' "$dir/$name.log" | head -12 || true)
    [ -z "$defects" ] || fail pdf "layout or reference defects in $name:
$defects"
    pdfinfo "$dir/$name.pdf" | grep -qE '^Pages:[[:space:]]+[1-9]' || fail pdf "empty PDF: $name"
    [ "$(pdftotext "$dir/$name.pdf" - | wc -c)" -gt 1000 ] || fail pdf "no extractable text: $name"
    fonts=$(pdffonts "$dir/$name.pdf" | tail -n +3)
    while read -r row; do
      [ -n "$row" ] || continue
      case "$row" in *"Type 3"*) fail pdf "bitmap font in $name: $row" ;; esac
      read -r -a cols <<<"$row"
      [ "${cols[${#cols[@]}-5]}" = yes ] || fail pdf "unembedded font in $name: $row"
    done <<<"$fonts"
  done
  echo "PDFs: 3 built; references resolved; fonts embedded; no overfull boxes or missing glyphs"
}

self_test() {
  local build base
  build=$(build_root)
  mkdir -p -- "$build"
  base=$(mktemp -d "$build/self-test.XXXXXX")
  trap "rm -rf -- '$base'" EXIT
  mkdir -p "$base/clean"
  (cd "$here" && cp -r artifacts data evidence figures sections appendices supplement scripts \
     main.tex supplement.tex archive.tex references.bib README.md EDITORIAL_DECISIONS.md "$base/clean/")
  PAPER_DIR="$base/clean" TMPDIR_PAPER="$base/tmp" bash "$here/scripts/check.sh" >/dev/null \
    || fail self-test "the unmodified fixture was refused"
  local cases=(bytes digest duplicate escape program elo qualified frontier derived citation input section vocabulary)
  local c fx m
  for c in "${cases[@]}"; do
    fx="$base/$c"; cp -r "$base/clean" "$fx"; m="$fx/evidence/artifact-manifest.json"
    case $c in
      bytes)      jq '.artifacts[0].bytes += 1' "$m" > "$m.new" && mv "$m.new" "$m" ;;
      digest)     printf ' ' >> "$fx/artifacts/challenge/participant.md" ;;
      duplicate)  jq '.artifacts += [.artifacts[0]]' "$m" > "$m.new" && mv "$m.new" "$m" ;;
      escape)     jq '.artifacts[0].path = "artifacts/../README.md"' "$m" > "$m.new" && mv "$m.new" "$m" ;;
      program)    printf '\n' >> "$fx/evidence/pilot-2026-10-22/programs/047e62b52a3f4ede.rs" ;;
      elo)        jq '.ratings[0].rating += 1' "$fx/evidence/elo-summary.json" > "$fx/e.json" && mv "$fx/e.json" "$fx/evidence/elo-summary.json" ;;
      qualified)  jq '.panel.innovationzero_k1_entrant = true' "$fx/evidence/elo-summary.json" > "$fx/e.json" && mv "$fx/e.json" "$fx/evidence/elo-summary.json" ;;
      frontier)   sed -i 's/^Opus5p5_Extra,1,1641.30/Opus5p5_Extra,1,1651.30/' "$fx/data/frontier-entrants.csv" ;;
      derived)    sed -i '2s/,0\.[0-9]*,/,0.9999,/' "$fx/data/frontier-pairwise.csv"; sed -i '3s/,0\.5990,/,0.6990,/' "$fx/data/frontier-pairwise.csv" ;;
      citation)   printf '\\citep{nonexistent2026}\n' >> "$fx/sections/10-conclusion.tex" ;;
      input)      printf '\\input{figures/missing-figure}\n' >> "$fx/sections/10-conclusion.tex" ;;
      section)    printf '%% orphan\n' > "$fx/sections/99-orphan.tex" ;;
      vocabulary) printf 'The proposer is a genetic search.\n' >> "$fx/sections/10-conclusion.tex" ;;
    esac
    if PAPER_DIR="$fx" TMPDIR_PAPER="$base/tmp" bash "$here/scripts/check.sh" >/dev/null 2>&1; then
      fail self-test "accepted the broken '$c' fixture"
    fi
  done
  echo "Validator self-test: ${#cases[@]} refusal cases fired (${cases[*]}); the clean fixture passed"
}

case "${1:-}" in
  --self-test) self_test ;;
  --prepare)   check_sources; prepare_archive "${2:?--prepare needs a directory}"
               echo "Publication sources verified; archive index written to ${2}/archive-index.tex" ;;
  --pdf-dir)   check_sources; check_pdfs "${2:?--pdf-dir needs a directory}" ;;
  '')          check_sources
               echo "Publication sources: $(jq '.artifacts | length' "$P/evidence/artifact-manifest.json") archived artifacts and 10 programs verified; Elo, frontier and derived tables bound to receipts; inputs, citations and vocabulary clean" ;;
  *) echo "usage: $0 [--self-test | --prepare DIR | --pdf-dir DIR]" >&2; exit 2 ;;
esac
