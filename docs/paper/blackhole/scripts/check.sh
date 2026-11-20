#!/usr/bin/env bash
# docs/paper/blackhole/scripts/check.sh — source, IP-disclosure, privacy, ledger, and PDF contract.
#
#   scripts/check.sh                  check sources (no TeX needed)
#   scripts/check.sh --prepare DIR    check sources, then write DIR/claim-ledger.tex
#   scripts/check.sh --pdf-dir DIR    check sources and the built PDF and its log in DIR
#   scripts/check.sh --self-test      prove that every refusal below fires on a broken copy
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
P=${PAPER_DIR:-$here}
export LC_ALL=C
JOB=EurekaZero-BlackHole

# Repository shipped-text IP scrub + protected solution-encoding grammar/emitter scrub
VOCAB='\bGP\b|genome|genetic|evolut|evolv|\bpopulation|chromosome|mutation|crossover|lexicase|fitness|keijzer|koza|expression tree|parse tree|syntax tree|symbolic regression|detokeniz|\(pipe[[:space:]]|grammar mask|operator inventory'
LEAK_PAT='proc_uid|BLIND_KEY_RECIPE_TEST|wp2_recipe/runs'

fail() { printf 'blackhole-paper check (%s): %s\n' "$1" "$2" >&2; exit 1; }
build_root() { printf '%s' "${BH_BUILD_DIR:-$(git -C "$here" rev-parse --show-toplevel)/target/blackhole-paper}"; }
strip_tex() { sed -E 's/(^|[^\\])%.*$/\1/' "$1"; }
manuscript() { printf '%s\n' "$P/main.tex" "$P"/sections/*.tex "$P"/appendices/*.tex "$P"/figures/*.tex; }

check_tex() {
  local bib="$P/references.bib" keys cited f rel inputs=""
  [ -f "$bib" ] || fail tex "missing references.bib"
  keys=$(grep -oE '^@[A-Za-z]+\{[^,[:space:]]+' "$bib" | sed 's/.*{//' | sort)
  [ -n "$keys" ] || fail tex "empty bibliography"
  [ -z "$(printf '%s\n' "$keys" | uniq -d)" ] || fail tex "duplicate bibliography key"
  while read -r f; do
    [ -f "$f" ] || fail tex "missing manuscript file: ${f#"$P"/}"
    if strip_tex "$f" | grep -E '\b(TODO|TBD|TBC|PLACEHOLDER|XXX)\b' >/dev/null; then fail tex "placeholder in ${f#"$P"/}"; fi
    if grep -q 'write18' "$f"; then fail tex "shell escape in ${f#"$P"/}"; fi
    while read -r rel; do
      [ -n "$rel" ] || continue
      case "$rel" in claim-ledger) continue ;; esac
      inputs="$inputs ${rel%.tex} "
      [ -f "$P/${rel%.tex}.tex" ] || fail tex "unresolved \\input{$rel} in ${f#"$P"/}"
    done < <(strip_tex "$f" | grep -oE '\\(input|include)\{[^}]+\}' | sed -E 's/.*\{//; s/\}$//')
  done < <(manuscript)
  cited=$(manuscript | xargs cat | sed -E 's/(^|[^\\])%.*$/\1/' \
    | grep -oE '\\(cite|citep|citet|citealp|citeauthor|citeyear|nocite)\*?(\[[^]]*\]){0,2}\{[^}]+\}' \
    | sed -E 's/.*\{//; s/\}$//' | tr ',' '\n' | tr -d ' ' | sort -u)
  for f in $cited; do printf '%s\n' "$keys" | grep -xF -- "$f" >/dev/null || fail tex "unknown citation key: $f"; done
  for f in "$P"/sections/*.tex "$P"/appendices/*.tex "$P"/figures/*.tex; do
    rel=${f#"$P"/}; rel=${rel%.tex}
    case "$inputs" in *" $rel "*) ;; *) fail tex "file not included anywhere: $rel" ;; esac
  done
}

check_vocabulary() {
  local f hits
  for f in $(manuscript) "$P"/*.md "$P"/bh.sty "$P"/generated/*.tex "$P"/evidence/claims.json "$P"/data/measured/*.csv; do
    [ -f "$f" ] || continue
    hits=$(sed -E 's/\\cite[a-z]*\*?(\[[^]]*\]){0,2}\{[^}]*\}//g; s/[Aa]lpha[Ee]volve//g; s/[Pp]ermutation//g' "$f" \
      | grep -inE "$VOCAB" || true)
    [ -z "$hits" ] || fail vocabulary "restricted vocabulary in ${f#"$P"/}: $(printf '%s' "$hits" | head -1 | cut -c1-120)"
    hits=$(grep -inE "$LEAK_PAT" "$f" || true)
    [ -z "$hits" ] || fail privacy "blind/sealed identifier in ${f#"$P"/}: $(printf '%s' "$hits" | head -1 | cut -c1-120)"
  done
}

check_data() {
  local f sz count=0
  for f in "$P"/data/measured/*.csv; do
    [ -f "$f" ] || fail data "no measured CSV files in data/measured/"
    sz=$(wc -c < "$f")
    [ "$sz" -gt 10 ] || fail data "empty CSV: ${f#"$P"/}"
    [ "$sz" -le 524288 ] || fail data "CSV exceeds 512 KiB cap: ${f#"$P"/} ($sz bytes)"
    count=$((count + 1))
  done
  [ "$count" -ge 4 ] || fail data "expected >=4 measured CSV files, found $count"
  [ -f "$P/generated/measured_numbers.tex" ] || fail data "missing generated/measured_numbers.tex"
}

check_ledgers() {
  local claims="$P/evidence/claims.json"
  [ -f "$claims" ] || fail ledgers "missing evidence/claims.json"
  jq -e '.schema == "blackhole.claims/v1" and (.claims | type == "array" and length >= 8)' "$claims" >/dev/null \
    || fail ledgers "evidence/claims.json is not a valid blackhole.claims/v1 ledger"
  jq -e 'all(.claims[]; (.id | test("^BH-C[0-9]{2}$"))
           and (.claim | length > 0)
           and (.regime | IN("recall","in-corpus","fair","memory","disjoint-val","protocol","pending-read"))
           and (.evidence | length > 0)
           and (.status | IN("established","pre-registered","pending-v7-read")))' "$claims" >/dev/null \
    || fail ledgers "a claim lacks a valid id, claim, regime, evidence, or status"
  [ -z "$(jq -r '.claims[].id' "$claims" | sort | uniq -d)" ] || fail ledgers "duplicate claim id"
}

check_sources() { check_tex; check_vocabulary; check_data; check_ledgers; }

tex_escape() { sed -e 's/\\/\\textbackslash{}/g' -e 's/[&%$#_{}]/\\&/g' -e 's/~/\\textasciitilde{}/g'; }

prepare_ledgers() {
  local out=$1 id claim regime evidence status
  mkdir -p -- "$out"
  {
    echo "% Generated by scripts/check.sh --prepare from evidence/claims.json; do not edit."
    while IFS=$'\t' read -r id regime status claim evidence; do
      printf '\\claimentry{%s}{%s}{%s}{%s}{%s}\n' "$id" \
        "$(printf '%s' "$regime" | tex_escape)" \
        "$(printf '%s' "$status" | tex_escape)" \
        "$(printf '%s' "$claim" | tex_escape)" \
        "$(printf '%s' "$evidence" | tex_escape)"
    done < <(jq -r '.claims[] | [.id, .regime, .status, .claim, .evidence] | @tsv' "$P/evidence/claims.json")
  } > "$out/claim-ledger.tex"
}

check_pdf() {
  local dir=$1 defects fonts row cols
  [ -s "$dir/$JOB.pdf" ] || fail pdf "missing PDF: $JOB"
  [ -f "$dir/$JOB.log" ] || fail pdf "missing TeX log: $JOB"
  defects=$(grep -E 'Overfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined|Missing character:|multiply defined|LaTeX Font Warning: Font shape|LaTeX Error:|Package .* Error:' "$dir/$JOB.log" | head -12 || true)
  [ -z "$defects" ] || fail pdf "layout or reference defects:
$defects"
  pdfinfo "$dir/$JOB.pdf" | grep -E '^Pages:[[:space:]]+[1-9]' >/dev/null || fail pdf "empty PDF"
  fonts=$(pdffonts "$dir/$JOB.pdf" | tail -n +3)
  while read -r row; do
    [ -n "$row" ] || continue
    case "$row" in *"Type 3"*) fail pdf "bitmap font: $row" ;; esac
    read -r -a cols <<<"$row"
    [ "${cols[${#cols[@]}-5]}" = yes ] || fail pdf "unembedded font: $row"
  done <<<"$fonts"
  echo "PDF: built; references resolved; fonts 100% embedded; 0 overfull boxes or missing glyphs"
}

self_test() {
  local build base c fx
  build=$(build_root)
  mkdir -p -- "$build"
  base=$(mktemp -d "$build/self-test.XXXXXX")
  # shellcheck disable=SC2064
  trap "rm -rf -- '$base'" EXIT
  mkdir -p "$base/clean"
  (cd "$here" && cp -r data generated evidence figures sections appendices scripts \
     main.tex bh.sty references.bib "$base/clean/" && { for m in ./*.md; do [ ! -f "$m" ] || cp -- "$m" "$base/clean/"; done; })
  PAPER_DIR="$base/clean" bash "$here/scripts/check.sh" >/dev/null || fail self-test "the unmodified fixture was refused"
  local cases=(placeholder citation input orphan vocabulary privacy ledger)
  for c in "${cases[@]}"; do
    fx="$base/$c"; cp -r "$base/clean" "$fx"
    case $c in
      placeholder) printf 'An open marker: TBD.\n' >> "$fx/sections/13-conclusion.tex" ;;
      citation)    printf '\\citep{nonexistent2099}\n' >> "$fx/sections/13-conclusion.tex" ;;
      input)       printf '\\input{figures/missing-figure}\n' >> "$fx/sections/13-conclusion.tex" ;;
      orphan)      printf '%% orphan\n' > "$fx/sections/99-orphan.tex" ;;
      vocabulary)  printf 'The proposer uses genetic mutation.\n' >> "$fx/sections/13-conclusion.tex" ;;
      privacy)     printf 'Leaked blind key proc_uid=abc.\n' >> "$fx/sections/13-conclusion.tex" ;;
      ledger)      jq '.claims += [.claims[0]]' "$fx/evidence/claims.json" > "$fx/c.json" && mv "$fx/c.json" "$fx/evidence/claims.json" ;;
    esac
    if PAPER_DIR="$fx" bash "$here/scripts/check.sh" >/dev/null 2>&1; then
      fail self-test "accepted the broken '$c' fixture"
    fi
  done
  echo "Validator self-test: ${#cases[@]} refusal cases fired (${cases[*]}); clean fixture passed"
}

case "${1:-}" in
  --self-test) self_test ;;
  --prepare)   check_sources; prepare_ledgers "${2:?--prepare needs a directory}"
               echo "Black-Hole paper sources verified; claim ledger written to ${2}" ;;
  --pdf-dir)   check_sources; check_pdf "${2:?--pdf-dir needs a directory}" ;;
  '')          check_sources
               echo "Black-Hole paper sources: $(jq '.claims | length' "$P/evidence/claims.json") claims ledgered; inputs, citations, privacy and IP vocabulary clean" ;;
  *) echo "usage: $0 [--self-test | --prepare DIR | --pdf-dir DIR]" >&2; exit 2 ;;
esac
