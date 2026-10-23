#!/usr/bin/env bash
# docs/paper/eurekazero/scripts/check.sh — the north-star draft's source and output contract.
#
#   scripts/check.sh                  check sources (no TeX needed)
#   scripts/check.sh --prepare DIR    check sources, then write DIR/gap-ledger.tex and DIR/claim-ledger.tex
#   scripts/check.sh --pdf-dir DIR    check sources and the built PDF and its log in DIR
#   scripts/check.sh --submission     refuse while anything synthetic remains (this draft must fail it)
#   scripts/check.sh --self-test      prove that every refusal below fires on a broken copy
#
# Refusals, each a separate check:
#   tex         every \input resolves (except the two generated ledgers), every citation is in
#               references.bib, every section, appendix and figure file is included, no placeholder
#               or shell escape
#   vocabulary  the manuscript and its Markdown do not use the repository's restricted vocabulary
#               (the deploy/ci/ip-scrub.sh pattern; only the proper name AlphaEvolve is exempt)
#   synthetic   every data/synthetic/*.csv and generated/numbers.tex carries its SYNTHETIC banner and
#               regenerates byte for byte from scripts/synthesize.sh; every \syn names a gap in
#               northstar/gaps.json; no synthetic macro appears outside \syn; every figure that reads
#               data/synthetic draws \synfigtag; every gap is cited in the manuscript
#   ledgers     northstar/gaps.json and evidence/claims.json are well formed, ids are unique, and
#               every gap a claim depends on exists
#   pdf         no overfull box, unresolved reference or citation, missing glyph, unembedded or
#               Type 3 font, and the north-star watermark is present on the output
#
# Bash, jq, awk and coreutils only (ADR-0006: no new Python). PAPER_DIR overrides the paper
# directory, which the self-test uses to run these checks against disposable copies.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
P=${PAPER_DIR:-$here}
export LC_ALL=C
JOB=EurekaZero-NorthStar
BANNER_CSV='# SYNTHETIC north-star data: illustrative, not a measured result.'
BANNER_TEX='% SYNTHETIC north-star numbers: illustrative targets, not measured results.'
VOCAB='\bGP\b|genome|genetic|evolut|evolv|\bpopulation|chromosome|mutation|crossover|lexicase|fitness|keijzer|koza|expression tree|parse tree|syntax tree|symbolic regression'

fail() { printf 'north-star check (%s): %s\n' "$1" "$2" >&2; exit 1; }
build_root() { printf '%s' "${EZ_BUILD_DIR:-$(git -C "$here" rev-parse --show-toplevel)/target/eurekazero-paper}"; }
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
      case "$rel" in gap-ledger|claim-ledger) continue ;; esac
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
  for f in $(manuscript) "$P"/*.md "$P"/ez.sty "$P"/northstar/gaps.json "$P"/evidence/claims.json "$P"/evidence/*.md; do
    [ -f "$f" ] || continue
    hits=$(sed -E 's/\\cite[a-z]*\*?(\[[^]]*\]){0,2}\{[^}]*\}//g; s/[Aa]lpha[Ee]volve//g; s/[Pp]ermutation//g' "$f" \
      | grep -inE "$VOCAB" || true)
    [ -z "$hits" ] || fail vocabulary "restricted vocabulary in ${f#"$P"/}: $(printf '%s' "$hits" | head -1 | cut -c1-120)"
  done
}

check_ledgers() {
  local gaps="$P/northstar/gaps.json" claims="$P/evidence/claims.json" id file
  jq -e '.schema == "eurekazero.gaps/v1" and (.gaps | type == "array" and length > 0)' "$gaps" >/dev/null \
    || fail ledgers "northstar/gaps.json is not a eurekazero.gaps/v1 ledger"
  jq -e 'all(.gaps[]; (.id | test("^EZ-G[0-9]{2}$")) and (.title|length>0) and (.work_package|length>0)
           and (.acceptance|length>0) and (.status == "synthetic"))' "$gaps" >/dev/null \
    || fail ledgers "a gap lacks id, title, work package, acceptance test or synthetic status"
  [ -z "$(jq -r '.gaps[].id' "$gaps" | sort | uniq -d)" ] || fail ledgers "duplicate gap id"
  while read -r file; do
    [ -n "$file" ] || continue
    [ -f "$P/$file" ] || fail ledgers "gap ledger names a missing file: $file"
  done < <(jq -r '.gaps[].files[]?' "$gaps")
  jq -e '.schema == "eurekazero.claims/v1" and (.claims | type == "array" and length > 0)' "$claims" >/dev/null \
    || fail ledgers "evidence/claims.json is not a eurekazero.claims/v1 ledger"
  jq -e 'all(.claims[]; (.id | test("^EZ-C[0-9]{2}$")) and (.status | IN("proposed","protocol","synthetic-north-star","do-not-assert")))' \
    "$claims" >/dev/null || fail ledgers "a claim has a malformed id or an unknown status"
  [ -z "$(jq -r '.claims[].id' "$claims" | sort | uniq -d)" ] || fail ledgers "duplicate claim id"
  while read -r id; do
    [ -n "$id" ] || continue
    jq -e --arg id "$id" 'any(.gaps[]; .id == $id)' "$gaps" >/dev/null || fail ledgers "claim depends on unknown gap $id"
  done < <(jq -r '.claims[].gaps[]?' "$claims")
}

# Positions of \syn[...]{...} arguments, then every synthetic macro outside them.
naked_synthetic() {
  awk -v names="$1" '
    BEGIN { n = split(names, N, " "); for (i = 1; i <= n; i++) known[N[i]] = 1 }
    { sub(/(^|[^\\])%.*$/, ""); text = text $0 "\n" }
    END {
      L = length(text); r = 0; pos = 1
      while ((i = index(substr(text, pos), "\\syn[")) > 0) {
        s = pos + i - 1; j = index(substr(text, s), "]"); if (!j) break
        k = s + j; if (substr(text, k, 1) != "{") { pos = k; continue }
        depth = 0
        for (q = k; q <= L; q++) { c = substr(text, q, 1)
          if (c == "{") depth++; else if (c == "}") { depth--; if (depth == 0) break } }
        r++; rs[r] = k; re[r] = q; pos = q + 1
      }
      pos = 1; bad = 0
      while (match(substr(text, pos), /\\EZ[A-Za-z]+/)) {
        st = pos + RSTART - 1; name = substr(text, st + 1, RLENGTH - 1); pos = st + RLENGTH
        if (!(name in known)) continue
        inside = 0; for (t = 1; t <= r; t++) if (st > rs[t] && st < re[t]) { inside = 1; break }
        if (!inside) { print name; bad = 1 }
      }
      exit bad
    }' "$2"
}

check_synthetic() {
  local f out names ids id bad build
  for f in "$P"/data/synthetic/*.csv; do
    [ -f "$f" ] || fail synthetic "no synthetic data files"
    [ "$(head -n 1 "$f")" = "$BANNER_CSV" ] || fail synthetic "missing SYNTHETIC banner: ${f#"$P"/}"
  done
  [ "$(head -n 1 "$P/generated/numbers.tex")" = "$BANNER_TEX" ] || fail synthetic "missing SYNTHETIC banner: generated/numbers.tex"
  build=$(build_root)
  out="$build/synthetic-check/$$"
  rm -rf -- "$out"; mkdir -p -- "$out"
  bash "$here/scripts/synthesize.sh" "$out"
  for f in "$out"/*.csv; do
    cmp -s "$f" "$P/data/synthetic/$(basename "$f")" || fail synthetic "data/synthetic/$(basename "$f") does not regenerate from scripts/synthesize.sh"
  done
  [ "$(find "$P/data/synthetic" -name '*.csv' | wc -l)" = "$(find "$out" -name '*.csv' | wc -l)" ] \
    || fail synthetic "data/synthetic holds a file the generator does not write"
  cmp -s "$out/numbers.tex" "$P/generated/numbers.tex" || fail synthetic "generated/numbers.tex does not regenerate from scripts/synthesize.sh"
  rm -rf -- "$out"
  names=$(grep -oE '^\\newcommand\\EZ[A-Za-z]+' "$P/generated/numbers.tex" | sed 's/.*\\//' | tr '\n' ' ')
  ids=$(jq -r '.gaps[].id' "$P/northstar/gaps.json")
  while read -r f; do
    while read -r id; do
      [ -n "$id" ] || continue
      printf '%s\n' "$ids" | grep -xF -- "$id" >/dev/null || fail synthetic "\\syn names unknown gap $id in ${f#"$P"/}"
    done < <(strip_tex "$f" | grep -oE '\\syn\[[^]]*\]' | sed -E 's/\\syn\[//; s/\]$//' | sort -u)
    # A figure that draws \synfigtag is labelled synthetic as a whole; elsewhere every synthetic
    # macro must sit inside \syn[gap]{...}.
    if ! grep -q '\\synfigtag' "$f" && ! bad=$(naked_synthetic "$names" "$f"); then
      fail synthetic "synthetic number outside \\syn in ${f#"$P"/}: $(printf '%s' "$bad" | head -1)"
    fi
    if grep -q 'data/synthetic/' "$f" && ! grep -q '\\synfigtag' "$f"; then
      fail synthetic "figure reads synthetic data without \\synfigtag: ${f#"$P"/}"
    fi
  done < <(manuscript)
  while read -r id; do
    manuscript | xargs cat | grep -F -- "$id" >/dev/null || fail synthetic "gap $id is not cited anywhere in the manuscript"
  done <<<"$ids"
}

check_sources() { check_tex; check_vocabulary; check_ledgers; check_synthetic; }

tex_escape() { sed -e 's/\\/\\textbackslash{}/g' -e 's/[&%$#_{}]/\\&/g' -e 's/~/\\textasciitilde{}/g'; }

prepare_ledgers() {
  local out=$1 id title domain wp acceptance replaces claims status claim evidence gaps
  mkdir -p -- "$out"
  {
    echo "% Generated by scripts/check.sh --prepare from northstar/gaps.json; do not edit."
    while IFS=$'\t' read -r id title domain wp acceptance replaces claims; do
      printf '\\gapentry{%s}{%s}{%s}{%s}{%s}{%s}{%s}\n' "$id" \
        "$(printf '%s' "$title" | tex_escape)" "$(printf '%s' "$domain" | tex_escape)" \
        "$(printf '%s' "$wp" | tex_escape)" "$(printf '%s' "$acceptance" | tex_escape)" \
        "$(printf '%s' "$replaces" | tex_escape)" "$(printf '%s' "$claims" | tex_escape)"
    done < <(jq -r '.gaps[] | [.id, .title, .domain, .work_package, .acceptance,
                     (.replaces | join("; ")), ((.claims // []) | join(", "))] | @tsv' "$P/northstar/gaps.json")
  } > "$out/gap-ledger.tex"
  {
    echo "% Generated by scripts/check.sh --prepare from evidence/claims.json; do not edit."
    while IFS=$'\t' read -r id claim evidence status gaps; do
      printf '\\claimentry{%s}{%s}{%s}{%s}{%s}\n' "$id" "$(printf '%s' "$claim" | tex_escape)" \
        "$(printf '%s' "$evidence" | tex_escape)" "$(printf '%s' "$status" | tex_escape)" "$(printf '%s' "$gaps" | tex_escape)"
    done < <(jq -r '.claims[] | [.id, .claim, .minimum_evidence, .status, ((.gaps // []) | join(", "))] | @tsv' "$P/evidence/claims.json")
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
  # Every page carries the shipout footer; letter-spaced title text is not a reliable probe.
  local pages marked
  pages=$(pdfinfo "$dir/$JOB.pdf" | awk '/^Pages:/ {print $2}')
  marked=$(pdftotext "$dir/$JOB.pdf" - | grep -cF 'results in purple are SYNTHETIC' || true)
  [ "$marked" -ge "$pages" ] || fail pdf "the north-star footer is missing from $((pages - marked)) of $pages pages"
  fonts=$(pdffonts "$dir/$JOB.pdf" | tail -n +3)
  while read -r row; do
    [ -n "$row" ] || continue
    case "$row" in *"Type 3"*) fail pdf "bitmap font: $row" ;; esac
    read -r -a cols <<<"$row"
    [ "${cols[${#cols[@]}-5]}" = yes ] || fail pdf "unembedded font: $row"
  done <<<"$fonts"
  echo "PDF: built; references resolved; fonts embedded; no overfull boxes or missing glyphs; every page carries the synthetic footer"
}

check_submission() {
  local left
  left=$(manuscript | xargs grep -nE '\\syn\[|\\synbadge|\\synfigtag|data/synthetic/|generated/numbers' 2>/dev/null | wc -l)
  grep -q '\\EZsyntheticfalse' "$P/main.tex" || left=$((left + 1))
  [ "$left" -eq 0 ] || fail submission "$left synthetic element(s) remain; this manuscript is a north-star draft and cannot be submitted"
  echo "Submission check: no synthetic elements remain"
}

self_test() {
  local build base c fx
  build=$(build_root)
  mkdir -p -- "$build"
  base=$(mktemp -d "$build/self-test.XXXXXX")
  # Expand now: the local variable is out of scope when the EXIT trap runs.
  # shellcheck disable=SC2064
  trap "rm -rf -- '$base'" EXIT
  mkdir -p "$base/clean"
  (cd "$here" && cp -r data generated northstar evidence figures sections appendices scripts \
     main.tex ez.sty references.bib "$base/clean/" && { for m in ./*.md; do [ ! -f "$m" ] || cp -- "$m" "$base/clean/"; done; })
  PAPER_DIR="$base/clean" bash "$here/scripts/check.sh" >/dev/null || fail self-test "the unmodified fixture was refused"
  if PAPER_DIR="$base/clean" bash "$here/scripts/check.sh" --submission >/dev/null 2>&1; then
    fail self-test "the submission guard accepted the synthetic draft"
  fi
  local cases=(gapid naked figtag banner regen ledger citation input orphan vocabulary)
  for c in "${cases[@]}"; do
    fx="$base/$c"; cp -r "$base/clean" "$fx"
    case $c in
      gapid)      printf '\\syn[EZ-G99]{\\EZdA}\n' >> "$fx/sections/13-conclusion.tex" ;;
      naked)      printf 'An unlabelled number: \\EZdA.\n' >> "$fx/sections/13-conclusion.tex" ;;
      figtag)     sed -i 's/\\synfigtag//' "$fx/figures/fig-primary.tex" ;;
      banner)     sed -i '1d' "$fx/data/synthetic/primary.csv" ;;
      regen)      sed -i '5s/,0\./,0.9/' "$fx/data/synthetic/primary.csv" ;;
      ledger)     jq '.gaps += [.gaps[0]]' "$fx/northstar/gaps.json" > "$fx/g.json" && mv "$fx/g.json" "$fx/northstar/gaps.json" ;;
      citation)   printf '\\citep{nonexistent2026}\n' >> "$fx/sections/13-conclusion.tex" ;;
      input)      printf '\\input{figures/missing-figure}\n' >> "$fx/sections/13-conclusion.tex" ;;
      orphan)     printf '%% orphan\n' > "$fx/sections/99-orphan.tex" ;;
      vocabulary) printf 'The proposer is a genetic search.\n' >> "$fx/sections/13-conclusion.tex" ;;
    esac
    if PAPER_DIR="$fx" bash "$here/scripts/check.sh" >/dev/null 2>&1; then
      fail self-test "accepted the broken '$c' fixture"
    fi
  done
  echo "Validator self-test: ${#cases[@]} refusal cases fired (${cases[*]}); the clean fixture passed; the submission guard refused the synthetic draft"
}

case "${1:-}" in
  --self-test)  self_test ;;
  --prepare)    check_sources; prepare_ledgers "${2:?--prepare needs a directory}"
                echo "North-star sources verified; gap and claim ledgers written to ${2}" ;;
  --pdf-dir)    check_sources; check_pdf "${2:?--pdf-dir needs a directory}" ;;
  --submission) check_submission ;;
  '')           check_sources
                echo "North-star sources: $(jq '.gaps | length' "$P/northstar/gaps.json") gaps and $(jq '.claims | length' "$P/evidence/claims.json") claims ledgered; synthetic data regenerate and are labelled; inputs, citations and vocabulary clean" ;;
  *) echo "usage: $0 [--self-test | --prepare DIR | --pdf-dir DIR | --submission]" >&2; exit 2 ;;
esac
