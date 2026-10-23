#!/usr/bin/env bash
# docs/paper/innovationzero/scripts/build.sh — check the sources, then typeset the three volumes.
#
#   bash scripts/build.sh            check, typeset and check the PDFs (make pdf)
#   bash scripts/build.sh --clean    remove only the files this build produces
#
# Products go to the lane's reused target directory, $repo/target/innovationzero-paper (override
# with IZ_BUILD_DIR): InnovationZero.pdf (the paper), InnovationZero-Supplement.pdf and
# InnovationZero-Archive.pdf, with their TeX logs and build-environment.txt. Needs XeLaTeX, latexmk
# and BibTeX (TeX Live 2023 or later), Poppler, jq, curl, unzip and sha256sum. Fonts come from
# fonts.lock: pinned release archives, verified by SHA-256 and cached outside the working tree
# (IZ_FONT_CACHE, default ~/.cache/innovationzero-fonts). The three font archives are the only
# network reads, and only when they are not already cached.
set -euo pipefail
paper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(git -C "$paper" rev-parse --show-toplevel)
build=${IZ_BUILD_DIR:-$repo/target/innovationzero-paper}
cache=${IZ_FONT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/innovationzero-fonts}
volumes=(InnovationZero-Supplement:supplement InnovationZero:main InnovationZero-Archive:archive)

if [ "${1:-}" = --clean ]; then
  # Never blanket-delete target/: only this build's known products.
  for v in "${volumes[@]}"; do
    for ext in aux bbl blg fdb_latexmk fls log out toc xdv pdf build.txt; do rm -f -- "$build/${v%%:*}.$ext"; done
  done
  rm -f -- "$build/archive-index.tex" "$build/build-environment.txt"
  exit 0
fi
[ $# -eq 0 ] || { echo "usage: $0 [--clean]" >&2; exit 2; }

for tool in xelatex latexmk bibtex pdfinfo pdftotext pdffonts jq curl unzip sha256sum; do
  command -v "$tool" >/dev/null || { echo "build.sh: $tool is required" >&2; exit 2; }
done
mkdir -p -- "$build" "$cache/archives" "$cache/fonts"

while read -r digest url members; do
  case "$digest" in ''|'#'*) continue ;; esac
  archive="$cache/archives/$(basename "$url")"
  if [ ! -f "$archive" ] || ! echo "$digest  $archive" | sha256sum -c --status; then
    echo "fetching $url"
    curl -fsSL --retry 3 -o "$archive.part" "$url"
    mv -- "$archive.part" "$archive"
  fi
  echo "$digest  $archive" | sha256sum -c --status || { echo "build.sh: SHA-256 mismatch for $url" >&2; exit 3; }
  for m in $members; do
    # Re-extract from the verified archive, so a modified cached font cannot be reused.
    unzip -o -q -j "$archive" "$m" -d "$cache/fonts"
  done
done < "$paper/fonts.lock"

bash "$paper/scripts/check.sh" --prepare "$build"

# Reproducible dates: the committer time of the pinned main snapshot unless overridden.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1792655756} FORCE_SOURCE_DATE=1 TZ=UTC
export TEXINPUTS="$paper//:$build:" BIBINPUTS="$paper:"
{
  printf 'SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'source_head=%s\n' "$(git -C "$repo" rev-parse HEAD)"
  printf 'source_dirty=%s\n' "$(git -C "$repo" status --porcelain -- "$paper" | wc -l)"
  latexmk -version | head -n 2
  xelatex --version | head -n 1
  bibtex --version | head -n 1
  pdfinfo -v 2>&1 | head -n 1
  sha256sum "$paper/fonts.lock"
} > "$build/build-environment.txt"

pretex="\\def\\IZFontDir{$cache/fonts/}"
for v in "${volumes[@]}"; do
  job=${v%%:*} src=${v#*:}
  (cd "$paper" && latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
     -no-shell-escape -outdir="$build" -jobname="$job" -usepretex="$pretex" "$src.tex" \
     > "$build/$job.build.txt" 2>&1) || {
    tail -n 40 "$build/$job.build.txt" >&2
    echo "build.sh: $src.tex failed (full output: $build/$job.build.txt)" >&2; exit 1; }
done

bash "$paper/scripts/check.sh" --pdf-dir "$build"
for v in "${volumes[@]}"; do
  printf '%-28s %4s pages\n' "${v%%:*}.pdf" "$(pdfinfo "$build/${v%%:*}.pdf" | awk '/^Pages:/{print $2}')"
done
echo "Built in $build"
