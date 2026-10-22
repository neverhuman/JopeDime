#!/usr/bin/env bash
set -euo pipefail
paper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_dir=$(git -C "$paper_dir" rev-parse --show-toplevel)
build_dir="$repo_dir/target/innovationzero-paper"
cd "$paper_dir"
if [[ "${1:-}" == "--clean" ]]; then
  # Never blanket-delete target: only this build's known generated products.
  for stem in InnovationZero InnovationZero-Supplement InnovationZero-Archive; do
    for extension in aux bbl blg fdb_latexmk fls log out toc xdv pdf; do
      rm -f -- "$build_dir/$stem.$extension"
    done
  done
  rm -f -- "$build_dir/archive-index.tex" "$build_dir/build-environment.txt"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf 'usage: %s [--clean]\n' "$0" >&2
  exit 2
fi
for tool in python3 latexmk pdflatex xelatex bibtex pdfinfo pdftotext pdffonts; do
  command -v "$tool" >/dev/null || { printf 'Missing build tool: %s\n' "$tool" >&2; exit 1; }
done
mkdir -p -- "$build_dir"
# Stable source epoch: the pinned main snapshot's committer timestamp (2026-10-22).
# Override explicitly for release reproducibility; never use the current wall clock.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1792655756}"
export FORCE_SOURCE_DATE=1
export TZ=UTC
export LC_ALL=C.UTF-8
export TEXINPUTS="$paper_dir:$build_dir:${TEXINPUTS:-}"
python3 scripts/check.py --prepare "$build_dir"
{
  printf 'SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'source_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
  latexmk -version
  pdflatex --version
  xelatex --version
  python3 --version
} > "$build_dir/build-environment.txt"
# Main text links into the separately paginated methods supplement.
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape \
  -outdir="$build_dir" -jobname=InnovationZero-Supplement supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape \
  -outdir="$build_dir" -jobname=InnovationZero main.tex
# XeLaTeX preserves Unicode in the archived prompt and source comments.
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape \
  -outdir="$build_dir" -jobname=InnovationZero-Archive listings.tex
python3 scripts/check.py --pdf-dir "$build_dir"
printf '\nBuilt publication PDFs in %s\n' "$build_dir"
