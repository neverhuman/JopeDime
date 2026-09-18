#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[[ -n "$root" && "$root" == /* ]] || {
  printf 'workflow lint root must be an absolute path\n' >&2
  exit 1
}
canonical_root="$(realpath -e -- "$root")"
[[ "$root" == "$canonical_root" && -d "$root" && ! -L "$root" ]] || {
  printf 'workflow lint root must be a canonical real directory: %s\n' "$root" >&2
  exit 1
}
cd "$root"

command -v actionlint >/dev/null 2>&1 || {
  printf 'required tool missing: actionlint\n' >&2
  exit 127
}

workflow_dir=.github/workflows
if [[ ! -e "$workflow_dir" && ! -L "$workflow_dir" ]]; then
  printf 'actionlint-workflows status=not_applicable files=0 reason=no-workflow-directory\n'
  exit 0
fi
[[ -d "$workflow_dir" && ! -L "$workflow_dir" ]] || {
  printf 'workflow directory must be a real directory: %s\n' "$workflow_dir" >&2
  exit 1
}

mapfile -d '' -t workflow_paths < <(
  find "$workflow_dir" -mindepth 1 -maxdepth 1 \
    \( -name '*.yml' -o -name '*.yaml' \) -print0 | sort -z
)
for path in "${workflow_paths[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'workflow path must be a regular non-symlink file: %s\n' "$path" >&2
    exit 1
  }
done

if (( ${#workflow_paths[@]} == 0 )); then
  printf 'actionlint-workflows status=not_applicable files=0 reason=no-workflow-files\n'
  exit 0
fi

actionlint "${workflow_paths[@]}"
printf 'actionlint-workflows status=pass files=%s\n' "${#workflow_paths[@]}"
