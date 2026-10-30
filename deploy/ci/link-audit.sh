#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <binary> <fused-cpu|fused-cuda>" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
binary=$1
profile=$2
[[ -f $binary && -x $binary ]] || {
  echo "link-audit: executable not found: $binary" >&2
  exit 2
}

case $profile in
  fused-cpu|fused-cuda) ;;
  *) usage ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)

case $(uname -s) in
  Linux*) platform=linux ;;
  Darwin*) platform=macos ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT*) platform=windows ;;
  *)
    echo "link-audit: unsupported operating system: $(uname -s)" >&2
    exit 2
    ;;
esac

if [[ $profile == fused-cuda && $platform != linux ]]; then
  echo "link-audit: fused-cuda is supported only on Linux" >&2
  exit 2
fi

allowlist="$repo_root/deploy/ci/link-allowlist/$platform-$profile.txt"
[[ -r $allowlist ]] || {
  echo "link-audit: missing allow-list: $allowlist" >&2
  exit 2
}

case $platform in
  linux)
    command -v ldd >/dev/null || { echo "link-audit: ldd is required" >&2; exit 2; }
    if ! raw=$(ldd "$binary" 2>&1); then
      echo "link-audit: ldd failed for $binary:" >&2
      echo "$raw" >&2
      exit 1
    fi
    if [[ $raw == *"not found"* ]]; then
      echo "link-audit: unresolved dynamic library for $binary:" >&2
      echo "$raw" >&2
      exit 1
    fi
    dependencies=$(printf '%s\n' "$raw" | awk '
      /statically linked/ { next }
      $2 == "=>" { print $1; next }
      NF > 0 { print $1 }
    ' | sed 's|.*/||' | sed '/^linux-vdso\.so\./d' | sort -u)
    ;;
  macos)
    command -v otool >/dev/null || { echo "link-audit: otool is required" >&2; exit 2; }
    if ! raw=$(otool -L "$binary" 2>&1); then
      echo "link-audit: otool failed for $binary:" >&2
      echo "$raw" >&2
      exit 1
    fi
    dependencies=$(printf '%s\n' "$raw" | awk 'NR > 1 { print $1 }' | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sort -u)
    ;;
  windows)
    command -v dumpbin >/dev/null || { echo "link-audit: dumpbin is required" >&2; exit 2; }
    if ! raw=$(dumpbin /dependents "$binary" 2>&1); then
      echo "link-audit: dumpbin failed for $binary:" >&2
      echo "$raw" >&2
      exit 1
    fi
    dependencies=$(printf '%s\n' "$raw" | awk '
      /following dependencies:/ { in_dependencies = 1; next }
      in_dependencies && /^[[:space:]]*Summary[[:space:]]*$/ { exit }
      in_dependencies {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "")
        if ($0 ~ /^[[:alnum:]_.+-]+[.]dll$/) print tolower($0)
      }
    ' | sort -u)
    ;;
esac

if [[ -z $dependencies ]]; then
  echo "link-audit: dependency scanner returned no libraries for $binary" >&2
  exit 1
fi

unexpected=()
while IFS= read -r library; do
  [[ -n $library ]] || continue
  allowed=false
  while IFS= read -r pattern || [[ -n $pattern ]]; do
    [[ -n $pattern && $pattern != \#* ]] || continue
    # Allow-list entries intentionally use shell globs for versioned system/CUDA libraries.
    # shellcheck disable=SC2254
    case $library in
      $pattern)
        allowed=true
        break
        ;;
    esac
  done < "$allowlist"
  [[ $allowed == true ]] || unexpected+=("$library")
done <<< "$dependencies"

if ((${#unexpected[@]})); then
  printf 'link-audit: unexpected libraries for %s (%s):\n' "$profile" "$platform" >&2
  printf '  %s\n' "${unexpected[@]}" >&2
  exit 1
fi

printf 'link-audit: PASS platform=%s profile=%s binary=%s libraries=%s\n' \
  "$platform" "$profile" "$binary" "$(printf '%s\n' "$dependencies" | paste -sd, -)"
