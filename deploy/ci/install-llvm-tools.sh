#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

sysroot="$(rustc --print sysroot)"
host="$(rustc -vV | sed -n 's/^host: //p')"
if [[ -z "$host" ]]; then
  echo "unable to determine rustc host triple" >&2
  exit 1
fi

sys_profdata="$sysroot/lib/rustlib/$host/bin/llvm-profdata"
sys_cov="$sysroot/lib/rustlib/$host/bin/llvm-cov"
if [[ -x "$sys_profdata" && -x "$sys_cov" ]]; then
  "$sys_profdata" --version >/dev/null
  "$sys_cov" --version >/dev/null
  printf '%s\n%s\n' "$sys_profdata" "$sys_cov"
  exit 0
fi

if [[ "$host" != "x86_64-unknown-linux-gnu" ]]; then
  echo "llvm-tools not installed in sysroot for $host; run: rustup component add llvm-tools-preview" >&2
  exit 1
fi

version="1.97.1"
dist_date="2026-07-16"
digest="056e39b861ad9cba6435cb40eab5095bfb421228c4a20ebf473a9ace238d6245"
cache_dir="${JOPE_LLVM_TOOLS_CACHE:-target/llvm-tools-$version}"
mkdir -p "$cache_dir"

pkg_name="llvm-tools-$version-$host"
archive="$cache_dir/$pkg_name.tar.gz"
bin_dir="$cache_dir/$pkg_name/llvm-tools-preview/lib/rustlib/$host/bin"
lib_dir="$cache_dir/$pkg_name/llvm-tools-preview/lib/rustlib/$host/lib"
profdata="$bin_dir/llvm-profdata"
cov="$bin_dir/llvm-cov"

if [[ ! -x "$profdata" || ! -x "$cov" ]]; then
  if [[ -e "$archive" || -L "$archive" ]]; then
    [[ -f "$archive" && ! -L "$archive" ]] || { echo "refusing non-regular archive cache entry: $archive" >&2; exit 1; }
  else
    download="$(mktemp "$archive.XXXXXX")"
    trap 'rm -f "$download"' EXIT
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error --retry 3 \
      "https://static.rust-lang.org/dist/$dist_date/$pkg_name.tar.gz" \
      --output "$download"
    echo "$digest  $download" | sha256sum --check --status
    ln -T -- "$download" "$archive"
    rm -f -- "$download"
    trap - EXIT
  fi
  echo "$digest  $archive" | sha256sum --check --status
  tar --no-same-owner -xzf "$archive" -C "$cache_dir"
fi

mkdir -p "$lib_dir"
for so_file in "$sysroot"/lib/libLLVM*; do
  if [[ -e "$so_file" ]]; then
    base_so="$(basename "$so_file")"
    if [[ ! -e "$lib_dir/$base_so" ]]; then
      ln -s "$so_file" "$lib_dir/$base_so"
    fi
  fi
done

"$profdata" --version >/dev/null
"$cov" --version >/dev/null
printf '%s\n%s\n' "$profdata" "$cov"
