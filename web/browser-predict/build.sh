#!/usr/bin/env bash
# Build the trusted runtime; customer model objects are never compiled or executed as code.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
target=${CARGO_TARGET_DIR:-$root/target}
source_sha=${JOPE_BROWSER_SOURCE_SHA:-$(git rev-parse HEAD)}
[[ "$source_sha" =~ ^[a-f0-9]{40}$ ]] || { echo 'full source SHA required' >&2; exit 2; }
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
# Runner images have an immutable native toolchain. Provision the matching WASM
# toolchain in this lane's target cache instead of modifying that shared image.
wasm_libdir=$(rustc --print target-libdir --target wasm32-unknown-unknown)
if ! compgen -G "$wasm_libdir/libstd-*.rlib" >/dev/null; then
  export RUSTUP_HOME="$target/browser-predict/rustup"
  RUSTUP_TOOLCHAIN=$(python3 -c 'import tomllib; print(tomllib.load(open("rust-toolchain.toml", "rb"))["toolchain"]["channel"])')
  export RUSTUP_TOOLCHAIN
  rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal --target wasm32-unknown-unknown --no-self-update
fi
JOPE_BROWSER_SOURCE_SHA="$source_sha" RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=--max-memory=536870912" \
  cargo build --locked --manifest-path web/browser-predict/Cargo.toml --target-dir "$target" \
  --lib --target wasm32-unknown-unknown --release
python3 - "$target" "$source_sha" <<'PY'
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);wasm=root/'wasm32-unknown-unknown/release/jope_browser_predict.wasm'
out=root/'browser-predict';out.mkdir(exist_ok=True)
receipt={'schema':'veox.browser-runtime/v1','source_repository':'https://github.com/neverhuman/JopeDime','source_commit':sys.argv[2],'wasm_sha256':hashlib.sha256(wasm.read_bytes()).hexdigest(),'abi':1,'max_memory_bytes':536870912,'inference_only':True,'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'lock_sha256':hashlib.sha256(pathlib.Path('web/browser-predict/Cargo.lock').read_bytes()).hexdigest()}
(out/'runtime-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(wasm)
PY
