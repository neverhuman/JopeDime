#!/usr/bin/env bash
# Apply native dependencies only to browser execution, never the numerical goldens.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
evidence="$repo/target/typescript-evidence"
mkdir -p "$evidence" "$repo/target/tmp/typescript-browser/cache" "$repo/target/tmp/typescript-browser/home"
runtime=$(python3 "$repo/deploy/ci/browser-deps.py" --lock "$repo/deploy/ci/typescript/browser-deps.lock.json" --cache "$repo/target/tool-cache/typescript-browser" --receipt "$evidence/browser-dependencies.json")
export JOPE_TS_NATIVE_RUNTIME="$runtime"
export LD_LIBRARY_PATH="$runtime/usr/lib/x86_64-linux-gnu:$runtime/lib/x86_64-linux-gnu:$runtime/usr/lib:$runtime/lib:$runtime/usr/lib/x86_64-linux-gnu/blas:$runtime/usr/lib/x86_64-linux-gnu/lapack:$runtime/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
FONTCONFIG_FILE=$(node -e 'console.log(JSON.parse(require("fs").readFileSync(process.argv[1])).fontconfig)' "$evidence/browser-dependencies.json")
export FONTCONFIG_FILE
export PLAYWRIGHT_BROWSERS_PATH="$repo/target/browser-cache/typescript-playwright" PLAYWRIGHT_SKIP_BROWSER_GC=1
export TMPDIR="$repo/target/tmp/typescript-browser" HOME="$repo/target/tmp/typescript-browser/home" XDG_CACHE_HOME="$repo/target/tmp/typescript-browser/cache"
export GST_PLUGIN_SYSTEM_PATH_1_0="$runtime/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
export __EGL_VENDOR_LIBRARY_FILENAMES="$runtime/usr/share/glvnd/egl_vendor.d/50_mesa.json"
export LIBGL_DRIVERS_PATH="$runtime/usr/lib/x86_64-linux-gnu/dri" LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
"$repo/deploy/ci/typescript/node_modules/.bin/playwright" install chromium firefox webkit
node "$repo/deploy/ci/typescript/browser.mjs" 2>&1 | tee "$evidence/browser.log"
