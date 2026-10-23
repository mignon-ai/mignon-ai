#!/bin/bash
# Builds the demo: the @mignon-ai/mignon-web package (the WebAssembly engine and its JavaScript API) copied next
# to the page, and optionally the model the page downloads.
#   examples/order-search/build.sh [path/to/model.mgn]
#   python3 -m http.server -d examples/order-search 8000      then open http://localhost:8000
# Tests for the page logic: node --test examples/order-search/orders.test.js; score the model: node examples/order-search/check.js
set -euo pipefail
here="$(dirname "$0")"
cd "$here/../.."
node packages/mignon-web/scripts/build.mjs
cp packages/mignon-web/dist/mignon.js packages/mignon-web/dist/mignon-web.wasm "$here/"
if [ $# -ge 1 ]; then cp "$1" "$here/orders.mgn"; fi
ls -lh "$here/mignon.js" "$here/mignon-web.wasm" "$here/orders.mgn"
