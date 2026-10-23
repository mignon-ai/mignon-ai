# @mignon-ai/mignon-web

Run [mignon-ai](https://github.com/mignon-ai/mignon-ai) models in the browser or in Node. The engine is a 4.9 MB
WebAssembly file (about 1.4 MB compressed) with a small JavaScript API. A model is a separate `.mgn` file you train
with the engine; this package contains no model and no task logic.

```js
const model = await Mignon.load({ model: "orders.mgn" });
const reply = model.generate("unpaid electronics orders over $500 from last month");
reply.text      // "orders find --status unpaid --category electronics --min-amount 500 --when last-month"
reply.tokens    // [{ text: "orders", prob: 1 }, ...] the probability the model gave each piece
reply.minProb   // the weakest piece: how sure the model was
```

Everything runs on the user's device. Nothing is sent anywhere.

## Use it

**In a page, with a script tag** (`mignon.js` finds `mignon-web.wasm` in its own folder, so any CDN that mirrors npm works):

```html
<script src="https://cdn.jsdelivr.net/npm/@mignon-ai/mignon-web@0.1.0/dist/mignon.js"></script>
<script>
  Mignon.load({ model: "orders.mgn" }).then(model => {
    console.log(model.generate("unpaid orders").text);
  });
</script>
```

Pin the version. To lock the file as well, add `integrity="sha384-..." crossorigin="anonymous"` to the tag; the values
for every file are in `integrity.json`. You can also host `mignon.js` and `mignon-web.wasm` yourself, side by side.

**As a module** (npm, any bundler or `<script type="module">`):

```js
import { load } from "@mignon-ai/mignon-web";
const model = await load({ model: "/models/orders.mgn" });   // wasm: found next to the module
```

If your bundler does not copy the wasm file, import its URL and pass it:
`load({ model, wasm: new URL("@mignon-ai/mignon-web/mignon-web.wasm", import.meta.url) })` (Vite and webpack 5 handle
this form), or pass the bytes.

**In Node** (18+): `load({ model: "path/to/orders.mgn" })` reads files directly; `require()` works too.

## API

- `load({ model, wasm? })` returns a `Model`. `model` and `wasm` can each be a URL string, a `URL`, a `Response`, an
  `ArrayBuffer` or a `Uint8Array` (in Node also a file path). Several models can be loaded at once and share one
  copy of the program.
- `model.generate(prompt, { maxTokens = 80 })` returns `{ text, tokens, minProb }`. It is synchronous and takes
  15 to 80 ms; for a page that must never stutter, run it in a Web Worker.
- `model.info` is `{ quantized, task, params, vocab_size, context_size, ... }`; `model.extra` is whatever JSON the
  model's author stored in the file (for example a description of the task), or `null`.
- `model.free()` releases the model.
- `version()` is the engine version inside the wasm file.

`generate` returns the model's most likely reply (greedy decoding). What a reply means is up to your code: the
engine only turns text into text. Treat low `minProb` as "not sure" and ask the user to confirm.

## Hosting the files

- Serve `.wasm` as `application/wasm`, or streaming compilation fails.
- Cross-origin pages need `Access-Control-Allow-Origin` on `mignon.js`, the wasm and the model.
- Compress with brotli or gzip, and use `Cache-Control: public, max-age=31536000, immutable` on versioned paths.
- Keep `mignon.js` and `mignon-web.wasm` of the same version together: the script embeds Go's loader, which has to
  match the Go version that built the wasm.

## Install

```
npm install @mignon-ai/mignon-web
```

## Build and release

```bash
npm test                  # builds dist/ (needs Go), runs the tests
npm run pack:tarball      # the tarball npm pack would make
```

Releases are published to npm by the `Publish web package` workflow when a tag `vX.Y.Z` (the same tag is the Go module version) is pushed; the tag must
match the version in `package.json`. The package is under the [Elastic License 2.0](LICENSE). The Go loader inside
`dist/` comes with Go's BSD licence, see `NOTICE`.
