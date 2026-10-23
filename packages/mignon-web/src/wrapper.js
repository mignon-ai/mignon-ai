// The part of the package that is the same in every build. scripts/build.mjs puts Go's wasm_exec.js in front of
// it and wraps it as an ES module (dist/mignon.mjs) or as a plain script that defines `Mignon` (dist/mignon.js).
// DEFAULT_WASM is replaced with an expression that finds mignon-web.wasm next to the built file.

const isNode = typeof process !== "undefined" && !!process.versions && !!process.versions.node;

/** Bytes of a model or wasm file from a URL, a path (Node), a Response, an ArrayBuffer or a Uint8Array. */
async function toBytes(source, what) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (ArrayBuffer.isView(source)) return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  if (typeof Response !== "undefined" && source instanceof Response) {
    if (!source.ok) throw new Error(`could not download the ${what} (${source.status})`);
    return new Uint8Array(await source.arrayBuffer());
  }
  if (isNode) {
    const href = typeof source === "string" ? source : source instanceof URL ? source.href : "";
    if (!/^https?:/i.test(href)) { // a file path or a file: URL
      const fs = await import(/* webpackIgnore: true */ /* @vite-ignore */ "node:fs/promises");
      const path = /^file:/i.test(href) ? (await import(/* webpackIgnore: true */ /* @vite-ignore */ "node:url")).fileURLToPath(href) : href;
      return new Uint8Array(await fs.readFile(path));
    }
  }
  return toBytes(await fetch(source), what);
}

let enginePromise = null;

// Starts the Go program once per page: it registers mignonLoad, mignonGenerate and mignonFree.
function engine(wasm) {
  if (!enginePromise) {
    enginePromise = (async () => {
      const go = new Go();
      let instance;
      if (typeof wasm === "string" || wasm instanceof URL || (typeof Response !== "undefined" && wasm instanceof Response)) {
        if (!isNode && typeof WebAssembly.instantiateStreaming === "function") { // streaming compile when the server sends application/wasm
          try {
            const response = wasm instanceof Response ? wasm : await fetch(wasm);
            ({ instance } = await WebAssembly.instantiateStreaming(response, go.importObject));
          } catch (e) {
            instance = null;
          }
        }
      }
      if (!instance) ({ instance } = await WebAssembly.instantiate(await toBytes(wasm, "wasm file"), go.importObject));
      go.run(instance); // returns when the program exits, which it never does
      const g = globalThis;
      return { load: g.mignonLoad, generate: g.mignonGenerate, free: g.mignonFree, version: g.mignonVersion };
    })();
    enginePromise.catch(() => { enginePromise = null; }); // allow another attempt after a failure
  }
  return enginePromise;
}

/** A loaded model. */
class Model {
  constructor(eng, id, info, extra) {
    this._eng = eng;
    this._id = id;
    /** Shape and metadata of the model: { quantized, task, params, vocab_size, context_size, ... }. */
    this.info = info;
    /** The JSON stored in the model file by whoever trained it (for example a task description), or null. */
    this.extra = extra;
  }

  /**
   * Answers a prompt with the model's most likely reply. Synchronous: a reply takes 15 to 80 ms.
   * @param {string} prompt
   * @param {{maxTokens?: number}} [options]
   * @returns {{text: string, tokens: {text: string, prob: number}[], minProb: number}}
   */
  generate(prompt, options = {}) {
    if (this._id === 0) throw new Error("the model has been freed");
    const r = JSON.parse(this._eng.generate(this._id, String(prompt), options.maxTokens || 80));
    if (r.error) throw new Error(r.error);
    const tokens = r.tokens || [];
    return { text: r.text, tokens, minProb: tokens.reduce((least, t) => Math.min(least, t.prob), 1) };
  }

  /** Releases the model's memory. Using it afterwards throws. */
  free() {
    if (this._id !== 0) this._eng.free(this._id);
    this._id = 0;
  }
}

/**
 * Loads a model. Several models can be loaded; they share one copy of the wasm program.
 * @param {{model: string|URL|Response|ArrayBuffer|Uint8Array, wasm?: string|URL|Response|ArrayBuffer|Uint8Array}} options
 *   model: where the model file (.mgn) is; wasm: where mignon-web.wasm is (default: next to this file).
 * @returns {Promise<Model>}
 */
async function load(options) {
  if (!options || options.model === undefined) throw new Error("load: a model is required");
  const eng = await engine(options.wasm !== undefined ? options.wasm : DEFAULT_WASM);
  const bytes = await toBytes(options.model, "model");
  const r = JSON.parse(eng.load(bytes));
  if (r.error) throw new Error(r.error);
  return new Model(eng, r.id, r.info, r.extra === undefined ? null : r.extra);
}

/** Version of the engine inside the wasm file ("dev" for an unversioned build). */
async function version(wasm) {
  return (await engine(wasm !== undefined ? wasm : DEFAULT_WASM)).version;
}
